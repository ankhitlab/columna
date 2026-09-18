/**
 * Specialized CPU kernels for common plan shapes used in benchmarks.
 * Falls back to generic eval paths when patterns don't match.
 */
import {
  allocateData,
  getColumn,
  getValue,
  isNumeric,
  isValid,
  setValid,
  tableFromColumns,
  takeColumn,
  takeTable,
  type Column,
  type DType,
  type TableView,
} from '@columna/arrow'
import type { AggKind, ExprNode, MathOp, QuantileMethod } from './types.js'
import { applyMathOp, roundHalfAway } from './math.js'
import { tryFused } from './fused.js'
import {
  getNativeKernels,
  isNativeKernelsLoaded,
  NATIVE_FILTER_MIN_ROWS,
  NATIVE_GATHER_MIN_ROWS,
  NATIVE_GROUPBY_MIN_ROWS,
  NATIVE_JOIN_MIN_ROWS,
  NATIVE_JOIN_BUILD_MIN_ROWS,
  NATIVE_SORT_MIN_ROWS,
  NATIVE_SORT_MULTI_MIN_ROWS,
  NATIVE_UNIQUE_MIN_ROWS,
  NATIVE_STR_MIN_ROWS,
} from './native_kernels.js'

type NumArr = Float64Array | Float32Array | Int32Array | Uint32Array

function numericView(col: Column): NumArr | null {
  if (!isNumeric(col.field.dtype) && col.field.dtype !== 'datetime' && col.field.dtype !== 'category') {
    return null
  }
  return col.data as NumArr
}

export function gather(table: TableView, indices: ArrayLike<number>, keep?: readonly string[]): TableView {
  const n = indices.length
  const cols =
    keep && keep.length > 0
      ? keep.map((name) => {
          const col = table.columns.find((c) => c.field.name === name)
          if (!col) throw new Error(`Unknown column "${name}" in gather`)
          return col
        })
      : table.columns
  const idx =
    indices instanceof Uint32Array ? indices : Uint32Array.from({ length: n }, (_, i) => indices[i] as number)

  // Prefer native gather when loaded; otherwise a tight typed JS gather for no-null columns.
  const k = n >= NATIVE_GATHER_MIN_ROWS && isNativeKernelsLoaded() ? getNativeKernels() : null
  if (k && (k.gatherF64 || k.gatherI32)) {
    return tableFromColumns(
      cols.map((c) => {
        if (c.nullBitmap) return takeColumn(c, idx)
        const d = c.field.dtype
        if ((d === 'f64' || d === 'datetime') && k.gatherF64 && c.data instanceof Float64Array) {
          return { field: c.field, data: k.gatherF64(c.data, idx), dictionary: c.dictionary }
        }
        if ((d === 'i32' || d === 'category' || d === 'u32') && k.gatherI32) {
          const src =
            c.data instanceof Int32Array
              ? c.data
              : c.data instanceof Uint32Array
                ? new Int32Array(c.data.buffer, c.data.byteOffset, c.data.length)
                : null
          if (src && d === 'i32') {
            return { field: c.field, data: k.gatherI32(src, idx), dictionary: c.dictionary }
          }
          if (src && (d === 'category' || d === 'u32')) {
            const out = k.gatherI32(src, idx)
            return {
              field: c.field,
              data: new Uint32Array(out.buffer, out.byteOffset, out.length),
              dictionary: c.dictionary,
            }
          }
        }
        return takeColumn(c, idx)
      }),
    )
  }

  // JS: reuse takeColumn's typed no-null paths with a single Uint32Array index buffer.
  return tableFromColumns(cols.map((c) => takeColumn(c, idx)))
}

export function flattenAnd(expr: ExprNode): ExprNode[] | null {
  if (expr.type === 'binary' && expr.op === 'and') {
    const left = flattenAnd(expr.left)
    const right = flattenAnd(expr.right)
    if (!left || !right) return null
    return [...left, ...right]
  }
  if (expr.type === 'binary') return [expr]
  return null
}

type Cmp = { data: NumArr; bitmap: Uint8Array | undefined; op: string; lit: number }

function cmpAt(r: Cmp, i: number): boolean {
  if (r.bitmap && !isValid(r.bitmap, i)) return false
  const v = r.data[i]!
  switch (r.op) {
    case 'gt':
      return v > r.lit
    case 'gte':
      return v >= r.lit
    case 'lt':
      return v < r.lit
    case 'lte':
      return v <= r.lit
    case 'eq':
      return v === r.lit
    case 'neq':
      return v !== r.lit
    default:
      return false
  }
}

function resolveCmps(table: TableView, predicate: ExprNode): Cmp[] | null {
  const comps = flattenAnd(predicate)
  if (!comps || comps.length === 0) return null
  const resolved: Cmp[] = []
  for (const c of comps) {
    if (c.type !== 'binary' || c.left.type !== 'col' || c.right.type !== 'lit') return null
    if (typeof c.right.value !== 'number') return null
    if (!['gt', 'gte', 'lt', 'lte', 'eq', 'neq'].includes(c.op)) return null
    const col = getColumn(table, c.left.name)
    const data = numericView(col)
    if (!data) return null
    resolved.push({ data, bitmap: col.nullBitmap, op: c.op, lit: c.right.value })
  }
  return resolved
}

const CMP_OP_CODE: Record<string, number> = {
  eq: 0,
  neq: 1,
  gt: 2,
  gte: 3,
  lt: 4,
  lte: 5,
}

/**
 * Dual comparison indices (shared by sync filter + parallel path).
 * Uses `@columna/native` Rayon when loaded and n is large enough; any cmp op pair.
 */
export function dualAndIndices(
  a: NumArr,
  b: NumArr,
  opA: string,
  litA: number,
  opB: string,
  litB: number,
): Uint32Array {
  const n = a.length
  const codeA = CMP_OP_CODE[opA]
  const codeB = CMP_OP_CODE[opB]
  if (
    codeA !== undefined &&
    codeB !== undefined &&
    n >= NATIVE_FILTER_MIN_ROWS &&
    isNativeKernelsLoaded()
  ) {
    const k = getNativeKernels()
    if (a instanceof Int32Array && b instanceof Float64Array && k.filterAnd2I32F64) {
      return k.filterAnd2I32F64(a, b, codeA, litA, codeB, litB)
    }
    if (a instanceof Int32Array && b instanceof Int32Array && k.filterAnd2I32I32) {
      return k.filterAnd2I32I32(a, b, codeA, litA, codeB, litB)
    }
    if (a instanceof Float64Array && b instanceof Float64Array && k.filterAnd2F64F64) {
      return k.filterAnd2F64F64(a, b, codeA, litA, codeB, litB)
    }
    // i32+f64 with columns swapped
    if (a instanceof Float64Array && b instanceof Int32Array && k.filterAnd2I32F64) {
      return k.filterAnd2I32F64(b, a, codeB, litB, codeA, litA)
    }
  }
  const idx = new Uint32Array(n)
  let j = 0
  const cmp = (op: string, v: number, lit: number) => {
    switch (op) {
      case 'gt':
        return v > lit
      case 'gte':
        return v >= lit
      case 'lt':
        return v < lit
      case 'lte':
        return v <= lit
      case 'eq':
        return v === lit
      case 'neq':
        return v !== lit
      default:
        return false
    }
  }
  for (let i = 0; i < n; i++) {
    if (cmp(opA, a[i]!, litA) && cmp(opB, b[i]!, litB)) idx[j++] = i
  }
  return idx.subarray(0, j)
}

/** @deprecated Prefer dualAndIndices — kept for parallel worker path that still assumes gt∧gt. */
export function dualGtIndices(a: NumArr, b: NumArr, la: number, lb: number): Uint32Array {
  return dualAndIndices(a, b, 'gt', la, 'gt', lb)
}

/**
 * Single-pass filter → Uint32 indices.
 * Optional `keep` prunes columns at gather (project pushdown).
 */
export function tryFastFilter(
  table: TableView,
  predicate: ExprNode,
  keep?: readonly string[],
): TableView | null {
  const resolved = resolveCmps(table, predicate)
  if (resolved) {
    const n = table.numRows

    if (resolved.length === 2 && !resolved[0]!.bitmap && !resolved[1]!.bitmap) {
      const idx = dualAndIndices(
        resolved[0]!.data,
        resolved[1]!.data,
        resolved[0]!.op,
        resolved[0]!.lit,
        resolved[1]!.op,
        resolved[1]!.lit,
      )
      return gather(table, idx, keep)
    }

    const idx = new Uint32Array(n)
    let j = 0
    outerFill: for (let i = 0; i < n; i++) {
      for (const r of resolved) if (!cmpAt(r, i)) continue outerFill
      idx[j++] = i
    }
    return gather(table, idx.subarray(0, j), keep)
  }

  // Vectorized mask for isIn / isBetween / when / compound predicates
  const n = table.numRows
  if (n === 0) return gather(table, [], keep)
  const mask = evalVec(table, predicate, n)
  if (!mask) return null
  const idx = new Uint32Array(n)
  let j = 0
  for (let i = 0; i < n; i++) if (truthy(mask, i)) idx[j++] = i
  return gather(table, idx.subarray(0, j), keep)
}

/** Match dual-gt filter shape for parallel dispatch. */
export function matchDualGtFilter(
  table: TableView,
  predicate: ExprNode,
): { a: NumArr; b: NumArr; la: number; lb: number } | null {
  const resolved = resolveCmps(table, predicate)
  if (
    !resolved ||
    resolved.length !== 2 ||
    resolved[0]!.op !== 'gt' ||
    resolved[1]!.op !== 'gt' ||
    resolved[0]!.bitmap ||
    resolved[1]!.bitmap
  ) {
    return null
  }
  return { a: resolved[0]!.data, b: resolved[1]!.data, la: resolved[0]!.lit, lb: resolved[1]!.lit }
}

const HASH_F64 = new Float64Array(1)
const HASH_U32 = new Uint32Array(HASH_F64.buffer)

/**
 * Distinct count of a[0..len): dense marks when the values are integers in a ≤16M span (the common
 * id / code / bucket case), otherwise an open-addressing set on the raw f64 bits. Both avoid
 * `new Set(doubles)`, which boxes every value (~10× slower at 1M+).
 */
export function countUniqueNumeric(a: ArrayLike<number>, len: number): number {
  if (len === 0) return 0
  let min = Infinity
  let max = -Infinity
  let ints = true
  for (let i = 0; i < len; i++) {
    const v = a[i]!
    if (v < min) min = v
    if (v > max) max = v
    if (ints && !Number.isInteger(v)) ints = false
  }
  if (ints && max - min < 16_777_216) {
    const span = max - min + 1
    const seen = new Uint8Array(span)
    let count = 0
    for (let i = 0; i < len; i++) {
      const k = a[i]! - min
      if (!seen[k]) {
        seen[k] = 1
        count++
      }
    }
    return count
  }
  let cap = 16
  while (cap < len * 2) cap <<= 1
  const keys = new Float64Array(cap)
  const used = new Uint8Array(cap)
  const mask = cap - 1
  let count = 0
  for (let i = 0; i < len; i++) {
    const v = a[i]! + 0 // -0 → +0 so both hash alike (Set semantics)
    HASH_F64[0] = v
    let h = (HASH_U32[0]! ^ HASH_U32[1]!) | 0
    h = Math.imul(h ^ (h >>> 16), 0x7feb352d)
    h = Math.imul(h ^ (h >>> 15), 0x846ca68b)
    let slot = (h ^ (h >>> 16)) & mask
    for (;;) {
      if (!used[slot]) {
        used[slot] = 1
        keys[slot] = v
        count++
        break
      }
      if (keys[slot] === v || (v !== v && keys[slot] !== keys[slot])) break
      slot = (slot + 1) & mask
    }
  }
  return count
}

/** In-place quickselect: after the call, a[k] holds the k-th smallest of a[0..len) (median-of-3 pivots). */
function selectKth(a: Float64Array, len: number, k: number): number {
  let lo = 0
  let hi = len - 1
  while (hi > lo) {
    // median of three → pivot value
    const mid = (lo + hi) >>> 1
    if (a[mid]! < a[lo]!) [a[mid], a[lo]] = [a[lo]!, a[mid]!]
    if (a[hi]! < a[lo]!) [a[hi], a[lo]] = [a[lo]!, a[hi]!]
    if (a[hi]! < a[mid]!) [a[hi], a[mid]] = [a[mid]!, a[hi]!]
    const pivot = a[mid]!
    let i = lo
    let j = hi
    while (i <= j) {
      while (a[i]! < pivot) i++
      while (a[j]! > pivot) j--
      if (i <= j) {
        const t = a[i]!
        a[i] = a[j]!
        a[j] = t
        i++
        j--
      }
    }
    if (k <= j) hi = j
    else if (k >= i) lo = i
    else break
  }
  return a[k]!
}

/**
 * Fractional 0-based rank of quantile q in a sample of `len` sorted values.
 * 'linear' (default): (len − 1)·q — type 7. 'minitab': q·(len + 1) − 1, clamped to [0, len − 1] — type 6,
 * so Q1 of {1,2,3,4} is 1.25 (Minitab) rather than 1.75 (pandas).
 */
export function quantilePos(len: number, q: number, method?: QuantileMethod): number {
  const qq = Math.min(1, Math.max(0, q))
  if (method === 'minitab') return Math.min(len - 1, Math.max(0, qq * (len + 1) - 1))
  return (len - 1) * qq
}

/** Linear-interpolated quantile (type 7 by default, Minitab type 6 on request) of a[0..len) via quickselect — O(n), no full sort. Mutates `a`. */
export function quantileSelect(a: Float64Array, len: number, q: number, method?: QuantileMethod): number {
  if (len <= 0) return NaN
  const pos = quantilePos(len, q, method)
  const lo = Math.floor(pos)
  const hi = Math.ceil(pos)
  const vlo = selectKth(a, len, lo)
  if (lo === hi) return vlo
  // after selecting lo, everything at index > lo is ≥ a[lo]; the (lo+1)-th smallest is the min of that tail
  let vhi = Infinity
  for (let i = lo + 1; i < len; i++) if (a[i]! < vhi) vhi = a[i]!
  const w = pos - lo
  return vlo * (1 - w) + vhi * w
}

const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1
const RADIX_BUCKETS = 65536

/**
 * Stable argsort of a numeric column: LSD radix sort on the sign-corrected 64-bit pattern of each
 * value (4 passes × 16 bits). Nulls go last in input order. ~5–10× faster than `idx.sort(cmp)` at
 * 1M+ rows and independent of key distribution. NaN sorts after +Infinity; −0 sorts before +0.
 *
 * When `order` is provided, re-sort that permutation stably by `data[order[i]]` (successive multi-key
 * radix: pass previous pass's indices here).
 */
export function argsortNumeric(
  data: ArrayLike<number>,
  n: number,
  nullBitmap: Uint8Array | undefined,
  descending = false,
  order?: Uint32Array,
  nullsLast = true,
): Uint32Array {
  const tmp = new Float64Array(1)
  const tv = new Uint32Array(tmp.buffer)
  const LO = LITTLE_ENDIAN ? 0 : 1
  const HI = LITTLE_ENDIAN ? 1 : 0
  const len = order?.length ?? n

  let idx = new Uint32Array(len)
  let keyLo = new Uint32Array(len)
  let keyHi = new Uint32Array(len)
  let m = 0
  for (let i = 0; i < len; i++) {
    const row = order ? order[i]! : i
    if (nullBitmap && !isValid(nullBitmap, row)) continue
    tmp[0] = data[row]!
    let lo = tv[LO]!
    let hi = tv[HI]!
    // negative: flip everything; positive: flip the sign bit → unsigned order == numeric order
    if (hi & 0x80000000) {
      lo = ~lo >>> 0
      hi = ~hi >>> 0
    } else hi = (hi ^ 0x80000000) >>> 0
    if (descending) {
      lo = ~lo >>> 0
      hi = ~hi >>> 0
    }
    idx[m] = row
    keyLo[m] = lo
    keyHi[m] = hi
    m++
  }

  let idx2 = new Uint32Array(m)
  let lo2 = new Uint32Array(m)
  let hi2 = new Uint32Array(m)
  const count = new Uint32Array(RADIX_BUCKETS)
  for (let pass = 0; pass < 4; pass++) {
    const src = pass < 2 ? keyLo : keyHi
    const shift = (pass & 1) * 16
    count.fill(0)
    for (let j = 0; j < m; j++) count[(src[j]! >>> shift) & 0xffff]++
    // all keys share this digit → pass is a no-op
    let single = false
    for (let b = 0; b < RADIX_BUCKETS; b++) {
      if (count[b] === m) single = true
      if (count[b]) break
    }
    if (single) continue
    let sum = 0
    for (let b = 0; b < RADIX_BUCKETS; b++) {
      const c = count[b]!
      count[b] = sum
      sum += c
    }
    for (let j = 0; j < m; j++) {
      const k = count[(src[j]! >>> shift) & 0xffff]!++
      idx2[k] = idx[j]!
      lo2[k] = keyLo[j]!
      hi2[k] = keyHi[j]!
    }
    ;[idx, idx2] = [idx2, idx]
    ;[keyLo, lo2] = [lo2, keyLo]
    ;[keyHi, hi2] = [hi2, keyHi]
  }

  if (m === len) return idx.length === len ? idx : idx.slice(0, len)
  const out = new Uint32Array(len)
  if (nullsLast) {
    out.set(idx.subarray(0, m))
    let k = m
    for (let i = 0; i < len; i++) {
      const row = order ? order[i]! : i
      if (nullBitmap && !isValid(nullBitmap, row)) out[k++] = row
    }
  } else {
    let k = 0
    for (let i = 0; i < len; i++) {
      const row = order ? order[i]! : i
      if (nullBitmap && !isValid(nullBitmap, row)) out[k++] = row
    }
    out.set(idx.subarray(0, m), k)
  }
  return out
}

/**
 * Map a column to numeric codes suitable for radix argsort (lexical order for text).
 * Category codes are remapped via sorted(dictionary) ranks; utf8 gets dense ranks over uniques.
 */
export function sortKeyCodes(col: Column): { codes: NumArr; nullBitmap?: Uint8Array } | null {
  const dtype = col.field.dtype
  if (dtype === 'category' && col.dictionary) {
    const dict = col.dictionary
    const card = dict.length
    const orderIdx = Array.from({ length: card }, (_, i) => i)
    orderIdx.sort((a, b) => {
      const sa = dict[a]!
      const sb = dict[b]!
      return sa < sb ? -1 : sa > sb ? 1 : 0
    })
    const rank = new Uint32Array(card)
    for (let r = 0; r < card; r++) rank[orderIdx[r]!] = r
    const src = col.data as Uint32Array
    const n = src.length
    const codes = new Uint32Array(n)
    for (let i = 0; i < n; i++) {
      const c = src[i]!
      codes[i] = c < card ? rank[c]! : 0
    }
    return { codes, nullBitmap: col.nullBitmap }
  }
  if (dtype === 'utf8') {
    const strings = col.data as string[]
    const n = strings.length
    const seen = new Map<string, number>()
    const uniques: string[] = []
    for (let i = 0; i < n; i++) {
      if (col.nullBitmap && !isValid(col.nullBitmap, i)) continue
      const s = strings[i] ?? ''
      if (!seen.has(s)) {
        seen.set(s, uniques.length)
        uniques.push(s)
      }
    }
    uniques.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0))
    for (let i = 0; i < uniques.length; i++) seen.set(uniques[i]!, i)
    const codes = new Uint32Array(n)
    for (let i = 0; i < n; i++) {
      if (col.nullBitmap && !isValid(col.nullBitmap, i)) continue
      codes[i] = seen.get(strings[i] ?? '') ?? 0
    }
    return { codes, nullBitmap: col.nullBitmap }
  }
  const data = numericView(col)
  if (!data) return null
  return { codes: data, nullBitmap: col.nullBitmap }
}

export function tryFastSort(
  table: TableView,
  by: Array<{ expr: ExprNode; descending: boolean; nullsLast?: boolean }>,
  limit?: number,
): TableView | null {
  if (by.length === 0) return null
  for (const k of by) if (k.expr.type !== 'col') return null

  const keys: Array<{ codes: NumArr; nullBitmap?: Uint8Array; descending: boolean; nullsLast: boolean }> = []
  for (const k of by) {
    const col = getColumn(table, (k.expr as { type: 'col'; name: string }).name)
    const sk = sortKeyCodes(col)
    if (!sk) return null
    keys.push({
      codes: sk.codes,
      nullBitmap: sk.nullBitmap,
      descending: k.descending,
      nullsLast: k.nullsLast !== false,
    })
  }

  const n = table.numRows

  // Top-k heap path.
  // Only when the key has no nulls: otherwise the full sort path is responsible
  // for placing nulls according to nullsLast/nullsFirst.
  if (
    limit !== undefined &&
    limit > 0 &&
    limit < n &&
    by.length === 1 &&
    keys[0]!.nullsLast &&
    !keys[0]!.nullBitmap
  ) {
    const key = keys[0]!

    const idx = new Uint32Array(n)
    for (let i = 0; i < n; i++) idx[i] = i

    topKIndices(idx, key.codes, undefined, key.descending, limit)

    return gather(table, idx.subarray(0, limit))
  }

  // Native Rayon argsort for large single-key numeric sorts (no limit / limit ≥ n).
  if (
    by.length === 1 &&
    n >= NATIVE_SORT_MIN_ROWS &&
    isNativeKernelsLoaded() &&
    (limit === undefined || limit >= n)
  ) {
    const key = keys[0]!
    const native = getNativeKernels()
    let idx: Uint32Array | null = null
    if (key.codes instanceof Float64Array && native.argsortF64 && !key.nullBitmap) {
      // NaNs sorted via nulls_last; without bitmap treat NaN as null-like
      idx = native.argsortF64(key.codes, key.descending, key.nullsLast)
    } else if (key.codes instanceof Int32Array && native.argsortI32) {
      idx = native.argsortI32(key.codes, key.descending, key.nullsLast, key.nullBitmap)
    }
    if (idx) {
      if (limit !== undefined && limit > 0 && limit < n) return gather(table, idx.subarray(0, limit))
      return gather(table, idx)
    }
  }

  // Native Rayon multi-key argsort for large all-f64 sorts (no per-key null bitmap; NaN = null).
  if (
    by.length > 1 &&
    n >= NATIVE_SORT_MULTI_MIN_ROWS &&
    isNativeKernelsLoaded() &&
    (limit === undefined || limit >= n)
  ) {
    const native = getNativeKernels()
    if (native.argsortMultiF64 && keys.every((k) => k.codes instanceof Float64Array && !k.nullBitmap)) {
      const idx = native.argsortMultiF64(
        keys.map((k) => k.codes as Float64Array),
        keys.map((k) => k.descending),
        keys.map((k) => k.nullsLast),
      )
      if (limit !== undefined && limit > 0 && limit < n) return gather(table, idx.subarray(0, limit))
      return gather(table, idx)
    }
  }

  // Successive stable radix: last key first, then earlier keys.
  let idx: Uint32Array | undefined
  for (let k = keys.length - 1; k >= 0; k--) {
    const key = keys[k]!
    idx = argsortNumeric(key.codes, n, key.nullBitmap, key.descending, idx, key.nullsLast)
  }
  const final = idx!
  if (limit !== undefined && limit > 0 && limit < n) {
    return gather(table, final.subarray(0, limit))
  }
  return gather(table, final)
}

function topKIndices(
  idx: Uint32Array,
  data: NumArr,
  bitmap: Uint8Array | undefined,
  descending: boolean,
  k: number,
): void {
  const heapIdx = new Uint32Array(k)
  const heapVal = new Float64Array(k)
  let size = 0
  const worse = (a: number, b: number) => (descending ? a < b : a > b)

  // Root = worst of the current top-k (min-heap by "worse").
  const siftUp = (pos: number) => {
    while (pos > 0) {
      const parent = (pos - 1) >> 1
      if (!worse(heapVal[pos]!, heapVal[parent]!)) break
      const ti = heapIdx[parent]!
      const tv = heapVal[parent]!
      heapIdx[parent] = heapIdx[pos]!
      heapVal[parent] = heapVal[pos]!
      heapIdx[pos] = ti
      heapVal[pos] = tv
      pos = parent
    }
  }
  const siftDown = (pos: number) => {
    for (;;) {
      let worst = pos
      const l = pos * 2 + 1
      const r = l + 1
      if (l < size && worse(heapVal[l]!, heapVal[worst]!)) worst = l
      if (r < size && worse(heapVal[r]!, heapVal[worst]!)) worst = r
      if (worst === pos) break
      const ti = heapIdx[worst]!
      const tv = heapVal[worst]!
      heapIdx[worst] = heapIdx[pos]!
      heapVal[worst] = heapVal[pos]!
      heapIdx[pos] = ti
      heapVal[pos] = tv
      pos = worst
    }
  }

  for (let i = 0; i < idx.length; i++) {
    if (bitmap && !isValid(bitmap, i)) continue
    const v = data[i]!
    if (size < k) {
      heapIdx[size] = i
      heapVal[size] = v
      siftUp(size++)
    } else if (worse(heapVal[0]!, v)) {
      heapIdx[0] = i
      heapVal[0] = v
      siftDown(0)
    }
  }

  // Argsort heap slots without allocating {i,v} objects (k is small).
  const order = new Uint32Array(size)
  for (let i = 0; i < size; i++) order[i] = i
  order.sort((ia, ib) => {
    const d = heapVal[ia]! - heapVal[ib]!
    return descending ? (d > 0 ? -1 : d < 0 ? 1 : 0) : d > 0 ? 1 : d < 0 ? -1 : 0
  })
  for (let i = 0; i < size; i++) idx[i] = heapIdx[order[i]!]!
}

type Acc = {
  count: number
  sum: number
  min: number
  max: number
  /** Welford M2 for std/var */
  m2: number
  /** Collected values for median/quantile */
  values: number[] | null
}

function freshAcc(needValues: boolean): Acc {
  return {
    count: 0,
    sum: 0,
    min: Infinity,
    max: -Infinity,
    m2: 0,
    values: needValues ? [] : null,
  }
}

function updateAcc(g: Acc, op: AggKind, v: number | null): void {
  if (op === 'count') {
    g.count++
    return
  }
  if (v === null) return
  if (g.values) g.values.push(v)
  const prevCount = g.count
  const meanPrev = prevCount === 0 ? 0 : g.sum / prevCount
  g.count++
  g.sum += v
  if (v < g.min) g.min = v
  if (v > g.max) g.max = v
  const mean = g.sum / g.count
  g.m2 += (v - meanPrev) * (v - mean)
}

function finishAcc(g: Acc, op: AggKind, q = 0.5, method?: QuantileMethod): number {
  switch (op) {
    case 'sum':
      return g.sum
    case 'mean':
      return g.count ? g.sum / g.count : NaN
    case 'min':
      return g.min === Infinity ? NaN : g.min
    case 'max':
      return g.max === -Infinity ? NaN : g.max
    case 'count':
      return g.count
    case 'var':
      return g.count > 1 ? g.m2 / (g.count - 1) : g.count === 1 ? 0 : NaN
    case 'std': {
      const v = g.count > 1 ? g.m2 / (g.count - 1) : g.count === 1 ? 0 : NaN
      return Number.isFinite(v) ? Math.sqrt(v) : NaN
    }
    case 'median':
    case 'quantile': {
      if (!g.values || g.values.length === 0) return NaN
      const qq = op === 'median' ? 0.5 : Math.min(1, Math.max(0, q))
      return quantileFromValues(g.values, qq, g.values.length, method)
    }
    default:
      return NaN
  }
}

type ParsedAgg = { name: string; op: AggKind; col: Column; data: NumArr | null; q?: number; qm?: QuantileMethod }

const AGG_FAST = new Set([
  'sum',
  'mean',
  'min',
  'max',
  'count',
  'std',
  'var',
  'median',
  'quantile',
])

function parseAggs(table: TableView, aggs: Array<{ name: string; expr: ExprNode }>): ParsedAgg[] | null {
  const parsed: ParsedAgg[] = []
  for (const agg of aggs) {
    if (agg.expr.type !== 'agg' || agg.expr.expr.type !== 'col') return null
    if (!AGG_FAST.has(agg.expr.op)) return null
    const c = getColumn(table, agg.expr.expr.name)
    parsed.push({
      name: agg.name,
      op: agg.expr.op,
      col: c,
      data: numericView(c),
      q: agg.expr.q,
      qm: agg.expr.qm,
    })
  }
  if (parsed.some((p) => p.op !== 'count' && !p.data)) return null
  return parsed
}

function needsValueList(op: AggKind): boolean {
  return op === 'median' || op === 'quantile'
}

function applyAggsAt(accs: Acc[], parsed: ParsedAgg[], i: number): void {
  for (let a = 0; a < parsed.length; a++) {
    const agg = parsed[a]!
    if (agg.op === 'count') {
      accs[a]!.count++
      continue
    }
    if (agg.col.nullBitmap && !isValid(agg.col.nullBitmap, i)) continue
    updateAcc(accs[a]!, agg.op, agg.data![i]!)
  }
}

const OP_SUM = 0
const OP_MEAN = 1
const OP_MIN = 2
const OP_MAX = 3
const OP_COUNT = 4
const OP_STD = 5
const OP_VAR = 6
const OP_MEDIAN = 7
const OP_QUANTILE = 8

/**
 * Struct-of-arrays accumulators for dense group codes: numeric op codes and
 * typed accumulator buffers replace per-group objects and string switches in
 * the row loop.
 */
type DenseAggs = {
  naggs: number
  ops: Int32Array
  qs: Float64Array
  /** Per-agg quantile definition (undefined = linear / type 7). */
  qms: Array<QuantileMethod | undefined>
  /** For median/quantile: index of agg that owns the shared value store (−1 = this agg). */
  valueShare: Int32Array
  datas: Array<NumArr | null>
  bitmaps: Array<Uint8Array | undefined>
  counts: Float64Array
  sums: Float64Array
  mins: Float64Array | null
  maxs: Float64Array | null
  m2s: Float64Array | null
  /** Per-(pack,agg) typed value buffers for median/quantile (fallback when no hist) */
  valueStores: Array<ValStore | null> | null
  valueCache: Array<Map<number, number> | null> | null
  /** Packed per-group histograms for integer quantile column: hist[pack * histSpan + (v - histMin)] */
  hist: Uint32Array | null
  histMin: number
  histSpan: number
  /** Owner agg index using hist, or -1 */
  histOwner: number
}

type ValStore = { data: Float64Array; len: number }

function newValStore(hint = 64): ValStore {
  return { data: new Float64Array(Math.max(8, hint)), len: 0 }
}

function pushVal(store: ValStore, v: number): void {
  if (store.len >= store.data.length) {
    const nd = new Float64Array(store.data.length << 1)
    nd.set(store.data)
    store.data = nd
  }
  store.data[store.len++] = v
}

function probeIntSpan(
  data: NumArr,
  bitmap: Uint8Array | undefined,
  n: number,
): { min: number; max: number } | null {
  let min = Infinity
  let max = -Infinity
  let seen = 0
  for (let i = 0; i < n; i++) {
    if (bitmap && !isValid(bitmap, i)) continue
    const v = data[i]!
    if (!Number.isInteger(v)) return null
    if (v < min) min = v
    if (v > max) max = v
    seen++
  }
  if (!seen) return null
  const span = max - min + 1
  if (span <= 0 || span > 2_000_000 || span > Math.max(seen * 4, 1024)) return null
  return { min, max }
}

function makeDenseAggs(parsed: ParsedAgg[], card: number, nRows = 0): DenseAggs {
  const naggs = parsed.length
  const ops = new Int32Array(naggs)
  const qs = new Float64Array(naggs)
  const qms: Array<QuantileMethod | undefined> = new Array(naggs)
  const valueShare = new Int32Array(naggs)
  valueShare.fill(-1)
  let needMin = false
  let needMax = false
  let needM2 = false
  let needVals = false
  for (let a = 0; a < naggs; a++) {
    switch (parsed[a]!.op) {
      case 'sum':
        ops[a] = OP_SUM
        break
      case 'mean':
        ops[a] = OP_MEAN
        break
      case 'min':
        ops[a] = OP_MIN
        needMin = true
        break
      case 'max':
        ops[a] = OP_MAX
        needMax = true
        break
      case 'std':
        ops[a] = OP_STD
        needM2 = true
        break
      case 'var':
        ops[a] = OP_VAR
        needM2 = true
        break
      case 'median':
        ops[a] = OP_MEDIAN
        qs[a] = 0.5
        qms[a] = parsed[a]!.qm
        needVals = true
        break
      case 'quantile':
        ops[a] = OP_QUANTILE
        qs[a] = parsed[a]!.q ?? 0.5
        qms[a] = parsed[a]!.qm
        needVals = true
        break
      default:
        ops[a] = OP_COUNT
    }
  }
  if (needVals) {
    for (let a = 0; a < naggs; a++) {
      if (ops[a] !== OP_MEDIAN && ops[a] !== OP_QUANTILE) continue
      for (let b = 0; b < a; b++) {
        if (
          (ops[b] === OP_MEDIAN || ops[b] === OP_QUANTILE) &&
          parsed[b]!.data === parsed[a]!.data &&
          parsed[b]!.col === parsed[a]!.col
        ) {
          valueShare[a] = b
          break
        }
      }
    }
  }

  let hist: Uint32Array | null = null
  let histMin = 0
  let histSpan = 0
  let histOwner = -1
  let useStores = needVals

  if (needVals && nRows > 0) {
    const owners: number[] = []
    for (let a = 0; a < naggs; a++) {
      if ((ops[a] === OP_MEDIAN || ops[a] === OP_QUANTILE) && valueShare[a]! < 0) owners.push(a)
    }
    if (owners.length === 1) {
      const a = owners[0]!
      const data = parsed[a]!.data
      if (data) {
        const range = probeIntSpan(data, parsed[a]!.col.nullBitmap, nRows)
        if (range) {
          histSpan = range.max - range.min + 1
          if (card * histSpan <= 32_000_000) {
            hist = new Uint32Array(card * histSpan)
            histMin = range.min
            histOwner = a
            useStores = false
          }
        }
      }
    }
  }

  const size = card * naggs
  const mins = needMin ? new Float64Array(size).fill(Infinity) : null
  const maxs = needMax ? new Float64Array(size).fill(-Infinity) : null
  const m2s = needM2 ? new Float64Array(size) : null
  const valueStores = useStores ? new Array<ValStore | null>(size).fill(null) : null
  const valueCache = needVals ? new Array<Map<number, number> | null>(size).fill(null) : null
  return {
    naggs,
    ops,
    qs,
    qms,
    valueShare,
    datas: parsed.map((p) => p.data),
    bitmaps: parsed.map((p) => p.col.nullBitmap),
    counts: new Float64Array(size),
    sums: new Float64Array(size),
    mins,
    maxs,
    m2s,
    valueStores,
    valueCache,
    hist,
    histMin,
    histSpan,
    histOwner,
  }
}

function denseUpdate(acc: DenseAggs, pack: number, i: number): void {
  const {
    naggs,
    ops,
    datas,
    bitmaps,
    counts,
    sums,
    mins,
    maxs,
    m2s,
    valueStores,
    valueShare,
    hist,
    histMin,
    histSpan,
    histOwner,
  } = acc
  const base = pack * naggs
  for (let a = 0; a < naggs; a++) {
    const op = ops[a]!
    if (op === OP_COUNT) {
      counts[base + a]!++
      continue
    }
    const bm = bitmaps[a]
    if (bm && !isValid(bm, i)) continue
    const v = datas[a]![i]!
    const idx = base + a
    const prev = counts[idx]!
    counts[idx]!++
    sums[idx]! += v
    if (mins && op === OP_MIN && v < mins[idx]!) mins[idx] = v
    if (maxs && op === OP_MAX && v > maxs[idx]!) maxs[idx] = v
    if (m2s && (op === OP_STD || op === OP_VAR)) {
      const meanPrev = prev === 0 ? 0 : (sums[idx]! - v) / prev
      const mean = sums[idx]! / counts[idx]!
      m2s[idx]! += (v - meanPrev) * (v - mean)
    }
    if ((op === OP_MEDIAN || op === OP_QUANTILE) && valueShare[a]! < 0) {
      if (hist && a === histOwner) {
        hist[pack * histSpan + (v - histMin)]!++
      } else if (valueStores) {
        let store = valueStores[idx]
        if (!store) {
          store = newValStore(64)
          valueStores[idx] = store
        }
        pushVal(store, v)
      }
    }
  }
}

/** Quantile from a contiguous histogram slice. */
function quantileFromHist(
  hist: Uint32Array,
  offset: number,
  span: number,
  min: number,
  count: number,
  q: number,
  method?: QuantileMethod,
): number {
  if (count <= 0) return NaN
  if (count === 1) {
    for (let o = 0; o < span; o++) if (hist[offset + o]) return min + o
    return NaN
  }
  const pos = quantilePos(count, q, method)
  const loRank = Math.floor(pos)
  const hiRank = Math.ceil(pos)
  let seen = 0
  let loVal = min
  let hiVal = min
  let gotLo = false
  let gotHi = false
  for (let o = 0; o < span; o++) {
    seen += hist[offset + o]!
    if (!gotLo && seen > loRank) {
      loVal = min + o
      gotLo = true
    }
    if (!gotHi && seen > hiRank) {
      hiVal = min + o
      gotHi = true
      break
    }
  }
  if (loRank === hiRank) return loVal
  const w = pos - loRank
  return loVal * (1 - w) + hiVal * w
}

/** Quantile from unsorted values: integer histogram when cheap, else quickselect. */
function quantileFromValues(
  values: ArrayLike<number> | number[],
  q: number,
  len = values.length,
  method?: QuantileMethod,
): number {
  if (len <= 0) return NaN
  if (len === 1) return values[0]!
  let min = Infinity
  let max = -Infinity
  let allInt = true
  for (let i = 0; i < len; i++) {
    const v = values[i]!
    if (v < min) min = v
    if (v > max) max = v
    if (allInt && !Number.isInteger(v)) allInt = false
  }
  if (allInt && Number.isFinite(min) && Number.isFinite(max)) {
    const span = max - min + 1
    if (span > 0 && span <= Math.max(len * 4, 1024) && span <= 2_000_000) {
      const hist = new Uint32Array(span)
      for (let i = 0; i < len; i++) hist[values[i]! - min]!++
      return quantileFromHist(hist, 0, span, min, len, q, method)
    }
  }
  const buf =
    values instanceof Float64Array
      ? values.slice(0, len)
      : Float64Array.from({ length: len }, (_, i) => Number(values[i]))
  return quantileFromSortedOrSelect(buf, len, q, false, method)
}

function denseFinish(acc: DenseAggs, pack: number, a: number): number {
  const idx = pack * acc.naggs + a
  switch (acc.ops[a]!) {
    case OP_SUM:
      return acc.sums[idx]!
    case OP_MEAN:
      return acc.counts[idx]! ? acc.sums[idx]! / acc.counts[idx]! : NaN
    case OP_MIN: {
      const v = acc.mins![idx]!
      return v === Infinity ? NaN : v
    }
    case OP_MAX: {
      const v = acc.maxs![idx]!
      return v === -Infinity ? NaN : v
    }
    case OP_VAR: {
      const c = acc.counts[idx]!
      return c > 1 ? acc.m2s![idx]! / (c - 1) : c === 1 ? 0 : NaN
    }
    case OP_STD: {
      const c = acc.counts[idx]!
      const v = c > 1 ? acc.m2s![idx]! / (c - 1) : c === 1 ? 0 : NaN
      return Number.isFinite(v) ? Math.sqrt(v) : NaN
    }
    case OP_MEDIAN:
    case OP_QUANTILE: {
      const share = acc.valueShare[a]!
      const ownerAgg = share >= 0 ? share : a
      const listIdx = pack * acc.naggs + ownerAgg
      const qq = acc.qs[a]!
      const qm = acc.qms[a]
      // cache key: q ∈ [0, 1] for linear, q + 2 for the Minitab definition
      const ck = qm === 'minitab' ? qq + 2 : qq
      let cache = acc.valueCache?.[listIdx]
      if (!cache) {
        cache = new Map()
        if (acc.valueCache) acc.valueCache[listIdx] = cache
      }
      let hit = cache.get(ck)
      if (hit !== undefined) return hit

      if (acc.hist && ownerAgg === acc.histOwner) {
        const count = acc.counts[listIdx]!
        hit = quantileFromHist(acc.hist, pack * acc.histSpan, acc.histSpan, acc.histMin, count, qq, qm)
      } else {
        const store = acc.valueStores?.[listIdx]
        if (!store || store.len === 0) return NaN
        hit = quantileFromValues(store.data, qq, store.len, qm)
      }
      cache.set(ck, hit)
      return hit
    }
    default:
      return acc.counts[idx]!
  }
}

/** Shared dense-group materialization for category groupby + fused filter→groupby. */
function materializeDenseGroups(
  used: Uint8Array,
  card: number,
  acc: DenseAggs,
  parsed: ParsedAgg[],
  writeKeys: (pack: number, row: number, keyOuts: string[][]) => void,
  keyNames: string[],
): TableView {
  let size = 0
  for (let c = 0; c < card; c++) if (used[c]) size++
  const keyOuts = keyNames.map(() => new Array<string>(size))
  const aggCols = parsed.map(() => new Float64Array(size))
  let row = 0
  for (let pack = 0; pack < card; pack++) {
    if (!used[pack]) continue
    writeKeys(pack, row, keyOuts)
    for (let a = 0; a < parsed.length; a++) aggCols[a]![row] = denseFinish(acc, pack, a)
    row++
  }
  return tableFromColumns([
    ...keyNames.map((name, k) => ({
      field: { name, dtype: 'utf8' as const, nullable: false },
      data: keyOuts[k]!,
    })),
    ...parsed.map((agg, i) => ({
      field: { name: agg.name, dtype: 'f64' as const, nullable: false },
      data: aggCols[i]!,
    })),
  ])
}

/** Native Rayon path: category codes + sum/mean/count only (no nulls on keys/values). */
function tryNativeDenseGroupBy(
  table: TableView,
  keyCols: Column[],
  cards: number[],
  strides: number[],
  card: number,
  parsed: ParsedAgg[],
  keys: string[],
): TableView | null {
  const n = table.numRows
  if (n < NATIVE_GROUPBY_MIN_ROWS || !isNativeKernelsLoaded()) return null
  const kernels = getNativeKernels()
  if (!kernels.groupbySumsF64 && !kernels.groupbyMinmaxF64) return null

  const opsOk = parsed.every(
    (p) => p.op === 'sum' || p.op === 'mean' || p.op === 'count' || p.op === 'min' || p.op === 'max',
  )
  if (!opsOk) return null
  if (keyCols.some((c) => c.nullBitmap)) return null
  if (parsed.some((p) => p.op !== 'count' && p.col.nullBitmap)) return null

  const wantSums = parsed.some((p) => p.op === 'sum' || p.op === 'mean' || p.op === 'count')
  const wantMinMax = parsed.some((p) => p.op === 'min' || p.op === 'max')
  if (wantSums && !kernels.groupbySumsF64) return null
  if (wantMinMax && !kernels.groupbyMinmaxF64) return null

  const codeBufs = keyCols.map((c) => c.data as Uint32Array)
  let codes: Uint32Array
  if (keyCols.length === 1) {
    codes = codeBufs[0]!
  } else {
    codes = new Uint32Array(n)
    for (let i = 0; i < n; i++) {
      let pack = 0
      for (let k = 0; k < keyCols.length; k++) {
        const code = codeBufs[k]![i]!
        if (code >= cards[k]!) {
          pack = card // mark invalid; native skips code >= card
          break
        }
        pack += code * strides[k]!
      }
      codes[i] = pack
    }
  }

  // Deduplicate value columns (same physical buffer can back sum+mean / min+max).
  const colSlots: Float64Array[] = []
  const colIndex = new Map<object, number>()
  const aggColIx: number[] = []
  for (const p of parsed) {
    if (p.op === 'count') {
      aggColIx.push(-1)
      continue
    }
    const src = p.data!
    let ix = colIndex.get(src)
    if (ix === undefined) {
      ix = colSlots.length
      colIndex.set(src, ix)
      colSlots.push(src instanceof Float64Array ? src : Float64Array.from(src as ArrayLike<number>))
    }
    aggColIx.push(ix)
  }

  let sumsOut: { sums: Float64Array; counts: Float64Array; used: Uint8Array } | null = null
  let mmOut: { mins: Float64Array; maxs: Float64Array; counts: Float64Array; used: Uint8Array } | null =
    null
  try {
    if (wantSums) sumsOut = kernels.groupbySumsF64!(codes, card, colSlots)
    if (wantMinMax) mmOut = kernels.groupbyMinmaxF64!(codes, card, colSlots)
  } catch {
    return null
  }

  const used = (sumsOut ?? mmOut)!.used
  const counts = (sumsOut ?? mmOut)!.counts
  const acc = makeDenseAggs(parsed, card, n)
  for (let c = 0; c < card; c++) {
    if (!used[c]) continue
    const rowCount = counts[c]!
    for (let a = 0; a < parsed.length; a++) {
      const idx = c * parsed.length + a
      const op = acc.ops[a]!
      if (op === OP_COUNT) {
        acc.counts[idx] = rowCount
        continue
      }
      const ci = aggColIx[a]!
      if (op === OP_MIN && mmOut) {
        acc.mins![idx] = mmOut.mins[ci * card + c]!
        acc.counts[idx] = rowCount
        continue
      }
      if (op === OP_MAX && mmOut) {
        acc.maxs![idx] = mmOut.maxs[ci * card + c]!
        acc.counts[idx] = rowCount
        continue
      }
      if (sumsOut) {
        const sum = sumsOut.sums[ci * card + c]!
        acc.sums[idx] = sum
        acc.counts[idx] = rowCount
      }
    }
  }

  return materializeDenseGroups(used, card, acc, parsed, (pack, row, keyOuts) => {
    let rem = pack
    for (let k = keyCols.length - 1; k >= 0; k--) {
      const stride = strides[k]!
      const code = Math.floor(rem / stride)
      rem = rem % stride
      keyOuts[k]![row] = keyCols[k]!.dictionary![code]!
    }
  }, keys)
}

/** Dense groupby for 1..k category keys packed into one code (product ≤ 65_536). */
function tryDenseCategoryGroupBy(
  table: TableView,
  keys: string[],
  parsed: ParsedAgg[],
): TableView | null {
  const keyCols: Column[] = []
  const cards: number[] = []
  const strides: number[] = []
  let card = 1
  for (const name of keys) {
    const col = getColumn(table, name)
    if (col.field.dtype !== 'category' || !col.dictionary) return null
    const c = col.dictionary.length
    if (c <= 0 || card > Math.floor(65_536 / c)) return null
    strides.push(card)
    cards.push(c)
    card *= c
    keyCols.push(col)
  }

  const nativeHit = tryNativeDenseGroupBy(table, keyCols, cards, strides, card, parsed, keys)
  if (nativeHit) return nativeHit

  const n = table.numRows
  const used = new Uint8Array(card)
  const acc = makeDenseAggs(parsed, card, n)
  const codeBufs = keyCols.map((c) => c.data as Uint32Array)

  if (keyCols.length === 1) {
    const codes = codeBufs[0]!
    const keyBitmap = keyCols[0]!.nullBitmap
    const card0 = cards[0]!
    for (let i = 0; i < n; i++) {
      if (keyBitmap && !isValid(keyBitmap, i)) continue
      const code = codes[i]!
      if (code >= card0) continue
      used[code] = 1
      denseUpdate(acc, code, i)
    }
  } else {
    outer: for (let i = 0; i < n; i++) {
      let pack = 0
      for (let k = 0; k < keyCols.length; k++) {
        const col = keyCols[k]!
        if (col.nullBitmap && !isValid(col.nullBitmap, i)) continue outer
        const code = codeBufs[k]![i]!
        if (code >= cards[k]!) continue outer
        pack += code * strides[k]!
      }
      used[pack] = 1
      denseUpdate(acc, pack, i)
    }
  }

  return materializeDenseGroups(used, card, acc, parsed, (pack, row, keyOuts) => {
    let rem = pack
    for (let k = keyCols.length - 1; k >= 0; k--) {
      const stride = strides[k]!
      const code = Math.floor(rem / stride)
      rem = rem % stride
      keyOuts[k]![row] = keyCols[k]!.dictionary![code]!
    }
  }, keys)
}

export function tryFastGroupBy(
  table: TableView,
  keys: string[],
  aggs: Array<{ name: string; expr: ExprNode }>,
): TableView | null {
  if (keys.length === 0) return null
  const parsed = parseAggs(table, aggs)
  if (!parsed) return null

  const dense = tryDenseCategoryGroupBy(table, keys, parsed)
  if (dense) return dense

  if (keys.length !== 1) return null
  const keyCol = getColumn(table, keys[0]!)
  const n = table.numRows

  const multi = new Map<string | number, Acc[]>()
  const keyIsStr = keyCol.field.dtype === 'utf8'
  const needVals = parsed.some((p) => needsValueList(p.op))

  for (let i = 0; i < n; i++) {
    if (keyCol.nullBitmap && !isValid(keyCol.nullBitmap, i)) continue
    let key: string | number
    if (keyCol.field.dtype === 'utf8') key = (keyCol.data as string[])[i]!
    else {
      const kd = numericView(keyCol)
      if (!kd) return null
      key = kd[i]!
    }

    let accs = multi.get(key)
    if (!accs) {
      accs = parsed.map((p) => freshAcc(needVals && needsValueList(p.op)))
      multi.set(key, accs)
    }
    applyAggsAt(accs, parsed, i)
  }

  const size = multi.size
  const keyOut = allocateData(keyIsStr ? 'utf8' : keyCol.field.dtype, size)
  const aggCols = parsed.map(() => new Float64Array(size))
  let row = 0
  for (const [key, accs] of multi) {
    if (keyIsStr) (keyOut as string[])[row] = String(key)
    else (keyOut as Float64Array | Int32Array | Uint32Array)[row] = Number(key)
    for (let a = 0; a < parsed.length; a++) {
      aggCols[a]![row] = finishAcc(accs[a]!, parsed[a]!.op, parsed[a]!.q ?? 0.5, parsed[a]!.qm)
    }
    row++
  }

  return tableFromColumns([
    {
      field: { name: keyCol.field.name, dtype: keyIsStr ? 'utf8' : keyCol.field.dtype, nullable: false },
      data: keyOut,
    },
    ...parsed.map((agg, i) => ({
      field: { name: agg.name, dtype: 'f64' as const, nullable: false },
      data: aggCols[i]!,
    })),
  ])
}

function mergeAcc(into: Acc, from: Acc): void {
  if (from.count === 0) return
  if (into.count === 0) {
    into.count = from.count
    into.sum = from.sum
    into.min = from.min
    into.max = from.max
    into.m2 = from.m2
    if (from.values) into.values = from.values.slice()
    return
  }
  const n1 = into.count
  const n2 = from.count
  const mean1 = into.sum / n1
  const mean2 = from.sum / n2
  const n = n1 + n2
  into.m2 = into.m2 + from.m2 + ((mean1 - mean2) * (mean1 - mean2) * n1 * n2) / n
  into.sum += from.sum
  into.count = n
  if (from.min < into.min) into.min = from.min
  if (from.max > into.max) into.max = from.max
  if (into.values && from.values) {
    for (const v of from.values) into.values.push(v)
  }
}

/**
 * Chunked / one-pass groupBy under a memory budget: accumulate per-group stats without
 * retaining every row index, merging partial maps across row batches.
 */
export function tryChunkedGroupBy(
  table: TableView,
  keys: string[],
  aggs: Array<{ name: string; expr: ExprNode }>,
  chunkRows: number,
): TableView | null {
  if (keys.length === 0) return null
  const parsed = parseAggs(table, aggs)
  if (!parsed) return null
  const needVals = parsed.some((p) => needsValueList(p.op))
  const keyCols = keys.map((k) => getColumn(table, k))
  const global = new Map<string, Acc[]>()
  const keySample = new Map<string, number>() // first row index for key materialization

  const n = table.numRows
  const step = Math.max(1, chunkRows)
  for (let start = 0; start < n; start += step) {
    const end = Math.min(n, start + step)
    const local = new Map<string, Acc[]>()
    for (let i = start; i < end; i++) {
      let nullKey = false
      const parts: string[] = []
      for (const kc of keyCols) {
        if (kc.nullBitmap && !isValid(kc.nullBitmap, i)) {
          nullKey = true
          parts.push('∅')
        } else if (kc.field.dtype === 'category' && kc.dictionary) {
          parts.push(kc.dictionary[Number((kc.data as Uint32Array)[i])] ?? '∅')
        } else if (kc.field.dtype === 'utf8') {
          parts.push(String((kc.data as string[])[i]))
        } else {
          const kd = numericView(kc)
          if (!kd) return null
          parts.push(String(kd[i]))
        }
      }
      void nullKey
      const key = parts.join('\0')
      let accs = local.get(key)
      if (!accs) {
        accs = parsed.map((p) => freshAcc(needVals && needsValueList(p.op)))
        local.set(key, accs)
        if (!keySample.has(key)) keySample.set(key, i)
      }
      applyAggsAt(accs, parsed, i)
    }
    for (const [key, accs] of local) {
      const g = global.get(key)
      if (!g) {
        global.set(key, accs)
      } else {
        for (let a = 0; a < accs.length; a++) mergeAcc(g[a]!, accs[a]!)
      }
    }
  }

  const size = global.size
  const outCols: Column[] = []
  for (let ki = 0; ki < keys.length; ki++) {
    const src = keyCols[ki]!
    const data = allocateData(src.field.dtype, size)
    const nullBitmap = new Uint8Array(Math.ceil(size / 8) || 1)
    let anyNull = false
    let row = 0
    for (const key of global.keys()) {
      const srcIdx = keySample.get(key)!
      if (!isValid(src.nullBitmap, srcIdx)) {
        anyNull = true
      } else {
        setValid(nullBitmap, row, true)
        if (src.field.dtype === 'utf8') {
          ;(data as string[])[row] = (src.data as string[])[srcIdx]!
        } else if (src.field.dtype === 'category') {
          ;(data as Uint32Array)[row] = (src.data as Uint32Array)[srcIdx]!
        } else {
          const kd = numericView(src)!
          ;(data as Float64Array | Int32Array | Uint32Array)[row] = kd[srcIdx]!
        }
      }
      row++
    }
    outCols.push({
      field: { ...src.field },
      data,
      nullBitmap: anyNull ? nullBitmap : undefined,
      dictionary: src.dictionary ? [...src.dictionary] : undefined,
    })
  }

  for (let a = 0; a < parsed.length; a++) {
    const agg = parsed[a]!
    const data = new Float64Array(size)
    let row = 0
    for (const accs of global.values()) {
      data[row++] = finishAcc(accs[a]!, agg.op, agg.q ?? 0.5, agg.qm)
    }
    outCols.push({
      field: { name: agg.name, dtype: 'f64', nullable: false },
      data,
    })
  }

  return tableFromColumns(outCols)
}

/**
 * Equi-join on one numeric key.
 * Dense Int32 probe + identity-left reuse + fused right gather (no extra index copy).
 * Supports inner/left (full row assemble) and semi/anti (left-only filter).
 */
/** Right-side key codes expressed in the left dictionary; null when the column cannot be remapped. */
function remapKeyToDictionary(col: Column, dict: string[]): NumArr | null {
  const index = new Map<string, number>()
  for (let i = 0; i < dict.length; i++) if (!index.has(dict[i]!)) index.set(dict[i]!, i)
  const n = col.data.length
  if (col.field.dtype === 'category' && col.dictionary) {
    const rd = col.dictionary
    let same = rd === dict || rd.length === dict.length
    if (same && rd !== dict) for (let i = 0; i < rd.length && same; i++) same = rd[i] === dict[i]
    if (same) return col.data as NumArr
    const codeMap = new Int32Array(rd.length)
    for (let i = 0; i < rd.length; i++) codeMap[i] = index.get(rd[i]!) ?? -1
    const codes = col.data as Uint32Array
    const out = new Float64Array(n)
    for (let i = 0; i < n; i++) out[i] = codeMap[codes[i]!] ?? -1
    return out
  }
  if (col.field.dtype === 'utf8') {
    const strs = col.data as string[]
    const out = new Float64Array(n)
    for (let i = 0; i < n; i++) out[i] = index.get(strs[i]!) ?? -1
    return out
  }
  return null
}

function keepJoinColumns(table: TableView, keep?: readonly string[]): TableView {
  if (!keep) return table
  const byName = new Map(table.columns.map((c) => [c.field.name, c]))
  const ordered: Column[] = []
  for (const name of keep) {
    const c = byName.get(name)
    if (c) ordered.push(c)
  }
  return tableFromColumns(ordered)
}

/** Set by tryFastJoin when a specialized kernel runs; cleared at start of each attempt. */
export let lastFastJoinKernel: { kernel: string; reason?: string } | null = null

export function tryFastJoin(
  left: TableView,
  right: TableView,
  leftOn: string[],
  rightOn: string[],
  how: 'inner' | 'left' | 'right' | 'outer' | 'semi' | 'anti',
  lSuffix = '',
  rSuffix = '_right',
  keep?: readonly string[],
): TableView | null {
  lastFastJoinKernel = null
  if (how !== 'inner' && how !== 'left' && how !== 'semi' && how !== 'anti') return null
  if (leftOn.length !== 1 || rightOn.length !== 1) return null

  const lCol = getColumn(left, leftOn[0]!)
  const rCol = getColumn(right, rightOn[0]!)
  const lData = numericView(lCol)
  let rData = numericView(rCol)
  if (!lData || !rData) return null
  // Category keys are compared by dictionary code, which is only meaningful inside one dictionary. Two
  // frames built separately encode the same strings under different codes, so the right key is remapped
  // into the left dictionary first (a string absent from it can never match: code -1).
  if (lCol.field.dtype === 'category' || rCol.field.dtype === 'category') {
    if (lCol.field.dtype !== 'category' || !lCol.dictionary) return null
    const remapped = remapKeyToDictionary(rCol, lCol.dictionary)
    if (!remapped) return null
    rData = remapped
  }

  const rn = right.numRows
  const ln = left.numRows

  let rMin = Infinity
  let rMax = -Infinity
  let rCount = 0
  for (let i = 0; i < rn; i++) {
    if (rCol.nullBitmap && !isValid(rCol.nullBitmap, i)) continue
    const k = rData[i]!
    if (k < rMin) rMin = k
    if (k > rMax) rMax = k
    rCount++
  }
  if (rCount === 0) {
    if (how === 'inner' || how === 'semi') return tableFromColumns(left.columns.map((c) => takeColumn(c, [])))
    if (how === 'anti') return left
  }

  const span = rMax - rMin + 1
  const useDense = Number.isFinite(rMin) && span > 0 && span <= Math.max(rCount * 4, 1_048_576) && span <= 50_000_000

  // This fast path stores one build-row per key. Duplicate right keys need the
  // generic join (Map → number[]); returning null preserves many-match semantics
  // when the optimizer swaps a non-unique side onto the build (right).
  // semi/anti only need existence, so duplicates on the build side are fine.
  const buildMustBeUnique = how === 'inner' || how === 'left'

  let dense: Int32Array | null = null
  let hash: Map<number, number> | null = null
  if (useDense) {
    const nativeBuild = getNativeKernels().joinBuildDenseI32
    if (
      nativeBuild &&
      rn >= NATIVE_JOIN_BUILD_MIN_ROWS &&
      rData instanceof Int32Array &&
      !rCol.nullBitmap &&
      isNativeKernelsLoaded()
    ) {
      if (buildMustBeUnique) {
        const seen = new Uint8Array(span)
        for (let i = 0; i < rn; i++) {
          const off = rData[i]! - (rMin | 0)
          if (off < 0 || off >= span) continue
          if (seen[off]) return null
          seen[off] = 1
        }
      }
      dense = nativeBuild(rData, rMin | 0)
    } else {
      dense = new Int32Array(span)
      dense.fill(-1)
      for (let i = 0; i < rn; i++) {
        if (rCol.nullBitmap && !isValid(rCol.nullBitmap, i)) continue
        const off = rData[i]! - rMin
        if (buildMustBeUnique && dense[off] !== -1) return null
        dense[off] = i
      }
    }
  } else {
    hash = new Map()
    for (let i = 0; i < rn; i++) {
      if (rCol.nullBitmap && !isValid(rCol.nullBitmap, i)) continue
      const k = rData[i]!
      if (buildMustBeUnique && hash.has(k)) return null
      hash.set(k, i)
    }
  }

  const probe = (k: number): number => {
    if (dense) {
      const off = k - rMin
      if (off < 0 || off >= dense.length) return -1
      return dense[off]!
    }
    return hash!.get(k) ?? -1
  }

  if (how === 'semi' || how === 'anti') {
    if (
      dense &&
      ln >= NATIVE_JOIN_MIN_ROWS &&
      !lCol.nullBitmap &&
      lData instanceof Int32Array &&
      isNativeKernelsLoaded()
    ) {
      const semi = getNativeKernels().joinSemiDenseI32
      if (semi) {
        const idx = semi(lData, dense, rMin | 0, how === 'semi')
        lastFastJoinKernel = {
          kernel: 'native:joinSemiDenseI32',
          reason: `probe left=${ln} ≥ NATIVE_JOIN_MIN_ROWS=${NATIVE_JOIN_MIN_ROWS}; build(right)=${rn}`,
        }
        const taken = takeTable(left, idx)
        return keepJoinColumns(taken, keep)
      }
    }
    const idx = new Uint32Array(ln)
    let j = 0
    const wantHit = how === 'semi'
    for (let i = 0; i < ln; i++) {
      if (lCol.nullBitmap && !isValid(lCol.nullBitmap, i)) {
        if (!wantHit) idx[j++] = i
        continue
      }
      const hit = probe(lData[i]!) >= 0
      if (hit === wantHit) idx[j++] = i
    }
    lastFastJoinKernel = { kernel: 'js:joinSemiAnti' }
    return keepJoinColumns(takeTable(left, idx.subarray(0, j)), keep)
  }

  // Right columns to emit (skip join key when names collide / same)
  const leftNames = new Set(left.schema.map((f) => f.name))
  const rightNames = new Set(right.schema.map((f) => f.name))
  const rightEmit = right.columns.filter(
    (col) => !(rightOn[0] === col.field.name && leftOn[0] === col.field.name),
  )

  const renameLeft = (col: Column): Column => {
    const sharedKey = leftOn[0] === col.field.name && rightOn[0] === col.field.name
    if (sharedKey || !rightNames.has(col.field.name) || !lSuffix) return col
    return { ...col, field: { ...col.field, name: `${col.field.name}${lSuffix}` } }
  }
  const rightOutName = (name: string): string =>
    leftNames.has(name) ? `${name}${rSuffix}` : name

  // Single left probe: over-allocate to ln, then subarray. Identity-left when every row matches.
  const leftIdx = new Uint32Array(ln)
  let rightIdx = new Int32Array(ln)
  let outCount = 0
  let identityLeft = how === 'inner' && !lCol.nullBitmap
  let probed = false

  if (
    dense &&
    ln >= NATIVE_JOIN_MIN_ROWS &&
    !lCol.nullBitmap &&
    lData instanceof Int32Array &&
    isNativeKernelsLoaded()
  ) {
    const probeNative = getNativeKernels().joinProbeDenseI32
    if (probeNative) {
      const probedRight = probeNative(lData, dense, rMin | 0)
      probed = true
      lastFastJoinKernel = {
        kernel: 'native:joinProbeDenseI32',
        reason: `probe left=${ln} ≥ NATIVE_JOIN_MIN_ROWS=${NATIVE_JOIN_MIN_ROWS}; build(right)=${rn}`,
      }
      if (how === 'inner') {
        for (let i = 0; i < ln; i++) {
          const r = probedRight[i]!
          if (r < 0) {
            identityLeft = false
            continue
          }
          leftIdx[outCount] = i
          rightIdx[outCount++] = r
        }
        if (identityLeft) {
          outCount = ln
          // Copy to satisfy Int32Array<ArrayBuffer> vs ArrayBufferLike under newer lib typings.
          rightIdx = new Int32Array(probedRight)
        }
      } else {
        for (let i = 0; i < ln; i++) {
          leftIdx[i] = i
          rightIdx[i] = probedRight[i]!
        }
        outCount = ln
        identityLeft = false
      }
    }
  }

  if (!probed) {
    for (let i = 0; i < ln; i++) {
      if (lCol.nullBitmap && !isValid(lCol.nullBitmap, i)) {
        identityLeft = false
        if (how === 'left') {
          leftIdx[outCount] = i
          rightIdx[outCount++] = -1
        }
        continue
      }
      const r = probe(lData[i]!)
      if (r < 0) {
        identityLeft = false
        if (how === 'left') {
          leftIdx[outCount] = i
          rightIdx[outCount++] = -1
        }
        continue
      }
      leftIdx[outCount] = i
      rightIdx[outCount++] = r
    }
    if (identityLeft && outCount !== ln) identityLeft = false
  }

  const leftGather = identityLeft ? null : leftIdx.subarray(0, outCount)
  const rightGather = rightIdx.subarray(0, outCount)

  const outCols: Column[] = []
  const keepSet = keep ? new Set(keep) : null
  const want = (name: string) => !keepSet || keepSet.has(name)

  if (identityLeft) {
    for (const col of left.columns) {
      const renamed = renameLeft(col)
      if (want(renamed.field.name)) outCols.push(renamed)
    }
  } else {
    for (const col of left.columns) {
      const renamed = renameLeft(takeColumn(col, leftGather!))
      if (want(renamed.field.name)) outCols.push(renamed)
    }
  }

  for (const col of rightEmit) {
    const name = rightOutName(col.field.name)
    if (!want(name)) continue
    if (how === 'inner') {
      const taken = takeColumn(col, rightGather)
      outCols.push({ ...taken, field: { ...taken.field, name } })
      continue
    }
    const gatherIdx = new Uint32Array(outCount)
    let anyNull = false
    const nullBitmap = new Uint8Array(Math.ceil(outCount / 8) || 1)
    for (let i = 0; i < outCount; i++) {
      const r = rightGather[i]!
      if (r < 0) {
        gatherIdx[i] = 0
        anyNull = true
      } else {
        gatherIdx[i] = r
        if (!col.nullBitmap || isValid(col.nullBitmap, r)) setValid(nullBitmap, i, true)
        else anyNull = true
      }
    }
    const taken = takeColumn(col, gatherIdx)
    outCols.push({
      ...taken,
      field: { ...taken.field, name, nullable: anyNull || taken.field.nullable },
      nullBitmap: anyNull ? nullBitmap : taken.nullBitmap,
    })
  }
  if (keep) {
    const byName = new Map(outCols.map((c) => [c.field.name, c]))
    const ordered: Column[] = []
    for (const name of keep) {
      const c = byName.get(name)
      if (c) ordered.push(c)
    }
    if (!lastFastJoinKernel) {
      lastFastJoinKernel = {
        kernel: useDense ? 'js:joinDense' : 'js:joinHash',
        reason:
          ln < NATIVE_JOIN_MIN_ROWS
            ? `probe left=${ln} < NATIVE_JOIN_MIN_ROWS=${NATIVE_JOIN_MIN_ROWS}; build(right)=${rn}`
            : `build(right)=${rn}`,
      }
    }
    return tableFromColumns(ordered)
  }
  if (!lastFastJoinKernel) {
    lastFastJoinKernel = {
      kernel: useDense ? 'js:joinDense' : 'js:joinHash',
      reason:
        ln < NATIVE_JOIN_MIN_ROWS
          ? `probe left=${ln} < NATIVE_JOIN_MIN_ROWS=${NATIVE_JOIN_MIN_ROWS}; build(right)=${rn}`
          : `build(right)=${rn}`,
    }
  }
  return tableFromColumns(outCols)
}

/**
 * Lazy-style fusion: filter ∧ groupBy without materializing the filtered table.
 */
export function tryFusedFilterGroupBy(
  table: TableView,
  predicate: ExprNode,
  keys: string[],
  aggs: Array<{ name: string; expr: ExprNode }>,
): TableView | null {
  if (keys.length !== 1) return null
  const keyCol = getColumn(table, keys[0]!)
  if (keyCol.field.dtype !== 'category' || !keyCol.dictionary || keyCol.dictionary.length > 65_536) {
    return null
  }

  const resolved = resolveCmps(table, predicate)
  if (!resolved) return null

  const parsed = parseAggs(table, aggs)
  if (!parsed) return null

  const codes = keyCol.data as Uint32Array
  const dict = keyCol.dictionary
  const card = dict.length
  const used = new Uint8Array(card)
  const n = table.numRows
  const acc = makeDenseAggs(parsed, card, n)

  const dualGt =
    resolved.length === 2 &&
    resolved[0]!.op === 'gt' &&
    resolved[1]!.op === 'gt' &&
    !resolved[0]!.bitmap &&
    !resolved[1]!.bitmap

  if (dualGt) {
    const a = resolved[0]!.data
    const b = resolved[1]!.data
    const la = resolved[0]!.lit
    const lb = resolved[1]!.lit
    for (let i = 0; i < n; i++) {
      if (a[i]! <= la || b[i]! <= lb) continue
      if (keyCol.nullBitmap && !isValid(keyCol.nullBitmap, i)) continue
      const code = codes[i]!
      if (code >= card) continue
      used[code] = 1
      denseUpdate(acc, code, i)
    }
  } else {
    outer: for (let i = 0; i < n; i++) {
      for (const r of resolved) if (!cmpAt(r, i)) continue outer
      if (keyCol.nullBitmap && !isValid(keyCol.nullBitmap, i)) continue
      const code = codes[i]!
      if (code >= card) continue
      used[code] = 1
      denseUpdate(acc, code, i)
    }
  }

  return materializeDenseGroups(used, card, acc, parsed, (pack, row, keyOuts) => {
    keyOuts[0]![row] = dict[pack]!
  }, [keyCol.field.name])
}

/**
 * Filter ∧ unique in one pass: only rows that pass the predicate enter the uniqueness map.
 * Avoids materializing a full filtered table when the predicate is cmp / vectorizable.
 * Key encoding matches uniqueTableInMemory (codes for category, not dictionary labels).
 */
export function tryFusedFilterUnique(
  table: TableView,
  predicate: ExprNode,
  columns: string[] | undefined,
  keep: 'first' | 'last' | 'none',
): TableView | null {
  const n = table.numRows
  if (n === 0) return gather(table, [])
  const cols = columns ?? table.schema.map((f) => f.name)
  if (cols.length === 0) return null

  const resolved = resolveCmps(table, predicate)
  let mask: ReturnType<typeof evalVec> | null = null
  if (!resolved) {
    mask = evalVec(table, predicate, n)
    if (!mask) return null
  }

  const passes = (i: number): boolean => {
    if (resolved) {
      for (const r of resolved) if (!cmpAt(r, i)) return false
      return true
    }
    return truthy(mask!, i)
  }

  const seen = new Map<string, number>()
  const order: string[] = []
  for (let i = 0; i < n; i++) {
    if (!passes(i)) continue
    const key = cols
      .map((name) => {
        const c = getColumn(table, name)
        if (!isValid(c.nullBitmap, i)) return '∅'
        return String(getValue(c.data, i))
      })
      .join('\0')
    if (!seen.has(key)) {
      seen.set(key, i)
      order.push(key)
    } else if (keep === 'last') {
      seen.set(key, i)
    } else if (keep === 'none') {
      seen.set(key, -1)
    }
  }
  const indices = order.map((k) => seen.get(k)!).filter((i) => i >= 0)
  return gather(table, indices)
}

/**
 * Filter ∧ sort ∧ limit without a full filtered intermediate when possible.
 * Prefers one-pass top-k over rows that pass the predicate; otherwise fast-filter then limited sort.
 */
export function tryFusedFilterSortLimit(
  table: TableView,
  predicate: ExprNode,
  by: Array<{ expr: ExprNode; descending: boolean; nullsLast?: boolean }>,
  limit: number,
): TableView | null {
  const n = table.numRows
  if (limit <= 0) return gather(table, [])
  if (n === 0) return gather(table, [])

  // One-pass top-k: cmp/vector filter ∧ single sort-key column (nulls last, no nulls).
  if (by.length === 1 && by[0]!.expr.type === 'col' && by[0]!.nullsLast !== false) {
    const col = getColumn(table, (by[0]!.expr as { type: 'col'; name: string }).name)
    const sortKey = sortKeyCodes(col)

    if (sortKey && !sortKey.nullBitmap) {
      const data = sortKey.codes
      const resolved = resolveCmps(table, predicate)
      let mask: ReturnType<typeof evalVec> | null = null
      if (!resolved) {
        mask = evalVec(table, predicate, n)
      }
      if (resolved || mask) {
        const descending = by[0]!.descending
        const heapIdx = new Uint32Array(limit)
        const heapVal = new Float64Array(limit)
        let size = 0
        const worse = (a: number, b: number) => (descending ? a < b : a > b)
        const siftUp = (pos: number) => {
          while (pos > 0) {
            const parent = (pos - 1) >> 1
            if (!worse(heapVal[pos]!, heapVal[parent]!)) break
            const ti = heapIdx[parent]!
            const tv = heapVal[parent]!
            heapIdx[parent] = heapIdx[pos]!
            heapVal[parent] = heapVal[pos]!
            heapIdx[pos] = ti
            heapVal[pos] = tv
            pos = parent
          }
        }
        const siftDown = (pos: number) => {
          for (;;) {
            let worst = pos
            const l = pos * 2 + 1
            const r = l + 1
            if (l < size && worse(heapVal[l]!, heapVal[worst]!)) worst = l
            if (r < size && worse(heapVal[r]!, heapVal[worst]!)) worst = r
            if (worst === pos) break
            const ti = heapIdx[worst]!
            const tv = heapVal[worst]!
            heapIdx[worst] = heapIdx[pos]!
            heapVal[worst] = heapVal[pos]!
            heapIdx[pos] = ti
            heapVal[pos] = tv
            pos = worst
          }
        }
        outer: for (let i = 0; i < n; i++) {
          if (resolved) {
            for (const r of resolved) if (!cmpAt(r, i)) continue outer
          } else if (!truthy(mask!, i)) continue
          const v = data[i]!
          if (size < limit) {
            heapIdx[size] = i
            heapVal[size] = v
            siftUp(size++)
          } else if (worse(heapVal[0]!, v)) {
            heapIdx[0] = i
            heapVal[0] = v
            siftDown(0)
          }
        }
        const order = new Uint32Array(size)
        for (let i = 0; i < size; i++) order[i] = i
        order.sort((ia, ib) => {
          const d = heapVal[ia]! - heapVal[ib]!
          return descending ? (d > 0 ? -1 : d < 0 ? 1 : 0) : d > 0 ? 1 : d < 0 ? -1 : 0
        })
        const idx = new Uint32Array(size)
        for (let i = 0; i < size; i++) idx[i] = heapIdx[order[i]!]!
        return gather(table, idx)
      }
    }
  }

  // Fallback: fast filter, then limited sort on the reduced table.
  const filtered = tryFastFilter(table, predicate)
  if (!filtered) return null
  const sorted = tryFastSort(filtered, by, limit)
  return sorted
}

/**
 * Rolling window on a numeric column. Sliding sum/mean/count; min/max rescan window.
 * Reuses input columns by reference (no full-frame clone).
 */
export function tryFastRolling(
  table: TableView,
  name: string,
  column: string,
  window: number,
  agg: AggKind,
): TableView | null {
  if (window < 1) return null
  if (!['sum', 'mean', 'min', 'max', 'count'].includes(agg)) return null
  const col = getColumn(table, column)
  const src = numericView(col)
  if (!src) return null

  const n = table.numRows
  const out = new Float64Array(n)
  const nullBitmap = new Uint8Array(Math.ceil(n / 8) || 1)
  let anyNull = false
  const bitmap = col.nullBitmap

  if (agg === 'count') {
    let cnt = 0
    for (let i = 0; i < n; i++) {
      if (!bitmap || isValid(bitmap, i)) cnt++
      const leave = i - window
      if (leave >= 0 && (!bitmap || isValid(bitmap, leave))) cnt--
      out[i] = cnt
      setValid(nullBitmap, i, true)
    }
  } else if (agg === 'sum' || agg === 'mean') {
    let sum = 0
    let cnt = 0
    for (let i = 0; i < n; i++) {
      if (!bitmap || isValid(bitmap, i)) {
        sum += src[i]!
        cnt++
      }
      const leave = i - window
      if (leave >= 0 && (!bitmap || isValid(bitmap, leave))) {
        sum -= src[leave]!
        cnt--
      }
      if (cnt === 0) {
        anyNull = true
      } else {
        out[i] = agg === 'mean' ? sum / cnt : sum
        setValid(nullBitmap, i, true)
      }
    }
  } else {
    // min / max — O(n·W) typed scan, no per-row allocations
    for (let i = 0; i < n; i++) {
      const start = Math.max(0, i - window + 1)
      let best = agg === 'min' ? Infinity : -Infinity
      let found = false
      for (let j = start; j <= i; j++) {
        if (bitmap && !isValid(bitmap, j)) continue
        const v = src[j]!
        found = true
        if (agg === 'min') {
          if (v < best) best = v
        } else if (v > best) best = v
      }
      if (!found) anyNull = true
      else {
        out[i] = best
        setValid(nullBitmap, i, true)
      }
    }
  }

  return tableFromColumns([
    ...table.columns,
    {
      field: { name, dtype: 'f64', nullable: anyNull },
      data: out,
      nullBitmap: anyNull ? nullBitmap : undefined,
    },
  ])
}

/**
 * Unique / dedupe. Dense Int32 probe when key span is small; else typed Map.
 */
export function tryFastUnique(
  table: TableView,
  columns: string[] | undefined,
  keep: 'first' | 'last' | 'none',
): TableView | null {
  const cols = columns ?? table.schema.map((f) => f.name)
  if (cols.length === 0) return null

  // Single numeric / category key (bench: user_id)
  if (cols.length === 1) {
    const col = getColumn(table, cols[0]!)
    const data = numericView(col)
    if (!data) return null
    const n = table.numRows
    const bitmap = col.nullBitmap

    let kMin = Infinity
    let kMax = -Infinity
    let valid = 0
    let nullRow = -1
    let allInt = true
    for (let i = 0; i < n; i++) {
      if (bitmap && !isValid(bitmap, i)) {
        // Use === -1 (not < 0): keep:'none' marks duplicates as -2, which must not be reset.
        if (nullRow === -1) nullRow = i
        else if (keep === 'last') nullRow = i
        else if (keep === 'none') nullRow = -2
        continue
      }
      const v = data[i]!
      if (!Number.isInteger(v)) allInt = false
      if (v < kMin) kMin = v
      if (v > kMax) kMax = v
      valid++
    }
    if (valid === 0) {
      if (nullRow >= 0) return gather(table, [nullRow])
      if (nullRow === -2) return gather(table, [])
      return gather(table, [])
    }

    const span = kMax - kMin + 1
    const useDense =
      allInt &&
      Number.isFinite(kMin) &&
      Number.isInteger(kMin) &&
      Number.isInteger(kMax) &&
      span > 0 &&
      span <= Math.max(valid * 4, 1_048_576) &&
      span <= 50_000_000

    if (useDense) {
      const first = new Int32Array(span)
      const last = keep === 'first' ? null : new Int32Array(span)
      first.fill(-1)
      if (last) last.fill(-1)
      const orderKeys = new Int32Array(Math.min(valid, span) + (nullRow !== -1 ? 1 : 0))
      let orderLen = 0
      let nullOrder = -1
      for (let i = 0; i < n; i++) {
        if (bitmap && !isValid(bitmap, i)) {
          if (nullOrder < 0) {
            nullOrder = orderLen
            orderKeys[orderLen++] = -1 // sentinel for null key
          }
          continue
        }
        const off = data[i]! - kMin
        if (first[off]! < 0) {
          first[off] = i
          orderKeys[orderLen++] = off
        }
        if (last) last[off] = i
      }
      let outCount = 0
      for (let o = 0; o < orderLen; o++) {
        const off = orderKeys[o]!
        if (off < 0) {
          if (nullRow >= 0) outCount++
          continue
        }
        if (keep === 'none' && last && last[off]! !== first[off]!) continue
        outCount++
      }
      const idx = new Uint32Array(outCount)
      let j = 0
      for (let o = 0; o < orderLen; o++) {
        const off = orderKeys[o]!
        if (off < 0) {
          if (nullRow >= 0) idx[j++] = nullRow
          continue
        }
        if (keep === 'none' && last && last[off]! !== first[off]!) continue
        idx[j++] = keep === 'last' && last ? last[off]! : first[off]!
      }
      return gather(table, idx)
    }

    // Hash on numbers (+ null key)
    const NULL_KEY = Number.NaN
    const seen = new Map<number, number>()
    const order: number[] = []
    const hasNullKey = (k: number) => Number.isNaN(k)
    for (let i = 0; i < n; i++) {
      if (bitmap && !isValid(bitmap, i)) {
        if (!seen.has(NULL_KEY)) {
          seen.set(NULL_KEY, i)
          order.push(NULL_KEY)
        } else if (keep === 'last') {
          seen.set(NULL_KEY, i)
        } else if (keep === 'none') {
          seen.set(NULL_KEY, -1)
        }
        continue
      }
      const key = data[i]!
      if (!seen.has(key)) {
        seen.set(key, i)
        order.push(key)
      } else if (keep === 'last') {
        seen.set(key, i)
      } else if (keep === 'none') {
        seen.set(key, -1)
      }
    }
    const idx = new Uint32Array(order.length)
    let j = 0
    for (const k of order) {
      const i = hasNullKey(k) ? seen.get(NULL_KEY)! : seen.get(k)!
      if (i >= 0) idx[j++] = i
    }
    return gather(table, idx.subarray(0, j))
  }

  // Multi category keys → packed dense unique
  // Native parallel unique for all-f64 columns (keep:'first' only).
  if (keep === 'first' && isNativeKernelsLoaded() && table.numRows >= NATIVE_UNIQUE_MIN_ROWS) {
    const nativeUnique = getNativeKernels().uniqueF64
    if (nativeUnique) {
      const f64Cols: Float64Array[] = []
      let ok = true
      for (const name of cols) {
        const c = getColumn(table, name)
        if (c.dictionary || c.nullBitmap) { ok = false; break }
        const data = numericView(c)
        if (!(data instanceof Float64Array)) { ok = false; break }
        f64Cols.push(data)
      }
      if (ok && f64Cols.length > 0) {
        const idx = nativeUnique(f64Cols)
        return gather(table, idx)
      }
    }
  }

  const keyCols: Column[] = []
  const cards: number[] = []
  const strides: number[] = []
  let card = 1
  for (const name of cols) {
    const c = getColumn(table, name)
    if (c.field.dtype !== 'category' || !c.dictionary) return null
    const len = c.dictionary.length
    if (len <= 0 || card > Math.floor(65_536 / len)) return null
    strides.push(card)
    cards.push(len)
    card *= len
    keyCols.push(c)
  }

  const n = table.numRows
  const first = new Int32Array(card)
  const last = keep === 'first' ? null : new Int32Array(card)
  first.fill(-1)
  if (last) last.fill(-1)
  const orderPack = new Int32Array(card)
  let orderLen = 0
  const codeBufs = keyCols.map((c) => c.data as Uint32Array)

  outer: for (let i = 0; i < n; i++) {
    let pack = 0
    for (let k = 0; k < keyCols.length; k++) {
      const col = keyCols[k]!
      if (col.nullBitmap && !isValid(col.nullBitmap, i)) continue outer
      const code = codeBufs[k]![i]!
      if (code >= cards[k]!) continue outer
      pack += code * strides[k]!
    }
    if (first[pack]! < 0) {
      first[pack] = i
      orderPack[orderLen++] = pack
    }
    if (last) last[pack] = i
  }

  let outCount = 0
  for (let o = 0; o < orderLen; o++) {
    const p = orderPack[o]!
    if (keep === 'none' && last && last[p]! !== first[p]!) continue
    outCount++
  }
  const idx = new Uint32Array(outCount)
  let j = 0
  for (let o = 0; o < orderLen; o++) {
    const p = orderPack[o]!
    if (keep === 'none' && last && last[p]! !== first[p]!) continue
    idx[j++] = keep === 'last' && last ? last[p]! : first[p]!
  }
  return gather(table, idx)
}

function swapNum(a: Float64Array, i: number, j: number): void {
  const t = a[i]!
  a[i] = a[j]!
  a[j] = t
}

/** In-place quickselect: a[k] becomes the k-th smallest (0-based). */
function selectK(a: Float64Array, left: number, right: number, k: number): number {
  while (left < right) {
    const pivotIdx = left + ((right - left) >> 1)
    const pivot = a[pivotIdx]!
    swapNum(a, pivotIdx, right)
    let store = left
    for (let i = left; i < right; i++) {
      if (a[i]! < pivot) {
        swapNum(a, store, i)
        store++
      }
    }
    swapNum(a, store, right)
    if (k === store) return a[k]!
    if (k < store) right = store - 1
    else left = store + 1
  }
  return a[left]!
}

function quantileFromSortedOrSelect(
  vals: Float64Array,
  count: number,
  p: number,
  sorted: boolean,
  method?: QuantileMethod,
): number {
  if (!count) return NaN
  const idx = quantilePos(count, p, method)
  const lo = Math.floor(idx)
  const hi = Math.ceil(idx)
  if (sorted) {
    if (lo === hi) return vals[lo]!
    return vals[lo]! * (hi - idx) + vals[hi]! * (idx - lo)
  }
  if (lo === hi) return selectK(vals, 0, count - 1, lo)
  const vh = selectK(vals, 0, count - 1, hi)
  const vl = selectK(vals, 0, hi, lo)
  return vl * (hi - idx) + vh * (idx - lo)
}

const QUANTILES = [0.25, 0.5, 0.75] as const

/**
 * Resolve all three quantiles from a value histogram in one ascending scan.
 * Each quantile needs the floor and ceil order statistics, so up to six ranks
 * are answered by a single cumulative walk instead of six partial ones.
 */
function histQuantiles(
  hist: Uint32Array,
  span: number,
  min: number,
  count: number,
  method?: QuantileMethod,
): [number, number, number] {
  const positions = QUANTILES.map((p) => quantilePos(count, p, method))
  const ranks = new Set<number>()
  for (const pos of positions) {
    ranks.add(Math.floor(pos))
    ranks.add(Math.ceil(pos))
  }
  const wanted = [...ranks].sort((a, b) => a - b)
  const resolved = new Map<number, number>()

  let seen = 0
  let ri = 0
  for (let o = 0; o < span && ri < wanted.length; o++) {
    seen += hist[o]!
    while (ri < wanted.length && seen > wanted[ri]!) {
      resolved.set(wanted[ri]!, min + o)
      ri++
    }
  }
  while (ri < wanted.length) {
    resolved.set(wanted[ri]!, min + span - 1)
    ri++
  }

  const out = positions.map((pos) => {
    const lo = Math.floor(pos)
    const hi = Math.ceil(pos)
    const vl = resolved.get(lo)!
    if (lo === hi) return vl
    return vl * (hi - pos) + resolved.get(hi)! * (pos - lo)
  })
  return [out[0]!, out[1]!, out[2]!]
}

function histogramSpan(min: number, max: number, count: number): number | null {
  if (!Number.isInteger(min) || !Number.isInteger(max)) return null
  const span = max - min + 1
  if (span <= 0 || span > Math.min(2_000_000, Math.max(count * 4, 65_536))) return null
  return span
}

/** Typed describe: Welford + histogram quantiles (ints) or select/sort fallback. */
export function tryFastDescribe(table: TableView, method?: QuantileMethod): TableView | null {
  const stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
  const numeric = table.columns.filter((c) => isNumeric(c.field.dtype) || c.field.dtype === 'datetime')
  if (numeric.length === 0) {
    return tableFromColumns([{ field: { name: 'stat', dtype: 'utf8', nullable: false }, data: stats }])
  }

  const out: Column[] = [{ field: { name: 'stat', dtype: 'utf8', nullable: false }, data: stats }]

  for (const col of numeric) {
    const src = numericView(col)
    if (!src) return null
    const n = table.numRows
    const bitmap = col.nullBitmap

    // Integer buffers can go straight to the histogram; float buffers only if
    // every value is integral, otherwise fractional offsets would be dropped.
    const dt = col.field.dtype
    const intDtype = dt === 'i32' || dt === 'u32'
    let allInt = true

    // Pass 1: extent + sum (+ integrality for float buffers). No per-element
    // division, so the loop stays free of a serial dependency on a divide.
    let sum = 0
    let min = Infinity
    let max = -Infinity
    let count = 0
    if (bitmap) {
      for (let i = 0; i < n; i++) {
        if (!isValid(bitmap, i)) continue
        const v = src[i]!
        if (v < min) min = v
        if (v > max) max = v
        sum += v
        count++
        if (!intDtype && allInt && !Number.isInteger(v)) allInt = false
      }
    } else if (intDtype) {
      count = n
      for (let i = 0; i < n; i++) {
        const v = src[i]!
        if (v < min) min = v
        if (v > max) max = v
        sum += v
      }
    } else {
      count = n
      for (let i = 0; i < n; i++) {
        const v = src[i]!
        if (v < min) min = v
        if (v > max) max = v
        sum += v
        if (allInt && !Number.isInteger(v)) allInt = false
      }
    }
    const mean = count ? sum / count : NaN

    // Pass 2: variance, fused with either histogram fill or the value copy the
    // quantile fallback needs.
    const span = count && allInt ? histogramSpan(min, max, count) : null
    const hist = span === null ? null : new Uint32Array(span)
    const buf = span === null && count ? new Float64Array(count) : null
    let m2 = 0
    let m = 0
    for (let i = 0; i < n; i++) {
      if (bitmap && !isValid(bitmap, i)) continue
      const v = src[i]!
      const d = v - mean
      m2 += d * d
      if (hist) hist[v - min]!++
      else if (buf) buf[m++] = v
    }
    const std = count > 1 ? Math.sqrt(m2 / (count - 1)) : NaN

    let q25 = NaN
    let q50 = NaN
    let q75 = NaN
    if (count) {
      if (hist && span !== null) {
        ;[q25, q50, q75] = histQuantiles(hist, span, min, count, method)
      } else if (buf) {
        if (count <= 10_000) {
          buf.sort()
          q25 = quantileFromSortedOrSelect(buf, count, 0.25, true, method)
          q50 = quantileFromSortedOrSelect(buf, count, 0.5, true, method)
          q75 = quantileFromSortedOrSelect(buf, count, 0.75, true, method)
        } else {
          // One buffer: high quantile first, then lower on the left partition
          const i75 = quantilePos(count, 0.75, method)
          const lo75 = Math.floor(i75)
          const hi75 = Math.ceil(i75)
          const vh75 = selectK(buf, 0, count - 1, hi75)
          const vl75 = lo75 === hi75 ? vh75 : selectK(buf, 0, hi75, lo75)
          q75 = lo75 === hi75 ? vh75 : vl75 * (hi75 - i75) + vh75 * (i75 - lo75)

          const i50 = quantilePos(count, 0.5, method)
          const lo50 = Math.floor(i50)
          const hi50 = Math.ceil(i50)
          const right50 = Math.min(hi75, count - 1)
          const vh50 = selectK(buf, 0, right50, hi50)
          const vl50 = lo50 === hi50 ? vh50 : selectK(buf, 0, hi50, lo50)
          q50 = lo50 === hi50 ? vh50 : vl50 * (hi50 - i50) + vh50 * (i50 - lo50)

          const i25 = quantilePos(count, 0.25, method)
          const lo25 = Math.floor(i25)
          const hi25 = Math.ceil(i25)
          const right25 = Math.min(hi50, count - 1)
          const vh25 = selectK(buf, 0, right25, hi25)
          const vl25 = lo25 === hi25 ? vh25 : selectK(buf, 0, hi25, lo25)
          q25 = lo25 === hi25 ? vh25 : vl25 * (hi25 - i25) + vh25 * (i25 - lo25)
        }
      }
    }

    out.push({
      field: { name: col.field.name, dtype: 'f64', nullable: true },
      data: new Float64Array([
        count,
        count ? mean : NaN,
        std,
        count ? min : NaN,
        q25,
        q50,
        q75,
        count ? max : NaN,
      ]),
    })
  }
  return tableFromColumns(out)
}

type FastVal =
  | { t: 'scalar'; v: number }
  | { t: 'str_scalar'; v: string }
  | {
      t: 'vec'
      data: ArrayLike<number>
      bitmap: Uint8Array | undefined
      kind: 'num' | 'bool'
      owned: boolean
      src?: Column
    }
  | { t: 'str_vec'; data: string[]; bitmap: Uint8Array | undefined; owned: boolean }

function bitAnd(a: Uint8Array | undefined, b: Uint8Array | undefined): Uint8Array | undefined {
  if (!a) return b
  if (!b) return a
  const len = Math.min(a.length, b.length)
  const out = new Uint8Array(len)
  for (let i = 0; i < len; i++) out[i] = a[i]! & b[i]!
  return out
}

function hasNulls(bitmap: Uint8Array, n: number): boolean {
  const fullBytes = n >> 3
  for (let i = 0; i < fullBytes; i++) if (bitmap[i]! !== 0xff) return true
  const restBits = n & 7
  if (restBits === 0) return false
  const mask = (1 << restBits) - 1
  return (bitmap[fullBytes]! & mask) !== mask
}

/** Truthiness of a vector cell, matching evalExprScalar (null is falsy for and/or/not). */
function truthy(v: FastVal, i: number): boolean {
  if (v.t === 'scalar') return v.v !== 0
  if (v.t === 'str_scalar') return v.v.length > 0
  if (v.t === 'str_vec') {
    if (v.bitmap && !isValid(v.bitmap, i)) return false
    return Boolean(v.data[i])
  }
  if (v.bitmap && !isValid(v.bitmap, i)) return false
  return v.data[i] !== 0
}

export { applyMathOp, roundHalfAway } from './math.js'

const MATH_OPS = new Set<string>(['sqrt', 'log', 'log10', 'log2', 'exp', 'round', 'floor', 'ceil', 'sign'])
const ARITH_OPS = new Set(['add', 'sub', 'mul', 'div', 'mod', 'pow'])
const CMP_OPS = new Set(['eq', 'neq', 'gt', 'gte', 'lt', 'lte'])

function arithVecVec(op: string, a: ArrayLike<number>, b: ArrayLike<number>, n: number): Float64Array {
  const out = new Float64Array(n)
  switch (op) {
    case 'add':
      for (let i = 0; i < n; i++) out[i] = a[i]! + b[i]!
      break
    case 'sub':
      for (let i = 0; i < n; i++) out[i] = a[i]! - b[i]!
      break
    case 'mul':
      for (let i = 0; i < n; i++) out[i] = a[i]! * b[i]!
      break
    case 'div':
      for (let i = 0; i < n; i++) out[i] = a[i]! / b[i]!
      break
    case 'pow':
      for (let i = 0; i < n; i++) out[i] = a[i]! ** b[i]!
      break
    default:
      for (let i = 0; i < n; i++) out[i] = a[i]! % b[i]!
  }
  return out
}

function arithVecScalar(op: string, a: ArrayLike<number>, s: number, n: number, flip: boolean): Float64Array {
  const out = new Float64Array(n)
  if (flip) {
    switch (op) {
      case 'add':
        for (let i = 0; i < n; i++) out[i] = s + a[i]!
        break
      case 'sub':
        for (let i = 0; i < n; i++) out[i] = s - a[i]!
        break
      case 'mul':
        for (let i = 0; i < n; i++) out[i] = s * a[i]!
        break
      case 'div':
        for (let i = 0; i < n; i++) out[i] = s / a[i]!
        break
      case 'pow':
        for (let i = 0; i < n; i++) out[i] = s ** a[i]!
        break
      default:
        for (let i = 0; i < n; i++) out[i] = s % a[i]!
    }
    return out
  }
  switch (op) {
    case 'add':
      for (let i = 0; i < n; i++) out[i] = a[i]! + s
      break
    case 'sub':
      for (let i = 0; i < n; i++) out[i] = a[i]! - s
      break
    case 'mul':
      for (let i = 0; i < n; i++) out[i] = a[i]! * s
      break
    case 'div':
      for (let i = 0; i < n; i++) out[i] = a[i]! / s
      break
    case 'pow':
      for (let i = 0; i < n; i++) out[i] = a[i]! ** s
      break
    default:
      for (let i = 0; i < n; i++) out[i] = a[i]! % s
  }
  return out
}

function cmpVecVec(op: string, a: ArrayLike<number>, b: ArrayLike<number>, n: number): Uint8Array {
  const out = new Uint8Array(n)
  switch (op) {
    case 'eq':
      for (let i = 0; i < n; i++) out[i] = a[i]! === b[i]! ? 1 : 0
      break
    case 'neq':
      for (let i = 0; i < n; i++) out[i] = a[i]! !== b[i]! ? 1 : 0
      break
    case 'gt':
      for (let i = 0; i < n; i++) out[i] = a[i]! > b[i]! ? 1 : 0
      break
    case 'gte':
      for (let i = 0; i < n; i++) out[i] = a[i]! >= b[i]! ? 1 : 0
      break
    case 'lt':
      for (let i = 0; i < n; i++) out[i] = a[i]! < b[i]! ? 1 : 0
      break
    default:
      for (let i = 0; i < n; i++) out[i] = a[i]! <= b[i]! ? 1 : 0
  }
  return out
}

function cmpVecScalar(op: string, a: ArrayLike<number>, s: number, n: number, flip: boolean): Uint8Array {
  const eff = flip
    ? op === 'gt'
      ? 'lt'
      : op === 'gte'
        ? 'lte'
        : op === 'lt'
          ? 'gt'
          : op === 'lte'
            ? 'gte'
            : op
    : op
  const out = new Uint8Array(n)
  switch (eff) {
    case 'eq':
      for (let i = 0; i < n; i++) out[i] = a[i]! === s ? 1 : 0
      break
    case 'neq':
      for (let i = 0; i < n; i++) out[i] = a[i]! !== s ? 1 : 0
      break
    case 'gt':
      for (let i = 0; i < n; i++) out[i] = a[i]! > s ? 1 : 0
      break
    case 'gte':
      for (let i = 0; i < n; i++) out[i] = a[i]! >= s ? 1 : 0
      break
    case 'lt':
      for (let i = 0; i < n; i++) out[i] = a[i]! < s ? 1 : 0
      break
    default:
      for (let i = 0; i < n; i++) out[i] = a[i]! <= s ? 1 : 0
  }
  return out
}

function evalVec(table: TableView, expr: ExprNode, n: number): FastVal | null {
  // Arithmetic / math chains compile to one loop with no per-node temporaries (see fused.ts).
  if (expr.type === 'binary' || expr.type === 'unary') {
    const fused = tryFused(table, expr, n)
    if (fused) return { t: 'vec', data: fused.data, bitmap: fused.bitmap, kind: fused.kind, owned: true }
  }
  switch (expr.type) {
    case 'lit': {
      if (typeof expr.value === 'number') return { t: 'scalar', v: expr.value }
      if (typeof expr.value === 'boolean') return { t: 'scalar', v: expr.value ? 1 : 0 }
      if (typeof expr.value === 'string') return { t: 'str_scalar', v: expr.value }
      return null
    }
    case 'col': {
      const col = getColumn(table, expr.name)
      const d = col.field.dtype
      // bool/utf8/category have non-numeric scalar semantics in evalExprScalar
      if (d !== 'f64' && d !== 'f32' && d !== 'i32' && d !== 'u32' && d !== 'datetime') return null
      return { t: 'vec', data: col.data as NumArr, bitmap: col.nullBitmap, kind: 'num', owned: false, src: col }
    }
    case 'alias':
      return evalVec(table, expr.expr, n)
    case 'unary': {
      const inner = evalVec(table, expr.expr, n)
      if (!inner) return null
      if (expr.op === 'isNull' || expr.op === 'isNotNull') {
        if (inner.t !== 'vec' && inner.t !== 'str_vec') return null
        const want = expr.op === 'isNull' ? 1 : 0
        const out = new Uint8Array(n)
        const bm = inner.bitmap
        if (bm) for (let i = 0; i < n; i++) out[i] = isValid(bm, i) ? 1 - want : want
        else out.fill(1 - want)
        return { t: 'vec', data: out, bitmap: undefined, kind: 'bool', owned: true }
      }
      if (expr.op === 'not') {
        const out = new Uint8Array(n)
        for (let i = 0; i < n; i++) out[i] = truthy(inner, i) ? 0 : 1
        return { t: 'vec', data: out, bitmap: undefined, kind: 'bool', owned: true }
      }
      if (MATH_OPS.has(expr.op)) {
        const op = expr.op as MathOp
        const decimals = expr.decimals ?? 0
        if (inner.t === 'scalar') return { t: 'scalar', v: applyMathOp(op, inner.v, decimals, expr.base) }
        if (inner.t !== 'vec' || inner.kind !== 'num') return null
        const src = inner.data
        const out = new Float64Array(n)
        switch (op) {
          case 'sqrt':
            for (let i = 0; i < n; i++) out[i] = Math.sqrt(src[i]!)
            break
          case 'log': {
            if (expr.base === undefined) for (let i = 0; i < n; i++) out[i] = Math.log(src[i]!)
            else {
              const inv = 1 / Math.log(expr.base)
              for (let i = 0; i < n; i++) out[i] = Math.log(src[i]!) * inv
            }
            break
          }
          case 'log10':
            for (let i = 0; i < n; i++) out[i] = Math.log10(src[i]!)
            break
          case 'log2':
            for (let i = 0; i < n; i++) out[i] = Math.log2(src[i]!)
            break
          case 'exp':
            for (let i = 0; i < n; i++) out[i] = Math.exp(src[i]!)
            break
          case 'round':
            for (let i = 0; i < n; i++) out[i] = roundHalfAway(src[i]!, decimals)
            break
          case 'floor':
            for (let i = 0; i < n; i++) out[i] = Math.floor(src[i]!)
            break
          case 'ceil':
            for (let i = 0; i < n; i++) out[i] = Math.ceil(src[i]!)
            break
          case 'sign':
            for (let i = 0; i < n; i++) out[i] = Math.sign(src[i]!)
            break
        }
        return { t: 'vec', data: out, bitmap: inner.bitmap, kind: 'num', owned: true }
      }
      if (inner.t === 'scalar') {
        return { t: 'scalar', v: expr.op === 'neg' ? -inner.v : Math.abs(inner.v) }
      }
      if (inner.t !== 'vec' || inner.kind !== 'num') return null
      const src = inner.data
      const out = new Float64Array(n)
      if (expr.op === 'neg') for (let i = 0; i < n; i++) out[i] = -src[i]!
      else for (let i = 0; i < n; i++) out[i] = Math.abs(src[i]!)
      return { t: 'vec', data: out, bitmap: inner.bitmap, kind: 'num', owned: true }
    }
    case 'binary': {
      const op = expr.op
      if (op === 'and' || op === 'or') {
        const l = evalVec(table, expr.left, n)
        const r = evalVec(table, expr.right, n)
        if (!l || !r) return null
        const out = new Uint8Array(n)
        if (op === 'and') for (let i = 0; i < n; i++) out[i] = truthy(l, i) && truthy(r, i) ? 1 : 0
        else for (let i = 0; i < n; i++) out[i] = truthy(l, i) || truthy(r, i) ? 1 : 0
        return { t: 'vec', data: out, bitmap: undefined, kind: 'bool', owned: true }
      }
      if (!ARITH_OPS.has(op) && !CMP_OPS.has(op)) return null
      const l = evalVec(table, expr.left, n)
      const r = evalVec(table, expr.right, n)
      if (!l || !r) return null
      if (l.t === 'str_scalar' || l.t === 'str_vec' || r.t === 'str_scalar' || r.t === 'str_vec') return null
      if (l.t === 'vec' && l.kind !== 'num') return null
      if (r.t === 'vec' && r.kind !== 'num') return null

      const isCmp = CMP_OPS.has(op)
      if (l.t === 'scalar' && r.t === 'scalar') {
        if (isCmp) {
          const a = l.v
          const b = r.v
          const res =
            op === 'eq'
              ? a === b
              : op === 'neq'
                ? a !== b
                : op === 'gt'
                  ? a > b
                  : op === 'gte'
                    ? a >= b
                    : op === 'lt'
                      ? a < b
                      : a <= b
          return { t: 'scalar', v: res ? 1 : 0 }
        }
        const a = l.v
        const b = r.v
        const res =
          op === 'add' ? a + b : op === 'sub' ? a - b : op === 'mul' ? a * b : op === 'div' ? a / b : op === 'pow' ? a ** b : a % b
        return { t: 'scalar', v: res }
      }

      if (l.t === 'vec' && r.t === 'vec') {
        const bitmap = bitAnd(l.bitmap, r.bitmap)
        const data = isCmp ? cmpVecVec(op, l.data, r.data, n) : arithVecVec(op, l.data, r.data, n)
        return { t: 'vec', data, bitmap, kind: isCmp ? 'bool' : 'num', owned: true }
      }

      const vec = l.t === 'vec' ? l : (r as Extract<FastVal, { t: 'vec' }>)
      const scalar = l.t === 'vec' ? (r as Extract<FastVal, { t: 'scalar' }>).v : (l as Extract<FastVal, { t: 'scalar' }>).v
      const flip = l.t === 'scalar'
      const data = isCmp
        ? cmpVecScalar(op, vec.data, scalar, n, flip)
        : arithVecScalar(op, vec.data, scalar, n, flip)
      return { t: 'vec', data, bitmap: vec.bitmap, kind: isCmp ? 'bool' : 'num', owned: true }
    }
    case 'isIn': {
      const inner = evalVec(table, expr.expr, n)
      if (!inner || (inner.t !== 'vec' && inner.t !== 'scalar')) return null
      const nums = expr.values.filter((v): v is number => typeof v === 'number')
      if (nums.length !== expr.values.length) return null
      const set = new Set(nums)
      let dense: Uint8Array | null = null
      let dMin = 0
      if (nums.length > 0) {
        let mn = nums[0]!
        let mx = nums[0]!
        for (const v of nums) {
          if (v < mn) mn = v
          if (v > mx) mx = v
        }
        const span = mx - mn + 1
        if (Number.isInteger(mn) && Number.isInteger(mx) && span > 0 && span <= 1_048_576) {
          dense = new Uint8Array(span)
          dMin = mn
          for (const v of nums) dense[v - mn] = 1
        }
      }
      const hit = (v: number): boolean => {
        if (dense) {
          const off = v - dMin
          return off >= 0 && off < dense.length && dense[off] === 1
        }
        return set.has(v)
      }
      if (inner.t === 'scalar') return { t: 'scalar', v: hit(inner.v) ? 1 : 0 }
      const out = new Uint8Array(n)
      const src = inner.data
      const bm = inner.bitmap
      if (bm) {
        for (let i = 0; i < n; i++) out[i] = isValid(bm, i) && hit(src[i]!) ? 1 : 0
      } else {
        for (let i = 0; i < n; i++) out[i] = hit(src[i]!) ? 1 : 0
      }
      return { t: 'vec', data: out, bitmap: undefined, kind: 'bool', owned: true }
    }
    case 'isBetween': {
      const inner = evalVec(table, expr.expr, n)
      const lo = evalVec(table, expr.low, n)
      const hi = evalVec(table, expr.high, n)
      if (!inner || !lo || !hi) return null
      if (inner.t === 'str_scalar' || inner.t === 'str_vec') return null
      if (lo.t === 'str_scalar' || lo.t === 'str_vec') return null
      if (hi.t === 'str_scalar' || hi.t === 'str_vec') return null
      if (inner.t !== 'vec' && inner.t !== 'scalar') return null
      if (lo.t !== 'vec' && lo.t !== 'scalar') return null
      if (hi.t !== 'vec' && hi.t !== 'scalar') return null

      const closed = expr.closed ?? 'both'
      const loAt = (i: number): number | null => {
        if (lo.t === 'scalar') return lo.v
        if (lo.bitmap && !isValid(lo.bitmap, i)) return null
        return lo.data[i]!
      }
      const hiAt = (i: number): number | null => {
        if (hi.t === 'scalar') return hi.v
        if (hi.bitmap && !isValid(hi.bitmap, i)) return null
        return hi.data[i]!
      }
      const inRange = (v: number, a: number, b: number): boolean => {
        const ge = closed === 'both' || closed === 'left' ? v >= a : v > a
        const le = closed === 'both' || closed === 'right' ? v <= b : v < b
        return ge && le
      }

      if (inner.t === 'scalar' && lo.t === 'scalar' && hi.t === 'scalar') {
        return { t: 'scalar', v: inRange(inner.v, lo.v, hi.v) ? 1 : 0 }
      }

      const out = new Uint8Array(n)
      for (let i = 0; i < n; i++) {
        if (inner.t === 'vec' && inner.bitmap && !isValid(inner.bitmap, i)) {
          out[i] = 0
          continue
        }
        const v = inner.t === 'scalar' ? inner.v : inner.data[i]!
        const a = loAt(i)
        const b = hiAt(i)
        out[i] = a !== null && b !== null && inRange(v, a, b) ? 1 : 0
      }
      return { t: 'vec', data: out, bitmap: undefined, kind: 'bool', owned: true }
    }
    case 'when': {
      const preds: FastVal[] = []
      const thens: FastVal[] = []
      for (const br of expr.branches) {
        const p = evalVec(table, br.when, n)
        const t = evalVec(table, br.then, n)
        if (!p || !t) return null
        preds.push(p)
        thens.push(t)
      }
      const otherwise = evalVec(table, expr.otherwise, n)
      if (!otherwise) return null

      const allStr =
        thens.every((t) => t.t === 'str_scalar' || t.t === 'str_vec') &&
        (otherwise.t === 'str_scalar' || otherwise.t === 'str_vec')
      const allNum =
        thens.every((t) => t.t === 'scalar' || (t.t === 'vec' && t.kind === 'num')) &&
        (otherwise.t === 'scalar' || (otherwise.t === 'vec' && otherwise.kind === 'num'))

      if (allStr) {
        const out = new Array<string>(n)
        const nullBitmap = new Uint8Array(Math.ceil(n / 8) || 1)
        let anyNull = false
        for (let i = 0; i < n; i++) {
          let chosen: FastVal = otherwise
          for (let b = 0; b < preds.length; b++) {
            if (truthy(preds[b]!, i)) {
              chosen = thens[b]!
              break
            }
          }
          if (chosen.t === 'str_scalar') {
            out[i] = chosen.v
            setValid(nullBitmap, i, true)
          } else if (chosen.t === 'str_vec') {
            if (chosen.bitmap && !isValid(chosen.bitmap, i)) {
              out[i] = ''
              anyNull = true
            } else {
              out[i] = chosen.data[i]!
              setValid(nullBitmap, i, true)
            }
          } else {
            return null
          }
        }
        return { t: 'str_vec', data: out, bitmap: anyNull ? nullBitmap : undefined, owned: true }
      }

      if (!allNum) return null
      const out = new Float64Array(n)
      let bitmap: Uint8Array | undefined
      let anyNull = false
      const ensureBm = () => {
        if (!bitmap) {
          bitmap = new Uint8Array(Math.ceil(n / 8) || 1)
          bitmap.fill(0xff)
        }
      }
      for (let i = 0; i < n; i++) {
        let chosen: FastVal = otherwise
        for (let b = 0; b < preds.length; b++) {
          if (truthy(preds[b]!, i)) {
            chosen = thens[b]!
            break
          }
        }
        if (chosen.t === 'scalar') {
          out[i] = chosen.v
        } else if (chosen.t === 'vec') {
          if (chosen.bitmap && !isValid(chosen.bitmap, i)) {
            ensureBm()
            setValid(bitmap!, i, false)
            anyNull = true
            out[i] = NaN
          } else out[i] = chosen.data[i]!
        } else return null
      }
      return { t: 'vec', data: out, bitmap: anyNull ? bitmap : undefined, kind: 'num', owned: true }
    }
    case 'rowOffset': {
      const periods = expr.periods | 0
      if (periods === 0 && expr.kind === 'shift') return evalVec(table, expr.expr, n)
      const inner = evalVec(table, expr.expr, n)
      if (!inner || inner.t !== 'vec' || inner.kind !== 'num') return null
      const src = inner.data
      const srcBm = inner.bitmap
      const out = new Float64Array(n)
      const nullBitmap = new Uint8Array(Math.ceil(n / 8) || 1)
      let anyNull = false
      if (expr.kind === 'shift') {
        for (let i = 0; i < n; i++) {
          const j = i - periods
          if (j < 0 || j >= n || (srcBm && !isValid(srcBm, j))) {
            anyNull = true
            out[i] = NaN
          } else {
            setValid(nullBitmap, i, true)
            out[i] = src[j]!
          }
        }
      } else if (expr.kind === 'diff') {
        for (let i = 0; i < n; i++) {
          const j = i - periods
          if (j < 0 || j >= n || (srcBm && (!isValid(srcBm, i) || !isValid(srcBm, j)))) {
            anyNull = true
            out[i] = NaN
          } else {
            setValid(nullBitmap, i, true)
            out[i] = src[i]! - src[j]!
          }
        }
      } else {
        // pctChange
        for (let i = 0; i < n; i++) {
          const j = i - periods
          if (j < 0 || j >= n || (srcBm && (!isValid(srcBm, i) || !isValid(srcBm, j)))) {
            anyNull = true
            out[i] = NaN
            continue
          }
          const prev = src[j]!
          if (prev === 0) {
            anyNull = true
            out[i] = NaN
            continue
          }
          setValid(nullBitmap, i, true)
          out[i] = (src[i]! - prev) / prev
        }
      }
      return { t: 'vec', data: out, bitmap: anyNull ? nullBitmap : undefined, kind: 'num', owned: true }
    }
    case 'str': {
      // Bool predicates for filter; string transforms handled in tryFastStrColumn
      if (
        expr.op !== 'contains' &&
        expr.op !== 'startsWith' &&
        expr.op !== 'endsWith' &&
        expr.op !== 'len'
      ) {
        return null
      }
      let inner = expr.expr
      while (inner.type === 'alias') inner = inner.expr
      if (inner.type !== 'col') return null
      const col = getColumn(table, inner.name)
      const pattern = expr.pattern ?? ''

      if (col.field.dtype === 'category' && col.dictionary) {
        const dict = col.dictionary
        const codes = col.data as Uint32Array
        const bm = col.nullBitmap
        if (expr.op === 'len') {
          const lens = dict.map((s) => s.length)
          const out = new Float64Array(n)
          if (bm) {
            for (let i = 0; i < n; i++) out[i] = isValid(bm, i) ? lens[codes[i]!]! : 0
          } else {
            for (let i = 0; i < n; i++) out[i] = lens[codes[i]!]!
          }
          return { t: 'vec', data: out, bitmap: bm, kind: 'num', owned: true }
        }
        const flags = new Uint8Array(dict.length)
        for (let d = 0; d < dict.length; d++) {
          const s = dict[d]!
          flags[d] =
            expr.op === 'contains'
              ? s.includes(pattern)
                ? 1
                : 0
              : expr.op === 'startsWith'
                ? s.startsWith(pattern)
                  ? 1
                  : 0
                : s.endsWith(pattern)
                  ? 1
                  : 0
        }
        const out = new Uint8Array(n)
        if (bm) {
          for (let i = 0; i < n; i++) out[i] = isValid(bm, i) && flags[codes[i]!]! ? 1 : 0
        } else {
          for (let i = 0; i < n; i++) out[i] = flags[codes[i]!]!
        }
        return { t: 'vec', data: out, bitmap: undefined, kind: 'bool', owned: true }
      }

      if (col.field.dtype !== 'utf8') return null
      const data = col.data as string[]
      const bm = col.nullBitmap
      if (expr.op === 'len') {
        const out = new Float64Array(n)
        if (bm) {
          for (let i = 0; i < n; i++) out[i] = isValid(bm, i) ? (data[i] ?? '').length : 0
        } else {
          for (let i = 0; i < n; i++) out[i] = (data[i] ?? '').length
        }
        return { t: 'vec', data: out, bitmap: bm, kind: 'num', owned: true }
      }
      if (
        expr.op === 'contains' &&
        !bm &&
        n >= NATIVE_STR_MIN_ROWS &&
        isNativeKernelsLoaded()
      ) {
        const sc = getNativeKernels().strContains
        if (sc) {
          return { t: 'vec', data: sc(data, pattern), bitmap: undefined, kind: 'bool', owned: true }
        }
      }
      const out = new Uint8Array(n)
      for (let i = 0; i < n; i++) {
        if (bm && !isValid(bm, i)) {
          out[i] = 0
          continue
        }
        const s = data[i] ?? ''
        out[i] =
          expr.op === 'contains'
            ? s.includes(pattern)
              ? 1
              : 0
            : expr.op === 'startsWith'
              ? s.startsWith(pattern)
                ? 1
                : 0
              : s.endsWith(pattern)
                ? 1
                : 0
      }
      return { t: 'vec', data: out, bitmap: undefined, kind: 'bool', owned: true }
    }
    case 'dt': {
      let inner = expr.expr
      while (inner.type === 'alias') inner = inner.expr
      const src = evalVec(table, inner, n)
      if (!src || src.t !== 'vec' || src.kind !== 'num') return null
      const data = src.data
      const bm = src.bitmap
      const out = new Float64Array(n)
      const op = expr.op
      if (op === 'epochMillis') {
        for (let i = 0; i < n; i++) out[i] = data[i]!
        return { t: 'vec', data: out, bitmap: bm, kind: 'num', owned: true }
      }
      for (let i = 0; i < n; i++) {
        if (bm && !isValid(bm, i)) {
          out[i] = NaN
          continue
        }
        out[i] = dtPartFromMs(data[i]!, op)
      }
      return { t: 'vec', data: out, bitmap: bm, kind: 'num', owned: true }
    }
    default:
      return null
  }
}

/** Howard Hinnant civil_from_days — UTC y/m/d without Date allocation. */
function civilFromDays(z: number): { y: number; m: number; d: number } {
  z += 719468
  const era = Math.floor((z >= 0 ? z : z - 146096) / 146097)
  const doe = z - era * 146097
  const yoe = Math.floor((doe - Math.floor(doe / 1460) + Math.floor(doe / 36524) - Math.floor(doe / 146096)) / 365)
  const y = yoe + era * 400
  const doy = doe - (365 * yoe + Math.floor(yoe / 4) - Math.floor(yoe / 100))
  const mp = Math.floor((5 * doy + 2) / 153)
  const d = doy - Math.floor((153 * mp + 2) / 5) + 1
  const m = mp < 10 ? mp + 3 : mp - 9
  return { y: m <= 2 ? y + 1 : y, m, d }
}

function dtPartFromMs(ms: number, op: string): number {
  if (!Number.isFinite(ms)) return NaN
  if (op === 'epochMillis') return ms
  const dayMs = 86_400_000
  const days = Math.floor(ms / dayMs)
  const tod = ms - days * dayMs
  if (op === 'hour') return Math.floor(tod / 3_600_000)
  if (op === 'minute') return Math.floor((tod % 3_600_000) / 60_000)
  if (op === 'second') return Math.floor((tod % 60_000) / 1000)
  if (op === 'weekday') return (days + 4) % 7 // 1970-01-01 was Thursday
  const { y, m, d } = civilFromDays(days)
  if (op === 'year') return y
  if (op === 'month') return m
  if (op === 'day') return d
  return NaN
}

function applyStrOp(s: string, op: string, expr: Extract<ExprNode, { type: 'str' }>): string | number | boolean {
  switch (op) {
    case 'len':
      return s.length
    case 'toLowerCase':
      return s.toLowerCase()
    case 'toUpperCase':
      return s.toUpperCase()
    case 'trim':
      return s.trim()
    case 'contains':
      return s.includes(expr.pattern ?? '')
    case 'startsWith':
      return s.startsWith(expr.pattern ?? '')
    case 'endsWith':
      return s.endsWith(expr.pattern ?? '')
    case 'replace': {
      const rep = expr.replacement ?? ''
      return s.replace(expr.pattern ?? '', () => rep)
    }
    case 'replaceAll':
      return s.split(expr.pattern ?? '').join(expr.replacement ?? '')
    case 'slice':
      return s.slice(expr.start ?? 0, expr.end)
    case 'split':
      return JSON.stringify(s.split(expr.pattern ?? ','))
    default:
      return s
  }
}

/** Collapse duplicate dictionary entries and remap codes (e.g. after toLowerCase merges "A"/"a"). */
function canonicalizeCategory(
  codes: Uint32Array,
  dictionary: string[],
  nullBitmap: Uint8Array | undefined,
): { codes: Uint32Array; dictionary: string[] } {
  const map = new Map<string, number>()
  const newDict: string[] = []
  const oldToNew = new Int32Array(dictionary.length)
  for (let d = 0; d < dictionary.length; d++) {
    const s = dictionary[d]!
    let code = map.get(s)
    if (code === undefined) {
      code = newDict.length
      newDict.push(s)
      map.set(s, code)
    }
    oldToNew[d] = code
  }
  if (newDict.length === dictionary.length) {
    // No collapse — keep original codes buffer when possible.
    let same = true
    for (let d = 0; d < dictionary.length; d++) {
      if (oldToNew[d] !== d) {
        same = false
        break
      }
    }
    if (same) return { codes, dictionary }
  }
  const n = codes.length
  const remapped = new Uint32Array(n)
  if (nullBitmap) {
    for (let i = 0; i < n; i++) {
      if (!isValid(nullBitmap, i)) continue
      const c = codes[i]!
      remapped[i] = c < oldToNew.length ? oldToNew[c]! : 0
    }
  } else {
    for (let i = 0; i < n; i++) {
      const c = codes[i]!
      remapped[i] = c < oldToNew.length ? oldToNew[c]! : 0
    }
  }
  return { codes: remapped, dictionary: newDict }
}

/** Category dict remap / utf8 typed loops for str.* column materialization. */
function tryFastStrColumn(table: TableView, expr: Extract<ExprNode, { type: 'str' }>, name: string): Column | null {
  let inner = expr.expr
  while (inner.type === 'alias') inner = inner.expr
  if (inner.type !== 'col') return null
  const col = getColumn(table, inner.name)
  const n = table.numRows
  const op = expr.op

  if (col.field.dtype === 'category' && col.dictionary) {
    const dict = col.dictionary
    const codes = col.data as Uint32Array
    const bm = col.nullBitmap

    if (op === 'toLowerCase' || op === 'toUpperCase' || op === 'trim' || op === 'replace' || op === 'replaceAll' || op === 'slice') {
      const newDict = dict.map((s) => String(applyStrOp(s, op, expr)))
      const canon = canonicalizeCategory(codes, newDict, bm)
      return {
        field: { name, dtype: 'category', nullable: col.field.nullable },
        data: canon.codes,
        nullBitmap: bm,
        dictionary: canon.dictionary,
      }
    }
    if (op === 'len') {
      const lens = dict.map((s) => s.length)
      const out = new Float64Array(n)
      if (bm) for (let i = 0; i < n; i++) out[i] = isValid(bm, i) ? lens[codes[i]!]! : NaN
      else for (let i = 0; i < n; i++) out[i] = lens[codes[i]!]!
      const anyNull = bm !== undefined && hasNulls(bm, n)
      return {
        field: { name, dtype: 'f64', nullable: anyNull },
        data: out,
        nullBitmap: anyNull ? bm : undefined,
      }
    }
    if (op === 'contains' || op === 'startsWith' || op === 'endsWith') {
      const flags = new Uint8Array(dict.length)
      for (let d = 0; d < dict.length; d++) flags[d] = applyStrOp(dict[d]!, op, expr) ? 1 : 0
      const out = new Uint8Array(n)
      if (bm) for (let i = 0; i < n; i++) out[i] = isValid(bm, i) && flags[codes[i]!]! ? 1 : 0
      else for (let i = 0; i < n; i++) out[i] = flags[codes[i]!]!
      return { field: { name, dtype: 'bool', nullable: false }, data: out }
    }
    if (op === 'split') {
      const newDict = dict.map((s) => String(applyStrOp(s, op, expr)))
      const canon = canonicalizeCategory(codes, newDict, bm)
      return {
        field: { name, dtype: 'category', nullable: col.field.nullable },
        data: canon.codes,
        nullBitmap: bm,
        dictionary: canon.dictionary,
      }
    }
    return null
  }

  if (col.field.dtype !== 'utf8') return null
  const data = col.data as string[]
  const bm = col.nullBitmap

  if (
    n >= NATIVE_STR_MIN_ROWS &&
    !bm &&
    isNativeKernelsLoaded() &&
    (op === 'contains' || op === 'toLowerCase')
  ) {
    const k = getNativeKernels()
    if (op === 'contains' && k.strContains) {
      return { field: { name, dtype: 'bool', nullable: false }, data: k.strContains(data, expr.pattern ?? '') }
    }
    if (op === 'toLowerCase' && k.strToLower) {
      return { field: { name, dtype: 'utf8', nullable: false }, data: k.strToLower(data) }
    }
  }

  if (op === 'len') {
    const out = new Float64Array(n)
    if (bm) for (let i = 0; i < n; i++) out[i] = isValid(bm, i) ? (data[i] ?? '').length : NaN
    else for (let i = 0; i < n; i++) out[i] = (data[i] ?? '').length
    const anyNull = bm !== undefined && hasNulls(bm, n)
    return { field: { name, dtype: 'f64', nullable: anyNull }, data: out, nullBitmap: anyNull ? bm : undefined }
  }
  if (op === 'contains' || op === 'startsWith' || op === 'endsWith') {
    const out = new Uint8Array(n)
    for (let i = 0; i < n; i++) {
      if (bm && !isValid(bm, i)) out[i] = 0
      else out[i] = applyStrOp(data[i] ?? '', op, expr) ? 1 : 0
    }
    return { field: { name, dtype: 'bool', nullable: false }, data: out }
  }
  const out = new Array<string>(n)
  let anyNull = false
  const nullBitmap = new Uint8Array(Math.ceil(n / 8) || 1)
  for (let i = 0; i < n; i++) {
    if (bm && !isValid(bm, i)) {
      anyNull = true
      out[i] = ''
    } else {
      setValid(nullBitmap, i, true)
      out[i] = String(applyStrOp(data[i] ?? '', op, expr))
    }
  }
  return {
    field: { name, dtype: 'utf8', nullable: anyNull },
    data: out,
    nullBitmap: anyNull ? nullBitmap : undefined,
  }
}

/**
 * Fuse multiple dt.* extracts on the same numeric/datetime column into one scan.
 * Returns null when the batch is not a pure same-source dt group.
 */
export function tryFusedDtColumns(
  table: TableView,
  specs: Array<{ name: string; expr: ExprNode }>,
): Column[] | null {
  if (specs.length < 2) return null
  let srcName: string | null = null
  const parts: Array<{ name: string; op: string }> = []
  for (const s of specs) {
    if (s.expr.type !== 'dt') return null
    let inner = s.expr.expr
    while (inner.type === 'alias') inner = inner.expr
    if (inner.type !== 'col') return null
    if (srcName === null) srcName = inner.name
    else if (srcName !== inner.name) return null
    parts.push({ name: s.name, op: s.expr.op })
  }
  const col = getColumn(table, srcName!)
  const data = numericView(col)
  if (!data) return null
  const n = table.numRows
  const bm = col.nullBitmap
  const outs = parts.map(() => new Float64Array(n))
  let anyNull = false
  for (let i = 0; i < n; i++) {
    if (bm && !isValid(bm, i)) {
      anyNull = true
      for (let p = 0; p < parts.length; p++) outs[p]![i] = NaN
      continue
    }
    const ms = data[i]!
    // Shared civil breakdown when any date-part is requested
    let y = 0
    let m = 0
    let d = 0
    let civil = false
    const dayMs = 86_400_000
    const days = Math.floor(ms / dayMs)
    const tod = ms - days * dayMs
    for (let p = 0; p < parts.length; p++) {
      const op = parts[p]!.op
      if (op === 'epochMillis') {
        outs[p]![i] = ms
        continue
      }
      if (op === 'hour') {
        outs[p]![i] = Math.floor(tod / 3_600_000)
        continue
      }
      if (op === 'minute') {
        outs[p]![i] = Math.floor((tod % 3_600_000) / 60_000)
        continue
      }
      if (op === 'second') {
        outs[p]![i] = Math.floor((tod % 60_000) / 1000)
        continue
      }
      if (op === 'weekday') {
        outs[p]![i] = (days + 4) % 7
        continue
      }
      if (!civil) {
        const c = civilFromDays(days)
        y = c.y
        m = c.m
        d = c.d
        civil = true
      }
      outs[p]![i] = op === 'year' ? y : op === 'month' ? m : op === 'day' ? d : NaN
    }
  }
  const nullBitmap = anyNull ? (bm ? new Uint8Array(bm) : undefined) : undefined
  return parts.map((p, i) => ({
    field: { name: p.name, dtype: 'f64' as const, nullable: anyNull },
    data: outs[i]!,
    nullBitmap: anyNull ? nullBitmap : undefined,
  }))
}

function tryFastDtColumn(table: TableView, expr: Extract<ExprNode, { type: 'dt' }>, name: string): Column | null {
  const n = table.numRows
  const val = evalVec(table, expr, n)
  if (!val || val.t !== 'vec') return null
  const anyNull = val.bitmap !== undefined && hasNulls(val.bitmap, n)
  return {
    field: { name, dtype: 'f64', nullable: anyNull },
    data: val.data as Float64Array,
    nullBitmap: anyNull ? val.bitmap : undefined,
  }
}

/**
 * Vectorized expression materialization: typed loops instead of per-row eval.
 * Covers numeric arithmetic, comparisons, str/dt, isIn/isBetween/when and and/or.
 */
export function tryFastExprColumn(table: TableView, expr: ExprNode, name: string): Column | null {
  const n = table.numRows
  if (n === 0) return null

  if (expr.type === 'str') {
    const fast = tryFastStrColumn(table, expr, name)
    if (fast) return fast
  }
  if (expr.type === 'dt') {
    const fast = tryFastDtColumn(table, expr, name)
    if (fast) return fast
  }

  const val = evalVec(table, expr, n)
  if (!val) return null

  if (val.t === 'str_vec') {
    if (!val.owned) return null
    const anyNull = val.bitmap !== undefined && hasNulls(val.bitmap, n)
    return {
      field: { name, dtype: 'utf8', nullable: anyNull },
      data: val.data,
      nullBitmap: anyNull ? val.bitmap : undefined,
    }
  }

  if (val.t === 'str_scalar') {
    const data = new Array<string>(n).fill(val.v)
    return { field: { name, dtype: 'utf8', nullable: false }, data }
  }

  if (val.t === 'scalar') {
    const data = new Float64Array(n).fill(val.v)
    return { field: { name, dtype: 'f64', nullable: false }, data }
  }

  if (val.t !== 'vec') return null

  if (!val.owned) {
    const src = val.src
    if (!src) return null
    return { ...src, field: { ...src.field, name } }
  }

  const anyNull = val.bitmap !== undefined && hasNulls(val.bitmap, n)
  return {
    field: { name, dtype: val.kind === 'bool' ? 'bool' : 'f64', nullable: anyNull },
    data: val.data as Float64Array | Uint8Array,
    nullBitmap: anyNull ? val.bitmap : undefined,
  }
}

/**
 * Input column names a projection must read (strings + col / alias(col)).
 * Alias output names are intentionally not returned — use `isIdentityProjection`
 * before skipping a final project rename.
 */
export function requiredInputColumns(columns: Array<string | ExprNode>): string[] | null {
  const names: string[] = []
  for (const c of columns) {
    if (typeof c === 'string') names.push(c)
    else if (c.type === 'col') names.push(c.name)
    else if (c.type === 'alias' && c.expr.type === 'col') names.push(c.expr.name)
    else return null // complex expr — cannot prune safely to just inputs without eval
  }
  return names
}

/** @deprecated Use requiredInputColumns — kept for callers that prune by source names. */
export function projectColumnNames(columns: Array<string | ExprNode>): string[] | null {
  return requiredInputColumns(columns)
}

/** True when every projection entry is an identity column ref (no rename / compute). */
export function isIdentityProjection(columns: Array<string | ExprNode>): boolean {
  for (const c of columns) {
    if (typeof c === 'string') continue
    if (c.type === 'col') continue
    if (c.type === 'alias' && c.expr.type === 'col' && c.name === c.expr.name) continue
    return false
  }
  return true
}
