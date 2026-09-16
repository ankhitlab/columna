/**
 * Window aggregates: `agg(expr).over(partitionBy)` — the aggregate is computed per partition and
 * broadcast back to every row of that partition (polars `over` / SQL `agg() OVER (PARTITION BY …)`).
 *
 * Two pieces: `partitionIds` maps every row to a dense group id (category codes / small-span ints
 * are used directly, everything else goes through one string-keyed map, multi-key partitions are
 * packed with mixed radix), and `overAggregate` runs per-group typed accumulators over a numeric
 * column and scatters the result into an n-length column. Non-numeric inputs (first/last/count/nunique
 * on strings) fall back to boxed per-group evaluation.
 */
import { getColumn, getValue, isNumeric, isValid, type Column, type TableView } from '@columna/arrow'
import type { AggKind, QuantileMethod } from './types.js'
import { argsortNumeric, countUniqueNumeric, quantileSelect } from './fast.js'

export type Partition = { ids: Uint32Array; groups: number }

const DENSE_MAX = 1 << 24

/** Group id per row for one column; nulls form their own group. */
function singleKeyIds(col: Column, n: number): Partition {
  const bm = col.nullBitmap
  const ids = new Uint32Array(n)
  // Dictionary-encoded: codes are already dense ids
  if (col.field.dtype === 'category' && col.dictionary) {
    const codes = col.data as Uint32Array
    const card = col.dictionary.length
    for (let i = 0; i < n; i++) ids[i] = bm && !isValid(bm, i) ? card : codes[i]!
    return { ids, groups: card + 1 }
  }
  // Integer-valued numeric column with a small span → offset
  if (isNumeric(col.field.dtype) || col.field.dtype === 'datetime') {
    const data = col.data as ArrayLike<number>
    let min = Infinity
    let max = -Infinity
    let ints = true
    for (let i = 0; i < n; i++) {
      if (bm && !isValid(bm, i)) continue
      const v = data[i]!
      if (!Number.isInteger(v)) {
        ints = false
        break
      }
      if (v < min) min = v
      if (v > max) max = v
    }
    if (ints && (min === Infinity || max - min < DENSE_MAX)) {
      const span = min === Infinity ? 0 : max - min + 1
      for (let i = 0; i < n; i++) ids[i] = bm && !isValid(bm, i) ? span : data[i]! - min
      return { ids, groups: span + 1 }
    }
  }
  // Generic: one map keyed by the stringified value
  const map = new Map<string, number>()
  let next = 0
  const nullId = -1
  let nullGroup = nullId
  for (let i = 0; i < n; i++) {
    if (bm && !isValid(bm, i)) {
      if (nullGroup === nullId) nullGroup = next++
      ids[i] = nullGroup
      continue
    }
    const k = String(getValue(col.data, i))
    let id = map.get(k)
    if (id === undefined) map.set(k, (id = next++))
    ids[i] = id
  }
  return { ids, groups: next }
}

/** Dense group ids for a (multi-)column partition. Ids may have unused values (sparse dense spans). */
export function partitionIds(table: TableView, keys: string[], n: number): Partition {
  if (keys.length === 1) return singleKeyIds(getColumn(table, keys[0]!), n)
  const parts = keys.map((k) => singleKeyIds(getColumn(table, k), n))
  // Mixed radix when the product of cardinalities is small enough
  let product = 1
  for (const p of parts) product *= p.groups
  if (product <= DENSE_MAX) {
    const ids = new Uint32Array(n)
    for (let i = 0; i < n; i++) {
      let id = 0
      let stride = 1
      for (const p of parts) {
        id += p.ids[i]! * stride
        stride *= p.groups
      }
      ids[i] = id
    }
    return { ids, groups: product }
  }
  const map = new Map<string, number>()
  const ids = new Uint32Array(n)
  let next = 0
  for (let i = 0; i < n; i++) {
    let k = ''
    for (const p of parts) k += p.ids[i] + '\0'
    let id = map.get(k)
    if (id === undefined) map.set(k, (id = next++))
    ids[i] = id
  }
  return { ids, groups: next }
}

/**
 * Per-partition aggregate of a numeric column, broadcast to rows. Returns null when the column is
 * not numeric (caller falls back to the boxed path). Rows whose partition has no valid values → null.
 */
export function overAggregateNumeric(
  col: Column,
  n: number,
  part: Partition,
  op: AggKind,
  q: number,
  method?: QuantileMethod,
): Column | null {
  if (!(isNumeric(col.field.dtype) || col.field.dtype === 'datetime')) return null
  const data = col.data as ArrayLike<number>
  const bm = col.nullBitmap
  const { ids, groups: G } = part
  const count = new Uint32Array(G)
  const result = new Float64Array(G)

  switch (op) {
    case 'count':
      for (let i = 0; i < n; i++) if (!bm || isValid(bm, i)) count[ids[i]!]++
      for (let g = 0; g < G; g++) result[g] = count[g]!
      break
    case 'sum':
    case 'mean': {
      for (let i = 0; i < n; i++) {
        if (bm && !isValid(bm, i)) continue
        const g = ids[i]!
        count[g]++
        result[g] += data[i]!
      }
      if (op === 'mean') for (let g = 0; g < G; g++) if (count[g]) result[g] /= count[g]!
      break
    }
    case 'min':
    case 'max': {
      result.fill(op === 'min' ? Infinity : -Infinity)
      if (op === 'min') {
        for (let i = 0; i < n; i++) {
          if (bm && !isValid(bm, i)) continue
          const g = ids[i]!
          count[g]++
          const v = data[i]!
          if (v < result[g]!) result[g] = v
        }
      } else {
        for (let i = 0; i < n; i++) {
          if (bm && !isValid(bm, i)) continue
          const g = ids[i]!
          count[g]++
          const v = data[i]!
          if (v > result[g]!) result[g] = v
        }
      }
      break
    }
    case 'first':
    case 'last': {
      for (let i = 0; i < n; i++) {
        if (bm && !isValid(bm, i)) continue
        const g = ids[i]!
        if (op === 'last' || count[g] === 0) result[g] = data[i]!
        count[g]++
      }
      break
    }
    case 'std':
    case 'var': {
      // two-pass centered: means, then squared deviations
      const sum = new Float64Array(G)
      for (let i = 0; i < n; i++) {
        if (bm && !isValid(bm, i)) continue
        const g = ids[i]!
        count[g]++
        sum[g] += data[i]!
      }
      for (let g = 0; g < G; g++) if (count[g]) sum[g] /= count[g]! // now means
      for (let i = 0; i < n; i++) {
        if (bm && !isValid(bm, i)) continue
        const g = ids[i]!
        const d = data[i]! - sum[g]!
        result[g] += d * d
      }
      for (let g = 0; g < G; g++) {
        const c = count[g]!
        const v = c > 1 ? result[g]! / (c - 1) : c === 1 ? 0 : NaN
        result[g] = op === 'std' ? Math.sqrt(v) : v
      }
      break
    }
    case 'median':
    case 'quantile':
    case 'nunique': {
      // Counting sort of valid values by group → contiguous segments
      for (let i = 0; i < n; i++) if (!bm || isValid(bm, i)) count[ids[i]!]++
      const offset = new Uint32Array(G + 1)
      for (let g = 0; g < G; g++) offset[g + 1] = offset[g]! + count[g]!
      const cursor = offset.slice(0, G)
      const sorted = new Float64Array(offset[G]!)
      for (let i = 0; i < n; i++) {
        if (bm && !isValid(bm, i)) continue
        sorted[cursor[ids[i]!]!++] = data[i]!
      }
      const qq = op === 'median' ? 0.5 : q
      for (let g = 0; g < G; g++) {
        const len = count[g]!
        if (!len) continue
        const seg = sorted.subarray(offset[g]!, offset[g + 1]!)
        result[g] = op === 'nunique' ? countUniqueNumeric(seg, len) : quantileSelect(seg, len, qq, method)
      }
      break
    }
    default:
      return null
  }

  const out = new Float64Array(n)
  let anyNull = false
  const outBm = new Uint8Array(Math.ceil(n / 8) || 1)
  for (let i = 0; i < n; i++) {
    const g = ids[i]!
    if (count[g] === 0 && op !== 'count') {
      anyNull = true
      continue
    }
    out[i] = result[g]!
    outBm[i >> 3] = outBm[i >> 3]! | (1 << (i & 7))
  }
  return {
    field: { name: col.field.name, dtype: 'f64', nullable: anyNull },
    data: out,
    nullBitmap: anyNull ? outBm : undefined,
  }
}

// ---- ordered (cumulative) windows ---------------------------------------------------------------

/**
 * Row order for `over(..., { orderBy })`: a stable argsort by the given columns (nulls last).
 * Single numeric key → radix argsort; otherwise a comparator sort over decoded values.
 */
export function orderRows(table: TableView, orderBy: string[], descending: boolean, n: number): Uint32Array {
  const cols = orderBy.map((k) => getColumn(table, k))
  if (cols.length === 1) {
    const c = cols[0]!
    if (isNumeric(c.field.dtype) || c.field.dtype === 'datetime') return argsortNumeric(c.data as ArrayLike<number>, n, c.nullBitmap, descending)
  }
  const keyAt = (c: Column, i: number): number | string | boolean | null => {
    if (!isValid(c.nullBitmap, i)) return null
    const v = getValue(c.data, i)
    return c.field.dtype === 'category' && c.dictionary ? (c.dictionary[Number(v)] ?? null) : (v as number | string | boolean)
  }
  const idx: number[] = new Array(n)
  for (let i = 0; i < n; i++) idx[i] = i
  idx.sort((a, b) => {
    for (const c of cols) {
      const va = keyAt(c, a)
      const vb = keyAt(c, b)
      if (va === vb) continue
      if (va === null) return 1
      if (vb === null) return -1
      const cmp = va < vb ? -1 : 1
      return descending ? -cmp : cmp
    }
    return a - b
  })
  return Uint32Array.from(idx)
}

/**
 * Running aggregate within each partition, rows visited in `order` (ROWS UNBOUNDED PRECEDING …
 * CURRENT ROW, like polars `cum_*().over()`): each row gets the aggregate of the partition's rows
 * up to and including itself. Null inputs leave the accumulator untouched; a row before any valid
 * value in its partition is null (count → 0).
 */
export function overCumulativeNumeric(
  col: Column,
  n: number,
  part: Partition,
  order: Uint32Array,
  op: AggKind,
): Column | null {
  if (!(isNumeric(col.field.dtype) || col.field.dtype === 'datetime')) return null
  if (op === 'median' || op === 'quantile' || op === 'nunique') {
    throw new Error(`over(orderBy): running ${op} is not supported (use sum/mean/min/max/count/std/var/first/last)`)
  }
  const data = col.data as ArrayLike<number>
  const bm = col.nullBitmap
  const { ids, groups: G } = part
  const count = new Uint32Array(G)
  const acc = new Float64Array(G) // sum / min / max / first / last / running mean
  const m2 = op === 'std' || op === 'var' ? new Float64Array(G) : null
  if (op === 'min') acc.fill(Infinity)
  if (op === 'max') acc.fill(-Infinity)
  const out = new Float64Array(n)
  const outBm = new Uint8Array(Math.ceil(n / 8) || 1)
  let anyNull = false
  for (let p = 0; p < n; p++) {
    const i = order[p]!
    const g = ids[i]!
    if (!bm || isValid(bm, i)) {
      const v = data[i]!
      count[g]++
      switch (op) {
        case 'sum':
          acc[g] += v
          break
        case 'mean':
          acc[g] += (v - acc[g]!) / count[g]!
          break
        case 'std':
        case 'var': {
          const c = count[g]!
          const d = v - acc[g]!
          acc[g] += d / c
          m2![g] += d * (v - acc[g]!)
          break
        }
        case 'min':
          if (v < acc[g]!) acc[g] = v
          break
        case 'max':
          if (v > acc[g]!) acc[g] = v
          break
        case 'first':
          if (count[g] === 1) acc[g] = v
          break
        case 'last':
          acc[g] = v
          break
      }
    }
    const c = count[g]!
    if (op === 'count') {
      out[i] = c
    } else if (c === 0) {
      anyNull = true
      continue
    } else if (op === 'var' || op === 'std') {
      const variance = c > 1 ? m2![g]! / (c - 1) : 0
      out[i] = op === 'std' ? Math.sqrt(variance) : variance
    } else out[i] = acc[g]!
    outBm[i >> 3] = outBm[i >> 3]! | (1 << (i & 7))
  }
  return {
    field: { name: col.field.name, dtype: 'f64', nullable: anyNull },
    data: out,
    nullBitmap: anyNull ? outBm : undefined,
  }
}
