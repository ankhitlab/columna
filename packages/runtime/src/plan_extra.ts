import {
  getColumn,
  getValue,
  isValid,
  setValid,
  tableFromColumns,
  takeColumn,
  takeTable,
  type Column,
  type TableView,
} from '@columna/arrow'
import type { AggKind, ExprNode } from './types.js'
import { tryFusedDtColumns } from './fast.js'

export function mulberry32(seed: number): () => number {
  let t = seed >>> 0
  return () => {
    t += 0x6d2b79f5
    let r = Math.imul(t ^ (t >>> 15), 1 | t)
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r)
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296
  }
}

export function sampleTable(
  table: TableView,
  opts: { n?: number; fraction?: number; seed?: number },
): TableView {
  const nRows = table.numRows
  let n: number
  if (opts.n !== undefined) n = Math.max(0, Math.min(nRows, Math.floor(opts.n)))
  else if (opts.fraction !== undefined) n = Math.max(0, Math.min(nRows, Math.round(nRows * opts.fraction)))
  else n = Math.min(nRows, 10)
  if (n <= 0) return takeTable(table, [])
  if (n >= nRows) return table
  const rand = mulberry32(opts.seed ?? 1)

  // Partial Fisher–Yates: only k swaps. Sparse map avoids O(nRows) index buffer when k ≪ n.
  if (n * 4 < nRows) {
    const out = new Uint32Array(n)
    const map = new Map<number, number>()
    const at = (i: number) => map.get(i) ?? i
    const setAt = (i: number, v: number) => {
      if (v === i) map.delete(i)
      else map.set(i, v)
    }
    for (let i = 0; i < n; i++) {
      const j = i + Math.floor(rand() * (nRows - i))
      const vi = at(i)
      const vj = at(j)
      setAt(i, vj)
      setAt(j, vi)
      out[i] = at(i)
    }
    return takeTable(table, out)
  }

  const idx = new Uint32Array(nRows)
  for (let i = 0; i < nRows; i++) idx[i] = i
  for (let i = 0; i < n; i++) {
    const j = i + Math.floor(rand() * (nRows - i))
    const tmp = idx[i]!
    idx[i] = idx[j]!
    idx[j] = tmp
  }
  return takeTable(table, idx.subarray(0, n))
}

function parseListCell(raw: unknown): unknown[] {
  if (raw == null) return []
  if (Array.isArray(raw)) return raw
  if (typeof raw !== 'string') return [raw]
  const s = raw.trim()
  if (!s) return []
  if (s.startsWith('[')) {
    try {
      const parsed = JSON.parse(s)
      return Array.isArray(parsed) ? parsed : [s]
    } catch {
      return [s]
    }
  }
  return s.split(',').map((x) => x.trim())
}

/** Repeat each source row `counts[i]` times — faster than gather via row index list. */
function expandByCounts(col: Column, counts: Uint32Array, outN: number): Column {
  const nIn = counts.length
  const dtype = col.field.dtype
  const dict = col.dictionary
  const srcBm = col.nullBitmap

  const expandBitmap = (): Uint8Array | undefined => {
    if (!srcBm) return undefined
    let anyNull = false
    for (let i = 0; i < nIn; i++) {
      if (!isValid(srcBm, i)) {
        anyNull = true
        break
      }
    }
    if (!anyNull) return undefined
    const nullBitmap = new Uint8Array(Math.ceil(outN / 8) || 1)
    let w = 0
    for (let i = 0; i < nIn; i++) {
      const c = counts[i]!
      if (isValid(srcBm, i)) {
        for (let k = 0; k < c; k++) setValid(nullBitmap, w + k, true)
      }
      w += c
    }
    return nullBitmap
  }

  if (dtype === 'utf8') {
    const src = col.data as string[]
    const data = new Array<string>(outN)
    let w = 0
    for (let i = 0; i < nIn; i++) {
      const c = counts[i]!
      const v = src[i]!
      if (c === 1) data[w++] = v
      else {
        data.fill(v, w, w + c)
        w += c
      }
    }
    const nullBitmap = expandBitmap()
    return {
      field: { ...col.field, nullable: Boolean(nullBitmap) || col.field.nullable },
      data,
      nullBitmap,
      dictionary: dict,
    }
  }

  const alloc =
    dtype === 'f64' || dtype === 'datetime'
      ? new Float64Array(outN)
      : dtype === 'f32'
        ? new Float32Array(outN)
        : dtype === 'i32'
          ? new Int32Array(outN)
          : dtype === 'bool'
            ? new Uint8Array(outN)
            : new Uint32Array(outN)
  const src = col.data as Float64Array | Float32Array | Int32Array | Uint32Array | Uint8Array
  let w = 0
  for (let i = 0; i < nIn; i++) {
    const c = counts[i]!
    const v = src[i]! as never
    if (c === 1) alloc[w++] = v
    else {
      alloc.fill(v, w, w + c)
      w += c
    }
  }
  const nullBitmap = expandBitmap()
  return {
    field: { ...col.field, nullable: Boolean(nullBitmap) || col.field.nullable },
    data: alloc,
    nullBitmap,
    dictionary: dict,
  }
}

export function explodeTable(table: TableView, column: string): TableView {
  const col = getColumn(table, column)
  const nIn = table.numRows
  const bm = col.nullBitmap
  const dtype = col.field.dtype
  const dict = col.dictionary
  const utf8Src = dtype === 'utf8' ? (col.data as string[]) : null
  const catSrc = dtype === 'category' && dict ? (col.data as Uint32Array) : null

  const parseCache = new Map<string, unknown[]>()
  const counts = new Uint32Array(nIn)
  const itemsPerRow = new Array<unknown[] | null>(nIn)
  let outN = 0
  let allStrings = true

  for (let i = 0; i < nIn; i++) {
    if (bm && !isValid(bm, i)) {
      counts[i] = 1
      itemsPerRow[i] = null
      outN++
      continue
    }

    let raw: unknown
    if (utf8Src) raw = utf8Src[i]
    else if (catSrc) raw = dict![catSrc[i]!] ?? null
    else raw = getValue(col.data, i)

    let items: unknown[]
    if (typeof raw === 'string') {
      let hit = parseCache.get(raw)
      if (!hit) {
        hit = parseListCell(raw)
        parseCache.set(raw, hit)
      }
      items = hit
    } else {
      items = parseListCell(raw)
    }

    if (items.length === 0) {
      counts[i] = 1
      itemsPerRow[i] = null
      outN++
    } else {
      counts[i] = items.length
      itemsPerRow[i] = items
      outN += items.length
      if (allStrings) {
        for (let k = 0; k < items.length; k++) {
          const v = items[k]
          if (v != null && typeof v !== 'string') {
            allStrings = false
            break
          }
        }
      }
    }
  }

  const outCols = table.columns.map((c) => {
    if (c.field.name !== column) return expandByCounts(c, counts, outN)

    if (allStrings) {
      const data = new Array<string>(outN)
      let anyNull = false
      const nullBitmap = new Uint8Array(Math.ceil(outN / 8) || 1)
      let w = 0
      for (let i = 0; i < nIn; i++) {
        const items = itemsPerRow[i]
        if (!items) {
          anyNull = true
          data[w++] = ''
          continue
        }
        for (let k = 0; k < items.length; k++) {
          const v = items[k]
          if (v == null) {
            anyNull = true
            data[w++] = ''
          } else {
            setValid(nullBitmap, w, true)
            data[w++] = v as string
          }
        }
      }
      return {
        field: { name: column, dtype: 'utf8' as const, nullable: anyNull },
        data,
        nullBitmap: anyNull ? nullBitmap : undefined,
      }
    }

    const data = new Float64Array(outN)
    let anyNull = false
    const nullBitmap = new Uint8Array(Math.ceil(outN / 8) || 1)
    let w = 0
    for (let i = 0; i < nIn; i++) {
      const items = itemsPerRow[i]
      if (!items) {
        anyNull = true
        data[w++] = NaN
        continue
      }
      for (let k = 0; k < items.length; k++) {
        const v = items[k]
        if (v == null) {
          anyNull = true
          data[w++] = NaN
        } else {
          setValid(nullBitmap, w, true)
          data[w++] = Number(v)
        }
      }
    }
    return {
      field: { name: column, dtype: 'f64' as const, nullable: anyNull },
      data,
      nullBitmap: anyNull ? nullBitmap : undefined,
    }
  })
  return tableFromColumns(outCols)
}

function flattenObject(
  input: unknown,
  sep: string,
  prefix = '',
  out: Record<string, unknown> = {},
): Record<string, unknown> {
  if (input === null || input === undefined) {
    if (prefix) out[prefix] = null
    return out
  }
  if (typeof input === 'object' && !Array.isArray(input)) {
    for (const [k, v] of Object.entries(input as Record<string, unknown>)) {
      const key = prefix ? `${prefix}${sep}${k}` : k
      flattenObject(v, sep, key, out)
    }
    return out
  }
  out[prefix || 'value'] = Array.isArray(input) ? JSON.stringify(input) : input
  return out
}

type FlatVariant = { flat: Record<string, unknown>; id: number }

/**
 * Unnest a JSON/object column into flat fields.
 * Caches parse+flatten by source string (repeated payloads in benches).
 */
export function unnestTable(table: TableView, column: string, separator = '.'): TableView {
  const col = getColumn(table, column)
  const n = table.numRows
  const stringCache = new Map<string, FlatVariant>()
  const variants: Record<string, unknown>[] = []
  const variantOf = new Int32Array(n)
  variantOf.fill(-1)
  const keys: string[] = []
  const seenKey = new Set<string>()

  const register = (flat: Record<string, unknown>): number => {
    const id = variants.length
    variants.push(flat)
    for (const k of Object.keys(flat)) {
      if (!seenKey.has(k)) {
        seenKey.add(k)
        keys.push(k)
      }
    }
    return id
  }

  const data = col.data as string[] | Float64Array | Int32Array | Uint32Array
  const isUtf8 = col.field.dtype === 'utf8'
  const isCat = col.field.dtype === 'category' && col.dictionary

  for (let i = 0; i < n; i++) {
    if (!isValid(col.nullBitmap, i)) continue

    if (isUtf8) {
      const raw = (data as string[])[i]!
      const hit = stringCache.get(raw)
      if (hit) {
        variantOf[i] = hit.id
        continue
      }
      let parsed: unknown = raw
      try {
        parsed = JSON.parse(raw)
      } catch {
        /* keep string */
      }
      const flat = flattenObject(parsed, separator)
      const id = register(flat)
      stringCache.set(raw, { flat, id })
      variantOf[i] = id
      continue
    }

    let raw: unknown
    if (isCat) {
      raw = col.dictionary![Number((data as Uint32Array)[i])] ?? null
      if (typeof raw === 'string') {
        const hit = stringCache.get(raw)
        if (hit) {
          variantOf[i] = hit.id
          continue
        }
        let parsed: unknown = raw
        try {
          parsed = JSON.parse(raw)
        } catch {
          /* keep */
        }
        const flat = flattenObject(parsed, separator)
        const id = register(flat)
        stringCache.set(raw, { flat, id })
        variantOf[i] = id
        continue
      }
    } else {
      raw = getValue(col.data, i)
    }
    variantOf[i] = register(flattenObject(raw, separator))
  }

  const other = table.columns.filter((c) => c.field.name !== column)
  const newCols: Column[] = [...other]

  for (const key of keys) {
    let asNum = true
    for (const flat of variants) {
      const v = flat[key]
      if (v == null) continue
      if (typeof v !== 'number' && typeof v !== 'boolean') {
        asNum = false
        break
      }
    }
    if (asNum) {
      const out = new Float64Array(n)
      const nullBitmap = new Uint8Array(Math.ceil(n / 8) || 1)
      let anyNull = false
      for (let i = 0; i < n; i++) {
        const vi = variantOf[i]!
        if (vi < 0) {
          anyNull = true
          out[i] = NaN
          continue
        }
        const v = variants[vi]![key]
        if (v == null) {
          anyNull = true
          out[i] = NaN
        } else {
          setValid(nullBitmap, i, true)
          out[i] = Number(v)
        }
      }
      newCols.push({
        field: { name: key, dtype: 'f64', nullable: anyNull },
        data: out,
        nullBitmap: anyNull ? nullBitmap : undefined,
      })
    } else {
      const out = new Array<string>(n)
      const nullBitmap = new Uint8Array(Math.ceil(n / 8) || 1)
      let anyNull = false
      for (let i = 0; i < n; i++) {
        const vi = variantOf[i]!
        if (vi < 0) {
          anyNull = true
          out[i] = ''
          continue
        }
        const v = variants[vi]![key]
        if (v == null) {
          anyNull = true
          out[i] = ''
        } else {
          setValid(nullBitmap, i, true)
          out[i] = String(v)
        }
      }
      newCols.push({
        field: { name: key, dtype: 'utf8', nullable: anyNull },
        data: out,
        nullBitmap: anyNull ? nullBitmap : undefined,
      })
    }
  }
  return tableFromColumns(newCols)
}

/** Cell as text: category codes are decoded through the dictionary, never shown as numbers. */
function cellText(c: Column, i: number): string {
  const v = getValue(c.data, i)
  if (c.field.dtype === 'category' && c.dictionary) return c.dictionary[Number(v)] ?? ''
  return String(v)
}

export function transposeTable(table: TableView, headerColumn?: string): TableView {
  const headers =
    headerColumn !== undefined
      ? Array.from({ length: table.numRows }, (_, i) => {
          const c = getColumn(table, headerColumn)
          return isValid(c.nullBitmap, i) ? cellText(c, i) : `row_${i}`
        })
      : Array.from({ length: table.numRows }, (_, i) => `row_${i}`)
  const valueCols = table.columns.filter((c) => c.field.name !== headerColumn)
  const out: Column[] = [
    {
      field: { name: 'column', dtype: 'utf8', nullable: false },
      data: valueCols.map((c) => c.field.name),
    },
  ]
  for (let r = 0; r < table.numRows; r++) {
    const data = valueCols.map((c) => {
      if (!isValid(c.nullBitmap, r)) return ''
      return cellText(c, r)
    })
    out.push({ field: { name: headers[r]!, dtype: 'utf8', nullable: true }, data })
  }
  return tableFromColumns(out)
}

export function interpolateTable(table: TableView, columns?: string[]): TableView {
  const names =
    columns ??
    table.schema.filter((f) => f.dtype !== 'utf8' && f.dtype !== 'category').map((f) => f.name)
  return tableFromColumns(
    table.columns.map((col) => {
      if (!names.includes(col.field.name)) return col
      const n = table.numRows
      const data = new Float64Array(n)
      const vals: Array<number | null> = []
      for (let i = 0; i < n; i++) {
        if (!isValid(col.nullBitmap, i)) vals.push(null)
        else vals.push(Number(getValue(col.data, i)))
      }
      for (let i = 0; i < n; i++) {
        if (vals[i] !== null) {
          data[i] = vals[i]!
          continue
        }
        let lo = i - 1
        while (lo >= 0 && vals[lo] === null) lo--
        let hi = i + 1
        while (hi < n && vals[hi] === null) hi++
        if (lo < 0 && hi >= n) data[i] = NaN
        else if (lo < 0) data[i] = vals[hi]!
        else if (hi >= n) data[i] = vals[lo]!
        else {
          const t = (i - lo) / (hi - lo)
          data[i] = vals[lo]! * (1 - t) + vals[hi]! * t
        }
      }
      return { field: { ...col.field, dtype: 'f64' as const, nullable: false }, data }
    }),
  )
}

export function expandingTable(
  table: TableView,
  name: string,
  column: string,
  agg: AggKind,
  aggregateValues: (
    op: AggKind,
    expr: ExprNode,
    table: TableView,
    rows: number[],
  ) => number | string | boolean | null,
): TableView {
  const src = getColumn(table, column)
  const data = new Float64Array(table.numRows)
  const nullBitmap = new Uint8Array(Math.ceil(table.numRows / 8) || 1)
  let anyNull = false
  if (agg === 'sum' || agg === 'mean' || agg === 'count' || agg === 'min' || agg === 'max') {
    let sum = 0
    let count = 0
    let min = Infinity
    let max = -Infinity
    for (let i = 0; i < table.numRows; i++) {
      if (isValid(src.nullBitmap, i)) {
        const v = Number(getValue(src.data, i))
        if (Number.isFinite(v)) {
          count++
          sum += v
          if (v < min) min = v
          if (v > max) max = v
        }
      }
      let out: number | null = null
      if (agg === 'count') out = count
      else if (count === 0) out = null
      else if (agg === 'sum') out = sum
      else if (agg === 'mean') out = sum / count
      else if (agg === 'min') out = min
      else if (agg === 'max') out = max
      if (out === null) {
        anyNull = true
        data[i] = NaN
      } else {
        setValid(nullBitmap, i, true)
        data[i] = out
      }
    }
  } else {
    for (let i = 0; i < table.numRows; i++) {
      const rows = Array.from({ length: i + 1 }, (_, j) => j)
      const v = aggregateValues(agg, { type: 'col', name: column }, table, rows)
      if (v === null) {
        anyNull = true
        data[i] = NaN
      } else {
        setValid(nullBitmap, i, true)
        data[i] = Number(v)
      }
    }
  }
  const col: Column = {
    field: { name, dtype: 'f64', nullable: anyNull },
    data,
    nullBitmap: anyNull ? nullBitmap : undefined,
  }
  return tableFromColumns([...table.columns.filter((c) => c.field.name !== name), col])
}

export function asofJoinTables(
  left: TableView,
  right: TableView,
  leftOn: string,
  rightOn: string,
  strategy: 'backward' | 'forward' | 'nearest',
): TableView {
  const lCol = getColumn(left, leftOn)
  const rCol = getColumn(right, rightOn)
  const ln = left.numRows
  const rn = right.numRows

  const rVals: number[] = []
  const rOrig: number[] = []
  for (let i = 0; i < rn; i++) {
    if (!isValid(rCol.nullBitmap, i)) continue
    rVals.push(Number(getValue(rCol.data, i)))
    rOrig.push(i)
  }
  const orderR = Array.from({ length: rVals.length }, (_, i) => i)
  orderR.sort((a, b) => rVals[a]! - rVals[b]! || a - b)
  const rSortedVal = new Float64Array(orderR.length)
  const rSortedIdx = new Int32Array(orderR.length)
  for (let i = 0; i < orderR.length; i++) {
    const o = orderR[i]!
    rSortedVal[i] = rVals[o]!
    rSortedIdx[i] = rOrig[o]!
  }
  const rm = rSortedVal.length

  const lKeys = new Float64Array(ln)
  const lOrder = new Uint32Array(ln)
  let needSort = false
  let prev = -Infinity
  for (let i = 0; i < ln; i++) {
    lOrder[i] = i
    if (!isValid(lCol.nullBitmap, i)) {
      lKeys[i] = Number.NaN
      needSort = true
    } else {
      const v = Number(getValue(lCol.data, i))
      lKeys[i] = v
      if (v < prev) needSort = true
      prev = v
    }
  }
  if (needSort) {
    lOrder.sort((a, b) => {
      const av = lKeys[a]!
      const bv = lKeys[b]!
      const aNull = Number.isNaN(av)
      const bNull = Number.isNaN(bv)
      if (aNull && bNull) return a - b
      if (aNull) return 1
      if (bNull) return -1
      if (av !== bv) return av < bv ? -1 : 1
      return a - b
    })
  }

  const matchRight = new Int32Array(ln)
  matchRight.fill(-1)

  if (rm === 0) {
    /* unmatched */
  } else if (strategy === 'backward') {
    let j = -1
    for (let oi = 0; oi < ln; oi++) {
      const i = lOrder[oi]!
      const lv = lKeys[i]!
      if (Number.isNaN(lv)) continue
      while (j + 1 < rm && rSortedVal[j + 1]! <= lv) j++
      if (j >= 0) matchRight[i] = rSortedIdx[j]!
    }
  } else if (strategy === 'forward') {
    let j = 0
    for (let oi = 0; oi < ln; oi++) {
      const i = lOrder[oi]!
      const lv = lKeys[i]!
      if (Number.isNaN(lv)) continue
      while (j < rm && rSortedVal[j]! < lv) j++
      if (j < rm) matchRight[i] = rSortedIdx[j]!
    }
  } else {
    for (let i = 0; i < ln; i++) {
      const lv = lKeys[i]!
      if (Number.isNaN(lv)) continue
      let lo = 0
      let hi = rm
      while (lo < hi) {
        const mid = (lo + hi) >> 1
        if (rSortedVal[mid]! < lv) lo = mid + 1
        else hi = mid
      }
      let best = -1
      let bestDist = Infinity
      if (lo < rm) {
        best = rSortedIdx[lo]!
        bestDist = Math.abs(rSortedVal[lo]! - lv)
      }
      if (lo > 0) {
        const d = Math.abs(rSortedVal[lo - 1]! - lv)
        if (d < bestDist) {
          best = rSortedIdx[lo - 1]!
          bestDist = d
        }
      }
      matchRight[i] = best
    }
  }

  const out: Column[] = [...left.columns]
  const leftNames = new Set(left.schema.map((f) => f.name))
  const gatherR = new Uint32Array(ln)
  let hasNull = false
  for (let i = 0; i < ln; i++) {
    if (matchRight[i]! < 0) {
      gatherR[i] = 0
      hasNull = true
    } else gatherR[i] = matchRight[i]!
  }
  for (const col of right.columns) {
    if (col.field.name === rightOn && leftOn === rightOn) continue
    const name = leftNames.has(col.field.name) ? `${col.field.name}_right` : col.field.name
    const cloned = takeColumn(col, gatherR)
    if (!hasNull) {
      out.push({ ...cloned, field: { ...cloned.field, name } })
      continue
    }
    const nullBitmap = cloned.nullBitmap
      ? new Uint8Array(cloned.nullBitmap)
      : new Uint8Array(Math.ceil(ln / 8) || 1)
    if (!cloned.nullBitmap) nullBitmap.fill(0xff)
    let anyNull = Boolean(cloned.nullBitmap)
    for (let i = 0; i < ln; i++) {
      if (matchRight[i]! < 0) {
        setValid(nullBitmap, i, false)
        anyNull = true
      }
    }
    out.push({
      ...cloned,
      field: { ...cloned.field, name, nullable: anyNull },
      nullBitmap: anyNull ? nullBitmap : undefined,
    })
  }
  return tableFromColumns(out)
}

export function withColumnsTable(
  table: TableView,
  columns: Array<{ name: string; expr: ExprNode }>,
  materialize: (table: TableView, expr: ExprNode, name: string) => Column,
): TableView {
  let cols = [...table.columns]
  let i = 0
  while (i < columns.length) {
    // Greedy fuse consecutive dt.* on the same source column
    const batch: Array<{ name: string; expr: ExprNode }> = [columns[i]!]
    let j = i + 1
    if (columns[i]!.expr.type === 'dt') {
      while (j < columns.length && columns[j]!.expr.type === 'dt') {
        batch.push(columns[j]!)
        j++
      }
    }
    if (batch.length >= 2) {
      const fused = tryFusedDtColumns(table, batch)
      if (fused) {
        for (const col of fused) {
          cols = cols.filter((c) => c.field.name !== col.field.name)
          cols.push(col)
        }
        i = j
        continue
      }
    }
    const { name, expr } = columns[i]!
    const col = materialize(table, expr, name)
    cols = cols.filter((c) => c.field.name !== name)
    cols.push(col)
    i++
  }
  return tableFromColumns(cols)
}
