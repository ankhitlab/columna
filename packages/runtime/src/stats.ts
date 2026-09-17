/**
 * Lightweight cardinality / selectivity helpers for the rule-based optimiser.
 * Sample-based — not a persistent stats catalog.
 */
import {
  getColumn,
  getValue,
  isValid,
  type Column,
  type TableView,
} from '@columna/arrow'
import type { ExprNode, PlanNode } from './types.js'

const FILTER_PRIOR = 0.45
const DEFAULT_SAMPLE = 4096
const NDV_SAMPLE = 8192

type CacheKey = string
const ndvCache = new WeakMap<object, Map<CacheKey, number>>()
const selCache = new WeakMap<object, Map<CacheKey, number>>()

function cacheGet(map: WeakMap<object, Map<CacheKey, number>>, table: TableView, key: string): number | undefined {
  return map.get(table as object)?.get(key)
}

function cacheSet(map: WeakMap<object, Map<CacheKey, number>>, table: TableView, key: string, v: number): void {
  let m = map.get(table as object)
  if (!m) {
    m = new Map()
    map.set(table as object, m)
  }
  m.set(key, v)
}

/** Flatten top-level AND into conjuncts. */
export function splitAnd(expr: ExprNode): ExprNode[] {
  if (expr.type === 'binary' && expr.op === 'and') {
    return [...splitAnd(expr.left), ...splitAnd(expr.right)]
  }
  return [expr]
}

/**
 * Deterministic stride sample of row indices into `out` (length ≤ maxSample).
 * Returns number of indices written.
 */
export function sampleIndices(n: number, maxSample = DEFAULT_SAMPLE, seed = 1): Uint32Array {
  if (n <= 0) return new Uint32Array(0)
  const k = Math.min(n, maxSample)
  const out = new Uint32Array(k)
  if (k === n) {
    for (let i = 0; i < n; i++) out[i] = i
    return out
  }
  // Mixed congruential stride from seed for reproducibility without storing RNG state.
  let x = (seed * 1664525 + 1013904223) >>> 0
  const used = new Set<number>()
  let j = 0
  while (j < k) {
    x = (x * 1664525 + 1013904223) >>> 0
    const i = x % n
    if (!used.has(i)) {
      used.add(i)
      out[j++] = i
    }
  }
  return out
}

export function sampleColumnValues(col: Column, nRows: number, maxSample = DEFAULT_SAMPLE, seed = 1): unknown[] {
  const idx = sampleIndices(nRows, maxSample, seed)
  const out: unknown[] = []
  for (let j = 0; j < idx.length; j++) {
    const i = idx[j]!
    if (!isValid(col.nullBitmap, i)) {
      out.push(null)
      continue
    }
    const v = getValue(col.data, i)
    if (col.field.dtype === 'category' && col.dictionary) {
      out.push(col.dictionary[Number(v)] ?? null)
    } else out.push(v)
  }
  return out
}

/** Approx number of distinct values; sample unique scaled to full n. */
export function approxNdv(col: Column, nRows: number, maxSample = NDV_SAMPLE, seed = 1): number {
  if (nRows <= 0) return 0
  if (col.field.dtype === 'category' && col.dictionary) {
    // Codes in use
    const data = col.data as Uint32Array
    const seen = new Set<number>()
    const idx = sampleIndices(nRows, maxSample, seed)
    for (let j = 0; j < idx.length; j++) {
      const i = idx[j]!
      if (!isValid(col.nullBitmap, i)) continue
      seen.add(data[i]!)
    }
    const u = seen.size
    const s = idx.length
    if (s >= nRows) return Math.min(nRows, u + (col.nullBitmap ? 1 : 0))
    const scaled = Math.ceil((u * nRows) / s)
    return Math.max(u, Math.min(nRows, scaled))
  }
  const vals = sampleColumnValues(col, nRows, maxSample, seed)
  const seen = new Set<string>()
  let nulls = 0
  for (const v of vals) {
    if (v === null || v === undefined) nulls++
    else seen.add(String(v))
  }
  const u = seen.size + (nulls > 0 ? 1 : 0)
  const s = vals.length
  if (s >= nRows) return Math.min(nRows, u)
  return Math.max(u, Math.min(nRows, Math.ceil((u * nRows) / s)))
}

export function approxNdvCached(table: TableView, colName: string): number {
  const hit = cacheGet(ndvCache, table, colName)
  if (hit !== undefined) return hit
  const col = getColumn(table, colName)
  const v = approxNdv(col, table.numRows)
  cacheSet(ndvCache, table, colName, v)
  return v
}

type CmpOp = 'eq' | 'neq' | 'gt' | 'gte' | 'lt' | 'lte'

function matchCmp(expr: ExprNode): { col: string; op: CmpOp; lit: number | string | boolean } | null {
  if (expr.type !== 'binary') return null
  if (!['eq', 'neq', 'gt', 'gte', 'lt', 'lte'].includes(expr.op)) return null
  if (expr.left.type === 'col' && expr.right.type === 'lit') {
    return { col: expr.left.name, op: expr.op as CmpOp, lit: expr.right.value as number | string | boolean }
  }
  if (expr.right.type === 'col' && expr.left.type === 'lit') {
    // Flip comparison
    const flip: Record<CmpOp, CmpOp> = {
      eq: 'eq',
      neq: 'neq',
      gt: 'lt',
      gte: 'lte',
      lt: 'gt',
      lte: 'gte',
    }
    return { col: expr.right.name, op: flip[expr.op as CmpOp], lit: expr.left.value as number | string | boolean }
  }
  return null
}

function cmpHolds(op: CmpOp, v: number | string | boolean, lit: number | string | boolean): boolean {
  if (typeof v === 'number' && typeof lit === 'number') {
    switch (op) {
      case 'eq':
        return v === lit
      case 'neq':
        return v !== lit
      case 'gt':
        return v > lit
      case 'gte':
        return v >= lit
      case 'lt':
        return v < lit
      case 'lte':
        return v <= lit
    }
  }
  const a = String(v)
  const b = String(lit)
  switch (op) {
    case 'eq':
      return a === b
    case 'neq':
      return a !== b
    case 'gt':
      return a > b
    case 'gte':
      return a >= b
    case 'lt':
      return a < b
    case 'lte':
      return a <= b
  }
}

/**
 * Estimate filter selectivity in (0, 1] from a concrete table + predicate.
 * Unknown shapes → FILTER_PRIOR per conjunct (multiplied).
 */
export function estimateFilterSelectivity(table: TableView, predicate: ExprNode): number {
  const key = JSON.stringify(predicate)
  const hit = cacheGet(selCache, table, key)
  if (hit !== undefined) return hit

  const parts = splitAnd(predicate)
  let sel = 1
  for (const p of parts) {
    const cmp = matchCmp(p)
    if (!cmp || cmp.lit === null) {
      sel *= FILTER_PRIOR
      continue
    }
    let col: Column
    try {
      col = getColumn(table, cmp.col)
    } catch {
      sel *= FILTER_PRIOR
      continue
    }
    const vals = sampleColumnValues(col, table.numRows)
    let ok = 0
    let total = 0
    for (const v of vals) {
      if (v === null || v === undefined) continue
      total++
      if (cmpHolds(cmp.op, v as number | string | boolean, cmp.lit)) ok++
    }
    if (total === 0) sel *= FILTER_PRIOR
    else sel *= Math.max(0.001, Math.min(1, ok / total))
  }
  const out = Math.max(0.001, Math.min(1, sel))
  cacheSet(selCache, table, key, out)
  return out
}

/** Walk through project/drop/rename/filter to find an underlying scan table, if any. */
export function findBaseScan(plan: PlanNode): { table: TableView; plan: PlanNode } | null {
  switch (plan.type) {
    case 'scan':
      return { table: plan.table, plan }
    case 'project':
    case 'drop':
    case 'rename':
    case 'filter':
    case 'limit':
    case 'sort':
    case 'slice':
      return findBaseScan(plan.input)
    default:
      return null
  }
}

/** Resolve a named column on a near-scan plan for NDV. */
export function findColumnOnNearScan(plan: PlanNode, name: string): { table: TableView; col: Column } | null {
  const base = findBaseScan(plan)
  if (!base) return null
  // After rename, map name back through rename chain
  let colName = name
  let p: PlanNode = plan
  const renames: Array<Record<string, string>> = []
  while (p.type !== 'scan') {
    if (p.type === 'rename') renames.push(p.mapping)
    if (!('input' in p) || !p.input) break
    p = p.input
  }
  // reverse: output name → input name
  for (let i = renames.length - 1; i >= 0; i--) {
    const m = renames[i]!
    let found = false
    for (const [from, to] of Object.entries(m)) {
      if (to === colName) {
        colName = from
        found = true
        break
      }
    }
    if (!found && Object.hasOwn(m, colName)) {
      // name was renamed away
    }
  }
  try {
    return { table: base.table, col: getColumn(base.table, colName) }
  } catch {
    return null
  }
}

/**
 * Lightweight cardinality estimate for cost heuristics.
 */
export function estimatePlanRows(plan: PlanNode): number {
  switch (plan.type) {
    case 'scan':
      return plan.table.numRows
    case 'filter': {
      const n = estimatePlanRows(plan.input)
      if (n === 0) return 0
      const base = findBaseScan(plan.input)
      let sel: number
      if (base) {
        try {
          sel = estimateFilterSelectivity(base.table, plan.predicate)
        } catch {
          sel = Math.pow(FILTER_PRIOR, Math.max(1, splitAnd(plan.predicate).length))
        }
      } else {
        sel = Math.pow(FILTER_PRIOR, Math.max(1, splitAnd(plan.predicate).length))
      }
      return Math.max(1, Math.floor(n * sel))
    }
    case 'limit': {
      const n = estimatePlanRows(plan.input)
      const start = plan.offset ?? 0
      return Math.max(0, Math.min(n, Math.max(0, plan.n) + Math.max(0, start)) - Math.max(0, start))
    }
    case 'slice': {
      const n = estimatePlanRows(plan.input)
      const start = plan.start < 0 ? Math.max(0, n + plan.start) : plan.start
      const end = plan.end === undefined ? n : plan.end < 0 ? n + plan.end : plan.end
      return Math.max(0, Math.min(n, end) - Math.min(n, Math.max(0, start)))
    }
    case 'sample': {
      const n = estimatePlanRows(plan.input)
      if (plan.n !== undefined) return Math.min(n, Math.max(0, plan.n))
      if (plan.fraction !== undefined) return Math.max(0, Math.floor(n * Math.min(1, Math.max(0, plan.fraction))))
      return n
    }
    case 'unique': {
      const n = estimatePlanRows(plan.input)
      return Math.max(1, Math.floor(n * 0.7))
    }
    case 'groupBy': {
      const n = estimatePlanRows(plan.input)
      return Math.max(1, Math.min(n, Math.floor(Math.sqrt(n) * 4) || 1))
    }
    case 'join': {
      const l = estimatePlanRows(plan.left)
      const r = estimatePlanRows(plan.right)
      if (plan.how === 'cross') return l * r
      if (plan.how === 'left') return l
      if (plan.how === 'right') return r
      if (plan.how === 'outer') return l + r

      const ndvPair = (side: PlanNode, keys: string[]): number | null => {
        if (keys.length === 0) return null
        let ndv = 1
        for (const k of keys) {
          const found = findColumnOnNearScan(side, k)
          if (!found) return null
          ndv = Math.max(ndv, approxNdv(found.col, found.table.numRows))
        }
        return ndv
      }

      if (plan.how === 'semi' || plan.how === 'anti') {
        const ndvL = ndvPair(plan.left, plan.leftOn)
        const ndvR = ndvPair(plan.right, plan.rightOn)
        if (ndvL && ndvR && ndvL > 0) {
          const overlap = Math.min(1, ndvR / ndvL)
          const sel = plan.how === 'semi' ? overlap : 1 - overlap * 0.9
          return Math.max(1, Math.floor(l * Math.max(0.05, Math.min(1, sel))))
        }
        return Math.max(1, Math.floor(l * 0.5))
      }

      if (plan.how === 'inner') {
        if (l === 0 || r === 0) return 0
        const ndvL = ndvPair(plan.left, plan.leftOn)
        const ndvR = ndvPair(plan.right, plan.rightOn)
        if (ndvL && ndvR) {
          const d = Math.max(ndvL, ndvR, 1)
          const raw = (l * r) / d
          const cap = Math.min(l * r, Math.max(l, r) * 4)
          return Math.max(1, Math.floor(Math.min(raw, cap)))
        }
        const hi = Math.max(l, r)
        const lo = Math.min(l, r)
        return Math.max(1, Math.floor(lo * Math.min(2, hi / Math.max(lo, 1))))
      }
      return l + r
    }
    case 'asofJoin':
      return estimatePlanRows(plan.left)
    case 'concat':
      if (plan.how === 'vertical') {
        return plan.frames.reduce((s, f) => s + estimatePlanRows(f), 0)
      }
      return Math.max(0, ...plan.frames.map(estimatePlanRows))
    case 'project':
    case 'drop':
    case 'rename':
    case 'withColumn':
    case 'withColumns':
    case 'sort':
    case 'fillNull':
    case 'ffill':
    case 'bfill':
    case 'dropNull':
    case 'interpolate':
    case 'window':
    case 'rolling':
    case 'expanding':
    case 'take':
      return estimatePlanRows(plan.input)
    case 'valueCounts':
    case 'describe':
    case 'corr':
    case 'pivot':
      return Math.max(1, Math.min(estimatePlanRows(plan.input), 10_000))
    case 'melt':
      return estimatePlanRows(plan.input) * Math.max(1, plan.valueVars.length)
    default:
      if ('input' in plan && plan.input) return estimatePlanRows(plan.input)
      return maxScanRows(plan)
  }
}

function maxScanRows(plan: PlanNode): number {
  if (plan.type === 'scan') return plan.table.numRows
  if (plan.type === 'join' || plan.type === 'asofJoin') {
    return Math.max(maxScanRows(plan.left), maxScanRows(plan.right))
  }
  if (plan.type === 'concat') return Math.max(0, ...plan.frames.map(maxScanRows))
  if ('input' in plan && plan.input) return maxScanRows(plan.input)
  return 0
}
