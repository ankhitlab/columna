import {
  allocateData,
  cloneColumn,
  getColumn,
  getField,
  getValue,
  isNumeric,
  isValid,
  setValid,
  setValue,
  sliceTable,
  tableFromColumns,
  takeColumn,
  takeTable,
  type Column,
  type DType,
  type TableView,
} from '@columna/arrow'
import type { AggKind, Backend, CorrMethod, ExprNode, JoinKind, PlanNode, QuantileMethod, RankMethod } from './types.js'
import { orderRows, overAggregateNumeric, overCumulativeNumeric, partitionIds, type Partition } from './over.js'
import {
  applyMathOp,
  matchDualGtFilter,
  projectColumnNames,
  tryFastDescribe,
  argsortNumeric,
  countUniqueNumeric,
  quantilePos,
  quantileSelect,
  tryFastExprColumn,
  tryFastFilter,
  tryFastGroupBy,
  tryFastJoin,
  tryFastRolling,
  tryFastSort,
  tryFastUnique,
  tryFusedFilterGroupBy,
} from './fast.js'
import { parallelDualGtIndices, parallelTakeTable } from './parallel.js'
import {
  asofJoinTables,
  expandingTable,
  explodeTable,
  interpolateTable,
  sampleTable,
  transposeTable,
  unnestTable,
  withColumnsTable,
} from './plan_extra.js'

/** Display/grouping label for a row, decoding dictionary-encoded columns. */
function labelAt(col: Column, row: number): string {
  const raw = getValue(col.data, row)
  if (col.field.dtype === 'category' && col.dictionary) return String(col.dictionary[Number(raw)])
  return String(raw)
}

function resolveColumnName(expr: ExprNode): string | null {
  if (expr.type === 'col') return expr.name
  if (expr.type === 'alias') return expr.name
  if (expr.type === 'agg' && expr.expr.type === 'col') return `${expr.op}_${expr.expr.name}`
  return null
}

function evalExprScalar(
  expr: ExprNode,
  table: TableView,
  row: number,
  groupRows?: number[],
): number | string | boolean | null {
  switch (expr.type) {
    case 'col': {
      const col = getColumn(table, expr.name)
      if (!isValid(col.nullBitmap, row)) return null
      const v = getValue(col.data, row)
      if (col.field.dtype === 'bool') return Boolean(v)
      if (col.field.dtype === 'category' && col.dictionary) return col.dictionary[Number(v)] ?? null
      return v
    }
    case 'lit':
      return expr.value
    case 'alias':
      return evalExprScalar(expr.expr, table, row, groupRows)
    case 'cast': {
      const v = evalExprScalar(expr.expr, table, row, groupRows)
      if (v === null) return null
      return castValue(v, expr.dtype)
    }
    case 'fillNull': {
      const v = evalExprScalar(expr.expr, table, row, groupRows)
      return v === null ? expr.value : v
    }
    case 'unary': {
      const v = evalExprScalar(expr.expr, table, row, groupRows)
      if (expr.op === 'isNull') return v === null
      if (expr.op === 'isNotNull') return v !== null
      if (v === null) return null
      if (expr.op === 'not') return !v
      if (expr.op === 'neg') return -Number(v)
      if (expr.op === 'abs') return Math.abs(Number(v))
      return applyMathOp(expr.op, Number(v), expr.decimals ?? 0, expr.base)
    }
    case 'binary': {
      const l = evalExprScalar(expr.left, table, row, groupRows)
      const r = evalExprScalar(expr.right, table, row, groupRows)
      if (expr.op === 'and') return Boolean(l) && Boolean(r)
      if (expr.op === 'or') return Boolean(l) || Boolean(r)
      if (l === null || r === null) return null
      switch (expr.op) {
        case 'eq':
          return l === r
        case 'neq':
          return l !== r
        case 'gt':
          return (l as number | string) > (r as number | string)
        case 'gte':
          return (l as number | string) >= (r as number | string)
        case 'lt':
          return (l as number | string) < (r as number | string)
        case 'lte':
          return (l as number | string) <= (r as number | string)
        case 'add':
          return Number(l) + Number(r)
        case 'sub':
          return Number(l) - Number(r)
        case 'mul':
          return Number(l) * Number(r)
        case 'div':
          return Number(l) / Number(r)
        case 'mod':
          return Number(l) % Number(r)
        case 'pow':
          return Number(l) ** Number(r)
      }
      break
    }
    case 'agg':
      // Scalar-context aggregates are resolved up front by broadcastAggregates(); reaching here
      // means a plan node evaluated an expression without that pass.
      if (!groupRows) throw new Error('Aggregation used outside groupBy or a broadcast scalar context')
      return aggregateValues(expr.op, expr.expr, table, groupRows, expr.q ?? 0.5, expr.qm)
    case 'str': {
      const v = evalExprScalar(expr.expr, table, row, groupRows)
      if (v === null) return null
      const s = String(v)
      switch (expr.op) {
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
        case 'concat': {
          const o = expr.other ? evalExprScalar(expr.other, table, row, groupRows) : ''
          if (o === null) return null
          return s + (expr.separator ?? '') + String(o)
        }
        case 'padStart':
          return s.padStart(expr.length ?? 0, expr.fill ?? ' ')
        case 'padEnd':
          return s.padEnd(expr.length ?? 0, expr.fill ?? ' ')
      }
      return null
    }
    case 'dt': {
      const v = evalExprScalar(expr.expr, table, row, groupRows)
      if (v === null) return null
      const ms = Number(v)
      if (!Number.isFinite(ms)) return null
      if (expr.op === 'epochMillis') return ms
      // UTC parts without Date allocation (civil_from_days)
      const dayMs = 86_400_000
      const days = Math.floor(ms / dayMs)
      const tod = ms - days * dayMs
      if (expr.op === 'hour') return Math.floor(tod / 3_600_000)
      if (expr.op === 'minute') return Math.floor((tod % 3_600_000) / 60_000)
      if (expr.op === 'second') return Math.floor((tod % 60_000) / 1000)
      if (expr.op === 'weekday') return (days + 4) % 7
      const z = days + 719468
      const era = Math.floor((z >= 0 ? z : z - 146096) / 146097)
      const doe = z - era * 146097
      const yoe = Math.floor((doe - Math.floor(doe / 1460) + Math.floor(doe / 36524) - Math.floor(doe / 146096)) / 365)
      const y = yoe + era * 400
      const doy = doe - (365 * yoe + Math.floor(yoe / 4) - Math.floor(yoe / 100))
      const mp = Math.floor((5 * doy + 2) / 153)
      const d = doy - Math.floor((153 * mp + 2) / 5) + 1
      const m = mp < 10 ? mp + 3 : mp - 9
      const year = m <= 2 ? y + 1 : y
      if (expr.op === 'year') return year
      if (expr.op === 'month') return m
      if (expr.op === 'day') return d
      return null
    }
    case 'when': {
      for (const branch of expr.branches) {
        if (evalExprScalar(branch.when, table, row, groupRows)) {
          return evalExprScalar(branch.then, table, row, groupRows)
        }
      }
      return evalExprScalar(expr.otherwise, table, row, groupRows)
    }
    case 'isIn': {
      const v = evalExprScalar(expr.expr, table, row, groupRows)
      return expr.values.some((x) => Object.is(x, v) || x === v)
    }
    case 'isBetween': {
      const v = evalExprScalar(expr.expr, table, row, groupRows)
      const lo = evalExprScalar(expr.low, table, row, groupRows)
      const hi = evalExprScalar(expr.high, table, row, groupRows)
      if (v === null || lo === null || hi === null) return null
      const n = v as number | string
      const a = lo as number | string
      const b = hi as number | string
      const closed = expr.closed ?? 'both'
      const ge = closed === 'both' || closed === 'left' ? n >= a : n > a
      const le = closed === 'both' || closed === 'right' ? n <= b : n < b
      return ge && le
    }
    case 'clip': {
      const v = evalExprScalar(expr.expr, table, row, groupRows)
      if (v === null) return null
      let n = Number(v)
      if (expr.min !== undefined) n = Math.max(expr.min, n)
      if (expr.max !== undefined) n = Math.min(expr.max, n)
      return n
    }
    case 'rowOffset': {
      const other = row - expr.periods
      if (other < 0 || other >= table.numRows) return null
      if (expr.kind === 'shift') return evalExprScalar(expr.expr, table, other, groupRows)
      const cur = evalExprScalar(expr.expr, table, row, groupRows)
      const prev = evalExprScalar(expr.expr, table, other, groupRows)
      if (cur === null || prev === null) return null
      const c = Number(cur)
      const p = Number(prev)
      if (expr.kind === 'diff') return c - p
      if (p === 0) return null
      return (c - p) / p
    }
    case 'mapElements': {
      const v = evalExprScalar(expr.expr, table, row, groupRows)
      return expr.fn(v)
    }
    case 'over':
      throw new Error('over() is only supported in a scalar context (withColumn / select / filter / sort)')
  }
  return null
}

function castValue(v: number | string | boolean, dtype: DType): number | string | boolean {
  switch (dtype) {
    case 'utf8':
      return String(v)
    case 'bool':
      return Boolean(v)
    case 'f64':
    case 'f32':
    case 'i32':
    case 'u32':
    case 'datetime':
    case 'category':
      return Number(v)
  }
}

function quantileSorted(sorted: number[], q: number, method?: QuantileMethod): number {
  if (sorted.length === 0) return NaN
  const pos = quantilePos(sorted.length, q, method)
  const lo = Math.floor(pos)
  const hi = Math.ceil(pos)
  if (lo === hi) return sorted[lo]!
  const w = pos - lo
  return sorted[lo]! * (1 - w) + sorted[hi]! * w
}

function aggregateValues(
  op: AggKind,
  expr: ExprNode,
  table: TableView,
  rows: number[],
  q = 0.5,
  method?: QuantileMethod,
): number | string | boolean | null {
  const values: Array<number | string | boolean> = []
  for (const r of rows) {
    const v = evalExprScalar(expr, table, r)
    if (v !== null) values.push(v)
  }
  return aggregateCollected(op, values, q, method)
}

function aggregateCollected(
  op: AggKind,
  values: Array<number | string | boolean>,
  q = 0.5,
  method?: QuantileMethod,
): number | string | boolean | null {
  if (op === 'count') return values.length
  if (op === 'nunique') return new Set(values.map(String)).size
  if (values.length === 0) return null
  if (op === 'first') return values[0]!
  if (op === 'last') return values[values.length - 1]!
  const nums = values.map(Number)
  if (op === 'sum') return nums.reduce((a, b) => a + b, 0)
  if (op === 'mean') return nums.reduce((a, b) => a + b, 0) / nums.length
  if (op === 'min' || op === 'max') {
    // no spread: Math.min(...nums) overflows the argument stack past ~120k values
    let acc = op === 'min' ? Infinity : -Infinity
    for (const v of nums) if (op === 'min' ? v < acc : v > acc) acc = v
    return acc
  }
  if (op === 'var' || op === 'std') {
    const n = nums.length
    if (n < 2) return op === 'std' ? 0 : 0
    const mean = nums.reduce((a, b) => a + b, 0) / n
    let m2 = 0
    for (const x of nums) m2 += (x - mean) ** 2
    const variance = m2 / (n - 1)
    return op === 'var' ? variance : Math.sqrt(variance)
  }
  if (op === 'median' || op === 'quantile') {
    const sorted = [...nums].sort((a, b) => a - b)
    return quantileSorted(sorted, op === 'median' ? 0.5 : q, method)
  }
  return null
}

type AggValue = number | string | boolean | null
type AggMemo = Map<string, Map<string, AggValue>>

/** Memo key for an aggregate's inner expression; UDFs are not serializable, so they never share a key. */
function aggKey(inner: ExprNode): string | null {
  let hasFn = false
  const json = JSON.stringify(inner, (_, v) => {
    if (typeof v === 'function') hasFn = true
    return v
  })
  return hasFn ? null : json
}

function hasAgg(expr: ExprNode): boolean {
  switch (expr.type) {
    case 'col':
    case 'lit':
      return false
    case 'agg':
    case 'over':
      return true
    case 'binary':
      return hasAgg(expr.left) || hasAgg(expr.right)
    case 'isBetween':
      return hasAgg(expr.expr) || hasAgg(expr.low) || hasAgg(expr.high)
    case 'when':
      return expr.branches.some((b) => hasAgg(b.when) || hasAgg(b.then)) || hasAgg(expr.otherwise)
    default:
      return hasAgg(expr.expr)
  }
}

/** Every agg op requested on each inner expression in the tree — lets one pass serve mean + std + min … */
function collectAggNeeds(expr: ExprNode, needs: Map<string, Set<AggKind>>): void {
  switch (expr.type) {
    case 'col':
    case 'lit':
      return
    case 'agg': {
      const key = aggKey(expr.expr)
      if (key !== null) {
        let set = needs.get(key)
        if (!set) needs.set(key, (set = new Set()))
        set.add(expr.op)
      }
      collectAggNeeds(expr.expr, needs)
      return
    }
    case 'over':
      return // per-partition aggregates are computed separately (see rewriteOver)
    case 'binary':
      collectAggNeeds(expr.left, needs)
      collectAggNeeds(expr.right, needs)
      return
    case 'isBetween':
      collectAggNeeds(expr.expr, needs)
      collectAggNeeds(expr.low, needs)
      collectAggNeeds(expr.high, needs)
      return
    case 'when':
      for (const b of expr.branches) {
        collectAggNeeds(b.when, needs)
        collectAggNeeds(b.then, needs)
      }
      collectAggNeeds(expr.otherwise, needs)
      return
    default:
      collectAggNeeds(expr.expr, needs)
  }
}

/**
 * Aggregate over every row of the table (scalar-context broadcast). Computes every op in `ops` for the
 * same expression in one streaming pass over a typed column (Welford only when std/var is needed);
 * median / quantile / nunique take one extra typed copy each. Non-numeric inputs fall back to boxing once.
 */
function aggregateWholeTable(
  ops: Set<AggKind>,
  expr: ExprNode,
  table: TableView,
  qs: Map<AggKind, number>,
  method?: QuantileMethod,
): Map<string, AggValue> {
  const n = table.numRows
  const inner = expr.type === 'alias' ? expr.expr : expr
  const out = new Map<string, AggValue>()
  // A column, or any expression the typed vector path can materialize (e.g. mean(log x)), streams below.
  const col = inner.type === 'col' ? getColumn(table, inner.name) : tryFastExprColumn(table, inner, '__agg')
  if (col && (isNumeric(col.field.dtype) || col.field.dtype === 'datetime')) {
    const data = col.data as ArrayLike<number>
    const bm = col.nullBitmap
    const needStd = ops.has('std') || ops.has('var')
    let count = 0
    let sum = 0
    let min = Infinity
    let max = -Infinity
    let first: number | null = null
    let last: number | null = null
    const needMinMax = ops.has('min') || ops.has('max')
    if (!bm) {
      // Dense column: separate tight loops beat one branchy loop (min/max branches mispredict on
      // random data and cost more than a second sweep over the array).
      count = n
      for (let i = 0; i < n; i++) sum += data[i]!
      if (needMinMax) {
        for (let i = 0; i < n; i++) {
          const v = data[i]!
          if (v < min) min = v
          if (v > max) max = v
        }
      }
      if (n > 0) {
        first = data[0]!
        last = data[n - 1]!
      }
    } else {
      for (let i = 0; i < n; i++) {
        if (!isValid(bm, i)) continue
        const v = data[i]!
        count++
        sum += v
        if (v < min) min = v
        if (v > max) max = v
        if (first === null) first = v
        last = v
      }
    }
    // Two-pass centered variance: as stable as Welford, but no per-element divisions (~3× faster).
    let m2 = 0
    if (needStd && count > 1) {
      const mean = sum / count
      if (!bm) {
        for (let i = 0; i < n; i++) {
          const d = data[i]! - mean
          m2 += d * d
        }
      } else {
        for (let i = 0; i < n; i++) {
          if (!isValid(bm, i)) continue
          const d = data[i]! - mean
          m2 += d * d
        }
      }
    }
    for (const op of ops) {
      let v: AggValue
      switch (op) {
        case 'count':
          v = count
          break
        case 'sum':
          v = count ? sum : null
          break
        case 'mean':
          v = count ? sum / count : null
          break
        case 'min':
          v = count ? min : null
          break
        case 'max':
          v = count ? max : null
          break
        case 'first':
          v = first
          break
        case 'last':
          v = last
          break
        case 'var':
          v = count === 0 ? null : count > 1 ? m2 / (count - 1) : 0
          break
        case 'std':
          v = count === 0 ? null : count > 1 ? Math.sqrt(m2 / (count - 1)) : 0
          break
        case 'nunique':
        case 'median':
        case 'quantile': {
          const vals = new Float64Array(n)
          let k = 0
          for (let i = 0; i < n; i++) if (!bm || isValid(bm, i)) vals[k++] = data[i]!
          if (op === 'nunique') v = countUniqueNumeric(vals, k)
          else v = k === 0 ? null : quantileSelect(vals, k, op === 'median' ? 0.5 : (qs.get(op) ?? 0.5), method)
          break
        }
        default:
          v = null
      }
      out.set(op, v)
    }
    return out
  }
  const values: Array<number | string | boolean> = []
  for (let i = 0; i < n; i++) {
    const v = evalExprScalar(inner, table, i)
    if (v !== null) values.push(v)
  }
  for (const op of ops) out.set(op, aggregateCollected(op, values, qs.get(op) ?? 0.5, method))
  return out
}

/** Rewrite context: the table may grow temporary `__over_k` columns holding per-partition aggregates. */
type BroadcastCtx = {
  table: TableView
  needs: Map<string, Set<AggKind>>
  memo: AggMemo
  parts: Map<string, Partition>
  orders: Map<string, Uint32Array>
  temps: number
}

/** Partition plus (for ordered windows) the row visiting order. */
type Window = { part: Partition; order?: Uint32Array }

export const OVER_TEMP_PREFIX = '__over_'

/** Drop temporary over() columns from an output table. */
function stripTemps(table: TableView): TableView {
  if (!table.columns.some((c) => c.field.name.startsWith(OVER_TEMP_PREFIX))) return table
  return tableFromColumns(table.columns.filter((c) => !c.field.name.startsWith(OVER_TEMP_PREFIX)))
}

/**
 * Aggregates used in a scalar context (withColumn / select / filter / sort) are broadcast over the
 * whole table, polars-style: `col('x').sub(col('x').mean())`. Each `agg` node is evaluated once and
 * replaced by a literal, so row evaluation stays O(1) per row and fast paths still see plain shapes.
 * `agg.over(partitionBy)` is computed per partition and becomes a temporary column appended to the
 * returned table (evaluate the returned expression against the returned table; strip `__over_*`
 * columns from anything you hand back).
 * Returns the same expression object and table when the expression contains no aggregates.
 */
function broadcastAggregates(expr: ExprNode, table: TableView): { expr: ExprNode; table: TableView } {
  if (!hasAgg(expr)) return { expr, table }
  const needs = new Map<string, Set<AggKind>>()
  collectAggNeeds(expr, needs)
  const ctx: BroadcastCtx = { table, needs, memo: new Map(), parts: new Map(), orders: new Map(), temps: 0 }
  const out = rewriteAggs(expr, ctx)
  return { expr: out, table: ctx.table }
}

/** Per-partition aggregate → temporary column; falls back to boxed per-group evaluation for non-numeric inputs. */
function overColumn(agg: Extract<ExprNode, { type: 'agg' }>, inner: ExprNode, win: Window, ctx: BroadcastCtx): ExprNode {
  const table = ctx.table
  const n = table.numRows
  const name = `${OVER_TEMP_PREFIX}${ctx.temps++}`
  const q = agg.q ?? 0.5
  const qm = agg.qm
  const part = win.part
  const src = inner.type === 'col' ? getColumn(table, inner.name) : materializeExprColumn(table, inner, name)
  let col = win.order ? overCumulativeNumeric(src, n, part, win.order, agg.op) : overAggregateNumeric(src, n, part, agg.op, q, qm)
  if (!col && win.order) throw new Error(`over(orderBy): running ${agg.op} needs a numeric column (got ${src.field.dtype})`)
  if (!col) {
    // Boxed path: rows per group, existing per-group evaluator, scatter back
    const rowsByGroup = new Map<number, number[]>()
    for (let i = 0; i < n; i++) {
      const g = part.ids[i]!
      let rows = rowsByGroup.get(g)
      if (!rows) rowsByGroup.set(g, (rows = []))
      rows.push(i)
    }
    const perGroup = new Map<number, number | string | boolean | null>()
    for (const [g, rows] of rowsByGroup) perGroup.set(g, aggregateValues(agg.op, inner, table, rows, q, qm))
    const values: Array<number | string | boolean | null> = new Array(n)
    for (let i = 0; i < n; i++) values[i] = perGroup.get(part.ids[i]!) ?? null
    const sample = values.find((v) => v !== null)
    const dtype: DType = typeof sample === 'string' ? 'utf8' : typeof sample === 'boolean' ? 'bool' : 'f64'
    const data = allocateData(dtype, n)
    const bitmap = new Uint8Array(Math.ceil(n / 8) || 1)
    let anyNull = false
    for (let i = 0; i < n; i++) {
      const v = values[i]
      if (v === null) {
        anyNull = true
        continue
      }
      setValue(data, i, v, dtype)
      setValid(bitmap, i, true)
    }
    col = { field: { name, dtype, nullable: anyNull }, data, nullBitmap: anyNull ? bitmap : undefined }
  }
  col = { ...col, field: { ...col.field, name } }
  ctx.table = tableFromColumns([...table.columns, col])
  return { type: 'col', name }
}

/** Inside over(): every agg node becomes a per-partition temp column; nested over() is handled recursively. */
function rewriteOver(expr: ExprNode, win: Window, ctx: BroadcastCtx): ExprNode {
  const rec = (e: ExprNode) => rewriteOver(e, win, ctx)
  switch (expr.type) {
    case 'col':
    case 'lit':
      return expr
    case 'agg': {
      const inner = rec(expr.expr)
      return overColumn(expr, inner, win, ctx)
    }
    case 'over':
      return rewriteAggs(expr, ctx)
    case 'binary': {
      const left = rec(expr.left)
      const right = rec(expr.right)
      return left === expr.left && right === expr.right ? expr : { ...expr, left, right }
    }
    case 'isBetween': {
      const inner = rec(expr.expr)
      const low = rec(expr.low)
      const high = rec(expr.high)
      return inner === expr.expr && low === expr.low && high === expr.high ? expr : { ...expr, expr: inner, low, high }
    }
    case 'when': {
      const branches = expr.branches.map((b) => ({ when: rec(b.when), then: rec(b.then) }))
      return { ...expr, branches, otherwise: rec(expr.otherwise) }
    }
    default: {
      const inner = rec(expr.expr)
      return inner === expr.expr ? expr : { ...expr, expr: inner }
    }
  }
}

function rewriteAggs(expr: ExprNode, ctx: BroadcastCtx): ExprNode {
  const broadcastAggregates = (e: ExprNode) => rewriteAggs(e, ctx)
  switch (expr.type) {
    case 'col':
    case 'lit':
      return expr
    case 'over': {
      const n = ctx.table.numRows
      const key = expr.partitionBy.join('\0')
      let part = ctx.parts.get(key)
      if (!part) {
        // no partition columns (orderBy only) → one group
        part = expr.partitionBy.length ? partitionIds(ctx.table, expr.partitionBy, n) : { ids: new Uint32Array(n), groups: 1 }
        ctx.parts.set(key, part)
      }
      let order: Uint32Array | undefined
      if (expr.orderBy && expr.orderBy.length) {
        const okey = expr.orderBy.join('\0') + (expr.descending ? '\0desc' : '')
        order = ctx.orders.get(okey)
        if (!order) ctx.orders.set(okey, (order = orderRows(ctx.table, expr.orderBy, Boolean(expr.descending), n)))
      }
      return rewriteOver(expr.expr, { part, order }, ctx)
    }
    case 'agg': {
      const inner = broadcastAggregates(expr.expr)
      const key = aggKey(inner)
      // quantile / median are keyed by q and definition as well; everything else shares one pass per inner expression
      const qKey = (op: AggKind) => (op === 'quantile' || op === 'median' ? `${op}:${op === 'median' ? 0.5 : (expr.q ?? 0.5)}:${expr.qm ?? 'linear'}` : op)
      const opKey = qKey(expr.op)
      let vals = key !== null ? ctx.memo.get(key) : undefined
      if (!vals || !vals.has(opKey)) {
        const ops = new Set<AggKind>(key !== null ? (ctx.needs.get(key) ?? []) : [])
        ops.add(expr.op)
        const qs = new Map<AggKind, number>([['quantile', expr.q ?? 0.5]])
        const computed = aggregateWholeTable(ops, inner, ctx.table, qs, expr.qm)
        if (!vals) vals = new Map()
        for (const [op, v] of computed) vals.set(qKey(op as AggKind), v)
        if (key !== null) ctx.memo.set(key, vals)
      }
      return { type: 'lit', value: vals.get(opKey) ?? null }
    }
    case 'str': {
      const inner = broadcastAggregates(expr.expr)
      const other = expr.other ? broadcastAggregates(expr.other) : undefined
      return inner === expr.expr && other === expr.other ? expr : { ...expr, expr: inner, other }
    }
    case 'unary':
    case 'alias':
    case 'cast':
    case 'fillNull':
    case 'dt':
    case 'isIn':
    case 'clip':
    case 'rowOffset':
    case 'mapElements': {
      const inner = broadcastAggregates(expr.expr)
      return inner === expr.expr ? expr : { ...expr, expr: inner }
    }
    case 'binary': {
      const left = broadcastAggregates(expr.left)
      const right = broadcastAggregates(expr.right)
      return left === expr.left && right === expr.right ? expr : { ...expr, left, right }
    }
    case 'isBetween': {
      const inner = broadcastAggregates(expr.expr)
      const low = broadcastAggregates(expr.low)
      const high = broadcastAggregates(expr.high)
      return inner === expr.expr && low === expr.low && high === expr.high
        ? expr
        : { ...expr, expr: inner, low, high }
    }
    case 'when': {
      let changed = false
      const branches = expr.branches.map((b) => {
        const when = broadcastAggregates(b.when)
        const then = broadcastAggregates(b.then)
        if (when !== b.when || then !== b.then) changed = true
        return when === b.when && then === b.then ? b : { when, then }
      })
      const otherwise = broadcastAggregates(expr.otherwise)
      if (otherwise !== expr.otherwise) changed = true
      return changed ? { ...expr, branches, otherwise } : expr
    }
  }
}

function materializeExprColumn(table: TableView, expr: ExprNode, name: string): Column {
  ;({ expr, table } = broadcastAggregates(expr, table))
  const fast = tryFastExprColumn(table, expr, name)
  if (fast) return fast
  const sample = table.numRows > 0 ? evalExprScalar(expr, table, 0) : null
  let dtype: DType = 'f64'
  if (typeof sample === 'string') dtype = 'utf8'
  else if (typeof sample === 'boolean') dtype = 'bool'
  else if (expr.type === 'col') dtype = getColumn(table, expr.name).field.dtype
  else if (expr.type === 'cast') dtype = expr.dtype
  else if (expr.type === 'agg') dtype = 'f64'
  else if (expr.type === 'str') {
    dtype =
      expr.op === 'len' ||
      expr.op === 'contains' ||
      expr.op === 'startsWith' ||
      expr.op === 'endsWith'
        ? expr.op === 'len'
          ? 'f64'
          : 'bool'
        : 'utf8'
  } else if (expr.type === 'dt') dtype = 'f64'
  else if (expr.type === 'isIn' || expr.type === 'isBetween') dtype = 'bool'

  if (dtype === 'utf8') {
    const data: string[] = new Array(table.numRows)
    let anyNull = false
    const nullBitmap = new Uint8Array(Math.ceil(table.numRows / 8) || 1)
    for (let i = 0; i < table.numRows; i++) {
      const v = evalExprScalar(expr, table, i)
      if (v === null) {
        anyNull = true
        data[i] = ''
      } else {
        setValid(nullBitmap, i, true)
        data[i] = String(v)
      }
    }
    return {
      field: { name, dtype, nullable: anyNull },
      data,
      nullBitmap: anyNull ? nullBitmap : undefined,
    }
  }

  const data = allocateData(dtype, table.numRows)
  let anyNull = false
  const nullBitmap = new Uint8Array(Math.ceil(table.numRows / 8) || 1)
  for (let i = 0; i < table.numRows; i++) {
    const v = evalExprScalar(expr, table, i)
    if (v === null) {
      anyNull = true
      continue
    }
    setValid(nullBitmap, i, true)
    setValue(data, i, v, dtype)
  }
  return {
    field: { name, dtype, nullable: anyNull },
    data,
    nullBitmap: anyNull ? nullBitmap : undefined,
  }
}

// Columns are immutable once built: every writer allocates a fresh buffer, so
// pass-through outputs share the input buffers instead of copying them.
function project(table: TableView, columns: Array<string | ExprNode>): TableView {
  const out: Column[] = []
  for (const c of columns) {
    if (typeof c === 'string') {
      out.push(getColumn(table, c))
    } else {
      const name = resolveColumnName(c) ?? `expr_${out.length}`
      if (c.type === 'col') out.push(getColumn(table, c.name))
      else if (c.type === 'alias' && c.expr.type === 'col') {
        const col = getColumn(table, c.expr.name)
        out.push({ ...col, field: { ...col.field, name: c.name } })
      } else {
        out.push(materializeExprColumn(table, c.type === 'alias' ? c.expr : c, name))
      }
    }
  }
  return tableFromColumns(out)
}

function filterTable(table: TableView, predicate: ExprNode, keep?: readonly string[]): TableView {
  const b = broadcastAggregates(predicate, table)
  if (b.table !== table) {
    // over() temps live only in b.table: evaluate there, but keep the caller's columns
    predicate = b.expr
    keep = keep ?? table.columns.map((c) => c.field.name)
    table = b.table
  } else predicate = b.expr
  const fast = tryFastFilter(table, predicate, keep)
  if (fast) return fast
  const indices: number[] = []
  for (let i = 0; i < table.numRows; i++) {
    if (evalExprScalar(predicate, table, i)) indices.push(i)
  }
  return takeTable(table, indices, keep)
}

function sortTable(
  table: TableView,
  by: Array<{ expr: ExprNode; descending: boolean }>,
  limit?: number,
): TableView {
  let extended = false
  by = by.map((k) => {
    const b = broadcastAggregates(k.expr, table)
    if (b.table !== table) {
      table = b.table
      extended = true
    }
    return b.expr === k.expr ? k : { ...k, expr: b.expr }
  })
  if (extended) return stripTemps(sortTable(table, by, limit))
  const fast = tryFastSort(table, by, limit)
  if (fast) return fast
  const indices = Array.from({ length: table.numRows }, (_, i) => i)
  indices.sort((a, b) => {
    for (const key of by) {
      const va = evalExprScalar(key.expr, table, a)
      const vb = evalExprScalar(key.expr, table, b)
      if (va === vb) continue
      if (va === null) return 1
      if (vb === null) return -1
      const cmp = va < vb ? -1 : 1
      return key.descending ? -cmp : cmp
    }
    return 0
  })
  const sliced = limit !== undefined ? indices.slice(0, limit) : indices
  return tableFromColumns(table.columns.map((c) => takeColumn(c, sliced)))
}

function groupByTable(
  table: TableView,
  keys: string[],
  aggs: Array<{ name: string; expr: ExprNode }>,
): TableView {
  if (keys.length === 0) throw new Error('groupBy requires at least one key')
  const fast = tryFastGroupBy(table, keys, aggs)
  if (fast) return fast
  const groups = new Map<string, number[]>()
  for (let i = 0; i < table.numRows; i++) {
    const parts = keys.map((k) => {
      const col = getColumn(table, k)
      if (!isValid(col.nullBitmap, i)) return '∅'
      const v = getValue(col.data, i)
      if (col.field.dtype === 'category' && col.dictionary) return col.dictionary[Number(v)] ?? '∅'
      return String(v)
    })
    const key = parts.join('\0')
    let arr = groups.get(key)
    if (!arr) {
      arr = []
      groups.set(key, arr)
    }
    arr.push(i)
  }

  const outCols: Column[] = []
  for (const key of keys) {
    const src = getColumn(table, key)
    const data = allocateData(src.field.dtype, groups.size)
    let anyNull = false
    const nullBitmap = new Uint8Array(Math.ceil(groups.size / 8) || 1)
    let row = 0
    for (const idxs of groups.values()) {
      const srcIdx = idxs[0]!
      if (!isValid(src.nullBitmap, srcIdx)) {
        anyNull = true
      } else {
        setValid(nullBitmap, row, true)
        setValue(data, row, getValue(src.data, srcIdx), src.field.dtype)
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

  for (const agg of aggs) {
    const values: Array<number | string | boolean | null> = []
    for (const idxs of groups.values()) {
      if (agg.expr.type === 'agg') {
        values.push(
          aggregateValues(agg.expr.op, agg.expr.expr, table, idxs, agg.expr.q ?? 0.5, agg.expr.qm),
        )
      } else {
        values.push(evalExprScalar(agg.expr, table, idxs[0]!, idxs))
      }
    }
    const dtype: DType = typeof values.find((v) => v !== null) === 'string' ? 'utf8' : 'f64'
    const data = allocateData(dtype, values.length)
    let anyNull = false
    const nullBitmap = new Uint8Array(Math.ceil(values.length / 8) || 1)
    for (let i = 0; i < values.length; i++) {
      const v = values[i]
      if (v === null || v === undefined) {
        anyNull = true
        continue
      }
      setValid(nullBitmap, i, true)
      setValue(data, i, v, dtype)
    }
    outCols.push({
      field: { name: agg.name, dtype, nullable: anyNull },
      data,
      nullBitmap: anyNull ? nullBitmap : undefined,
    })
  }

  return tableFromColumns(outCols)
}

function joinKey(table: TableView, cols: string[], row: number): string {
  return cols
    .map((name) => {
      const col = getColumn(table, name)
      if (!isValid(col.nullBitmap, row)) return '∅'
      return String(getValue(col.data, row))
    })
    .join('\0')
}

function expandCrossColumn(col: Column, mode: 'repeat' | 'tile', ln: number, rn: number): Column {
  const outN = ln * rn
  const dtype = col.field.dtype
  const dict = col.dictionary
  const srcBm = col.nullBitmap

  const expandBitmap = (): Uint8Array | undefined => {
    if (!srcBm) return undefined
    if (mode === 'repeat') {
      let anyNull = false
      for (let i = 0; i < ln; i++) {
        if (!isValid(srcBm, i)) {
          anyNull = true
          break
        }
      }
      if (!anyNull) return undefined
      const nullBitmap = new Uint8Array(Math.ceil(outN / 8) || 1)
      for (let i = 0; i < ln; i++) {
        if (!isValid(srcBm, i)) continue
        const base = i * rn
        for (let j = 0; j < rn; j++) setValid(nullBitmap, base + j, true)
      }
      return nullBitmap
    }
    // Tile right validity pattern for each left row
    let rightAnyNull = false
    for (let j = 0; j < rn; j++) {
      if (!isValid(srcBm, j)) {
        rightAnyNull = true
        break
      }
    }
    if (!rightAnyNull) return undefined
    const nullBitmap = new Uint8Array(Math.ceil(outN / 8) || 1)
    const rightValid = new Uint8Array(rn)
    for (let j = 0; j < rn; j++) if (isValid(srcBm, j)) rightValid[j] = 1
    for (let i = 0; i < ln; i++) {
      const base = i * rn
      for (let j = 0; j < rn; j++) if (rightValid[j]) setValid(nullBitmap, base + j, true)
    }
    return nullBitmap
  }

  if (dtype === 'utf8') {
    const src = col.data as string[]
    const data = new Array<string>(outN)
    if (mode === 'repeat') {
      for (let i = 0; i < ln; i++) data.fill(src[i]!, i * rn, i * rn + rn)
    } else if (rn > 0) {
      for (let j = 0; j < rn; j++) data[j] = src[j]!
      let filled = rn
      while (filled < outN) {
        const copy = Math.min(filled, outN - filled)
        data.copyWithin(filled, 0, copy)
        filled += copy
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

  if (mode === 'repeat') {
    for (let i = 0; i < ln; i++) alloc.fill(src[i]! as never, i * rn, i * rn + rn)
  } else if (rn > 0) {
    for (let j = 0; j < rn; j++) alloc[j] = src[j]! as never
    let filled = rn
    while (filled < outN) {
      const copy = Math.min(filled, outN - filled)
      alloc.copyWithin(filled, 0, copy)
      filled += copy
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

function crossJoinTables(left: TableView, right: TableView): TableView {
  const ln = left.numRows
  const rn = right.numRows
  if (ln === 0 || rn === 0) {
    return tableFromColumns([
      ...left.columns.map((c) => takeColumn(c, [])),
      ...right.columns.map((c) => {
        const name = left.schema.some((f) => f.name === c.field.name) ? `${c.field.name}_right` : c.field.name
        return { ...takeColumn(c, []), field: { ...c.field, name } }
      }),
    ])
  }
  const leftNames = new Set(left.schema.map((f) => f.name))
  const out: Column[] = []
  for (const col of left.columns) out.push(expandCrossColumn(col, 'repeat', ln, rn))
  for (const col of right.columns) {
    const name = leftNames.has(col.field.name) ? `${col.field.name}_right` : col.field.name
    const expanded = expandCrossColumn(col, 'tile', ln, rn)
    out.push({ ...expanded, field: { ...expanded.field, name } })
  }
  return tableFromColumns(out)
}

function joinTables(
  left: TableView,
  right: TableView,
  leftOn: string[],
  rightOn: string[],
  how: JoinKind,
): TableView {
  if (how === 'cross') return crossJoinTables(left, right)

  if (how === 'inner' || how === 'left' || how === 'right' || how === 'outer' || how === 'semi' || how === 'anti') {
    const fast = tryFastJoin(left, right, leftOn, rightOn, how)
    if (fast) return fast
  }

  const rightIndex = new Map<string, number[]>()
  for (let i = 0; i < right.numRows; i++) {
    const k = joinKey(right, rightOn, i)
    let arr = rightIndex.get(k)
    if (!arr) {
      arr = []
      rightIndex.set(k, arr)
    }
    arr.push(i)
  }

  if (how === 'semi' || how === 'anti') {
    const indices: number[] = []
    for (let i = 0; i < left.numRows; i++) {
      const hits = rightIndex.get(joinKey(left, leftOn, i))
      const has = Boolean(hits && hits.length)
      if ((how === 'semi' && has) || (how === 'anti' && !has)) indices.push(i)
    }
    return takeTable(left, indices)
  }

  const leftIdx: number[] = []
  const rightIdx: Array<number | null> = []
  const matchedRight = new Set<number>()

  for (let i = 0; i < left.numRows; i++) {
    const k = joinKey(left, leftOn, i)
    const hits = rightIndex.get(k)
    if (hits && hits.length) {
      for (const r of hits) {
        leftIdx.push(i)
        rightIdx.push(r)
        matchedRight.add(r)
      }
    } else if (how === 'left' || how === 'outer') {
      leftIdx.push(i)
      rightIdx.push(null)
    }
  }

  if (how === 'right' || how === 'outer') {
    for (let r = 0; r < right.numRows; r++) {
      if (!matchedRight.has(r)) {
        leftIdx.push(-1)
        rightIdx.push(r)
      }
    }
  }

  return assembleJoin(left, right, leftIdx, rightIdx, leftOn, rightOn)
}

function assembleJoin(
  left: TableView,
  right: TableView,
  leftIdx: number[],
  rightIdx: Array<number | null>,
  leftOn: string[],
  rightOn: string[],
): TableView {
  const outCols: Column[] = []
  const leftGather = new Uint32Array(leftIdx.length)
  let leftHasSentinel = false
  for (let i = 0; i < leftIdx.length; i++) {
    const v = leftIdx[i]!
    if (v < 0) {
      leftGather[i] = 0
      leftHasSentinel = true
    } else leftGather[i] = v
  }
  for (const col of left.columns) {
    const cloned = takeColumn(col, leftGather)
    if (!leftHasSentinel) {
      outCols.push(cloned)
      continue
    }
    let anyNull = Boolean(cloned.nullBitmap)
    const nullBitmap = cloned.nullBitmap
      ? new Uint8Array(cloned.nullBitmap)
      : new Uint8Array(Math.ceil(leftIdx.length / 8) || 1)
    if (!cloned.nullBitmap) nullBitmap.fill(0xff)
    for (let i = 0; i < leftIdx.length; i++) {
      if (leftIdx[i]! < 0) {
        setValid(nullBitmap, i, false)
        anyNull = true
      }
    }
    outCols.push({
      ...cloned,
      nullBitmap: anyNull ? nullBitmap : undefined,
      field: { ...cloned.field, nullable: anyNull || cloned.field.nullable },
    })
  }

  const leftNames = new Set(left.schema.map((f) => f.name))
  const rightGather = new Uint32Array(rightIdx.length)
  let rightHasSentinel = false
  for (let i = 0; i < rightIdx.length; i++) {
    const v = rightIdx[i]
    if (v === null) {
      rightGather[i] = 0
      rightHasSentinel = true
    } else rightGather[i] = v
  }
  for (const col of right.columns) {
    if (rightOn.includes(col.field.name) && leftOn.includes(col.field.name)) continue
    const name = leftNames.has(col.field.name) ? `${col.field.name}_right` : col.field.name
    const cloned = takeColumn(col, rightGather)
    if (!rightHasSentinel) {
      outCols.push({ ...cloned, field: { ...cloned.field, name } })
      continue
    }
    let anyNull = Boolean(cloned.nullBitmap)
    const nullBitmap = cloned.nullBitmap
      ? new Uint8Array(cloned.nullBitmap)
      : new Uint8Array(Math.ceil(rightIdx.length / 8) || 1)
    if (!cloned.nullBitmap) nullBitmap.fill(0xff)
    for (let i = 0; i < rightIdx.length; i++) {
      if (rightIdx[i] === null) {
        setValid(nullBitmap, i, false)
        anyNull = true
      }
    }
    outCols.push({
      ...cloned,
      field: { ...cloned.field, name, nullable: anyNull || cloned.field.nullable },
      nullBitmap: anyNull ? nullBitmap : undefined,
    })
  }

  return tableFromColumns(outCols)
}

function fillNullTable(table: TableView, value: number | string | boolean, columns?: string[]): TableView {
  const names = columns ?? table.schema.map((f) => f.name)
  return tableFromColumns(
    table.columns.map((col) => {
      if (!names.includes(col.field.name) || !col.nullBitmap) return col
      const data = copyTyped(col.data)
      for (let i = 0; i < table.numRows; i++) {
        if (!isValid(col.nullBitmap, i)) setValue(data, i, value, col.field.dtype)
      }
      return { field: { ...col.field, nullable: false }, data, dictionary: col.dictionary ? [...col.dictionary] : undefined }
    }),
  )
}

function copyTyped(data: Column['data']): Column['data'] {
  if (Array.isArray(data)) return [...data]
  return data.slice() as Column['data']
}

function dropNullTable(table: TableView, columns?: string[]): TableView {
  const names = columns ?? table.schema.map((f) => f.name)
  const cols = names.map((n) => getColumn(table, n))
  const indices: number[] = []
  for (let i = 0; i < table.numRows; i++) {
    if (cols.every((c) => isValid(c.nullBitmap, i))) indices.push(i)
  }
  return tableFromColumns(table.columns.map((c) => cloneColumn(c, indices)))
}

/**
 * Typed melt: id columns are tiled with TypedArray.set (memcpy per repeat),
 * values are concatenated, and `variable` reuses one interned string per block.
 */
function tryFastMelt(
  table: TableView,
  idVars: string[],
  valueVars: string[],
  varName: string,
  valueName: string,
): TableView | null {
  const rows = table.numRows
  if (rows === 0 || valueVars.length === 0) return null

  const idSrc = idVars.map((name) => getColumn(table, name))
  const valSrc = valueVars.map((name) => getColumn(table, name))
  const typed = (c: Column) => !Array.isArray(c.data) && !c.nullBitmap
  if (!idSrc.every(typed) || !valSrc.every(typed)) return null
  if (valSrc.some((c) => c.field.dtype === 'utf8' || c.field.dtype === 'category')) return null

  const n = rows * valueVars.length
  const idCols: Column[] = idSrc.map((src) => {
    const data = allocateData(src.field.dtype, n) as Exclude<Column['data'], string[]>
    for (let k = 0; k < valueVars.length; k++) {
      ;(data as unknown as { set(a: ArrayLike<number>, off: number): void }).set(
        src.data as ArrayLike<number>,
        k * rows,
      )
    }
    return { field: { ...src.field }, data, dictionary: src.dictionary }
  })

  // `variable` is dictionary-encoded: one code per row instead of a 2M-entry
  // string array, which removes the pointer writes and the GC pressure.
  const varCodes = new Uint32Array(n)
  const valueData = new Float64Array(n)
  for (let k = 0; k < valueVars.length; k++) {
    const off = k * rows
    varCodes.fill(k, off, off + rows)
    valueData.set(valSrc[k]!.data as ArrayLike<number>, off)
  }

  return tableFromColumns([
    ...idCols,
    {
      field: { name: varName, dtype: 'category', nullable: false },
      data: varCodes,
      dictionary: [...valueVars],
    },
    { field: { name: valueName, dtype: 'f64', nullable: false }, data: valueData },
  ])
}

function meltTable(
  table: TableView,
  idVars: string[],
  valueVars: string[],
  varName: string,
  valueName: string,
): TableView {
  const fast = tryFastMelt(table, idVars, valueVars, varName, valueName)
  if (fast) return fast
  const n = table.numRows * valueVars.length
  const idCols = idVars.map((name) => {
    const src = getColumn(table, name)
    const data = allocateData(src.field.dtype, n)
    const nullBitmap = new Uint8Array(Math.ceil(n / 8) || 1)
    let anyNull = false
    let out = 0
    for (const vv of valueVars) {
      void vv
      for (let i = 0; i < table.numRows; i++) {
        if (isValid(src.nullBitmap, i)) {
          setValid(nullBitmap, out, true)
          setValue(data, out, getValue(src.data, i), src.field.dtype)
        } else anyNull = true
        out++
      }
    }
    return {
      field: { ...src.field },
      data,
      nullBitmap: anyNull ? nullBitmap : undefined,
      dictionary: src.dictionary ? [...src.dictionary] : undefined,
    } satisfies Column
  })

  const varData: string[] = []
  const valueCol = getColumn(table, valueVars[0]!)
  const valueData = allocateData(valueCol.field.dtype === 'utf8' ? 'utf8' : 'f64', n)
  const valueNull = new Uint8Array(Math.ceil(n / 8) || 1)
  let anyNull = false
  let out = 0
  for (const vv of valueVars) {
    const src = getColumn(table, vv)
    for (let i = 0; i < table.numRows; i++) {
      varData.push(vv)
      if (isValid(src.nullBitmap, i)) {
        setValid(valueNull, out, true)
        setValue(valueData, out, getValue(src.data, i), src.field.dtype === 'utf8' ? 'utf8' : 'f64')
      } else anyNull = true
      out++
    }
  }

  return tableFromColumns([
    ...idCols,
    { field: { name: varName, dtype: 'utf8', nullable: false }, data: varData },
    {
      field: { name: valueName, dtype: valueCol.field.dtype === 'utf8' ? 'utf8' : 'f64', nullable: anyNull },
      data: valueData,
      nullBitmap: anyNull ? valueNull : undefined,
    },
  ])
}

function concatVertical(tables: TableView[]): TableView {
  if (tables.length === 0) return tableFromColumns([])
  const schema = tables[0]!.schema
  for (const t of tables) {
    if (t.schema.length !== schema.length || t.schema.some((f, i) => f.name !== schema[i]!.name)) {
      throw new Error('concat vertical requires matching schemas')
    }
  }
  const total = tables.reduce((s, t) => s + t.numRows, 0)
  const cols = schema.map((field, ci) => {
    const data = allocateData(field.dtype, total)
    let anyNull = false
    const nullBitmap = new Uint8Array(Math.ceil(total / 8) || 1)
    let offset = 0
    for (const t of tables) {
      const src = t.columns[ci]!
      for (let i = 0; i < t.numRows; i++) {
        if (isValid(src.nullBitmap, i)) {
          setValid(nullBitmap, offset + i, true)
          setValue(data, offset + i, getValue(src.data, i), field.dtype)
        } else anyNull = true
      }
      offset += t.numRows
    }
    return {
      field: { ...field },
      data,
      nullBitmap: anyNull ? nullBitmap : undefined,
      dictionary: tables[0]!.columns[ci]!.dictionary ? [...tables[0]!.columns[ci]!.dictionary!] : undefined,
    } satisfies Column
  })
  return tableFromColumns(cols)
}

/** Average ranks (ties share the mean rank) for the first `len` values. */
function averageRanks(values: ArrayLike<number>, len: number): Float64Array {
  // Radix argsort + one linear run-walk: O(n) after the sort, no comparator, no binary search.
  const order = argsortNumeric(values, len, undefined, false)
  const ranks = new Float64Array(len)
  for (let i = 0; i < len; ) {
    const v = values[order[i]!]!
    let j = i
    while (j + 1 < len && values[order[j + 1]!] === v) j++
    const r = (i + 1 + j + 1) / 2
    for (let p = i; p <= j; p++) ranks[order[p]!] = r
    i = j + 1
  }
  return ranks
}

/**
 * Sample covariance / correlation of two numeric columns over pairwise-complete rows (pandas semantics).
 * Pearson and covariance stream in one Welford pass; Spearman ranks the complete pairs first.
 * Fewer than two complete pairs (or zero variance for corr) → NaN.
 */
function pairStat(
  a: Column,
  b: Column,
  n: number,
  kind: 'corr' | 'cov',
  method: CorrMethod,
  rankCache?: Map<Column, Float64Array>,
): number {
  const ad = a.data as ArrayLike<number>
  const bd = b.data as ArrayLike<number>
  const abm = a.nullBitmap
  const bbm = b.nullBitmap

  let xs: ArrayLike<number> = ad
  let ys: ArrayLike<number> = bd
  let rows: Uint32Array | null = null
  let len = n

  if (kind === 'corr' && method === 'spearman') {
    if (!abm && !bbm) {
      // No nulls: ranks over the full column are the same for every pair — computed once per column.
      xs = rankCache?.get(a) ?? averageRanks(ad, n)
      ys = rankCache?.get(b) ?? averageRanks(bd, n)
      rankCache?.set(a, xs as Float64Array)
      rankCache?.set(b, ys as Float64Array)
    } else {
      const xv = new Float64Array(n)
      const yv = new Float64Array(n)
      let k = 0
      for (let i = 0; i < n; i++) {
        if ((abm && !isValid(abm, i)) || (bbm && !isValid(bbm, i))) continue
        xv[k] = ad[i]!
        yv[k] = bd[i]!
        k++
      }
      xs = averageRanks(xv, k)
      ys = averageRanks(yv, k)
      len = k
    }
  } else if (abm || bbm) {
    const idx = new Uint32Array(n)
    let k = 0
    for (let i = 0; i < n; i++) {
      if ((abm && !isValid(abm, i)) || (bbm && !isValid(bbm, i))) continue
      idx[k++] = i
    }
    rows = idx
    len = k
  }

  // Two-pass centered sums: numerically equivalent to Welford for this use and ~2× faster
  // (no per-element divisions); the second pass is over the same two cache-friendly arrays.
  const count = len
  if (count < 2) return NaN
  let sx = 0
  let sy = 0
  for (let p = 0; p < len; p++) {
    const i = rows ? rows[p]! : p
    sx += xs[i]!
    sy += ys[i]!
  }
  const mx = sx / count
  const my = sy / count
  let m2x = 0
  let m2y = 0
  let cxy = 0
  for (let p = 0; p < len; p++) {
    const i = rows ? rows[p]! : p
    const dx = xs[i]! - mx
    const dy = ys[i]! - my
    cxy += dx * dy
    m2x += dx * dx
    m2y += dy * dy
  }
  if (kind === 'cov') return cxy / (count - 1)
  const denom = Math.sqrt(m2x * m2y)
  return denom > 0 ? cxy / denom : NaN
}

/** pandas-style corr()/cov(): symmetric matrix with a leading `column` label column. */
function corrTable(table: TableView, kind: 'corr' | 'cov', method: CorrMethod, columns?: string[]): TableView {
  const isNum = (c: Column) => isNumeric(c.field.dtype) || c.field.dtype === 'datetime'
  let cols: Column[]
  if (columns) {
    cols = columns.map((name) => {
      const c = getColumn(table, name)
      if (!isNum(c)) throw new Error(`${kind}: column "${name}" is not numeric (${c.field.dtype})`)
      return c
    })
  } else {
    cols = table.columns.filter(isNum)
  }
  const k = cols.length
  const n = table.numRows
  const m = cols.map(() => new Float64Array(k))
  const rankCache = method === 'spearman' ? new Map<Column, Float64Array>() : undefined
  for (let i = 0; i < k; i++) {
    for (let j = i; j < k; j++) {
      const v = i === j && kind === 'corr' ? pairStat(cols[i]!, cols[i]!, n, 'cov', 'pearson') : NaN
      // diagonal of corr is 1 when the column has variance, NaN otherwise (matches pandas)
      const s = i === j && kind === 'corr' ? (v > 0 ? 1 : NaN) : pairStat(cols[i]!, cols[j]!, n, kind, method, rankCache)
      m[i]![j] = s
      m[j]![i] = s
    }
  }
  const out: Column[] = [
    { field: { name: 'column', dtype: 'utf8', nullable: false }, data: cols.map((c) => c.field.name) },
  ]
  for (let j = 0; j < k; j++) {
    const data = new Float64Array(k)
    for (let i = 0; i < k; i++) data[i] = m[i]![j]!
    out.push({ field: { name: cols[j]!.field.name, dtype: 'f64', nullable: false }, data })
  }
  return tableFromColumns(out)
}

function describeTable(table: TableView, method?: QuantileMethod): TableView {
  const fast = tryFastDescribe(table, method)
  if (fast) return fast
  const numeric = table.columns.filter((c) => isNumeric(c.field.dtype) || c.field.dtype === 'datetime')
  const stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
  const out: Column[] = [
    { field: { name: 'stat', dtype: 'utf8', nullable: false }, data: stats },
  ]
  for (const col of numeric) {
    const vals: number[] = []
    for (let i = 0; i < table.numRows; i++) {
      if (isValid(col.nullBitmap, i)) vals.push(Number(getValue(col.data, i)))
    }
    vals.sort((a, b) => a - b)
    const count = vals.length
    const mean = count ? vals.reduce((a, b) => a + b, 0) / count : NaN
    const variance = count > 1 ? vals.reduce((s, v) => s + (v - mean) ** 2, 0) / (count - 1) : NaN
    const std = Math.sqrt(variance)
    const q = (p: number) => {
      if (!count) return NaN
      const idx = quantilePos(count, p, method)
      const lo = Math.floor(idx)
      const hi = Math.ceil(idx)
      if (lo === hi) return vals[lo]!
      return vals[lo]! * (hi - idx) + vals[hi]! * (idx - lo)
    }
    const data = new Float64Array([count, mean, std, vals[0] ?? NaN, q(0.25), q(0.5), q(0.75), vals[count - 1] ?? NaN])
    out.push({ field: { name: col.field.name, dtype: 'f64', nullable: true }, data })
  }
  return tableFromColumns(out)
}

function valueCounts(table: TableView, column: string, normalize: boolean): TableView {
  const col = getColumn(table, column)

  // Dense histogram over dictionary codes — no per-row string decode
  if (col.field.dtype === 'category' && col.dictionary) {
    const codes = col.data as Uint32Array
    const card = col.dictionary.length
    const hist = new Uint32Array(card)
    const order = new Int32Array(card)
    let size = 0
    let total = 0
    for (let i = 0; i < table.numRows; i++) {
      if (col.nullBitmap && !isValid(col.nullBitmap, i)) continue
      const code = codes[i]!
      if (code >= card) continue
      if (hist[code]! === 0) order[size++] = code
      hist[code]!++
      total++
    }
    const keys = new Array<string>(size)
    const values = new Float64Array(size)
    for (let row = 0; row < size; row++) {
      const code = order[row]!
      keys[row] = col.dictionary[code]!
      values[row] = normalize ? hist[code]! / total : hist[code]!
    }
    return tableFromColumns([
      { field: { name: column, dtype: 'utf8', nullable: false }, data: keys },
      {
        field: { name: normalize ? 'proportion' : 'count', dtype: 'f64', nullable: false },
        data: values,
      },
    ])
  }

  const counts = new Map<string, number>()
  let total = 0
  for (let i = 0; i < table.numRows; i++) {
    if (!isValid(col.nullBitmap, i)) continue
    const raw = getValue(col.data, i)
    const key =
      col.field.dtype === 'category' && col.dictionary
        ? String(col.dictionary[Number(raw)])
        : String(raw)
    counts.set(key, (counts.get(key) ?? 0) + 1)
    total++
  }
  const keys = [...counts.keys()]
  const values = keys.map((k) => {
    const c = counts.get(k)!
    return normalize ? c / total : c
  })
  return tableFromColumns([
    { field: { name: column, dtype: 'utf8', nullable: false }, data: keys },
    { field: { name: normalize ? 'proportion' : 'count', dtype: 'f64', nullable: false }, data: new Float64Array(values) },
  ])
}

function uniqueTable(table: TableView, columns: string[] | undefined, keep: 'first' | 'last' | 'none'): TableView {
  const fast = tryFastUnique(table, columns, keep)
  if (fast) return fast
  const cols = columns ?? table.schema.map((f) => f.name)
  const seen = new Map<string, number>()
  const order: string[] = []
  for (let i = 0; i < table.numRows; i++) {
    const key = cols
      .map((n) => {
        const c = getColumn(table, n)
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
  return tableFromColumns(table.columns.map((c) => cloneColumn(c, indices)))
}

/**
 * Rank rows of one partition with tie handling. Rows are already in orderBy order when
 * orderBy is given; otherwise they are sorted by `keyExprs` here (ascending, stable).
 * A row with a null key gets a null rank. Returns the number of null ranks written.
 */
/**
 * Single numeric key, non-ordinal: O(n) run-walk over rows already in key order (orderBy), or over
 * a radix argsort of the key (expr) — no comparator sort, no per-element binary search.
 * Descending flips the sign so rank 1 is the largest value.
 */
function rankPartitionNumeric(
  rows: number[],
  key: Column,
  presorted: boolean,
  descending: boolean,
  method: RankMethod,
  out: Float64Array,
  nullBitmap: Uint8Array,
): number {
  const data = key.data as ArrayLike<number>
  const bm = key.nullBitmap
  const vals = new Float64Array(rows.length)
  const valid = new Uint32Array(rows.length)
  let len = 0
  for (let p = 0; p < rows.length; p++) {
    const i = rows[p]!
    if (bm && !isValid(bm, i)) continue
    vals[len] = data[i]!
    valid[len] = i
    len++
  }
  // order[k] = position (in vals/valid) of the k-th smallest (or largest) value
  let order: ArrayLike<number>
  if (presorted) {
    const seq = new Uint32Array(len)
    for (let k = 0; k < len; k++) seq[k] = k
    order = seq
  } else order = argsortNumeric(vals, len, undefined, descending)

  let dense = 0
  for (let i = 0; i < len; ) {
    const v = vals[order[i]!]!
    let j = i
    while (j + 1 < len && vals[order[j + 1]!] === v) j++
    dense++
    const r = method === 'min' ? i + 1 : method === 'max' ? j + 1 : method === 'dense' ? dense : (i + 1 + j + 1) / 2
    for (let p = i; p <= j; p++) {
      const row = valid[order[p]!]!
      out[row] = r
      setValid(nullBitmap, row, true)
    }
    i = j + 1
  }
  return rows.length - len
}

function rankPartition(
  rows: number[],
  keyCols: Column[],
  presorted: boolean,
  method: RankMethod,
  out: Float64Array,
  nullBitmap: Uint8Array,
  descending = false,
): number {
  if (keyCols.length === 1 && method !== 'ordinal') {
    const d = keyCols[0]!.field.dtype
    if (isNumeric(d) || d === 'datetime') return rankPartitionNumeric(rows, keyCols[0]!, presorted, descending, method, out, nullBitmap)
  }
  // Keys are pre-materialized columns (typed for numeric expressions) — no per-row boxing.
  const nk = keyCols.length
  const keyAt = (pos: number, k: number): number | string | boolean => {
    const c = keyCols[k]!
    const v = getValue(c.data, rows[pos]!)
    return c.field.dtype === 'category' && c.dictionary ? c.dictionary[Number(v)]! : (v as number | string | boolean)
  }
  const order = rows.map((_, pos) => pos)
  const isNullKey = (pos: number): boolean => {
    for (let k = 0; k < nk; k++) if (!isValid(keyCols[k]!.nullBitmap, rows[pos]!)) return true
    return false
  }
  const cmp = (a: number, b: number): number => {
    for (let k = 0; k < nk; k++) {
      const va = keyAt(a, k)
      const vb = keyAt(b, k)
      if (va < vb) return -1
      if (va > vb) return 1
    }
    return 0
  }
  const equalKeys = (a: number, b: number): boolean => {
    for (let k = 0; k < nk; k++) if (keyAt(a, k) !== keyAt(b, k)) return false
    return true
  }

  // Nulls go last and are ranked as null; ordinal keeps position order for ties.
  const ranked = order.filter((pos) => !isNullKey(pos))
  const nulls = order.filter(isNullKey)
  if (!presorted && method !== 'ordinal') ranked.sort(cmp)
  else if (!presorted) ranked.sort((a, b) => cmp(a, b) || a - b)

  let dense = 0
  for (let i = 0; i < ranked.length; ) {
    let j = i
    while (j + 1 < ranked.length && equalKeys(ranked[i]!, ranked[j + 1]!)) j++
    dense++
    for (let p = i; p <= j; p++) {
      let r: number
      switch (method) {
        case 'min':
          r = i + 1
          break
        case 'max':
          r = j + 1
          break
        case 'dense':
          r = dense
          break
        case 'ordinal':
          r = p + 1
          break
        default:
          r = (i + 1 + j + 1) / 2
      }
      const row = rows[ranked[p]!]!
      out[row] = r
      setValid(nullBitmap, row, true)
    }
    i = j + 1
  }
  return nulls.length
}

function windowTable(
  table: TableView,
  name: string,
  fn: 'rank' | 'lag' | 'lead' | 'cumsum' | 'rowNumber',
  expr: ExprNode | undefined,
  offset: number,
  partitionBy: string[] | undefined,
  orderBy: Array<{ expr: ExprNode; descending: boolean }> | undefined,
  method: RankMethod = 'average',
): TableView {
  if (expr) ({ expr, table } = broadcastAggregates(expr, table))
  let working = table
  if (orderBy && orderBy.length) working = sortTable(table, orderBy)

  const partitions = new Map<string, number[]>()
  for (let i = 0; i < working.numRows; i++) {
    const key = (partitionBy ?? [])
      .map((n) => {
        const c = getColumn(working, n)
        return isValid(c.nullBitmap, i) ? String(getValue(c.data, i)) : '∅'
      })
      .join('\0')
    let arr = partitions.get(key)
    if (!arr) {
      arr = []
      partitions.set(key, arr)
    }
    arr.push(i)
  }

  const data = new Float64Array(working.numRows)
  const nullBitmap = new Uint8Array(Math.ceil(working.numRows / 8) || 1)
  let anyNull = false

  // Rank key: orderBy columns (partition already sorted), else `expr`, else row position.
  const rankKeyExprs = orderBy && orderBy.length ? orderBy.map((o) => o.expr) : expr ? [expr] : null
  const rankKeys = rankKeyExprs ? rankKeyExprs.map((e, k) => materializeExprColumn(working, e, `__rank_key_${k}`)) : null
  const rankPresorted = Boolean(orderBy && orderBy.length)

  for (const rows of partitions.values()) {
    if (fn === 'rank' && rankKeys) {
      if (rankPartition(rows, rankKeys, rankPresorted, method, data, nullBitmap, Boolean(orderBy?.[0]?.descending)) > 0) anyNull = true
      continue
    }
    let cum = 0
    for (let pos = 0; pos < rows.length; pos++) {
      const i = rows[pos]!
      if (fn === 'rowNumber' || fn === 'rank') {
        data[i] = pos + 1
        setValid(nullBitmap, i, true)
      } else if (fn === 'cumsum') {
        const v = expr ? evalExprScalar(expr, working, i) : 0
        cum += Number(v ?? 0)
        data[i] = cum
        setValid(nullBitmap, i, true)
      } else if (fn === 'lag' || fn === 'lead') {
        const srcPos = fn === 'lag' ? pos - offset : pos + offset
        if (srcPos < 0 || srcPos >= rows.length) {
          anyNull = true
        } else {
          const srcRow = rows[srcPos]!
          const v = expr ? evalExprScalar(expr, working, srcRow) : srcRow
          if (v === null) anyNull = true
          else {
            data[i] = Number(v)
            setValid(nullBitmap, i, true)
          }
        }
      }
    }
  }

  return stripTemps(
    tableFromColumns([
      ...working.columns,
      {
        field: { name, dtype: 'f64', nullable: anyNull },
        data,
        nullBitmap: anyNull ? nullBitmap : undefined,
      },
    ]),
  )
}

function rollingTable(table: TableView, name: string, column: string, window: number, agg: AggKind): TableView {
  getField(table.schema, column)
  const fast = tryFastRolling(table, name, column, window, agg)
  if (fast) return fast
  const data = new Float64Array(table.numRows)
  const nullBitmap = new Uint8Array(Math.ceil(table.numRows / 8) || 1)
  let anyNull = false
  for (let i = 0; i < table.numRows; i++) {
    const start = Math.max(0, i - window + 1)
    const rows = Array.from({ length: i - start + 1 }, (_, k) => start + k)
    const v = aggregateValues(agg, { type: 'col', name: column }, table, rows)
    if (v === null) anyNull = true
    else {
      data[i] = Number(v)
      setValid(nullBitmap, i, true)
    }
  }
  return tableFromColumns([
    ...table.columns,
    { field: { name, dtype: 'f64', nullable: anyNull }, data, nullBitmap: anyNull ? nullBitmap : undefined },
  ])
}

function pivotTable(
  table: TableView,
  index: string[],
  columns: string,
  values: string,
  agg: AggKind,
): TableView {
  const colsCol = getColumn(table, columns)
  const indexCols = index.map((name) => getColumn(table, name))

  const colValues = new Set<string>()
  for (let i = 0; i < table.numRows; i++) {
    if (isValid(colsCol.nullBitmap, i)) colValues.add(labelAt(colsCol, i))
  }
  const pivoted = [...colValues]
  const groups = new Map<string, number[]>()
  for (let i = 0; i < table.numRows; i++) {
    const key = indexCols.map((c) => (isValid(c.nullBitmap, i) ? labelAt(c, i) : '∅')).join('\0')
    let arr = groups.get(key)
    if (!arr) {
      arr = []
      groups.set(key, arr)
    }
    arr.push(i)
  }

  const outCols: Column[] = indexCols.map((src) => {
    const data = allocateData(src.field.dtype, groups.size)
    let row = 0
    for (const idxs of groups.values()) {
      setValue(data, row, getValue(src.data, idxs[0]!), src.field.dtype)
      row++
    }
    return { field: { ...src.field, nullable: false }, data, dictionary: src.dictionary }
  })

  for (const pv of pivoted) {
    const data = new Float64Array(groups.size)
    let row = 0
    for (const idxs of groups.values()) {
      const filtered = idxs.filter((i) => isValid(colsCol.nullBitmap, i) && labelAt(colsCol, i) === pv)
      const v = aggregateValues(agg, { type: 'col', name: values }, table, filtered)
      data[row++] = v === null ? NaN : Number(v)
    }
    outCols.push({ field: { name: pv, dtype: 'f64', nullable: true }, data })
  }
  return tableFromColumns(outCols)
}

export function executeCpu(plan: PlanNode): TableView {
  switch (plan.type) {
    case 'scan':
      return plan.table
    case 'project': {
      // A: push column pruning into filter gather
      if (plan.input.type === 'filter') {
        const names = projectColumnNames(plan.columns)
        if (names) {
          const base = executeCpu(plan.input.input)
          const filtered = filterTable(base, plan.input.predicate, names)
          // If projection is exactly those columns in order, skip re-project when possible
          return project(filtered, plan.columns)
        }
      }
      return project(executeCpu(plan.input), plan.columns)
    }
    case 'filter':
      return filterTable(executeCpu(plan.input), plan.predicate)
    case 'sort':
      return sortTable(executeCpu(plan.input), plan.by)
    case 'limit': {
      // Fuse sort + limit into top-k when possible
      if (plan.input.type === 'sort' && (plan.offset ?? 0) === 0) {
        const sorted = sortTable(executeCpu(plan.input.input), plan.input.by, plan.n)
        return sorted
      }
      const t = executeCpu(plan.input)
      const start = plan.offset ?? 0
      const end = Math.min(t.numRows, Math.max(0, start) + Math.max(0, plan.n))
      return sliceTable(t, Math.max(0, start), end)
    }
    case 'withColumn': {
      const t = executeCpu(plan.input)
      const col = materializeExprColumn(t, plan.expr, plan.name)
      const others = t.columns.filter((c) => c.field.name !== plan.name)
      return tableFromColumns([...others, col])
    }
    case 'withColumns': {
      const t = executeCpu(plan.input)
      return withColumnsTable(t, plan.columns, materializeExprColumn)
    }
    case 'drop': {
      const t = executeCpu(plan.input)
      return tableFromColumns(t.columns.filter((c) => !plan.columns.includes(c.field.name)))
    }
    case 'rename': {
      const t = executeCpu(plan.input)
      for (const from of Object.keys(plan.mapping)) getField(t.schema, from)
      const renamed = t.columns.map((c) => {
        const name = Object.hasOwn(plan.mapping, c.field.name) ? plan.mapping[c.field.name]! : c.field.name
        return name === c.field.name ? c : { ...c, field: { ...c.field, name } }
      })
      const seen = new Set<string>()
      for (const c of renamed) {
        if (seen.has(c.field.name)) throw new Error(`rename: duplicate column name "${c.field.name}"`)
        seen.add(c.field.name)
      }
      return tableFromColumns(renamed)
    }
    case 'groupBy': {
      // Fuse filter → groupBy (lazy collect style) — avoids materializing filtered rows.
      if (plan.input.type === 'filter') {
        const base = executeCpu(plan.input.input)
        const fused = tryFusedFilterGroupBy(base, plan.input.predicate, plan.keys, plan.aggs)
        if (fused) return fused
      }
      return groupByTable(executeCpu(plan.input), plan.keys, plan.aggs)
    }
    case 'join':
      return joinTables(executeCpu(plan.left), executeCpu(plan.right), plan.leftOn, plan.rightOn, plan.how)
    case 'fillNull':
      return fillNullTable(executeCpu(plan.input), plan.value, plan.columns)
    case 'dropNull':
      return dropNullTable(executeCpu(plan.input), plan.columns)
    case 'melt':
      return meltTable(executeCpu(plan.input), plan.idVars, plan.valueVars, plan.varName, plan.valueName)
    case 'pivot':
      return pivotTable(executeCpu(plan.input), plan.index, plan.columns, plan.values, plan.agg)
    case 'concat':
      return plan.how === 'vertical'
        ? concatVertical(plan.frames.map(executeCpu))
        : tableFromColumns(plan.frames.flatMap((f) => executeCpu(f).columns))
    case 'window':
      return windowTable(
        executeCpu(plan.input),
        plan.name,
        plan.fn,
        plan.expr,
        plan.offset ?? 1,
        plan.partitionBy,
        plan.orderBy,
        plan.method ?? 'average',
      )
    case 'rolling':
      return rollingTable(executeCpu(plan.input), plan.name, plan.column, plan.window, plan.agg)
    case 'expanding':
      return expandingTable(executeCpu(plan.input), plan.name, plan.column, plan.agg, aggregateValues)
    case 'slice': {
      const t = executeCpu(plan.input)
      const start = plan.start < 0 ? Math.max(0, t.numRows + plan.start) : plan.start
      const end = plan.end === undefined ? t.numRows : plan.end < 0 ? t.numRows + plan.end : plan.end
      return sliceTable(t, start, end)
    }
    case 'take': {
      const t = executeCpu(plan.input)
      for (const i of plan.indices) {
        if (!Number.isInteger(i) || i < 0 || i >= t.numRows) {
          throw new RangeError(`take: index ${i} out of range for ${t.numRows} rows`)
        }
      }
      return tableFromColumns(t.columns.map((c) => takeColumn(c, plan.indices)))
    }
    case 'sample':
      return sampleTable(executeCpu(plan.input), plan)
    case 'explode':
      return explodeTable(executeCpu(plan.input), plan.column)
    case 'unnest':
      return unnestTable(executeCpu(plan.input), plan.column, plan.separator)
    case 'transpose':
      return transposeTable(executeCpu(plan.input), plan.headerColumn)
    case 'interpolate':
      return interpolateTable(executeCpu(plan.input), plan.columns)
    case 'asofJoin':
      return asofJoinTables(
        executeCpu(plan.left),
        executeCpu(plan.right),
        plan.leftOn,
        plan.rightOn,
        plan.strategy,
      )
    case 'unique':
      return uniqueTable(executeCpu(plan.input), plan.columns, plan.keep)
    case 'valueCounts':
      return valueCounts(executeCpu(plan.input), plan.column, plan.normalize)
    case 'describe':
      return describeTable(executeCpu(plan.input), plan.quantileMethod)
    case 'corr':
      return corrTable(executeCpu(plan.input), plan.kind, plan.method, plan.columns)
  }
}

export class CpuBackend implements Backend {
  readonly name = 'cpu' as const
  readonly capabilities = { name: 'cpu' as const }

  supports(_plan: PlanNode): boolean {
    return true
  }

  async execute(plan: PlanNode): Promise<TableView> {
    // B: parallel dual-gt + typed gather (helpers no-op to sync below thresholds)
    if (plan.type === 'filter') {
      const input = executeCpu(plan.input)
      const dual = matchDualGtFilter(input, plan.predicate)
      if (dual) {
        const idx = await parallelDualGtIndices(dual.a, dual.b, dual.la, dual.lb)
        return parallelTakeTable(input, idx)
      }
      return filterTable(input, plan.predicate)
    }
    if (plan.type === 'project' && plan.input.type === 'filter') {
      const names = projectColumnNames(plan.columns)
      const input = executeCpu(plan.input.input)
      const dual = matchDualGtFilter(input, plan.input.predicate)
      if (names && dual) {
        const idx = await parallelDualGtIndices(dual.a, dual.b, dual.la, dual.lb)
        return project(await parallelTakeTable(input, idx, names), plan.columns)
      }
    }
    return executeCpu(plan)
  }
}
