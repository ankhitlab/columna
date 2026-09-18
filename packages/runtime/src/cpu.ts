import {
  allocateData,
  cloneColumn,
  estimateTableBytes,
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
import type { AggKind, Backend, CorrMethod, ExecContext, ExprNode, JoinKind, PlanNode, QuantileMethod, RankMethod } from './types.js'
import { parallelDualGtIndices, parallelTakeTable, parallelFilter, parallelSort, parallelGroupBy, parallelUnique, PARALLEL_FILTER_MIN_ROWS, PARALLEL_SORT_MIN_ROWS, PARALLEL_GROUPBY_MIN_ROWS, PARALLEL_UNIQUE_MIN_ROWS } from './parallel.js'
import { optimizePlan, joinOrderChanged, estimatePlanRows } from './optimize.js'
import { exprSome, forEachChildExpr, mapExprChildren } from './expr_walk.js'
import { encodeCompositeKey, type KeyPart } from './composite_key.js'
import { tryLoadNativeKernels, NATIVE_JOIN_MIN_ROWS } from './native_kernels.js'
import { orderRows, overAggregateNumeric, overCumulativeNumeric, partitionIds, type Partition } from './over.js'
import {
  applyMathOp,
  matchDualGtFilter,
  projectColumnNames,
  requiredInputColumns,
  isIdentityProjection,
  tryFastDescribe,
  argsortNumeric,
  countUniqueNumeric,
  quantilePos,
  quantileSelect,
  tryFastExprColumn,
  tryFastFilter,
  tryFastGroupBy,
  tryFastJoin,
  lastFastJoinKernel,
  tryFastRolling,
  tryFastSort,
  tryFastUnique,
  tryFusedFilterGroupBy,
  tryFusedFilterUnique,
  tryFusedFilterSortLimit,
  tryChunkedGroupBy,
  gather,
} from './fast.js'
import { memoryBudget, recordLiveBytes } from './memory.js'
import { externalSortTable, joinTablesSpilled, needsSpill, uniqueTableSpilled } from './spill_ops.js'
import { spillRead, spillUnlink, spillWrite } from './spill.js'
import { createFilterView, ensureMaterialized } from './views.js'
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
    case 'i32': {
      const n = Number(v)
      if (!Number.isInteger(n) || n < -2147483648 || n > 2147483647) {
        throw new RangeError(`Cannot cast ${String(v)} to i32 (need integer in Int32 range)`)
      }
      return n
    }
    case 'u32': {
      const n = Number(v)
      if (!Number.isInteger(n) || n < 0 || n > 4294967295) {
        throw new RangeError(`Cannot cast ${String(v)} to u32 (need integer in Uint32 range)`)
      }
      return n
    }
    case 'f64':
    case 'f32':
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
  return exprSome(expr, (node) => node.type === 'agg' || node.type === 'over')
}

/** Every agg op requested on each inner expression in the tree — lets one pass serve mean + std + min … */
function collectAggNeeds(expr: ExprNode, needs: Map<string, Set<AggKind>>): void {
  if (expr.type === 'over') return // per-partition aggregates are computed separately (see rewriteOver)
  if (expr.type === 'agg') {
    const key = aggKey(expr.expr)
    if (key !== null) {
      let set = needs.get(key)
      if (!set) needs.set(key, (set = new Set()))
      set.add(expr.op)
    }
  }
  forEachChildExpr(expr, (child) => collectAggNeeds(child, needs))
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
    default:
      return mapExprChildren(expr, rec)
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
  table = ensureMaterialized(table)
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
  table = ensureMaterialized(table)
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
  // Defer gather: share column buffers until sort/join/unique/mutate materializes.
  return createFilterView(table, Uint32Array.from(indices), keep)
}

function sortTable(
  table: TableView,
  by: Array<{ expr: ExprNode; descending: boolean; nullsLast?: boolean }>,
  limit?: number,
): TableView {
  table = ensureMaterialized(table)
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
  if (needsSpill(table)) {
    return externalSortTable(table, by, limit, sortTableInMemory, evalExprScalar)
  }
  return sortTableInMemory(table, by, limit)
}

function sortTableInMemory(
  table: TableView,
  by: Array<{ expr: ExprNode; descending: boolean; nullsLast?: boolean }>,
  limit?: number,
): TableView {
  const fast = tryFastSort(table, by, limit)
  if (fast) return fast
  const n = table.numRows
  const indices = new Uint32Array(n)
  for (let i = 0; i < n; i++) indices[i] = i

  // Prefetch column keys for bare col exprs to avoid evalExprScalar per comparison.
  type KeyFn = (row: number) => number | string | boolean | null
  const keyFns: Array<{ at: KeyFn; descending: boolean; nullsLast: boolean }> = by.map((key) => {
    const nullsLast = key.nullsLast !== false
    if (key.expr.type === 'col') {
      const col = getColumn(table, key.expr.name)
      const dtype = col.field.dtype
      const at: KeyFn = (row) => {
        if (!isValid(col.nullBitmap, row)) return null
        const v = getValue(col.data, row)
        if (dtype === 'category' && col.dictionary) return col.dictionary[Number(v)] ?? null
        return v as number | string | boolean
      }
      return { at, descending: key.descending, nullsLast }
    }
    return {
      at: (row) => evalExprScalar(key.expr, table, row),
      descending: key.descending,
      nullsLast,
    }
  })

  indices.sort((a, b) => {
    for (const key of keyFns) {
      const va = key.at(a)
      const vb = key.at(b)
      if (va === vb) continue
      if (va === null) return key.nullsLast ? 1 : -1
      if (vb === null) return key.nullsLast ? -1 : 1
      const cmp = va < vb ? -1 : 1
      return key.descending ? -cmp : cmp
    }
    return 0
  })
  const sliced = limit !== undefined ? indices.subarray(0, Math.min(limit, n)) : indices
  return gather(table, sliced)
}

function groupByTable(
  table: TableView,
  keys: string[],
  aggs: Array<{ name: string; expr: ExprNode }>,
): TableView {
  table = ensureMaterialized(table)
  if (keys.length === 0) throw new Error('groupBy requires at least one key')
  if (needsSpill(table)) {
    const budget = memoryBudget() ?? 1
    const chunkRows = Math.max(1, Math.floor(table.numRows / Math.max(2, Math.ceil(estimateTableBytes(table) / budget))))
    const chunked = tryChunkedGroupBy(table, keys, aggs, chunkRows)
    if (chunked) return chunked
  }
  const fast = tryFastGroupBy(table, keys, aggs)
  if (fast) return fast
  return groupByTableSlow(table, keys, aggs)
}

function groupByTableSlow(
  table: TableView,
  keys: string[],
  aggs: Array<{ name: string; expr: ExprNode }>,
): TableView {
  const groups = new Map<string, number[]>()
  for (let i = 0; i < table.numRows; i++) {
    const parts: KeyPart[] = keys.map((k) => {
      const col = getColumn(table, k)
      if (!isValid(col.nullBitmap, i)) return null
      const v = getValue(col.data, i)
      if (col.field.dtype === 'category' && col.dictionary) return col.dictionary[Number(v)] ?? null
      if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') return v
      return String(v)
    })
    const key = encodeCompositeKey(parts)
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
      if (v === null) {
        anyNull = true
      } else {
        setValid(nullBitmap, i, true)
        setValue(data, i, v, dtype)
      }
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
  return encodeCompositeKey(
    cols.map((name) => {
      const col = getColumn(table, name)
      if (!isValid(col.nullBitmap, row)) return null
      const v = getValue(col.data, row)
      // category keys must compare by their string, never by the per-frame dictionary code
      if (col.field.dtype === 'category' && col.dictionary) return col.dictionary[Number(v)] ?? null
      if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') return v
      return String(v)
    }),
  )
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

/** Resolve output name for a join side column when the other side shares the name. */
function joinSideName(
  name: string,
  side: 'left' | 'right',
  otherHasName: boolean,
  sharedKey: boolean,
  lSuffix: string,
  rSuffix: string,
): string {
  if (sharedKey || !otherHasName) return name
  if (side === 'left') return lSuffix ? `${name}${lSuffix}` : name
  return `${name}${rSuffix}`
}

function crossJoinTables(
  left: TableView,
  right: TableView,
  lSuffix = '',
  rSuffix = '_right',
): TableView {
  const ln = left.numRows
  const rn = right.numRows
  const leftNames = new Set(left.schema.map((f) => f.name))
  const rightNames = new Set(right.schema.map((f) => f.name))
  if (ln === 0 || rn === 0) {
    return tableFromColumns([
      ...left.columns.map((c) => {
        const name = joinSideName(c.field.name, 'left', rightNames.has(c.field.name), false, lSuffix, rSuffix)
        const taken = takeColumn(c, [])
        return name === c.field.name ? taken : { ...taken, field: { ...taken.field, name } }
      }),
      ...right.columns.map((c) => {
        const name = joinSideName(c.field.name, 'right', leftNames.has(c.field.name), false, lSuffix, rSuffix)
        return { ...takeColumn(c, []), field: { ...c.field, name } }
      }),
    ])
  }
  const out: Column[] = []
  for (const col of left.columns) {
    const name = joinSideName(col.field.name, 'left', rightNames.has(col.field.name), false, lSuffix, rSuffix)
    const expanded = expandCrossColumn(col, 'repeat', ln, rn)
    out.push(name === col.field.name ? expanded : { ...expanded, field: { ...expanded.field, name } })
  }
  for (const col of right.columns) {
    const name = joinSideName(col.field.name, 'right', leftNames.has(col.field.name), false, lSuffix, rSuffix)
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
  lSuffix = '',
  rSuffix = '_right',
  validate?: '1:1' | '1:m' | 'm:1',
  keep?: readonly string[],
): TableView {
  left = ensureMaterialized(left)
  right = ensureMaterialized(right)
  if (how === 'cross') return crossJoinTables(left, right, lSuffix, rSuffix)

  if (validate) assertJoinValidate(left, right, leftOn, rightOn, validate)

  const working = estimateTableBytes(left) + estimateTableBytes(right)
  recordLiveBytes(working)
  if (needsSpill(left) || needsSpill(right) || (memoryBudgetSafe() && working * 2 > memoryBudgetSafe()!)) {
    if (how === 'inner') {
      return joinTablesSpilled(left, right, (l, r) =>
        joinTablesInMemory(l, r, leftOn, rightOn, how, lSuffix, rSuffix, keep),
      )
    }
    // Non-inner: spill the larger side to account for budget, then join in memory after reload.
    const spillTarget = estimateTableBytes(right) >= estimateTableBytes(left) ? right : left
    const path = spillWrite(spillTarget)
    try {
      const reloaded = spillRead(path)
      if (spillTarget === right) return joinTablesInMemory(left, reloaded, leftOn, rightOn, how, lSuffix, rSuffix, keep)
      return joinTablesInMemory(reloaded, right, leftOn, rightOn, how, lSuffix, rSuffix, keep)
    } finally {
      spillUnlink(path)
    }
  }
  return joinTablesInMemory(left, right, leftOn, rightOn, how, lSuffix, rSuffix, keep)
}

function assertJoinValidate(
  left: TableView,
  right: TableView,
  leftOn: string[],
  rightOn: string[],
  validate: '1:1' | '1:m' | 'm:1',
): void {
  const needLeft = validate === '1:1' || validate === '1:m'
  const needRight = validate === '1:1' || validate === 'm:1'
  if (needLeft && !joinKeysUnique(left, leftOn)) {
    throw new Error(`join validate '${validate}': left keys are not unique`)
  }
  if (needRight && !joinKeysUnique(right, rightOn)) {
    throw new Error(`join validate '${validate}': right keys are not unique`)
  }
}

function joinKeysUnique(table: TableView, cols: string[]): boolean {
  const seen = new Set<string>()
  for (let i = 0; i < table.numRows; i++) {
    const k = joinKey(table, cols, i)
    if (seen.has(k)) return false
    seen.add(k)
  }
  return true
}

function memoryBudgetSafe(): number | undefined {
  return memoryBudget()
}

function joinTablesInMemory(
  left: TableView,
  right: TableView,
  leftOn: string[],
  rightOn: string[],
  how: JoinKind,
  lSuffix = '',
  rSuffix = '_right',
  keep?: readonly string[],
): TableView {
  if (how === 'cross') return crossJoinTables(left, right, lSuffix, rSuffix)

  if (how === 'inner' || how === 'left' || how === 'right' || how === 'outer' || how === 'semi' || how === 'anti') {
    const fast = tryFastJoin(left, right, leftOn, rightOn, how, lSuffix, rSuffix, keep)
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
    const taken = takeTable(left, indices)
    return keep ? project(taken, [...keep]) : taken
  }

  const leftIdx: number[] = []
  const rightIdx: Array<number | null> = []
  const matchedRight = new Set<number>()

  for (let i = 0; i < left.numRows; i++) {
    const k = joinKey(left, leftOn, i)
    const hits = rightIndex.get(k)
    if (hits && hits.length) {
      for (const j of hits) {
        leftIdx.push(i)
        rightIdx.push(j)
        matchedRight.add(j)
      }
    } else if (how === 'left' || how === 'outer') {
      leftIdx.push(i)
      rightIdx.push(null)
    }
  }

  if (how === 'right' || how === 'outer') {
    for (let j = 0; j < right.numRows; j++) {
      if (!matchedRight.has(j)) {
        leftIdx.push(-1)
        rightIdx.push(j)
      }
    }
  }

  return assembleJoin(left, right, leftIdx, rightIdx, leftOn, rightOn, lSuffix, rSuffix, keep)
}

function assembleJoin(
  left: TableView,
  right: TableView,
  leftIdx: number[],
  rightIdx: Array<number | null>,
  leftOn: string[],
  rightOn: string[],
  lSuffix = '',
  rSuffix = '_right',
  keep?: readonly string[],
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
  const rightNames = new Set(right.schema.map((f) => f.name))
  for (const col of left.columns) {
    const keyIdx = leftOn.indexOf(col.field.name)
    const sharedKey = keyIdx >= 0 && rightOn.includes(col.field.name)
    const name = joinSideName(col.field.name, 'left', rightNames.has(col.field.name), sharedKey, lSuffix, rSuffix)
    const cloned = takeColumn(col, leftGather)
    let renamed = name === col.field.name ? cloned : { ...cloned, field: { ...cloned.field, name } }
    if (!leftHasSentinel) {
      outCols.push(renamed)
      continue
    }
    let anyNull = Boolean(renamed.nullBitmap)
    const nullBitmap = renamed.nullBitmap
      ? new Uint8Array(renamed.nullBitmap)
      : new Uint8Array(Math.ceil(leftIdx.length / 8) || 1)
    if (!renamed.nullBitmap) nullBitmap.fill(0xff)
    const data = sharedKey ? copyTyped(renamed.data) : renamed.data
    const rKeyCol = sharedKey ? getColumn(right, rightOn[keyIdx]!) : null
    for (let i = 0; i < leftIdx.length; i++) {
      if (leftIdx[i]! >= 0) continue
      if (sharedKey && rKeyCol) {
        const rj = rightIdx[i]
        if (rj !== null && isValid(rKeyCol.nullBitmap, rj)) {
          if (rKeyCol.field.dtype === 'category' && rKeyCol.dictionary) {
            const code = Number(getValue(rKeyCol.data, rj))
            const label = String(rKeyCol.dictionary[code] ?? '')
            let dict = renamed.dictionary ? [...renamed.dictionary] : []
            let leftCode = dict.indexOf(label)
            if (leftCode < 0) {
              leftCode = dict.length
              dict = [...dict, label]
            }
            ;(data as Uint32Array)[i] = leftCode
            renamed = { ...renamed, dictionary: dict, data }
          } else {
            const v = getValue(rKeyCol.data, rj)
            setValue(data, i, v as number | string | boolean, renamed.field.dtype)
          }
          setValid(nullBitmap, i, true)
          continue
        }
      }
      setValid(nullBitmap, i, false)
      anyNull = true
    }
    outCols.push({
      ...renamed,
      data,
      nullBitmap: anyNull ? nullBitmap : undefined,
      field: { ...renamed.field, nullable: anyNull || renamed.field.nullable },
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
    const sharedKey = rightOn.includes(col.field.name) && leftOn.includes(col.field.name)
    if (sharedKey) continue
    const name = joinSideName(col.field.name, 'right', leftNames.has(col.field.name), false, lSuffix, rSuffix)
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

  if (keep) {
    const byName = new Map(outCols.map((c) => [c.field.name, c]))
    const ordered: Column[] = []
    for (const name of keep) {
      const c = byName.get(name)
      if (c) ordered.push(c)
    }
    return tableFromColumns(ordered)
  }
  return tableFromColumns(outCols)
}

function fillNullTable(
  table: TableView,
  value?: number | string | boolean,
  columns?: string[],
  values?: Record<string, number | string | boolean>,
): TableView {
  const fillOne = (col: Column, fill: number | string | boolean): Column => {
    if (!col.nullBitmap) return col
    if (col.field.dtype === 'category') {
      const dict = col.dictionary ? [...col.dictionary] : []
      let code = dict.indexOf(String(fill))
      if (code < 0) {
        code = dict.length
        dict.push(String(fill))
      }
      const data = copyTyped(col.data) as Uint32Array
      for (let i = 0; i < table.numRows; i++) {
        if (!isValid(col.nullBitmap, i)) data[i] = code
      }
      return { field: { ...col.field, nullable: false }, data, dictionary: dict }
    }
    const data = copyTyped(col.data)
    for (let i = 0; i < table.numRows; i++) {
      if (!isValid(col.nullBitmap, i)) setValue(data, i, fill, col.field.dtype)
    }
    return {
      field: { ...col.field, nullable: false },
      data,
      dictionary: col.dictionary ? [...col.dictionary] : undefined,
    }
  }

  if (values) {
    return tableFromColumns(
      table.columns.map((col) => (col.field.name in values ? fillOne(col, values[col.field.name]!) : col)),
    )
  }
  const names = columns ?? table.schema.map((f) => f.name)
  const fill = value!
  return tableFromColumns(table.columns.map((col) => (names.includes(col.field.name) ? fillOne(col, fill) : col)))
}

/** Copy a cell without dtype decoding (works for category codes). */
function copyCell(data: Column['data'], from: number, to: number): void {
  if (Array.isArray(data)) {
    ;(data as unknown[])[to] = (data as unknown[])[from]
    return
  }
  ;(data as Float64Array)[to] = (data as Float64Array)[from]!
}

function ffillTable(table: TableView, columns?: string[]): TableView {
  const names = columns ?? table.schema.map((f) => f.name)
  return tableFromColumns(
    table.columns.map((col) => {
      if (!names.includes(col.field.name) || !col.nullBitmap) return col
      const data = copyTyped(col.data)
      const nullBitmap = new Uint8Array(col.nullBitmap)
      let last = -1
      let anyNull = false
      for (let i = 0; i < table.numRows; i++) {
        if (isValid(col.nullBitmap, i)) {
          last = i
          setValid(nullBitmap, i, true)
        } else if (last >= 0) {
          copyCell(data, last, i)
          setValid(nullBitmap, i, true)
        } else {
          anyNull = true
          setValid(nullBitmap, i, false)
        }
      }
      return {
        field: { ...col.field, nullable: anyNull },
        data,
        nullBitmap: anyNull ? nullBitmap : undefined,
        dictionary: col.dictionary ? [...col.dictionary] : undefined,
      }
    }),
  )
}

function bfillTable(table: TableView, columns?: string[]): TableView {
  const names = columns ?? table.schema.map((f) => f.name)
  return tableFromColumns(
    table.columns.map((col) => {
      if (!names.includes(col.field.name) || !col.nullBitmap) return col
      const data = copyTyped(col.data)
      const nullBitmap = new Uint8Array(col.nullBitmap)
      let next = -1
      let anyNull = false
      for (let i = table.numRows - 1; i >= 0; i--) {
        if (isValid(col.nullBitmap, i)) {
          next = i
          setValid(nullBitmap, i, true)
        } else if (next >= 0) {
          copyCell(data, next, i)
          setValid(nullBitmap, i, true)
        } else {
          anyNull = true
          setValid(nullBitmap, i, false)
        }
      }
      return {
        field: { ...col.field, nullable: anyNull },
        data,
        nullBitmap: anyNull ? nullBitmap : undefined,
        dictionary: col.dictionary ? [...col.dictionary] : undefined,
      }
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
  table = ensureMaterialized(table)
  if (needsSpill(table)) {
    return uniqueTableSpilled(table, (chunk) => uniqueTableInMemory(chunk, columns, keep))
  }
  return uniqueTableInMemory(table, columns, keep)
}

function uniqueTableInMemory(table: TableView, columns: string[] | undefined, keep: 'first' | 'last' | 'none'): TableView {
  const fast = tryFastUnique(table, columns, keep)
  if (fast) return fast
  const cols = columns ?? table.schema.map((f) => f.name)
  const seen = new Map<string, number>()
  const order: string[] = []
  for (let i = 0; i < table.numRows; i++) {
    const key = encodeCompositeKey(
      cols.map((n) => {
        const c = getColumn(table, n)
        if (!isValid(c.nullBitmap, i)) return null
        const v = getValue(c.data, i)
        if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') return v
        return String(v)
      }),
    )
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
    const key = encodeCompositeKey(
      (partitionBy ?? []).map((n) => {
        const c = getColumn(working, n)
        if (!isValid(c.nullBitmap, i)) return null
        const v = getValue(c.data, i)
        if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') return v
        return String(v)
      }),
    )
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
    const key = encodeCompositeKey(
      indexCols.map((c) => (isValid(c.nullBitmap, i) ? labelAt(c, i) : null)),
    )
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
  // Ensure direct executeCpu callers (sync DataFrame helpers / tests) get the same rewrites as Runtime.
  plan = optimizePlan(plan)
  return ensureMaterialized(executeCpuNode(plan))
}

/**
 * Execute a plan without logical rewrites.
 * Used by optimizer differential tests: results must match `executeCpuUnoptimized(optimizePlan(plan))`.
 */
export function executeCpuUnoptimized(plan: PlanNode): TableView {
  return ensureMaterialized(executeCpuNode(plan))
}

function executeCpuNode(plan: PlanNode): TableView {
  switch (plan.type) {
    case 'scan':
      return plan.table
    case 'project': {
      // A: push column pruning into filter gather
      if (plan.input.type === 'filter') {
        const names = projectColumnNames(plan.columns)
        if (names) {
          const base = executeCpuNode(plan.input.input)
          const filtered = filterTable(base, plan.input.predicate, names)
          // If projection is exactly those columns in order, skip re-project when possible
          return project(ensureMaterialized(filtered), plan.columns)
        }
      }
      // Project after join: only materialize requested output columns.
      if (plan.input.type === 'join' && plan.input.how !== 'cross') {
        const names = requiredInputColumns(plan.columns)
        if (names) {
          const joined = joinTables(
            executeCpuNode(plan.input.left),
            executeCpuNode(plan.input.right),
            plan.input.leftOn,
            plan.input.rightOn,
            plan.input.how,
            plan.input.lSuffix ?? '',
            plan.input.rSuffix ?? '_right',
            plan.input.validate,
            names,
          )
          // Skip final project only for identity column picks (no alias rename).
          if (
            isIdentityProjection(plan.columns) &&
            joined.schema.length === names.length &&
            names.every((n, i) => joined.schema[i]!.name === n)
          ) {
            return joined
          }
          return project(ensureMaterialized(joined), plan.columns)
        }
      }
      return project(ensureMaterialized(executeCpuNode(plan.input)), plan.columns)
    }
    case 'filter':
      return filterTable(executeCpuNode(plan.input), plan.predicate)
    case 'sort':
      return sortTable(executeCpuNode(plan.input), plan.by)
    case 'limit': {
      // Fuse filter → sort → limit into filtered top-k when possible
      if (plan.input.type === 'sort' && (plan.offset ?? 0) === 0) {
        if (plan.input.input.type === 'filter') {
          const base = executeCpuNode(plan.input.input.input)
          const fused = tryFusedFilterSortLimit(
            base,
            plan.input.input.predicate,
            plan.input.by,
            plan.n,
          )
          if (fused) return fused
          const filtered = filterTable(base, plan.input.input.predicate)
          return sortTable(filtered, plan.input.by, plan.n)
        }
        return sortTable(executeCpuNode(plan.input.input), plan.input.by, plan.n)
      }
      const t = ensureMaterialized(executeCpuNode(plan.input))
      const start = plan.offset ?? 0
      const end = Math.min(t.numRows, Math.max(0, start) + Math.max(0, plan.n))
      return sliceTable(t, Math.max(0, start), end)
    }
    case 'withColumn': {
      const t = ensureMaterialized(executeCpuNode(plan.input))
      const col = materializeExprColumn(t, plan.expr, plan.name)
      const others = t.columns.filter((c) => c.field.name !== plan.name)
      return tableFromColumns([...others, col])
    }
    case 'withColumns': {
      const t = ensureMaterialized(executeCpuNode(plan.input))
      return withColumnsTable(t, plan.columns, materializeExprColumn)
    }
    case 'drop': {
      const t = ensureMaterialized(executeCpuNode(plan.input))
      return tableFromColumns(t.columns.filter((c) => !plan.columns.includes(c.field.name)))
    }
    case 'rename': {
      const t = ensureMaterialized(executeCpuNode(plan.input))
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
        const base = executeCpuNode(plan.input.input)
        const fused = tryFusedFilterGroupBy(base, plan.input.predicate, plan.keys, plan.aggs)
        if (fused) return fused
      }
      return groupByTable(executeCpuNode(plan.input), plan.keys, plan.aggs)
    }
    case 'join':
      return joinTables(
        executeCpuNode(plan.left),
        executeCpuNode(plan.right),
        plan.leftOn,
        plan.rightOn,
        plan.how,
        plan.lSuffix ?? '',
        plan.rSuffix ?? '_right',
        plan.validate,
      )
    case 'fillNull':
      return fillNullTable(executeCpuNode(plan.input), plan.value, plan.columns, plan.values)
    case 'ffill':
      return ffillTable(executeCpuNode(plan.input), plan.columns)
    case 'bfill':
      return bfillTable(executeCpuNode(plan.input), plan.columns)
    case 'dropNull':
      return dropNullTable(executeCpuNode(plan.input), plan.columns)
    case 'melt':
      return meltTable(executeCpuNode(plan.input), plan.idVars, plan.valueVars, plan.varName, plan.valueName)
    case 'pivot':
      return pivotTable(executeCpuNode(plan.input), plan.index, plan.columns, plan.values, plan.agg)
    case 'concat':
      return plan.how === 'vertical'
        ? concatVertical(plan.frames.map(executeCpu))
        : tableFromColumns(plan.frames.flatMap((f) => executeCpuNode(f).columns))
    case 'window':
      return windowTable(
        executeCpuNode(plan.input),
        plan.name,
        plan.fn,
        plan.expr,
        plan.offset ?? 1,
        plan.partitionBy,
        plan.orderBy,
        plan.method ?? 'average',
      )
    case 'rolling':
      return rollingTable(executeCpuNode(plan.input), plan.name, plan.column, plan.window, plan.agg)
    case 'expanding':
      return expandingTable(executeCpuNode(plan.input), plan.name, plan.column, plan.agg, aggregateValues)
    case 'slice': {
      const t = executeCpuNode(plan.input)
      const start = plan.start < 0 ? Math.max(0, t.numRows + plan.start) : plan.start
      const end = plan.end === undefined ? t.numRows : plan.end < 0 ? t.numRows + plan.end : plan.end
      return sliceTable(t, start, end)
    }
    case 'take': {
      const t = executeCpuNode(plan.input)
      for (const i of plan.indices) {
        if (!Number.isInteger(i) || i < 0 || i >= t.numRows) {
          throw new RangeError(`take: index ${i} out of range for ${t.numRows} rows`)
        }
      }
      return tableFromColumns(t.columns.map((c) => takeColumn(c, plan.indices)))
    }
    case 'sample':
      return sampleTable(executeCpuNode(plan.input), plan)
    case 'explode':
      return explodeTable(executeCpuNode(plan.input), plan.column)
    case 'unnest':
      return unnestTable(executeCpuNode(plan.input), plan.column, plan.separator)
    case 'transpose':
      return transposeTable(executeCpuNode(plan.input), plan.headerColumn)
    case 'interpolate':
      return interpolateTable(executeCpuNode(plan.input), plan.columns)
    case 'asofJoin':
      return asofJoinTables(
        executeCpuNode(plan.left),
        executeCpuNode(plan.right),
        plan.leftOn,
        plan.rightOn,
        plan.strategy,
      )
    case 'unique':
      if (plan.input.type === 'filter') {
        const base = executeCpuNode(plan.input.input)
        const fused = tryFusedFilterUnique(base, plan.input.predicate, plan.columns, plan.keep)
        if (fused) return fused
      }
      return uniqueTable(executeCpuNode(plan.input), plan.columns, plan.keep)
    case 'valueCounts':
      return valueCounts(executeCpuNode(plan.input), plan.column, plan.normalize)
    case 'describe':
      return describeTable(executeCpuNode(plan.input), plan.quantileMethod)
    case 'corr':
      return corrTable(executeCpuNode(plan.input), plan.kind, plan.method, plan.columns)
  }
}

// --- Protocol v2: parallel sort/groupBy/unique interception (numeric, SAB-backed) ---

type ArrKind = 'Float64Array' | 'Float32Array' | 'Int32Array' | 'Uint32Array' | 'Uint8Array'

function dtypeToKind(dtype: DType): ArrKind | null {
  switch (dtype) {
    case 'f64': case 'datetime': return 'Float64Array'
    case 'f32': return 'Float32Array'
    case 'i32': return 'Int32Array'
    case 'u32': return 'Uint32Array'
    case 'bool': return 'Uint8Array'
    default: return null
  }
}

function copyToShared(src: ArrayLike<number>, kind: ArrKind): SharedArrayBuffer {
  const buf = new SharedArrayBuffer(src.length * bytesPerElem(kind))
  const view = viewTyped(kind, buf)
  for (let i = 0; i < src.length; i++) view[i] = src[i]!
  return buf
}

function bytesPerElem(kind: ArrKind): number {
  switch (kind) {
    case 'Float64Array': return 8
    case 'Float32Array': return 4
    case 'Int32Array': return 4
    case 'Uint32Array': return 4
    case 'Uint8Array': return 1
  }
}

function viewTyped(kind: ArrKind, buf: SharedArrayBuffer): ArrayLike<number> & { [i: number]: number } {
  switch (kind) {
    case 'Float64Array': return new Float64Array(buf)
    case 'Float32Array': return new Float32Array(buf)
    case 'Int32Array': return new Int32Array(buf)
    case 'Uint32Array': return new Uint32Array(buf)
    case 'Uint8Array': return new Uint8Array(buf)
  }
}

/** Try parallel sort when input is large and all keys are bare numeric (non-dict) columns. */
async function tryParallelSort(plan: Extract<PlanNode, { type: 'sort' }>): Promise<TableView | null> {
  const input = ensureMaterialized(executeCpuNode(plan.input))
  const n = input.numRows
  if (n < PARALLEL_SORT_MIN_ROWS) return null
  const specs: Array<{ buffer: SharedArrayBuffer; kind: ArrKind; length: number; descending: boolean; nullsLast: boolean; nullBitmap: SharedArrayBuffer | null }> = []
  for (const k of plan.by) {
    if (k.expr.type !== 'col') return null
    const col = getColumn(input, k.expr.name)
    if (col.dictionary) return null
    const kind = dtypeToKind(col.field.dtype)
    if (!kind) return null
    const data = col.data as ArrayLike<number>
    if (!(data instanceof Float64Array || data instanceof Float32Array || data instanceof Int32Array || data instanceof Uint32Array || data instanceof Uint8Array)) return null
    const nullsLast = k.nullsLast !== false
    let nullBitmap: SharedArrayBuffer | null = null
    if (col.nullBitmap) {
      nullBitmap = new SharedArrayBuffer(n)
      const nb = new Uint8Array(nullBitmap)
      for (let i = 0; i < n; i++) nb[i] = isValid(col.nullBitmap, i) ? 1 : 0
    }
    specs.push({
      buffer: copyToShared(data, kind),
      kind, length: n,
      descending: k.descending,
      nullsLast,
      nullBitmap,
    })
  }
  const idx = await parallelSort(specs, n)
  return gather(input, idx)
}

/** Try parallel unique when input is large and all subset columns are bare numeric (non-dict). */
async function tryParallelUnique(plan: Extract<PlanNode, { type: 'unique' }>): Promise<TableView | null> {
  const input = ensureMaterialized(executeCpuNode(plan.input))
  const n = input.numRows
  if (n < PARALLEL_UNIQUE_MIN_ROWS) return null
  if (plan.keep !== 'first') return null
  const colNames = plan.columns ?? input.schema.map((f) => f.name)
  const cols: ArrayLike<number>[] = []
  for (const name of colNames) {
    const col = getColumn(input, name)
    if (col.dictionary) return null
    const kind = dtypeToKind(col.field.dtype)
    if (!kind) return null
    const data = col.data as ArrayLike<number>
    if (!(data instanceof Float64Array || data instanceof Float32Array || data instanceof Int32Array || data instanceof Uint32Array || data instanceof Uint8Array)) return null
    cols.push(data)
  }
  const idx = await parallelUnique(cols as unknown as (Float64Array | Float32Array | Int32Array | Uint32Array)[], n)
  if (!idx) return null
  return gather(input, idx)
}

export class CpuBackend implements Backend {
  readonly name = 'cpu' as const
  readonly capabilities = { name: 'cpu' as const }

  supports(_plan: PlanNode): boolean {
    return true
  }

  async execute(plan: PlanNode, ctx?: ExecContext): Promise<TableView> {
    const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now()
    const raw = plan
    plan = optimizePlan(plan)
    if (ctx && joinOrderChanged(raw, plan)) {
      ctx.trace({
        node: 'join',
        backend: 'cpu',
        kernel: 'optimized:joinReorder',
        reason: 'inner join build/order changed by optimizePlan',
      })
    }
    const done = (table: TableView, kernel?: string, reason?: string): TableView => {
      ctx?.trace({
        node: plan.type,
        backend: 'cpu',
        kernel,
        ms: (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0,
        rows: table.numRows,
        reason:
          reason ??
          (ctx.requested !== 'cpu' && ctx.requested !== 'auto' ? 'plan executed by the CPU engine' : undefined),
      })
      return table
    }
    // B: parallel dual-gt + typed gather (helpers no-op to sync below thresholds)
    if (plan.type === 'filter') {
      const input = executeCpuNode(plan.input)
      const dual = matchDualGtFilter(input, plan.predicate)
      if (dual) {
        await tryLoadNativeKernels()
        const { indices, kernel } = await parallelDualGtIndices(dual.a, dual.b, dual.la, dual.lb)
        return done(await parallelTakeTable(input, indices), kernel)
      }
      return done(ensureMaterialized(filterTable(input, plan.predicate)))
    }
    if (plan.type === 'project' && plan.input.type === 'filter') {
      const names = projectColumnNames(plan.columns)
      const input = executeCpuNode(plan.input.input)
      const dual = matchDualGtFilter(input, plan.input.predicate)
      if (names && dual) {
        await tryLoadNativeKernels()
        const { indices, kernel } = await parallelDualGtIndices(dual.a, dual.b, dual.la, dual.lb)
        return done(project(await parallelTakeTable(input, indices, names), plan.columns), kernel)
      }
    }
    // Protocol v2: parallel sort / unique for huge numeric inputs
    if (plan.type === 'sort') {
      const par = await tryParallelSort(plan)
      if (par) return done(par, 'workers:sort', `parallel sort when rows≥${PARALLEL_SORT_MIN_ROWS}`)
    }
    if (plan.type === 'unique') {
      const par = await tryParallelUnique(plan)
      if (par) return done(par, 'workers:unique', `parallel unique when rows≥${PARALLEL_UNIQUE_MIN_ROWS}`)
    }
    // Join (or project→join keep): surface fast-join kernel + native threshold note
    const joinRoot =
      plan.type === 'join'
        ? plan
        : plan.type === 'project' && plan.input.type === 'join'
          ? plan.input
          : null
    if (joinRoot) {
      const table = ensureMaterialized(executeCpuNode(plan))
      const info = lastFastJoinKernel
      const rightEst = estimatePlanRows(joinRoot.right)
      const reason =
        info?.reason ??
        (rightEst > 0
          ? `build(right)≈${rightEst}; native probe when left≥${NATIVE_JOIN_MIN_ROWS}`
          : undefined)
      return done(table, info?.kernel ?? 'js:join', reason)
    }
    return done(executeCpu(plan))
  }
}
