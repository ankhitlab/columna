import type { DType, TableView } from '@columna/arrow'
import { estimatePlanRows } from './stats.js'

export type EngineKind = 'webgpu' | 'wasm' | 'cpu' | 'auto'

/**
 * Sample-quantile definition. 'linear' — Hyndman–Fan type 7, position (n − 1)·q (pandas / numpy / polars default).
 * 'minitab' — type 6, position q·(n + 1) clamped to the sample (Minitab, SPSS, R type = 6; numpy method='weibull').
 */
export type QuantileMethod = 'linear' | 'minitab'

export type AggKind =
  | 'sum'
  | 'mean'
  | 'min'
  | 'max'
  | 'count'
  | 'nunique'
  | 'first'
  | 'last'
  | 'std'
  | 'var'
  | 'median'
  | 'quantile'

export type JoinKind = 'inner' | 'left' | 'right' | 'outer' | 'cross' | 'semi' | 'anti'

/** Element-wise math on numeric expressions; null in → null out, domain errors → NaN. */
export type MathOp = 'sqrt' | 'log' | 'log10' | 'log2' | 'exp' | 'round' | 'floor' | 'ceil' | 'sign'

/** Correlation coefficient: Pearson (linear) or Spearman (rank-based, average ties). */
export type CorrMethod = 'pearson' | 'spearman'

/** Tie handling for the `rank` window function (pandas / Minitab semantics). */
export type RankMethod = 'average' | 'min' | 'max' | 'dense' | 'ordinal'

export type StrOp =
  | 'len'
  | 'toLowerCase'
  | 'toUpperCase'
  | 'trim'
  | 'contains'
  | 'startsWith'
  | 'endsWith'
  | 'replace'
  | 'replaceAll'
  | 'slice'
  | 'split'
  | 'concat'
  | 'padStart'
  | 'padEnd'

export type DtOp = 'year' | 'month' | 'day' | 'hour' | 'minute' | 'second' | 'weekday' | 'epochMillis'

export type ExprNode =
  | { type: 'col'; name: string }
  | { type: 'lit'; value: number | string | boolean | null }
  | {
      type: 'unary'
      op: 'not' | 'isNull' | 'isNotNull' | 'abs' | 'neg' | MathOp
      expr: ExprNode
      /** round: number of decimals (default 0, half away from zero) */
      decimals?: number
      /** log: base (default e) */
      base?: number
    }
  | {
      type: 'binary'
      op:
        | 'eq'
        | 'neq'
        | 'gt'
        | 'gte'
        | 'lt'
        | 'lte'
        | 'and'
        | 'or'
        | 'add'
        | 'sub'
        | 'mul'
        | 'div'
        | 'mod'
        | 'pow'
      left: ExprNode
      right: ExprNode
    }
  | { type: 'alias'; expr: ExprNode; name: string }
  | { type: 'agg'; op: AggKind; expr: ExprNode; q?: number; qm?: QuantileMethod }
  | { type: 'cast'; expr: ExprNode; dtype: DType }
  | { type: 'fillNull'; expr: ExprNode; value: number | string | boolean }
  | {
      type: 'str'
      op: StrOp
      expr: ExprNode
      pattern?: string
      replacement?: string
      start?: number
      end?: number
      /** concat: the expression appended after `expr` (with optional `separator`). */
      other?: ExprNode
      separator?: string
      /** padStart / padEnd: target length and fill string. */
      length?: number
      fill?: string
    }
  | { type: 'dt'; op: DtOp; expr: ExprNode }
  | {
      type: 'when'
      branches: Array<{ when: ExprNode; then: ExprNode }>
      otherwise: ExprNode
    }
  | { type: 'isIn'; expr: ExprNode; values: Array<number | string | boolean | null> }
  | {
      type: 'isBetween'
      expr: ExprNode
      low: ExprNode
      high: ExprNode
      closed?: 'both' | 'left' | 'right' | 'neither'
    }
  | { type: 'clip'; expr: ExprNode; min?: number; max?: number }
  | {
      type: 'rowOffset'
      expr: ExprNode
      periods: number
      kind: 'shift' | 'diff' | 'pctChange'
    }
  | {
      /** Aggregates inside `expr` are computed per partition and broadcast to its rows (polars over / SQL PARTITION BY). */
      type: 'over'
      expr: ExprNode
      partitionBy: string[]
      /** With orderBy the aggregates become running (cumulative) within each partition, rows visited in this order. */
      orderBy?: string[]
      descending?: boolean
    }
  | {
      type: 'mapElements'
      expr: ExprNode
      /** CPU-only UDF; not serializable across workers. */
      fn: (value: number | string | boolean | null) => number | string | boolean | null
    }

export type PlanNode =
  | { type: 'scan'; table: TableView }
  | { type: 'project'; input: PlanNode; columns: Array<string | ExprNode> }
  | { type: 'filter'; input: PlanNode; predicate: ExprNode }
  | {
      type: 'sort'
      input: PlanNode
      by: Array<{ expr: ExprNode; descending: boolean; nullsLast?: boolean }>
    }
  | { type: 'limit'; input: PlanNode; n: number; offset?: number }
  | { type: 'withColumn'; input: PlanNode; name: string; expr: ExprNode }
  | { type: 'withColumns'; input: PlanNode; columns: Array<{ name: string; expr: ExprNode }> }
  | { type: 'drop'; input: PlanNode; columns: string[] }
  | { type: 'rename'; input: PlanNode; mapping: Record<string, string> }
  | {
      type: 'groupBy'
      input: PlanNode
      keys: string[]
      aggs: Array<{ name: string; expr: ExprNode }>
    }
  | {
      type: 'join'
      left: PlanNode
      right: PlanNode
      leftOn: string[]
      rightOn: string[]
      how: JoinKind
      /** Applied to colliding left non-key columns; empty/omitted keeps the left name (pandas-like). */
      lSuffix?: string
      /** Applied to colliding right non-key columns; default `_right`. */
      rSuffix?: string
      /** Assert join-key uniqueness before joining (pandas-style). */
      validate?: '1:1' | '1:m' | 'm:1'
    }
  | {
      type: 'asofJoin'
      left: PlanNode
      right: PlanNode
      leftOn: string
      rightOn: string
      strategy: 'backward' | 'forward' | 'nearest'
    }
  | {
      type: 'fillNull'
      input: PlanNode
      columns?: string[]
      value?: number | string | boolean
      /** Per-column fill values; when set, `value` / `columns` are ignored. */
      values?: Record<string, number | string | boolean>
    }
  | { type: 'ffill'; input: PlanNode; columns?: string[] }
  | { type: 'bfill'; input: PlanNode; columns?: string[] }
  | { type: 'dropNull'; input: PlanNode; columns?: string[] }
  | {
      type: 'melt'
      input: PlanNode
      idVars: string[]
      valueVars: string[]
      varName: string
      valueName: string
    }
  | {
      type: 'pivot'
      input: PlanNode
      index: string[]
      columns: string
      values: string
      agg: AggKind
    }
  | { type: 'concat'; frames: PlanNode[]; how: 'vertical' | 'horizontal' }
  | {
      type: 'window'
      input: PlanNode
      name: string
      fn: 'rank' | 'lag' | 'lead' | 'cumsum' | 'rowNumber'
      expr?: ExprNode
      offset?: number
      /** rank only; default 'average' */
      method?: RankMethod
      partitionBy?: string[]
      orderBy?: Array<{ expr: ExprNode; descending: boolean }>
    }
  | {
      type: 'rolling'
      input: PlanNode
      name: string
      column: string
      window: number
      agg: AggKind
    }
  | {
      type: 'expanding'
      input: PlanNode
      name: string
      column: string
      agg: AggKind
    }
  | { type: 'slice'; input: PlanNode; start: number; end?: number }
  | { type: 'take'; input: PlanNode; indices: number[] }
  | { type: 'sample'; input: PlanNode; n?: number; fraction?: number; seed?: number }
  | { type: 'explode'; input: PlanNode; column: string }
  | { type: 'unnest'; input: PlanNode; column: string; separator?: string }
  | { type: 'transpose'; input: PlanNode; headerColumn?: string }
  | { type: 'interpolate'; input: PlanNode; columns?: string[] }
  | { type: 'unique'; input: PlanNode; columns?: string[]; keep: 'first' | 'last' | 'none' }
  | { type: 'valueCounts'; input: PlanNode; column: string; normalize: boolean }
  | { type: 'describe'; input: PlanNode; quantileMethod?: QuantileMethod }
  | {
      /** Pairwise correlation / covariance matrix over numeric columns (pairwise-complete rows, n−1). */
      type: 'corr'
      input: PlanNode
      kind: 'corr' | 'cov'
      method: CorrMethod
      columns?: string[]
    }

export interface BackendCapabilities {
  name: EngineKind
  gpuFriendlyOnly?: boolean
  minRows?: number
}

/** One executed plan node (or kernel) as reported by the backend that ran it. */
export interface ExecutionEvent {
  /** Plan node type, e.g. "filter", "groupBy". */
  node: string
  /** Backend that actually did the work for this node. */
  backend: EngineKind
  /** Named kernel when one applied (e.g. "rust:filterAnd2I32F64", "gpu:filter", "native:parallelGather"). */
  kernel?: string
  /** Why the requested backend did not handle this node (set when `backend` differs from the requested engine). */
  reason?: string
  /** Wall time of the node, ms (best effort; nested nodes overlap). */
  ms?: number
  /** For GPU nodes: host↔device transfer vs. shader time, ms. */
  transferMs?: number
  computeMs?: number
  rows?: number
}

/** What actually ran, as opposed to what `explain()` planned. */
export interface ExecutionReport {
  requested: EngineKind
  /** Backend the runtime dispatched the plan to. */
  dispatched: EngineKind
  strict: boolean
  events: ExecutionEvent[]
  /** Whole-plan fallbacks performed by the runtime (backend threw or declined). */
  fallbacks: Array<{ from: EngineKind; to: EngineKind; reason: string }>
  totalMs: number
  /** Distinct backends that executed at least one node. */
  backendsUsed: EngineKind[]
  /** Bytes written to spill files during this execution (Node, when MemoryPolicy.spill is on). */
  spilledBytes?: number
  /** Peak estimated live table bytes observed during this execution. */
  peakBytes?: number
  /** True when `persist()` served the result from the LRU cache. */
  cacheHit?: boolean
}

/** Per-execution context handed to backends: records events; `strict` forbids silent delegation. */
export interface ExecContext {
  readonly requested: EngineKind
  readonly strict: boolean
  trace(event: ExecutionEvent): void
  /** Cancellation / deadline guard (`collect({ signal, timeoutMs })`); the CPU engine yields between operators when set. */
  readonly guard?: import('./cancel.js').ExecGuard
  /** Set by the cooperative executor on the context it hands to each execution unit (prevents re-entry). */
  readonly unit?: boolean
}

/** Options accepted by `collect()`, `collectWithReport()` and `Runtime.execute()`. */
export interface ExecuteOptions {
  memory?: import('./memory.js').MemoryPolicy
  /**
   * Abort cooperatively: checked before every operator (and the CPU engine yields to the event loop between
   * operators so the signal can actually fire). Rejects with `ExecutionAbortedError`; a running kernel
   * finishes first — granularity is one operator, not one row.
   */
  signal?: AbortSignal
  /** Deadline in milliseconds, checked at the same points (works without any timer firing). */
  timeoutMs?: number
}

export class EngineStrictError extends Error {
  constructor(
    readonly engine: EngineKind,
    readonly reasons: string[],
  ) {
    super(`engine "${engine}" (strict): ${reasons.join('; ') || 'no kernel of this engine executed'}`)
    this.name = 'EngineStrictError'
  }
}

export function formatExecutionReport(r: ExecutionReport): string {
  const lines = [
    `requested: ${r.requested}${r.strict ? ' (strict)' : ''} · dispatched: ${r.dispatched} · used: ${r.backendsUsed.join(', ') || '—'} · ${r.totalMs.toFixed(1)} ms`,
  ]
  for (const f of r.fallbacks) lines.push(`  fallback ${f.from} → ${f.to}: ${f.reason}`)
  for (const e of r.events) {
    const t = e.ms !== undefined ? ` ${e.ms.toFixed(1)} ms` : ''
    const gpu = e.transferMs !== undefined || e.computeMs !== undefined ? ` (transfer ${(e.transferMs ?? 0).toFixed(1)} / compute ${(e.computeMs ?? 0).toFixed(1)} ms)` : ''
    lines.push(`  ${e.node}: ${e.backend}${e.kernel ? ` [${e.kernel}]` : ''}${t}${gpu}${e.rows !== undefined ? ` rows=${e.rows}` : ''}${e.reason ? ` — ${e.reason}` : ''}`)
  }
  return lines.join('\n')
}

export interface Backend {
  readonly name: EngineKind
  readonly capabilities: BackendCapabilities
  supports(plan: PlanNode): boolean
  /** `ctx` is optional for backward compatibility; backends that receive it should trace what they run. */
  execute(plan: PlanNode, ctx?: ExecContext): Promise<TableView> | TableView
}

export interface RuntimeOptions {
  engine?: EngineKind
  webgpuMinRows?: number
  wasmMinRows?: number
  preferGpu?: boolean
  /**
   * Strict engine selection: the requested engine must execute the plan itself. Any runtime fallback
   * (backend throws, plan unsupported) or a backend that delegates every node to another engine raises
   * `EngineStrictError` with the reasons instead of silently running elsewhere.
   */
  strict?: boolean
  /** Soft memory budget / spill / persist cache (also settable via `setMemoryPolicy`). */
  memory?: import('./memory.js').MemoryPolicy
}

export const DEFAULT_WEBGPU_MIN_ROWS = 10_000
export const DEFAULT_WASM_MIN_ROWS = 1_000

export function collectLeafTables(plan: PlanNode): TableView[] {
  switch (plan.type) {
    case 'scan':
      return [plan.table]
    case 'project':
    case 'filter':
    case 'sort':
    case 'limit':
    case 'withColumn':
    case 'withColumns':
    case 'drop':
    case 'rename':
    case 'groupBy':
    case 'fillNull':
    case 'ffill':
    case 'bfill':
    case 'dropNull':
    case 'melt':
    case 'pivot':
    case 'window':
    case 'rolling':
    case 'expanding':
    case 'slice':
    case 'take':
    case 'sample':
    case 'explode':
    case 'unnest':
    case 'transpose':
    case 'interpolate':
    case 'unique':
    case 'valueCounts':
    case 'describe':
    case 'corr':
      return collectLeafTables(plan.input)
    case 'join':
    case 'asofJoin':
      return [...collectLeafTables(plan.left), ...collectLeafTables(plan.right)]
    case 'concat':
      return plan.frames.flatMap(collectLeafTables)
  }
}

export function estimateRows(plan: PlanNode): number {
  const tables = collectLeafTables(plan)
  if (tables.length === 0) return 0
  return Math.max(...tables.map((t) => t.numRows))
}

export function planUsesOnlyGpuFriendly(plan: PlanNode): boolean {
  const tables = collectLeafTables(plan)
  return tables.every((t) =>
    t.schema.every(
      (f) =>
        f.dtype === 'f32' ||
        f.dtype === 'i32' ||
        f.dtype === 'u32' ||
        f.dtype === 'bool' ||
        f.dtype === 'category' ||
        f.dtype === 'f64',
    ),
  )
}

export function explainPlan(plan: PlanNode, indent = 0): string {
  const pad = '  '.repeat(indent)
  const rowsApprox =
    plan.type === 'scan' ? '' : ` rows≈${estimatePlanRows(plan)}`
  switch (plan.type) {
    case 'scan':
      return `${pad}Scan(rows=${plan.table.numRows}, cols=${plan.table.schema.map((f) => f.name).join(',')})`
    case 'join':
      return `${pad}Join(${plan.how})${rowsApprox}\n${explainPlan(plan.left, indent + 1)}\n${explainPlan(plan.right, indent + 1)}`
    case 'asofJoin':
      return `${pad}AsofJoin(${plan.strategy})${rowsApprox}\n${explainPlan(plan.left, indent + 1)}\n${explainPlan(plan.right, indent + 1)}`
    case 'concat':
      return `${pad}Concat(${plan.how})${rowsApprox}\n${plan.frames.map((f) => explainPlan(f, indent + 1)).join('\n')}`
    case 'groupBy':
      return `${pad}GroupBy(keys=${plan.keys.join(',')})${rowsApprox}\n${explainPlan(plan.input, indent + 1)}`
    case 'project':
      return `${pad}Project${rowsApprox}\n${explainPlan(plan.input, indent + 1)}`
    case 'filter':
      return `${pad}Filter${rowsApprox}\n${explainPlan(plan.input, indent + 1)}`
    case 'sort':
      return `${pad}Sort${rowsApprox}\n${explainPlan(plan.input, indent + 1)}`
    case 'limit':
      return `${pad}Limit(${plan.n})${rowsApprox}\n${explainPlan(plan.input, indent + 1)}`
    case 'withColumns':
      return `${pad}WithColumns(${plan.columns.map((c) => c.name).join(',')})${rowsApprox}\n${explainPlan(plan.input, indent + 1)}`
    default:
      return `${pad}${plan.type}${rowsApprox}\n${'input' in plan ? explainPlan((plan as { input: PlanNode }).input, indent + 1) : ''}`
  }
}
