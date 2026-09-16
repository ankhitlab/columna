import type { DType, TableView } from '@columna/arrow'

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
  | { type: 'sort'; input: PlanNode; by: Array<{ expr: ExprNode; descending: boolean }> }
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
    }
  | {
      type: 'asofJoin'
      left: PlanNode
      right: PlanNode
      leftOn: string
      rightOn: string
      strategy: 'backward' | 'forward' | 'nearest'
    }
  | { type: 'fillNull'; input: PlanNode; columns?: string[]; value: number | string | boolean }
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

export interface Backend {
  readonly name: EngineKind
  readonly capabilities: BackendCapabilities
  supports(plan: PlanNode): boolean
  execute(plan: PlanNode): Promise<TableView> | TableView
}

export interface RuntimeOptions {
  engine?: EngineKind
  webgpuMinRows?: number
  wasmMinRows?: number
  preferGpu?: boolean
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
  switch (plan.type) {
    case 'scan':
      return `${pad}Scan(rows=${plan.table.numRows}, cols=${plan.table.schema.map((f) => f.name).join(',')})`
    case 'join':
      return `${pad}Join(${plan.how})\n${explainPlan(plan.left, indent + 1)}\n${explainPlan(plan.right, indent + 1)}`
    case 'asofJoin':
      return `${pad}AsofJoin(${plan.strategy})\n${explainPlan(plan.left, indent + 1)}\n${explainPlan(plan.right, indent + 1)}`
    case 'concat':
      return `${pad}Concat(${plan.how})\n${plan.frames.map((f) => explainPlan(f, indent + 1)).join('\n')}`
    case 'groupBy':
      return `${pad}GroupBy(keys=${plan.keys.join(',')})\n${explainPlan(plan.input, indent + 1)}`
    case 'project':
      return `${pad}Project\n${explainPlan(plan.input, indent + 1)}`
    case 'filter':
      return `${pad}Filter\n${explainPlan(plan.input, indent + 1)}`
    case 'sort':
      return `${pad}Sort\n${explainPlan(plan.input, indent + 1)}`
    case 'limit':
      return `${pad}Limit(${plan.n})\n${explainPlan(plan.input, indent + 1)}`
    case 'withColumns':
      return `${pad}WithColumns(${plan.columns.map((c) => c.name).join(',')})\n${explainPlan(plan.input, indent + 1)}`
    default:
      return `${pad}${plan.type}\n${'input' in plan ? explainPlan((plan as { input: PlanNode }).input, indent + 1) : ''}`
  }
}
