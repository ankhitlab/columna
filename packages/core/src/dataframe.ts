import {
  allocateData,
  encodeCategory,
  fromArrowLike,
  getRowField,
  getValue,
  inferDtype,
  inferDtypeFromValues,
  isValid,
  setValid,
  setValue,
  tableFromColumns,
  toArrowLike,
  toRowObjects,
  type ArrowLike,
  type Column,
  type DType,
  type Field,
  type TableView,
} from '@columna/arrow'
import {
  getDefaultRuntime,
  type AggKind,
  type CorrMethod,
  type EngineKind,
  type ExecutionReport,
  type ExprNode,
  type JoinKind,
  type MemoryPolicy,
  type PlanNode,
  type QuantileMethod,
  type RankMethod,
  type Runtime,
  markPlanPersist,
  unmarkPlanPersist,
} from '@columna/runtime'
import { Expr, col, lit, when, type AnyExpr, type ColRefs } from './expr.js'
import {
  parseCsvToRows,
  parseCsvToTable,
  readCsvTable,
  parseJsonToRows,
  readCsvRows,
  readDatabaseRows,
  readExcelRows,
  readJsonRows,
  readKafkaBatch,
  readParquetRows,
  type IoSource,
  type KafkaConnection,
  type ReadCsvOptions,
  type ReadExcelOptions,
  type ReadJsonOptions,
  type ReadKafkaOptions,
  type ReadParquetOptions,
  type ReadSqlOptions,
  type SqlConnection,
} from './io/index.js'
import { writeCsvText, writeParquetBytes, writeParquetLikeBytes, tableToCsv, tableToParquetLike, type CsvWriteOptions } from './io/write.js'
import { profileTable, tableToHtml, tableToMarkdown, type ProfileReport } from './io/present.js'
export type AggSpec = Record<string, AggKind | AnyExpr>

// ---------------------------------------------------------------------------------------------------
// Schema typing. Frames carry a phantom row type `S`; every operation that changes columns computes the
// next `S`, so the compiler knows which columns exist and what `toArray()` returns. Everything defaults to
// `Row` (`Record<string, unknown>`), i.e. untyped code keeps compiling unchanged.
// ---------------------------------------------------------------------------------------------------

/** A row: column name → value. The default schema when nothing is known. */
export type Row = Record<string, unknown>
type Simplify<T> = { [K in keyof T]: T[K] } & {}
type UnionToIntersection<U> = (U extends unknown ? (k: U) => void : never) extends (k: infer I) => void ? I : never
type NameOf<E> = E extends Expr<any, infer N> ? N : never
type TypeOf<E> = E extends Expr<infer T, any> ? T : never
/** Columns produced by a list of (aliased) expressions; an unaliased expression contributes nothing the compiler can name. */
type ExprRow<E extends readonly AnyExpr[]> = UnionToIntersection<
  { [I in keyof E]: string extends NameOf<E[I]> ? Record<never, never> : Record<NameOf<E[I]>, TypeOf<E[I]>> }[number]
>
type Nullable<S> = { [K in keyof S]: S[K] | null }
/** Schema of a join result. Suffixes for colliding names are not modelled: a shared non-key column keeps the left type here. */
export type JoinResult<L, R, How extends JoinKind> = How extends 'inner' | 'cross'
  ? Simplify<L & R>
  : How extends 'left'
    ? Simplify<L & Nullable<R>>
    : How extends 'right'
      ? Simplify<Nullable<L> & R>
      : How extends 'outer'
        ? Simplify<Nullable<L> & Nullable<R>>
        : L // semi / anti keep the left rows
type Renamed<S, M> = { [K in keyof S as K extends keyof M ? (M[K] extends string ? M[K] : K) : K]: S[K] }
/** Result columns of `groupBy().agg()`: keys keep their type, aggregates are numbers unless the expression says otherwise. */
type AggResult<A> = { [K in keyof A]: A[K] extends Expr<infer T, any> ? (unknown extends T ? number : T) : number }
/** Value type of a column as `Series.toArray()` reports it: the schema's type, or the cell union when unknown. */
type Cell<T> = unknown extends T ? number | string | boolean | null : T
/** Schema inferred from `fromColumns` input. */
type ColumnValue<C> = C extends Float64Array | Float32Array | Int32Array | Uint32Array
  ? number
  : C extends Uint8Array
    ? boolean
    : C extends { codes: Uint32Array; dictionary: string[] }
      ? string
      : C extends ReadonlyArray<infer V>
        ? V extends Date
          ? number
          : V
        : unknown
export type InferColumns<C> = Simplify<{ [K in keyof C]: ColumnValue<C[K]> }>
/** Expression or a callback receiving typed column refs. */
type ExprOrFn<S, E> = E | ((c: ColRefs<S>) => E)
function resolveExpr<S extends Row, E>(v: ExprOrFn<S, E>): E {
  return typeof v === 'function' ? (v as (c: ColRefs<S>) => E)(colRefs<S>()) : v
}
function colRefs<S extends Row>(): ColRefs<S> {
  return new Proxy({} as ColRefs<S>, { get: (_t, key) => (typeof key === 'string' ? col(key) : undefined) })
}
type SortKey = string | AnyExpr | { expr: AnyExpr; descending: boolean }
export type {
  IoSource,
  KafkaConnection,
  ReadCsvOptions,
  ReadExcelOptions,
  ReadJsonOptions,
  ReadKafkaOptions,
  ReadParquetOptions,
  ReadSqlOptions,
  SqlConnection,
}
export type { SqlClient, SqlConnectionConfig, SqlDialect, SqlParams } from './io/sql/types.js'
export type {
  KafkaClient,
  KafkaCodec,
  KafkaFlattenOptions,
  KafkaMetaField,
  KafkaRawMessage,
  KafkaSaslOptions,
} from './io/kafka/types.js'

export class LazyFrame<S extends Row = Row> {
  /** @internal phantom schema; never assigned at runtime */
  declare readonly __schema?: S

  constructor(
    readonly plan: PlanNode,
    private runtime: Runtime = getDefaultRuntime(),
  ) {}

  /**
   * Force a backend. `strict: true` makes the choice a contract: if that engine cannot run the plan, throws,
   * or hands every node to another engine, `collect()` rejects with `EngineStrictError` listing the reasons
   * instead of silently running elsewhere. Without it, `engine()` is a preference with CPU fallback.
   */
  engine(kind: EngineKind, options: { strict?: boolean } = {}): LazyFrame<S> {
    return new LazyFrame<S>(this.plan, this.runtime.withEngine(kind, options))
  }

  /** The plan and the backend it is dispatched to. What ran is only known after execution: see `collectWithReport()`. */
  explain(): string {
    return this.runtime.explain(this.plan)
  }

  /** Keep only these columns (names are checked against the schema) … */
  select<K extends keyof S & string>(...columns: K[]): LazyFrame<Simplify<Pick<S, K>>>
  /** … or build new ones from (aliased) expressions, optionally from typed column refs. */
  select<E extends readonly AnyExpr[]>(...exprs: E): LazyFrame<Simplify<ExprRow<E>>>
  select<const E extends readonly AnyExpr[]>(fn: (c: ColRefs<S>) => E): LazyFrame<Simplify<ExprRow<E>>>
  select(...columns: Array<(keyof S & string) | AnyExpr>): LazyFrame<Row>
  select(...args: Array<string | AnyExpr | ((c: ColRefs<S>) => readonly AnyExpr[])>): LazyFrame<any> {
    const columns = args.length === 1 && typeof args[0] === 'function' ? [...args[0](colRefs<S>())] : (args as Array<string | AnyExpr>)
    return new LazyFrame<any>(
      {
        type: 'project',
        input: this.plan,
        columns: columns.map((c) => (typeof c === 'string' ? c : c.node)),
      },
      this.runtime,
    )
  }

  drop<K extends keyof S & string>(...columns: K[]): LazyFrame<Simplify<Omit<S, K>>> {
    return new LazyFrame<any>({ type: 'drop', input: this.plan, columns }, this.runtime)
  }

  rename<const M extends { [K in keyof S & string]?: string }>(mapping: M): LazyFrame<Simplify<Renamed<S, M>>> {
    return new LazyFrame<any>({ type: 'rename', input: this.plan, mapping: mapping as Record<string, string> }, this.runtime)
  }

  /** Add or replace one column; the result schema knows its name and the expression's value type. */
  withColumn<N extends string, T>(name: N, expr: ExprOrFn<S, Expr<T, any>>): LazyFrame<Simplify<Omit<S, N> & Record<N, T>>> {
    return new LazyFrame<any>({ type: 'withColumn', input: this.plan, name, expr: resolveExpr<S, Expr<T, any>>(expr).node }, this.runtime)
  }

  /** Add/replace multiple columns in one plan node. Aliased expressions extend the schema by name. */
  withColumns<E extends readonly AnyExpr[]>(...exprs: E): LazyFrame<Simplify<Omit<S, keyof ExprRow<E>> & ExprRow<E>>>
  withColumns<const E extends readonly AnyExpr[]>(fn: (c: ColRefs<S>) => E): LazyFrame<Simplify<Omit<S, keyof ExprRow<E>> & ExprRow<E>>>
  withColumns(...cols: Array<AnyExpr | Record<string, AnyExpr> | [string, AnyExpr]>): LazyFrame<Row>
  withColumns(...args: Array<AnyExpr | Record<string, AnyExpr> | [string, AnyExpr] | ((c: ColRefs<S>) => readonly AnyExpr[])>): LazyFrame<any> {
    const cols = args.length === 1 && typeof args[0] === 'function' ? [...args[0](colRefs<S>())] : (args as Array<AnyExpr | Record<string, AnyExpr> | [string, AnyExpr]>)
    const columns: Array<{ name: string; expr: ExprNode }> = []
    for (const c of cols) {
      if (Array.isArray(c)) {
        columns.push({ name: c[0], expr: c[1].node })
      } else if (c instanceof Expr) {
        const name =
          c.node.type === 'alias'
            ? c.node.name
            : c.node.type === 'col'
              ? c.node.name
              : `expr_${columns.length}`
        columns.push({ name, expr: c.node.type === 'alias' ? c.node.expr : c.node })
      } else {
        for (const [name, expr] of Object.entries(c)) {
          columns.push({ name, expr: expr.node })
        }
      }
    }
    return new LazyFrame<any>({ type: 'withColumns', input: this.plan, columns }, this.runtime)
  }

  /** Keep rows where the predicate is true. A typed frame checks the predicate is boolean: `df.filter((c) => c.age.gt(18))`. */
  filter(predicate: ExprOrFn<S, Expr<boolean, any>>): LazyFrame<S> {
    return new LazyFrame<S>({ type: 'filter', input: this.plan, predicate: resolveExpr<S, Expr<boolean, any>>(predicate).node }, this.runtime)
  }

  sort(...by: Array<(keyof S & string) | AnyExpr | { expr: AnyExpr; descending: boolean } | ((c: ColRefs<S>) => SortKey | SortKey[])>): LazyFrame<S>
  sort(...by: Array<SortKey | ((c: ColRefs<S>) => SortKey | SortKey[])>): LazyFrame<S> {
    const flat: SortKey[] = []
    for (const b of by) {
      if (typeof b === 'function') {
        const r = b(colRefs<S>())
        flat.push(...(Array.isArray(r) ? r : [r]))
      } else flat.push(b)
    }
    const keys = flat.map((b) => {
      if (typeof b === 'string') return { expr: col(b).node, descending: false }
      if (b instanceof Expr) return { expr: b.node, descending: false }
      return { expr: b.expr.node, descending: b.descending }
    })
    return new LazyFrame<S>({ type: 'sort', input: this.plan, by: keys }, this.runtime)
  }

  head(n = 5): LazyFrame<S> {
    return new LazyFrame<any>({ type: 'limit', input: this.plan, n }, this.runtime)
  }

  tail(n = 5): LazyFrame<S> {
    return new LazyFrame<any>({ type: 'slice', input: this.plan, start: -n }, this.runtime)
  }

  limit(n: number, offset = 0): LazyFrame<S> {
    return new LazyFrame<any>({ type: 'limit', input: this.plan, n, offset }, this.runtime)
  }

  slice(start: number, end?: number): LazyFrame<S> {
    return new LazyFrame<any>({ type: 'slice', input: this.plan, start, end }, this.runtime)
  }

  take(indices: number[]): LazyFrame<S> {
    return new LazyFrame<any>({ type: 'take', input: this.plan, indices }, this.runtime)
  }

  sample(options: { n?: number; fraction?: number; seed?: number } = {}): LazyFrame<S> {
    return new LazyFrame<any>({ type: 'sample', input: this.plan, ...options }, this.runtime)
  }

  explode(column: string): LazyFrame<Row> {
    return new LazyFrame<any>({ type: 'explode', input: this.plan, column }, this.runtime)
  }

  unnest(column: string, separator = '.'): LazyFrame<Row> {
    return new LazyFrame<any>({ type: 'unnest', input: this.plan, column, separator }, this.runtime)
  }

  transpose(headerColumn?: string): LazyFrame<Row> {
    return new LazyFrame<any>({ type: 'transpose', input: this.plan, headerColumn }, this.runtime)
  }

  interpolate(columns?: string[]): LazyFrame<S> {
    return new LazyFrame<any>({ type: 'interpolate', input: this.plan, columns }, this.runtime)
  }

  groupBy<K extends keyof S & string>(...keys: K[]): GroupBy<S, K> {
    return new GroupBy<S, K>(this.plan, keys, this.runtime)
  }

  join<R extends Row, How extends JoinKind = 'inner'>(
    other: LazyFrame<R> | DataFrame<R>,
    options: { on?: string | string[]; leftOn?: string | string[]; rightOn?: string | string[]; how?: How } = {},
  ): LazyFrame<JoinResult<S, R, How>> {
    const rightPlan = other instanceof DataFrame ? other.lazy().plan : other.plan
    const how = options.how ?? 'inner'
    if (how === 'cross') {
      return new LazyFrame<any>(
        { type: 'join', left: this.plan, right: rightPlan, leftOn: [], rightOn: [], how: 'cross' },
        this.runtime,
      )
    }
    const on = options.on ? (Array.isArray(options.on) ? options.on : [options.on]) : undefined
    const leftOn = options.leftOn
      ? Array.isArray(options.leftOn)
        ? options.leftOn
        : [options.leftOn]
      : on
    const rightOn = options.rightOn
      ? Array.isArray(options.rightOn)
        ? options.rightOn
        : [options.rightOn]
      : on
    if (!leftOn || !rightOn) throw new Error('join requires on or leftOn/rightOn')
    return new LazyFrame<any>(
      {
        type: 'join',
        left: this.plan,
        right: rightPlan,
        leftOn,
        rightOn,
        how,
      },
      this.runtime,
    )
  }

  leftJoin<R extends Row>(other: LazyFrame<R> | DataFrame<R>, on: string | string[]): LazyFrame<JoinResult<S, R, 'left'>> {
    return this.join(other, { on, how: 'left' })
  }

  innerJoin<R extends Row>(other: LazyFrame<R> | DataFrame<R>, on: string | string[]): LazyFrame<JoinResult<S, R, 'inner'>> {
    return this.join(other, { on, how: 'inner' })
  }

  semiJoin<R extends Row>(other: LazyFrame<R> | DataFrame<R>, on: string | string[]): LazyFrame<S> {
    return this.join(other, { on, how: 'semi' })
  }

  antiJoin<R extends Row>(other: LazyFrame<R> | DataFrame<R>, on: string | string[]): LazyFrame<S> {
    return this.join(other, { on, how: 'anti' })
  }

  crossJoin<R extends Row>(other: LazyFrame<R> | DataFrame<R>): LazyFrame<JoinResult<S, R, 'cross'>> {
    return this.join(other, { how: 'cross' })
  }

  joinAsof<R extends Row>(
    other: LazyFrame<R> | DataFrame<R>,
    options: { leftOn: string; rightOn?: string; strategy?: 'backward' | 'forward' | 'nearest' },
  ): LazyFrame<JoinResult<S, R, 'left'>> {
    const rightPlan = other instanceof DataFrame ? other.lazy().plan : other.plan
    return new LazyFrame<any>(
      {
        type: 'asofJoin',
        left: this.plan,
        right: rightPlan,
        leftOn: options.leftOn,
        rightOn: options.rightOn ?? options.leftOn,
        strategy: options.strategy ?? 'backward',
      },
      this.runtime,
    )
  }

  fillNull(value: number | string | boolean, columns?: string[]): LazyFrame<S> {
    return new LazyFrame<any>({ type: 'fillNull', input: this.plan, value, columns }, this.runtime)
  }

  dropNull(columns?: string[]): LazyFrame<S> {
    return new LazyFrame<any>({ type: 'dropNull', input: this.plan, columns }, this.runtime)
  }

  melt(options: { idVars: string[]; valueVars: string[]; varName?: string; valueName?: string }): LazyFrame<Row> {
    return new LazyFrame<any>(
      {
        type: 'melt',
        input: this.plan,
        idVars: options.idVars,
        valueVars: options.valueVars,
        varName: options.varName ?? 'variable',
        valueName: options.valueName ?? 'value',
      },
      this.runtime,
    )
  }

  pivot(options: { index: string | string[]; columns: string; values: string; agg?: AggKind }): LazyFrame<Row> {
    return new LazyFrame<any>(
      {
        type: 'pivot',
        input: this.plan,
        index: Array.isArray(options.index) ? options.index : [options.index],
        columns: options.columns,
        values: options.values,
        agg: options.agg ?? 'sum',
      },
      this.runtime,
    )
  }

  unique(columns?: string[], keep: 'first' | 'last' | 'none' = 'first'): LazyFrame<S> {
    return new LazyFrame<any>({ type: 'unique', input: this.plan, columns, keep }, this.runtime)
  }

  valueCounts(column: string, normalize = false): LazyFrame<Row> {
    return new LazyFrame<any>({ type: 'valueCounts', input: this.plan, column, normalize }, this.runtime)
  }

  /**
   * count / mean / std / min / 25% / 50% / 75% / max per numeric column.
   * `quantileMethod: 'minitab'` switches the quartiles to Minitab's p(n + 1) definition (type 6).
   */
  describe(options?: { quantileMethod?: QuantileMethod }): LazyFrame<Row> {
    return new LazyFrame<any>(
      { type: 'describe', input: this.plan, ...(options?.quantileMethod ? { quantileMethod: options.quantileMethod } : {}) },
      this.runtime,
    )
  }

  /**
   * Pairwise correlation matrix of numeric columns (pandas `df.corr()`): a `column` label column plus one
   * f64 column per variable. Uses pairwise-complete rows; NaN when fewer than two pairs or zero variance.
   */
  corr(options: { columns?: string[]; method?: CorrMethod } = {}): LazyFrame<Row> {
    return new LazyFrame<any>(
      { type: 'corr', input: this.plan, kind: 'corr', method: options.method ?? 'pearson', columns: options.columns },
      this.runtime,
    )
  }

  /** Sample covariance matrix (n−1) of numeric columns (pandas `df.cov()`); same layout as `corr()`. */
  cov(options: { columns?: string[] } = {}): LazyFrame<Row> {
    return new LazyFrame<any>(
      { type: 'corr', input: this.plan, kind: 'cov', method: 'pearson', columns: options.columns },
      this.runtime,
    )
  }

  withWindow<N extends string>(
    name: N,
    fn: 'rank' | 'lag' | 'lead' | 'cumsum' | 'rowNumber',
    options: {
      expr?: AnyExpr
      offset?: number
      /** rank only: tie handling, default 'average' (pandas / Minitab) */
      method?: RankMethod
      partitionBy?: string[]
      orderBy?: Array<string | AnyExpr | { expr: AnyExpr; descending: boolean }>
    } = {},
  ): LazyFrame<Simplify<Omit<S, N> & Record<N, number>>> {
    const orderBy = options.orderBy?.map((b) => {
      if (typeof b === 'string') return { expr: col(b).node, descending: false }
      if (b instanceof Expr) return { expr: b.node, descending: false }
      return { expr: b.expr.node, descending: b.descending }
    })
    return new LazyFrame<any>(
      {
        type: 'window',
        input: this.plan,
        name,
        fn,
        expr: options.expr?.node,
        offset: options.offset,
        method: options.method,
        partitionBy: options.partitionBy,
        orderBy,
      },
      this.runtime,
    )
  }

  rolling<N extends string>(name: N, column: keyof S & string, window: number, agg: AggKind = 'mean'): LazyFrame<Simplify<Omit<S, N> & Record<N, number>>> {
    if (!Number.isInteger(window) || window < 1) throw new RangeError(`rolling: window must be a positive integer (got ${window})`)
    return new LazyFrame<any>({ type: 'rolling', input: this.plan, name, column, window, agg }, this.runtime)
  }

  expanding<N extends string>(name: N, column: keyof S & string, agg: AggKind = 'mean'): LazyFrame<Simplify<Omit<S, N> & Record<N, number>>> {
    return new LazyFrame<any>({ type: 'expanding', input: this.plan, name, column, agg }, this.runtime)
  }

  pipe<T>(fn: (lf: LazyFrame<S>) => T): T {
    return fn(this)
  }

  static concat<F extends LazyFrame<any> | DataFrame<any>>(frames: F[], how: 'vertical' | 'horizontal' = 'vertical'): LazyFrame<F extends LazyFrame<infer X> ? X : F extends DataFrame<infer Y> ? Y : Row> {
    return new LazyFrame<any>({
      type: 'concat',
      frames: frames.map((f) => (f instanceof DataFrame ? f.lazy().plan : f.plan)),
      how,
    })
  }

  async collect(opts?: { memory?: MemoryPolicy }): Promise<DataFrame<S>> {
    const table = await this.runtime.execute(this.plan, opts)
    return new DataFrame<S>(table, this.runtime)
  }

  /**
   * `collect()` plus an execution report: which backend ran each node, which kernel (Rust / native / workers /
   * GPU), why a node fell back to the CPU, and timings (for GPU nodes transfer vs. compute, plus the CPU
   * gather). Use it for benchmarks — "engine('wasm')" is a request, the report is the fact.
   */
  async collectWithReport(opts?: { memory?: MemoryPolicy }): Promise<{ frame: DataFrame<S>; report: ExecutionReport }> {
    const { table, report } = await this.runtime.executeWithReport(this.plan, opts)
    return { frame: new DataFrame<S>(table, this.runtime), report }
  }

  /**
   * Mark this plan for LRU caching: the next `collect` materializes and stores the table; subsequent
   * collects of an identical plan return the cached table (`report.cacheHit`).
   */
  persist(): this {
    markPlanPersist(this.plan)
    return this
  }

  /** Drop a previously persisted plan from the LRU cache. */
  unpersist(): this {
    unmarkPlanPersist(this.plan)
    return this
  }

  async toArray(): Promise<S[]> {
    const df = await this.collect()
    return df.toArray()
  }

  async toArrow(): Promise<ArrowLike> {
    const df = await this.collect()
    return df.toArrow()
  }
}

export class GroupBy<S extends Row = Row, K extends keyof S & string = keyof S & string> {
  constructor(
    private input: PlanNode,
    private keys: K[],
    private runtime: Runtime,
  ) {}

  /** Aggregate per group. Result columns: the keys, then one column per entry (its type from the expression, number for AggKind names). */
  agg<A extends Record<string, AggKind | AnyExpr>>(spec: A | ((c: ColRefs<S>) => A)): LazyFrame<Simplify<Pick<S, K> & AggResult<A>>> {
    const resolved: Record<string, AggKind | AnyExpr> = typeof spec === 'function' ? spec(colRefs<S>()) : spec
    const aggs: Array<{ name: string; expr: ExprNode }> = []
    for (const [name, value] of Object.entries(resolved)) {
      if (value instanceof Expr) {
        const node = value.node.type === 'agg' ? value.node : value.node
        const outName =
          value.node.type === 'alias'
            ? value.node.name
            : name
        aggs.push({
          name: outName,
          expr:
            value.node.type === 'agg'
              ? value.node
              : value.node.type === 'alias' && value.node.expr.type === 'agg'
                ? value.node.expr
                : { type: 'agg', op: 'first', expr: value.node },
        })
        // Prefer dictionary key as output name when not aliased
        if (value.node.type !== 'alias') aggs[aggs.length - 1]!.name = name
      } else {
        aggs.push({ name, expr: { type: 'agg', op: value, expr: { type: 'col', name } } })
      }
    }
    return new LazyFrame<any>({ type: 'groupBy', input: this.input, keys: this.keys, aggs }, this.runtime)
  }
}

export class Series<T = number | string | boolean | null> {
  constructor(
    readonly name: string,
    readonly column: Column,
    readonly numRows: number,
  ) {}

  get length(): number {
    return this.numRows
  }

  get dtype(): DType {
    return this.column.field.dtype
  }

  toArray(): T[] {
    const out: Array<number | string | boolean | null> = []
    for (let i = 0; i < this.numRows; i++) {
      if (!isValid(this.column.nullBitmap, i)) {
        out.push(null)
        continue
      }
      const v = getValue(this.column.data, i)
      if (this.column.field.dtype === 'bool') out.push(Boolean(v))
      else if (this.column.field.dtype === 'category' && this.column.dictionary) {
        out.push(this.column.dictionary[Number(v)] ?? null)
      } else out.push(v as number | string)
    }
    return out as T[]
  }

  nullCount(): number {
    let n = 0
    for (let i = 0; i < this.numRows; i++) if (!isValid(this.column.nullBitmap, i)) n++
    return n
  }

  /** Single streaming pass over finite numeric values — no boxing, no spread (Math.max(...xs) overflows the stack past ~120k). */
  private reduceNums(): { count: number; sum: number; min: number; max: number } {
    const { data, nullBitmap, field } = this.column
    const acc = { count: 0, sum: 0, min: Infinity, max: -Infinity }
    if (field.dtype === 'bool' || field.dtype === 'utf8' || field.dtype === 'category') return acc
    for (let i = 0; i < this.numRows; i++) {
      if (!isValid(nullBitmap, i)) continue
      const v = Number(getValue(data, i))
      if (!Number.isFinite(v)) continue
      acc.count++
      acc.sum += v
      if (v < acc.min) acc.min = v
      if (v > acc.max) acc.max = v
    }
    return acc
  }

  sum(): number {
    return this.reduceNums().sum
  }
  mean(): number | null {
    const { count, sum } = this.reduceNums()
    return count ? sum / count : null
  }
  min(): number | null {
    const { count, min } = this.reduceNums()
    return count ? min : null
  }
  max(): number | null {
    const { count, max } = this.reduceNums()
    return count ? max : null
  }

  unique(): T[] {
    const seen = new Set<string>()
    const out: T[] = []
    for (const v of this.toArray()) {
      const k = String(v)
      if (seen.has(k)) continue
      seen.add(k)
      out.push(v)
    }
    return out
  }

  head(n = 5): T[] {
    return this.toArray().slice(0, n)
  }

  tail(n = 5): T[] {
    return this.toArray().slice(-n)
  }
}

export class DataFrame<S extends Row = Row> {
  /** @internal phantom schema; never assigned at runtime */
  declare readonly __schema?: S

  constructor(
    readonly table: TableView,
    private runtime: Runtime = getDefaultRuntime(),
  ) {}

  get shape(): [number, number] {
    return [this.table.numRows, this.table.schema.length]
  }

  get columns(): string[] {
    return this.table.schema.map((f) => f.name)
  }

  get dtypes(): Record<string, DType> {
    return Object.fromEntries(this.table.schema.map((f) => [f.name, f.dtype]))
  }

  lazy(): LazyFrame<S> {
    return new LazyFrame<S>({ type: 'scan', table: this.table }, this.runtime)
  }

  select<K extends keyof S & string>(...columns: K[]): LazyFrame<Simplify<Pick<S, K>>>
  select<E extends readonly AnyExpr[]>(...exprs: E): LazyFrame<Simplify<ExprRow<E>>>
  select<const E extends readonly AnyExpr[]>(fn: (c: ColRefs<S>) => E): LazyFrame<Simplify<ExprRow<E>>>
  select(...columns: Array<(keyof S & string) | AnyExpr>): LazyFrame<Row>
  select(...args: Array<string | AnyExpr | ((c: ColRefs<S>) => readonly AnyExpr[])>): LazyFrame<any> {
    return (this.lazy().select as (...a: unknown[]) => LazyFrame<any>)(...args)
  }
  drop<K extends keyof S & string>(...columns: K[]): LazyFrame<Simplify<Omit<S, K>>> {
    return this.lazy().drop(...columns)
  }
  rename<const M extends { [K in keyof S & string]?: string }>(mapping: M): LazyFrame<Simplify<Renamed<S, M>>> {
    return this.lazy().rename(mapping)
  }
  withColumn<N extends string, T>(name: N, expr: ExprOrFn<S, Expr<T, any>>): LazyFrame<Simplify<Omit<S, N> & Record<N, T>>> {
    return this.lazy().withColumn(name, expr)
  }
  withColumns<E extends readonly AnyExpr[]>(...exprs: E): LazyFrame<Simplify<Omit<S, keyof ExprRow<E>> & ExprRow<E>>>
  withColumns<const E extends readonly AnyExpr[]>(fn: (c: ColRefs<S>) => E): LazyFrame<Simplify<Omit<S, keyof ExprRow<E>> & ExprRow<E>>>
  withColumns(...cols: Array<AnyExpr | Record<string, AnyExpr> | [string, AnyExpr]>): LazyFrame<Row>
  withColumns(...args: Array<AnyExpr | Record<string, AnyExpr> | [string, AnyExpr] | ((c: ColRefs<S>) => readonly AnyExpr[])>): LazyFrame<any> {
    return (this.lazy().withColumns as (...a: unknown[]) => LazyFrame<any>)(...args)
  }
  filter(predicate: ExprOrFn<S, Expr<boolean, any>>): LazyFrame<S> {
    return this.lazy().filter(predicate)
  }
  sort(...by: Array<(keyof S & string) | AnyExpr | { expr: AnyExpr; descending: boolean } | ((c: ColRefs<S>) => SortKey | SortKey[])>): LazyFrame<S> {
    return this.lazy().sort(...by)
  }
  head(n = 5): LazyFrame<S> {
    return this.lazy().head(n)
  }
  tail(n = 5): LazyFrame<S> {
    return this.lazy().tail(n)
  }
  limit(n: number, offset = 0): LazyFrame<S> {
    return this.lazy().limit(n, offset)
  }
  slice(start: number, end?: number): LazyFrame<S> {
    return this.lazy().slice(start, end)
  }
  take(indices: number[]): LazyFrame<S> {
    return this.lazy().take(indices)
  }
  sample(options?: { n?: number; fraction?: number; seed?: number }): LazyFrame<S> {
    return this.lazy().sample(options)
  }
  explode(column: string): LazyFrame<Row> {
    return this.lazy().explode(column)
  }
  unnest(column: string, separator?: string): LazyFrame<Row> {
    return this.lazy().unnest(column, separator)
  }
  transpose(headerColumn?: string): LazyFrame<Row> {
    return this.lazy().transpose(headerColumn)
  }
  interpolate(columns?: string[]): LazyFrame<S> {
    return this.lazy().interpolate(columns)
  }
  groupBy<K extends keyof S & string>(...keys: K[]): GroupBy<S, K> {
    return this.lazy().groupBy(...keys)
  }
  join<R extends Row, How extends JoinKind = 'inner'>(
    other: LazyFrame<R> | DataFrame<R>,
    options?: { on?: string | string[]; leftOn?: string | string[]; rightOn?: string | string[]; how?: How },
  ): LazyFrame<JoinResult<S, R, How>> {
    return this.lazy().join(other, options)
  }
  leftJoin<R extends Row>(other: LazyFrame<R> | DataFrame<R>, on: string | string[]): LazyFrame<JoinResult<S, R, 'left'>> {
    return this.lazy().leftJoin(other, on)
  }
  innerJoin<R extends Row>(other: LazyFrame<R> | DataFrame<R>, on: string | string[]): LazyFrame<JoinResult<S, R, 'inner'>> {
    return this.lazy().innerJoin(other, on)
  }
  semiJoin<R extends Row>(other: LazyFrame<R> | DataFrame<R>, on: string | string[]): LazyFrame<S> {
    return this.lazy().semiJoin(other, on)
  }
  antiJoin<R extends Row>(other: LazyFrame<R> | DataFrame<R>, on: string | string[]): LazyFrame<S> {
    return this.lazy().antiJoin(other, on)
  }
  crossJoin<R extends Row>(other: LazyFrame<R> | DataFrame<R>): LazyFrame<JoinResult<S, R, 'cross'>> {
    return this.lazy().crossJoin(other)
  }
  joinAsof<R extends Row>(
    other: LazyFrame<R> | DataFrame<R>,
    options: { leftOn: string; rightOn?: string; strategy?: 'backward' | 'forward' | 'nearest' },
  ): LazyFrame<JoinResult<S, R, 'left'>> {
    return this.lazy().joinAsof(other, options)
  }
  fillNull(value: number | string | boolean, columns?: string[]): LazyFrame<S> {
    return this.lazy().fillNull(value, columns)
  }
  dropNull(columns?: string[]): LazyFrame<S> {
    return this.lazy().dropNull(columns)
  }
  melt(options: { idVars: string[]; valueVars: string[]; varName?: string; valueName?: string }): LazyFrame<Row> {
    return this.lazy().melt(options)
  }
  pivot(options: { index: string | string[]; columns: string; values: string; agg?: AggKind }): LazyFrame<Row> {
    return this.lazy().pivot(options)
  }
  unique(columns?: string[], keep?: 'first' | 'last' | 'none'): LazyFrame<S> {
    return this.lazy().unique(columns, keep)
  }
  valueCounts(column: string, normalize?: boolean): LazyFrame<Row> {
    return this.lazy().valueCounts(column, normalize)
  }
  describe(options?: { quantileMethod?: QuantileMethod }): LazyFrame<Row> {
    return this.lazy().describe(options)
  }
  corr(options?: { columns?: string[]; method?: CorrMethod }): LazyFrame<Row> {
    return this.lazy().corr(options)
  }
  cov(options?: { columns?: string[] }): LazyFrame<Row> {
    return this.lazy().cov(options)
  }

  withWindow<N extends string>(
    name: N,
    fn: 'rank' | 'lag' | 'lead' | 'cumsum' | 'rowNumber',
    options?: {
      expr?: AnyExpr
      offset?: number
      /** rank only: tie handling, default 'average' (pandas / Minitab) */
      method?: RankMethod
      partitionBy?: string[]
      orderBy?: Array<string | AnyExpr | { expr: AnyExpr; descending: boolean }>
    },
  ): LazyFrame<Simplify<Omit<S, N> & Record<N, number>>> {
    return this.lazy().withWindow(name, fn, options)
  }
  rolling<N extends string>(name: N, column: keyof S & string, window: number, agg?: AggKind): LazyFrame<Simplify<Omit<S, N> & Record<N, number>>> {
    return this.lazy().rolling(name, column, window, agg)
  }
  expanding<N extends string>(name: N, column: keyof S & string, agg?: AggKind): LazyFrame<Simplify<Omit<S, N> & Record<N, number>>> {
    return this.lazy().expanding(name, column, agg)
  }
  pipe<T>(fn: (df: DataFrame<S>) => T): T {
    return fn(this)
  }

  /**
   * CSV text. `{ escapeFormulas: true }` neutralises text cells a spreadsheet would run as formulas
   * (leading = + - @ tab CR → "'"-prefixed, quoted) — use it for exports of untrusted text that will be
   * opened in Excel-like applications; the default writes the data unchanged.
   */
  toCsv(options: CsvWriteOptions = {}): string {
    return tableToCsv(this.table, options)
  }

  /** Write CSV to a path (streamed, memory bounded by one chunk; resolves to '') or return the text when no path is given. */
  async writeCsv(path?: string, options: CsvWriteOptions = {}): Promise<string> {
    return writeCsvText(this.table, path, options)
  }

  async writeParquet(path?: string): Promise<Uint8Array> {
    return writeParquetBytes(this.table, path)
  }

  /** Write columna's JSON "parquet-like" format (round-trips with `readParquetLike`, not Apache Parquet). */
  async writeParquetLike(path?: string): Promise<Uint8Array> {
    return writeParquetLikeBytes(this.table, path)
  }

  toMarkdown(maxRows?: number): string {
    return tableToMarkdown(this.table, maxRows)
  }

  toHTML(maxRows?: number): string {
    return tableToHtml(this.table, maxRows)
  }

  toBlob(format: 'csv' | 'parquet-like' = 'csv', options: CsvWriteOptions = {}): Blob {
    if (format === 'csv') return new Blob([this.toCsv(options)], { type: 'text/csv' })
    const bytes = tableToParquetLike(this.table)
    const copy = new Uint8Array(bytes.byteLength)
    copy.set(bytes)
    return new Blob([copy], { type: 'application/json' })
  }

  profile(): ProfileReport {
    return profileTable(this.table)
  }
  engine(kind: EngineKind, options?: { strict?: boolean }): LazyFrame<S> {
    return this.lazy().engine(kind, options)
  }
  explain(): string {
    return this.lazy().explain()
  }
  collectWithReport(opts?: { memory?: MemoryPolicy }): Promise<{ frame: DataFrame<S>; report: ExecutionReport }> {
    return this.lazy().collectWithReport(opts)
  }

  collect(): Promise<DataFrame<S>> {
    return Promise.resolve(this)
  }

  persist(): LazyFrame<S> {
    return this.lazy().persist()
  }

  unpersist(): LazyFrame<S> {
    return this.lazy().unpersist()
  }

  /** Rows as objects, typed by the schema. */
  toArray(): S[] {
    return toRowObjects(this.table) as S[]
  }

  toArrow(): ArrowLike {
    return toArrowLike(this.table)
  }

  getColumn<K extends keyof S & string>(name: K): Series<Cell<S[K]>> {
    const idx = this.table.schema.findIndex((f) => f.name === name)
    if (idx < 0) throw new Error(`Unknown column "${name}"`)
    return new Series(name, this.table.columns[idx]!, this.table.numRows)
  }

  /**
   * Build a frame from row objects. The schema is the union of the keys of every row, in order of first
   * appearance (a key missing from a row reads as null); dtypes are inferred over the whole column, so a
   * late fractional / out-of-Int32 / string value widens the column instead of being coerced.
   */
  static fromRows<T extends Row>(rows: readonly T[]): DataFrame<T> {
    if (rows.length === 0) return new DataFrame<T>(tableFromColumns([]))
    const n = rows.length
    const names = Object.keys(rows[0]!)
    // Union of keys across rows (pandas semantics); the fast path is rows sharing the first row's shape.
    const seen = new Set(names)
    for (let i = 1; i < n; i++) {
      const row = rows[i]!
      for (const k in row) if (Object.hasOwn(row, k) && !seen.has(k)) {
        seen.add(k)
        names.push(k)
      }
    }

    // Infer dtypes from the full column (not a 256-row sample) so late floats/strings
    // cannot be silently coerced into a wrong integer/utf8 layout.
    const dtypes: DType[] = names.map((name) =>
      inferDtypeFromValues(n, (i) => getRowField(rows[i]! as Record<string, unknown>, name)),
    )

    const columns: Column[] = names.map((name, ci) => {
      const dtype = dtypes[ci]!
      if (dtype === 'utf8') {
        // Dictionary-encode low-cardinality strings (city, category, …)
        const dict: string[] = []
        const map = new Map<string, number>()
        const codes = new Uint32Array(n)
        let anyNull = false
        const nullBitmap = new Uint8Array(Math.ceil(n / 8) || 1)
        for (let i = 0; i < n; i++) {
          const v = getRowField(rows[i]! as Record<string, unknown>, name)
          if (v === null || v === undefined) {
            anyNull = true
            codes[i] = 0
            continue
          }
          setValid(nullBitmap, i, true)
          const s = String(v)
          let code = map.get(s)
          if (code === undefined) {
            code = dict.length
            dict.push(s)
            map.set(s, code)
          }
          codes[i] = code
        }
        // Keep as category when cardinality is modest; else materialize utf8
        if (dict.length <= Math.max(1024, n / 4)) {
          return {
            field: { name, dtype: 'category', nullable: anyNull },
            data: codes,
            nullBitmap: anyNull ? nullBitmap : undefined,
            dictionary: dict,
          } satisfies Column
        }
        const data = new Array<string>(n)
        for (let i = 0; i < n; i++) {
          if (anyNull && !isValid(nullBitmap, i)) data[i] = ''
          else data[i] = dict[codes[i]!]!
        }
        return {
          field: { name, dtype: 'utf8', nullable: anyNull },
          data,
          nullBitmap: anyNull ? nullBitmap : undefined,
        } satisfies Column
      }

      const data = allocateData(dtype, n)
      let anyNull = false
      const nullBitmap = new Uint8Array(Math.ceil(n / 8) || 1)
      for (let i = 0; i < n; i++) {
        const v = getRowField(rows[i]! as Record<string, unknown>, name)
        if (v === null || v === undefined) {
          anyNull = true
          continue
        }
        setValid(nullBitmap, i, true)
        if (dtype === 'bool') setValue(data, i, Boolean(v), dtype)
        else if (dtype === 'datetime') setValue(data, i, v instanceof Date ? v.getTime() : Number(v), dtype)
        else setValue(data, i, v as number | string | boolean, dtype)
      }
      return {
        field: { name, dtype, nullable: anyNull } satisfies Field,
        data,
        nullBitmap: anyNull ? nullBitmap : undefined,
      }
    })
    return new DataFrame<any>(tableFromColumns(columns))
  }

  static fromColumns<
    C extends Record<
      string,
      | ReadonlyArray<number | string | boolean | null | Date>
      | Float64Array
      | Float32Array
      | Int32Array
      | Uint32Array
      | Uint8Array
      | { codes: Uint32Array; dictionary: string[] }
    >,
  >(cols: C, options: { copy?: boolean } = {}): DataFrame<InferColumns<C>> {
    const names = Object.keys(cols)
    if (names.length === 0) return new DataFrame<any>(tableFromColumns([]))
    type Input =
      | ReadonlyArray<number | string | boolean | null | Date>
      | Float64Array
      | Float32Array
      | Int32Array
      | Uint32Array
      | Uint8Array
      | { codes: Uint32Array; dictionary: string[] }
    const colLen = (v: Input): number =>
      v && typeof v === 'object' && 'codes' in v && v.codes instanceof Uint32Array ? v.codes.length : (v as { length: number }).length
    const length = colLen(cols[names[0]!] as Input)
    const columns: Column[] = names.map((name) => {
      const values = cols[name]! as Input
      if (colLen(values) !== length) throw new Error(`Column length mismatch for ${name}`)

      // Pre-encoded category (needed for 10M–100M row benches — avoid string[])
      if (
        values &&
        typeof values === 'object' &&
        'codes' in values &&
        'dictionary' in values &&
        values.codes instanceof Uint32Array &&
        Array.isArray(values.dictionary)
      ) {
        return {
          field: { name, dtype: 'category', nullable: false },
          data: options.copy ? values.codes.slice() : values.codes,
          dictionary: options.copy ? values.dictionary.slice() : values.dictionary,
        } satisfies Column
      }

      // Typed buffers are shared, not copied (README → "Buffer ownership"): the frame reads the caller's
      // array, so a later write to that array changes the frame. Pass { copy: true } to detach.
      if (ArrayBuffer.isView(values) && !(values instanceof DataView)) {
        const typed = (options.copy ? values.slice() : values) as Float64Array | Float32Array | Int32Array | Uint32Array | Uint8Array
        let dtype: DType = 'f64'
        if (typed instanceof Float32Array) dtype = 'f32'
        else if (typed instanceof Int32Array) dtype = 'i32'
        else if (typed instanceof Uint32Array) dtype = 'u32'
        else if (typed instanceof Uint8Array) dtype = 'bool'
        else dtype = 'f64'
        return {
          field: { name, dtype, nullable: false },
          data: typed,
        } satisfies Column
      }

      const arr = values as Array<number | string | boolean | null | Date>
      const dtype = inferDtype(arr)
      if (dtype === 'utf8') {
        const encoded = encodeCategory(arr.map((v) => (v == null ? null : String(v))))
        if (encoded.dictionary.length <= Math.max(1024, length / 4)) {
          return {
            field: { name, dtype: 'category', nullable: Boolean(encoded.nullBitmap) },
            data: encoded.codes,
            nullBitmap: encoded.nullBitmap,
            dictionary: encoded.dictionary,
          } satisfies Column
        }
        const data = arr.map((v) => (v == null ? '' : String(v)))
        return {
          field: { name, dtype: 'utf8', nullable: Boolean(encoded.nullBitmap) },
          data,
          nullBitmap: encoded.nullBitmap,
        } satisfies Column
      }

      const data = allocateData(dtype === 'datetime' ? 'datetime' : dtype, length)
      let anyNull = false
      const nullBitmap = new Uint8Array(Math.ceil(length / 8) || 1)
      for (let i = 0; i < length; i++) {
        const v = arr[i]
        if (v === null || v === undefined) {
          anyNull = true
          continue
        }
        setValid(nullBitmap, i, true)
        if (dtype === 'bool') setValue(data, i, Boolean(v), 'bool')
        else if (dtype === 'datetime') setValue(data, i, v instanceof Date ? v.getTime() : Number(v), 'datetime')
        else setValue(data, i, Number(v), dtype)
      }
      return {
        field: { name, dtype, nullable: anyNull },
        data,
        nullBitmap: anyNull ? nullBitmap : undefined,
      }
    })
    return new DataFrame<any>(tableFromColumns(columns))
  }

  static fromJSON<S extends Row = Row>(data: Record<string, unknown>[] | string, options: ReadJsonOptions = {}): DataFrame<S> {
    if (typeof data !== 'string' && options.orient === undefined && !options.lines) {
      return DataFrame.fromRows(data) as DataFrame<S>
    }
    return DataFrame.fromRows(parseJsonToRows(data, options)) as DataFrame<S>
  }

  static fromArrow<S extends Row = Row>(arrow: ArrowLike): DataFrame<S> {
    return new DataFrame<any>(fromArrowLike(arrow))
  }

  /** Sync parse of an in-memory CSV string (pandas/polars-style options). */
  /** Parse CSV text. `S` is an assertion about the file, not something the compiler can verify. */
  static fromCSV<S extends Row = Row>(csv: string, options: ReadCsvOptions = {}): DataFrame<S> {
    return new DataFrame<S>(parseCsvToTable(csv, options))
  }

  /** @deprecated internal: previous row-object implementation of fromCSV, kept for parity tests. */
  static fromCSVRows(csv: string, options: ReadCsvOptions = {}): DataFrame {
    return DataFrame.fromRows(parseCsvToRows(csv, { ...options, content: true }))
  }

  /** @deprecated Prefer `readCsv(url, options)`. */
  static async fromCSVUrl(url: string, options?: ReadCsvOptions): Promise<DataFrame> {
    return DataFrame.readCsv(url, options)
  }

  /** Read CSV from path, URL, string content, or bytes. */
  /** Read CSV from a path / URL / bytes / text. `S` is an assertion about the file, not something the compiler can verify. */
  static async readCsv<S extends Row = Row>(source: IoSource, options: ReadCsvOptions = {}): Promise<DataFrame<S>> {
    return new DataFrame<S>(await readCsvTable(source, options), getDefaultRuntime())
  }

  /** @deprecated internal: previous row-object implementation of readCsv, kept for parity tests. */
  static async readCsvRows(source: IoSource, options: ReadCsvOptions = {}): Promise<DataFrame> {
    return DataFrame.fromRows(await readCsvRows(source, options))
  }

  /** Read JSON / NDJSON from path, URL, string content, or bytes. */
  static async readJson<S extends Row = Row>(source: IoSource, options: ReadJsonOptions = {}): Promise<DataFrame<S>> {
    return DataFrame.fromRows(await readJsonRows(source, options)) as DataFrame<S>
  }

  /** Read Excel (.xls / .xlsx) from path, URL, or bytes. */
  static async readExcel<S extends Row = Row>(source: IoSource, options: ReadExcelOptions = {}): Promise<DataFrame<S>> {
    return DataFrame.fromRows(await readExcelRows(source, options)) as DataFrame<S>
  }

  /** Read Parquet from path, URL, or bytes. */
  static async readParquet<S extends Row = Row>(source: IoSource, options: ReadParquetOptions = {}): Promise<DataFrame<S>> {
    return DataFrame.fromRows(await readParquetRows(source, options)) as DataFrame<S>
  }

  /**
   * Load a query result from a database (pandas `read_sql` / polars `read_database`).
   *
   * Built-in dialects: `postgres`, `mssql`, `clickhouse`, `mysql`, `sqlite`.
   * Drivers are optional peer packages (`pg`, `mssql`, `@clickhouse/client`, `mysql2`, `better-sqlite3`).
   */
  static async readSql(
    sql: string,
    connection: SqlConnection,
    options: ReadSqlOptions = {},
  ): Promise<DataFrame> {
    return DataFrame.fromRows(await readDatabaseRows(sql, connection, options))
  }

  /** Alias of `readSql`. */
  static async readDatabase(
    sql: string,
    connection: SqlConnection,
    options: ReadSqlOptions = {},
  ): Promise<DataFrame> {
    return DataFrame.readSql(sql, connection, options)
  }

  /**
   * Consume a bounded batch from Kafka into a DataFrame.
   *
   * Nested JSON values are flattened to dotted columns by default
   * (`user.address.city`). Driver: optional peer `kafkajs`.
   *
   * @example
   * await DataFrame.readKafka({
   *   brokers: ['localhost:9092'],
   *   topic: 'events',
   *   fromBeginning: true,
   *   nMessages: 1000,
   * })
   */
  static async readKafka(
    connection: KafkaConnection,
    options: Partial<ReadKafkaOptions> = {},
  ): Promise<DataFrame> {
    return DataFrame.fromRows(await readKafkaBatch(connection, options))
  }

  /** Read from a browser `File` / `Blob` (CSV/JSON inferred by name or `format`). */
  static async readFile(
    file: Blob & { name?: string },
    options: { format?: 'csv' | 'json' | 'parquet' | 'excel' } & ReadCsvOptions & ReadJsonOptions = {},
  ): Promise<DataFrame> {
    const name = (file as { name?: string }).name ?? ''
    const format =
      options.format ??
      (/\.parquet$/i.test(name)
        ? 'parquet'
        : /\.xlsx?$/i.test(name)
          ? 'excel'
          : /\.jsonl?$/i.test(name) || /\.ndjson$/i.test(name)
            ? 'json'
            : 'csv')
    if (format === 'csv') return DataFrame.readCsv(file, options)
    if (format === 'json') return DataFrame.readJson(file, options)
    if (format === 'excel') return DataFrame.readExcel(file, options)
    return DataFrame.readParquet(file, options)
  }
}

export { col, lit, Expr, when }
