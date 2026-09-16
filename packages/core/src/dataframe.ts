import {
  allocateData,
  encodeCategory,
  fromArrowLike,
  getValue,
  inferDtype,
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
  type ExprNode,
  type JoinKind,
  type PlanNode,
  type QuantileMethod,
  type RankMethod,
  type Runtime,
} from '@columna/runtime'
import { Expr, col, lit, when } from './expr.js'
import {
  parseCsvToRows,
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
import { writeCsvText, writeParquetBytes, tableToCsv, tableToParquetLike } from './io/write.js'
import { profileTable, tableToHtml, tableToMarkdown, type ProfileReport } from './io/present.js'
export type AggSpec = Record<string, AggKind | Expr>
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

export class LazyFrame {
  constructor(
    readonly plan: PlanNode,
    private runtime: Runtime = getDefaultRuntime(),
  ) {}

  engine(kind: EngineKind): LazyFrame {
    return new LazyFrame(this.plan, this.runtime.withEngine(kind))
  }

  explain(): string {
    return this.runtime.explain(this.plan)
  }

  select(...columns: Array<string | Expr>): LazyFrame {
    return new LazyFrame(
      {
        type: 'project',
        input: this.plan,
        columns: columns.map((c) => (typeof c === 'string' ? c : c.node)),
      },
      this.runtime,
    )
  }

  drop(...columns: string[]): LazyFrame {
    return new LazyFrame({ type: 'drop', input: this.plan, columns }, this.runtime)
  }

  rename(mapping: Record<string, string>): LazyFrame {
    return new LazyFrame({ type: 'rename', input: this.plan, mapping }, this.runtime)
  }

  withColumn(name: string, expr: Expr): LazyFrame {
    return new LazyFrame({ type: 'withColumn', input: this.plan, name, expr: expr.node }, this.runtime)
  }

  /** Add/replace multiple columns in one plan node. */
  withColumns(
    ...cols: Array<Expr | Record<string, Expr> | [string, Expr]>
  ): LazyFrame {
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
    return new LazyFrame({ type: 'withColumns', input: this.plan, columns }, this.runtime)
  }

  filter(predicate: Expr): LazyFrame {
    return new LazyFrame({ type: 'filter', input: this.plan, predicate: predicate.node }, this.runtime)
  }

  sort(...by: Array<string | Expr | { expr: Expr; descending: boolean }>): LazyFrame {
    const keys = by.map((b) => {
      if (typeof b === 'string') return { expr: col(b).node, descending: false }
      if (b instanceof Expr) return { expr: b.node, descending: false }
      return { expr: b.expr.node, descending: b.descending }
    })
    return new LazyFrame({ type: 'sort', input: this.plan, by: keys }, this.runtime)
  }

  head(n = 5): LazyFrame {
    return new LazyFrame({ type: 'limit', input: this.plan, n }, this.runtime)
  }

  tail(n = 5): LazyFrame {
    return new LazyFrame({ type: 'slice', input: this.plan, start: -n }, this.runtime)
  }

  limit(n: number, offset = 0): LazyFrame {
    return new LazyFrame({ type: 'limit', input: this.plan, n, offset }, this.runtime)
  }

  slice(start: number, end?: number): LazyFrame {
    return new LazyFrame({ type: 'slice', input: this.plan, start, end }, this.runtime)
  }

  take(indices: number[]): LazyFrame {
    return new LazyFrame({ type: 'take', input: this.plan, indices }, this.runtime)
  }

  sample(options: { n?: number; fraction?: number; seed?: number } = {}): LazyFrame {
    return new LazyFrame({ type: 'sample', input: this.plan, ...options }, this.runtime)
  }

  explode(column: string): LazyFrame {
    return new LazyFrame({ type: 'explode', input: this.plan, column }, this.runtime)
  }

  unnest(column: string, separator = '.'): LazyFrame {
    return new LazyFrame({ type: 'unnest', input: this.plan, column, separator }, this.runtime)
  }

  transpose(headerColumn?: string): LazyFrame {
    return new LazyFrame({ type: 'transpose', input: this.plan, headerColumn }, this.runtime)
  }

  interpolate(columns?: string[]): LazyFrame {
    return new LazyFrame({ type: 'interpolate', input: this.plan, columns }, this.runtime)
  }

  groupBy(...keys: string[]): GroupBy {
    return new GroupBy(this.plan, keys, this.runtime)
  }

  join(
    other: LazyFrame | DataFrame,
    options: { on?: string | string[]; leftOn?: string | string[]; rightOn?: string | string[]; how?: JoinKind } = {},
  ): LazyFrame {
    const rightPlan = other instanceof DataFrame ? other.lazy().plan : other.plan
    const how = options.how ?? 'inner'
    if (how === 'cross') {
      return new LazyFrame(
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
    return new LazyFrame(
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

  leftJoin(other: LazyFrame | DataFrame, on: string | string[]): LazyFrame {
    return this.join(other, { on, how: 'left' })
  }

  innerJoin(other: LazyFrame | DataFrame, on: string | string[]): LazyFrame {
    return this.join(other, { on, how: 'inner' })
  }

  semiJoin(other: LazyFrame | DataFrame, on: string | string[]): LazyFrame {
    return this.join(other, { on, how: 'semi' })
  }

  antiJoin(other: LazyFrame | DataFrame, on: string | string[]): LazyFrame {
    return this.join(other, { on, how: 'anti' })
  }

  crossJoin(other: LazyFrame | DataFrame): LazyFrame {
    return this.join(other, { how: 'cross' })
  }

  joinAsof(
    other: LazyFrame | DataFrame,
    options: { leftOn: string; rightOn?: string; strategy?: 'backward' | 'forward' | 'nearest' },
  ): LazyFrame {
    const rightPlan = other instanceof DataFrame ? other.lazy().plan : other.plan
    return new LazyFrame(
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

  fillNull(value: number | string | boolean, columns?: string[]): LazyFrame {
    return new LazyFrame({ type: 'fillNull', input: this.plan, value, columns }, this.runtime)
  }

  dropNull(columns?: string[]): LazyFrame {
    return new LazyFrame({ type: 'dropNull', input: this.plan, columns }, this.runtime)
  }

  melt(options: { idVars: string[]; valueVars: string[]; varName?: string; valueName?: string }): LazyFrame {
    return new LazyFrame(
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

  pivot(options: { index: string | string[]; columns: string; values: string; agg?: AggKind }): LazyFrame {
    return new LazyFrame(
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

  unique(columns?: string[], keep: 'first' | 'last' | 'none' = 'first'): LazyFrame {
    return new LazyFrame({ type: 'unique', input: this.plan, columns, keep }, this.runtime)
  }

  valueCounts(column: string, normalize = false): LazyFrame {
    return new LazyFrame({ type: 'valueCounts', input: this.plan, column, normalize }, this.runtime)
  }

  /**
   * count / mean / std / min / 25% / 50% / 75% / max per numeric column.
   * `quantileMethod: 'minitab'` switches the quartiles to Minitab's p(n + 1) definition (type 6).
   */
  describe(options?: { quantileMethod?: QuantileMethod }): LazyFrame {
    return new LazyFrame(
      { type: 'describe', input: this.plan, ...(options?.quantileMethod ? { quantileMethod: options.quantileMethod } : {}) },
      this.runtime,
    )
  }

  /**
   * Pairwise correlation matrix of numeric columns (pandas `df.corr()`): a `column` label column plus one
   * f64 column per variable. Uses pairwise-complete rows; NaN when fewer than two pairs or zero variance.
   */
  corr(options: { columns?: string[]; method?: CorrMethod } = {}): LazyFrame {
    return new LazyFrame(
      { type: 'corr', input: this.plan, kind: 'corr', method: options.method ?? 'pearson', columns: options.columns },
      this.runtime,
    )
  }

  /** Sample covariance matrix (n−1) of numeric columns (pandas `df.cov()`); same layout as `corr()`. */
  cov(options: { columns?: string[] } = {}): LazyFrame {
    return new LazyFrame(
      { type: 'corr', input: this.plan, kind: 'cov', method: 'pearson', columns: options.columns },
      this.runtime,
    )
  }

  withWindow(
    name: string,
    fn: 'rank' | 'lag' | 'lead' | 'cumsum' | 'rowNumber',
    options: {
      expr?: Expr
      offset?: number
      /** rank only: tie handling, default 'average' (pandas / Minitab) */
      method?: RankMethod
      partitionBy?: string[]
      orderBy?: Array<string | Expr | { expr: Expr; descending: boolean }>
    } = {},
  ): LazyFrame {
    const orderBy = options.orderBy?.map((b) => {
      if (typeof b === 'string') return { expr: col(b).node, descending: false }
      if (b instanceof Expr) return { expr: b.node, descending: false }
      return { expr: b.expr.node, descending: b.descending }
    })
    return new LazyFrame(
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

  rolling(name: string, column: string, window: number, agg: AggKind = 'mean'): LazyFrame {
    if (!Number.isInteger(window) || window < 1) throw new RangeError(`rolling: window must be a positive integer (got ${window})`)
    return new LazyFrame({ type: 'rolling', input: this.plan, name, column, window, agg }, this.runtime)
  }

  expanding(name: string, column: string, agg: AggKind = 'mean'): LazyFrame {
    return new LazyFrame({ type: 'expanding', input: this.plan, name, column, agg }, this.runtime)
  }

  pipe<T>(fn: (lf: LazyFrame) => T): T {
    return fn(this)
  }

  static concat(frames: Array<LazyFrame | DataFrame>, how: 'vertical' | 'horizontal' = 'vertical'): LazyFrame {
    return new LazyFrame({
      type: 'concat',
      frames: frames.map((f) => (f instanceof DataFrame ? f.lazy().plan : f.plan)),
      how,
    })
  }

  async collect(): Promise<DataFrame> {
    const table = await this.runtime.execute(this.plan)
    return new DataFrame(table, this.runtime)
  }

  async toArray(): Promise<Record<string, unknown>[]> {
    const df = await this.collect()
    return df.toArray()
  }

  async toArrow(): Promise<ArrowLike> {
    const df = await this.collect()
    return df.toArrow()
  }
}

export class GroupBy {
  constructor(
    private input: PlanNode,
    private keys: string[],
    private runtime: Runtime,
  ) {}

  agg(spec: AggSpec): LazyFrame {
    const aggs: Array<{ name: string; expr: ExprNode }> = []
    for (const [name, value] of Object.entries(spec)) {
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
    return new LazyFrame({ type: 'groupBy', input: this.input, keys: this.keys, aggs }, this.runtime)
  }
}

export class Series {
  constructor(
    readonly name: string,
    readonly column: Column,
    readonly numRows: number,
  ) {}

  get length(): number {
    return this.numRows
  }

  toArray(): Array<number | string | boolean | null> {
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
    return out
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

  unique(): Array<number | string | boolean | null> {
    const seen = new Set<string>()
    const out: Array<number | string | boolean | null> = []
    for (const v of this.toArray()) {
      const k = String(v)
      if (seen.has(k)) continue
      seen.add(k)
      out.push(v)
    }
    return out
  }

  head(n = 5): Array<number | string | boolean | null> {
    return this.toArray().slice(0, n)
  }

  tail(n = 5): Array<number | string | boolean | null> {
    return this.toArray().slice(-n)
  }
}

export class DataFrame {
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

  lazy(): LazyFrame {
    return new LazyFrame({ type: 'scan', table: this.table }, this.runtime)
  }

  select(...columns: Array<string | Expr>): LazyFrame {
    return this.lazy().select(...columns)
  }
  drop(...columns: string[]): LazyFrame {
    return this.lazy().drop(...columns)
  }
  rename(mapping: Record<string, string>): LazyFrame {
    return this.lazy().rename(mapping)
  }
  withColumn(name: string, expr: Expr): LazyFrame {
    return this.lazy().withColumn(name, expr)
  }
  withColumns(...cols: Array<Expr | Record<string, Expr> | [string, Expr]>): LazyFrame {
    return this.lazy().withColumns(...cols)
  }
  filter(predicate: Expr): LazyFrame {
    return this.lazy().filter(predicate)
  }
  sort(...by: Array<string | Expr | { expr: Expr; descending: boolean }>): LazyFrame {
    return this.lazy().sort(...by)
  }
  head(n = 5): LazyFrame {
    return this.lazy().head(n)
  }
  tail(n = 5): LazyFrame {
    return this.lazy().tail(n)
  }
  limit(n: number, offset = 0): LazyFrame {
    return this.lazy().limit(n, offset)
  }
  slice(start: number, end?: number): LazyFrame {
    return this.lazy().slice(start, end)
  }
  take(indices: number[]): LazyFrame {
    return this.lazy().take(indices)
  }
  sample(options?: { n?: number; fraction?: number; seed?: number }): LazyFrame {
    return this.lazy().sample(options)
  }
  explode(column: string): LazyFrame {
    return this.lazy().explode(column)
  }
  unnest(column: string, separator?: string): LazyFrame {
    return this.lazy().unnest(column, separator)
  }
  transpose(headerColumn?: string): LazyFrame {
    return this.lazy().transpose(headerColumn)
  }
  interpolate(columns?: string[]): LazyFrame {
    return this.lazy().interpolate(columns)
  }
  groupBy(...keys: string[]): GroupBy {
    return this.lazy().groupBy(...keys)
  }
  join(
    other: LazyFrame | DataFrame,
    options?: { on?: string | string[]; leftOn?: string | string[]; rightOn?: string | string[]; how?: JoinKind },
  ): LazyFrame {
    return this.lazy().join(other, options)
  }
  leftJoin(other: LazyFrame | DataFrame, on: string | string[]): LazyFrame {
    return this.lazy().leftJoin(other, on)
  }
  innerJoin(other: LazyFrame | DataFrame, on: string | string[]): LazyFrame {
    return this.lazy().innerJoin(other, on)
  }
  semiJoin(other: LazyFrame | DataFrame, on: string | string[]): LazyFrame {
    return this.lazy().semiJoin(other, on)
  }
  antiJoin(other: LazyFrame | DataFrame, on: string | string[]): LazyFrame {
    return this.lazy().antiJoin(other, on)
  }
  crossJoin(other: LazyFrame | DataFrame): LazyFrame {
    return this.lazy().crossJoin(other)
  }
  joinAsof(
    other: LazyFrame | DataFrame,
    options: { leftOn: string; rightOn?: string; strategy?: 'backward' | 'forward' | 'nearest' },
  ): LazyFrame {
    return this.lazy().joinAsof(other, options)
  }
  fillNull(value: number | string | boolean, columns?: string[]): LazyFrame {
    return this.lazy().fillNull(value, columns)
  }
  dropNull(columns?: string[]): LazyFrame {
    return this.lazy().dropNull(columns)
  }
  melt(options: { idVars: string[]; valueVars: string[]; varName?: string; valueName?: string }): LazyFrame {
    return this.lazy().melt(options)
  }
  pivot(options: { index: string | string[]; columns: string; values: string; agg?: AggKind }): LazyFrame {
    return this.lazy().pivot(options)
  }
  unique(columns?: string[], keep?: 'first' | 'last' | 'none'): LazyFrame {
    return this.lazy().unique(columns, keep)
  }
  valueCounts(column: string, normalize?: boolean): LazyFrame {
    return this.lazy().valueCounts(column, normalize)
  }
  describe(options?: { quantileMethod?: QuantileMethod }): LazyFrame {
    return this.lazy().describe(options)
  }
  corr(options?: { columns?: string[]; method?: CorrMethod }): LazyFrame {
    return this.lazy().corr(options)
  }
  cov(options?: { columns?: string[] }): LazyFrame {
    return this.lazy().cov(options)
  }

  withWindow(
    name: string,
    fn: 'rank' | 'lag' | 'lead' | 'cumsum' | 'rowNumber',
    options?: {
      expr?: Expr
      offset?: number
      /** rank only: tie handling, default 'average' (pandas / Minitab) */
      method?: RankMethod
      partitionBy?: string[]
      orderBy?: Array<string | Expr | { expr: Expr; descending: boolean }>
    },
  ): LazyFrame {
    return this.lazy().withWindow(name, fn, options)
  }
  rolling(name: string, column: string, window: number, agg?: AggKind): LazyFrame {
    return this.lazy().rolling(name, column, window, agg)
  }
  expanding(name: string, column: string, agg?: AggKind): LazyFrame {
    return this.lazy().expanding(name, column, agg)
  }
  pipe<T>(fn: (df: DataFrame) => T): T {
    return fn(this)
  }

  toCsv(): string {
    return tableToCsv(this.table)
  }

  async writeCsv(path?: string): Promise<string> {
    return writeCsvText(this.table, path)
  }

  async writeParquet(path?: string): Promise<Uint8Array> {
    return writeParquetBytes(this.table, path)
  }

  toMarkdown(maxRows?: number): string {
    return tableToMarkdown(this.table, maxRows)
  }

  toHTML(maxRows?: number): string {
    return tableToHtml(this.table, maxRows)
  }

  toBlob(format: 'csv' | 'parquet' = 'csv'): Blob {
    if (format === 'csv') return new Blob([this.toCsv()], { type: 'text/csv' })
    const bytes = tableToParquetLike(this.table)
    const copy = new Uint8Array(bytes.byteLength)
    copy.set(bytes)
    return new Blob([copy], { type: 'application/octet-stream' })
  }

  profile(): ProfileReport {
    return profileTable(this.table)
  }
  engine(kind: EngineKind): LazyFrame {
    return this.lazy().engine(kind)
  }
  explain(): string {
    return this.lazy().explain()
  }

  collect(): Promise<DataFrame> {
    return Promise.resolve(this)
  }

  toArray(): Record<string, unknown>[] {
    return toRowObjects(this.table)
  }

  toArrow(): ArrowLike {
    return toArrowLike(this.table)
  }

  getColumn(name: string): Series {
    const idx = this.table.schema.findIndex((f) => f.name === name)
    if (idx < 0) throw new Error(`Unknown column "${name}"`)
    return new Series(name, this.table.columns[idx]!, this.table.numRows)
  }

  static fromRows(rows: Record<string, unknown>[]): DataFrame {
    if (rows.length === 0) return new DataFrame(tableFromColumns([]))
    const n = rows.length
    const names = Object.keys(rows[0]!)
    const sampleLimit = Math.min(n, 256)

    // Infer dtypes from a sample, then fill columnar buffers in one pass.
    const dtypes: DType[] = names.map((name) => {
      const sample: unknown[] = []
      for (let i = 0; i < sampleLimit; i++) sample.push(rows[i]![name])
      return inferDtype(sample)
    })

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
          const v = rows[i]![name]
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
        const v = rows[i]![name]
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
    return new DataFrame(tableFromColumns(columns))
  }

  static fromColumns(
    cols: Record<
      string,
      | Array<number | string | boolean | null | Date>
      | Float64Array
      | Float32Array
      | Int32Array
      | Uint32Array
      | Uint8Array
      | { codes: Uint32Array; dictionary: string[] }
    >,
  ): DataFrame {
    const names = Object.keys(cols)
    if (names.length === 0) return new DataFrame(tableFromColumns([]))
    const colLen = (v: (typeof cols)[string]): number =>
      v && typeof v === 'object' && 'codes' in v && v.codes instanceof Uint32Array ? v.codes.length : (v as { length: number }).length
    const length = colLen(cols[names[0]!]!)
    const columns: Column[] = names.map((name) => {
      const values = cols[name]!
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
          data: values.codes,
          dictionary: values.dictionary,
        } satisfies Column
      }

      // Zero-copy path for typed numeric buffers without nulls
      if (ArrayBuffer.isView(values) && !(values instanceof DataView)) {
        const typed = values as Float64Array | Float32Array | Int32Array | Uint32Array | Uint8Array
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
    return new DataFrame(tableFromColumns(columns))
  }

  static fromJSON(data: Record<string, unknown>[] | string, options: ReadJsonOptions = {}): DataFrame {
    if (typeof data !== 'string' && options.orient === undefined && !options.lines) {
      return DataFrame.fromRows(data)
    }
    return DataFrame.fromRows(parseJsonToRows(data, options))
  }

  static fromArrow(arrow: ArrowLike): DataFrame {
    return new DataFrame(fromArrowLike(arrow))
  }

  /** Sync parse of an in-memory CSV string (pandas/polars-style options). */
  static fromCSV(csv: string, options: ReadCsvOptions = {}): DataFrame {
    return DataFrame.fromRows(parseCsvToRows(csv, { ...options, content: true }))
  }

  /** @deprecated Prefer `readCsv(url, options)`. */
  static async fromCSVUrl(url: string, options?: ReadCsvOptions): Promise<DataFrame> {
    return DataFrame.readCsv(url, options)
  }

  /** Read CSV from path, URL, string content, or bytes. */
  static async readCsv(source: IoSource, options: ReadCsvOptions = {}): Promise<DataFrame> {
    return DataFrame.fromRows(await readCsvRows(source, options))
  }

  /** Read JSON / NDJSON from path, URL, string content, or bytes. */
  static async readJson(source: IoSource, options: ReadJsonOptions = {}): Promise<DataFrame> {
    return DataFrame.fromRows(await readJsonRows(source, options))
  }

  /** Read Excel (.xls / .xlsx) from path, URL, or bytes. */
  static async readExcel(source: IoSource, options: ReadExcelOptions = {}): Promise<DataFrame> {
    return DataFrame.fromRows(await readExcelRows(source, options))
  }

  /** Read Parquet from path, URL, or bytes. */
  static async readParquet(source: IoSource, options: ReadParquetOptions = {}): Promise<DataFrame> {
    return DataFrame.fromRows(await readParquetRows(source, options))
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
