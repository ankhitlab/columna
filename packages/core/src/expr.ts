import type { AggKind, ExprNode, QuantileMethod } from '@columna/runtime'
import type { DType } from '@columna/arrow'

const DTYPES: ReadonlySet<DType> = new Set<DType>(['f64', 'f32', 'i32', 'u32', 'bool', 'utf8', 'category', 'datetime'])
import { StrNamespace } from './expr/str.js'
import { DtNamespace } from './expr/dt.js'

/**
 * Static value type of an expression (`T`) and its output column name (`N`, set by `alias`). Both are
 * phantom: `col('x')` is `Expr<any>` (nothing is known about a column named by string), `c.x` from a typed
 * frame's column refs is `Expr<S['x'], 'x'>`, comparisons produce `Expr<boolean>`, arithmetic `Expr<number>`.
 * Frames use them to type `filter`, `withColumn(s)`, `select` and `agg` results.
 */
export class Expr<T = any, N extends string = string> {
  /** @internal phantom — carries the value type; never assigned at runtime */
  declare readonly __type?: T
  /** @internal phantom — carries the output name; never assigned at runtime */
  declare readonly __name?: N

  constructor(readonly node: ExprNode) {}

  get str(): StrNamespace {
    return new StrNamespace(this)
  }

  get dt(): DtNamespace {
    return new DtNamespace(this)
  }

  alias<M extends string>(name: M): Expr<T, M> {
    return new Expr({ type: 'alias', expr: this.node, name })
  }

  cast<D extends DType>(dtype: D): Expr<DTypeValue<D>, N> {
    if (!DTYPES.has(dtype)) throw new RangeError(`cast: unknown dtype "${String(dtype)}" (expected ${[...DTYPES].join(' | ')})`)
    return new Expr({ type: 'cast', expr: this.node, dtype })
  }

  fillNull<V extends number | string | boolean>(value: V): Expr<Exclude<T, null> | V, N> {
    return new Expr({ type: 'fillNull', expr: this.node, value })
  }

  isNull(): Expr<boolean, N> {
    return new Expr({ type: 'unary', op: 'isNull', expr: this.node })
  }

  isNotNull(): Expr<boolean, N> {
    return new Expr({ type: 'unary', op: 'isNotNull', expr: this.node })
  }

  not(): Expr<boolean, N> {
    return new Expr({ type: 'unary', op: 'not', expr: this.node })
  }

  abs(): Expr<number, N> {
    return new Expr({ type: 'unary', op: 'abs', expr: this.node })
  }

  neg(): Expr<number, N> {
    return new Expr({ type: 'unary', op: 'neg', expr: this.node })
  }

  // Element-wise math (null → null, domain errors → NaN). Run on the typed fast path.
  sqrt(): Expr<number, N> {
    return new Expr({ type: 'unary', op: 'sqrt', expr: this.node })
  }

  /** Natural log by default; pass `base` for log_b (e.g. `log(10)`). */
  log(base?: number): Expr<number, N> {
    return new Expr({ type: 'unary', op: 'log', expr: this.node, base })
  }

  log10(): Expr {
    return new Expr({ type: 'unary', op: 'log10', expr: this.node })
  }

  log2(): Expr {
    return new Expr({ type: 'unary', op: 'log2', expr: this.node })
  }

  exp(): Expr<number, N> {
    return new Expr({ type: 'unary', op: 'exp', expr: this.node })
  }

  /** Round half away from zero (Minitab / Excel ROUND) to `decimals` places. */
  round(decimals = 0): Expr<number, N> {
    return new Expr({ type: 'unary', op: 'round', expr: this.node, decimals })
  }

  floor(): Expr<number, N> {
    return new Expr({ type: 'unary', op: 'floor', expr: this.node })
  }

  ceil(): Expr<number, N> {
    return new Expr({ type: 'unary', op: 'ceil', expr: this.node })
  }

  sign(): Expr<number, N> {
    return new Expr({ type: 'unary', op: 'sign', expr: this.node })
  }

  pow(other: AnyExpr | number): Expr<number, N> {
    return bin('pow', this, other)
  }

  eq(other: AnyExpr | number | string | boolean | null): Expr<boolean, N> {
    return bin('eq', this, other)
  }
  neq(other: AnyExpr | number | string | boolean | null): Expr<boolean, N> {
    return bin('neq', this, other)
  }
  gt(other: AnyExpr | number | string | boolean | null): Expr<boolean, N> {
    return bin('gt', this, other)
  }
  gte(other: AnyExpr | number | string | boolean | null): Expr<boolean, N> {
    return bin('gte', this, other)
  }
  lt(other: AnyExpr | number | string | boolean | null): Expr<boolean, N> {
    return bin('lt', this, other)
  }
  lte(other: AnyExpr | number | string | boolean | null): Expr<boolean, N> {
    return bin('lte', this, other)
  }
  and(other: AnyExpr | boolean): Expr<boolean, N> {
    return bin('and', this, other)
  }
  or(other: AnyExpr | boolean): Expr<boolean, N> {
    return bin('or', this, other)
  }
  add(other: AnyExpr | number): Expr<number, N> {
    return bin('add', this, other)
  }
  sub(other: AnyExpr | number): Expr<number, N> {
    return bin('sub', this, other)
  }
  mul(other: AnyExpr | number): Expr<number, N> {
    return bin('mul', this, other)
  }
  div(other: AnyExpr | number): Expr<number, N> {
    return bin('div', this, other)
  }
  mod(other: AnyExpr | number): Expr<number, N> {
    return bin('mod', this, other)
  }

  isIn(values: Array<number | string | boolean | null>): Expr<boolean, N> {
    return new Expr({ type: 'isIn', expr: this.node, values })
  }

  isBetween(
    low: AnyExpr | number | string,
    high: AnyExpr | number | string,
    closed: 'both' | 'left' | 'right' | 'neither' = 'both',
  ): Expr<boolean, N> {
    return new Expr({
      type: 'isBetween',
      expr: this.node,
      low: toExpr(low).node,
      high: toExpr(high).node,
      closed,
    })
  }

  clip(min?: number, max?: number): Expr<T, N> {
    return new Expr({ type: 'clip', expr: this.node, min, max })
  }

  shift(periods = 1): Expr<T, N> {
    return new Expr({ type: 'rowOffset', expr: this.node, periods, kind: 'shift' })
  }

  diff(periods = 1): Expr<number, N> {
    return new Expr({ type: 'rowOffset', expr: this.node, periods, kind: 'diff' })
  }

  pctChange(periods = 1): Expr<number, N> {
    return new Expr({ type: 'rowOffset', expr: this.node, periods, kind: 'pctChange' })
  }

  mapElements<R extends number | string | boolean | null>(fn: (value: number | string | boolean | null) => R): Expr<R, N> {
    return new Expr({ type: 'mapElements', expr: this.node, fn })
  }

  sum(): Expr<number, N> {
    return new Expr({ type: 'agg', op: 'sum', expr: this.node })
  }
  mean(): Expr<number, N> {
    return new Expr({ type: 'agg', op: 'mean', expr: this.node })
  }
  min(): Expr<T, N> {
    return new Expr({ type: 'agg', op: 'min', expr: this.node })
  }
  max(): Expr<T, N> {
    return new Expr({ type: 'agg', op: 'max', expr: this.node })
  }
  count(): Expr<number, N> {
    return new Expr({ type: 'agg', op: 'count', expr: this.node })
  }
  nunique(): Expr<number, N> {
    return new Expr({ type: 'agg', op: 'nunique', expr: this.node })
  }
  first(): Expr<T, N> {
    return new Expr({ type: 'agg', op: 'first', expr: this.node })
  }
  last(): Expr<T, N> {
    return new Expr({ type: 'agg', op: 'last', expr: this.node })
  }
  std(): Expr<number, N> {
    return new Expr({ type: 'agg', op: 'std', expr: this.node })
  }
  var(): Expr<number, N> {
    return new Expr({ type: 'agg', op: 'var', expr: this.node })
  }
  /**
   * Median. `method`: 'linear' (default; type 7, pandas / polars) or 'minitab' (type 6, position p(n + 1)) —
   * both give the middle value / midpoint for the median, the choice matters for other quantiles.
   */
  median(method?: QuantileMethod): Expr<number, N> {
    return new Expr({ type: 'agg', op: 'median', expr: this.node, ...(method ? { qm: method } : {}) })
  }
  /**
   * Sample quantile q ∈ [0, 1].
   *   col('x').quantile(0.25)             // type 7: position (n − 1)q, linear interpolation (pandas default)
   *   col('x').quantile(0.25, 'minitab')  // type 6: position q(n + 1), clamped to the sample — Minitab / SPSS / R type=6
   */
  quantile(q: number, method?: QuantileMethod): Expr<number, N> {
    return new Expr({ type: 'agg', op: 'quantile', expr: this.node, q, ...(method ? { qm: method } : {}) })
  }

  /**
   * Window: aggregates inside this expression are computed per partition and broadcast back to each
   * row of the partition (polars `over`, SQL `… OVER (PARTITION BY …)`).
   *   col('x').mean().over('g')                          // group mean on every row
   *   col('x').sub(col('x').mean()).over('g')            // deviation from own group's mean
   *   col('x').div(col('x').sum().over(['a', 'b']))      // share within (a, b)
   */
  over(
    partitionBy: string | string[],
    options: {
      /** Running (cumulative) aggregates in this row order — SQL `OVER (PARTITION BY … ORDER BY …)` / polars `cum_sum().over()`. */
      orderBy?: string | string[]
      descending?: boolean
    } = {},
  ): Expr<T, N> {
    const keys = Array.isArray(partitionBy) ? partitionBy : [partitionBy]
    const orderBy = options.orderBy === undefined ? undefined : Array.isArray(options.orderBy) ? options.orderBy : [options.orderBy]
    if (keys.length === 0 && !orderBy?.length) throw new Error('over() needs at least one partition column or an orderBy')
    return new Expr({
      type: 'over',
      expr: this.node,
      partitionBy: keys,
      ...(orderBy?.length ? { orderBy, descending: options.descending ?? false } : {}),
    })
  }

  asc(): { expr: Expr<T, N>; descending: boolean } {
    return { expr: this, descending: false }
  }
  desc(): { expr: Expr<T, N>; descending: boolean } {
    return { expr: this, descending: true }
  }
}

/** TypeScript value type of a dtype. */
export type DTypeValue<D extends DType> = D extends 'utf8' | 'category' ? string : D extends 'bool' ? boolean : number

/** Any expression regardless of value type / name — parameter positions that only need the plan node. */
export type AnyExpr = Expr<any, any>

function toExpr(v: AnyExpr | number | string | boolean | null): AnyExpr {
  if (v instanceof Expr) return v
  return lit(v)
}

function bin(
  op: Extract<ExprNode, { type: 'binary' }>['op'],
  left: AnyExpr,
  right: AnyExpr | number | string | boolean | null,
): Expr<any, any> {
  return new Expr({ type: 'binary', op, left: left.node, right: toExpr(right).node })
}

/**
 * Column reference by name. Untyped (`Expr<any>`): nothing is known about a column named by a string.
 * For compile-time checked names and value types use the column refs a typed frame hands to callbacks
 * (`df.filter((c) => c.age.gt(18))`) or `cols<Schema>()`.
 */
export function col<M extends string = string>(name: M): Expr<any, M> {
  return new Expr({ type: 'col', name })
}

export function lit<V extends number | string | boolean | null>(value: V): Expr<V> {
  return new Expr({ type: 'lit', value })
}

/** Typed column references for a schema: `const c = cols<{ age: number; city: string }>(); c.age.gt(18)`. */
export function cols<S extends Record<string, unknown>>(): ColRefs<S> {
  return new Proxy({} as ColRefs<S>, {
    get: (_target, key) => (typeof key === 'string' ? col(key) : undefined),
    has: () => true,
  })
}

/** One `Expr<S[K], K>` per column of a schema. */
export type ColRefs<S> = { readonly [K in keyof S & string]: Expr<S[K], K> }

export function aggExpr(op: AggKind, name: string, q?: number): Expr<number> {
  return new Expr({ type: 'agg', op, expr: { type: 'col', name }, q })
}

type ValueOf<V> = V extends Expr<infer T, any> ? T : V

export class WhenBuilder<V = never> {
  private branches: Array<{ when: AnyExpr; then: AnyExpr }> = []
  private pendingWhen: AnyExpr | null = null

  constructor(predicate?: AnyExpr) {
    if (predicate) this.pendingWhen = predicate
  }

  then<W extends AnyExpr | number | string | boolean | null>(value: W): WhenThenBuilder<V | ValueOf<W>> {
    if (!this.pendingWhen) throw new Error('when().then() called without predicate')
    const pred = this.pendingWhen
    this.pendingWhen = null
    return new WhenThenBuilder<V | ValueOf<W>>(this as WhenBuilder<any>, pred, toExpr(value))
  }

  /** @internal */
  setPending(predicate: AnyExpr): void {
    this.pendingWhen = predicate
  }

  /** @internal */
  push(whenExpr: AnyExpr, thenExpr: AnyExpr): void {
    this.branches.push({ when: whenExpr, then: thenExpr })
  }

  /** @internal */
  build(otherwise: AnyExpr): Expr<any> {
    return new Expr({
      type: 'when',
      branches: this.branches.map((b) => ({ when: b.when.node, then: b.then.node })),
      otherwise: otherwise.node,
    })
  }
}

export class WhenThenBuilder<V = never> {
  constructor(
    private root: WhenBuilder<any>,
    private pred: AnyExpr,
    private thenExpr: AnyExpr,
  ) {}

  when(predicate: Expr<boolean, any> | Expr<any, any>): WhenBuilder<V> {
    this.root.push(this.pred, this.thenExpr)
    this.root.setPending(predicate)
    return this.root as WhenBuilder<V>
  }

  otherwise<W extends AnyExpr | number | string | boolean | null>(value: W): Expr<V | ValueOf<W>> {
    this.root.push(this.pred, this.thenExpr)
    return this.root.build(toExpr(value)) as Expr<V | ValueOf<W>>
  }
}

/** polars-style `when(pred).then(v).otherwise(v)` — the result type is the union of the branch types. */
export function when(predicate: Expr<boolean, any> | Expr<any, any>): WhenBuilder {
  return new WhenBuilder(predicate)
}
