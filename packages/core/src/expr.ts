import type { AggKind, ExprNode, QuantileMethod } from '@columna/runtime'
import type { DType } from '@columna/arrow'

const DTYPES: ReadonlySet<DType> = new Set<DType>(['f64', 'f32', 'i32', 'u32', 'bool', 'utf8', 'category', 'datetime'])
import { StrNamespace } from './expr/str.js'
import { DtNamespace } from './expr/dt.js'

export class Expr {
  constructor(readonly node: ExprNode) {}

  get str(): StrNamespace {
    return new StrNamespace(this)
  }

  get dt(): DtNamespace {
    return new DtNamespace(this)
  }

  alias(name: string): Expr {
    return new Expr({ type: 'alias', expr: this.node, name })
  }

  cast(dtype: DType): Expr {
    if (!DTYPES.has(dtype)) throw new RangeError(`cast: unknown dtype "${String(dtype)}" (expected ${[...DTYPES].join(' | ')})`)
    return new Expr({ type: 'cast', expr: this.node, dtype })
  }

  fillNull(value: number | string | boolean): Expr {
    return new Expr({ type: 'fillNull', expr: this.node, value })
  }

  isNull(): Expr {
    return new Expr({ type: 'unary', op: 'isNull', expr: this.node })
  }

  isNotNull(): Expr {
    return new Expr({ type: 'unary', op: 'isNotNull', expr: this.node })
  }

  not(): Expr {
    return new Expr({ type: 'unary', op: 'not', expr: this.node })
  }

  abs(): Expr {
    return new Expr({ type: 'unary', op: 'abs', expr: this.node })
  }

  neg(): Expr {
    return new Expr({ type: 'unary', op: 'neg', expr: this.node })
  }

  // Element-wise math (null → null, domain errors → NaN). Run on the typed fast path.
  sqrt(): Expr {
    return new Expr({ type: 'unary', op: 'sqrt', expr: this.node })
  }

  /** Natural log by default; pass `base` for log_b (e.g. `log(10)`). */
  log(base?: number): Expr {
    return new Expr({ type: 'unary', op: 'log', expr: this.node, base })
  }

  log10(): Expr {
    return new Expr({ type: 'unary', op: 'log10', expr: this.node })
  }

  log2(): Expr {
    return new Expr({ type: 'unary', op: 'log2', expr: this.node })
  }

  exp(): Expr {
    return new Expr({ type: 'unary', op: 'exp', expr: this.node })
  }

  /** Round half away from zero (Minitab / Excel ROUND) to `decimals` places. */
  round(decimals = 0): Expr {
    return new Expr({ type: 'unary', op: 'round', expr: this.node, decimals })
  }

  floor(): Expr {
    return new Expr({ type: 'unary', op: 'floor', expr: this.node })
  }

  ceil(): Expr {
    return new Expr({ type: 'unary', op: 'ceil', expr: this.node })
  }

  sign(): Expr {
    return new Expr({ type: 'unary', op: 'sign', expr: this.node })
  }

  pow(other: Expr | number): Expr {
    return bin('pow', this, other)
  }

  eq(other: Expr | number | string | boolean | null): Expr {
    return bin('eq', this, other)
  }
  neq(other: Expr | number | string | boolean | null): Expr {
    return bin('neq', this, other)
  }
  gt(other: Expr | number | string | boolean | null): Expr {
    return bin('gt', this, other)
  }
  gte(other: Expr | number | string | boolean | null): Expr {
    return bin('gte', this, other)
  }
  lt(other: Expr | number | string | boolean | null): Expr {
    return bin('lt', this, other)
  }
  lte(other: Expr | number | string | boolean | null): Expr {
    return bin('lte', this, other)
  }
  and(other: Expr | boolean): Expr {
    return bin('and', this, other)
  }
  or(other: Expr | boolean): Expr {
    return bin('or', this, other)
  }
  add(other: Expr | number): Expr {
    return bin('add', this, other)
  }
  sub(other: Expr | number): Expr {
    return bin('sub', this, other)
  }
  mul(other: Expr | number): Expr {
    return bin('mul', this, other)
  }
  div(other: Expr | number): Expr {
    return bin('div', this, other)
  }
  mod(other: Expr | number): Expr {
    return bin('mod', this, other)
  }

  isIn(values: Array<number | string | boolean | null>): Expr {
    return new Expr({ type: 'isIn', expr: this.node, values })
  }

  isBetween(
    low: Expr | number | string,
    high: Expr | number | string,
    closed: 'both' | 'left' | 'right' | 'neither' = 'both',
  ): Expr {
    return new Expr({
      type: 'isBetween',
      expr: this.node,
      low: toExpr(low).node,
      high: toExpr(high).node,
      closed,
    })
  }

  clip(min?: number, max?: number): Expr {
    return new Expr({ type: 'clip', expr: this.node, min, max })
  }

  shift(periods = 1): Expr {
    return new Expr({ type: 'rowOffset', expr: this.node, periods, kind: 'shift' })
  }

  diff(periods = 1): Expr {
    return new Expr({ type: 'rowOffset', expr: this.node, periods, kind: 'diff' })
  }

  pctChange(periods = 1): Expr {
    return new Expr({ type: 'rowOffset', expr: this.node, periods, kind: 'pctChange' })
  }

  mapElements(fn: (value: number | string | boolean | null) => number | string | boolean | null): Expr {
    return new Expr({ type: 'mapElements', expr: this.node, fn })
  }

  sum(): Expr {
    return new Expr({ type: 'agg', op: 'sum', expr: this.node })
  }
  mean(): Expr {
    return new Expr({ type: 'agg', op: 'mean', expr: this.node })
  }
  min(): Expr {
    return new Expr({ type: 'agg', op: 'min', expr: this.node })
  }
  max(): Expr {
    return new Expr({ type: 'agg', op: 'max', expr: this.node })
  }
  count(): Expr {
    return new Expr({ type: 'agg', op: 'count', expr: this.node })
  }
  nunique(): Expr {
    return new Expr({ type: 'agg', op: 'nunique', expr: this.node })
  }
  first(): Expr {
    return new Expr({ type: 'agg', op: 'first', expr: this.node })
  }
  last(): Expr {
    return new Expr({ type: 'agg', op: 'last', expr: this.node })
  }
  std(): Expr {
    return new Expr({ type: 'agg', op: 'std', expr: this.node })
  }
  var(): Expr {
    return new Expr({ type: 'agg', op: 'var', expr: this.node })
  }
  /**
   * Median. `method`: 'linear' (default; type 7, pandas / polars) or 'minitab' (type 6, position p(n + 1)) —
   * both give the middle value / midpoint for the median, the choice matters for other quantiles.
   */
  median(method?: QuantileMethod): Expr {
    return new Expr({ type: 'agg', op: 'median', expr: this.node, ...(method ? { qm: method } : {}) })
  }
  /**
   * Sample quantile q ∈ [0, 1].
   *   col('x').quantile(0.25)             // type 7: position (n − 1)q, linear interpolation (pandas default)
   *   col('x').quantile(0.25, 'minitab')  // type 6: position q(n + 1), clamped to the sample — Minitab / SPSS / R type=6
   */
  quantile(q: number, method?: QuantileMethod): Expr {
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
  ): Expr {
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

  asc(): { expr: Expr; descending: boolean } {
    return { expr: this, descending: false }
  }
  desc(): { expr: Expr; descending: boolean } {
    return { expr: this, descending: true }
  }
}

function toExpr(v: Expr | number | string | boolean | null): Expr {
  if (v instanceof Expr) return v
  return lit(v)
}

function bin(
  op: Extract<ExprNode, { type: 'binary' }>['op'],
  left: Expr,
  right: Expr | number | string | boolean | null,
): Expr {
  return new Expr({ type: 'binary', op, left: left.node, right: toExpr(right).node })
}

export function col(name: string): Expr {
  return new Expr({ type: 'col', name })
}

export function lit(value: number | string | boolean | null): Expr {
  return new Expr({ type: 'lit', value })
}

export function aggExpr(op: AggKind, name: string, q?: number): Expr {
  return new Expr({ type: 'agg', op, expr: { type: 'col', name }, q })
}

export class WhenBuilder {
  private branches: Array<{ when: Expr; then: Expr }> = []
  private pendingWhen: Expr | null = null

  constructor(predicate?: Expr) {
    if (predicate) this.pendingWhen = predicate
  }

  then(value: Expr | number | string | boolean | null): WhenThenBuilder {
    if (!this.pendingWhen) throw new Error('when().then() called without predicate')
    const pred = this.pendingWhen
    this.pendingWhen = null
    return new WhenThenBuilder(this, pred, toExpr(value))
  }

  /** @internal */
  setPending(predicate: Expr): void {
    this.pendingWhen = predicate
  }

  /** @internal */
  push(whenExpr: Expr, thenExpr: Expr): void {
    this.branches.push({ when: whenExpr, then: thenExpr })
  }

  /** @internal */
  build(otherwise: Expr): Expr {
    return new Expr({
      type: 'when',
      branches: this.branches.map((b) => ({ when: b.when.node, then: b.then.node })),
      otherwise: otherwise.node,
    })
  }
}

export class WhenThenBuilder {
  constructor(
    private root: WhenBuilder,
    private pred: Expr,
    private thenExpr: Expr,
  ) {}

  when(predicate: Expr): WhenBuilder {
    this.root.push(this.pred, this.thenExpr)
    this.root.setPending(predicate)
    return this.root
  }

  otherwise(value: Expr | number | string | boolean | null): Expr {
    this.root.push(this.pred, this.thenExpr)
    return this.root.build(toExpr(value))
  }
}

/** polars-style `when(pred).then(v).otherwise(v)` */
export function when(predicate: Expr): WhenBuilder {
  return new WhenBuilder(predicate)
}
