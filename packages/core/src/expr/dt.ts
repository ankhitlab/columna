import type { DtOp } from '@columna/runtime'
import { Expr, type AnyExpr } from '../expr.js'

export class DtNamespace {
  constructor(private readonly expr: AnyExpr) {}

  year(): Expr<number> {
    return this.op('year')
  }
  month(): Expr<number> {
    return this.op('month')
  }
  day(): Expr<number> {
    return this.op('day')
  }
  hour(): Expr<number> {
    return this.op('hour')
  }
  minute(): Expr<number> {
    return this.op('minute')
  }
  second(): Expr<number> {
    return this.op('second')
  }
  weekday(): Expr<number> {
    return this.op('weekday')
  }
  epochMillis(): Expr<number> {
    return this.op('epochMillis')
  }

  private op(op: DtOp): Expr<number> {
    return new Expr({ type: 'dt', op, expr: this.expr.node })
  }
}
