import type { DtOp } from '@columna/runtime'
import { Expr } from '../expr.js'

export class DtNamespace {
  constructor(private readonly expr: Expr) {}

  year(): Expr {
    return this.op('year')
  }
  month(): Expr {
    return this.op('month')
  }
  day(): Expr {
    return this.op('day')
  }
  hour(): Expr {
    return this.op('hour')
  }
  minute(): Expr {
    return this.op('minute')
  }
  second(): Expr {
    return this.op('second')
  }
  weekday(): Expr {
    return this.op('weekday')
  }
  epochMillis(): Expr {
    return this.op('epochMillis')
  }

  private op(op: DtOp): Expr {
    return new Expr({ type: 'dt', op, expr: this.expr.node })
  }
}
