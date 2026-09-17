import type { ExprNode, StrOp } from '@columna/runtime'
import { Expr, type AnyExpr } from '../expr.js'

export class StrNamespace {
  constructor(private readonly expr: AnyExpr) {}

  len(): Expr<number> {
    return this.op('len')
  }
  toLowerCase(): Expr<string> {
    return this.op('toLowerCase')
  }
  toUpperCase(): Expr<string> {
    return this.op('toUpperCase')
  }
  trim(): Expr<string> {
    return this.op('trim')
  }
  contains(pattern: string): Expr<boolean> {
    return this.op('contains', { pattern })
  }
  startsWith(pattern: string): Expr<boolean> {
    return this.op('startsWith', { pattern })
  }
  endsWith(pattern: string): Expr<boolean> {
    return this.op('endsWith', { pattern })
  }
  replace(pattern: string, replacement: string): Expr<string> {
    return this.op('replace', { pattern, replacement })
  }
  replaceAll(pattern: string, replacement: string): Expr<string> {
    return this.op('replaceAll', { pattern, replacement })
  }
  slice(start: number, end?: number): Expr<string> {
    return this.op('slice', { start, end })
  }
  /** Split into JSON-array string (use `explode` to expand rows). */
  split(pattern = ','): Expr<string> {
    return this.op('split', { pattern })
  }
  /** Concatenate with another expression or literal string: `col('a').str.concat(col('b'), ' ')`. */
  concat(other: AnyExpr | string, separator = ''): Expr<string> {
    const node: ExprNode = typeof other === 'string' ? { type: 'lit', value: other } : other.node
    return this.op('concat', { other: node, separator })
  }
  /** Pad on the left to `length` with `fill` (default space). */
  padStart(length: number, fill = ' '): Expr<string> {
    return this.op('padStart', { length, fill })
  }
  /** Pad on the right to `length` with `fill` (default space). */
  padEnd(length: number, fill = ' '): Expr<string> {
    return this.op('padEnd', { length, fill })
  }

  private op<R = string>(
    op: StrOp,
    extra: Partial<Extract<ExprNode, { type: 'str' }>> = {},
  ): Expr<R> {
    return new Expr({ type: 'str', op, expr: this.expr.node, ...extra })
  }
}
