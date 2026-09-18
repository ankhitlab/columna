/**
 * Shared ExprNode walk / rewrite primitives.
 *
 * New ExprNode variants must be handled in `forEachChildExpr` / `mapExprChildren`
 * (exhaustiveness via `never`) so columnRefs, pushdown barriers, and renames stay aligned.
 */
import type { ExprNode } from './types.js'

/** Visit every direct child expression (not the node itself). */
export function forEachChildExpr(expr: ExprNode, visit: (child: ExprNode) => void): void {
  switch (expr.type) {
    case 'col':
    case 'lit':
      return

    case 'alias':
    case 'cast':
    case 'fillNull':
    case 'agg':
    case 'unary':
    case 'dt':
    case 'clip':
    case 'mapElements':
    case 'isIn':
    case 'rowOffset':
    case 'over':
      visit(expr.expr)
      return

    case 'binary':
      visit(expr.left)
      visit(expr.right)
      return

    case 'when':
      for (const branch of expr.branches) {
        visit(branch.when)
        visit(branch.then)
      }
      visit(expr.otherwise)
      return

    case 'isBetween':
      visit(expr.expr)
      visit(expr.low)
      visit(expr.high)
      return

    case 'str':
      visit(expr.expr)
      if (expr.other) visit(expr.other)
      return

    default: {
      const exhaustive: never = expr
      return exhaustive
    }
  }
}

/**
 * Rebuild `expr` with each child replaced by `map(child)`.
 * Structural sharing when every child is unchanged (===).
 */
export function mapExprChildren(expr: ExprNode, map: (child: ExprNode) => ExprNode): ExprNode {
  switch (expr.type) {
    case 'col':
    case 'lit':
      return expr

    case 'alias':
    case 'cast':
    case 'fillNull':
    case 'agg':
    case 'unary':
    case 'dt':
    case 'clip':
    case 'mapElements':
    case 'isIn':
    case 'rowOffset':
    case 'over': {
      const inner = map(expr.expr)
      return inner === expr.expr ? expr : ({ ...expr, expr: inner } as ExprNode)
    }

    case 'binary': {
      const left = map(expr.left)
      const right = map(expr.right)
      return left === expr.left && right === expr.right ? expr : { ...expr, left, right }
    }

    case 'when': {
      let changed = false
      const branches = expr.branches.map((b) => {
        const when = map(b.when)
        const then = map(b.then)
        if (when !== b.when || then !== b.then) changed = true
        return when === b.when && then === b.then ? b : { when, then }
      })
      const otherwise = map(expr.otherwise)
      if (otherwise !== expr.otherwise) changed = true
      return changed ? { ...expr, branches, otherwise } : expr
    }

    case 'isBetween': {
      const inner = map(expr.expr)
      const low = map(expr.low)
      const high = map(expr.high)
      return inner === expr.expr && low === expr.low && high === expr.high
        ? expr
        : { ...expr, expr: inner, low, high }
    }

    case 'str': {
      const inner = map(expr.expr)
      const other = expr.other ? map(expr.other) : undefined
      return inner === expr.expr && other === expr.other
        ? expr
        : { ...expr, expr: inner, other }
    }

    default: {
      const exhaustive: never = expr
      return exhaustive
    }
  }
}

/** Depth-first pre-order visit of every node in the tree. */
export function visitExpr(expr: ExprNode, visit: (node: ExprNode) => void): void {
  visit(expr)
  forEachChildExpr(expr, (child) => visitExpr(child, visit))
}

/** True if `pred` holds for this node or any descendant. */
export function exprSome(expr: ExprNode, pred: (node: ExprNode) => boolean): boolean {
  if (pred(expr)) return true
  let found = false
  forEachChildExpr(expr, (child) => {
    if (!found && exprSome(child, pred)) found = true
  })
  return found
}

/** Collect every column referenced by an expression. */
export function exprColumnRefs(expr: ExprNode, out = new Set<string>()): Set<string> {
  visitExpr(expr, (node) => {
    if (node.type === 'col') out.add(node.name)
    if (node.type === 'over') {
      for (const name of node.partitionBy ?? []) out.add(name)
      for (const name of node.orderBy ?? []) out.add(name)
    }
  })
  return out
}

/**
 * True if expr must not be merged/pushed past filters or withColumn.
 * Barriers: mapElements, over, whole-column aggregates, and row-position ops (shift/offset).
 */
export function exprBlocksPushdown(expr: ExprNode): boolean {
  return exprSome(
    expr,
    (node) =>
      node.type === 'mapElements' ||
      node.type === 'over' ||
      node.type === 'agg' ||
      node.type === 'rowOffset',
  )
}

/** Rename column refs in an expression tree (does not rewrite over partition/order strings). */
export function renameColsInExpr(expr: ExprNode, mapName: (name: string) => string): ExprNode {
  if (expr.type === 'col') {
    const n = mapName(expr.name)
    return n === expr.name ? expr : { type: 'col', name: n }
  }
  return mapExprChildren(expr, (child) => renameColsInExpr(child, mapName))
}
