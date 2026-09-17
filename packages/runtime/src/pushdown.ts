import type { ExprNode, PlanNode } from './types.js'

/** Collect column names referenced by an expression (best-effort). */
export function exprColumnRefs(expr: ExprNode, out = new Set<string>()): Set<string> {
  switch (expr.type) {
    case 'col':
      out.add(expr.name)
      break
    case 'alias':
    case 'cast':
    case 'fillNull':
    case 'agg':
    case 'unary':
    case 'dt':
    case 'clip':
    case 'mapElements':
      exprColumnRefs((expr as { expr: ExprNode }).expr, out)
      break
    case 'binary':
      exprColumnRefs(expr.left, out)
      exprColumnRefs(expr.right, out)
      break
    case 'when':
      for (const w of expr.branches) {
        exprColumnRefs(w.when, out)
        exprColumnRefs(w.then, out)
      }
      exprColumnRefs(expr.otherwise, out)
      break
    case 'isIn':
      exprColumnRefs(expr.expr, out)
      break
    case 'isBetween':
      exprColumnRefs(expr.expr, out)
      break
    case 'str':
      exprColumnRefs(expr.expr, out)
      break
    case 'rowOffset':
      exprColumnRefs(expr.expr, out)
      break
    case 'over':
      exprColumnRefs(expr.expr, out)
      for (const p of expr.partitionBy ?? []) out.add(p)
      for (const o of expr.orderBy ?? []) out.add(o)
      break
    default:
      break
  }
  return out
}

function projectIsColumnSubset(columns: Array<string | ExprNode>): string[] | null {
  const names: string[] = []
  for (const c of columns) {
    if (typeof c === 'string') names.push(c)
    else if (c.type === 'col') names.push(c.name)
    else if (c.type === 'alias' && c.expr.type === 'col') names.push(c.expr.name)
    else return null
  }
  return names
}

function mapInput(plan: PlanNode, input: PlanNode): PlanNode {
  return { ...(plan as object), input } as PlanNode
}

/**
 * Light projection pushdown: move simple column `project` nodes below sort / through
 * filter, and prune join inputs to columns needed by the projection + join keys.
 */
export function pushdownProjections(plan: PlanNode): PlanNode {
  switch (plan.type) {
    case 'project': {
      const input = pushdownProjections(plan.input)
      const names = projectIsColumnSubset(plan.columns)
      if (!names) return { ...plan, input }

      if (input.type === 'sort') {
        const sortCols = new Set<string>()
        for (const k of input.by) exprColumnRefs(k.expr, sortCols)
        const need = [...new Set([...names, ...sortCols])]
        return {
          type: 'sort',
          input: pushdownProjections({ type: 'project', input: input.input, columns: need }),
          by: input.by,
        }
      }

      if (input.type === 'filter') {
        const predCols = exprColumnRefs(input.predicate)
        const need = [...new Set([...names, ...predCols])]
        const pruned: PlanNode = {
          type: 'filter',
          input: pushdownProjections({ type: 'project', input: input.input, columns: need }),
          predicate: input.predicate,
        }
        return { type: 'project', input: pruned, columns: plan.columns }
      }

      if (input.type === 'join' && input.how !== 'cross') {
        const leftNeed = [...new Set([...input.leftOn, ...names])]
        const rightNeed = [...new Set([...input.rightOn, ...names])]
        return {
          type: 'project',
          input: {
            type: 'join',
            left: pushdownProjections({ type: 'project', input: input.left, columns: leftNeed }),
            right: pushdownProjections({ type: 'project', input: input.right, columns: rightNeed }),
            leftOn: input.leftOn,
            rightOn: input.rightOn,
            how: input.how,
          },
          columns: plan.columns,
        }
      }

      return { ...plan, input }
    }
    case 'scan':
      return plan
    case 'concat':
      return { ...plan, frames: plan.frames.map(pushdownProjections) }
    case 'join':
    case 'asofJoin':
      return {
        ...plan,
        left: pushdownProjections(plan.left),
        right: pushdownProjections(plan.right),
      }
    default: {
      if ('input' in plan && plan.input) {
        return mapInput(plan, pushdownProjections(plan.input))
      }
      return plan
    }
  }
}
