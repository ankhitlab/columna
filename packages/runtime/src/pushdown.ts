import type { ExprNode, PlanNode } from './types.js'
import { collectLeafTables } from './types.js'

/** Collect every column referenced by an expression.
 *
 * IMPORTANT: this function is used for correctness-sensitive optimizer decisions.
 * A newly added ExprNode variant must be handled explicitly.
 */
export function exprColumnRefs(expr: ExprNode, out = new Set<string>()): Set<string> {
  switch (expr.type) {
    case 'col':
      out.add(expr.name)
      return out

    case 'lit':
      return out

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
      return exprColumnRefs(expr.expr, out)

    case 'binary':
      exprColumnRefs(expr.left, out)
      exprColumnRefs(expr.right, out)
      return out

    case 'when':
      for (const branch of expr.branches) {
        exprColumnRefs(branch.when, out)
        exprColumnRefs(branch.then, out)
      }
      exprColumnRefs(expr.otherwise, out)
      return out

    case 'isBetween':
      exprColumnRefs(expr.expr, out)
      exprColumnRefs(expr.low, out)
      exprColumnRefs(expr.high, out)
      return out

    case 'str':
      exprColumnRefs(expr.expr, out)
      if (expr.other) exprColumnRefs(expr.other, out)
      return out

    case 'over':
      exprColumnRefs(expr.expr, out)
      for (const name of expr.partitionBy ?? []) out.add(name)
      for (const name of expr.orderBy ?? []) out.add(name)
      return out

    default: {
      const exhaustive: never = expr
      return exhaustive
    }
  }
}

/**
 * Pure projection that preserves column identity and names.
 *
 * Aliases intentionally return null. Supporting aliases safely requires
 * source→output provenance, not just string column names.
 */
function projectIsColumnSubset(columns: Array<string | ExprNode>): string[] | null {
  const names: string[] = []

  for (const column of columns) {
    if (typeof column === 'string') {
      names.push(column)
      continue
    }

    if (column.type === 'col') {
      names.push(column.name)
      continue
    }

    return null
  }

  return names
}

/** Simple column-subset project → output names (or null if complex / aliased). */
export function projectColumnSubset(columns: Array<string | ExprNode>): string[] | null {
  return projectIsColumnSubset(columns)
}

function mapInput(plan: PlanNode, input: PlanNode): PlanNode {
  return { ...(plan as object), input } as PlanNode
}

/** Best-effort column names available from leaf scans (misses withColumn/rename). */
export function leafColumnNames(plan: PlanNode): Set<string> {
  const names = new Set<string>()
  for (const t of collectLeafTables(plan)) {
    for (const f of t.schema) names.add(f.name)
  }
  return names
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
        const sorted: PlanNode = {
          type: 'sort',
          input: pushdownProjections({ type: 'project', input: input.input, columns: need }),
          by: input.by,
        }
        // Keep the original projection so sort-only columns are not exposed.
        return { type: 'project', input: sorted, columns: plan.columns }
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
        const leftCols = leafColumnNames(input.left)
        const rightCols = leafColumnNames(input.right)
        const lSuffix = input.lSuffix ?? ''
        const rSuffix = input.rSuffix ?? '_right'
        const leftNeed = new Set(input.leftOn)
        const rightNeed = new Set(input.rightOn)
        let classified = true
        for (const name of names) {
          let placed = false
          if (leftCols.has(name)) {
            leftNeed.add(name)
            placed = true
          }
          if (lSuffix && name.endsWith(lSuffix)) {
            const base = name.slice(0, -lSuffix.length)
            if (base && leftCols.has(base) && rightCols.has(base)) {
              leftNeed.add(base)
              placed = true
            }
          }
          if (rSuffix && name.endsWith(rSuffix)) {
            const base = name.slice(0, -rSuffix.length)
            if (base && rightCols.has(base)) {
              rightNeed.add(base)
              placed = true
            }
          }
          if (!leftCols.has(name) && rightCols.has(name)) {
            rightNeed.add(name)
            placed = true
          }
          if (input.leftOn.includes(name) || input.rightOn.includes(name)) {
            if (input.leftOn.includes(name)) leftNeed.add(name)
            if (input.rightOn.includes(name)) rightNeed.add(name)
            placed = true
          }
          if (!placed) {
            classified = false
            break
          }
        }
        if (classified) {
          return {
            type: 'project',
            input: {
              type: 'join',
              left: pushdownProjections({ type: 'project', input: input.left, columns: [...leftNeed] }),
              right: pushdownProjections({ type: 'project', input: input.right, columns: [...rightNeed] }),
              leftOn: input.leftOn,
              rightOn: input.rightOn,
              how: input.how,
              lSuffix: input.lSuffix,
              rSuffix: input.rSuffix,
              validate: input.validate,
            },
            columns: plan.columns,
          }
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
