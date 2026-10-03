import type { ExprNode, PlanNode } from './types.js'
import { collectLeafTables } from './types.js'
import { exprColumnRefs } from './expr_walk.js'
export { exprColumnRefs } from './expr_walk.js'

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

/** Exact output names for plan shapes whose naming rules are known here. Unknown is deliberately
 * not approximated by leaf scans: that loses rename/drop/withColumn provenance and can corrupt a join.
 */
function outputColumnNames(plan: PlanNode): Set<string> | null {
  switch (plan.type) {
    case 'scan':
      return new Set(plan.table.schema.map((field) => field.name))
    case 'project':
      return new Set(plan.columns.map((column, i) => typeof column === 'string' ? column
        : column.type === 'col' || column.type === 'alias' ? column.name
          : column.type === 'agg' && column.expr.type === 'col' ? `${column.op}_${column.expr.name}` : `expr_${i}`))
    case 'groupBy':
      return new Set([...plan.keys, ...plan.aggs.map((agg) => agg.name)])
    case 'rename': {
      const input = outputColumnNames(plan.input)
      return input && new Set([...input].map((name) => plan.mapping[name] ?? name))
    }
    case 'drop': {
      const input = outputColumnNames(plan.input)
      if (input) for (const name of plan.columns) input.delete(name)
      return input
    }
    case 'withColumn':
    case 'window':
    case 'rolling':
    case 'expanding': {
      const input = outputColumnNames(plan.input)
      if (input) input.add(plan.name)
      return input
    }
    case 'withColumns': {
      const input = outputColumnNames(plan.input)
      if (input) for (const column of plan.columns) input.add(column.name)
      return input
    }
    case 'filter':
    case 'sort':
    case 'limit':
    case 'slice':
    case 'take':
    case 'sample':
    case 'fillNull':
    case 'ffill':
    case 'bfill':
    case 'dropNull':
    case 'unique':
    case 'interpolate':
      return outputColumnNames(plan.input)
    default:
      // Nested joins, dynamic pivots/unnests and concatenation need richer provenance.
      return null
  }
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
        const leftCols = outputColumnNames(input.left)
        const rightCols = outputColumnNames(input.right)
        if (!leftCols || !rightCols) return { ...plan, input }
        // Dropping either side of a collision changes suffix allocation. Until that provenance is
        // modelled, retain both branches even when the requested name itself has no suffix.
        const hasCollision = [...leftCols].some((name) => rightCols.has(name)
          && !input.leftOn.some((key, i) => key === name && input.rightOn[i] === name))
        if (hasCollision) return { ...plan, input }
        const lSuffix = input.lSuffix ?? ''
        const rSuffix = input.rSuffix ?? '_right'

        // Hotfix: never prune join children when the projection asks for a
        // collision/suffix output name. Stripping `_right` and dropping the
        // left collidee changes the join schema (score_right → score).
        // Full provenance (OutputField) is the long-term fix.
        const asksCollisionSuffix = names.some((name) => {
          if (rSuffix && name.endsWith(rSuffix)) {
            const base = name.slice(0, -rSuffix.length)
            if (base && leftCols.has(base) && rightCols.has(base)) return true
          }
          if (lSuffix && name.endsWith(lSuffix)) {
            const base = name.slice(0, -lSuffix.length)
            if (base && leftCols.has(base) && rightCols.has(base)) return true
          }
          return false
        })
        if (asksCollisionSuffix) return { ...plan, input }

        const leftNeed = new Set(input.leftOn)
        const rightNeed = new Set(input.rightOn)
        let classified = true
        for (const name of names) {
          let placed = false
          if (leftCols.has(name)) {
            leftNeed.add(name)
            placed = true
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
