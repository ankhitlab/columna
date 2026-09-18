/**
 * Rule-based logical plan rewrite (not cost-based).
 * Deterministic, idempotent within a few rounds. Physical fuse stays in cpu/fast.
 */
import type { ExprNode, PlanNode } from './types.js'
import {
  exprColumnRefs,
  leafColumnNames,
  projectColumnSubset,
  pushdownProjections,
} from './pushdown.js'
import { estimatePlanRows, splitAnd } from './stats.js'

export { estimatePlanRows, splitAnd } from './stats.js'

const MAX_ROUNDS = 5

function andExpr(left: ExprNode, right: ExprNode): ExprNode {
  return { type: 'binary', op: 'and', left, right }
}

function combineAnd(parts: ExprNode[]): ExprNode | null {
  if (parts.length === 0) return null
  let acc = parts[0]!
  for (let i = 1; i < parts.length; i++) acc = andExpr(acc, parts[i]!)
  return acc
}

/**
 * True if expr must not be merged/pushed past filters or withColumn.
 * Barriers: mapElements, over, whole-column aggregates, and row-position ops (shift/offset).
 */
export function exprBlocksPushdown(expr: ExprNode): boolean {
  switch (expr.type) {
    case 'mapElements':
    case 'over':
    case 'agg':
    case 'rowOffset':
      return true
    case 'alias':
    case 'cast':
    case 'fillNull':
    case 'unary':
    case 'dt':
    case 'clip':
    case 'str':
    case 'isIn':
      return exprBlocksPushdown(expr.expr)
    case 'isBetween':
      return (
        exprBlocksPushdown(expr.expr) ||
        exprBlocksPushdown(expr.low) ||
        exprBlocksPushdown(expr.high)
      )
    case 'binary':
      return exprBlocksPushdown(expr.left) || exprBlocksPushdown(expr.right)
    case 'when':
      if (exprBlocksPushdown(expr.otherwise)) return true
      for (const b of expr.branches) {
        if (exprBlocksPushdown(b.when) || exprBlocksPushdown(b.then)) return true
      }
      return false
    default:
      return false
  }
}

/**
 * Best-effort output column names for a plan subtree.
 * Returns null when schema cannot be tracked (complex project exprs, melt/pivot, etc.).
 */
export function planOutputColumns(plan: PlanNode): Set<string> | null {
  const list = planOutputColumnList(plan)
  return list ? new Set(list) : null
}

/** Ordered output column names (join: left then right, matching assembleJoin). */
export function planOutputColumnList(plan: PlanNode): string[] | null {
  switch (plan.type) {
    case 'scan':
      return plan.table.schema.map((f) => f.name)
    case 'project': {
      const names = projectColumnSubset(plan.columns)
      return names
    }
    case 'drop': {
      const inner = planOutputColumnList(plan.input)
      if (!inner) return null
      const dropped = new Set(plan.columns)
      return inner.filter((n) => !dropped.has(n))
    }
    case 'rename': {
      const inner = planOutputColumnList(plan.input)
      if (!inner) return null
      return inner.map((n) => (Object.hasOwn(plan.mapping, n) ? plan.mapping[n]! : n))
    }
    case 'withColumn': {
      const inner = planOutputColumnList(plan.input)
      if (!inner) return null
      if (inner.includes(plan.name)) return [...inner]
      return [...inner, plan.name]
    }
    case 'withColumns': {
      const inner = planOutputColumnList(plan.input)
      if (!inner) return null
      const names = new Set(plan.columns.map((c) => c.name))
      const kept = inner.filter((n) => !names.has(n))
      return [...kept, ...plan.columns.map((c) => c.name)]
    }
    case 'groupBy':
      return [...plan.keys, ...plan.aggs.map((a) => a.name)]
    case 'join': {
      if (plan.how === 'semi' || plan.how === 'anti') return planOutputColumnList(plan.left)
      const left = planOutputColumnList(plan.left)
      const right = planOutputColumnList(plan.right)
      if (!left || !right) {
        const l = [...leafColumnNames(plan.left)]
        const r = [...leafColumnNames(plan.right)]
        return [...l, ...r.filter((n) => !l.includes(n))]
      }
      const leftSet = new Set(left)
      const rightSet = new Set(right)
      const lSuffix = plan.lSuffix ?? ''
      const rSuffix = plan.rSuffix ?? '_right'
      const out: string[] = []
      for (const n of left) {
        const sharedKey = plan.leftOn.includes(n) && plan.rightOn.includes(n)
        if (sharedKey || !rightSet.has(n) || !lSuffix) out.push(n)
        else out.push(`${n}${lSuffix}`)
      }
      for (const n of right) {
        const sharedKey = plan.rightOn.includes(n) && plan.leftOn.includes(n)
        if (sharedKey) continue
        out.push(leftSet.has(n) ? `${n}${rSuffix}` : n)
      }
      return out
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
      return planOutputColumnList(plan.input)
    case 'valueCounts':
      return [plan.column, 'count']
    default:
      if ('input' in plan && plan.input) return planOutputColumnList(plan.input)
      return null
  }
}

function mapInput(plan: PlanNode, input: PlanNode): PlanNode {
  return { ...(plan as object), input } as PlanNode
}

function rewriteChildren(plan: PlanNode, rewrite: (p: PlanNode) => PlanNode): PlanNode {
  switch (plan.type) {
    case 'scan':
      return plan
    case 'concat':
      return { ...plan, frames: plan.frames.map(rewrite) }
    case 'join':
    case 'asofJoin':
      return { ...plan, left: rewrite(plan.left), right: rewrite(plan.right) }
    default:
      if ('input' in plan && plan.input) return mapInput(plan, rewrite(plan.input))
      return plan
  }
}

function renameColsInExpr(expr: ExprNode, mapName: (name: string) => string): ExprNode {
  switch (expr.type) {
    case 'col': {
      const n = mapName(expr.name)
      return n === expr.name ? expr : { type: 'col', name: n }
    }
    case 'binary':
      return {
        ...expr,
        left: renameColsInExpr(expr.left, mapName),
        right: renameColsInExpr(expr.right, mapName),
      }
    case 'unary':
    case 'alias':
    case 'cast':
    case 'fillNull':
    case 'agg':
    case 'dt':
    case 'clip':
    case 'str':
    case 'rowOffset':
    case 'isIn':
      return { ...expr, expr: renameColsInExpr(expr.expr, mapName) } as ExprNode
    case 'isBetween':
      return {
        ...expr,
        expr: renameColsInExpr(expr.expr, mapName),
        low: renameColsInExpr(expr.low, mapName),
        high: renameColsInExpr(expr.high, mapName),
      }
    case 'when':
      return {
        ...expr,
        branches: expr.branches.map((b) => ({
          when: renameColsInExpr(b.when, mapName),
          then: renameColsInExpr(b.then, mapName),
        })),
        otherwise: renameColsInExpr(expr.otherwise, mapName),
      }
    default:
      return expr
  }
}

/** R1: merge consecutive filters into one AND predicate. */
function mergeFilters(plan: PlanNode): PlanNode {
  plan = rewriteChildren(plan, mergeFilters)
  if (plan.type === 'filter' && plan.input.type === 'filter') {
    if (exprBlocksPushdown(plan.predicate) || exprBlocksPushdown(plan.input.predicate)) {
      return plan
    }
    return {
      type: 'filter',
      input: plan.input.input,
      predicate: andExpr(plan.input.predicate, plan.predicate),
    }
  }
  return plan
}

/**
 * R2: filter past simple project when predicate refs ⊆ project output names
 * (and project is column subset / alias-of-col).
 */
function filterPastProject(plan: PlanNode): PlanNode {
  plan = rewriteChildren(plan, filterPastProject)
  if (plan.type !== 'filter' || plan.input.type !== 'project') return plan
  if (exprBlocksPushdown(plan.predicate)) return plan
  const names = projectColumnSubset(plan.input.columns)
  if (!names) return plan
  const refs = exprColumnRefs(plan.predicate)
  for (const r of refs) {
    if (!names.includes(r)) return plan
  }
  const inputNames = new Set<string>()
  for (const c of plan.input.columns) {
    if (typeof c === 'string') inputNames.add(c)
    else if (c.type === 'col') inputNames.add(c.name)
    else if (c.type === 'alias' && c.expr.type === 'col') {
      if (refs.has(c.name) && c.name !== c.expr.name) return plan
      inputNames.add(c.expr.name)
    }
  }
  for (const r of refs) {
    if (!inputNames.has(r) && !names.includes(r)) return plan
  }
  return {
    type: 'project',
    columns: plan.input.columns,
    input: {
      type: 'filter',
      input: plan.input.input,
      predicate: plan.predicate,
    },
  }
}

/**
 * R3: push filter conjuncts into join children when refs are side-local.
 * Uses tracked output schemas when available, else leaf scans.
 */
function filterIntoJoin(plan: PlanNode): PlanNode {
  plan = rewriteChildren(plan, filterIntoJoin)
  if (plan.type !== 'filter' || plan.input.type !== 'join') return plan
  if (plan.input.how === 'cross' || plan.input.how === 'semi' || plan.input.how === 'anti') return plan
  if (exprBlocksPushdown(plan.predicate)) return plan

  const join = plan.input
  // Outer joins invent nulls on the non-preserved side; filters on that side must stay post-join.
  const canPushLeft = join.how === 'inner' || join.how === 'left'
  const canPushRight = join.how === 'inner' || join.how === 'right'
  const leftCols = planOutputColumns(join.left)
  const rightCols = planOutputColumns(join.right)

  // Unknown provenance => no pushdown.
  // leafColumnNames() is not safe here because rename/project may have changed names.
  if (!leftCols || !rightCols) return plan

  const resolveSide = (name: string): 'left' | 'right' | 'both' | 'unknown' => {
    const onLeft = leftCols.has(name)
    const onRight = rightCols.has(name)

    if (onLeft && onRight) return 'both'
    if (onLeft) return 'left'
    if (onRight) return 'right'

    // This includes synthetic `foo_right` / `foo_left` output names.
    // Keep those predicates above the join until explicit provenance metadata exists.
    return 'unknown'
  }

  const toChildName = (name: string): string => name

  const leftParts: ExprNode[] = []
  const rightParts: ExprNode[] = []
  const topParts: ExprNode[] = []

  for (const conj of splitAnd(plan.predicate)) {
    if (exprBlocksPushdown(conj)) {
      topParts.push(conj)
      continue
    }
    const refs = [...exprColumnRefs(conj)]
    const sides = new Set(refs.map(resolveSide))
    if (sides.has('unknown') || sides.has('both') || sides.size !== 1) {
      topParts.push(conj)
      continue
    }
    if (sides.has('left')) {
      if (canPushLeft) {
        leftParts.push(renameColsInExpr(conj, toChildName))
      } else {
        topParts.push(conj)
      }
    } else if (sides.has('right')) {
      if (canPushRight) {
        rightParts.push(renameColsInExpr(conj, toChildName))
      } else {
        topParts.push(conj)
      }
    } else {
      topParts.push(conj)
    }
  }

  if (leftParts.length === 0 && rightParts.length === 0) return plan

  let left: PlanNode = join.left
  let right: PlanNode = join.right
  const leftPred = combineAnd(leftParts)
  const rightPred = combineAnd(rightParts)
  if (leftPred) left = { type: 'filter', input: left, predicate: leftPred }
  if (rightPred) right = { type: 'filter', input: right, predicate: rightPred }

  const newJoin: PlanNode = {
    type: 'join',
    left,
    right,
    leftOn: join.leftOn,
    rightOn: join.rightOn,
    how: join.how,
    lSuffix: join.lSuffix,
    rSuffix: join.rSuffix,
    validate: join.validate,
  }
  const topPred = combineAnd(topParts)
  if (topPred) return { type: 'filter', input: newJoin, predicate: topPred }
  return newJoin
}

/** R4: push filter under withColumn(s) when predicate does not reference new names. */
function filterUnderWithColumn(plan: PlanNode): PlanNode {
  plan = rewriteChildren(plan, filterUnderWithColumn)
  if (plan.type !== 'filter') return plan
  if (exprBlocksPushdown(plan.predicate)) return plan
  const refs = exprColumnRefs(plan.predicate)

  if (plan.input.type === 'withColumn') {
    if (refs.has(plan.input.name)) return plan
    if (exprBlocksPushdown(plan.input.expr)) return plan
    return {
      type: 'withColumn',
      name: plan.input.name,
      expr: plan.input.expr,
      input: { type: 'filter', input: plan.input.input, predicate: plan.predicate },
    }
  }
  if (plan.input.type === 'withColumns') {
    for (const c of plan.input.columns) {
      if (refs.has(c.name)) return plan
      if (exprBlocksPushdown(c.expr)) return plan
    }
    return {
      type: 'withColumns',
      columns: plan.input.columns,
      input: { type: 'filter', input: plan.input.input, predicate: plan.predicate },
    }
  }
  return plan
}

/** R5: limit under simple project (offset 0 only). */
function limitUnderProject(plan: PlanNode): PlanNode {
  plan = rewriteChildren(plan, limitUnderProject)
  if (plan.type !== 'limit' || plan.input.type !== 'project') return plan
  if ((plan.offset ?? 0) !== 0) return plan
  if (!projectColumnSubset(plan.input.columns)) return plan
  return {
    type: 'project',
    columns: plan.input.columns,
    input: { type: 'limit', input: plan.input.input, n: plan.n, offset: plan.offset },
  }
}

/**
 * R6 helper: push filter under sort so execute-time filter→sort→limit fuse can fire
 * (filter(sort(x)) ≡ sort(filter(x)) for pure row predicates).
 */
function filterUnderSort(plan: PlanNode): PlanNode {
  plan = rewriteChildren(plan, filterUnderSort)
  if (plan.type !== 'filter' || plan.input.type !== 'sort') return plan
  if (exprBlocksPushdown(plan.predicate)) return plan
  return {
    type: 'sort',
    by: plan.input.by,
    input: { type: 'filter', input: plan.input.input, predicate: plan.predicate },
  }
}

/** Push filter under drop when predicate does not use dropped columns. */
function filterUnderDrop(plan: PlanNode): PlanNode {
  plan = rewriteChildren(plan, filterUnderDrop)
  if (plan.type !== 'filter' || plan.input.type !== 'drop') return plan
  if (exprBlocksPushdown(plan.predicate)) return plan
  const dropped = new Set(plan.input.columns)
  for (const r of exprColumnRefs(plan.predicate)) {
    if (dropped.has(r)) return plan
  }
  return {
    type: 'drop',
    columns: plan.input.columns,
    input: { type: 'filter', input: plan.input.input, predicate: plan.predicate },
  }
}

/**
 * Push filter under rename: rewrite predicate column names to pre-rename names, then push.
 * mapping is from→to; we need to→from for the inner predicate.
 */
function filterUnderRename(plan: PlanNode): PlanNode {
  plan = rewriteChildren(plan, filterUnderRename)
  if (plan.type !== 'filter' || plan.input.type !== 'rename') return plan
  if (exprBlocksPushdown(plan.predicate)) return plan
  const reverse = new Map<string, string>()
  for (const [from, to] of Object.entries(plan.input.mapping)) {
    if (reverse.has(to)) return plan // ambiguous
    reverse.set(to, from)
  }
  const refs = exprColumnRefs(plan.predicate)
  for (const r of refs) {
    // After rename, refs use new names; unmapped names stay as-is if not a rename target.
    if (!reverse.has(r) && Object.values(plan.input.mapping).includes(r)) return plan
  }
  const innerPred = renameColsInExpr(plan.predicate, (n) => reverse.get(n) ?? n)
  return {
    type: 'rename',
    mapping: plan.input.mapping,
    input: { type: 'filter', input: plan.input.input, predicate: innerPred },
  }
}

function composeRenameMappings(
  inner: Record<string, string>,
  outer: Record<string, string>,
): Record<string, string> {
  const result: Record<string, string> = {}
  const innerTargets = new Set(Object.values(inner))

  // Original column -> name after inner -> name after outer.
  for (const [source, intermediate] of Object.entries(inner)) {
    const target = Object.hasOwn(outer, intermediate)
      ? outer[intermediate]!
      : intermediate

    if (target !== source) result[source] = target
  }

  // Outer rename of a column untouched by the inner rename.
  for (const [source, target] of Object.entries(outer)) {
    if (Object.hasOwn(inner, source)) continue
    if (innerTargets.has(source)) continue

    if (target !== source) result[source] = target
  }

  return result
}

/**
 * R7: fold adjacent project / drop / rename into fewer nodes.
 */
function foldProjectDrop(plan: PlanNode): PlanNode {
  plan = rewriteChildren(plan, foldProjectDrop)

  if (plan.type === 'project' && plan.input.type === 'project') {
    const outer = projectColumnSubset(plan.columns)
    const inner = projectColumnSubset(plan.input.columns)
    if (outer && inner) {
      const innerSet = new Set(inner)
      if (outer.every((n) => innerSet.has(n))) {
        return { type: 'project', input: plan.input.input, columns: plan.columns }
      }
    }
  }

  if (plan.type === 'project' && plan.input.type === 'drop') {
    const names = projectColumnSubset(plan.columns)
    if (names) {
      const dropped = new Set(plan.input.columns)
      if (names.every((n) => !dropped.has(n))) {
        return { type: 'project', input: plan.input.input, columns: plan.columns }
      }
    }
  }

  if (plan.type === 'drop' && plan.input.type === 'drop') {
    return {
      type: 'drop',
      input: plan.input.input,
      columns: [...new Set([...plan.input.columns, ...plan.columns])],
    }
  }

  // Compose renames: rename(rename(x))
  if (plan.type === 'rename' && plan.input.type === 'rename') {
    return {
      type: 'rename',
      mapping: composeRenameMappings(plan.input.mapping, plan.mapping),
      input: plan.input.input,
    }
  }

  return plan
}

/** R8: project over groupBy → prune groupBy input to keys + agg column refs. */
function groupByColumnPrune(plan: PlanNode): PlanNode {
  plan = rewriteChildren(plan, groupByColumnPrune)
  if (plan.type !== 'project' || plan.input.type !== 'groupBy') return plan
  const names = projectColumnSubset(plan.columns)
  if (!names) return plan
  const gb = plan.input
  const need = new Set(gb.keys)
  for (const a of gb.aggs) exprColumnRefs(a.expr, need)
  return {
    type: 'project',
    columns: plan.columns,
    input: {
      type: 'groupBy',
      keys: gb.keys,
      aggs: gb.aggs,
      input: pushdownProjections({
        type: 'project',
        input: gb.input,
        columns: [...need],
      }),
    },
  }
}

/**
 * Push limit under sort when offset===0 so execute fuse limit(sort) stays a single shape
 * after other rewrites (no new TopK node).
 */
function limitUnderSort(plan: PlanNode): PlanNode {
  // Intentionally no structural change: limit(sort) is already the fuse shape.
  // Keep pass as identity marker for documentation / future top-k plan node.
  return rewriteChildren(plan, limitUnderSort)
}

type JoinNode = Extract<PlanNode, { type: 'join' }>

function isInnerEquiJoin(plan: JoinNode): boolean {
  return (
    plan.how === 'inner' &&
    plan.leftOn.length > 0 &&
    plan.leftOn.length === plan.rightOn.length &&
    !plan.validate
  )
}

type JoinEdge = {
  leftLeaf: number
  rightLeaf: number
  leftOn: string[]
  rightOn: string[]
}

type JoinGraph = {
  leaves: PlanNode[]
  edges: JoinEdge[]
  lSuffix: string | undefined
  rSuffix: string | undefined
}

/**
 * Collect a left-deep inner equi-join spine into leaves + edges (any key names).
 * Falls back to null when a join is not a simple inner equi or provenance fails.
 */
function collectInnerJoinGraph(plan: PlanNode): JoinGraph | null {
  if (plan.type !== 'join' || !isInnerEquiJoin(plan)) return null

  const leaves: PlanNode[] = []
  const edges: JoinEdge[] = []
  let lSuffix = plan.lSuffix
  let rSuffix = plan.rSuffix

  function addLeaf(p: PlanNode): number {
    const id = leaves.length
    leaves.push(p)
    return id
  }

  /** Recursively flatten; returns leaf-id set represented by this subtree (as bitmask via Set). */
  function walk(p: PlanNode): Set<number> | null {
    if (p.type === 'join' && isInnerEquiJoin(p)) {
      if (p.lSuffix !== undefined) lSuffix = p.lSuffix
      if (p.rSuffix !== undefined) rSuffix = p.rSuffix
      const leftIds = walk(p.left)
      const rightIds = walk(p.right)
      if (!leftIds || !rightIds) return null

      // Merge all key components between the same leaf pair into one equi-edge.
      // Splitting them into separate edges would drop composite-key conjuncts on rebuild.
      const edgeByPair = new Map<string, JoinEdge>()
      for (let k = 0; k < p.leftOn.length; k++) {
        const lk = p.leftOn[k]!
        const rk = p.rightOn[k]!
        const leftLeaf = findLeafWithColumn(leaves, leftIds, lk)
        const rightLeaf = findLeafWithColumn(leaves, rightIds, rk)
        if (leftLeaf === null || rightLeaf === null) return null
        const pairKey =
          leftLeaf < rightLeaf ? `${leftLeaf}:${rightLeaf}` : `${rightLeaf}:${leftLeaf}`
        let edge = edgeByPair.get(pairKey)
        if (!edge) {
          edge = { leftLeaf, rightLeaf, leftOn: [], rightOn: [] }
          edgeByPair.set(pairKey, edge)
          edges.push(edge)
        }
        if (edge.leftLeaf === leftLeaf && edge.rightLeaf === rightLeaf) {
          edge.leftOn.push(lk)
          edge.rightOn.push(rk)
        } else {
          // Edge was oriented the other way; keep orientation, swap this component.
          edge.leftOn.push(rk)
          edge.rightOn.push(lk)
        }
      }
      const all = new Set<number>()
      for (const id of leftIds) all.add(id)
      for (const id of rightIds) all.add(id)
      return all
    }
    const id = addLeaf(p)
    return new Set([id])
  }

  const ids = walk(plan)
  if (!ids || leaves.length < 2) return null
  return { leaves, edges, lSuffix, rSuffix }
}

function findLeafWithColumn(leaves: PlanNode[], ids: Set<number>, col: string): number | null {
  for (const id of ids) {
    const cols = planOutputColumns(leaves[id]!)
    if (cols?.has(col)) return id
  }

  return null
}

function makeInnerEquiJoin(
  left: PlanNode,
  right: PlanNode,
  leftOn: string[],
  rightOn: string[],
  lSuffix: string | undefined,
  rSuffix: string | undefined,
): JoinNode {
  return {
    type: 'join',
    left,
    right,
    leftOn,
    rightOn,
    how: 'inner',
    lSuffix,
    rSuffix,
  }
}

/**
 * Preserve original join output column order after a reorder via trailing project.
 * Prefer folding into a later project→join keep pass when possible.
 */
function withOriginalJoinColumns(original: PlanNode, reordered: PlanNode): PlanNode {
  const order = planOutputColumnList(original)
  if (!order || order.length === 0) return reordered
  const have = planOutputColumns(reordered)
  if (!have || order.some((c) => !have.has(c))) return reordered
  const current = planOutputColumnList(reordered)
  if (current && current.length === order.length && current.every((c, i) => c === order[i])) {
    return reordered
  }
  return { type: 'project', columns: order, input: reordered }
}

/**
 * Fast-path kernels build a hash/dense table on the **right** and probe with the left.
 * For inner joins only: put the smaller estimated side on the right (build).
 * Left / semi / anti never swap (output schema is left-driven).
 */
function swapInnerJoinBuildSide(plan: PlanNode): PlanNode {
  plan = rewriteChildren(plan, swapInnerJoinBuildSide)
  if (plan.type !== 'join') return plan
  // Explicit: never swap left/semi/anti/right/outer
  if (plan.how !== 'inner' || plan.validate) return plan
  if (plan.leftOn.length === 0 || plan.leftOn.length !== plan.rightOn.length) return plan

  const leftN = estimatePlanRows(plan.left)
  const rightN = estimatePlanRows(plan.right)
  if (rightN <= leftN) return plan
  if (leftN === 0 || rightN / Math.max(leftN, 1) < 1.25) return plan

  // Bare names come from the left; colliding right cols get rSuffix. Swapping sides
  // would invert provenance under the same names — skip when non-key names overlap.
  const leftCols = planOutputColumns(plan.left) ?? leafColumnNames(plan.left)
  const rightCols = planOutputColumns(plan.right) ?? leafColumnNames(plan.right)
  const leftKeySet = new Set(plan.leftOn)
  const rightKeySet = new Set(plan.rightOn)
  for (const name of leftCols) {
    if (leftKeySet.has(name) && rightKeySet.has(name)) continue
    if (rightCols.has(name)) return plan
  }

  const swapped: JoinNode = {
    type: 'join',
    left: plan.right,
    right: plan.left,
    leftOn: plan.rightOn,
    rightOn: plan.leftOn,
    how: 'inner',
    lSuffix: plan.lSuffix,
    rSuffix: plan.rSuffix,
    validate: plan.validate,
  }
  return withOriginalJoinColumns(plan, swapped)
}

/**
 * Greedy left-deep reorder for multi-leaf inner equi-join graphs (cross-name keys OK).
 * Sort leaves by estimate; grow by joining the connected candidate with smallest build size as right.
 */
function reorderInnerJoinGraph(plan: PlanNode): PlanNode {
  plan = rewriteChildren(plan, reorderInnerJoinGraph)
  if (plan.type !== 'join' || plan.how !== 'inner' || plan.validate) return plan

  const graph = collectInnerJoinGraph(plan)
  if (!graph || graph.leaves.length < 3) {
    // Same-key 2-way handled by swap; fall back to legacy same-key chain if graph failed
    return reorderSameKeyChainFallback(plan)
  }

  const { leaves, edges, lSuffix, rSuffix } = graph
  // Same provenance hazard as swapInnerJoinBuildSide: do not reorder when bare/_suffix
  // names would flip under a side swap.
  if (leavesHaveAmbiguousCollisions(leaves, edges)) return plan

  const n = leaves.length
  const ranked = leaves
    .map((leaf, i) => ({ i, rows: estimatePlanRows(leaf) }))
    .sort((a, b) => a.rows - b.rows || a.i - b.i)

  // Start with two smallest that are connected
  let startA = -1
  let startB = -1
  let startEdge: JoinEdge | null = null
  outer: for (let ai = 0; ai < ranked.length; ai++) {
    for (let bi = ai + 1; bi < ranked.length; bi++) {
      const a = ranked[ai]!.i
      const b = ranked[bi]!.i
      const e = findEdge(edges, a, b)
      if (e) {
        startA = a
        startB = b
        startEdge = e
        break outer
      }
    }
  }
  if (startA < 0 || !startEdge) return reorderSameKeyChainFallback(plan)

  // Smaller as build (right)
  const aRows = estimatePlanRows(leaves[startA]!)
  const bRows = estimatePlanRows(leaves[startB]!)
  let probe = startA
  let build = startB
  let leftOn = startEdge.leftOn
  let rightOn = startEdge.rightOn
  if (startEdge.leftLeaf === startB) {
    // edge oriented startB→startA
    if (bRows <= aRows) {
      probe = startA
      build = startB
      leftOn = startEdge.rightOn
      rightOn = startEdge.leftOn
    } else {
      probe = startB
      build = startA
      leftOn = startEdge.leftOn
      rightOn = startEdge.rightOn
    }
  } else {
    // edge oriented startA→startB
    if (aRows <= bRows) {
      probe = startB
      build = startA
      leftOn = startEdge.rightOn
      rightOn = startEdge.leftOn
    } else {
      probe = startA
      build = startB
      leftOn = startEdge.leftOn
      rightOn = startEdge.rightOn
    }
  }

  const used = new Set<number>([probe, build])
  let acc: PlanNode = makeInnerEquiJoin(
    leaves[probe]!,
    leaves[build]!,
    leftOn,
    rightOn,
    lSuffix,
    rSuffix,
  )

  while (used.size < n) {
    let bestLeaf = -1
    let bestEdges: JoinEdge[] | null = null
    let bestRows = Infinity
    for (let i = 0; i < n; i++) {
      if (used.has(i)) continue

      const connectingEdges = findEdgesToSet(edges, i, used)
      if (connectingEdges.length === 0) continue

      const rows = estimatePlanRows(leaves[i]!)

      if (rows < bestRows || (rows === bestRows && i < bestLeaf)) {
        bestRows = rows
        bestLeaf = i
        bestEdges = connectingEdges
      }
    }

    if (bestLeaf < 0 || !bestEdges) {
      return reorderSameKeyChainFallback(plan)
    }

    // A single physical join node currently represents one leaf-to-subtree
    // equi relation. Multiple independent edges would require preserving
    // all predicates explicitly. Until that representation exists, do not reorder.
    if (bestEdges.length !== 1) {
      return plan
    }

    const bestEdge = bestEdges[0]!

    // Attach new leaf as build (right); keys: used-side on left, new leaf on right
    const newIsRight = bestEdge.rightLeaf === bestLeaf
    const joinLeftOn = newIsRight ? bestEdge.leftOn : bestEdge.rightOn
    const joinRightOn = newIsRight ? bestEdge.rightOn : bestEdge.leftOn
    acc = makeInnerEquiJoin(acc, leaves[bestLeaf]!, joinLeftOn, joinRightOn, lSuffix, rSuffix)
    used.add(bestLeaf)
  }

  return withOriginalJoinColumns(plan, acc)
}

function findEdge(edges: JoinEdge[], a: number, b: number): JoinEdge | null {
  for (const e of edges) {
    if ((e.leftLeaf === a && e.rightLeaf === b) || (e.leftLeaf === b && e.rightLeaf === a)) return e
  }
  return null
}

/**
 * True when an equi-edge connects leaves that also share a non-key column name.
 * Reordering/swapping that edge's sides would invert bare vs suffixed provenance.
 */
function leavesHaveAmbiguousCollisions(leaves: PlanNode[], edges: JoinEdge[]): boolean {
  const colSets = leaves.map((l) => planOutputColumns(l) ?? leafColumnNames(l))
  for (const edge of edges) {
    const leftCols = colSets[edge.leftLeaf]!
    const rightCols = colSets[edge.rightLeaf]!
    const keys = new Set<string>([...edge.leftOn, ...edge.rightOn])
    for (const name of leftCols) {
      if (keys.has(name)) continue
      if (rightCols.has(name)) return true
    }
  }
  return false
}

function findEdgesToSet(
  edges: JoinEdge[],
  leaf: number,
  used: Set<number>,
): JoinEdge[] {
  const matches: JoinEdge[] = []

  for (const edge of edges) {
    if (edge.leftLeaf === leaf && used.has(edge.rightLeaf)) {
      matches.push(edge)
      continue
    }

    if (edge.rightLeaf === leaf && used.has(edge.leftLeaf)) {
      matches.push(edge)
    }
  }

  return matches
}

/** Legacy same-key chain when multi-key provenance fails. */
function reorderSameKeyChainFallback(plan: PlanNode): PlanNode {
  if (plan.type !== 'join' || plan.how !== 'inner' || plan.validate) return plan
  if (plan.leftOn.length !== 1 || plan.rightOn.length !== 1) return plan
  if (plan.leftOn[0] !== plan.rightOn[0]) return plan
  const key = plan.leftOn[0]!

  function flatten(p: PlanNode): PlanNode[] | null {
    if (p.type !== 'join') return [p]
    if (
      p.how !== 'inner' ||
      p.validate ||
      p.leftOn.length !== 1 ||
      p.rightOn.length !== 1 ||
      p.leftOn[0] !== key ||
      p.rightOn[0] !== key
    ) {
      return null
    }
    const leftLeaves = p.left.type === 'join' ? flatten(p.left) : [p.left]
    if (!leftLeaves) return null
    return [...leftLeaves, p.right]
  }

  const leaves = flatten(plan)
  if (!leaves || leaves.length < 3) return plan

  // Same-key chain still swaps leaf order; skip when non-key names collide.
  const syntheticEdges: JoinEdge[] = []
  for (let i = 0; i < leaves.length - 1; i++) {
    syntheticEdges.push({ leftLeaf: i, rightLeaf: i + 1, leftOn: [key], rightOn: [key] })
  }
  if (leavesHaveAmbiguousCollisions(leaves, syntheticEdges)) return plan

  const ranked = leaves
    .map((leaf, i) => ({ leaf, i, rows: estimatePlanRows(leaf) }))
    .sort((a, b) => a.rows - b.rows || a.i - b.i)

  let acc: PlanNode = makeInnerEquiJoin(
    ranked[1]!.leaf,
    ranked[0]!.leaf,
    [key],
    [key],
    plan.lSuffix,
    plan.rSuffix,
  )
  for (let i = 2; i < ranked.length; i++) {
    acc = makeInnerEquiJoin(acc, ranked[i]!.leaf, [key], [key], plan.lSuffix, plan.rSuffix)
  }
  return withOriginalJoinColumns(plan, acc)
}

/**
 * Fold project(join) → join when project is a pure column subset/reorder.
 * Execute already accepts keep on join; this removes the trailing project node.
 */
function foldProjectIntoJoin(plan: PlanNode): PlanNode {
  plan = rewriteChildren(plan, foldProjectIntoJoin)
  if (plan.type !== 'project' || plan.input.type !== 'join') return plan
  const names = projectColumnSubset(plan.columns)
  if (!names) return plan
  const joinCols = planOutputColumns(plan.input)
  if (!joinCols || names.some((n) => !joinCols.has(n))) return plan
  // Cannot attach keep on PlanNode join type without schema change — leave project
  // but mark via a rename-free project that executeCpu already fuses to keep.
  // If order matches default join order, drop project entirely.
  const defaultOrder = planOutputColumnList(plan.input)
  if (defaultOrder && defaultOrder.length === names.length && defaultOrder.every((c, i) => c === names[i])) {
    return plan.input
  }
  return plan
}

function oneRound(plan: PlanNode): PlanNode {
  let p = plan
  p = mergeFilters(p)
  p = filterPastProject(p)
  p = filterIntoJoin(p)
  p = filterUnderWithColumn(p)
  p = filterUnderSort(p)
  p = filterUnderDrop(p)
  p = filterUnderRename(p)
  p = limitUnderProject(p)
  p = limitUnderSort(p)
  p = foldProjectDrop(p)
  p = groupByColumnPrune(p)
  p = pushdownProjections(p)
  p = mergeFilters(p)
  p = reorderInnerJoinGraph(p)
  p = swapInnerJoinBuildSide(p)
  p = foldProjectIntoJoin(p)
  return p
}

function planEqualShallow(a: PlanNode, b: PlanNode): boolean {
  return stablePlanJson(a) === stablePlanJson(b)
}

/** Structural JSON for plan compare — never stringify column payloads. */
function stablePlanJson(plan: PlanNode): string {
  return JSON.stringify(plan, (_k, v) => {
    if (typeof v === 'function') return '[fn]'
    if (typeof v === 'bigint') return v.toString()
    if (
      v &&
      typeof v === 'object' &&
      'numRows' in v &&
      'columns' in v &&
      Array.isArray((v as { columns: unknown }).columns) &&
      'schema' in v
    ) {
      const t = v as { numRows: number; columns: Array<{ field: { name: string } }> }
      return {
        __table: true,
        rows: t.numRows,
        cols: t.columns.map((c) => c.field.name),
      }
    }
    if (ArrayBuffer.isView(v)) return { __view: (v as ArrayBufferView).byteLength }
    return v
  })
}

/**
 * Optimize a lazy plan with rule-based rewrites.
 * Idempotent within MAX_ROUNDS; safe to call on already-optimized plans.
 */
export function optimizePlan(plan: PlanNode): PlanNode {
  let current = plan
  for (let i = 0; i < MAX_ROUNDS; i++) {
    const next = oneRound(current)
    if (planEqualShallow(current, next)) return next
    current = next
  }
  return current
}

/** Cheap fingerprint of join topology + build side — detects reorder without full plan stringify. */
export function joinPlanFingerprint(plan: PlanNode): string {
  const parts: string[] = []
  const walk = (p: PlanNode, path: string): void => {
    if (p.type === 'join') {
      // Include path + how/on + estimated sizes + leaf scan shape so topology-only reorders differ.
      const leftShape = leafShape(p.left)
      const rightShape = leafShape(p.right)
      parts.push(
        `${path}:j:${p.how}|${p.leftOn.join(',')}=${p.rightOn.join(',')}|L${estimatePlanRows(p.left)}@${leftShape}|R${estimatePlanRows(p.right)}@${rightShape}`,
      )
      walk(p.left, `${path}L`)
      walk(p.right, `${path}R`)
      return
    }
    if (
      p.type === 'project' ||
      p.type === 'filter' ||
      p.type === 'limit' ||
      p.type === 'sort' ||
      p.type === 'drop' ||
      p.type === 'rename'
    ) {
      walk(p.input, path)
      return
    }
    if (p.type === 'scan') {
      parts.push(`${path}:s:${p.table.numRows}:${p.table.schema.map((f) => f.name).join(',')}`)
    }
  }
  walk(plan, '')
  return parts.join(';')
}

/** Compact identity of near-scan leaves under a join child (row count + column names). */
function leafShape(plan: PlanNode): string {
  if (plan.type === 'scan') {
    return `s${plan.table.numRows}[${plan.table.schema.map((f) => f.name).join(',')}]`
  }
  if (plan.type === 'join') {
    return `j(${leafShape(plan.left)},${leafShape(plan.right)})`
  }
  if (
    plan.type === 'project' ||
    plan.type === 'filter' ||
    plan.type === 'limit' ||
    plan.type === 'sort' ||
    plan.type === 'drop' ||
    plan.type === 'rename'
  ) {
    const tag = plan.type === 'filter' ? 'f' : plan.type[0]!
    return `${tag}(${leafShape(plan.input)})`
  }
  if ('input' in plan && plan.input) return `u(${leafShape(plan.input)})`
  return '?'
}

/** True when optimize changed join build/order shape. */
export function joinOrderChanged(before: PlanNode, after: PlanNode): boolean {
  return joinPlanFingerprint(before) !== joinPlanFingerprint(after)
}
