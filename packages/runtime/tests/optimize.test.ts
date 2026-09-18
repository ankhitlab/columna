import { describe, expect, it } from 'vitest'
import { DataFrame, col } from '@columna/core'
import { optimizePlan, estimatePlanRows, joinOrderChanged } from '../src/optimize.js'
import { exprColumnRefs, pushdownProjections } from '../src/pushdown.js'
import type { ExprNode, PlanNode } from '../src/types.js'

function scan(rows: Record<string, unknown>[]): PlanNode {
  return { type: 'scan', table: DataFrame.fromRows(rows).table }
}

function colRef(name: string): ExprNode {
  return { type: 'col', name }
}

function gt(name: string, lit: number): ExprNode {
  return { type: 'binary', op: 'gt', left: colRef(name), right: { type: 'lit', value: lit } }
}

function strip(plan: PlanNode): unknown {
  return JSON.parse(JSON.stringify(plan, (_k, v) => (typeof v === 'function' ? '[fn]' : v)))
}

describe('optimizePlan rules', () => {
  it('R1: merges consecutive filters into one AND', () => {
    const plan: PlanNode = {
      type: 'filter',
      predicate: gt('b', 0),
      input: {
        type: 'filter',
        predicate: gt('a', 1),
        input: scan([{ a: 1, b: 2 }]),
      },
    }
    const out = optimizePlan(plan)
    expect(out.type).toBe('filter')
    if (out.type === 'filter') {
      expect(out.input.type).toBe('scan')
      expect(out.predicate).toEqual({
        type: 'binary',
        op: 'and',
        left: gt('a', 1),
        right: gt('b', 0),
      })
    }
  })

  it('R2: pushes filter past simple project', () => {
    const plan: PlanNode = {
      type: 'filter',
      predicate: gt('a', 1),
      input: {
        type: 'project',
        columns: ['a', 'b'],
        input: scan([{ a: 1, b: 2, c: 3 }]),
      },
    }
    const out = optimizePlan(plan)
    expect(out.type).toBe('project')
    if (out.type === 'project') {
      expect(out.input.type).toBe('filter')
      if (out.input.type === 'filter') {
        expect(out.input.predicate).toEqual(gt('a', 1))
      }
    }
  })

  it('R3: pushes left-only filter into join left child', () => {
    const left = scan([{ id: 1, a: 10, x: 1 }])
    const right = scan([{ id: 1, b: 20, y: 2 }])
    const plan: PlanNode = {
      type: 'filter',
      predicate: gt('a', 5),
      input: {
        type: 'join',
        left,
        right,
        leftOn: ['id'],
        rightOn: ['id'],
        how: 'inner',
      },
    }
    const out = optimizePlan(plan)
    // After push + projection passes, join should have filtered left
    const findJoin = (p: PlanNode): Extract<PlanNode, { type: 'join' }> | null => {
      if (p.type === 'join') return p
      if (p.type === 'project' || p.type === 'filter') return findJoin(p.input)
      return null
    }
    const join = findJoin(out)
    expect(join).not.toBeNull()
    expect(join!.left.type).toBe('filter')
    if (join!.left.type === 'filter') {
      expect(join!.left.predicate).toEqual(gt('a', 5))
    }
    // Top-level should not still be a lone filter(join) with the same predicate
    if (out.type === 'filter') {
      expect(out.predicate).not.toEqual(gt('a', 5))
    }
  })

  it('R4: pushes filter under withColumn when predicate ignores new col', () => {
    const plan: PlanNode = {
      type: 'filter',
      predicate: gt('a', 0),
      input: {
        type: 'withColumn',
        name: 'z',
        expr: { type: 'binary', op: 'add', left: colRef('a'), right: { type: 'lit', value: 1 } },
        input: scan([{ a: 1 }]),
      },
    }
    const out = optimizePlan(plan)
    expect(out.type).toBe('withColumn')
    if (out.type === 'withColumn') {
      expect(out.input.type).toBe('filter')
    }
  })

  it('R5: pushes limit under simple project', () => {
    const plan: PlanNode = {
      type: 'limit',
      n: 2,
      input: {
        type: 'project',
        columns: ['a'],
        input: scan([{ a: 1, b: 2 }]),
      },
    }
    const out = optimizePlan(plan)
    expect(out.type).toBe('project')
    if (out.type === 'project') {
      expect(out.input.type).toBe('limit')
      if (out.input.type === 'limit') expect(out.input.n).toBe(2)
    }
  })

  it('R7: folds nested projects', () => {
    const plan: PlanNode = {
      type: 'project',
      columns: ['a'],
      input: {
        type: 'project',
        columns: ['a', 'b'],
        input: scan([{ a: 1, b: 2, c: 3 }]),
      },
    }
    const out = optimizePlan(plan)
    expect(out.type).toBe('project')
    if (out.type === 'project') {
      expect(out.columns).toEqual(['a'])
      expect(out.input.type).toBe('scan')
    }
  })

  it('R8: prunes groupBy input columns under trailing project', () => {
    const plan: PlanNode = {
      type: 'project',
      columns: ['g', 's'],
      input: {
        type: 'groupBy',
        keys: ['g'],
        aggs: [{ name: 's', expr: { type: 'agg', op: 'sum', expr: colRef('v') } }],
        input: scan([{ g: 'a', v: 1, noise: 9 }]),
      },
    }
    const out = optimizePlan(plan)
    expect(out.type).toBe('project')
    if (out.type === 'project' && out.input.type === 'groupBy') {
      expect(out.input.input.type).toBe('project')
      if (out.input.input.type === 'project') {
        const cols = out.input.input.columns
        expect(cols).toEqual(expect.arrayContaining(['g', 'v']))
        expect(cols).not.toContain('noise')
      }
    }
  })

  it('pushes filter under sort (enables filter→sort→limit fuse)', () => {
    const plan: PlanNode = {
      type: 'filter',
      predicate: gt('a', 1),
      input: {
        type: 'sort',
        by: [{ expr: colRef('a'), descending: false }],
        input: scan([{ a: 1 }, { a: 2 }]),
      },
    }
    const out = optimizePlan(plan)
    expect(out.type).toBe('sort')
    if (out.type === 'sort') expect(out.input.type).toBe('filter')
  })

  it('pushes filter under rename with remapped predicate', () => {
    const plan: PlanNode = {
      type: 'filter',
      predicate: gt('x', 0),
      input: {
        type: 'rename',
        mapping: { a: 'x' },
        input: scan([{ a: 1, b: 2 }]),
      },
    }
    const out = optimizePlan(plan)
    expect(out.type).toBe('rename')
    if (out.type === 'rename' && out.input.type === 'filter') {
      expect(out.input.predicate).toEqual(gt('a', 0))
    }
  })

  it('composes nested renames', () => {
    const plan: PlanNode = {
      type: 'rename',
      mapping: { y: 'z' },
      input: {
        type: 'rename',
        mapping: { a: 'y' },
        input: scan([{ a: 1 }]),
      },
    }
    const out = optimizePlan(plan)
    expect(out.type).toBe('rename')
    if (out.type === 'rename') {
      expect(out.mapping).toEqual({ a: 'z' })
      expect(out.input.type).toBe('scan')
    }
  })

  it('swaps inner join so smaller side is build (right)', () => {
    const big = scan(Array.from({ length: 100 }, (_, i) => ({ id: i % 10, a: i })))
    const small = scan(Array.from({ length: 5 }, (_, i) => ({ id: i, b: i * 10 })))
    const plan: PlanNode = {
      type: 'join',
      left: small,
      right: big,
      leftOn: ['id'],
      rightOn: ['id'],
      how: 'inner',
    }
    const out = optimizePlan(plan)
    // After swap+project: innermost join should have big as left (probe) and small as right (build)
    const findJoin = (p: PlanNode): Extract<PlanNode, { type: 'join' }> | null => {
      if (p.type === 'join') return p
      if ('input' in p && p.input) return findJoin(p.input as PlanNode)
      return null
    }
    const j = findJoin(out)
    expect(j).not.toBeNull()
    expect(estimatePlanRows(j!.right)).toBeLessThanOrEqual(estimatePlanRows(j!.left))
  })

  it('reorders 3-way inner equi chain by cardinality', () => {
    const a = scan(Array.from({ length: 80 }, (_, i) => ({ id: i % 8, a: i })))
    const b = scan(Array.from({ length: 8 }, (_, i) => ({ id: i, b: i })))
    const c = scan(Array.from({ length: 3 }, (_, i) => ({ id: i, c: i })))
    const plan: PlanNode = {
      type: 'join',
      left: {
        type: 'join',
        left: a,
        right: b,
        leftOn: ['id'],
        rightOn: ['id'],
        how: 'inner',
      },
      right: c,
      leftOn: ['id'],
      rightOn: ['id'],
      how: 'inner',
    }
    const out = optimizePlan(plan)
    expect(joinOrderChanged(plan, out)).toBe(true)
    const findJoins = (p: PlanNode, acc: Array<Extract<PlanNode, { type: 'join' }>> = []) => {
      if (p.type === 'join') {
        acc.push(p)
        findJoins(p.left, acc)
        findJoins(p.right, acc)
      } else if ('input' in p && p.input) findJoins(p.input as PlanNode, acc)
      return acc
    }
    const joins = findJoins(out)
    expect(joins.length).toBeGreaterThanOrEqual(1)
    // Smallest leaf (c, 3 rows) should appear as a build (right) of some join
    const hasSmallBuild = joins.some((j) => estimatePlanRows(j.right) <= 3)
    expect(hasSmallBuild).toBe(true)
  })

  it('pushes filter under drop when predicate does not use dropped cols', () => {
    const plan: PlanNode = {
      type: 'filter',
      predicate: gt('a', 0),
      input: {
        type: 'drop',
        columns: ['c'],
        input: scan([{ a: 1, b: 2, c: 3 }]),
      },
    }
    const out = optimizePlan(plan)
    expect(out.type).toBe('drop')
    if (out.type === 'drop') {
      expect(out.columns).toEqual(['c'])
      expect(out.input.type).toBe('filter')
    }
  })

  it('is idempotent', () => {
    const plan: PlanNode = {
      type: 'filter',
      predicate: gt('b', 0),
      input: {
        type: 'filter',
        predicate: gt('a', 1),
        input: {
          type: 'project',
          columns: ['a', 'b'],
          input: scan([{ a: 1, b: 2, c: 3 }]),
        },
      },
    }
    const once = optimizePlan(plan)
    const twice = optimizePlan(once)
    expect(strip(twice)).toEqual(strip(once))
  })

  it('pushdownProjections remains available and composes', () => {
    const plan: PlanNode = {
      type: 'project',
      columns: ['a'],
      input: {
        type: 'sort',
        input: scan([{ a: 1, b: 2 }]),
        by: [{ expr: colRef('a'), descending: false }],
      },
    }
    const viaPush = pushdownProjections(plan)
    const viaOpt = optimizePlan(plan)
    // Projection wraps sort so sort-only columns are not exposed in the output schema.
    expect(viaPush.type).toBe('project')
    expect(viaOpt.type).toBe('project')
    if (viaPush.type === 'project') expect(viaPush.input.type).toBe('sort')
    if (viaOpt.type === 'project') expect(viaOpt.input.type).toBe('sort')
  })
})

describe('optimizePlan identity via collect', () => {
  it('merged filters match stepwise filter', async () => {
    const df = DataFrame.fromRows([
      { a: 1, b: 10 },
      { a: 2, b: 20 },
      { a: 3, b: 5 },
      { a: 4, b: 40 },
    ])
    const fused = await df.filter(col('a').gt(1)).filter(col('b').gt(10)).collect()
    const one = await df.filter(col('a').gt(1).and(col('b').gt(10))).collect()
    expect(fused.toArray()).toEqual(one.toArray())
  })

  it('join then left filter matches filter then join', async () => {
    const left = DataFrame.fromRows([
      { id: 1, a: 10 },
      { id: 2, a: 3 },
      { id: 3, a: 30 },
    ])
    const right = DataFrame.fromRows([
      { id: 1, b: 100 },
      { id: 2, b: 200 },
      { id: 3, b: 300 },
    ])
    const pushed = await left.join(right, { on: 'id' }).filter(col('a').gt(5)).sort('id').collect()
    const early = await left
      .filter(col('a').gt(5))
      .join(right, { on: 'id' })
      .sort('id')
      .collect()
    expect(pushed.toArray()).toEqual(early.toArray())
  })

  it('select + limit identity', async () => {
    const df = DataFrame.fromRows([
      { a: 1, b: 2 },
      { a: 3, b: 4 },
      { a: 5, b: 6 },
    ])
    const out = await df.select('a').limit(2).collect()
    expect(out.toArray()).toEqual([{ a: 1 }, { a: 3 }])
  })

  it('filter after sort matches filter then sort', async () => {
    const df = DataFrame.fromRows([
      { a: 3, b: 1 },
      { a: 1, b: 2 },
      { a: 2, b: 3 },
    ])
    const a = await df.sort('a').filter(col('a').gt(1)).collect()
    const b = await df.filter(col('a').gt(1)).sort('a').collect()
    expect(a.toArray()).toEqual(b.toArray())
  })

  it('filter after rename matches rename after filter on old name', async () => {
    const df = DataFrame.fromRows([
      { a: 1 },
      { a: 5 },
      { a: 3 },
    ])
    const a = await df.rename({ a: 'x' }).filter(col('x').gt(2)).sort('x').collect()
    const b = await df.filter(col('a').gt(2)).rename({ a: 'x' }).sort('x').collect()
    expect(a.toArray()).toEqual(b.toArray())
  })

  it('inner join swap preserves row content', async () => {
    const left = DataFrame.fromRows([
      { id: 0, a: 1 },
      { id: 1, a: 2 },
      { id: 2, a: 3 },
    ])
    const right = DataFrame.fromRows(
      Array.from({ length: 50 }, (_, i) => ({ id: i % 3, b: i })),
    )
    // Small left, large right → optimizer swaps build side
    const out = await left.join(right, { on: 'id' }).collect()
    const rows = out.toArray().sort((x, y) => Number(x.a) - Number(y.a) || Number(x.b) - Number(y.b))
    expect(rows.length).toBeGreaterThan(0)
    expect(rows.every((r) => r.id === 0 || r.id === 1 || r.id === 2)).toBe(true)
    // Spot-check: every output row's id matches both sides' join key
    for (const r of rows) {
      expect(left.toArray().some((l) => l.id === r.id && l.a === r.a)).toBe(true)
    }
  })

  it('explain shows merged filter', () => {
    const df = DataFrame.fromRows([{ a: 1, b: 2 }])
    const text = df.filter(col('a').gt(0)).filter(col('b').gt(0)).explain()
    const filters = text.split('\n').filter((l) => l.trimStart().startsWith('Filter'))
    expect(filters.length).toBe(1)
  })

  it('explain includes rows≈ after optimize', () => {
    const df = DataFrame.fromRows(
      Array.from({ length: 20 }, (_, i) => ({ a: i, b: i % 2 })),
    )
    const text = df.filter(col('a').gt(5)).explain()
    expect(text).toMatch(/Filter rows≈\d+/)
  })

  it('left join never swaps sides', () => {
    const big = scan(Array.from({ length: 100 }, (_, i) => ({ id: i % 10, a: i })))
    const small = scan(Array.from({ length: 5 }, (_, i) => ({ id: i, b: i * 10 })))
    const plan: PlanNode = {
      type: 'join',
      left: small,
      right: big,
      leftOn: ['id'],
      rightOn: ['id'],
      how: 'left',
    }
    const out = optimizePlan(plan)
    expect(out.type).toBe('join')
    if (out.type === 'join') {
      expect(estimatePlanRows(out.left)).toBe(5)
      expect(estimatePlanRows(out.right)).toBe(100)
    }
  })

  it('semi join never swaps sides', () => {
    const small = scan(Array.from({ length: 5 }, (_, i) => ({ id: i, a: i })))
    const big = scan(Array.from({ length: 80 }, (_, i) => ({ id: i % 5, b: i })))
    const plan: PlanNode = {
      type: 'join',
      left: small,
      right: big,
      leftOn: ['id'],
      rightOn: ['id'],
      how: 'semi',
    }
    const out = optimizePlan(plan)
    expect(out.type).toBe('join')
    if (out.type === 'join') {
      expect(out.how).toBe('semi')
      expect(estimatePlanRows(out.left)).toBe(5)
      expect(estimatePlanRows(out.right)).toBe(80)
    }
  })

  it('anti join never swaps sides', () => {
    const small = scan(Array.from({ length: 5 }, (_, i) => ({ id: i, a: i })))
    const big = scan(Array.from({ length: 80 }, (_, i) => ({ id: i % 5, b: i })))
    const plan: PlanNode = {
      type: 'join',
      left: small,
      right: big,
      leftOn: ['id'],
      rightOn: ['id'],
      how: 'anti',
    }
    const out = optimizePlan(plan)
    expect(out.type).toBe('join')
    if (out.type === 'join') {
      expect(out.how).toBe('anti')
      expect(estimatePlanRows(out.left)).toBe(5)
      expect(estimatePlanRows(out.right)).toBe(80)
    }
  })

  it('reorders multi-key / cross-name inner chain', async () => {
    const a = DataFrame.fromRows(
      Array.from({ length: 80 }, (_, i) => ({ id: i % 8, a: i })),
    )
    const b = DataFrame.fromRows(
      Array.from({ length: 8 }, (_, i) => ({ a_id: i, b: i * 10 })),
    )
    const c = DataFrame.fromRows(
      Array.from({ length: 3 }, (_, i) => ({ id: i, c: i * 100 })),
    )
    // ((A⋈B)⋈C) with A large — greedy should rebuild so smallest (C) is a build side
    const plan: PlanNode = {
      type: 'join',
      left: {
        type: 'join',
        left: { type: 'scan', table: a.table },
        right: { type: 'scan', table: b.table },
        leftOn: ['id'],
        rightOn: ['a_id'],
        how: 'inner',
      },
      right: { type: 'scan', table: c.table },
      leftOn: ['id'],
      rightOn: ['id'],
      how: 'inner',
    }
    const out = optimizePlan(plan)
    expect(joinOrderChanged(plan, out)).toBe(true)
    const findJoins = (p: PlanNode, acc: Array<Extract<PlanNode, { type: 'join' }>> = []) => {
      if (p.type === 'join') {
        acc.push(p)
        findJoins(p.left, acc)
        findJoins(p.right, acc)
      } else if ('input' in p && p.input) findJoins(p.input as PlanNode, acc)
      return acc
    }
    const joins = findJoins(out)
    expect(joins.length).toBeGreaterThanOrEqual(1)
    expect(joins.some((j) => estimatePlanRows(j.right) <= 3)).toBe(true)
    const collected = await a
      .join(b, { leftOn: 'id', rightOn: 'a_id' })
      .join(c, { on: 'id' })
      .sort('id')
      .collect()
    expect(collected.toArray().length).toBeGreaterThan(0)
  })

  it('join reorder preserves output column order after collect', async () => {
    const left = DataFrame.fromRows([
      { id: 0, a: 1 },
      { id: 1, a: 2 },
    ])
    const right = DataFrame.fromRows(
      Array.from({ length: 30 }, (_, i) => ({ id: i % 2, b: i })),
    )
    // Small left + large right → swap; schema order must stay left-then-right (id, a, b)
    const out = await left.join(right, { on: 'id' }).collect()
    expect(out.columns).toEqual(['id', 'a', 'b'])
  })

  it('joinReorder reported when collectWithReport swaps build side', async () => {
    const left = DataFrame.fromRows([
      { id: 0, a: 1 },
      { id: 1, a: 2 },
    ])
    const right = DataFrame.fromRows(
      Array.from({ length: 40 }, (_, i) => ({ id: i % 2, b: i })),
    )
    const { report } = await left.join(right, { on: 'id' }).collectWithReport()
    const reorder = report.events.find((e) => e.kernel === 'optimized:joinReorder')
    expect(reorder).toBeDefined()
  })
})

describe('optimizer correctness regressions', () => {
  it('does not fold an alias projection as a plain column subset', async () => {
    const out = await DataFrame.fromRows([{ a: 1, b: 2 }])
      .select(col('a').alias('b'), col('b').alias('a'))
      .select('a')
      .collect()

    expect(out.toArray()).toEqual([{ a: 2 }])
  })

  it('two simultaneous rename swaps cancel each other', async () => {
    const out = await DataFrame.fromRows([{ a: 1, b: 2 }])
      .rename({ a: 'b', b: 'a' })
      .rename({ a: 'b', b: 'a' })
      .collect()

    expect(out.toArray()).toEqual([{ a: 1, b: 2 }])
  })

  it('isBetween dependencies include low and high expressions', () => {
    const expr = col('x').isBetween(col('lo'), col('hi')).node

    expect([...exprColumnRefs(expr)].sort()).toEqual(['hi', 'lo', 'x'])
  })

  it('str.concat dependency includes the other expression', () => {
    const expr = {
      type: 'str',
      op: 'concat',
      expr: { type: 'col', name: 'a' },
      other: { type: 'col', name: 'b' },
    } as const

    expect([...exprColumnRefs(expr)].sort()).toEqual(['a', 'b'])
  })

  it('select after isBetween keeps bound columns until predicate execution', async () => {
    const out = await DataFrame.fromRows([
      { x: 5, lo: 0, hi: 10 },
      { x: 20, lo: 0, hi: 10 },
    ])
      .filter(col('x').isBetween(col('lo'), col('hi')))
      .select('x')
      .collect()

    expect(out.toArray()).toEqual([{ x: 5 }])
  })

  it('does not infer join side from a real _right suffix', async () => {
    const left = DataFrame.fromRows([{ id: 1, score_right: 0 }])
    const right = DataFrame.fromRows([{ id: 1, score: 10 }])

    const out = await left.join(right, { on: 'id' }).filter(col('score_right').gt(0)).collect()

    expect(out.toArray()).toEqual([])
  })

  it('does not drop a second edge to an already joined subtree', async () => {
    const a = DataFrame.fromRows([{ aid: 1, a: 10 }])
    const b = DataFrame.fromRows([{ bid: 1, b: 20 }])
    const c = DataFrame.fromRows([{ ca: 10, cb: 999 }])

    const out = await a
      .join(b, { leftOn: 'aid', rightOn: 'bid' })
      .join(c, { leftOn: ['a', 'b'], rightOn: ['ca', 'cb'] })
      .collect()

    expect(out.toArray()).toEqual([])
  })
})
