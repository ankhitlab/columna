/**
 * Integration regressions for 1684ac48625c9155da46b3973fe463e8a3cd4210.
 * Copy into packages/core/tests/review-1684ac4.test.ts.
 * Run: pnpm exec vitest run packages/core/tests/review-1684ac4.test.ts
 *
 * NOT run against the full repository in the review environment.
 * Expected values are independent fixtures: do not compute them with executeCpu,
 * because executeCpu / Runtime may apply the same optimizer under review.
 */
import { afterEach, describe, expect, it } from 'vitest'
import { DataFrame, col } from '@columna/core'
import { clearPersistCache, hashPlan, optimizePlan } from '@columna/runtime'

// Explicit cache clearing prevents test-order dependence.
afterEach(() => clearPersistCache())

describe('optimizer preserves query semantics', () => {
  it('does not remove a post-left-join filter on the nullable right side', async () => {
    const left = DataFrame.fromRows([{ id: 1 }, { id: 2 }])
    const right = DataFrame.fromRows([{ id: 1, score: 0 }])
    const out = await left.leftJoin(right, 'id').filter(col('score').gt(0)).collect()
    expect(out.toArray()).toEqual([])
  })

  it('select after sort does not expose the sort-only column', async () => {
    const out = await DataFrame.fromRows([
      { public: 'B', secret: 2 },
      { public: 'A', secret: 1 },
    ]).sort('secret').select('public').collect()
    expect(out.columns).toEqual(['public'])
    expect(out.toArray()).toEqual([{ public: 'A' }, { public: 'B' }])
  })

  it('computes a broadcast mean on the input of its own filter', async () => {
    const out = await DataFrame.fromColumns({ x: [1, 2, 3, 4] })
      .filter(col('x').gt(2))
      .filter(col('x').gt(col('x').mean()))
      .collect()
    expect(out.toArray()).toEqual([{ x: 4 }])
  })

  it('does not push a filter below a whole-column aggregate', async () => {
    const out = await DataFrame.fromColumns({ x: [1, 3] })
      .withColumn('avg', col('x').mean())
      .filter(col('x').gt(1))
      .collect()
    expect(out.toArray()).toEqual([{ x: 3, avg: 2 }])
  })

  it('does not push a filter below shift, which depends on row positions', async () => {
    const out = await DataFrame.fromColumns({ x: [1, 3] })
      .withColumn('previous', col('x').shift())
      .filter(col('x').gt(1))
      .collect()
    expect(out.toArray()).toEqual([{ x: 3, previous: 1 }])
  })

  it('swapping the build side preserves the origin of colliding column names', async () => {
    const left = DataFrame.fromRows([{ id: 1, value: 'LEFT' }])
    const right = DataFrame.fromRows([
      { id: 1, value: 'RIGHT' },
      { id: 2, value: 'OTHER' },
    ])
    const out = await left.join(right, { on: 'id' }).collect()
    expect(out.toArray()).toEqual([{ id: 1, value: 'LEFT', value_right: 'RIGHT' }])
  })

  it('preserves every component of a composite join key when reordering three leaves', async () => {
    const a = DataFrame.fromRows([{ a1: 1, a2: 10 }])
    const b = DataFrame.fromRows([{ b1: 1, b2: 20 }])
    const c = DataFrame.fromRows([{ c1: 1 }])
    const out = await a
      .join(b, { leftOn: ['a1', 'a2'], rightOn: ['b1', 'b2'] })
      .join(c, { leftOn: 'b1', rightOn: 'c1' })
      .collect()
    // 10 !== 20: the first join has no matches, irrespective of join order.
    expect(out.toArray()).toEqual([])
  })
})

describe('persist does not confuse distinct expressions', () => {
  it('uses distinct cache keys for distinct UDF identities', () => {
    const df = DataFrame.fromColumns({ x: [1] })
    const a = df.withColumn('y', col('x').mapElements(v => Number(v) + 1))
    const b = df.withColumn('y', col('x').mapElements(v => Number(v) + 100))
    expect(hashPlan(optimizePlan(a.plan))).not.toBe(hashPlan(optimizePlan(b.plan)))
  })

  it('does not return a previously persisted result for another UDF', async () => {
    clearPersistCache()
    const df = DataFrame.fromColumns({ x: [1] })
    const first = await df.withColumn('y', col('x').mapElements(v => Number(v) + 1)).persist().collect()
    expect(first.toArray()).toEqual([{ x: 1, y: 2 }])
    const second = await df.withColumn('y', col('x').mapElements(v => Number(v) + 100)).collect()
    expect(second.toArray()).toEqual([{ x: 1, y: 101 }])
  })

  it('distinguishes NaN, positive infinity, negative infinity and null literals', () => {
    const df = DataFrame.fromColumns({ x: [1] })
    const plans = [NaN, Infinity, -Infinity, null].map(value => df.filter(col('x').eq(value)).plan)
    expect(new Set(plans.map(hashPlan)).size).toBe(4)
  })
})
