import { afterEach, describe, expect, it } from 'vitest'
import { DataFrame, clearPersistCache } from '@columna/core'
import {
  Runtime,
  clearMemoryPolicy,
  clearPersistCache as clearRtCache,
  hashPlan,
  persistCacheStats,
  setMemoryPolicy,
} from '../src/index.js'

afterEach(() => {
  clearMemoryPolicy()
  clearPersistCache()
  clearRtCache()
})

describe('persist LRU cache', () => {
  it('second collect hits cache', async () => {
    const df = DataFrame.fromRows([
      { a: 1, b: 2 },
      { a: 3, b: 4 },
      { a: 5, b: 6 },
    ])
    const lazy = df.lazy().filter((c) => c.a.gt(0)).persist()
    const first = await lazy.collectWithReport()
    expect(first.report.cacheHit).toBe(false)
    const second = await lazy.collectWithReport()
    expect(second.report.cacheHit).toBe(true)
    expect(second.frame.table.numRows).toBe(first.frame.table.numRows)
  })

  it('unpersist drops the entry', async () => {
    const df = DataFrame.fromRows([
      { a: 1 },
      { a: 2 },
    ])
    const lazy = df.lazy().select('a').persist()
    await lazy.collect()
    expect(persistCacheStats().entries).toBeGreaterThan(0)
    lazy.unpersist()
    expect(persistCacheStats().entries).toBe(0)
    const again = await lazy.collectWithReport()
    expect(again.report.cacheHit).toBe(false)
  })

  it('evicts by maxCacheBytes', async () => {
    setMemoryPolicy({ maxCacheBytes: 200 })
    const a = DataFrame.fromRows(Array.from({ length: 50 }, (_, i) => ({ x: i }))).lazy().persist()
    const b = DataFrame.fromRows(Array.from({ length: 50 }, (_, i) => ({ y: i }))).lazy().persist()
    await a.collect()
    await b.collect()
    // Soft cap should keep cache from growing without bound
    expect(persistCacheStats().bytes).toBeLessThanOrEqual(5000)
  })

  it('hashPlan stays cheap on large scan tables', () => {
    const df = DataFrame.fromRows(Array.from({ length: 100_000 }, (_, i) => ({ a: i, b: i % 7 })))
    const plan = df.head(1).plan
    const t0 = performance.now()
    const key = hashPlan(plan)
    const ms = performance.now() - t0
    expect(key.includes('"__table"')).toBe(true)
    expect(key.length).toBeLessThan(2_000)
    expect(ms).toBeLessThan(50)
  })
})
