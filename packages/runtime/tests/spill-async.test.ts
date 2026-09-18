/**
 * Under a memory budget the runtime executes unit by unit and the spilling operators (sort / unique / join)
 * take the fs/promises twins: results equal the in-memory kernels, the report names the spill kernel, and the
 * event loop keeps turning while the disk works (a synchronous spill would freeze the interval below).
 */
import { afterEach, describe, expect, it } from 'vitest'
import { DataFrame, ExecutionAbortedError, clearPersistCache, col } from '@columna/core'
import { clearMemoryPolicy } from '../src/index.js'

afterEach(() => {
  clearMemoryPolicy()
  clearPersistCache()
})

const N = 60_000
const base = DataFrame.fromColumns({
  id: Int32Array.from({ length: N }, (_, i) => i),
  x: Float64Array.from({ length: N }, (_, i) => ((i * 7919) % 10_007) / 7),
  g: Array.from({ length: N }, (_, i) => `g${i % 211}`),
})
const dims = DataFrame.fromRows(Array.from({ length: 211 }, (_, i) => ({ g: `g${i}`, w: i * 0.5 })))
const tiny = { memory: { maxBytes: 64 * 1024, spill: true } }

describe('async spill under a memory budget', () => {
  const plans = () => ({
    sort: { plan: base.lazy().sort(col('x').desc(), 'id'), kernel: 'js:sort+spill' },
    unique: { plan: base.lazy().unique(['g']).sort('g'), kernel: 'js:unique+spill' },
    'inner join': { plan: base.lazy().join(dims.lazy(), { on: 'g' }).sort('id'), kernel: 'js:join+spill' },
    'left join projected': { plan: base.lazy().join(dims.lazy(), { on: 'g', how: 'left' }).select('id', 'w').sort('id'), kernel: 'js:join+spill' },
  })
  for (const [name, { plan, kernel }] of Object.entries(plans())) {
    it(`${name}: same rows as in memory; report names ${kernel} and spilledBytes > 0`, async () => {
      const plain = await plan.collect()
      const { frame, report } = await plan.collectWithReport(tiny)
      expect(frame.toArray()).toEqual(plain.toArray())
      expect(report.events.map((e) => e.kernel)).toContain(kernel)
      expect(report.spilledBytes ?? 0).toBeGreaterThan(0)
    })
  }

  it('the event loop keeps turning while runs are written and read (non-blocking I/O)', async () => {
    let ticks = 0
    const timer = setInterval(() => ticks++, 1)
    try {
      const { report } = await base.lazy().sort('x').collectWithReport(tiny)
      expect(report.events.map((e) => e.kernel)).toContain('js:sort+spill')
    } finally {
      clearInterval(timer)
    }
    expect(ticks).toBeGreaterThan(5)
  })

  it('an abort during a spilling plan is honoured between units and leaves no temp files behind', async () => {
    const { readdirSync } = await import('node:fs')
    const { resolveSpillDir } = await import('../src/memory.js')
    const ac = new AbortController()
    setTimeout(() => ac.abort(), 2)
    let lf = base.lazy()
    for (let k = 0; k < 4; k++) lf = lf.withColumn(`y${k}`, col('x').add(k)).sort(`y${k}`)
    const err = await lf.collect({ ...tiny, signal: ac.signal }).catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ExecutionAbortedError)
    const dir = resolveSpillDir()
    let leftovers: string[] = []
    try {
      leftovers = readdirSync(dir).filter((f) => /^(sort|uniq|join)-/.test(f))
    } catch {
      /* no spill dir created yet */
    }
    expect(leftovers).toEqual([])
  })
})
