/**
 * persist() cache semantics, pinned down:
 *  1. strict engine contract survives the cache: a strict request is served only from an entry that engine
 *     produced; otherwise the plan runs (engine executes, or EngineStrictError);
 *  2. immutable inputs: fromColumns copies by default; zero-copy frames (copy: false) are never cached;
 *  3. UDFs: plans calling mapElements are not cached unless persist({ trustUdfs: true });
 *  4. bounded metadata: maxEntries, maxPending, ttlMs, pendingTtlMs, and counters.
 */
import { describe, expect, it } from 'vitest'
import { executeCpu, type Backend, type ExecContext, type PlanNode } from '@columna/runtime'
import { DataFrame, EngineStrictError, PersistCache, Runtime, col } from '../src/index.js'

/** A stand-in accelerator: runs plans on the CPU kernels but reports itself, and can be switched off. */
function fakeGpu(): Backend & { enabled: boolean; runs: number } {
  const b = {
    name: 'webgpu' as const,
    capabilities: { name: 'webgpu' as const },
    enabled: true,
    runs: 0,
    supports(_plan: PlanNode) {
      return b.enabled
    },
    execute(plan: PlanNode, ctx?: ExecContext) {
      b.runs++
      const t = executeCpu(plan)
      ctx?.trace({ node: plan.type, backend: 'webgpu', kernel: 'fake:gpu', rows: t.numRows })
      return t
    },
  }
  return b
}

function runtimeWithGpu() {
  const gpu = fakeGpu()
  const rt = new Runtime({ persist: new PersistCache(), webgpuMinRows: Number.MAX_SAFE_INTEGER })
  rt.register(gpu)
  return { rt, gpu }
}

const rows = Array.from({ length: 500 }, (_, i) => ({ id: i, x: (i * 37) % 101, g: `g${i % 5}` }))

describe('strict engine × cache', () => {
  it('a CPU-produced entry does not satisfy a strict webgpu request: the plan runs on the GPU instead', async () => {
    const { rt, gpu } = runtimeWithGpu()
    const lf = DataFrame.fromRows(rows).withRuntime(rt).lazy().filter(col('x').gt(50)).persist()
    const first = await lf.collectWithReport() // auto → CPU (threshold), stored with provenance cpu
    expect(first.report.backendsUsed).toEqual(['cpu'])
    const strictGpu = await lf.engine('webgpu', { strict: true }).collectWithReport()
    expect(strictGpu.report.cacheHit).toBe(false)
    expect(strictGpu.report.backendsUsed).toContain('webgpu')
    expect(gpu.runs).toBe(1)
    expect(rt.persist.stats().strictBypasses).toBe(1)
    // now the entry was produced by webgpu → the next strict request is a hit, with honest provenance
    const again = await lf.engine('webgpu', { strict: true }).collectWithReport()
    expect(again.report.cacheHit).toBe(true)
    expect(again.report.requested).toBe('webgpu')
    expect(again.report.strict).toBe(true)
    expect(again.report.cachedFrom?.backendsUsed).toContain('webgpu')
    expect(again.report.dispatched).toBe('webgpu')
    expect(gpu.runs).toBe(1)
    expect(again.frame.toArray()).toEqual(first.frame.toArray())
  })

  it('when the requested engine cannot run the plan, a cached CPU result is not returned: EngineStrictError', async () => {
    const { rt, gpu } = runtimeWithGpu()
    const lf = DataFrame.fromRows(rows).withRuntime(rt).lazy().filter(col('x').gt(50)).persist()
    await lf.collect()
    gpu.enabled = false
    await expect(lf.engine('webgpu', { strict: true }).collect()).rejects.toBeInstanceOf(EngineStrictError)
    // a failed strict run stores nothing and does not replace the entry
    expect(rt.persist.stats().entries).toBe(1)
  })

  it('a strict cpu request is not served a GPU-produced entry; non-strict requests take any entry', async () => {
    const { rt, gpu } = runtimeWithGpu()
    const lf = DataFrame.fromRows(rows).withRuntime(rt).lazy().filter(col('x').gt(10)).persist()
    await lf.engine('webgpu', { strict: true }).collect()
    const nonStrict = await lf.collectWithReport()
    expect(nonStrict.report.cacheHit).toBe(true)
    expect(nonStrict.report.backendsUsed).toContain('webgpu') // provenance, not a claim that CPU ran
    const strictCpu = await lf.engine('cpu', { strict: true }).collectWithReport()
    expect(strictCpu.report.cacheHit).toBe(false)
    expect(strictCpu.report.backendsUsed).toEqual(['cpu'])
    expect(gpu.runs).toBe(1)
  })
})

describe('buffer ownership × cache', () => {
  it('fromColumns copies by default: writing to the source array changes neither the frame nor a cached result', async () => {
    const values = new Float64Array([1, 2, 3])
    const rt = new Runtime({ persist: new PersistCache() })
    const q = DataFrame.fromColumns({ x: values }).withRuntime(rt).lazy().filter(col('x').gt(1)).persist()
    const before = (await q.collect()).toArray()
    values[2] = -100
    const after = await q.collectWithReport()
    expect(after.report.cacheHit).toBe(true)
    expect(after.frame.toArray()).toEqual(before)
    expect(before).toEqual([{ x: 2 }, { x: 3 }])
  })

  it('copy: false shares the buffer and such plans are never cached — results follow the buffer, and the report says why', async () => {
    const values = new Float64Array([1, 2, 3])
    const rt = new Runtime({ persist: new PersistCache() })
    const q = DataFrame.fromColumns({ x: values }, { copy: false }).withRuntime(rt).lazy().filter(col('x').gt(1)).persist()
    const first = await q.collectWithReport()
    expect(first.report.cacheSkipped).toMatch(/caller-owned buffers/)
    values[2] = -100
    const second = await q.collectWithReport()
    expect(second.report.cacheHit).toBe(false)
    expect(second.frame.toArray()).toEqual([{ x: 2 }])
    expect(rt.persist.stats()).toMatchObject({ entries: 0, skipped: 2 })
  })

  it('a frame derived from a zero-copy frame without copying (select / slice views) is still refused', async () => {
    const values = new Int32Array([5, 6, 7, 8])
    const rt = new Runtime({ persist: new PersistCache() })
    const derived = DataFrame.fromColumns({ a: values }, { copy: false }).withRuntime(rt).head(2)
    const r = await derived.lazy().persist().collectWithReport()
    expect(r.report.cacheSkipped).toMatch(/caller-owned/)
    expect(() => rt.persist.store(derived.lazy().plan, derived.table)).toThrow(/caller-owned/)
  })
})

describe('UDF × cache', () => {
  it('plans calling mapElements are executed every time unless trustUdfs asserts purity', async () => {
    let factor = 2
    const rt = new Runtime({ persist: new PersistCache() })
    const df = DataFrame.fromRows([{ x: 1 }, { x: 2 }]).withRuntime(rt)
    const q = df.lazy().withColumn('y', col('x').mapElements((v) => (v as number) * factor))
    const a = await q.persist().collectWithReport()
    expect(a.report.cacheSkipped).toMatch(/UDF/)
    factor = 10
    expect((await q.collect()).getColumn('y').toArray()).toEqual([10, 20])
    const trusted = q.persist({ trustUdfs: true })
    await trusted.collect()
    const hit = await trusted.collectWithReport()
    expect(hit.report.cacheHit).toBe(true)
  })
})

describe('bounded cache metadata', () => {
  it('maxPending bounds marks of plans that are never collected', () => {
    const cache = new PersistCache({ maxPending: 10 })
    const rt = new Runtime({ persist: cache })
    const df = DataFrame.fromRows([{ x: 1 }]).withRuntime(rt)
    for (let i = 0; i < 1000; i++) df.lazy().filter(col('x').gt(i)).persist()
    const s = cache.stats()
    expect(s.pending).toBe(10)
    expect(s.pendingEvictions).toBe(990)
    expect(s.entries).toBe(0)
  })

  it('maxEntries bounds cached tables (LRU: a recently used entry survives)', async () => {
    const cache = new PersistCache({ maxEntries: 3 })
    const rt = new Runtime({ persist: cache })
    const df = DataFrame.fromRows(rows).withRuntime(rt)
    const plans = Array.from({ length: 5 }, (_, i) => df.lazy().filter(col('x').gt(i)).persist())
    await plans[0]!.collect()
    await plans[1]!.collect()
    await plans[2]!.collect()
    await plans[0]!.collect() // touch 0 → most recent
    await plans[3]!.collect() // evicts 1
    expect(cache.stats().entries).toBe(3)
    expect((await plans[0]!.collectWithReport()).report.cacheHit).toBe(true)
    expect((await plans[1]!.collectWithReport()).report.cacheHit).toBe(false)
    expect(cache.stats().evictions).toBeGreaterThanOrEqual(1)
  })

  it('ttlMs expires entries and pendingTtlMs expires marks (injectable clock)', async () => {
    let t = 0
    const cache = new PersistCache({ ttlMs: 1000, pendingTtlMs: 500, now: () => t })
    const rt = new Runtime({ persist: cache })
    const df = DataFrame.fromRows(rows).withRuntime(rt)
    const q = df.lazy().filter(col('x').gt(3)).persist()
    await q.collect()
    t = 999
    expect((await q.collectWithReport()).report.cacheHit).toBe(true)
    t = 1000
    expect((await q.collectWithReport()).report.cacheHit).toBe(false) // expired, recomputed (mark is gone too at t ≥ 500)
    df.lazy().filter(col('x').gt(99)).persist() // marked at t = 1000
    t = 1600
    expect(cache.stats().expired).toBeGreaterThanOrEqual(1)
    cache.probe(df.lazy().filter(col('x').gt(99)).plan)
    expect(cache.stats().pending).toBe(0)
  })

  it('hits and misses are counted; invalid limits are rejected', async () => {
    const cache = new PersistCache()
    const rt = new Runtime({ persist: cache })
    const q = DataFrame.fromRows(rows).withRuntime(rt).lazy().select('id').persist()
    await q.collect()
    await q.collect()
    await q.collect()
    expect(cache.stats()).toMatchObject({ hits: 2, misses: 1, entries: 1 })
    expect(() => new PersistCache({ maxEntries: -1 })).toThrow(RangeError)
  })
})
