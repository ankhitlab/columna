/**
 * `Session`: per-tenant isolation of the three otherwise process-wide pieces of state.
 *  - persist(): a session's cache is invisible to other sessions and to the default runtime; close() drops it;
 *  - IO policy: the session's policy is a floor every read must pass, on top of setIoPolicy() and narrowed by
 *    per-call options — never widened by them;
 *  - runtime: engine / strict / memory are per session; `new Runtime({ memory })` no longer touches the process
 *    default; backends registered on the default runtime are visible to sessions.
 */
import { afterEach, describe, expect, it } from 'vitest'
import { DataFrame, PersistCache, Runtime, clearPersistCache, col, createSession, getDefaultRuntime, getMemoryPolicy, setIoPolicy } from '../src/index.js'
import { loadBytes } from '../src/io/source.js'
import { persistCacheStats } from '@columna/runtime'

afterEach(() => {
  setIoPolicy({})
  clearPersistCache()
})

const rows = Array.from({ length: 2000 }, (_, i) => ({ id: i, g: `g${i % 7}`, x: (i * 31) % 97 }))
const plan = (df: DataFrame) => df.lazy().groupBy('g').agg({ n: col('id').count(), s: col('x').sum() }).sort('g')

describe('Session persist cache isolation', () => {
  it('a persisted plan is served from the session cache only; other sessions and the default runtime miss', async () => {
    const a = createSession()
    const b = createSession()
    const dfA = a.fromRows(rows)
    const { report: first } = await plan(dfA).persist().collectWithReport()
    expect(first.cacheHit).toBe(false)
    const { report: second } = await plan(dfA).collectWithReport()
    expect(second.cacheHit).toBe(true)
    expect(a.persistCache.stats().entries).toBe(1)
    // the very same table and plan — but B's runtime has its own cache
    const { report: other } = await plan(b.bind(dfA)).collectWithReport()
    expect(other.cacheHit).toBe(false)
    expect(b.persistCache.stats().entries).toBe(0)
    // the process-wide cache never saw it either
    expect(persistCacheStats().entries).toBe(0)
    const { report: def } = await plan(dfA.withRuntime(getDefaultRuntime())).collectWithReport()
    expect(def.cacheHit).toBe(false)
    a.close()
    expect(a.persistCache.stats().entries).toBe(0)
    expect(() => a.fromRows(rows)).toThrow(/closed/)
    b.close()
  })

  it('persist: false shares the process-wide cache; a PersistCache instance can be shared explicitly', async () => {
    const shared = createSession({ persist: false })
    expect(shared.persistCache).toBe(getDefaultRuntime().persist)
    const cache = new PersistCache({ maxBytes: 1_000_000 })
    const s1 = createSession({ persist: cache })
    const s2 = createSession({ persist: cache })
    const df = DataFrame.fromRows(rows)
    await plan(s1.bind(df)).persist().collect()
    const { report } = await plan(s2.bind(df)).collectWithReport()
    expect(report.cacheHit).toBe(true)
    s1.close() // owns the (shared) cache → clears it
    expect(cache.stats().entries).toBe(0)
  })

  it('a per-session LRU cap evicts within that session only', async () => {
    const big = createSession({ persist: { maxBytes: 1 } })
    await plan(big.fromRows(rows)).persist().collect()
    expect(big.persistCache.stats().entries).toBe(0) // evicted immediately: nothing fits in 1 byte
    const normal = createSession()
    await plan(normal.fromRows(rows)).persist().collect()
    expect(normal.persistCache.stats().entries).toBe(1)
  })
})

describe('Session IO policy floor', () => {
  const ok = () => new Response('a,b\n1,2\n', { status: 200, headers: { 'content-type': 'text/csv' } })

  it('the session policy is enforced on top of the process floor, and per-call options cannot widen it', async () => {
    let fetched = 0
    const fetchImpl = (async () => (fetched++, ok())) as unknown as typeof fetch
    const tenant = createSession({ io: { allowedHosts: ['data.tenant-a.example'], maxBytes: 100, fetch: fetchImpl } })
    const df = await tenant.readCsv({ url: 'https://data.tenant-a.example/x.csv' })
    expect(df.shape).toEqual([1, 2])
    expect(df.getRuntime()).toBe(tenant.runtime)
    // another host: refused by the session floor even though the call allows it
    await expect(tenant.readCsv({ url: 'https://data.tenant-b.example/x.csv', allowedHosts: ['data.tenant-b.example'] })).rejects.toThrow(/allowedHosts/)
    // per-call maxBytes above the session cap does not widen it
    const bigFetch = (async () => new Response('a'.repeat(500), { status: 200 })) as unknown as typeof fetch
    await expect(loadBytes({ url: 'https://data.tenant-a.example/big.csv' }, tenant.ioOptions({ maxBytes: 10_000, fetch: bigFetch }))).rejects.toThrow(/maxBytes = 100/)
    // the process floor still applies underneath the session
    setIoPolicy({ allowedHosts: ['nothing.example'] })
    await expect(tenant.readCsv({ url: 'https://data.tenant-a.example/x.csv' })).rejects.toThrow(/allowedHosts/)
    expect(fetched).toBe(1)
  })

  it('two sessions with different policies do not see each other', async () => {
    const seen: string[] = []
    const fetchImpl = (async (url: string) => (seen.push(url), ok())) as unknown as typeof fetch
    const a = createSession({ io: { allowedHosts: ['a.example'], fetch: fetchImpl } })
    const b = createSession({ io: { allowedHosts: ['b.example'], fetch: fetchImpl } })
    await a.readCsv({ url: 'https://a.example/x.csv' })
    await b.readCsv({ url: 'https://b.example/x.csv' })
    await expect(a.readCsv({ url: 'https://b.example/x.csv' })).rejects.toThrow(/allowedHosts/)
    await expect(b.readCsv({ url: 'https://a.example/x.csv' })).rejects.toThrow(/allowedHosts/)
    expect(seen).toEqual(['https://a.example/x.csv', 'https://b.example/x.csv'])
  })
})

describe('Session runtime isolation', () => {
  it('memory / engine / strict are per session; new Runtime({ memory }) leaves the process policy alone', async () => {
    const before = getMemoryPolicy()
    const s = createSession({ runtime: { engine: 'cpu', strict: true, memory: { maxBytes: 16 * 1024, spill: true } } })
    expect(getMemoryPolicy()).toEqual(before)
    const { report } = await s.fromRows(rows).lazy().sort('x').collectWithReport()
    expect(report.requested).toBe('cpu')
    expect(report.strict).toBe(true)
    expect(report.events.map((e) => e.kernel)).toContain('js:sort+spill') // the session's budget applied
    const { report: plain } = await DataFrame.fromRows(rows).lazy().sort('x').collectWithReport()
    expect(plain.events.map((e) => e.kernel)).not.toContain('js:sort+spill')
    expect(new Runtime({ memory: { maxBytes: 1 } }).memoryPolicy?.maxBytes).toBe(1)
    expect(getMemoryPolicy()).toEqual(before)
  })

  it('bind / withRuntime move a frame between runtimes; sessions see the default runtime backends', () => {
    const s = createSession()
    const df = DataFrame.fromRows(rows)
    expect(df.getRuntime()).toBe(getDefaultRuntime())
    expect(s.bind(df).getRuntime()).toBe(s.runtime)
    expect(s.bind(df.lazy()).getRuntime()).toBe(s.runtime)
    expect(df.getRuntime()).toBe(getDefaultRuntime()) // the original is untouched
    const names = s.runtime.listBackends().map((b) => b.name)
    expect(names).toEqual(getDefaultRuntime().listBackends().map((b) => b.name))
    expect(createSession({ backends: [] }).runtime.listBackends().map((b) => b.name)).toEqual(['cpu'])
  })
})
