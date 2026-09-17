import { describe, expect, it } from 'vitest'
import { DataFrame, LazyFrame, col, executeCpu } from 'columna'
import { Runtime } from '@columna/runtime'
import { WasmBackend, readParquetLike } from '@columna/wasm'

/**
 * Data-preservation invariants on seeded random frames with every dtype, nulls, ties, duplicates and awkward
 * names: identities that must hold regardless of the kernel path taken (dense / hashed / native / fused),
 * format round-trips, and the requested-vs-actual engine equivalence.
 */
function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

type R = { id: number; k: number; g: string; s: string; f: number | null; i: number | null; b: boolean; 'we ird,name': number }
function randomRows(seed: number, n: number): R[] {
  const rnd = mulberry32(seed)
  const groups = ['a', 'b', 'c', '', 'ünï', 'x"y']
  return Array.from({ length: n }, (_, id) => ({
    id,
    k: Math.floor(rnd() * 7), // heavy ties
    g: groups[Math.floor(rnd() * groups.length)]!,
    s: rnd() < 0.3 ? 'dup' : `s${Math.floor(rnd() * 5000)}`, // high cardinality with duplicates
    f: rnd() < 0.15 ? null : Math.round((rnd() - 0.5) * 1e6) / 1000,
    i: rnd() < 0.1 ? null : Math.floor((rnd() - 0.5) * 2 ** 32),
    b: rnd() < 0.5,
    'we ird,name': rnd(),
  }))
}
const sum = (xs: Array<number | null>) => xs.reduce<number>((s, v) => s + (v ?? 0), 0)
const sortRows = (rows: Array<Record<string, unknown>>) => [...rows].sort((a, b) => Number(a.id) - Number(b.id))

describe('invariants on seeded random frames', () => {
  const seeds = [1, 2, 3]

  it('groupBy sums / counts partition the column sums; filter ∪ complement is the frame; sort is a permutation', async () => {
    for (const seed of seeds) {
      const rows = randomRows(seed, 5000)
      const df = DataFrame.fromRows(rows)
      const g = (await df.groupBy('g').agg({ n: col('id').count(), f: col('f').sum(), i: col('i').sum(), k: col('k').sum() }).collect()).toArray()
      expect(sum(g.map((r) => r.n))).toBe(rows.length)
      expect(sum(g.map((r) => r.f))).toBeCloseTo(sum(rows.map((r) => r.f)), 6)
      expect(sum(g.map((r) => r.i))).toBe(sum(rows.map((r) => r.i)))
      expect(sum(g.map((r) => r.k))).toBe(sum(rows.map((r) => r.k)))
      // multi-key groupBy (dense mixed-radix path) partitions the same way
      const g2 = (await df.groupBy('g', 'k').agg({ n: col('id').count() }).collect()).toArray()
      expect(sum(g2.map((r) => r.n))).toBe(rows.length)
      expect(g2.length).toBe(new Set(rows.map((r) => `${r.g}\0${r.k}`)).size)

      const yes = (await df.filter(col('f').gt(0)).collect()).toArray()
      const no = (await df.filter(col('f').lte(0)).collect()).toArray()
      const nulls = (await df.filter(col('f').isNull()).collect()).toArray()
      expect(yes.length + no.length + nulls.length).toBe(rows.length)
      expect(sortRows([...yes, ...no, ...nulls])).toEqual(rows)

      const sorted = (await df.sort(col('f').desc(), 'id').collect()).toArray()
      expect(sortRows(sorted)).toEqual(rows)
      for (let i = 1; i < sorted.length; i++) {
        const a = sorted[i - 1]!.f
        const b = sorted[i]!.f
        if (a !== null && b !== null) expect(a).toBeGreaterThanOrEqual(b)
        if (a === null) expect(b).toBeNull() // nulls sort last, together
      }
    }
  })

  it('unique / valueCounts / nunique agree with each other and with a Set; slice ∘ concat is the identity', async () => {
    for (const seed of seeds) {
      const rows = randomRows(seed, 4000)
      const df = DataFrame.fromRows(rows)
      const distinct = new Set(rows.map((r) => r.s)).size
      expect((await df.unique(['s']).collect()).shape[0]).toBe(distinct)
      const vc = (await df.valueCounts('s').collect()).toArray()
      expect(vc.length).toBe(distinct)
      expect(sum(vc.map((r) => Number(r.count)))).toBe(rows.length)
      expect((await df.select(col('s').nunique().alias('n')).collect()).toArray()[0]!.n).toBe(distinct)

      const parts = [df.slice(0, 1000), df.slice(1000, 2500), df.slice(2500)]
      const back = await LazyFrame.concat(parts).collect()
      expect(back.toArray()).toEqual(rows)
      expect(back.dtypes).toEqual(df.dtypes)
    }
  })

  it('join invariants: inner ⊆ left, |left| = |frame|, semi + anti partition the frame, and key equality holds row by row', async () => {
    for (const seed of seeds) {
      const rows = randomRows(seed, 3000)
      const df = DataFrame.fromRows(rows)
      const right = DataFrame.fromRows([
        { g: 'a', region: 'A' },
        { g: 'c', region: 'C' },
        { g: '', region: 'EMPTY' },
        { g: 'zzz', region: 'NONE' },
      ])
      const inner = (await df.join(right, { on: 'g' }).collect()).toArray()
      const left = (await df.leftJoin(right, 'g').collect()).toArray()
      const semi = (await df.semiJoin(right, 'g').collect()).toArray()
      const anti = (await df.antiJoin(right, 'g').collect()).toArray()
      const matched = new Set(['a', 'c', ''])
      expect(inner.length).toBe(rows.filter((r) => matched.has(r.g)).length)
      expect(left.length).toBe(rows.length)
      expect(semi.length + anti.length).toBe(rows.length)
      expect(sortRows([...semi, ...anti])).toEqual(rows)
      for (const r of inner) expect(r.region).toBe({ a: 'A', c: 'C', '': 'EMPTY' }[String(r.g)])
      for (const r of left) expect(r.region).toBe(matched.has(String(r.g)) ? { a: 'A', c: 'C', '': 'EMPTY' }[String(r.g)] : null)
    }
  })

  it('format round-trips preserve values and dtypes: CSV, JSON, parquet-like, Arrow-like', async () => {
    for (const seed of seeds) {
      const rows = randomRows(seed, 1500)
      const df = DataFrame.fromRows(rows)
      // CSV: numbers, nulls, quotes, commas and spaces in names survive; booleans come back as booleans
      const csv = DataFrame.fromCSV(df.toCsv())
      expect(csv.columns).toEqual(df.columns)
      expect(csv.toArray()).toEqual(rows)
      expect(csv.dtypes).toEqual(df.dtypes)
      // JSON records
      const json = DataFrame.fromJSON(JSON.stringify(rows))
      expect(json.toArray()).toEqual(rows)
      // parquet-like (columna's own JSON container)
      const pq = new DataFrame(readParquetLike(await df.writeParquetLike()))
      expect(pq.toArray()).toEqual(rows)
      expect(pq.dtypes).toEqual(df.dtypes)
      // Arrow-like
      const arrow = DataFrame.fromArrow(df.toArrow())
      expect(arrow.toArray()).toEqual(rows)
      expect(arrow.dtypes).toEqual(df.dtypes)
    }
  })

  it('cross-engine equivalence: the same plans give identical rows on the cpu engine, the wasm engine and executeCpu directly', async () => {
    const rt = new Runtime({ wasmMinRows: 1 })
    rt.register(new WasmBackend(executeCpu))
    for (const seed of seeds) {
      const rows = randomRows(seed, 6000)
      const df = DataFrame.fromRows(rows)
      const plans = [
        df.filter(col('k').gt(2).and(col('f').gt(-100))),
        df.filter(col('i').gt(0).and(col('k').lt(5))).select('id', 'i', 'k'),
        df.groupBy('g').agg({ n: col('id').count(), m: col('f').mean(), x: col('i').max() }),
        df.sort(col('k').desc(), col('id').asc()).head(500),
        df.withColumn('z', col('f').sub(col('f').mean()).div(col('f').std())).select('id', 'z'),
        df.unique(['k', 'g']),
      ]
      for (const plan of plans) {
        const cpu = (await plan.engine('cpu').collect()).toArray()
        const wasm = (await new LazyFrame(plan.plan, rt.withEngine('wasm')).collect()).toArray()
        const direct = new DataFrame(executeCpu(plan.plan)).toArray()
        expect(wasm).toEqual(cpu)
        expect(direct).toEqual(cpu)
      }
    }
  })
})
