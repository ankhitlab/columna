import { describe, expect, it } from 'vitest'
import { DataFrame, col, lit } from '@columna/core'

/** Reference: Hyndman–Fan type 6 (Minitab) and type 7 (pandas) on a sorted copy. */
function refQuantile(values: number[], p: number, method: 'linear' | 'minitab'): number {
  const s = [...values].sort((a, b) => a - b)
  const n = s.length
  const pos = method === 'minitab' ? Math.min(n - 1, Math.max(0, p * (n + 1) - 1)) : (n - 1) * p
  const lo = Math.floor(pos)
  const hi = Math.ceil(pos)
  return s[lo]! + (pos - lo) * (s[hi]! - s[lo]!)
}

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

describe('Minitab quantile method (type 6, position p(n + 1))', () => {
  it('textbook values: quartiles of 1..4 and 1..7, clamping at the ends', async () => {
    const df4 = DataFrame.fromColumns({ x: new Float64Array([4, 1, 3, 2]) })
    const r4 = (
      await df4
        .select(
          col('x').quantile(0.25).alias('q1_lin'),
          col('x').quantile(0.25, 'minitab').alias('q1_mtb'),
          col('x').quantile(0.75).alias('q3_lin'),
          col('x').quantile(0.75, 'minitab').alias('q3_mtb'),
          col('x').median().alias('med_lin'),
          col('x').median('minitab').alias('med_mtb'),
          col('x').quantile(0.05, 'minitab').alias('lo'),
          col('x').quantile(0.99, 'minitab').alias('hi'),
          col('x').quantile(0.25).sub(col('x').quantile(0.25, 'minitab')).alias('diff'),
        )
        .collect()
    ).toArray()[0]!
    expect(r4.q1_lin).toBe(1.75)
    expect(r4.q1_mtb).toBe(1.25)
    expect(r4.q3_lin).toBe(3.25)
    expect(r4.q3_mtb).toBe(3.75)
    expect(r4.med_lin).toBe(2.5)
    expect(r4.med_mtb).toBe(2.5)
    expect(r4.lo).toBe(1) // 0.05 · 5 = 0.25 < 1 → smallest value
    expect(r4.hi).toBe(4) // 0.99 · 5 = 4.95 > 4 → largest value
    expect(r4.diff).toBe(0.5) // same q, different definitions must not share a memo slot

    // n = 7 integers → histogram path in describe and grouped aggregates
    const df7 = DataFrame.fromColumns({ x: new Int32Array([7, 1, 4, 2, 6, 3, 5]) })
    const d = Object.fromEntries((await df7.describe({ quantileMethod: 'minitab' }).collect()).toArray().map((r) => [String(r.stat), Number(r.x)]))
    expect(d['25%']).toBe(2) // rank 0.25 · 8 = 2
    expect(d['50%']).toBe(4)
    expect(d['75%']).toBe(6)
    const dl = Object.fromEntries((await df7.describe().collect()).toArray().map((r) => [String(r.stat), Number(r.x)]))
    expect(dl['25%']).toBe(2.5)
    expect(dl['75%']).toBe(5.5)
  })

  it('agrees with the reference on random data through every engine path', async () => {
    const rng = mulberry32(20240915)
    const n = 3000
    const g: string[] = []
    const xf: number[] = []
    const xi: number[] = []
    for (let i = 0; i < n; i++) {
      g.push('g' + Math.floor(rng() * 4))
      xf.push(Math.round(rng() * 1000) / 7)
      xi.push(Math.floor(rng() * 40))
    }
    const df = DataFrame.fromColumns({ g, xf: new Float64Array(xf), xi: new Int32Array(xi) })
    const byGroup = (arr: number[]) => {
      const m = new Map<string, number[]>()
      arr.forEach((v, i) => {
        const key = g[i]!
        if (!m.has(key)) m.set(key, [])
        m.get(key)!.push(v)
      })
      return m
    }
    const gf = byGroup(xf)
    const gi = byGroup(xi)
    const ps = [0.1, 0.25, 0.5, 0.75, 0.9]

    for (const method of ['linear', 'minitab'] as const) {
      // whole-column broadcast (typed path) and expression inner (boxed path)
      const sel: Record<string, ReturnType<typeof col>> = {}
      for (const p of ps) {
        sel[`f${p}`] = col('xf').quantile(p, method)
        sel[`i${p}`] = col('xi').quantile(p, method)
        sel[`e${p}`] = col('xf').mul(lit(2)).quantile(p, method)
      }
      const row = (await df.select(...Object.entries(sel).map(([k, e]) => e.alias(k))).collect()).toArray()[0]!
      for (const p of ps) {
        expect(row[`f${p}`]).toBeCloseTo(refQuantile(xf, p, method), 10)
        expect(row[`i${p}`]).toBeCloseTo(refQuantile(xi, p, method), 10)
        expect(row[`e${p}`]).toBeCloseTo(2 * refQuantile(xf, p, method), 10)
      }

      // groupBy: float value store, integer histogram, and non-column inner (generic path)
      const grouped = (
        await df
          .groupBy('g')
          .agg({
            qf: col('xf').quantile(0.25, method),
            qi: col('xi').quantile(0.9, method),
            qe: col('xi').add(lit(1)).quantile(0.75, method),
            md: col('xf').median(method),
          })
          .sort('g')
          .collect()
      ).toArray()
      for (const r of grouped) {
        const key = String(r.g)
        expect(r.qf).toBeCloseTo(refQuantile(gf.get(key)!, 0.25, method), 10)
        expect(r.qi).toBeCloseTo(refQuantile(gi.get(key)!, 0.9, method), 10)
        expect(r.qe).toBeCloseTo(refQuantile(gi.get(key)!.map((v) => v + 1), 0.75, method), 10)
        expect(r.md).toBeCloseTo(refQuantile(gf.get(key)!, 0.5, method), 10)
      }

      // window: per-partition quantile broadcast to rows
      const win = (await df.withColumn('w', col('xf').quantile(0.75, method).over('g')).collect()).toArray()
      for (let i = 0; i < n; i += 97) expect(win[i]!.w).toBeCloseTo(refQuantile(gf.get(g[i]!)!, 0.75, method), 10)

      // describe: sorted (n ≤ 10 000) fallback for floats, histogram for ints
      const desc = (await df.describe({ quantileMethod: method }).collect()).toArray()
      const stat = (name: string, c: string) => Number(desc.find((r) => r.stat === name)![c])
      expect(stat('25%', 'xf')).toBeCloseTo(refQuantile(xf, 0.25, method), 10)
      expect(stat('75%', 'xi')).toBeCloseTo(refQuantile(xi, 0.75, method), 10)
    }

    // describe select path (n > 10 000 floats)
    const big: number[] = []
    for (let i = 0; i < 20_001; i++) big.push(rng() * 100)
    const dbig = (await DataFrame.fromColumns({ v: new Float64Array(big) }).describe({ quantileMethod: 'minitab' }).collect()).toArray()
    // n = 20 001 puts every linear position on an integer rank (5000 / 10000 / 15000): regression for the
    // quickselect branch, which returned 0 there.
    const dlin = (await DataFrame.fromColumns({ v: new Float64Array(big) }).describe().collect()).toArray()
    for (const [name, p] of [['25%', 0.25], ['50%', 0.5], ['75%', 0.75]] as const) {
      expect(Number(dbig.find((r) => r.stat === name)!.v)).toBeCloseTo(refQuantile(big, p, 'minitab'), 10)
      expect(Number(dlin.find((r) => r.stat === name)!.v)).toBeCloseTo(refQuantile(big, p, 'linear'), 10)
    }
  })
})
