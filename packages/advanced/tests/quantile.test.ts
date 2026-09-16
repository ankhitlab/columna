import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DataFrame, col } from '@columna/core'
import { quantile, descriptiveStats, random } from '@columna/advanced'

const ref = JSON.parse(readFileSync(new URL('./fixtures/tier8-scipy.json', import.meta.url), 'utf8'))
const close = (got: number, want: number, digits = 10) => expect(got).toBeCloseTo(want, digits)

describe('quantile definitions: Minitab p(n + 1) vs pandas (n − 1)p', () => {
  const q = ref.quantiles as { x: number[]; p: number[]; minitab: number[]; linear: number[] }

  it('quantile() matches numpy method="weibull" (Minitab) and "linear" (pandas)', () => {
    const mtb = quantile(q.x, q.p)
    const lin = quantile(q.x, q.p, { method: 'linear' })
    q.p.forEach((_, i) => {
      close(mtb[i]!, q.minitab[i]!)
      close(lin[i]!, q.linear[i]!)
      close(quantile(q.x, q.p[i]!), q.minitab[i]!)
    })
    // agrees with descriptiveStats quartiles (same definition)
    const d = descriptiveStats(q.x) as { q1: number; median: number; q3: number }
    close(d.q1, quantile(q.x, 0.25))
    close(d.median, quantile(q.x, 0.5))
    close(d.q3, quantile(q.x, 0.75))
    // nulls / NaN skipped, p range checked, empty → NaN
    close(quantile([null, 3, NaN, 1, undefined, 2, 4], 0.25), 1.25)
    expect(() => quantile([1, 2], 1.5)).toThrow(RangeError)
    expect(Number.isNaN(quantile([], 0.5))).toBe(true)
  })

  it('DataFrame engine: col().quantile(p, "minitab") and describe({ quantileMethod }) reproduce numpy', async () => {
    const df = DataFrame.fromColumns({ x: new Float64Array(q.x) })
    const exprs = q.p.flatMap((p, i) => [
      col('x').quantile(p, 'minitab').alias(`m${i}`),
      col('x').quantile(p).alias(`l${i}`),
    ])
    const row = (await df.select(...exprs).collect()).toArray()[0]!
    q.p.forEach((_, i) => {
      close(Number(row[`m${i}`]), q.minitab[i]!)
      close(Number(row[`l${i}`]), q.linear[i]!)
    })
    const desc = (await df.describe({ quantileMethod: 'minitab' }).collect()).toArray()
    const stat = (name: string) => Number(desc.find((r) => r.stat === name)!.x)
    close(stat('25%'), quantile(q.x, 0.25))
    close(stat('50%'), quantile(q.x, 0.5))
    close(stat('75%'), quantile(q.x, 0.75))
  })

  it('properties on random samples: monotone in p, bracketed by min / max, both definitions agree on the median of odd n', () => {
    const rng = random(77)
    for (let trial = 0; trial < 50; trial++) {
      const n = 3 + Math.floor(rng.uniform(1)[0]! * 60)
      const x = rng.normal(n)
      const ps = [0, 0.05, 0.2, 0.5, 0.8, 0.95, 1]
      for (const method of ['minitab', 'linear'] as const) {
        const qs = quantile(x, ps, { method })
        for (let i = 1; i < qs.length; i++) expect(qs[i]!).toBeGreaterThanOrEqual(qs[i - 1]!)
        expect(qs[0]).toBe(Math.min(...x))
        expect(qs[qs.length - 1]).toBe(Math.max(...x))
      }
      if (n % 2 === 1) close(quantile(x, 0.5), quantile(x, 0.5, { method: 'linear' }), 12)
      // Minitab quantile is never closer to the centre than the pandas one: p(n + 1) − 1 ≤ (n − 1)p for p ≤ 0.5 and ≥ for p ≥ 0.5
      expect(quantile(x, 0.25)).toBeLessThanOrEqual(quantile(x, 0.25, { method: 'linear' }) + 1e-12)
      expect(quantile(x, 0.75)).toBeGreaterThanOrEqual(quantile(x, 0.75, { method: 'linear' }) - 1e-12)
    }
  })
})
