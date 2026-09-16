import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import { controlChart, cusum, ewma, movingAverage, nelsonRules, spcConstants } from '@columna/advanced'

const ref = JSON.parse(readFileSync(new URL('./fixtures/tier4-scipy.json', import.meta.url), 'utf8'))
const close = (got: number, want: number, digits = 10) => expect(got).toBeCloseTo(want, digits)

function rng(seed: number) {
  let a = seed >>> 0
  const u = () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296
  }
  const normal = () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u())
  return { u, normal }
}

describe('4.1 spcConstants', () => {
  it('matches ASTM / Minitab table and c4 closed form', () => {
    for (const c of ref.spcConstants) {
      const g = spcConstants(c.n)
      close(g.d2, c.d2, 3)
      close(g.d3, c.d3, 3)
      close(g.c4, c.c4, 12)
      close(g.A2, c.A2, 10)
      close(g.A3, c.A3, 10)
      close(g.B3, c.B3, 10)
      close(g.B4, c.B4, c.B4 < 1e-12 ? 12 : 10)
      close(g.D3, c.D3, 10)
      close(g.D4, c.D4, 10)
    }
    expect(() => spcConstants(1)).toThrow(RangeError)
    // c4 → 1 as n → ∞; A3 → 0
    expect(spcConstants(100).c4).toBeGreaterThan(0.99)
    expect(spcConstants(100).A3).toBeLessThan(spcConstants(5).A3)
  })
})

describe('4.2 control charts', () => {
  it('I-MR center / limits match the MR/d2 formula', () => {
    const r = controlChart(ref.imr.x, { type: 'i-mr' })
    close(r.center, ref.imr.center, 10)
    close(r.sigma, ref.imr.sigma, 10)
    close(r.ucl, ref.imr.ucl, 10)
    close(r.lcl, ref.imr.lcl, 10)
    expect(r.companion).toBeDefined()
    close(r.companion!.center, ref.imr.mrBar, 10)
    close(r.companion!.ucl, ref.imr.mrUcl, 8)
  })

  it('X̄-R limits match A2 / D3 / D4', () => {
    const r = controlChart(ref.xbarR.groups, { type: 'xbar-r' })
    close(r.center, ref.xbarR.center, 10)
    close(r.ucl, ref.xbarR.ucl, 10)
    close(r.lcl, ref.xbarR.lcl, 10)
    close(r.companion!.ucl, ref.xbarR.rUcl, 10)
    close(r.companion!.lcl, ref.xbarR.rLcl, 10)
  })

  it('P chart matches binomial SE limits', () => {
    const r = controlChart(ref.pChart.counts, { type: 'p', sizes: ref.pChart.sizes })
    close(r.center, ref.pChart.center, 12)
    for (let i = 0; i < r.points.length; i++) {
      close(r.points[i]!.ucl, ref.pChart.ucl[i], 10)
      close(r.points[i]!.lcl, ref.pChart.lcl[i], 10)
    }
  })

  it('Nelson rule 1 fires beyond 3σ; rule 2 on nine same-side points', () => {
    const center = 0
    const sigma = 1
    const vals = [0, 0.1, -0.2, 0, 0.05, -0.1, 0.2, -0.05, 0.1, 0.15] // 10 points same side after shift
    const same = Array.from({ length: 12 }, () => 0.5)
    const r2 = nelsonRules(same, center, sigma, [2])
    expect(r2[8]!.includes(2) || r2[11]!.includes(2)).toBe(true)
    const spike = [0, 0, 0, 3.5, 0]
    expect(nelsonRules(spike, center, sigma, [1])[3]).toContain(1)
    void vals
  })

  it('df.controlChart installs on DataFrame', () => {
    const df = DataFrame.fromColumns({ x: ref.imr.x })
    const r = df.controlChart('x', { type: 'i-mr' })
    expect(r.type).toBe('i-mr')
    expect(r.points.length).toBe(ref.imr.x.length)
  })

  it('ARL0 under H0 is large; shift 1σ is detected (seeded)', () => {
    const g = rng(42)
    // Approximate ARL0 with Shewhart rule 1 only on I chart with known σ: theoretical 370.4
    let run = 0
    let hits = 0
    const trials = 2000
    for (let t = 0; t < trials; t++) {
      const x = Array.from({ length: 500 }, () => g.normal())
      const r = controlChart(x, { type: 'i-mr', sigma: 1, mu: 0, rules: [1] })
      if (r.outOfControl.length) {
        hits++
        run += r.outOfControl[0]! + 1
      } else {
        run += 500
      }
    }
    const arl0 = run / Math.max(1, hits)
    // With 500-cap the estimate is biased low; require ARL0 ≫ 100
    expect(arl0).toBeGreaterThan(200)

    // Power: 1σ mean shift — first hit should be much earlier
    let run1 = 0
    let hits1 = 0
    for (let t = 0; t < 800; t++) {
      const x = Array.from({ length: 200 }, () => 1 + g.normal())
      const r = controlChart(x, { type: 'i-mr', sigma: 1, mu: 0, rules: [1] })
      if (r.outOfControl.length) {
        hits1++
        run1 += r.outOfControl[0]! + 1
      } else run1 += 200
    }
    const arl1 = run1 / Math.max(1, hits1)
    expect(arl1).toBeLessThan(80)
    expect(arl1).toBeLessThan(arl0 / 3)
  })
})

describe('4.3 attribute / Laney', () => {
  it('C chart and Laney P′ produce finite limits', () => {
    const counts = [3, 5, 2, 8, 4, 6, 3, 7, 5, 4]
    const c = controlChart(counts, { type: 'c' })
    expect(c.ucl).toBeGreaterThan(c.center)
    expect(c.lcl).toBeGreaterThanOrEqual(0)
    const sizes = Array(10).fill(100)
    const laney = controlChart(counts, { type: 'laney-p', sizes })
    expect(laney.type).toBe('laney-p')
    expect(laney.sigma).toBeGreaterThan(0)
  })
})

describe('4.4 EWMA / CUSUM / MA', () => {
  it('EWMA path matches Lucas–Saccucci recursion', () => {
    const r = ewma(ref.ewma.x, { lambda: ref.ewma.lambda, L: ref.ewma.L, mu: 0, sigma: 1 })
    for (let i = 0; i < r.z.length; i++) {
      close(r.z[i]!, ref.ewma.z[i], 10)
      close(r.ucl[i]!, ref.ewma.ucl[i], 10)
      close(r.lcl[i]!, ref.ewma.lcl[i], 10)
    }
  })

  it('CUSUM signals a sustained shift; MA limits tighten with span', () => {
    const g = rng(7)
    const x = Array.from({ length: 60 }, (_, i) => (i < 30 ? 0 : 1.5) + g.normal())
    const cu = cusum(x, { mu: 0, sigma: 1, k: 0.5, h: 5 })
    expect(cu.outOfControl.some((i) => i >= 30)).toBe(true)
    const ma = movingAverage(Array.from({ length: 40 }, () => g.normal()), { span: 5, mu: 0, sigma: 1 })
    expect(ma.ucl[0]! - ma.lcl[0]!).toBeGreaterThan(ma.ucl[20]! - ma.lcl[20]!)
  })

  it('keeps size properties: EWMA rarely signals under H0 (seeded)', () => {
    const g = rng(99)
    let signals = 0
    const reps = 300
    for (let i = 0; i < reps; i++) {
      const x = Array.from({ length: 100 }, () => g.normal())
      if (ewma(x, { mu: 0, sigma: 1, lambda: 0.2, L: 3 }).outOfControl.length) signals++
    }
    // ARL0 for λ=0.2, L=3 is ~500 → P(signal in 100) ≈ 100/500 = 0.2
    expect(signals / reps).toBeLessThan(0.35)
    expect(signals / reps).toBeGreaterThan(0.05)
  })
})
