import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import { boxCoxLambda, capability, johnsonFit, toleranceInterval, weibullFit } from '@columna/advanced'

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

describe('4.5 capability', () => {
  it('matches Pp / Ppk / within Cp on the fixture', () => {
    const r = capability(ref.capability.x, { lsl: ref.capability.lsl, usl: ref.capability.usl, subgroup: 5 })
    close(r.mean, ref.capability.mean, 10)
    close(r.sigmaOverall, ref.capability.sigmaOverall, 10)
    close(r.Pp!, ref.capability.Pp, 8)
    close(r.Ppk!, ref.capability.Ppk, 8)
    close(r.sigmaWithin, ref.capability.sigmaWithin_R, 6)
    close(r.Cp!, ref.capability.Cp, 6)
    close(r.Cpk!, ref.capability.Cpk, 6)
    expect(r.ppmOverall).toBeGreaterThan(0)
    expect(r.zBench).toBeDefined()
    expect(r.cpkCI).toBeDefined()
    expect(r.cpkCI![0]!).toBeLessThan(r.Cpk!)
    expect(r.cpkCI![1]!).toBeGreaterThan(r.Cpk!)
  })

  it('recovers Cp ≈ 1 for N(μ,σ) with limits μ±3σ (Monte-Carlo)', () => {
    const g = rng(11)
    let sumCp = 0
    let cover = 0
    const reps = 400
    for (let i = 0; i < reps; i++) {
      const x = Array.from({ length: 100 }, () => g.normal())
      const r = capability(x, { lsl: -3, usl: 3 })
      sumCp += r.Pp!
      // true Cpk = 1; CI should cover often
      if (r.cpkCI![0]! <= 1 && 1 <= r.cpkCI![1]!) cover++
    }
    expect(sumCp / reps).toBeGreaterThan(0.9)
    expect(sumCp / reps).toBeLessThan(1.1)
    expect(cover / reps).toBeGreaterThan(0.9)
  })

  it('df.capability works', () => {
    const df = DataFrame.fromColumns({ x: ref.capability.x })
    expect(df.capability('x', { lsl: -3, usl: 3 }).Pp).toBeGreaterThan(0.8)
  })

  it('rejects missing limits and usl ≤ lsl', () => {
    expect(() => capability([1, 2, 3], {})).toThrow(RangeError)
    expect(() => capability([1, 2, 3], { lsl: 5, usl: 1 })).toThrow(RangeError)
  })
})

describe('4.6 Box-Cox / Johnson / Weibull', () => {
  it('Box-Cox λ matches the profile-likelihood grid (± scipy)', () => {
    const r = boxCoxLambda(ref.boxcox.x)
    close(r.lambda, ref.boxcox.lambda_grid, 1)
    // scipy uses a continuous optimiser — within 0.15 of the grid max
    expect(Math.abs(r.lambda - ref.boxcox.lambda_scipy)).toBeLessThan(0.15)
  })

  it('Weibull MLE recovers scipy shape / scale', () => {
    const r = weibullFit(ref.weibull.x)
    close(r.shape, ref.weibull.shape, 3)
    close(r.scale, ref.weibull.scale, 3)
  })

  it('recovers planted λ and Weibull params (seeded)', () => {
    const g = rng(21)
    // λ = 0 → lognormal-ish via exp; plant λ=0.5: y = (x^0.5) roughly by transforming normals
    const z = Array.from({ length: 120 }, () => 2 + 0.4 * g.normal())
    const x = z.map((zi) => zi ** 2) // so λ≈0.5 brings back near-normal
    const bc = boxCoxLambda(x)
    expect(Math.abs(bc.lambda - 0.5)).toBeLessThan(0.25)

    // Weibull(1.8, 12)
    const w = Array.from({ length: 80 }, () => {
      const u = Math.max(1e-12, g.u())
      return 12 * (-Math.log(u)) ** (1 / 1.8)
    })
    const fit = weibullFit(w)
    expect(Math.abs(fit.shape - 1.8)).toBeLessThan(0.35)
    expect(Math.abs(fit.scale - 12)).toBeLessThan(2)

    const jf = johnsonFit(Array.from({ length: 100 }, () => g.normal()))
    expect(['SU', 'SB', 'SL']).toContain(jf.family)
    expect(jf.z.length).toBe(100)
  })
})

describe('4.7 tolerance intervals', () => {
  it('normal k-factor matches Howe formula', () => {
    const r = toleranceInterval(ref.tolerance.x, { coverage: 0.95, confidence: 0.95 })
    close(r.k!, ref.tolerance.k, 10)
    close(r.interval[0], ref.tolerance.interval[0], 8)
    close(r.interval[1], ref.tolerance.interval[1], 8)
  })

  it('covers ≥95% of a normal population about 95% of the time (seeded)', () => {
    const g = rng(33)
    let cover = 0
    const reps = 500
    // Population interval for N(0,1) covering 95%: ±1.96
    for (let i = 0; i < reps; i++) {
      const x = Array.from({ length: 40 }, () => g.normal())
      const ti = toleranceInterval(x, { coverage: 0.95, confidence: 0.95 })
      // Check that the interval covers at least 95% mass: Φ(hi)−Φ(lo) ≥ 0.95 for true N(0,1)
      // Approximate by requiring lo ≤ −1.645 and hi ≥ 1.645 is too strict; use Monte-Carlo pop sample
      const pop = Array.from({ length: 2000 }, () => g.normal())
      const frac = pop.filter((v) => v >= ti.interval[0] && v <= ti.interval[1]).length / pop.length
      if (frac >= 0.95) cover++
    }
    expect(cover / reps).toBeGreaterThan(0.9)
    expect(cover / reps).toBeLessThan(0.99)
  })

  it('nonparametric returns order-statistic bounds inside the sample range', () => {
    const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    const r = toleranceInterval(x, { method: 'nonparametric', coverage: 0.9, confidence: 0.9 })
    expect(r.interval[0]).toBeGreaterThanOrEqual(1)
    expect(r.interval[1]).toBeLessThanOrEqual(20)
  })
})
