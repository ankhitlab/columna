import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  analyzeTaguchi,
  arima,
  ets,
  reliabilityFit,
  taguchi,
  treeNet,
} from '@columna/advanced'

const ref = JSON.parse(readFileSync(new URL('./fixtures/tier5-scipy.json', import.meta.url), 'utf8'))
const close = (got: number, want: number, digits = 8) => expect(got).toBeCloseTo(want, digits)

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
  const exponential = (mean: number) => -mean * Math.log(u())
  const weibull = (shape: number, scale: number) => scale * Math.pow(-Math.log(u()), 1 / shape)
  return { u, normal, exponential, weibull }
}

describe('tier5 deepen — layer 1 fixtures', () => {
  it('Weibull complete MLE near scipy fit; SE/CI present', () => {
    const fit = reliabilityFit(ref.weibull_complete.time, { distribution: 'weibull' })
    close(fit.shape!, ref.weibull_complete.shape_mle, 1)
    close(fit.scale, ref.weibull_complete.scale_mle, 1)
    expect(fit.se?.shape).toBeGreaterThan(0)
    expect(fit.se?.scale).toBeGreaterThan(0)
    expect(fit.ci?.shape![0]).toBeLessThan(fit.shape!)
    expect(fit.ci?.shape![1]).toBeGreaterThan(fit.shape!)
    expect(fit.vcov?.length).toBe(2)
  })

  it('Taguchi L9 larger S/N matches fixture; A ranked first with best level 3', () => {
    const d = taguchi('L9', ref.taguchi_l9.factors)
    expect(d.matrix.length).toBe(9)
    const an = analyzeTaguchi(d, ref.taguchi_l9.outer, { snRatio: 'larger' })
    for (let i = 0; i < 9; i++) close(an.sn[i]!, ref.taguchi_l9.sn_larger[i], 4)
    expect(an.ranking[0]).toBe('A')
    const fa = an.factors.find((f) => f.factor === 'A')!
    expect(fa.bestLevel).toBe(ref.taguchi_l9.best_A)
    close(fa.meanSN[0]!, ref.taguchi_l9.sn_by_A['1'], 3)
    close(fa.meanSN[2]!, ref.taguchi_l9.sn_by_A['3'], 3)
  })

  it('canonical OA run counts L4–L27', () => {
    expect(taguchi('L4').matrix.length).toBe(4)
    expect(taguchi('L8').matrix[0]!.length).toBe(7)
    expect(taguchi('L12').matrix.length).toBe(12)
    expect(taguchi('L16').matrix.length).toBe(16)
    expect(taguchi('L18').matrix.length).toBe(18)
    expect(taguchi('L27').matrix.length).toBe(27)
    expect(taguchi('L27').matrix[0]!.length).toBe(13)
  })

  it('SES / ARIMA expose forecast intervals', () => {
    const e = ets(ref.ses.y, { method: 'ses', alpha: ref.ses.alpha, horizon: 5 })
    expect(e.forecastLower!.length).toBe(5)
    expect(e.forecastUpper!.length).toBe(5)
    for (let h = 0; h < 5; h++) {
      expect(e.forecastLower![h]!).toBeLessThan(e.forecast[h]!)
      expect(e.forecastUpper![h]!).toBeGreaterThan(e.forecast[h]!)
    }
    const fit = arima(ref.ar1.y, { p: 1, d: 0, q: 0, method: 'CSS-ML', horizon: 3 })
    expect(fit.method).toBe('CSS-ML')
    expect(fit.forecastLower!.length).toBe(3)
    expect(fit.se.length).toBeGreaterThan(0)
    expect(fit.ci[0]![0]).toBeLessThan(fit.ar[0]!)
  })
})

describe('tier5 deepen — layer 2 invariants', () => {
  it('exponential MLE handles left and interval censoring without NaN', () => {
    // censor: 0 exact, 1 right, 2 left, 3 interval (time2 = upper)
    const time = [10, 20, 30, 40, 50, 15, 25, 30]
    const censor = [0, 0, 0, 1, 1, 2, 3, 3]
    const time2 = [NaN, NaN, NaN, NaN, NaN, NaN, 40, 50]
    const fit = reliabilityFit(time, { distribution: 'exponential', censor, time2 })
    expect(Number.isFinite(fit.scale)).toBe(true)
    expect(Number.isFinite(fit.logLik)).toBe(true)
    expect(fit.se?.scale).toBeGreaterThan(0)
  })

  it('ARIMA CSS-ML recovers AR(1) and MA(1); stationarity |φ|<1', () => {
    const ar = arima(ref.ar1.y, { p: 1, d: 0, q: 0, method: 'CSS-ML' })
    expect(Math.abs(ar.ar[0]! - 0.7)).toBeLessThan(0.12)
    expect(Math.abs(ar.ar[0]!)).toBeLessThan(1)
    expect(ar.logLik).toBeLessThan(0)

    const ma = arima(ref.ma1.y, { p: 0, d: 0, q: 1, method: 'CSS-ML' })
    expect(Math.abs(ma.ma[0]! - 0.5)).toBeLessThan(0.2)
  })

  it('TreeNet classification: predictProba sums to 1; accuracy ≥ 0.9 on separable data', () => {
    const tn = treeNet(ref.treenet_clf.X, ref.treenet_clf.y, {
      task: 'classification',
      nTrees: 40,
      learningRate: 0.15,
      maxDepth: 2,
      seed: 4,
    })
    expect(tn.trainAccuracy!).toBeGreaterThanOrEqual(0.9)
    const proba = tn.predictProba!(ref.treenet_clf.X)
    for (const row of proba) {
      const s = Object.values(row).reduce((a, b) => a + b, 0)
      close(s, 1, 6)
    }
  })

  it('Taguchi Δ(S/N) ranking is consistent with meanSN spread', () => {
    const d = taguchi('L9', ['A', 'B', 'C', 'D'])
    const an = analyzeTaguchi(d, ref.taguchi_l9.outer, { snRatio: 'larger' })
    const deltas = an.factors.map((f) => f.deltaSN)
    const sorted = [...deltas].sort((a, b) => b - a)
    expect(an.factors.find((f) => f.factor === an.ranking[0])!.deltaSN).toBe(sorted[0])
  })
})

describe('tier5 deepen — layer 3 Monte-Carlo', () => {
  it('Weibull 90% CI covers truth ≥ 85% over seeded MC', () => {
    const g = rng(4242)
    const shape = 1.8
    const scale = 80
    let cover = 0
    const N = 40
    for (let i = 0; i < N; i++) {
      const time = Array.from({ length: 60 }, () => g.weibull(shape, scale))
      const fit = reliabilityFit(time, { distribution: 'weibull', confidence: 0.9 })
      const [lo, hi] = fit.ci!.shape!
      if (lo <= shape && shape <= hi) cover++
    }
    expect(cover / N).toBeGreaterThanOrEqual(0.85)
  })

  it('ETS PI widen with horizon; ARIMA PI contain ~h=1 forecast band', () => {
    const g = rng(7)
    const y = Array.from({ length: 60 }, (_, i) => 5 + 0.05 * i + g.normal() * 0.5)
    const e = ets(y, { method: 'des', horizon: 8 })
    const w1 = e.forecastUpper![0]! - e.forecastLower![0]!
    const w8 = e.forecastUpper![7]! - e.forecastLower![7]!
    expect(w8).toBeGreaterThan(w1)

    const a = arima(ref.ar1.y, { p: 1, d: 0, q: 0, horizon: 5 })
    const aw1 = a.forecastUpper![0]! - a.forecastLower![0]!
    const aw5 = a.forecastUpper![4]! - a.forecastLower![4]!
    expect(aw5).toBeGreaterThanOrEqual(aw1 * 0.99)
  })
})
