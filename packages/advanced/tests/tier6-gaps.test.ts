import { describe, expect, it } from 'vitest'
import {
  analyzeMixture,
  arima,
  autoArima,
  coxPH,
  definitiveScreening,
  logRank,
  mixedModel,
  mixtureDesign,
  responseOptimizer,
  taguchi,
} from '@columna/advanced'

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
  return { u, normal, exponential }
}

describe('tier6 — survival Cox / weighted log-rank', () => {
  it('Cox agrees in sign with log-rank; Wilcoxon/strata finite', () => {
    const g = rng(7)
    const timeA = Array.from({ length: 40 }, () => g.exponential(20))
    const timeB = Array.from({ length: 40 }, () => g.exponential(55))
    const time = [...timeA, ...timeB]
    const group = [...Array(40).fill(0), ...Array(40).fill(1)]
    const X = group.map((v) => [v])
    const lr = logRank(time, group)
    const cox = coxPH(time, X, { names: ['group'] })
    // group 1 has longer mean survival → lower hazard → coef < 0
    expect(cox.coefficients[0]!).toBeLessThan(0)
    expect(lr.pValue).toBeLessThan(0.05)
    expect(cox.pValue[0]!).toBeLessThan(0.1)

    const w = logRank(time, group, { weight: 'wilcoxon' })
    expect(Number.isFinite(w.statistic)).toBe(true)
    const st = time.map((_, i) => (i % 2 === 0 ? 'a' : 'b'))
    const stratified = logRank(time, group, { strata: st, weight: 'tarone-ware' })
    expect(stratified.stratified).toBe(true)
    expect(Number.isFinite(stratified.pValue)).toBe(true)
  })
})

describe('tier6 — autoArima / higher orders', () => {
  it('fits AR(2) and allows p=3', () => {
    const g = rng(3)
    const y = new Array(120).fill(0)
    const e = Array.from({ length: 120 }, () => g.normal())
    for (let t = 2; t < 120; t++) y[t] = 0.5 * y[t - 1]! - 0.3 * y[t - 2]! + e[t]!
    const manual = arima(y, { p: 3, d: 0, q: 0, includeMean: false })
    expect(manual.ar.length).toBe(3)
    const auto = autoArima(y, { maxP: 3, maxQ: 1, maxD: 0 })
    expect(auto.order.p + auto.order.q).toBeGreaterThanOrEqual(1)
    expect(auto.aic).toBeLessThan(arima(y, { p: 0, d: 0, q: 0, includeMean: false }).aic)
  })
})

describe('tier6 — Taguchi L32/L36 + DSD', () => {
  it('generates valid run counts', () => {
    expect(taguchi('L32').matrix.length).toBe(32)
    expect(taguchi('L36').matrix.length).toBe(36)
    const dsd = definitiveScreening(['A', 'B', 'C', 'D', 'E', 'F'])
    expect([13, 15].includes(dsd.matrix.length)).toBe(true)
    // center row exists
    expect(dsd.matrix.some((r) => r.every((v) => v === 0))).toBe(true)
  })
})

describe('tier6 — mixture + optimizer', () => {
  it('fits Scheffé quadratic and optimizes near planted peak', () => {
    const d = mixtureDesign(['A', 'B', 'C'], { type: 'lattice', degree: 2 })
    // planted: y = 10*A + 4*B + 2*C + 20*A*B  → peak toward A-B edge
    const y = d.matrix.map((r) => 10 * r[0]! + 4 * r[1]! + 2 * r[2]! + 20 * r[0]! * r[1]!)
    const fit = analyzeMixture(d, y, { model: 'quadratic' })
    expect(fit.r2).toBeGreaterThan(0.9)
    const opt = responseOptimizer(
      [
        {
          predict: (x) => fit.predict(x),
          goal: 'maximize',
          lower: 0,
          upper: 30,
        },
      ],
      { simplex: true, nComponents: 3, grid: 11 },
    )
    expect(opt.D).toBeGreaterThan(0.4)
    expect(opt.x[0]! + opt.x[1]!).toBeGreaterThan(opt.x[2]!)
  })
})

describe('tier6 — mixedModel random intercept', () => {
  it('recovers high ICC and group BLUPs', () => {
    const g = rng(11)
    const nGroups = 20
    const nPer = 5
    const sigmaU = 2
    const sigma = 1
    const y: number[] = []
    const X: number[][] = []
    const group: string[] = []
    const trueU: number[] = []
    for (let j = 0; j < nGroups; j++) {
      const u = sigmaU * g.normal()
      trueU.push(u)
      for (let i = 0; i < nPer; i++) {
        const x = g.normal()
        y.push(1 + 0.5 * x + u + sigma * g.normal())
        X.push([x])
        group.push(`g${j}`)
      }
    }
    const fit = mixedModel(y, { fixed: X, group, reml: true })
    expect(fit.icc).toBeGreaterThan(0.55)
    expect(fit.icc).toBeLessThan(0.95)
    // corr(blup, trueU)
    let num = 0
    let d1 = 0
    let d2 = 0
    const mB = fit.ranef.reduce((s, r) => s + r.blup, 0) / nGroups
    const mT = trueU.reduce((a, b) => a + b, 0) / nGroups
    for (let j = 0; j < nGroups; j++) {
      const a = fit.ranef[j]!.blup - mB
      const b = trueU[j]! - mT
      num += a * b
      d1 += a * a
      d2 += b * b
    }
    expect(num / Math.sqrt(d1 * d2)).toBeGreaterThan(0.5)
  })
})
