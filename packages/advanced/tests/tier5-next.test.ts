import { describe, expect, it } from 'vitest'
import { arima, factorAnalysis, logRank, mars, stl } from '@columna/advanced'

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

describe('tier5-next — log-rank', () => {
  it('separates different exponential groups; identical groups not significant', () => {
    const g = rng(11)
    const timeA = Array.from({ length: 40 }, () => g.exponential(20))
    const timeB = Array.from({ length: 40 }, () => g.exponential(50))
    const time = [...timeA, ...timeB]
    const group = [...Array(40).fill('A'), ...Array(40).fill('B')]
    const diff = logRank(time, group)
    expect(diff.pValue).toBeLessThan(0.05)
    expect(diff.df).toBe(1)

    // Same hazard: use one shared stream split — expect non-rejection at α=0.01
    const g2 = rng(101)
    const pool = Array.from({ length: 200 }, () => g2.exponential(40))
    const same = logRank(pool, [...Array(100).fill(0), ...Array(100).fill(1)])
    expect(same.pValue).toBeGreaterThan(0.01)
    expect(diff.pValue).toBeLessThan(same.pValue)
  })
})

describe('tier5-next — MARS GCV prune', () => {
  it('prunes forward terms and keeps high R²', () => {
    const g = rng(3)
    const X = Array.from({ length: 80 }, () => [g.normal(), g.normal(), g.normal()])
    const y = X.map((r) => 2 * Math.max(0, r[0]! - 0.2) - 1.5 * Math.max(0, 0.1 - r[0]!) + 0.2 * g.normal())
    const m = mars(X, y, { maxTerms: 12, prune: true })
    expect(m.nTerms).toBeLessThan(12)
    expect(m.r2).toBeGreaterThan(0.7)
    expect(Number.isFinite(m.gcv)).toBe(true)
  })
})

describe('tier5-next — STL', () => {
  it('recovers seasonal wave from trend + season + noise', () => {
    const g = rng(5)
    const n = 72
    const s = 12
    const trueSeason = Array.from({ length: n }, (_, i) => 3 * Math.sin((2 * Math.PI * i) / s))
    const y = trueSeason.map((seas, i) => 10 + 0.05 * i + seas + 0.3 * g.normal())
    const r = stl(y, { seasonLength: s })
    expect(r.trend.length).toBe(n)
    expect(r.seasonal.length).toBe(n)
    expect(r.residual.length).toBe(n)
    const mS = r.seasonal.reduce((a, b) => a + b, 0) / n
    const mT = trueSeason.reduce((a, b) => a + b, 0) / n
    let num = 0
    let d1 = 0
    let d2 = 0
    for (let i = 0; i < n; i++) {
      const a = r.seasonal[i]! - mS
      const b = trueSeason[i]! - mT
      num += a * b
      d1 += a * a
      d2 += b * b
    }
    const corr = num / Math.sqrt(d1 * d2)
    expect(corr).toBeGreaterThan(0.85)
  })
})

describe('tier5-next — ML factor analysis', () => {
  it('loads correlated variables on the common factor', () => {
    const g = rng(9)
    const n = 80
    const f = Array.from({ length: n }, () => g.normal())
    const x1 = f.map((fi) => fi + 0.3 * g.normal())
    const x2 = f.map((fi) => 0.9 * fi + 0.3 * g.normal())
    const x3 = Array.from({ length: n }, () => g.normal())
    const fa = factorAnalysis({ x1, x2, x3 }, { nFactors: 1, method: 'ml', rotate: 'none' })
    expect(fa.method).toBe('ml')
    expect(fa.logLik).toBeDefined()
    const L = fa.loadings.map((row) => Math.abs(row[0]!))
    expect(L[0]! + L[1]!).toBeGreaterThan(L[2]!)
    for (const u of fa.uniqueVariances) {
      expect(u).toBeGreaterThan(0)
      expect(u).toBeLessThan(1)
    }
  })
})

describe('tier5-next — SARIMA / ARIMAX', () => {
  it('recovers seasonal AR(1) Φ and ARIMAX slope sign', () => {
    const g = rng(42)
    const period = 12
    const Phi = 0.6
    const e = Array.from({ length: 160 }, () => g.normal())
    const y = new Array(160).fill(0)
    for (let t = period; t < 160; t++) y[t] = Phi * y[t - period]! + e[t]!
    const fit = arima(y, { p: 0, d: 0, q: 0, seasonal: { P: 1, D: 0, Q: 0, period }, includeMean: false })
    expect(Math.abs(fit.sar[0]! - Phi)).toBeLessThan(0.25)

    const x = Array.from({ length: 100 }, (_, i) => [i / 10])
    const yx = x.map((r, i) => 2 * r[0]! + g.normal() * 0.5)
    // reshape x as row regressors
    const fitX = arima(yx, { p: 0, d: 0, q: 0, includeMean: true, xreg: x })
    expect(fitX.xregCoef![0]!).toBeGreaterThan(1)
    expect(fitX.xregCoef![0]!).toBeLessThan(3)
  })
})
