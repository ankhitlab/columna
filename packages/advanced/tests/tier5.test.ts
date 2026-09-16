import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  acf,
  analyzeEffects,
  arima,
  boxBehnken,
  cart,
  ccd,
  correspondence,
  decompose,
  discriminant,
  ets,
  factorAnalysis,
  fractionalFactorial,
  fullFactorial,
  hclust,
  kaplanMeier,
  kmeans,
  ljungBox,
  mars,
  pacf,
  pca,
  plackettBurman,
  randomForest,
  reliabilityFit,
  taguchi,
  treeNet,
  trendAnalysis,
  warrantyPrediction,
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
  return { u, normal }
}

describe('5.1 time series', () => {
  it('linear trend matches OLS coefficients', () => {
    const r = trendAnalysis(ref.trend.y, { model: 'linear' })
    close(r.parameters.intercept!, ref.trend.coef[0], 6)
    close(r.parameters.slope!, ref.trend.coef[1], 6)
    expect(r.r2).toBeGreaterThan(0.95)
  })

  it('ACF lag-1 matches sample formula; PACF / Ljung–Box finite', () => {
    const a = acf(ref.ar1.y, { maxLag: 10 })
    close(a.acf[1]!, ref.ar1.acf1, 10)
    const p = pacf(ref.ar1.y, { maxLag: 5 })
    expect(Math.abs(p.pacf[1]!)).toBeGreaterThan(0.4)
    const lb = ljungBox(ref.ar1.y, { lags: 10 })
    expect(lb.pValue).toBeLessThan(0.05) // autocorrelated
  })

  it('SES with fixed α matches recursion SSE', () => {
    const r = ets(ref.ses.y, { method: 'ses', alpha: ref.ses.alpha })
    close(r.sse, ref.ses.sse, 6)
  })

  it('ARIMA recovers AR(1) φ ≈ 0.7 (seeded series)', () => {
    const fit = arima(ref.ar1.y, { p: 1, d: 0, q: 0 })
    expect(Math.abs(fit.ar[0]! - 0.7)).toBeLessThan(0.15)
  })

  it('decomposition seasonal indices sum ≈ 0 (additive)', () => {
    const g = rng(3)
    const y = Array.from({ length: 48 }, (_, i) => 10 + 0.1 * i + 2 * Math.sin((2 * Math.PI * i) / 12) + 0.2 * g.normal())
    const d = decompose(y, { seasonLength: 12, method: 'additive' })
    const s = d.seasonalIndices.reduce((a, b) => a + b, 0)
    expect(Math.abs(s)).toBeLessThan(1e-8)
  })
})

describe('5.2 DOE', () => {
  it('full factorial 2³ recovers planted effects via Lenth analysis', () => {
    const d = fullFactorial(['A', 'B', 'C'])
    expect(d.matrix.length).toBe(8)
    const eff = analyzeEffects({ type: 'custom', factors: ['A', 'B', 'C'], matrix: ref.doe.matrix }, ref.doe.y)
    const a = eff.effects.find((e) => e.term === 'A')!
    const b = eff.effects.find((e) => e.term === 'B')!
    expect(Math.abs(a.effect - 4)).toBeLessThan(0.5)
    expect(Math.abs(b.effect + 2)).toBeLessThan(0.5)
    expect(eff.significant).toContain('A')
  })

  it('fractional / PB / CCD / BBD / Taguchi generate valid run counts', () => {
    expect(fractionalFactorial(['A', 'B', 'C', 'D'], ['ABD']).matrix.length).toBe(8)
    expect(plackettBurman(['A', 'B', 'C', 'D', 'E', 'F', 'G']).matrix.length).toBe(8)
    expect(ccd(['A', 'B']).matrix.length).toBeGreaterThan(8)
    expect(boxBehnken(['A', 'B', 'C']).matrix.length).toBe(15)
    expect(taguchi('L9').matrix.length).toBe(9)
  })
})

describe('5.3 reliability', () => {
  it('Weibull MLE recovers shape/scale under ~30% censoring', () => {
    const fit = reliabilityFit(ref.weibull.time, { distribution: 'weibull', censor: ref.weibull.censor })
    expect(Math.abs(fit.shape! - ref.weibull.shape)).toBeLessThan(0.45)
    expect(Math.abs(fit.scale - ref.weibull.scale) / ref.weibull.scale).toBeLessThan(0.25)
    const w = warrantyPrediction(fit, { warranty: 50, nUnits: 100 })
    expect(w.expectedFailures).toBeGreaterThan(0)
    expect(w.reliability).toBeGreaterThan(0.3)
  })

  it('Kaplan–Meier is monotone and ends ≤ 1', () => {
    const km = kaplanMeier(ref.km.time, { censor: ref.km.censor })
    expect(km.curve.length).toBeGreaterThan(0)
    for (let i = 1; i < km.curve.length; i++) expect(km.curve[i]!.survival).toBeLessThanOrEqual(km.curve[i - 1]!.survival + 1e-12)
    expect(km.curve[0]!.survival).toBeLessThanOrEqual(1)
  })
})

describe('5.4 multivariate', () => {
  it('PCA: first component loads on correlated X1/X2', () => {
    const r = pca({ x1: ref.pca.x1, x2: ref.pca.x2, x3: ref.pca.x3 })
    expect(r.cumulative[0]!).toBeGreaterThan(0.4)
    const l0 = r.loadings.map((row) => Math.abs(row[0]!))
    expect(l0[0]! + l0[1]!).toBeGreaterThan(l0[2]!)
  })

  it('k-means separates two clusters; LDA accuracy high', () => {
    const km = kmeans(ref.kmeans.X, { k: 2, seed: 1 })
    expect(km.inertia).toBeLessThan(80)
    // majority label agreement
    const flip = km.cluster.map((c) => 1 - c)
    const acc = (lab: number[]) => lab.filter((c, i) => c === ref.kmeans.labels[i]).length / lab.length
    expect(Math.max(acc(km.cluster), acc(flip))).toBeGreaterThan(0.9)

    const lda = discriminant(ref.lda.X, ref.lda.y, { method: 'lda' })
    expect(lda.accuracy).toBeGreaterThan(0.9)
  })

  it('factor varimax / hclust / correspondence run', () => {
    const fa = factorAnalysis({ x1: ref.pca.x1, x2: ref.pca.x2, x3: ref.pca.x3 }, { nFactors: 2, rotate: 'varimax' })
    expect(fa.rotatedLoadings!.length).toBe(3)
    const hc = hclust(ref.kmeans.X.slice(0, 20), { method: 'average' })
    expect(hc.merge.length).toBe(19)
    const ca = correspondence([
      [10, 5, 2],
      [3, 12, 4],
      [1, 2, 15],
    ])
    expect(ca.singularValues.length).toBeGreaterThan(0)
  })
})

describe('5.5 predictive', () => {
  it('CART / RF recover signal; OOS R² positive', () => {
    const g = rng(9)
    const X = ref.rf.X as number[][]
    const y = ref.rf.y as number[]
    const tree = cart(X, y, { task: 'regression', maxDepth: 4 })
    const pred = tree.predict(X) as number[]
    let ss = 0
    let sst = 0
    const m = y.reduce((a, b) => a + b, 0) / y.length
    for (let i = 0; i < y.length; i++) {
      ss += (pred[i]! - y[i]!) ** 2
      sst += (y[i]! - m) ** 2
    }
    expect(1 - ss / sst).toBeGreaterThan(0.7)

    const rf = randomForest(X, y, { task: 'regression', nTrees: 30, maxDepth: 4, seed: 2 })
    const holdX = Array.from({ length: 40 }, () => [g.normal(), g.normal(), g.normal()])
    const holdY = holdX.map((r) => 2 * r[0]! - 1.5 * r[1]! + 0.3 * g.normal())
    const rp = rf.predict(holdX) as number[]
    let ss2 = 0
    let sst2 = 0
    const m2 = holdY.reduce((a, b) => a + b, 0) / holdY.length
    for (let i = 0; i < holdY.length; i++) {
      ss2 += (rp[i]! - holdY[i]!) ** 2
      sst2 += (holdY[i]! - m2) ** 2
    }
    expect(1 - ss2 / sst2).toBeGreaterThan(0.4)
  })

  it('TreeNet and MARS fit synthetic regression', () => {
    const X = ref.rf.X as number[][]
    const y = ref.rf.y as number[]
    const tn = treeNet(X, y, { nTrees: 40, learningRate: 0.1, maxDepth: 2 })
    expect(tn.trainRmse).toBeLessThan(1)
    const m = mars(X, y, { maxTerms: 6 })
    expect(m.r2).toBeGreaterThan(0.5)
  })
})
