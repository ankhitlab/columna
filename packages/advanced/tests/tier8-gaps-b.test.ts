import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import {
  altRegression,
  autoModel,
  clusterVariables,
  crossValidate,
  demonstrationTestPlan,
  dist,
  estimationTestPlan,
  factorAnalysis,
  hclust,
  itemAnalysis,
  lifeRegression,
  multipleCorrespondence,
  patterned,
  powerLawNHPP,
  probitAnalysis,
  promax,
  random,
  reliabilityFit,
} from '@columna/advanced'

const ref = JSON.parse(readFileSync(new URL('./fixtures/tier8-scipy.json', import.meta.url), 'utf8'))
const close = (got: number, want: number, digits = 10) => expect(got).toBeCloseTo(want, digits)
const closeArr = (got: ArrayLike<number>, want: number[], digits = 10) => {
  expect(got.length).toBe(want.length)
  for (let i = 0; i < want.length; i++) close(got[i]!, want[i]!, digits)
}

describe('continuous distributions added for life data', () => {
  it('weibull / lognormal / gamma / beta / logistic / SEV are consistent (cdf∘ppf, pdf integrates, moments)', () => {
    const cases = [dist.weibull(1.8, 50), dist.lognormal(2, 0.4), dist.gamma(2.5, 3), dist.beta(2, 5), dist.logistic(1, 2), dist.smallestExtremeValue(0, 1), dist.exponential(4)]
    for (const d of cases) {
      for (const q of [0.01, 0.2, 0.5, 0.9, 0.999]) close(d.cdf(d.ppf(q)), q, 9)
      close(d.sf(d.ppf(0.3)) + d.cdf(d.ppf(0.3)), 1, 12)
      // numeric mean from pdf
      const lo = d.ppf(1e-9)
      const hi = d.ppf(1 - 1e-9)
      const n = 20000
      const h = (hi - lo) / n
      let m = 0
      let mass = 0
      for (let i = 0; i < n; i++) {
        const x = lo + (i + 0.5) * h
        m += x * d.pdf(x) * h
        mass += d.pdf(x) * h
      }
      close(mass, 1, 5)
      expect(Math.abs(m - d.mean) / Math.max(1, Math.abs(d.mean))).toBeLessThan(1e-4)
    }
    close(dist.weibull(2, 3).mean, 3 * Math.sqrt(Math.PI) / 2, 12)
    expect(() => dist.gamma(-1)).toThrow(RangeError)
  })
})

describe('regression with life data (AFT)', () => {
  it('Weibull with right censoring: coefficients, σ, SEs, log-likelihood and percentiles match the direct likelihood fit', () => {
    const c = ref.lifereg
    const r = lifeRegression(c.t, { x: c.x }, { censor: c.censor })
    closeArr(r.coefficients.map((k) => k.coef), c.coef, 6)
    close(r.scale.estimate, c.sigma, 6)
    closeArr(r.coefficients.map((k) => k.se), c.se, 4)
    close(r.scale.se / r.scale.estimate, c.seLogSigma, 4)
    close(r.logLik, c.logLik, 6)
    close(r.percentile(0.1, [0]).time, c.p10_at0, 5)
    expect(r.shape).toBeCloseTo(1 / c.sigma, 5)
    expect(r.nFailures).toBe(c.censor.filter((v: number) => v === 0).length)
    expect(r.converged).toBe(true)
    expect(r.survival(c.p10_at0, [0])).toBeCloseTo(0.9, 5)
    const pct = r.percentile(0.5, [1])
    expect(pct.ci[0]).toBeLessThan(pct.time)
    expect(pct.ci[1]).toBeGreaterThan(pct.time)
    // without covariates it equals reliabilityFit
    const nox = lifeRegression(c.t, [], { censor: c.censor })
    const rf = reliabilityFit(c.t, { censor: c.censor, distribution: 'weibull' })
    close(Math.exp(nox.coefficients[0]!.coef), rf.scale, 4)
    close(1 / nox.scale.estimate, rf.shape!, 4)
  })
  it('lognormal AFT and validation', () => {
    const c = ref.lifereg_lognormal
    const r = lifeRegression(c.t, { x: ref.lifereg.x }, { distribution: 'lognormal' })
    closeArr(r.coefficients.map((k) => k.coef), c.coef, 6)
    close(r.scale.estimate, c.sigma, 6)
    close(r.logLik, c.logLik, 6)
    expect(r.shape).toBeUndefined()
    const e = lifeRegression(c.t, { x: ref.lifereg.x }, { distribution: 'exponential' })
    expect(e.scale.estimate).toBe(1)
    expect(() => lifeRegression([1, -2, 3], { x: [1, 2, 3] })).toThrow(RangeError)
    expect(() => lifeRegression([1, 2, 3, 4], { x: [1, 2, 3, 4] }, { censor: [3, 0, 0, 0] })).toThrow(RangeError)
  })
  it('Monte-Carlo (seeded): recovers β and σ with 30 % censoring; Wald CI coverage ≈ 95 %', () => {
    const g = random(8201)
    const reps = 150
    let sumB = 0
    let sumS = 0
    let cover = 0
    for (let i = 0; i < reps; i++) {
      const n = 100
      const x = Array.from(g.normal(n))
      const t = x.map((v) => Math.exp(2 + 0.5 * v + 0.4 * Math.log(-Math.log(g.next()))))
      const cens = Array.from(g.uniform(n, 5, 30))
      const obs = t.map((v, j) => Math.min(v, cens[j]!))
      const cc = t.map((v, j) => (v > cens[j]! ? 1 : 0))
      const r = lifeRegression(obs, { x }, { censor: cc })
      const b = r.coefficients[1]!
      sumB += b.coef
      sumS += r.scale.estimate
      if (b.ci[0] <= 0.5 && 0.5 <= b.ci[1]) cover++
    }
    expect(Math.abs(sumB / reps - 0.5)).toBeLessThan(0.02)
    expect(Math.abs(sumS / reps - 0.4)).toBeLessThan(0.02)
    expect(cover / reps).toBeGreaterThan(0.9)
  })
})

describe('accelerated life testing, test plans, repairable systems, probit', () => {
  it('ALT: Arrhenius transform, acceleration factor and use-condition percentiles', () => {
    const g = random(8202)
    const temps = [80, 80, 80, 100, 100, 100, 120, 120, 120].flatMap((t) => new Array(12).fill(t))
    const Ea = 0.5
    const t = temps.map((T) => Math.exp(Math.log(200) + (Ea * 11604.83) / (T + 273.15) - (Ea * 11604.83) / (150 + 273.15) + 0.3 * Math.log(-Math.log(g.next()))))
    const r = altRegression(t, temps, { relation: 'arrhenius', useStress: 50, percentiles: [0.1, 0.5] })
    expect(Math.abs(r.coefficients[1]!.coef - Ea)).toBeLessThan(0.1)
    expect(r.accelerationFactor(120, 50)).toBeGreaterThan(1)
    close(Math.log(r.accelerationFactor(120, 50)), r.coefficients[1]!.coef * (r.transform(50) - r.transform(120)), 10)
    expect(r.usePercentiles!.length).toBe(2)
    expect(r.usePercentiles![1]!.time).toBeGreaterThan(r.usePercentiles![0]!.time)
    expect(r.usePercentiles![0]!.ci[0]).toBeLessThan(r.usePercentiles![0]!.time)
    const ip = altRegression(t, temps, { relation: 'inverse-power' })
    expect(ip.relation).toBe('inverse-power')
    close(ip.transform(10), Math.log(10), 12)
  })
  it('demonstration and estimation test plans', () => {
    const c = ref.demo
    const d = demonstrationTestPlan({ reliability: c.R, time: c.t0, confidence: c.C, shape: c.beta, testTime: c.T })
    expect(d.sampleSize).toBe(c.n)
    // solving for the test time with that n gives back ≈ T (within one integer-n step)
    const tt = demonstrationTestPlan({ reliability: c.R, time: c.t0, confidence: c.C, shape: c.beta, sampleSize: c.n })
    expect(tt.testTime!).toBeLessThanOrEqual(c.T)
    expect(tt.testTime!).toBeGreaterThan(c.T * 0.9)
    // allowing failures needs more units
    expect(demonstrationTestPlan({ reliability: c.R, time: c.t0, confidence: c.C, shape: c.beta, testTime: c.T, allowedFailures: 1 }).sampleSize).toBeGreaterThan(c.n)
    const e1 = estimationTestPlan({ shape: 2, scale: 1000, percentile: 0.1, ratio: 2, censorTime: 800 })
    const e2 = estimationTestPlan({ shape: 2, scale: 1000, percentile: 0.1, ratio: 1.5, censorTime: 800 })
    expect(e2.sampleSize).toBeGreaterThan(e1.sampleSize)
    expect(e1.expectedFailures).toBeLessThan(e1.sampleSize)
    expect(e1.precision).toBeLessThanOrEqual(2)
    expect(() => estimationTestPlan({ shape: 2, scale: 1000, ratio: 0.5, censorTime: 800 })).toThrow(RangeError)
  })
  it('estimation plan (Monte-Carlo, seeded): the planned n gives CI ratios near the target', () => {
    const plan = estimationTestPlan({ shape: 2, scale: 1000, percentile: 0.1, ratio: 2, censorTime: 800 })
    const g = random(8203)
    let sumRatio = 0
    const reps = 60
    for (let i = 0; i < reps; i++) {
      const t = Array.from(g.weibull(plan.sampleSize, 2, 1000))
      const obs = t.map((v) => Math.min(v, 800))
      const c = t.map((v) => (v > 800 ? 1 : 0))
      const f = reliabilityFit(obs, { censor: c, distribution: 'weibull', percentiles: [0.1] })
      // CI ratio for the 10th percentile from the delta method on (ln η, ln β)
      const fit = lifeRegression(obs, [], { censor: c })
      const p = fit.percentile(0.1, [])
      sumRatio += p.ci[1] / p.ci[0]
      void f
    }
    expect(sumRatio / reps).toBeGreaterThan(1.5)
    expect(sumRatio / reps).toBeLessThan(2.6)
  })
  it('power-law NHPP MLE, Laplace / MIL-HDBK-189 trend tests match closed forms', () => {
    const c = ref.nhpp
    const r = powerLawNHPP(c.times, { endTime: c.T })
    close(r.shape, c.beta, 10)
    close(r.scale, c.lambda, 10)
    close(r.trend.laplace.statistic, c.laplace, 10)
    close(r.trend.milHdbk.statistic, c.mil, 10)
    expect(r.trend.milHdbk.df).toBe(2 * c.n)
    expect(r.n).toBe(c.n)
    close(r.cumulative(c.T), c.n, 8) // MLE reproduces the observed count at T
    expect(r.ci.shape[0]).toBeLessThan(r.shape)
    expect(r.ttt.length).toBe(c.n)
    const two = powerLawNHPP([c.times, c.times.map((v: number) => v * 0.9)], { endTime: [c.T, c.T] })
    expect(two.systems).toBe(2)
    expect(two.n).toBe(2 * c.n)
    expect(() => powerLawNHPP([1, 2], { endTime: 1 })).toThrow(RangeError)
  })
  it('Monte-Carlo (seeded): Laplace test keeps its size for a homogeneous process and detects deterioration', () => {
    const g = random(8204)
    let rej = 0
    let pow = 0
    const reps = 600
    for (let i = 0; i < reps; i++) {
      const hpp = Array.from(g.uniform(30, 0, 100)).sort((a, b) => a - b)
      if (powerLawNHPP(hpp, { endTime: 100 }).trend.laplace.pValue < 0.05) rej++
      // power law with β = 2: cumulative N(t) ∝ t² → times = 100·√U
      const nh = Array.from(g.uniform(30), (u) => 100 * Math.sqrt(u)).sort((a, b) => a - b)
      if (powerLawNHPP(nh, { endTime: 100 }).trend.laplace.pValue < 0.05) pow++
    }
    expect(rej / reps).toBeGreaterThan(0.025)
    expect(rej / reps).toBeLessThan(0.08)
    expect(pow / reps).toBeGreaterThan(0.9)
  })
  it('probit analysis: coefficients, ED50 / ED90 with Fieller limits, natural response', () => {
    const c = ref.probit
    const r = probitAnalysis(c.events, c.trials, c.dose)
    closeArr(r.coefficients.map((k) => k.coef), c.coef, 6)
    const ed50 = r.percentiles.find((p) => p.p === 0.5)!
    close(ed50.stress, c.ed50, 6)
    close(r.percentiles.find((p) => p.p === 0.9)!.stress, c.ed90, 6)
    expect(ed50.ci[0]).toBeLessThan(ed50.stress)
    expect(ed50.ci[1]).toBeGreaterThan(ed50.stress)
    close(r.location, c.ed50, 6)
    const lg = probitAnalysis(c.events, c.trials, c.dose, { distribution: 'logistic' })
    expect(lg.distribution).toBe('logistic')
    expect(Math.abs(lg.percentiles.find((p) => p.p === 0.5)!.stress - c.ed50)).toBeLessThan(0.3)
    const nr = probitAnalysis(c.events, c.trials, c.dose, { naturalResponse: 0.02 })
    expect(nr.naturalResponse).toBe(0.02)
    const ls = probitAnalysis(c.events, c.trials, c.dose, { logStress: true })
    expect(ls.percentiles.find((p) => p.p === 0.5)!.stress).toBeGreaterThan(0)
    expect(() => probitAnalysis(c.events, c.trials, c.dose, { naturalResponse: 1 })).toThrow(RangeError)
  })
})

describe('cluster variables, MCA, item analysis, promax', () => {
  it('cluster variables: merge heights match scipy linkage (average, 1 − r)', () => {
    const c = ref.clusterVars
    const data = Object.fromEntries([0, 1, 2, 3, 4].map((j) => [`v${j + 1}`, c.Z.map((r: number[]) => r[j])]))
    const r = clusterVariables(data, { nClusters: 2 })
    closeArr(r.merge.map((m) => m[2]), c.heights, 10)
    closeArr(r.correlation.flat(), c.R.flat(), 10)
    expect(r.membership!.length).toBe(5)
    expect(new Set(r.membership).size).toBe(2)
    expect(r.membership![0]).toBe(r.membership![1]) // v1, v2 highly correlated
    const abs = clusterVariables(data, { distance: 'absolute correlation', nClusters: 2 })
    expect(abs.membership![2]).toBe(abs.membership![3]) // v3, v4 negatively correlated → together under |r|
    expect(r.similarity[0]).toBeGreaterThan(r.similarity[r.similarity.length - 1]!)
    expect(() => clusterVariables({ a: [1, 2, 3] })).toThrow(RangeError)
  })
  it('hclust (NN-chain) merge heights match scipy linkage for single / complete / average / ward', () => {
    const c = ref.hclust
    for (const m of ['single', 'complete', 'average', 'ward'] as const) {
      const r = hclust(c.X, { method: m })
      closeArr(r.merge.map((x) => x[2]), c[m], 9)
    }
    expect(hclust(c.X).order.length).toBe(40)
    expect(() => hclust([[1, 2]])).toThrow(RangeError)
  })
  it('MCA: singular values and total inertia of the indicator matrix match numpy SVD', () => {
    const c = ref.mca
    const r = multipleCorrespondence({ a: c.a, b: c.b, c: c.c })
    closeArr(r.singularValues.slice(0, 4), c.singular, 8)
    close(r.inertia.reduce((s, v) => s + v, 0), c.inertia, 8)
    expect(r.categories.length).toBe(9)
    expect(r.rowCoords.length).toBe(80)
    const burt = multipleCorrespondence({ a: c.a, b: c.b, c: c.c }, { method: 'burt' })
    expect(burt.rowCoords.length).toBe(9)
    // Burt singular values are the squares of the indicator ones
    for (let i = 0; i < 3; i++) close(burt.singularValues[i]!, c.singular[i] ** 2, 8)
    expect(() => multipleCorrespondence({ a: c.a })).toThrow(RangeError)
  })
  it('item analysis: Cronbach α, standardized α, item-total correlations, SMC, α if deleted', () => {
    const c = ref.item
    const data = Object.fromEntries([0, 1, 2, 3].map((j) => [`i${j + 1}`, c.items.map((r: number[]) => r[j])]))
    const r = itemAnalysis(data)
    close(r.alpha, c.alpha, 10)
    close(r.standardizedAlpha, c.stdAlpha, 10)
    closeArr(r.items.map((i) => i.alphaIfDeleted), c.alphaDel, 10)
    closeArr(r.items.map((i) => i.itemTotalCorr), c.itemTotal, 10)
    closeArr(r.items.map((i) => i.adjustedItemTotalCorr), c.adjItemTotal, 10)
    closeArr(r.items.map((i) => i.squaredMultipleCorr), c.smc, 8)
    expect(r.items[0]!.adjustedItemTotalCorr).toBeGreaterThan(r.items[3]!.adjustedItemTotalCorr) // least noisy item correlates most
  })
  it('promax matches the numpy implementation and factorAnalysis loadings can be rotated', () => {
    const c = ref.promax
    const r = promax(c.A)
    closeArr(r.loadings.flat(), c.P.flat(), 8)
    closeArr(r.factorCorrelation.flat(), c.Phi.flat(), 8)
    close(r.factorCorrelation[0]![0]!, 1, 10)
    const g = random(8205)
    const f1 = Array.from(g.normal(120))
    const f2 = Array.from(g.normal(120))
    const data = {
      a: f1.map((v) => v + 0.4 * g.normal(1)[0]!),
      b: f1.map((v) => 0.9 * v + 0.5 * g.normal(1)[0]!),
      c: f1.map((v, i) => 0.8 * v + 0.3 * f2[i]! + 0.5 * g.normal(1)[0]!),
      d: f2.map((v) => v + 0.4 * g.normal(1)[0]!),
      e: f2.map((v) => 0.9 * v + 0.5 * g.normal(1)[0]!),
    }
    const fa = factorAnalysis(data, { nFactors: 2 })
    const pr = promax(fa.rotatedLoadings ?? fa.loadings)
    expect(Math.abs(pr.factorCorrelation[0]![1]!)).toBeLessThan(0.6)
    // the pattern keeps the simple structure: a, b load on one factor, d, e on the other
    const dom = pr.loadings.map((row) => (Math.abs(row[0]!) > Math.abs(row[1]!) ? 0 : 1))
    expect(dom[0]).toBe(dom[1])
    expect(dom[3]).toBe(dom[4])
    expect(dom[0]).not.toBe(dom[3])
  })
})

describe('model validation and AutoML', () => {
  it('cross-validation of OLS on linear data gives high out-of-fold R²; classification metrics are consistent', () => {
    const g = random(8206)
    const n = 120
    const X = Array.from({ length: n }, () => [g.normal(1)[0]!, g.normal(1)[0]!])
    const y = X.map((r) => 1 + 2 * r[0]! - r[1]! + 0.3 * g.normal(1)[0]!)
    const cv = crossValidate(X, y, { model: 'ols', folds: 5 })
    expect(cv.metrics.r2!).toBeGreaterThan(0.95)
    expect(cv.perFold.length).toBe(5)
    expect(cv.predictions.length).toBe(n)
    const yc = X.map((r) => (r[0]! + r[1]! > 0 ? 'yes' : 'no'))
    const cc = crossValidate(X, yc, { model: 'logistic', folds: 4 })
    expect(cc.task).toBe('classification')
    expect(cc.metrics.accuracy!).toBeGreaterThan(0.9)
    expect(cc.metrics.logLoss).toBeDefined()
    expect(Object.keys(cc.metrics.confusion!).sort()).toEqual(['no', 'yes'])
    expect(() => crossValidate(X.slice(0, 5), y.slice(0, 5), { model: 'ols', folds: 5 })).toThrow(RangeError)
    expect(() => crossValidate(X, yc, { model: 'mars' })).toThrow(RangeError)
  })
  it('autoModel ranks OLS first on linear data and a tree ensemble first on a step function', () => {
    const g = random(8207)
    const n = 150
    const X = Array.from({ length: n }, () => [g.normal(1)[0]!, g.normal(1)[0]!])
    const lin = X.map((r) => 1 + 2 * r[0]! - r[1]! + 0.3 * g.normal(1)[0]!)
    const a = autoModel(X, lin, { folds: 4, nTrees: 30 })
    expect(a.best).toBe('ols')
    expect(a.ranking.length).toBe(5)
    expect(a.fit.predict([[0, 0]])[0]).toBeCloseTo(1, 0)
    const step = X.map((r) => (r[0]! > 0 ? 5 : 0) + (r[1]! > 0.5 ? 3 : 0) + 0.2 * g.normal(1)[0]!)
    const b = autoModel(X, step, { folds: 4, nTrees: 30, models: ['ols', 'cart', 'random-forest'] })
    expect(b.best).not.toBe('ols')
    const cls = autoModel(X, X.map((r) => (r[0]! > 0 ? 'a' : 'b')), { folds: 3, nTrees: 20 })
    expect(cls.task).toBe('classification')
    expect(cls.ranking[0]!.metrics.accuracy!).toBeGreaterThan(0.85)
  })
})

describe('random data and patterned data', () => {
  it('seeded generators reproduce and have the right moments; patterned sequences', () => {
    const a = random(1)
    const b = random(1)
    expect(Array.from(a.normal(5))).toEqual(Array.from(b.normal(5)))
    const g = random(8208)
    const mean = (v: ArrayLike<number>) => Array.from(v).reduce((s, x) => s + x, 0) / v.length
    expect(Math.abs(mean(g.normal(20000, 3, 2)) - 3)).toBeLessThan(0.05)
    expect(Math.abs(mean(g.gamma(20000, 2.5, 2)) - 5)).toBeLessThan(0.1)
    expect(Math.abs(mean(g.beta(20000, 2, 3)) - 0.4)).toBeLessThan(0.01)
    expect(Math.abs(mean(g.poisson(20000, 4)) - 4)).toBeLessThan(0.05)
    expect(Math.abs(mean(g.poisson(5000, 50)) - 50)).toBeLessThan(0.4)
    expect(Math.abs(mean(g.binomial(20000, 10, 0.3)) - 3)).toBeLessThan(0.05)
    expect(Math.abs(mean(g.binomial(5000, 500, 0.3)) - 150)).toBeLessThan(0.6)
    expect(Math.abs(mean(g.weibull(20000, 2, 3)) - dist.weibull(2, 3).mean)).toBeLessThan(0.03)
    expect(Math.abs(mean(g.t(5000, 10)))).toBeLessThan(0.05)
    expect(Math.abs(mean(g.chi2(5000, 4)) - 4)).toBeLessThan(0.15)
    expect(Math.abs(mean(g.f(5000, 5, 20)) - 20 / 18)).toBeLessThan(0.05)
    expect(Math.abs(mean(g.exponential(20000, 2)) - 2)).toBeLessThan(0.05)
    expect(Math.abs(mean(g.lognormal(20000, 0, 0.5)) - Math.exp(0.125))).toBeLessThan(0.03)
    const ints = g.integer(1000, 1, 6)
    expect(Math.min(...ints)).toBe(1)
    expect(Math.max(...ints)).toBe(6)
    const s = g.sample(['a', 'b', 'c', 'd'], 3)
    expect(new Set(s).size).toBe(3)
    expect(g.sample([1, 2], 5, { replace: true }).length).toBe(5)
    expect(() => g.sample([1, 2], 3)).toThrow(RangeError)
    expect(g.shuffle([1, 2, 3, 4]).sort()).toEqual([1, 2, 3, 4])
    expect(patterned(1, 3)).toEqual([1, 2, 3])
    expect(patterned(1, 2, { repeat: 2, times: 2 })).toEqual([1, 1, 2, 2, 1, 1, 2, 2])
    expect(patterned(0, 1, { step: 0.25 })).toEqual([0, 0.25, 0.5, 0.75, 1])
    expect(patterned(3, 1)).toEqual([3, 2, 1])
  })
})

describe('DataFrame methods (gap-closing B)', () => {
  it('lifeRegression / altRegression / powerLawNHPP / probitAnalysis / clusterVariables / multipleCorrespondence / itemAnalysis / crossValidate / autoModel', async () => {
    const c = ref.lifereg
    const df = DataFrame.fromColumns({ t: c.t, x: c.x, cens: c.censor })
    closeArr(df.lifeRegression('t', ['x'], { censor: 'cens' }).coefficients.map((k) => k.coef), c.coef, 6)
    const alt = df.altRegression('t', 'x', { relation: 'linear', censor: 'cens' })
    close(alt.coefficients[1]!.coef, c.coef[1], 6)
    const nh = DataFrame.fromColumns({ t: ref.nhpp.times, sys: ref.nhpp.times.map(() => 'A') })
    close(nh.powerLawNHPP('t', { endTime: ref.nhpp.T }).shape, ref.nhpp.beta, 10)
    close(nh.powerLawNHPP('t', { endTime: [ref.nhpp.T], system: 'sys' }).shape, ref.nhpp.beta, 10)
    const pd = DataFrame.fromColumns({ e: ref.probit.events, n: ref.probit.trials, d: ref.probit.dose })
    close(pd.probitAnalysis('e', 'n', 'd').location, ref.probit.ed50, 6)
    const cv = ref.clusterVars
    const cd = DataFrame.fromColumns(Object.fromEntries([0, 1, 2, 3, 4].map((j) => [`v${j + 1}`, cv.Z.map((r: number[]) => r[j])])))
    closeArr(cd.clusterVariables(['v1', 'v2', 'v3', 'v4', 'v5']).merge.map((m) => m[2]), cv.heights, 10)
    const m = ref.mca
    const md = DataFrame.fromColumns({ a: m.a, b: m.b, c: m.c })
    closeArr(md.multipleCorrespondence(['a', 'b', 'c']).singularValues.slice(0, 4), m.singular, 8)
    const it2 = ref.item
    const id = DataFrame.fromColumns(Object.fromEntries([0, 1, 2, 3].map((j) => [`i${j + 1}`, it2.items.map((r: number[]) => r[j])])))
    close(id.itemAnalysis(['i1', 'i2', 'i3', 'i4']).alpha, it2.alpha, 10)
    const g = random(8209)
    const xs = Array.from(g.normal(80))
    const ys = xs.map((v) => 2 * v + 0.1 * g.normal(1)[0]!)
    const rd = DataFrame.fromColumns({ x: xs, y: ys })
    expect(rd.crossValidate('y', ['x'], { model: 'ols' }).metrics.r2!).toBeGreaterThan(0.95)
    expect(rd.autoModel('y', ['x'], { folds: 3, nTrees: 20, models: ['ols', 'cart'] }).best).toBe('ols')
    close((await rd.lazy().crossValidate('y', ['x'], { model: 'ols' })).metrics.r2!, rd.crossValidate('y', ['x'], { model: 'ols' }).metrics.r2!, 12)
  })
})
