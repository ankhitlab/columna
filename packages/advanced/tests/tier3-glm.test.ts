import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import { anova, glm, linearModel, logit, mlogit, ols, ologit, parseFormula, poissonRegression } from '@columna/advanced'

const ref = JSON.parse(readFileSync(new URL('./fixtures/tier3-scipy.json', import.meta.url), 'utf8'))
const close = (got: number, want: number, digits = 10) => expect(got).toBeCloseTo(want, digits)
const closeArr = (got: ArrayLike<number>, want: number[], digits = 10) => {
  expect(got.length).toBe(want.length)
  for (let i = 0; i < want.length; i++) close(got[i]!, want[i]!, digits)
}

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
  const pois = (lambda: number) => {
    const L = Math.exp(-lambda)
    let k = 0
    let p = 1
    do {
      k++
      p *= u()
    } while (p > L)
    return k - 1
  }
  return { u, normal, pois }
}
const sample = (n: number, g: () => number) => Array.from({ length: n }, g)

describe('3.4 binary logistic regression', () => {
  const c = ref.logit
  const r = logit(c.y, { a: c.a, b: c.b })
  it('coefficients, SE, odds ratios, deviance, G, Pearson, Hosmer–Lemeshow match the direct likelihood fit', () => {
    closeArr(r.coefficients.map((k) => k.coef), c.coef, 7)
    closeArr(r.coefficients.map((k) => k.se), c.se, 7)
    closeArr(r.coefficients.slice(1).map((k) => k.ratio!), c.or, 6)
    expect(r.coefficients[0]!.ratio).toBeUndefined()
    close(r.deviance, c.deviance, 7)
    close(r.nullDeviance, c.nullDeviance, 7)
    close(r.gTest.statistic, c.G, 7)
    close(r.gTest.pValue, c.pG, 8)
    expect(r.gTest.df).toBe(2)
    close(r.pearsonChi2, c.pearson, 6)
    close(r.goodnessOfFit.hosmerLemeshow!.statistic, c.hl, 6)
    close(r.goodnessOfFit.hosmerLemeshow!.pValue, c.pHL, 7)
    expect(r.goodnessOfFit.hosmerLemeshow!.df).toBe(8)
    close(r.logLik, c.logLik, 7)
    expect(r.converged).toBe(true)
    expect(r.dispersion).toBe(1)
    close(r.predict(c.predict.x)[0]!.fit, c.predict.p, 7)
    const lp = r.predict(c.predict.x, { type: 'link' })[0]!
    close(1 / (1 + Math.exp(-lp.fit)), c.predict.p, 7)
    expect(lp.ci[0]).toBeLessThan(lp.fit)
  })
  it('events / trials form, probit link, offsets and validation', () => {
    const t = ref.logit_trials
    const e = logit(t.events, { x: t.x }, { trials: t.trials })
    closeArr(e.coefficients.map((k) => k.coef), t.coef, 7)
    closeArr(e.coefficients.map((k) => k.se), t.se, 7)
    close(e.deviance, t.deviance, 6)
    close(e.pearsonChi2, t.pearson, 6)
    // non-integer proportions with trials give the same fit
    const pr = logit(t.events.map((v: number, i: number) => v / t.trials[i]), { x: t.x }, { trials: t.trials })
    closeArr(pr.coefficients.map((k) => k.coef), t.coef, 7)
    expect(() => logit([3, 2.5, 1], { x: [1, 2, 3] }, { trials: [5, 5, 5] })).toThrow(RangeError)
    const pb = glm(c.y, { a: c.a, b: c.b }, { family: 'binomial', link: 'probit' })
    closeArr(pb.coefficients.map((k) => k.coef), ref.probit.coef, 5)
    close(pb.logLik, ref.probit.logLik, 6)
    expect(pb.coefficients[1]!.ratio).toBeUndefined()
    // an offset of a constant shifts the intercept by exactly that constant
    const off = logit(c.y, { a: c.a, b: c.b }, { offset: new Array(c.y.length).fill(0.7) })
    close(off.coefficients[0]!.coef, c.coef[0] - 0.7, 6)
    close(off.coefficients[1]!.coef, c.coef[1], 6)
    expect(() => logit([0, 1, 2], { x: [1, 2, 3] })).toThrow(RangeError)
    expect(() => glm(c.y, { a: c.a }, { family: 'binomial', link: 'log' })).toThrow(RangeError)
    expect(() => glm(c.y, { a: c.a }, { family: 'weibull' as never })).toThrow(RangeError)
  })
  it('Monte-Carlo (seeded): near-unbiased β (n = 200), Wald CI coverage ≈ 95 %, G test size ≈ α', () => {
    const g = rng(3401)
    const reps = 600
    const n = 200
    let sum = 0
    let cover = 0
    let rejG = 0
    for (let i = 0; i < reps; i++) {
      const x = sample(n, g.normal)
      const z = sample(n, g.normal)
      const y = x.map((v) => (g.u() < 1 / (1 + Math.exp(-(0.3 + 1 * v))) ? 1 : 0))
      const r = logit(y, { x, z })
      const b = r.coefficients[1]!
      sum += b.coef
      if (b.ci[0] <= 1 && 1 <= b.ci[1]) cover++
      // z is noise: its Wald test should reject at ≈ α
      if (r.coefficients[2]!.pValue < 0.05) rejG++
    }
    expect(Math.abs(sum / reps - 1)).toBeLessThan(0.05)
    expect(cover / reps).toBeGreaterThan(0.92)
    expect(cover / reps).toBeLessThan(0.98)
    expect(rejG / reps).toBeGreaterThan(0.025)
    expect(rejG / reps).toBeLessThan(0.08)
  })
})

describe('3.5 Poisson regression', () => {
  const c = ref.poisson
  it('coefficients with a log-exposure offset, deviance, null deviance, Pearson, log-likelihood', () => {
    const r = poissonRegression(c.y, { c1: c.c1, c2: c.c2 }, { offset: c.exposure.map(Math.log) })
    closeArr(r.coefficients.map((k) => k.coef), c.coef, 7)
    closeArr(r.coefficients.map((k) => k.se), c.se, 6)
    close(r.deviance, c.deviance, 7)
    close(r.nullDeviance, c.nullDeviance, 6)
    close(r.pearsonChi2, c.pearson, 6)
    close(r.logLik, c.logLik, 7)
    expect(r.coefficients[1]!.ratio).toBeCloseTo(Math.exp(c.coef[1]), 8)
    expect(r.goodnessOfFit.hosmerLemeshow).toBeUndefined()
    expect(() => poissonRegression([1, -1, 2], { x: [1, 2, 3] })).toThrow(RangeError)
  })
  it('Monte-Carlo (seeded): recovers the rate ratio and keeps the deviance goodness-of-fit near α', () => {
    const g = rng(3501)
    const reps = 500
    const n = 120
    let sum = 0
    let rejGof = 0
    let cover = 0
    for (let i = 0; i < reps; i++) {
      const x = sample(n, g.normal)
      const y = x.map((v) => g.pois(Math.exp(2.5 + 0.5 * v))) // means ≈ 12: the χ² approximation to the deviance needs non-tiny counts
      const r = poissonRegression(y, { x })
      const b = r.coefficients[1]!
      sum += b.coef
      if (b.ci[0] <= 0.5 && 0.5 <= b.ci[1]) cover++
      if (r.goodnessOfFit.deviance.pValue < 0.05) rejGof++
    }
    expect(Math.abs(sum / reps - 0.5)).toBeLessThan(0.02)
    expect(cover / reps).toBeGreaterThan(0.92)
    expect(rejGof / reps).toBeLessThan(0.1)
  })
})

describe('3.4 ordinal and nominal logistic regression', () => {
  it('ordinal (proportional odds, Minitab sign convention) matches the direct likelihood maximum', () => {
    const c = ref.ologit
    const r = ologit(c.y, { x1: c.x1, x2: c.x2 })
    expect(r.levels).toEqual(['0', '1', '2'])
    closeArr(r.thresholds.map((k) => k.coef), c.theta, 5)
    closeArr(r.coefficients.map((k) => k.coef), c.coef, 5)
    closeArr([...r.thresholds, ...r.coefficients].map((k) => k.se), c.se, 4)
    close(r.logLik, c.logLik, 6)
    close(r.gTest.statistic, c.G, 5)
    expect(r.gTest.df).toBe(2)
    expect(r.converged).toBe(true)
    expect(r.thresholds[0]!.coef).toBeLessThan(r.thresholds[1]!.coef)
    expect(r.coefficients[0]!.oddsRatio).toBeCloseTo(Math.exp(c.coef[0]), 4)
    // probabilities sum to 1 and predict() agrees with the fitted rows
    for (const pr of r.probabilities.slice(0, 5)) close(pr[0]! + pr[1]! + pr[2]!, 1, 12)
    const p0 = r.predict([c.x1[0], c.x2[0]])[0]!
    closeArr(p0, Array.from(r.probabilities[0]!), 10)
    expect(() => ologit([1, 1, 2, 2], { x: [1, 2, 3, 4] })).toThrow(RangeError)
    // string levels are sorted lexicographically
    const s = ologit(['low', 'mid', 'high', 'low', 'mid', 'high', 'low', 'high', 'mid', 'low'], { x: [1, 2, 3, 1.5, 2.5, 3.5, 0.5, 4, 2, 1] })
    expect(s.levels).toEqual(['high', 'low', 'mid'])
  })
  it('nominal (multinomial) matches the direct likelihood maximum with analytic Hessian SEs', () => {
    const c = ref.mlogit
    const r = mlogit(c.y, { x: c.x })
    expect(r.reference).toBe('0')
    expect(r.equations.map((e) => e.level)).toEqual(['1', '2'])
    closeArr(r.equations[0]!.coefficients.map((k) => k.coef), c.coef[0], 6)
    closeArr(r.equations[1]!.coefficients.map((k) => k.coef), c.coef[1], 6)
    closeArr(r.equations[0]!.coefficients.map((k) => k.se), c.se[0], 5)
    closeArr(r.equations[1]!.coefficients.map((k) => k.se), c.se[1], 5)
    close(r.logLik, c.logLik, 7)
    close(r.gTest.statistic, c.G, 6)
    expect(r.gTest.df).toBe(2)
    const alt = mlogit(c.y, { x: c.x }, { reference: 2 })
    expect(alt.reference).toBe('2')
    close(alt.logLik, c.logLik, 7)
    // relabeling the reference: log-odds of 1 vs 2 = (1 vs 0) − (2 vs 0)
    close(alt.equations.find((e) => e.level === '1')!.coefficients[1]!.coef, c.coef[0][1] - c.coef[1][1], 6)
    const p = r.predict([0.2])[0]!
    close(p.reduce((s, v) => s + v, 0), 1, 12)
  })
})

describe('3.6 general linear model (Type III via effects coding)', () => {
  const c = ref.glm
  it('formula parsing', () => {
    const f = parseFormula('y ~ a*b + x^2 + c:d')
    expect(f.terms.map((t) => t.name)).toEqual(['a', 'b', 'a*b', 'x*x', 'c*d'])
    expect(f.intercept).toBe(true)
    expect(parseFormula('y ~ a + b - 1').intercept).toBe(false)
    expect(parseFormula('y ~ a*b*c').terms.length).toBe(7)
    expect(() => parseFormula('y a + b')).toThrow(RangeError)
  })
  it('adjusted and sequential SS, F, p and coefficients match the reduced-model computation', () => {
    const r = linearModel({ y: c.y, a: c.a, b: c.b, cv: c.cv }, 'y ~ a*b + cv')
    const byName = Object.fromEntries(r.anova.terms.map((t) => [t.term, t]))
    for (const k of ['a', 'b', 'a*b', 'cv']) {
      close(byName[k]!.adjSS, c.adjSS[k], 8)
      close(byName[k]!.seqSS, c.seqSS[k], 8)
      close(byName[k]!.f, c.F[k], 8)
      close(byName[k]!.pValue, c.p[k], 9)
    }
    expect(byName.a!.df).toBe(2)
    expect(byName['a*b']!.df).toBe(2)
    expect(byName.cv!.df).toBe(1)
    close(r.anova.error.ss, c.sse, 8)
    closeArr(r.coefficients.map((k) => k.coef), c.coef, 8)
    closeArr(r.coefficients.map((k) => k.se), c.se, 8)
    expect(r.columnNames).toEqual(['Constant', 'a[H]', 'a[L]', 'b[a]', 'a[H]*b[a]', 'a[L]*b[a]', 'cv'])
    expect(r.factors).toEqual({ a: ['H', 'L', 'M'], b: ['a', 'b'] })
    expect(r.means.a!.map((m) => m.level)).toEqual(['H', 'L', 'M'])
    close(r.anova.terms.reduce((s, t) => s + t.seqSS, 0), r.anova.model.ss, 8)
  })
  it('one factor reduces to One-Way ANOVA; balanced two-way has adjusted = sequential SS; numeric factors via `factors`', () => {
    const g = rng(3601)
    const grp = Array.from({ length: 24 }, (_, i) => ['A', 'B', 'C'][i % 3]!)
    const y = grp.map((v) => (v === 'A' ? 0 : v === 'B' ? 1 : 2.5) + g.normal())
    const lm = linearModel({ y, grp }, 'y ~ grp')
    const ow = anova({ A: y.filter((_, i) => grp[i] === 'A'), B: y.filter((_, i) => grp[i] === 'B'), C: y.filter((_, i) => grp[i] === 'C') })
    close(lm.anova.terms[0]!.f, ow.statistic, 10)
    close(lm.anova.terms[0]!.pValue, ow.pValue, 10)
    close(lm.anova.terms[0]!.adjSS, ow.ssBetween, 10)
    close(lm.means.grp![0]!.fittedMean, ow.groups[0]!.mean, 10)
    // balanced 2 × 2 with 3 replicates: Type III equals Type I
    const fa = Array.from({ length: 12 }, (_, i) => (i < 6 ? 1 : 2))
    const fb = Array.from({ length: 12 }, (_, i) => (i % 2 ? 1 : 2))
    const yb = fa.map((v, i) => v + 0.5 * fb[i]! + g.normal())
    const two = linearModel({ y: yb, fa, fb }, 'y ~ fa*fb', { factors: ['fa', 'fb'] })
    for (const t of two.anova.terms) close(t.adjSS, t.seqSS, 10)
    expect(two.factors).toEqual({ fa: ['1', '2'], fb: ['1', '2'] })
    // covariate-only model equals ols
    const x = sample(20, g.normal)
    const yc = x.map((v) => 1 + 2 * v + g.normal())
    const cov = linearModel({ yc, x }, 'yc ~ x')
    const o = ols(yc, { x })
    close(cov.coefficients[1]!.coef, o.coefficients[1]!.coef, 12)
    close(cov.r2, o.r2, 12)
    expect(() => linearModel({ y, grp }, 'y ~ nope')).toThrow(RangeError)
  })
  it('Monte-Carlo (seeded): Type III F for a null factor in an unbalanced design keeps its size', () => {
    const g = rng(3602)
    const reps = 1500
    let rej = 0
    for (let i = 0; i < reps; i++) {
      const n = 30
      const a = Array.from({ length: n }, () => (g.u() < 0.4 ? 'p' : g.u() < 0.5 ? 'q' : 'r'))
      const b = Array.from({ length: n }, () => (g.u() < 0.6 ? 'u' : 'v'))
      const y = a.map((v, j) => (v === 'p' ? 1 : 0) + (b[j] === 'u' ? 0.8 : 0) + g.normal())
      const lm = linearModel({ y, a, b }, 'y ~ a + b')
      // b has a real effect; a's test is not what we check — instead add a pure-noise factor
      const c = Array.from({ length: n }, () => (g.u() < 0.5 ? 'm' : 'n'))
      const lm2 = linearModel({ y, a, b, c }, 'y ~ a + b + c')
      void lm
      if (lm2.anova.terms[2]!.pValue < 0.05) rej++
    }
    expect(rej / reps).toBeGreaterThan(0.035)
    expect(rej / reps).toBeLessThan(0.07)
  })
})

describe('DataFrame methods (Tier 3 GLM)', () => {
  it('logistic / glm / ologit / mlogit / linearModel on columns', async () => {
    const c = ref.logit
    const df = DataFrame.fromColumns({ y: c.y, a: c.a, b: c.b, off: new Array(c.y.length).fill(0) })
    closeArr(df.logistic('y', ['a', 'b']).coefficients.map((k) => k.coef), c.coef, 7)
    closeArr(df.glm('y', ['a', 'b'], { family: 'binomial', offset: 'off' }).coefficients.map((k) => k.coef), c.coef, 7)
    const o = ref.ologit
    const od = DataFrame.fromColumns({ y: o.y, x1: o.x1, x2: o.x2 })
    closeArr(od.ologit('y', ['x1', 'x2']).coefficients.map((k) => k.coef), o.coef, 5)
    const m = ref.mlogit
    const md = DataFrame.fromColumns({ y: m.y, x: m.x })
    close(md.mlogit('y', ['x']).logLik, m.logLik, 7)
    const gd = DataFrame.fromColumns({ y: ref.glm.y, a: ref.glm.a, b: ref.glm.b, cv: ref.glm.cv })
    close(gd.linearModel('y ~ a*b + cv').anova.error.ss, ref.glm.sse, 8)
    close((await gd.lazy().linearModel('y ~ a + b')).n, 34, 12)
  })
})
