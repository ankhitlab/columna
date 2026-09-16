import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import { nls, ols, orthogonalRegression, pls } from '@columna/advanced'

const ref = JSON.parse(readFileSync(new URL('./fixtures/tier3-scipy.json', import.meta.url), 'utf8'))
const close = (got: number, want: number, digits = 10) => expect(got).toBeCloseTo(want, digits)
const closeArr = (got: ArrayLike<number>, want: number[], digits = 10) => {
  expect(got.length).toBe(want.length)
  for (let i = 0; i < want.length; i++) close(got[i]!, want[i]!, digits)
}
const rel = (got: number, want: number, tol: number) => expect(Math.abs(got - want) / Math.abs(want)).toBeLessThan(tol)

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
const sample = (n: number, g: () => number) => Array.from({ length: n }, g)

describe('3.7 nonlinear regression (Levenberg–Marquardt)', () => {
  const c = ref.nls
  const misra = (x: number | number[], [b1, b2]: number[]) => b1! * (1 - Math.exp(-b2! * (x as number)))
  it('NIST StRD Misra1a: certified parameters, SEs and residual SS from the harder start (500, 1e-4)', () => {
    const r = nls(misra, c.x, c.y, { start: [500, 1e-4], names: ['b1', 'b2'] })
    expect(r.converged).toBe(true)
    rel(r.parameters[0]!.estimate, c.nist.b[0], 1e-8)
    rel(r.parameters[1]!.estimate, c.nist.b[1], 1e-8)
    rel(r.parameters[0]!.se, c.nist.se[0], 1e-7)
    rel(r.parameters[1]!.se, c.nist.se[1], 1e-7)
    rel(r.sse, c.nist.sse, 1e-9) // SSE is flat near the optimum: agrees to 1e-9, parameters to 1e-8 (finite-difference Jacobian)
    // and scipy curve_fit on the same data
    closeArr(r.parameters.map((p) => p.estimate), c.params, 6)
    closeArr(r.parameters.map((p) => p.se), c.se, 6)
    expect(r.parameters.map((p) => p.name)).toEqual(['b1', 'b2'])
    expect(r.df).toBe(12)
    expect(r.history.length).toBeGreaterThan(1)
    expect(r.history[r.history.length - 1]!.sse).toBeLessThanOrEqual(r.history[0]!.sse)
    expect(Math.abs(r.correlation.data[1]!)).toBeLessThan(1)
    // analytic Jacobian gives the same answer
    const j = nls(misra, c.x, c.y, { start: [500, 1e-4], jacobian: (x, [b1, b2]) => [1 - Math.exp(-b2! * (x as number)), b1! * (x as number) * Math.exp(-b2! * (x as number))] })
    rel(j.parameters[1]!.estimate, c.nist.b[1], 1e-9)
    rel(j.parameters[1]!.se, c.nist.se[1], 1e-8)
  })
  it('predict, bounds, weights and multi-predictor models', () => {
    const r = nls(misra, c.x, c.y, { start: [500, 1e-4] })
    const p = r.predict(400)[0]!
    close(p.fit, misra(400, r.parameters.map((q) => q.estimate)), 10)
    expect(p.pi[0]).toBeLessThan(p.ci[0])
    expect(r.predict([100, 200]).length).toBe(2)
    const b = nls(misra, c.x, c.y, { start: [500, 1e-4], bounds: [[0, 200], [0, 1]] })
    close(b.parameters[0]!.estimate, 200, 8)
    // weighted: doubling all weights leaves the estimates unchanged
    const w = nls(misra, c.x, c.y, { start: [500, 1e-4], weights: new Array(c.x.length).fill(2) })
    closeArr(w.parameters.map((q) => q.estimate), r.parameters.map((q) => q.estimate), 7)
    // two predictors: y = a·x1 + b·x2² (linear in parameters → equals OLS)
    const g = rng(3701)
    const x1 = sample(30, g.normal)
    const x2 = sample(30, g.normal)
    const y = x1.map((v, i) => 2 * v + 0.5 * x2[i]! ** 2 + 0.1 * g.normal())
    const m = nls((x, [a, bb]) => a! * (x as number[])[0]! + bb! * (x as number[])[1]! ** 2, x1.map((v, i) => [v, x2[i]!]), y, { start: [1, 1] })
    const o = ols(y, { x1, x2sq: x2.map((v) => v * v) }, { intercept: false })
    closeArr(m.parameters.map((q) => q.estimate), o.coefficients.map((k) => k.coef), 7)
    closeArr(m.parameters.map((q) => q.se), o.coefficients.map((k) => k.se), 6)
    expect(m.predict([1, 2]).length).toBe(1)
    expect(() => nls(misra, c.x, c.y, { start: [] })).toThrow(RangeError)
    expect(() => nls(misra, [1, 2], [1, 2, 3], { start: [1, 1] })).toThrow(RangeError)
  })
  it('Monte-Carlo (seeded): recovers exponential-decay parameters, CI coverage ≈ 95 %', () => {
    const g = rng(3702)
    const reps = 400
    const xs = Array.from({ length: 20 }, (_, i) => i * 0.5)
    let cover = 0
    let sumB = 0
    for (let i = 0; i < reps; i++) {
      const y = xs.map((x) => 5 * Math.exp(-0.4 * x) + 0.1 * g.normal())
      const r = nls((x, [a, b]) => a! * Math.exp(-b! * (x as number)), xs, y, { start: [3, 0.2] })
      const b = r.parameters[1]!
      sumB += b.estimate
      if (b.ci[0] <= 0.4 && 0.4 <= b.ci[1]) cover++
    }
    expect(Math.abs(sumB / reps - 0.4)).toBeLessThan(0.005)
    expect(cover / reps).toBeGreaterThan(0.92)
    expect(cover / reps).toBeLessThan(0.98)
  })
})

describe('3.8 orthogonal regression', () => {
  const c = ref.orthogonal
  it('slope and intercept match scipy.odr for error-variance ratios 1, 2 and 0.5; jackknife SE agrees with ODR to 25 %', () => {
    for (const f of c.fits) {
      const r = orthogonalRegression(c.x, c.y, { errorVarianceRatio: f.delta })
      close(r.slope.estimate, f.slope, 4) // ODR iterates to ~1e-6; the closed form is exact
      close(r.intercept.estimate, f.intercept, 4)
      rel(r.slope.se, f.sd[1], 0.25)
      rel(r.intercept.se, f.sd[0], 0.25)
      expect(r.slope.ci[0]).toBeLessThan(r.slope.estimate)
    }
    const r = orthogonalRegression(c.x, c.y)
    expect(r.errorVarianceRatio).toBe(1)
    expect(r.slopeEqualsOne.pValue).toBeLessThan(0.05) // true slope 1.1
    // δ → ∞ is OLS of y on x; δ → 0 is the inverse OLS of x on y
    const big = orthogonalRegression(c.x, c.y, { errorVarianceRatio: 1e9 })
    close(big.slope.estimate, ols(c.y, { x: c.x }).coefficients[1]!.coef, 5)
    const small = orthogonalRegression(c.x, c.y, { errorVarianceRatio: 1e-9 })
    close(small.slope.estimate, 1 / ols(c.x, { y: c.y }).coefficients[1]!.coef, 5)
    expect(() => orthogonalRegression([1, 2], [1, 2])).toThrow(RangeError)
  })
  it('Monte-Carlo (seeded): unbiased slope under errors in both variables where OLS is attenuated', () => {
    const g = rng(3801)
    const reps = 400
    let sumOrth = 0
    let sumOls = 0
    let cover = 0
    for (let i = 0; i < reps; i++) {
      const xi = sample(40, () => 5 + 2 * g.normal())
      const x = xi.map((v) => v + g.normal())
      const y = xi.map((v) => 1 + 1.5 * v + g.normal())
      const r = orthogonalRegression(x, y, { errorVarianceRatio: 1 })
      sumOrth += r.slope.estimate
      sumOls += ols(y, { x }).coefficients[1]!.coef
      if (r.slope.ci[0] <= 1.5 && 1.5 <= r.slope.ci[1]) cover++
    }
    expect(Math.abs(sumOrth / reps - 1.5)).toBeLessThan(0.03)
    expect(sumOls / reps).toBeLessThan(1.3) // attenuation: 1.5 · 4/(4 + 1) = 1.2
    expect(cover / reps).toBeGreaterThan(0.9)
  })
})

describe('3.8 partial least squares', () => {
  const c = ref.pls
  const cols = [0, 1, 2, 3].map((j) => c.X.map((row: number[]) => row[j]))
  it('NIPALS coefficients, R²X / R²Y per component and LOO predicted R² match the numpy implementation', () => {
    for (const A of [1, 2, 4]) {
      const r = pls(c.y, cols, { components: A, crossValidate: false })
      const want = c[String(A)]
      closeArr(r.coefficients[0]!.coef, want.coef, 8)
      close(r.coefficients[0]!.constant, want.constant, 8)
      close(r.summary[A - 1]!.r2y, want.r2y, 8)
      close(r.summary[A - 1]!.r2x, want.r2x, 8)
    }
    const cv = pls(c.y, cols, { components: 4 })
    close(cv.summary[1]!.press!, c.press2, 7)
    close(cv.summary[1]!.r2pred!, c.r2pred2, 8)
    expect(cv.selected).toBeGreaterThanOrEqual(1)
    expect(cv.summary.map((s) => s.r2y)).toEqual([...cv.summary.map((s) => s.r2y)].sort((a, b) => a - b))
    // with all components PLS equals OLS
    const full = pls(c.y, cols, { components: 4, crossValidate: false })
    const o = ols(c.y, cols)
    closeArr(full.coefficients[0]!.coef, o.coefficients.slice(1).map((k) => k.coef), 8)
    close(full.summary[3]!.r2x, 1, 10)
    // standardized fit predicts the same as the centered fit with all components
    const st = pls(c.y, cols, { components: 4, standardize: true, crossValidate: false })
    closeArr(st.coefficients[0]!.coef, o.coefficients.slice(1).map((k) => k.coef), 8)
    close(full.predict(c.X[0])[0]![0]!, full.fitted[0]![0]!, 10)
    expect(full.leverage.every((h) => h > 0 && h < 1)).toBe(true)
    // two responses
    const y2 = c.y.map((v: number, i: number) => 2 * v - c.X[i][1])
    const multi = pls([c.y, y2], cols, { components: 2, crossValidate: false })
    expect(multi.m).toBe(2)
    expect(multi.coefficients.map((k) => k.response)).toEqual(['y1', 'y2'])
  })
  it('Monte-Carlo (seeded): with collinear predictors the CV-selected model predicts better than the full OLS out of sample', () => {
    const g = rng(3802)
    const reps = 60
    let plsWins = 0
    for (let i = 0; i < reps; i++) {
      const n = 25
      const latent = sample(n, g.normal)
      const X = Array.from({ length: 8 }, () => latent.map((v) => v + 0.3 * g.normal()))
      const y = latent.map((v) => 2 * v + 0.5 * g.normal())
      const r = pls(y, X)
      const o = ols(y, X)
      // fresh test data from the same process
      const lt = sample(50, g.normal)
      const Xt = Array.from({ length: 8 }, () => lt.map((v) => v + 0.3 * g.normal()))
      let ePls = 0
      let eOls = 0
      for (let t = 0; t < 50; t++) {
        const row = Xt.map((col) => col[t]!)
        const truth = 2 * lt[t]!
        ePls += (r.predict(row)[0]![0]! - truth) ** 2
        eOls += (o.predict(row)[0]!.fit - truth) ** 2
      }
      if (ePls < eOls) plsWins++
    }
    expect(plsWins / reps).toBeGreaterThan(0.8)
  })
})

describe('DataFrame methods (Tier 3 nonlinear / multivariate)', () => {
  it('nls / orthogonalRegression / pls on columns', async () => {
    const c = ref.nls
    const df = DataFrame.fromColumns({ x: c.x, y: c.y })
    const r = df.nls((x, [b1, b2]) => b1! * (1 - Math.exp(-b2! * (x as number))), 'x', 'y', { start: [500, 1e-4] })
    rel(r.parameters[0]!.estimate, c.nist.b[0], 1e-8)
    const o = ref.orthogonal
    const od = DataFrame.fromColumns({ x: o.x, y: o.y })
    close(od.orthogonalRegression('x', 'y').slope.estimate, o.fits[0].slope, 4)
    const p = ref.pls
    const pd = DataFrame.fromColumns({ y: p.y, a: p.X.map((r: number[]) => r[0]), b: p.X.map((r: number[]) => r[1]), c: p.X.map((r: number[]) => r[2]), d: p.X.map((r: number[]) => r[3]) })
    const pr = pd.pls('y', ['a', 'b', 'c', 'd'], { components: 2, crossValidate: false })
    closeArr(pr.coefficients[0]!.coef, p['2'].coef, 8)
    expect(pr.names).toEqual(['a', 'b', 'c', 'd'])
    const multi = df.nls((x, [b1, b2]) => b1! * (1 - Math.exp(-b2! * (x as number[])[0]!)), ['x'], 'y', { start: [500, 1e-4] })
    rel(multi.parameters[1]!.estimate, c.nist.b[1], 1e-8)
    close((await od.lazy().orthogonalRegression('x', 'y')).intercept.estimate, o.fits[0].intercept, 4)
  })
})
