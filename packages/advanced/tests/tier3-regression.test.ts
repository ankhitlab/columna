import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import { bestSubsets, fittedLine, lstsq, matrixFromColumns, ols, qr, stepwise, svd } from '@columna/advanced'

// Reference values from numpy / scipy (tests/refs/tier3_ref.py)
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
  return { u, normal }
}
const sample = (n: number, g: () => number) => Array.from({ length: n }, g)

describe('linear algebra', () => {
  it('QR / lstsq reproduce a known solution and detect rank deficiency; SVD recovers singular values', () => {
    const X = matrixFromColumns([[1, 1, 1, 1], [1, 2, 3, 4], [2, 4, 6, 8]])
    const f = lstsq(X, [3, 5, 7, 9])
    expect(f.rank).toBe(2)
    expect(f.dependent).toEqual([2])
    expect(f.coef[2]).toBe(0)
    close(f.coef[0]!, 1, 10)
    close(f.coef[1]!, 2, 10)
    close(f.sse, 0, 10)
    const q = qr(matrixFromColumns([[1, 2, 3], [4, 5, 6.5]]))
    // QᵀQ = I
    for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) {
      let s = 0
      for (let i = 0; i < 3; i++) s += q.q.data[i * 2 + a]! * q.q.data[i * 2 + b]!
      close(s, a === b ? 1 : 0, 12)
    }
    const s = svd(matrixFromColumns([[3, 0], [0, 4], [0, 0]]))
    closeArr(s.s, [4, 3], 12)
  })
})

describe('3.1 / 3.2 regression (OLS) with diagnostics', () => {
  const c = ref.ols
  const r = ols(c.y, { x1: c.x1, x2: c.x2, x3: c.x3 })
  it('coefficients, SE, t, p, CI, VIF match numpy', () => {
    closeArr(r.coefficients.map((k) => k.coef), c.coef, 9)
    closeArr(r.coefficients.map((k) => k.se), c.se, 9)
    closeArr(r.coefficients.map((k) => k.t), c.t, 8)
    closeArr(r.coefficients.map((k) => k.pValue), c.p, 10)
    closeArr(r.coefficients.map((k) => k.ci[0]), c.ci[0], 9)
    closeArr(r.coefficients.map((k) => k.ci[1]), c.ci[1], 9)
    closeArr(r.coefficients.slice(1).map((k) => k.vif!), c.vif, 8)
    expect(r.coefficients[0]!.vif).toBeUndefined()
    expect(r.coefficients.map((k) => k.name)).toEqual(['Constant', 'x1', 'x2', 'x3'])
  })
  it('model summary and ANOVA (sequential / adjusted SS)', () => {
    close(r.s, c.s)
    close(r.r2, c.r2)
    close(r.r2adj, c.r2adj)
    close(r.r2pred, c.r2pred, 8)
    close(r.press, c.press, 8)
    close(r.anova.regression.ss, c.ssr, 8)
    close(r.anova.error.ss, c.sse, 8)
    close(r.anova.regression.f!, c.F, 8)
    close(r.anova.regression.pValue!, c.pF, 10)
    expect(r.anova.regression.df).toBe(3)
    expect(r.anova.error.df).toBe(21)
    expect(r.anova.total.df).toBe(24)
    closeArr(r.terms.map((t) => t.seqSS), c.seqSS, 8)
    closeArr(r.terms.map((t) => t.adjSS), c.adjSS, 8)
    close(r.terms.reduce((s, t) => s + t.seqSS, 0), c.ssr, 8)
    close(r.durbinWatson, c.dw, 10)
    close(r.logLik, c.logLik, 9)
    close(r.aic, c.aic, 9)
    close(r.bic, c.bic, 9)
  })
  it('residual diagnostics: leverage, standardized / deleted-t residuals, Cook, DFFITS, unusual observations', () => {
    closeArr(r.leverage, c.leverage, 10)
    closeArr(r.standardizedResiduals, c.std, 9)
    closeArr(r.studentizedResiduals, c.stud, 9)
    closeArr(r.cooksD, c.cook, 9)
    closeArr(r.dffits, c.dffits, 9)
    const flagged = r.unusual.find((u) => u.index === 7)
    expect(flagged?.flags).toContain('R')
    for (const u of r.unusual) expect(u.flags.includes('R') ? Math.abs(u.stdResidual) > 2 : u.leverage > (3 * 4) / 25).toBe(true)
  })
  it('predict with confidence and prediction intervals', () => {
    const p = r.predict(c.predict.x)[0]!
    close(p.fit, c.predict.fit, 9)
    close(p.se, c.predict.se, 9)
    close(p.ci[0], c.predict.ci[0], 8)
    close(p.pi[1], c.predict.pi[1], 8)
    expect(r.predict([c.predict.x, c.predict.x]).length).toBe(2)
    expect(() => r.predict([1, 2])).toThrow(RangeError)
  })
  it('input handling: missing rows dropped, no intercept, weights, aliased column, errors', () => {
    const y2 = [...c.y]
    y2[3] = null
    const r2 = ols(y2, [c.x1, c.x2], { names: ['a', 'b'] })
    expect(r2.n).toBe(24)
    expect(r2.omitted).toEqual([3])
    expect(r2.coefficients[1]!.name).toBe('a')
    const noInt = ols(c.y, { x1: c.x1 }, { intercept: false })
    expect(noInt.coefficients.length).toBe(1)
    expect(noInt.anova.total.df).toBe(25)
    // unit weights reproduce OLS; doubled weights reproduce coefficients
    const w1 = ols(c.y, { x1: c.x1, x2: c.x2 }, { weights: new Array(25).fill(1) })
    const w2 = ols(c.y, { x1: c.x1, x2: c.x2 }, { weights: new Array(25).fill(2) })
    const plain = ols(c.y, { x1: c.x1, x2: c.x2 })
    closeArr(w1.coefficients.map((k) => k.se), plain.coefficients.map((k) => k.se), 10)
    closeArr(w2.coefficients.map((k) => k.coef), plain.coefficients.map((k) => k.coef), 10)
    // aliased predictor (x1 duplicated) gets coef 0 and the flag; the fit is unchanged
    const al = ols(c.y, { x1: c.x1, dup: c.x1.map((v: number) => 2 * v), x2: c.x2 })
    expect(al.coefficients[2]!.aliased).toBe(true)
    close(al.s, plain.s, 10)
    expect(() => ols([1, 2], { x: [1, 2] })).toThrow(RangeError)
    expect(() => ols([1, 2, 3, 4], { x: [1, 2, 3] })).toThrow(RangeError)
  })
  it('fitted line (quadratic) matches numpy polyfit and evaluates the curve', () => {
    const q = ref.fittedLine
    const f = fittedLine(q.x, q.y, { degree: 2 })
    closeArr(f.coefficients.map((k) => k.coef), q.coef, 8)
    close(f.r2, q.r2, 10)
    close(f.curve(3), q.coef[0] + q.coef[1] * 3 + q.coef[2] * 9, 8)
    expect(f.equation).toMatch(/^y = /)
    const lg = fittedLine([1, 10, 100, 1000], [2, 4, 8, 16], { logX: true, logY: true })
    close(lg.curve(100), 8, 6)
  })

  it('Monte-Carlo (seeded): unbiased β, 95 % CI coverage, uniform p under H0, R²pred < R²', () => {
    const g = rng(3101)
    const reps = 2000
    const n = 30
    let coverB1 = 0
    let coverPred = 0
    let rejNull = 0
    let sumB1 = 0
    let predBelow = 0
    for (let i = 0; i < reps; i++) {
      const x1 = sample(n, g.normal)
      const x2 = sample(n, g.normal)
      const y = x1.map((v, j) => 1 + 2 * v + 0 * x2[j]! + g.normal())
      const r = ols(y, { x1, x2 })
      const b1 = r.coefficients[1]!
      sumB1 += b1.coef
      if (b1.ci[0] <= 2 && 2 <= b1.ci[1]) coverB1++
      if (r.coefficients[2]!.pValue < 0.05) rejNull++
      // prediction interval coverage at a new point
      const xn = g.normal()
      const yn = 1 + 2 * xn + g.normal()
      const p = r.predict([xn, g.normal()])[0]!
      if (p.pi[0] <= yn && yn <= p.pi[1]) coverPred++
      if (r.r2pred < r.r2) predBelow++
    }
    expect(Math.abs(sumB1 / reps - 2)).toBeLessThan(0.02)
    expect(coverB1 / reps).toBeGreaterThan(0.93)
    expect(coverB1 / reps).toBeLessThan(0.97)
    expect(coverPred / reps).toBeGreaterThan(0.93)
    expect(coverPred / reps).toBeLessThan(0.97)
    expect(rejNull / reps).toBeGreaterThan(0.035)
    expect(rejNull / reps).toBeLessThan(0.065)
    expect(predBelow).toBe(reps)
  })
})

describe('3.3 stepwise and best subsets', () => {
  const c = ref.subsets
  const cols = Object.fromEntries([0, 1, 2, 3, 4].map((j) => [`z${j + 1}`, c.Z.map((row: number[]) => row[j])]))
  it('best subsets: top-2 per size by R² with Cp and S match the exhaustive numpy search', () => {
    const b = bestSubsets(c.y, cols)
    expect(b.subsets.length).toBe(c.best.length)
    close(b.mseFull, c.mseFull, 10)
    for (let i = 0; i < c.best.length; i++) {
      expect(b.subsets[i]!.variables).toEqual(c.best[i].vars)
      close(b.subsets[i]!.r2, c.best[i].r2, 10)
      close(b.subsets[i]!.cp, c.best[i].cp, 8)
      close(b.subsets[i]!.s, c.best[i].s, 10)
    }
    const k2 = bestSubsets(c.y, cols, { maxK: 2, nBest: 1 })
    expect(k2.subsets.map((s) => s.size)).toEqual([1, 2])
    const inc = bestSubsets(c.y, cols, { include: ['z5'], nBest: 1 })
    expect(inc.subsets.every((s) => s.variables.includes('z5'))).toBe(true)
    // the full model has Cp = p exactly
    close(b.subsets[b.subsets.length - 1]!.cp, 6, 8)
  })
  it('stepwise reproduces the p-value driven path and the final model', () => {
    const s = stepwise(c.y, cols)
    expect(s.steps.map((st) => `${st.action}:${st.variable}`)).toEqual(c.stepwise.map((st: { add?: string; remove?: string }) => (st.add ? `add:${st.add}` : `remove:${st.remove}`)))
    expect(s.selected).toEqual(c.final)
    closeArr(s.steps.map((st) => st.pValue), c.stepwise.map((st: { p: number }) => st.p), 10)
    expect(s.model.coefficients.map((k) => k.name)).toEqual(['Constant', ...c.final])
    const fw = stepwise(c.y, cols, { method: 'forward' })
    expect(fw.selected).toEqual(c.final)
    const bw = stepwise(c.y, cols, { method: 'backward', alphaOut: 0.15 })
    expect(bw.selected.sort()).toEqual([...c.final].sort())
    expect(bw.steps.every((st) => st.action === 'remove')).toBe(true)
    const keep = stepwise(c.y, cols, { include: ['z5'] })
    expect(keep.selected).toContain('z5')
    expect(() => stepwise(c.y, cols, { alphaIn: 0.2, alphaOut: 0.1 })).toThrow(RangeError)
  })
  it('Monte-Carlo (seeded): stepwise recovers the true predictors and rejects noise ones most of the time', () => {
    const g = rng(3303)
    const reps = 300
    let exact = 0
    let containsTrue = 0
    for (let i = 0; i < reps; i++) {
      const n = 60
      const Z = Array.from({ length: 6 }, () => sample(n, g.normal))
      const y = Array.from({ length: n }, (_, j) => 2 * Z[0]![j]! - 1.5 * Z[1]![j]! + g.normal())
      const s = stepwise(y, Z, { alphaIn: 0.05, alphaOut: 0.1 })
      const sel = [...s.selected].sort()
      if (sel.includes('x1') && sel.includes('x2')) containsTrue++
      if (sel.length === 2 && sel[0] === 'x1' && sel[1] === 'x2') exact++
    }
    expect(containsTrue / reps).toBeGreaterThan(0.99)
    expect(exact / reps).toBeGreaterThan(0.75) // 4 noise predictors at α = 0.05 → P(none enters) ≈ 0.8
  })
})

describe('DataFrame methods (Tier 3 regression)', () => {
  it('regress / fittedLine / stepwise / bestSubsets on columns', async () => {
    const c = ref.ols
    const df = DataFrame.fromColumns({ y: c.y, x1: c.x1, x2: c.x2, x3: c.x3 })
    const r = df.regress('y', ['x1', 'x2', 'x3'])
    closeArr(r.coefficients.map((k) => k.coef), c.coef, 9)
    expect(r.names).toEqual(['x1', 'x2', 'x3'])
    const w = df.regress('y', ['x1'], { weights: 'x2' })
    expect(w.n).toBe(25)
    close(df.fittedLine('x1', 'y').r2, ols(c.y, { x1: c.x1 }).r2, 12)
    expect(df.stepwise('y', ['x1', 'x2', 'x3']).selected.length).toBeGreaterThan(0)
    expect(df.bestSubsets('y', ['x1', 'x2', 'x3']).subsets.length).toBe(5) // 2 + 2 + 1
    close((await df.lazy().regress('y', ['x1'])).r2, ols(c.y, { x1: c.x1 }).r2, 12)
  })
})
