/**
 * Numerical convergence: columna/advanced vs scipy / numpy on identical inputs.
 *
 *   pnpm --filter @columna/bench run advanced:convergence
 */
import { mkdirSync, writeFileSync } from 'node:fs'
import * as A from 'columna/advanced'

const OUT = new URL('../results/advanced-convergence-columna.json', import.meta.url)
const g = A.random(20260915)

type Case = {
  id: string
  group: string
  label: string
  python: 'scipy' | 'numpy'
  columna: Record<string, number | number[]>
  inputs: Record<string, unknown>
  notes?: string
}

const cases: Case[] = []
const add = (c: Case) => cases.push(c)

// ---- Distributions ---------------------------------------------------------------------------------
{
  const x = Array.from({ length: 200 }, (_, i) => -3 + (6 * i) / 199)
  const u = Array.from({ length: 200 }, (_, i) => 0.001 + (0.998 * i) / 199)
  const xPos = x.map((v) => Math.abs(v) * 2 + 0.01)
  const xGam = x.map((v) => Math.abs(v) * 3 + 0.01)
  add({
    id: 'dist.normal.cdf',
    group: 'Distributions',
    label: 'normal.cdf (200-point grid)',
    python: 'scipy',
    inputs: { x },
    columna: { values: Array.from(A.normal().map('cdf', x)) },
  })
  add({
    id: 'dist.normal.ppf',
    group: 'Distributions',
    label: 'normal.ppf (200-point grid)',
    python: 'scipy',
    inputs: { u },
    columna: { values: Array.from(A.normal().map('ppf', u)) },
  })
  add({
    id: 'dist.t.cdf',
    group: 'Distributions',
    label: 't(10).cdf',
    python: 'scipy',
    inputs: { x },
    columna: { values: Array.from(A.t(10).map('cdf', x)) },
  })
  add({
    id: 'dist.t.ppf',
    group: 'Distributions',
    label: 't(10).ppf',
    python: 'scipy',
    inputs: { u },
    columna: { values: Array.from(A.t(10).map('ppf', u)) },
  })
  add({
    id: 'dist.chi2.sf',
    group: 'Distributions',
    label: 'chi2(5).sf',
    python: 'scipy',
    inputs: { x: xPos },
    columna: { values: Array.from(A.chi2(5).map('sf', xPos)) },
  })
  add({
    id: 'dist.gamma.cdf',
    group: 'Distributions',
    label: 'gamma(2.5, scale=3).cdf',
    python: 'scipy',
    inputs: { x: xGam },
    columna: { values: Array.from(A.gamma(2.5, 3).map('cdf', xGam)) },
  })
  add({
    id: 'dist.beta.ppf',
    group: 'Distributions',
    label: 'beta(2,5).ppf',
    python: 'scipy',
    inputs: { u },
    columna: { values: Array.from(A.beta(2, 5).map('ppf', u)) },
  })
  add({
    id: 'dist.weibull.ppf',
    group: 'Distributions',
    label: 'weibull(1.8,50).ppf',
    python: 'scipy',
    inputs: { u },
    columna: { values: Array.from(A.weibull(1.8, 50).map('ppf', u)) },
  })
  const q = Array.from({ length: 40 }, (_, i) => 0.5 + (4.5 * i) / 39)
  const p = Array.from({ length: 40 }, (_, i) => 0.5 + (0.49 * i) / 39)
  add({
    id: 'dist.ptukey',
    group: 'Distributions',
    label: 'ptukey(q; k=4, df=20)',
    python: 'scipy',
    inputs: { q },
    columna: { values: q.map((qi) => A.ptukey(qi, 4, 20)) },
  })
  add({
    id: 'dist.qtukey',
    group: 'Distributions',
    label: 'qtukey(p; k=4, df=20)',
    python: 'scipy',
    inputs: { p },
    columna: { values: p.map((pi) => A.qtukey(pi, 4, 20)) },
  })
}

// ---- Basic statistics ------------------------------------------------------------------------------
{
  const x1 = Array.from(g.normal(500, 50, 10))
  const a = Array.from(g.normal(400))
  const b = Array.from(g.normal(400, 0.15, 1.1))
  const t1 = A.ttest1(x1, { mu: 50 })
  const t2 = A.ttest2(a, b)
  const z1 = A.ztest1(x1, { sigma: 10, mu: 50 })
  add({
    id: 'ttest1',
    group: 'Basic statistics',
    label: 'one-sample t (n=500)',
    python: 'scipy',
    inputs: { x: x1, mu: 50 },
    columna: { statistic: t1.statistic, pValue: t1.pValue, estimate: t1.estimate },
  })
  add({
    id: 'ttest2',
    group: 'Basic statistics',
    label: 'Welch two-sample t (n=400+400)',
    python: 'scipy',
    inputs: { a, b },
    columna: { statistic: t2.statistic, pValue: t2.pValue, estimate: t2.estimate },
  })
  add({
    id: 'ztest1',
    group: 'Basic statistics',
    label: 'one-sample z (known σ)',
    python: 'numpy',
    inputs: { x: x1, mu: 50, sigma: 10 },
    columna: { statistic: z1.statistic, pValue: z1.pValue },
  })
  const p1 = A.propTest1(350, 1000, { p0: 0.3 })
  add({
    id: 'propTest1',
    group: 'Basic statistics',
    label: 'binomial exact 350/1000 vs 0.3',
    python: 'scipy',
    inputs: { k: 350, n: 1000, p0: 0.3 },
    columna: { pValue: p1.pValue, estimate: p1.estimate },
  })
  const fisher = A.propTest2(120, 400, 90, 380, { method: 'fisher' })
  add({
    id: 'propTest2.fisher',
    group: 'Basic statistics',
    label: 'Fisher exact 2×2',
    python: 'scipy',
    inputs: { table: [[120, 280], [90, 290]] },
    columna: { pValue: fisher.pValue },
  })
  const xa = Array.from(g.normal(300))
  const xb = xa.map((v) => 0.6 * v + g.normal(1)[0]!)
  const cp = A.corrTest(xa, xb)
  const cs = A.corrTest(xa, xb, { method: 'spearman' })
  add({
    id: 'corrTest.pearson',
    group: 'Basic statistics',
    label: 'Pearson r (n=300)',
    python: 'scipy',
    inputs: { a: xa, b: xb },
    columna: { estimate: cp.estimate, statistic: cp.statistic, pValue: cp.pValue },
  })
  add({
    id: 'corrTest.spearman',
    group: 'Basic statistics',
    label: 'Spearman ρ (n=300)',
    python: 'scipy',
    inputs: { a: xa, b: xb },
    columna: { estimate: cs.estimate, pValue: cs.pValue },
  })
  const xv = Array.from(g.normal(200, 0, 2))
  const vt = A.varTest1(xv, { sigma0: 2 })
  add({
    id: 'varTest1',
    group: 'Basic statistics',
    label: 'χ² variance test',
    python: 'scipy',
    inputs: { x: xv, sigma0: 2 },
    columna: { statistic: vt.statistic, pValue: vt.pValue },
  })
}

// ---- ANOVA / tables --------------------------------------------------------------------------------
{
  const groups: Record<string, number[]> = {}
  for (let i = 0; i < 4; i++) groups[`g${i}`] = Array.from(g.normal(80, i * 0.4))
  const an = A.anova(groups)
  const lv = A.levene(groups)
  const bt = A.bartlett(groups)
  add({
    id: 'anova',
    group: 'ANOVA',
    label: 'one-way ANOVA 4×80',
    python: 'scipy',
    inputs: { groups },
    columna: { statistic: an.statistic, pValue: an.pValue },
  })
  add({
    id: 'levene',
    group: 'ANOVA',
    label: 'Levene',
    python: 'scipy',
    inputs: { groups },
    columna: { statistic: lv.statistic, pValue: lv.pValue },
  })
  add({
    id: 'bartlett',
    group: 'ANOVA',
    label: 'Bartlett',
    python: 'scipy',
    inputs: { groups },
    columna: { statistic: bt.statistic, pValue: bt.pValue },
  })
  const table = Array.from({ length: 5 }, () => Array.from(g.integer(5, 8, 40)))
  const chi = A.chi2test(table)
  add({
    id: 'chi2test',
    group: 'Tables',
    label: 'χ² contingency 5×5',
    python: 'scipy',
    inputs: { table },
    columna: { statistic: chi.statistic, pValue: chi.pValue },
  })
  const observed = Array.from(g.integer(10, 20, 60))
  const gof = A.chi2gof(observed)
  add({
    id: 'chi2gof',
    group: 'Tables',
    label: 'χ² GOF (equal expected)',
    python: 'scipy',
    inputs: { observed },
    columna: { statistic: gof.statistic, pValue: gof.pValue },
  })
}

// ---- Normality / nonparametric ---------------------------------------------------------------------
{
  const xn = Array.from(g.normal(500))
  const ad = A.andersonDarling(xn)
  add({
    id: 'andersonDarling',
    group: 'Normality',
    label: 'Anderson–Darling A² (n=500)',
    python: 'scipy',
    inputs: { x: xn },
    columna: { statistic: ad.statistic },
    notes: 'Compare A²; p-value tables may differ from scipy',
  })
  const xsw = Array.from(g.normal(100))
  const sw = A.shapiroWilk(xsw)
  add({
    id: 'shapiroWilk',
    group: 'Normality',
    label: 'Shapiro–Wilk (n=100)',
    python: 'scipy',
    inputs: { x: xsw },
    columna: { statistic: sw.statistic, pValue: sw.pValue },
  })
  const mwA = Array.from(g.normal(120))
  const mwB = Array.from(g.normal(120, 0.25))
  const mw = A.mannWhitney(mwA, mwB)
  add({
    id: 'mannWhitney',
    group: 'Nonparametrics',
    label: 'Mann–Whitney U',
    python: 'scipy',
    inputs: { a: mwA, b: mwB },
    columna: { statistic: mw.statistic, pValue: mw.pValue },
  })
  const wx = Array.from(g.normal(80, 0.2))
  const wil = A.wilcoxonSigned(wx)
  const nNonZero = wx.filter((v) => v !== 0).length
  const wMinus = (nNonZero * (nNonZero + 1)) / 2 - wil.statistic
  add({
    id: 'wilcoxonSigned',
    group: 'Nonparametrics',
    label: 'Wilcoxon signed-rank',
    python: 'scipy',
    inputs: { x: wx },
    // Compare min(W+, W−) like scipy; p-value separately
    columna: { statistic: Math.min(wil.statistic, wMinus), pValue: wil.pValue },
    notes: 'columna exposes W+; comparison uses min(W+,W−) to match scipy',
  })
  const kg: Record<string, number[]> = {}
  for (let i = 0; i < 4; i++) kg[`g${i}`] = Array.from(g.normal(60, i * 0.3))
  const kr = A.kruskal(kg)
  add({
    id: 'kruskal',
    group: 'Nonparametrics',
    label: 'Kruskal–Wallis',
    python: 'scipy',
    inputs: { groups: kg },
    columna: { statistic: kr.statistic, pValue: kr.pValue },
  })
}

// ---- Regression ------------------------------------------------------------------------------------
{
  const n = 200
  const x1 = Array.from(g.normal(n))
  const x2 = Array.from(g.normal(n))
  const y = x1.map((v, i) => 1.5 + 2 * v - 0.7 * x2[i]! + 0.5 * g.normal(1)[0]!)
  const fit = A.ols(y, [x1, x2])
  add({
    id: 'ols',
    group: 'Regression',
    label: 'OLS + intercept (n=200, p=2)',
    python: 'numpy',
    inputs: { y, X: x1.map((v, i) => [v, x2[i]!]) },
    columna: {
      coefficients: fit.coefficients.map((c) => c.coef),
      r2: fit.r2,
      s: fit.s,
    },
  })
  const xg = Array.from(g.normal(150))
  const yg = xg.map((v) => (g.next() < 1 / (1 + Math.exp(-(0.2 + 1.2 * v))) ? 1 : 0))
  const gl = A.glm(yg, [xg], { family: 'binomial' })
  add({
    id: 'glm.binomial',
    group: 'Regression',
    label: 'GLM logit (n=150)',
    python: 'scipy',
    inputs: { y: yg, X: xg.map((v) => [v]) },
    columna: { coefficients: gl.coefficients.map((c) => c.coef) },
    notes: 'Python: scipy.optimize BFGS on Bernoulli log-likelihood',
  })
}

// ---- Time series / PCA / KM / linalg ---------------------------------------------------------------
{
  const ar: number[] = []
  let prev = 0
  for (let i = 0; i < 120; i++) {
    prev = 0.7 * prev + g.normal(1)[0]!
    ar.push(prev)
  }
  const ac = A.acf(ar, { maxLag: 5 })
  const arima = A.arima(ar, { p: 1, d: 0, q: 0, method: 'CSS-ML' })
  add({
    id: 'acf',
    group: 'Time series',
    label: 'ACF lags 0…5',
    python: 'numpy',
    inputs: { y: ar },
    columna: { acf: ac.acf.slice(0, 6) },
  })
  add({
    id: 'arima.ar1',
    group: 'Time series',
    label: 'ARIMA(1,0,0) φ',
    python: 'numpy',
    inputs: { y: ar },
    columna: { ar: arima.ar, intercept: arima.intercept },
    notes: 'numpy reference = OLS y_t ~ y_{t-1}; columna uses CSS-ML',
  })

  const z1 = Array.from(g.normal(80))
  const z2 = z1.map((v) => 0.8 * v + 0.6 * g.normal(1)[0]!)
  const z3 = Array.from(g.normal(80))
  const mat = z1.map((_, i) => [z1[i]!, z2[i]!, z3[i]!])
  const pc = A.pca(mat, { scale: true })
  add({
    id: 'pca',
    group: 'Multivariate',
    label: 'PCA eigenvalues (correlation)',
    python: 'numpy',
    inputs: { X: mat },
    columna: { eigenvalues: pc.eigenvalues },
  })

  const timeRaw = Array.from(g.exponential(60, 40))
  const censor = timeRaw.map((t) => (t > 50 ? 1 : 0))
  const time = timeRaw.map((t) => Math.min(t, 50))
  const km = A.kaplanMeier(time, { censor })
  add({
    id: 'kaplanMeier',
    group: 'Reliability',
    label: 'Kaplan–Meier S(t)',
    python: 'numpy',
    inputs: { time, censor },
    columna: {
      time: km.curve.map((p) => p.time),
      survival: km.curve.map((p) => p.survival),
    },
    notes: 'numpy: product-limit formula (no lifelines)',
  })

  const Amat = [
    [4, 1, 0.5],
    [1, 3, 0.2],
    [0.5, 0.2, 2],
  ]
  const bv = [1, 2, 0.5]
  const sol = A.lstsq(A.matrix(3, 3, Float64Array.from(Amat.flat())), bv)
  add({
    id: 'lstsq',
    group: 'Linear algebra',
    label: '3×3 lstsq',
    python: 'numpy',
    inputs: { A: Amat, b: bv },
    columna: { coef: Array.from(sol.coef) },
  })
}

mkdirSync(new URL('.', OUT), { recursive: true })
writeFileSync(
  OUT,
  JSON.stringify({
    generatedAt: new Date().toISOString(),
    seed: 20260915,
    nCases: cases.length,
    cases,
  }),
)
console.log(`Wrote ${cases.length} cases → ${OUT.pathname}`)
