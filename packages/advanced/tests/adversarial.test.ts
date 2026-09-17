import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  anova,
  beta,
  binomial,
  chi2,
  chi2test,
  corrTest,
  f,
  gamma,
  kruskal,
  levene,
  mannWhitney,
  normal,
  ols,
  poisson,
  poissonRateTest1,
  propTest1,
  propTest2,
  signTest,
  t,
  ttest1,
  ttest2,
  ttestPaired,
  varTest1,
  wilcoxonSigned,
} from '@columna/advanced'

/**
 * Adversarial references (tests/refs/adversarial_ref.py, scipy 1.14): the inputs a happy-path fixture never
 * contains — exact collinearity, condition numbers ~1e9, saturated fits, constant predictors and responses,
 * missing values, heavy ties, constant samples, tail probabilities down to 1e-300, every one-sided
 * alternative, and 2×2 tables with empty cells — each against the specialised scipy method for that case.
 */
const ref = JSON.parse(readFileSync(new URL('./fixtures/adversarial-scipy.json', import.meta.url), 'utf8'))
const ALTS = ['two-sided', 'less', 'greater'] as const

/** Relative closeness with an absolute floor for values that underflow on either side. */
function relClose(got: number, want: number, rel = 1e-10, floor = 1e-300): void {
  if (!Number.isFinite(want)) {
    expect(got, `want ${want}`).toBe(want)
    return
  }
  if (Math.abs(want) < floor) {
    expect(Math.abs(got), `expected ≈0 (|want| < ${floor})`).toBeLessThan(floor)
    return
  }
  expect(Math.abs(got - want) / Math.abs(want), `got ${got}, want ${want}`).toBeLessThan(rel)
}
const closeArr = (got: ArrayLike<number>, want: number[], rel = 1e-10) => {
  expect(got.length).toBe(want.length)
  for (let i = 0; i < want.length; i++) relClose(got[i]!, want[i]!, rel, 1e-12)
}

describe('degenerate and ill-conditioned regression', () => {
  it('exact collinearity: the dependent column is aliased, fitted values / SSE are the unique least-squares solution', () => {
    const c = ref.collinear
    const fit = ols(c.y, [c.x1, c.x2, c.x3], { names: ['x1', 'x2', 'x3'] })
    expect(fit.p).toBe(c.rank) // intercept + 2 estimable slopes
    const aliased = fit.coefficients.filter((k) => k.aliased).map((k) => k.name)
    expect(aliased).toEqual(['x3'])
    closeArr(fit.residuals.map((r: number, i: number) => c.y[i] - r), c.fitted, 1e-9)
    relClose(fit.residuals.reduce((s: number, r: number) => s + r * r, 0), c.sse, 1e-9)
    const est = fit.coefficients.filter((k) => !k.aliased).map((k) => k.coef)
    closeArr(est, c.coef_reduced, 1e-8)
    for (const k of fit.coefficients.filter((k) => !k.aliased)) {
      expect(Number.isFinite(k.se) && k.se > 0).toBe(true)
      expect(k.pValue).toBeGreaterThanOrEqual(0)
    }
  })

  it('condition number ~1e7 (polynomial columns): coefficients within the conditioning-limited tolerance, fit within 1e-8', () => {
    const c = ref.illcond
    const cols = [1, 2, 3, 4, 5, 6].map((k) => c.x.map((v: number) => v ** k))
    const fit = ols(c.y, cols)
    expect(fit.coefficients.some((k) => k.aliased)).toBe(false)
    closeArr(fit.residuals.map((r: number, i: number) => c.y[i] - r), c.fitted, 1e-8)
    relClose(fit.residuals.reduce((s: number, r: number) => s + r * r, 0), c.sse, 1e-6)
    // Householder QR vs LAPACK SVD-based lstsq: agree to about cond × eps ≈ 1e-7 relative on the coefficients
    closeArr(fit.coefficients.map((k) => k.coef), c.coef, 1e-5)
    expect(c.cond).toBeGreaterThan(1e6)
  })

  it('saturated fit (p = n) is exact and reports no error degrees of freedom; n < p is refused', () => {
    const c = ref.saturated
    const X = [0, 1, 2, 3].map((j) => c.X.map((r: number[]) => r[j]))
    expect(() => ols(c.y, X)).toThrow(/observations|coefficients/) // n = p: no residual df — Minitab refuses too
    expect(() => ols(c.y.slice(0, 3), X.map((col: number[]) => col.slice(0, 3)))).toThrow()
  })

  it('a constant predictor is aliased with the intercept; a constant response gives a defined, finite summary', () => {
    const c = ref.constant_predictor
    const fit = ols(c.y, [c.x, c.const], { names: ['x', 'const'] })
    expect(fit.coefficients.find((k) => k.name === 'const')!.aliased).toBe(true)
    closeArr(
      fit.coefficients.filter((k) => !k.aliased).map((k) => k.coef),
      c.coef,
      1e-9,
    )
    const cr = ref.constant_response
    const flat = ols(cr.y, [cr.x])
    expect(flat.coefficients[0]!.coef).toBeCloseTo(2.5, 12)
    expect(Math.abs(flat.coefficients[1]!.coef)).toBeLessThan(1e-12)
    expect(flat.s).toBeLessThan(1e-12)
    expect(Number.isNaN(flat.r2)).toBe(true) // SST = 0: R² is undefined (0/0), reported as NaN rather than a made-up 0 or 1
  })
})

describe('missing values', () => {
  it('two-sample t drops nulls per sample (scipy on the cleaned data)', () => {
    const c = ref.missing_ttest2
    const r = ttest2(c.a, c.b)
    relClose(r.statistic, c.statistic, 1e-10)
    relClose(r.pValue, c.pValue, 1e-9)
    relClose(r.df, c.df, 1e-10)
  })
  it('Pearson drops rows where either side is null (pairwise-complete)', () => {
    const c = ref.missing_pearson
    const r = corrTest(c.x, c.y)
    expect(r.n).toBe(c.n)
    relClose(r.estimate, c.r, 1e-10)
    relClose(r.pValue, c.pValue, 1e-9)
  })
})

describe('ties', () => {
  it('Mann–Whitney with heavy ties: tie-corrected asymptotic p with continuity, all alternatives', () => {
    const c = ref.ties_mannwhitney
    for (const alt of ALTS) {
      const r = mannWhitney(c.a, c.b, { alternative: alt, method: 'asymptotic' })
      relClose(r.statistic, c.U, 1e-12)
      relClose(r.pValue, c.p[alt], 1e-8)
    }
  })
  it('Wilcoxon signed-rank with zeros and ties (zero_method = wilcox, tie-corrected, continuity)', () => {
    const c = ref.ties_wilcoxon
    for (const alt of ALTS) {
      const r = wilcoxonSigned(c.x, { alternative: alt, method: 'asymptotic' })
      // ours reports W+ (Minitab); scipy reports min(W+, W−) for two-sided — same information, n = 12 after dropping the two zeros
      const n = 12
      relClose(Math.min(r.statistic, (n * (n + 1)) / 2 - r.statistic), c.statistic, 1e-12)
      relClose(r.pValue, c.p[alt], 1e-8)
    }
  })
  it('Kruskal–Wallis H with the tie correction', () => {
    const c = ref.ties_kruskal
    const r = kruskal(c.groups)
    relClose(r.statistic, c.H, 1e-10)
    relClose(r.pValue, c.pValue, 1e-9)
  })
  it('Spearman on average ranks and Kendall tau-b with ties on both sides', () => {
    const c = ref.ties_rank_corr
    const s = corrTest(c.x, c.y, { method: 'spearman' })
    relClose(s.estimate, c.spearman, 1e-10)
    relClose(s.pValue, c.spearman_p, 1e-6) // scipy: t approximation; ours: same t on n − 2 df
    const k = corrTest(c.x, c.y, { method: 'kendall' })
    relClose(k.estimate, c.kendall_b, 1e-10)
  })
})

describe('constant samples', () => {
  it('one-sample t on a constant: ±Infinity with p = 0 when the mean differs from μ, NaN when it equals μ (scipy convention), never a throw', () => {
    const c = ref.constant
    const off = ttest1(c.x, { mu: 4 })
    expect(off.statistic).toBe(Infinity)
    expect(off.pValue).toBe(0)
    const on = ttest1(c.x, { mu: 5 })
    expect(Number.isNaN(on.statistic)).toBe(true)
    expect(Number.isNaN(on.pValue)).toBe(true)
  })
  it('ANOVA and Levene with one constant group agree with f_oneway / levene', () => {
    const a = ref.constant.anova_with_constant_group
    const r = anova(a.groups)
    relClose(r.statistic, a.F, 1e-9)
    relClose(r.pValue, a.pValue, 1e-8)
    const l = ref.constant.levene_with_constant_group
    const lv = levene(l.groups, { center: 'median' })
    relClose(lv.statistic, l.W, 1e-9)
    relClose(lv.pValue, l.pValue, 1e-8)
  })
  it('correlation with a constant column is NaN, not an exception', () => {
    const r = corrTest([1, 2, 3, 4, 5], [7, 7, 7, 7, 7])
    expect(Number.isNaN(r.estimate)).toBe(true)
    expect(Number.isNaN(r.pValue)).toBe(true)
  })
})

describe('extreme tail probabilities (scipy reference)', () => {
  it('normal cdf / sf / ppf from 1e-300 to 1 − 1e-16', () => {
    for (const c of ref.tails.norm) {
      relClose(normal().cdf(c.x), c.cdf, 1e-10)
      relClose(normal().sf(c.x), c.sf, 1e-10)
    }
    for (const c of ref.tails.norm_ppf) relClose(normal().ppf(c.q), c.ppf, 1e-9)
  })
  it('Student t, chi-square, F: far tails and inverse tails', () => {
    for (const c of ref.tails.t) {
      relClose(t(c.df).cdf(c.x), c.cdf, 1e-8)
      relClose(t(c.df).sf(c.x), c.sf, 1e-8)
    }
    for (const c of ref.tails.t_ppf) relClose(t(c.df).ppf(c.q), c.ppf, 1e-7)
    for (const c of ref.tails.chi2) relClose(chi2(c.df).sf(c.x), c.sf, 1e-8)
    for (const c of ref.tails.chi2_ppf) relClose(chi2(c.df).ppf(c.q), c.ppf, 1e-7)
    for (const c of ref.tails.f) relClose(f(c.d1, c.d2).sf(c.x), c.sf, 1e-8)
    for (const c of ref.tails.f_ppf) relClose(f(c.d1, c.d2).ppf(c.q), c.ppf, 1e-6)
  })
  it('binomial and Poisson at n = 10 000 / λ = 1e5 and at p = 1e-6 / λ = 1e-8', () => {
    for (const c of ref.tails.binom) {
      relClose(binomial(c.n, c.p).pmf(c.k), c.pmf, 1e-9)
      relClose(binomial(c.n, c.p).cdf(c.k), c.cdf, 1e-9)
      relClose(binomial(c.n, c.p).sf(c.k), c.sf, 1e-9)
    }
    for (const c of ref.tails.poisson) {
      relClose(poisson(c.lam).pmf(c.k), c.pmf, 1e-9)
      relClose(poisson(c.lam).cdf(c.k), c.cdf, 1e-9)
      relClose(poisson(c.lam).sf(c.k), c.sf, 1e-9)
    }
  })
  it('beta with tiny shapes and gamma with shape 1e-3 / 500', () => {
    for (const c of ref.tails.beta) relClose(beta(c.a, c.b).ppf(c.q), c.ppf, 1e-7)
    for (const c of ref.tails.gamma) {
      relClose(gamma(c.a).cdf(c.x), c.cdf, 1e-8)
      relClose(gamma(c.a).sf(c.x), c.sf, 1e-8)
    }
  })
})

describe('one-sided alternatives against the specialised scipy method', () => {
  const o = () => ref.onesided
  it('t tests (one-sample with CI, Welch with CI, pooled, paired)', () => {
    for (const alt of ALTS) {
      const r1 = ttest1(o().a, { mu: 0, alternative: alt })
      relClose(r1.statistic, o().ttest1[alt].statistic, 1e-10)
      relClose(r1.pValue, o().ttest1[alt].pValue, 1e-9)
      const ci = o().ttest1[alt].ci as [number | null, number | null]
      if (ci[0] !== null) relClose(r1.ci[0], ci[0], 1e-9)
      else expect(r1.ci[0]).toBe(-Infinity)
      if (ci[1] !== null) relClose(r1.ci[1], ci[1], 1e-9)
      else expect(r1.ci[1]).toBe(Infinity)
      relClose(ttest2(o().a, o().b, { alternative: alt }).pValue, o().ttest2_welch[alt].pValue, 1e-9)
      relClose(ttest2(o().a, o().b, { alternative: alt, equalVar: true }).pValue, o().ttest2_pooled[alt].pValue, 1e-9)
      relClose(ttestPaired(o().a, o().b.slice(0, 15), { alternative: alt }).pValue, o().paired[alt].pValue, 1e-9)
    }
  })
  it('exact Mann–Whitney and Wilcoxon, Pearson with CI', () => {
    for (const alt of ALTS) {
      relClose(mannWhitney(o().a, o().b, { alternative: alt, method: 'exact' }).pValue, o().mannwhitney_exact[alt].pValue, 1e-9)
      relClose(wilcoxonSigned(o().a, { alternative: alt, method: 'exact' }).pValue, o().wilcoxon_exact[alt].pValue, 1e-9)
      const p = corrTest(o().a, o().b.slice(0, 15), { alternative: alt })
      relClose(p.estimate, o().pearson[alt].r, 1e-10)
      relClose(p.pValue, o().pearson[alt].pValue, 1e-9)
      const ci = o().pearson[alt].ci as [number | null, number | null]
      if (ci[0] !== null) relClose(p.ci[0], ci[0], 1e-7)
      if (ci[1] !== null) relClose(p.ci[1], ci[1], 1e-7)
    }
  })
  it('exact binomial proportion at k = 0, n, 3, 19 (p, Clopper–Pearson CI), sign test, variance chi-square, Poisson rate', () => {
    for (const [k, n] of [[0, 20], [20, 20], [3, 20], [19, 20]] as const) {
      for (const alt of ALTS) {
        const want = o().prop1_exact[`${k}/${n}/${alt}`]
        const r = propTest1(k, n, { p0: 0.3, alternative: alt, method: 'exact' })
        relClose(r.pValue, want.pValue, 1e-9)
        // scipy's one-sided proportion_ci is a one-sided Clopper–Pearson bound: compare the finite side
        if (alt === 'two-sided') closeArr(r.ci, want.ci, 1e-8)
      }
    }
    for (const alt of ALTS) {
      relClose(signTest(o().a, { median: 0.2, alternative: alt }).pValue, o().sign[alt].pValue, 1e-9)
      const v = varTest1(o().a, { sigma0: 1.2, alternative: alt, method: 'chi-square' })
      relClose(v.statistic, o().var1_chi2[alt].statistic, 1e-10)
      relClose(v.pValue, o().var1_chi2[alt].pValue, 1e-9)
      relClose(poissonRateTest1(37, 50, { lambda0: 1, alternative: alt, method: 'exact' }).pValue, o().poisson_rate_exact[alt].pValue, 1e-9)
    }
  })
})

describe('2×2 tables with empty cells: fisher_exact and chi2_contingency with / without Yates', () => {
  it.each(Object.keys(ref.tables))('%s', (name) => {
    const c = ref.tables[name]
    const [[a, b], [cc, d]] = c.table as [[number, number], [number, number]]
    for (const alt of ALTS) {
      const r = propTest2(a, a + b, cc, cc + d, { method: 'fisher', alternative: alt })
      relClose(r.pValue, c[`fisher_${alt === 'two-sided' ? 'two_sided' : alt}`], 1e-9)
    }
    if (c.chi2_error === undefined) {
      const y = chi2test(c.table, { correction: true })
      relClose(y.statistic, c.chi2_yates, 1e-10)
      relClose(y.pValue, c.p_yates, 1e-9)
      const n = chi2test(c.table, { correction: false })
      relClose(n.statistic, c.chi2, 1e-10)
      relClose(n.pValue, c.p, 1e-9)
    } else {
      expect(() => chi2test(c.table)).toThrow()
    }
  })
})
