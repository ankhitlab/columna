import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import { anova, chi2gof, chi2test, crosstab, ttest1, ttest2, ttestPaired } from '@columna/advanced'

// Reference values from scipy 1.14 (scratch script tests_ref.py)
const ref = JSON.parse(readFileSync(new URL('./fixtures/tests-scipy.json', import.meta.url), 'utf8'))
const close = (got: number, want: number, digits = 12) => expect(got).toBeCloseTo(want, digits)

describe('t-tests', () => {
  it('one-sample matches scipy ttest_1samp incl. one-sided p and confidence intervals', () => {
    const r = ref.ttest1
    const t = ttest1(r.x, { mu: r.mu })
    expect(t.test).toBe('one-sample t')
    expect(t.n).toBe(10)
    expect(t.df).toBe(r.df)
    close(t.statistic, r.t)
    close(t.pValue, r.p)
    // CI bounds use t.ppf; scipy's own t.ppf(0.975, 9) is off by 2.5e-11 (mpmath: 2.262157162798205…, columna matches to 1e-15)
    close(t.ci[0], r.lo, 10)
    close(t.ci[1], r.hi, 10)
    close(ttest1(r.x, { mu: r.mu, alternative: 'less' }).pValue, r.p_less)
    close(ttest1(r.x, { mu: r.mu, alternative: 'greater' }).pValue, r.p_greater)
    close(ttest1(r.x, { mu: r.mu, confidence: 0.9 }).ci[0], r.ci90_lo, 10) // scipy's t.ppf is itself only ~1e-11 accurate
    const g = ttest1(r.x, { mu: r.mu, alternative: 'greater' })
    close(g.ci[0], r.ci_greater_lo, 10) // same t.ppf(0.95) as above
    expect(g.ci[1]).toBe(Infinity)
    // nulls / NaN are skipped
    close(ttest1([...r.x, null, NaN], { mu: r.mu }).statistic, r.t)
    expect(() => ttest1([1])).toThrow(RangeError)
  })

  it('two-sample Welch (default) and pooled match scipy ttest_ind', () => {
    const r = ref.ttest2
    const w = ttest2(r.a, r.b)
    expect(w.test).toBe('two-sample t (Welch)')
    close(w.statistic, r.welch_t)
    close(w.pValue, r.welch_p)
    close(w.df, r.welch_df)
    close(w.ci[0], r.welch_lo)
    close(w.ci[1], r.welch_hi)
    expect(w.samples.map((s) => s.n)).toEqual([10, 8])
    const p = ttest2(r.a, r.b, { equalVar: true })
    expect(p.test).toBe('two-sample t (pooled)')
    close(p.statistic, r.pooled_t)
    close(p.pValue, r.pooled_p)
    expect(p.df).toBe(r.pooled_df)
    close(p.ci[0], r.pooled_lo)
    close(p.ci[1], r.pooled_hi)
  })

  it('paired matches scipy ttest_rel', () => {
    const r = ref.paired
    const t = ttestPaired(r.after, r.before)
    expect(t.test).toBe('paired t')
    close(t.statistic, r.t)
    close(t.pValue, r.p)
    expect(t.df).toBe(r.df)
    close(t.ci[0], r.lo)
    close(t.ci[1], r.hi)
    expect(() => ttestPaired([1, 2], [1])).toThrow(RangeError)
  })
})

describe('ANOVA', () => {
  it('one-way matches scipy f_oneway and reports the table', () => {
    const r = ref.anova
    const a = anova(r.groups)
    close(a.statistic, r.F)
    close(a.pValue, r.p)
    expect(a.dfBetween).toBe(2)
    expect(a.dfWithin).toBe(21)
    expect(a.n).toBe(24)
    expect(a.groups.map((g) => g.name)).toEqual(['a', 'b', 'c'])
    close(a.msBetween / a.msWithin, a.statistic, 14)
    close(a.etaSquared, a.ssBetween / (a.ssBetween + a.ssWithin), 15)
    expect(() => anova({ a: [1, 2] })).toThrow(RangeError)
  })
})

describe('chi-square', () => {
  it('independence on 3×3 and 2×2 (with / without Yates) matches scipy chi2_contingency', () => {
    const r = ref.chi2
    const c = chi2test(r.table)
    close(c.statistic, r.stat)
    close(c.pValue, r.p)
    expect(c.df).toBe(r.df)
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) close((c.expected as number[][])[i]![j]!, r.expected[i][j])
    expect(c.lowExpected).toBe(false)
    const r2 = ref.chi2_2x2
    close(chi2test(r2.table).statistic, r2.stat_nocorr)
    close(chi2test(r2.table).pValue, r2.p_nocorr)
    const yates = chi2test(r2.table, { correction: true })
    close(yates.statistic, r2.stat_corr)
    close(yates.pValue, r2.p_corr)
    expect(yates.correction).toBe(true)
    expect(() => chi2test([[1, 2]])).toThrow(RangeError)
  })

  it('goodness of fit matches scipy chisquare (uniform, given expected, ddof)', () => {
    const r = ref.gof
    const g = chi2gof(r.obs)
    close(g.statistic, r.stat)
    close(g.pValue, r.p)
    expect(g.df).toBe(5)
    const g2 = chi2gof(r.obs, [16, 16, 16, 16, 16, 8], { ddof: 1 })
    close(g2.statistic, r.stat2)
    close(g2.pValue, r.p2)
    expect(g2.df).toBe(4)
    // probabilities are rescaled to the observed total
    close(chi2gof([50, 30, 20], [0.5, 0.3, 0.2]).statistic, 0, 15)
  })

  it('crosstab builds a sorted contingency table and skips nulls', () => {
    const ct = crosstab(['x', 'y', 'x', null, 'y', 'x'], ['p', 'p', 'q', 'q', 'q', 'p'])
    expect(ct.rows).toEqual(['x', 'y'])
    expect(ct.cols).toEqual(['p', 'q'])
    expect(ct.table).toEqual([
      [2, 1],
      [1, 1],
    ])
  })
})

describe('DataFrame / LazyFrame test methods', () => {
  const ra = ref.ttest2.a as number[]
  const rb = ref.ttest2.b as number[]
  const df = DataFrame.fromRows([
    ...ra.map((x) => ({ x, g: 'A', k: x > 5.5 ? 'hi' : 'lo' })),
    ...rb.map((x) => ({ x, g: 'B', k: x > 5.0 ? 'hi' : 'lo' })),
  ])

  it('ttest: one-sample, two-sample by group, paired', async () => {
    const one = DataFrame.fromRows(ra.map((x) => ({ x }))).ttest('x', { mu: 5 })
    close(one.statistic, ref.ttest1.t)
    const two = df.ttest('x', { by: 'g' })
    close(two.statistic, ref.ttest2.welch_t)
    close(two.pValue, ref.ttest2.welch_p)
    close(df.ttest('x', { by: 'g', equalVar: true }).statistic, ref.ttest2.pooled_t)
    const lazy = await df.lazy().ttest('x', { by: 'g' })
    close(lazy.statistic, ref.ttest2.welch_t)
    const paired = DataFrame.fromRows(ref.paired.after.map((a: number, i: number) => ({ after: a, before: ref.paired.before[i] })))
    close(paired.ttest('after', { paired: 'before' }).statistic, ref.paired.t)
    expect(() => df.ttest('x', { by: 'k', paired: 'x' })).toThrow(RangeError)
    expect(() =>
      DataFrame.fromRows([
        { x: 1, g: 'a' },
        { x: 2, g: 'b' },
        { x: 3, g: 'c' },
      ]).ttest('x', { by: 'g' }),
    ).toThrow(/2 levels/)
  })

  it('anova by column and chi2test on two categorical columns', async () => {
    const three = DataFrame.fromRows([
      ...ra.map((x) => ({ x, g: 'a' })),
      ...rb.map((x) => ({ x, g: 'b' })),
      ...(ref.anova.groups.c as number[]).map((x) => ({ x, g: 'c' })),
    ])
    const a = three.anova('x', 'g')
    close(a.statistic, ref.anova.F)
    close(a.pValue, ref.anova.p)
    close((await three.lazy().anova('x', 'g')).statistic, ref.anova.F)

    const c = df.chi2test('g', 'k')
    expect(c.rows).toEqual(['A', 'B'])
    expect(c.cols).toEqual(['hi', 'lo'])
    const ct = crosstab(df.getColumn('g').toArray(), df.getColumn('k').toArray())
    close(c.statistic, chi2test(ct.table).statistic, 15)
    expect(c.df).toBe(1)
  })
})
