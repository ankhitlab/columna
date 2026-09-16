import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import { anova, power, propTest1, tost1, tost2, tostPaired, ttest1, ttest2, varTest1 } from '@columna/advanced'

// Reference values from scipy 1.14 (tests/refs/tier2_ref.py)
const ref = JSON.parse(readFileSync(new URL('./fixtures/tier2-scipy.json', import.meta.url), 'utf8'))
const close = (got: number, want: number, digits = 10) => expect(got).toBeCloseTo(want, digits)

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
  const binom = (n: number, p: number) => {
    let k = 0
    for (let i = 0; i < n; i++) if (u() < p) k++
    return k
  }
  return { u, normal, binom }
}
const sample = (n: number, g: () => number) => Array.from({ length: n }, g)

describe('2.9 equivalence tests (TOST)', () => {
  it('one-sample: t statistics, one-sided p-values and the 90 % CI match the closed form', () => {
    const c = ref.tost1
    const r = tost1(c.x, { limits: c.limits })
    close(r.estimate, c.mean, 12)
    close(r.se, c.se, 12)
    close(r.tLower, c.tLower, 10)
    close(r.tUpper, c.tUpper, 10)
    close(r.pLower, c.pLower, 10)
    close(r.pUpper, c.pUpper, 10)
    close(r.pValue, Math.max(c.pLower, c.pUpper), 12)
    close(r.ci[0], c.ci90[0], 10)
    close(r.ci[1], c.ci90[1], 10)
    expect(r.df).toBe(c.x.length - 1)
    expect(r.equivalent).toBe(r.pValue < 0.05)
    // equivalence ⇔ the 100(1 − 2α)% CI lies inside the limits
    expect(r.equivalent).toBe(r.ci[0] > c.limits[0] && r.ci[1] < c.limits[1])
    expect(() => tost1(c.x, { limits: [1, 0] })).toThrow(RangeError)
    expect(() => tost1(c.x, { limits: [-1, 1], alpha: 0.6 })).toThrow(RangeError)
    expect(() => tost1([1], { limits: [-1, 1] })).toThrow(RangeError)
  })

  it('two-sample (Welch) and paired variants', () => {
    const c = ref.tost2
    const r = tost2(c.a, c.b, { limits: c.limits })
    close(r.estimate, c.diff, 12)
    close(r.se, c.se, 12)
    close(r.df, c.df, 8)
    close(r.pLower, c.pLower, 10)
    close(r.pUpper, c.pUpper, 10)
    expect(r.test).toBe('equivalence (two-sample, Welch)')
    const pooled = tost2(c.a, c.b, { limits: c.limits, equalVar: true })
    expect(pooled.df).toBe(c.a.length + c.b.length - 2)
    expect(pooled.test).toBe('equivalence (two-sample, pooled)')
    // paired reduces to one-sample on the differences
    const p = tostPaired(c.a.slice(0, 10), c.b, { limits: c.limits })
    const d = c.a.slice(0, 10).map((v: number, i: number) => v - c.b[i])
    const one = tost1(d, { limits: c.limits })
    close(p.pValue, one.pValue, 12)
    expect(p.test).toBe('equivalence (paired)')
    expect(() => tostPaired([1, 2], [1], { limits: [-1, 1] })).toThrow(RangeError)
  })

  it('size at the boundary ≈ α, power inside the limits, never claims equivalence when the CI crosses a limit (Monte-Carlo, seeded)', () => {
    const g = rng(1212)
    const reps = 3000
    const n = 30
    let rejBoundary = 0
    let rejInside = 0
    let consistent = 0
    for (let i = 0; i < reps; i++) {
      const x = sample(n, () => 0.5 + g.normal()) // true mean exactly at the upper limit
      const r = tost1(x, { limits: [-0.5, 0.5] })
      if (r.equivalent) rejBoundary++
      if (r.equivalent === (r.ci[0] > -0.5 && r.ci[1] < 0.5)) consistent++
      const y = sample(n, () => 0.5 * g.normal())
      if (tost1(y, { limits: [-0.5, 0.5] }).equivalent) rejInside++
    }
    expect(rejBoundary / reps).toBeGreaterThan(0.035)
    expect(rejBoundary / reps).toBeLessThan(0.065)
    expect(rejInside / reps).toBeGreaterThan(0.95)
    expect(consistent).toBe(reps)
  })
})

describe('2.10 power and sample size', () => {
  it('t / z / ANOVA / variance powers match the noncentral formulas (scipy nct, ncf, chi2, f)', () => {
    for (const c of ref.power.t1) close(power({ test: '1-sample t', effect: c.d, n: c.n }).power, c.power, 7)
    for (const c of ref.power.t1_greater) close(power({ test: '1-sample t', effect: c.d, n: c.n, alternative: 'greater' }).power, c.power, 7)
    for (const c of ref.power.t1) close(power({ test: 'paired t', effect: c.d, n: c.n }).power, c.power, 7)
    for (const c of ref.power.t2) close(power({ test: '2-sample t', effect: c.d, n: c.n }).power, c.power, 7)
    for (const c of ref.power.anova) close(power({ test: 'one-way anova', effect: c.maxdiff, n: c.n, groups: c.k }).power, c.power, 6)
    for (const c of ref.power.z1) close(power({ test: '1-sample z', effect: c.d, n: c.n }).power, c.power, 10)
    for (const c of ref.power.var1) close(power({ test: '1 variance', effect: c.ratio, n: c.n }).power, c.power, 8)
    for (const c of ref.power.var2) close(power({ test: '2 variances', effect: c.ratio, n: c.n }).power, c.power, 8)
    // sigma scales the effect
    close(power({ test: '1-sample t', effect: 2, n: 10, sigma: 2 }).power, power({ test: '1-sample t', effect: 1, n: 10 }).power, 12)
    // classic table values: d = 0.5, power 0.8 → n = 34 (1-sample t), n = 64 per group (2-sample t)
    expect(power({ test: '1-sample t', effect: 0.5, power: 0.8 }).n).toBe(34)
    expect(power({ test: '2-sample t', effect: 0.5, power: 0.8 }).n).toBe(64)
    expect(power({ test: '1-sample z', effect: 0.5, power: 0.8 }).n).toBe(32)
  })

  it('solving for n gives the smallest n reaching the target; solving for effect inverts power()', () => {
    for (const test of ['1-sample t', '2-sample t', 'paired t', '1-sample z', 'one-way anova', '1 variance', '2 variances'] as const) {
      const extra = test === 'one-way anova' ? { groups: 4 } : {}
      const effect = test === '1 variance' || test === '2 variances' ? 1.6 : 0.6
      const r = power({ test, effect, power: 0.85, ...extra })
      expect(r.solvedFor).toBe('n')
      expect(r.power).toBeGreaterThanOrEqual(0.85)
      expect(power({ test, effect, n: r.n - 1, ...extra }).power).toBeLessThan(0.85)
      const e = power({ test, n: r.n, power: r.power, ...extra })
      expect(e.solvedFor).toBe('effect')
      close(e.effect, effect, 6)
    }
    for (const test of ['1 proportion', '2 proportions'] as const) {
      const extra = test === '1 proportion' ? { p0: 0.3 } : { p1: 0.3 }
      const r = power({ test, effect: 0.5, power: 0.9, ...extra })
      expect(r.power).toBeGreaterThanOrEqual(0.9)
      expect(power({ test, effect: 0.5, n: r.n - 1, ...extra }).power).toBeLessThan(0.9)
      close(power({ test, n: r.n, power: r.power, ...extra }).effect, 0.5, 6)
      // 'less' alternative solves below the null
      const l = power({ test, n: 100, power: 0.8, alternative: 'less', ...extra })
      expect(l.effect).toBeLessThan(0.3)
    }
    expect(() => power({ test: '1-sample t', effect: 1 })).toThrow(RangeError)
    expect(() => power({ test: '1-sample t', effect: 1, n: 10, power: 0.5 })).toThrow(RangeError)
    expect(() => power({ test: '1-sample t', effect: 1, n: 2.5 })).toThrow(RangeError)
    expect(() => power({ test: 'one-way anova', effect: 1, n: 10 })).toThrow(RangeError)
    expect(() => power({ test: '1 proportion', effect: 0.5, n: 10 })).toThrow(RangeError)
    expect(() => power({ test: '1-sample t', effect: 0.001, power: 0.99, maxN: 1000 })).toThrow(RangeError)
  })

  it('computed power matches the empirical rejection rate of the actual tests (Monte-Carlo, seeded, ±0.025)', () => {
    const g = rng(1313)
    const reps = 4000
    const check = (computed: number, reject: () => boolean) => {
      let k = 0
      for (let i = 0; i < reps; i++) if (reject()) k++
      expect(Math.abs(k / reps - computed)).toBeLessThan(0.025)
    }
    // 1-sample t: d = 0.5, n = 30
    check(power({ test: '1-sample t', effect: 0.5, n: 30 }).power, () => ttest1(sample(30, () => 0.5 + g.normal())).pValue < 0.05)
    // 2-sample t (pooled): d = 1, n = 17
    check(power({ test: '2-sample t', effect: 1, n: 17 }).power, () => ttest2(sample(17, () => 1 + g.normal()), sample(17, g.normal), { equalVar: true }).pValue < 0.05)
    // one-way ANOVA: 3 groups of 10, max difference 1 (means 0, 0.5, 1)
    check(power({ test: 'one-way anova', effect: 1, n: 10, groups: 3 }).power, () => anova({ a: sample(10, g.normal), b: sample(10, () => 0.5 + g.normal()), c: sample(10, () => 1 + g.normal()) }).pValue < 0.05)
    // 1 variance: ratio 1.5, n = 30 (χ² test)
    check(power({ test: '1 variance', effect: 1.5, n: 30 }).power, () => varTest1(sample(30, () => 1.5 * g.normal()), { sigma0: 1 }).pValue < 0.05)
    // 1 proportion (normal approximation on both sides): p0 = 0.5, p = 0.65, n = 80
    check(power({ test: '1 proportion', effect: 0.65, n: 80, p0: 0.5 }).power, () => propTest1(g.binom(80, 0.65), 80, { p0: 0.5, method: 'normal' }).pValue < 0.05)
  })
})

describe('DataFrame methods (Tier 2 equivalence)', () => {
  it('equivalence: one-sample, two-sample by group, paired', async () => {
    const c = ref.tost2
    const df = DataFrame.fromColumns({
      y: [...c.a, ...c.b],
      g: [...c.a.map(() => 'test'), ...c.b.map(() => 'ref')],
    })
    const two = df.equivalence('y', { limits: c.limits, by: 'g', reference: 'ref' })
    close(two.estimate, c.diff, 12)
    close(two.pLower, c.pLower, 10)
    // default reference = last sorted level ('test'), so the sign flips
    close(df.equivalence('y', { limits: c.limits, by: 'g' }).estimate, -c.diff, 12)
    expect(() => df.equivalence('y', { limits: c.limits, by: 'g', reference: 'nope' })).toThrow(RangeError)
    const one = DataFrame.fromColumns({ x: ref.tost1.x })
    close(one.equivalence('x', { limits: ref.tost1.limits }).pValue, tost1(ref.tost1.x, { limits: ref.tost1.limits }).pValue, 12)
    const paired = DataFrame.fromColumns({ a: c.a.slice(0, 10), b: c.b })
    close(paired.equivalence('a', { limits: c.limits, paired: 'b' }).pValue, tostPaired(c.a.slice(0, 10), c.b, { limits: c.limits }).pValue, 12)
    expect(() => paired.equivalence('a', { limits: c.limits, paired: 'b', by: 'a' })).toThrow(RangeError)
    close((await one.lazy().equivalence('x', { limits: ref.tost1.limits })).estimate, ref.tost1.mean, 12)
  })
})
