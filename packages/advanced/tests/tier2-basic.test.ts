import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import {
  binomial,
  corrTest,
  dixon,
  grubbs,
  hypergeomPmf,
  ncChi2Cdf,
  ncfCdf,
  nctCdf,
  poisson,
  poissonRateTest1,
  poissonRateTest2,
  propTest1,
  propTest2,
  varTest1,
  ztest1,
} from '@columna/advanced'

// Reference values from scipy 1.14 (tests/refs/tier2_ref.py)
const ref = JSON.parse(readFileSync(new URL('./fixtures/tier2-scipy.json', import.meta.url), 'utf8'))
const close = (got: number, want: number, digits = 10) => expect(got).toBeCloseTo(want, digits)
const alts = ['two-sided', 'less', 'greater'] as const

// Seeded generators: mulberry32 → uniform, Box–Muller → normal, t(3), Bernoulli sums, Poisson (Knuth)
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
  const t3 = () => normal() / Math.sqrt((normal() ** 2 + normal() ** 2 + normal() ** 2) / 3)
  const binom = (n: number, p: number) => {
    let k = 0
    for (let i = 0; i < n; i++) if (u() < p) k++
    return k
  }
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
  return { u, normal, t3, binom, pois }
}
const sample = (n: number, g: () => number) => Array.from({ length: n }, g)

describe('discrete and noncentral distributions', () => {
  it('binomial / poisson pmf, cdf, ppf match scipy', () => {
    for (const c of ref.binom) {
      const d = binomial(c.n, c.p)
      close(d.pmf(c.k), c.pmf, 12)
      close(d.cdf(c.k), c.cdf, 12)
      expect(d.ppf(c.q)).toBe(c.ppf)
    }
    for (const c of ref.poisson) {
      const d = poisson(c.lam)
      close(d.pmf(c.k), c.pmf, 12)
      close(d.cdf(c.k), c.cdf, 12)
      expect(d.ppf(c.q)).toBe(c.ppf)
    }
    for (const c of ref.hypergeom) close(hypergeomPmf(c.k, c.N, c.K, c.n), c.pmf, 12)
    // pmf sums to 1, mean/variance
    const b = binomial(12, 0.35)
    let s = 0
    for (let k = 0; k <= 12; k++) s += b.pmf(k)
    close(s, 1, 12)
    close(b.mean, 4.2, 12)
    close(b.variance, 12 * 0.35 * 0.65, 12)
  })

  it('noncentral t / F / χ² cdf match scipy nct, ncf, ncx2', () => {
    for (const c of ref.nct) close(nctCdf(c.x, c.df, c.nc), c.cdf, 7)
    for (const c of ref.ncf) close(ncfCdf(c.x, c.d1, c.d2, c.nc), c.cdf, 7)
    for (const c of ref.ncx2) close(ncChi2Cdf(c.x, c.k, c.nc), c.cdf, 7)
  })
})

describe('2.1 one-sample z', () => {
  it('matches the closed form (statistic, p, CI)', () => {
    const r = ref.ztest
    for (const alt of alts) {
      const z = ztest1(r.x, { sigma: r.sigma, mu: r.mu, alternative: alt })
      close(z.statistic, r.z)
      close(z.pValue, r.p[alt])
      if (alt === 'two-sided') {
        close(z.ci[0], r.ci95[0])
        close(z.ci[1], r.ci95[1])
      }
    }
    expect(() => ztest1([1, 2], { sigma: 0 })).toThrow(RangeError)
    expect(() => ztest1([], { sigma: 1 })).toThrow(RangeError)
  })

  it('keeps its size and CI coverage (Monte-Carlo, seeded)', () => {
    const g = rng(101)
    const reps = 4000
    let rej = 0
    let cover = 0
    for (let i = 0; i < reps; i++) {
      const x = sample(15, () => 3 + 2 * g.normal())
      const r = ztest1(x, { sigma: 2, mu: 3 })
      if (r.pValue < 0.05) rej++
      if (r.ci[0] <= 3 && 3 <= r.ci[1]) cover++
    }
    expect(rej / reps).toBeGreaterThan(0.035)
    expect(rej / reps).toBeLessThan(0.065)
    expect(cover / reps).toBeGreaterThan(0.935)
    expect(cover / reps).toBeLessThan(0.965)
  })
})

describe('2.2 proportions', () => {
  it('1 proportion: exact p and Clopper–Pearson CI match scipy binomtest', () => {
    for (const c of ref.prop1) {
      for (const alt of alts) {
        const r = propTest1(c.k, c.n, { p0: c.p0, alternative: alt })
        close(r.pValue, c.exact[alt].p, 10)
        close(r.ci[0], c.exact[alt].ci[0], 8)
        close(r.ci[1], c.exact[alt].ci[1], 8)
        expect(r.method).toBe('exact')
      }
    }
    // normal approximation: z = (p̂ − p0)/√(p0 q0/n)
    const r = propTest1(30, 100, { p0: 0.25, method: 'normal' })
    close(r.statistic!, 0.05 / Math.sqrt(0.1875 / 100))
    expect(() => propTest1(5, 4)).toThrow(RangeError)
    expect(() => propTest1(1, 4, { p0: 1 })).toThrow(RangeError)
  })

  it('2 proportions: pooled z, unpooled CI and Fisher exact match scipy', () => {
    for (const c of ref.prop2) {
      for (const alt of alts) {
        const r = propTest2(c.e1, c.n1, c.e2, c.n2, { alternative: alt })
        close(r.statistic!, c.z)
        close(r.pValue, c.p[alt])
        const f = propTest2(c.e1, c.n1, c.e2, c.n2, { alternative: alt, method: 'fisher' })
        close(f.pValue, c.fisher[alt], 10)
        expect(f.method).toBe('fisher')
      }
      const r = propTest2(c.e1, c.n1, c.e2, c.n2)
      close(r.ci[0], c.ci95[0])
      close(r.ci[1], c.ci95[1])
    }
  })

  it('exact tests are conservative, the pooled z holds its size, Clopper–Pearson covers ≥ 95% (Monte-Carlo, seeded)', () => {
    const g = rng(202)
    const reps = 4000
    let rejExact = 0
    let rejZ = 0
    let rejFisher = 0
    let cover = 0
    for (let i = 0; i < reps; i++) {
      const k = g.binom(25, 0.3)
      const e = propTest1(k, 25, { p0: 0.3 })
      if (e.pValue < 0.05) rejExact++
      if (e.ci[0] <= 0.3 && 0.3 <= e.ci[1]) cover++
      const k1 = g.binom(40, 0.4)
      const k2 = g.binom(35, 0.4)
      if (propTest2(k1, 40, k2, 35).pValue < 0.05) rejZ++
      if (propTest2(k1, 40, k2, 35, { method: 'fisher' }).pValue < 0.05) rejFisher++
    }
    expect(rejExact / reps).toBeLessThanOrEqual(0.055)
    expect(rejFisher / reps).toBeLessThanOrEqual(0.055)
    expect(rejZ / reps).toBeGreaterThan(0.03)
    expect(rejZ / reps).toBeLessThan(0.07)
    expect(cover / reps).toBeGreaterThanOrEqual(0.945)
    // power: p = 0.6 vs p0 = 0.3, n = 25 — exact binomial power ≈ 0.85
    let pw = 0
    for (let i = 0; i < 2000; i++) if (propTest1(g.binom(25, 0.6), 25, { p0: 0.3 }).pValue < 0.05) pw++
    expect(pw / 2000).toBeGreaterThan(0.75)
  })
})

describe('2.3 Poisson rates', () => {
  it('1-sample: exact p and Garwood CI match scipy poisson / chi2', () => {
    for (const c of ref.rate1) {
      for (const alt of alts) close(poissonRateTest1(c.k, c.t, { lambda0: c.lambda0, alternative: alt }).pValue, c.p[alt], 10)
      const r = poissonRateTest1(c.k, c.t, { lambda0: c.lambda0 })
      close(r.estimate, c.k / c.t)
      close(r.ci[0], c.ci95[0], 8)
      close(r.ci[1], c.ci95[1], 8)
    }
    expect(() => poissonRateTest1(2.5, 1)).toThrow(RangeError)
    expect(() => poissonRateTest1(2, 0)).toThrow(RangeError)
  })

  it('2-sample: exact conditional binomial and pooled normal match the reference', () => {
    for (const c of ref.rate2) {
      for (const alt of alts) close(poissonRateTest2(c.k1, c.t1, c.k2, c.t2, { alternative: alt }).pValue, c.exact[alt], 10)
      const r = poissonRateTest2(c.k1, c.t1, c.k2, c.t2, { method: 'normal' })
      close(r.statistic!, c.z)
      close(r.pValue, c.pNormal)
      close(r.estimate, c.k1 / c.t1 - c.k2 / c.t2)
    }
  })

  it('exact tests are conservative and the Garwood CI covers ≥ 95% (Monte-Carlo, seeded)', () => {
    const g = rng(303)
    const reps = 4000
    let rej1 = 0
    let rej2 = 0
    let cover = 0
    for (let i = 0; i < reps; i++) {
      const k = g.pois(6)
      const r = poissonRateTest1(k, 4, { lambda0: 1.5 })
      if (r.pValue < 0.05) rej1++
      if (r.ci[0] <= 1.5 && 1.5 <= r.ci[1]) cover++
      if (poissonRateTest2(g.pois(8), 4, g.pois(12), 6).pValue < 0.05) rej2++
    }
    expect(rej1 / reps).toBeLessThanOrEqual(0.055)
    expect(rej2 / reps).toBeLessThanOrEqual(0.055)
    expect(cover / reps).toBeGreaterThanOrEqual(0.945)
    // power: λ = 4 vs λ0 = 1.5 over 4 units (exact power ≈ 0.86)
    let pw = 0
    for (let i = 0; i < 2000; i++) if (poissonRateTest1(g.pois(16), 4, { lambda0: 1.5 }).pValue < 0.05) pw++
    expect(pw / 2000).toBeGreaterThan(0.75)
  })
})

describe('2.4 one-sample variance', () => {
  it('χ² statistic, p and CI match the closed form', () => {
    const c = ref.var1
    for (const alt of alts) {
      const r = varTest1(c.x, { sigma0: c.sigma0, alternative: alt })
      close(r.statistic!, c.stat)
      close(r.pValue, c.p[alt])
      close(r.estimate, c.s2)
      expect(r.df).toBe(c.x.length - 1)
    }
    const r = varTest1(c.x, { sigma0: c.sigma0 })
    close(r.ci[0], c.ci95[0])
    close(r.ci[1], c.ci95[1])
    expect(() => varTest1(c.x, { sigma0: -1 })).toThrow(RangeError)
    expect(() => varTest1([1], { sigma0: 1 })).toThrow(RangeError)
  })

  it('Bonett: p is the smallest α whose CI excludes σ₀², CI shrinks with confidence', () => {
    const c = ref.var1
    const b = varTest1(c.x, { sigma0: 0.5, method: 'bonett' })
    expect(b.method).toBe('bonett')
    expect(b.kurtosis).toBeGreaterThan(1)
    const atP = varTest1(c.x, { sigma0: 0.5, method: 'bonett', confidence: 1 - b.pValue })
    close(Math.min(Math.abs(atP.ci[0] - 0.25), Math.abs(atP.ci[1] - 0.25)), 0, 6)
    const wide = varTest1(c.x, { sigma0: 0.5, method: 'bonett', confidence: 0.99 })
    expect(wide.ci[0]).toBeLessThan(b.ci[0])
    expect(wide.ci[1]).toBeGreaterThan(b.ci[1])
    // one-sided variants
    const less = varTest1(c.x, { sigma0: 0.5, method: 'bonett', alternative: 'less' })
    expect(less.ci[0]).toBe(0)
    const greater = varTest1(c.x, { sigma0: 0.5, method: 'bonett', alternative: 'greater' })
    expect(greater.ci[1]).toBe(Infinity)
    // the α-dependent c = n/(n − z) keeps the one-sided p-values from summing exactly to 1
    expect(less.pValue + greater.pValue).toBeGreaterThan(0.85)
    expect(less.pValue + greater.pValue).toBeLessThan(1.15)
  })

  it('χ² holds its size on normal data but not on Laplace; Bonett stays near α on both (Monte-Carlo, seeded)', () => {
    // Laplace (kurtosis 6, variance 2b²) — t₃ has no finite fourth moment, so no variance test can be calibrated on it
    const g = rng(404)
    const reps = 3000
    const n = 40
    const laplace = () => (g.u() < 0.5 ? -1 : 1) * -Math.log(g.u())
    const count = { chiN: 0, chiT: 0, bonN: 0, bonT: 0, cover: 0 }
    for (let i = 0; i < reps; i++) {
      const xn = sample(n, () => 2 * g.normal())
      const xt = sample(n, laplace)
      const cn = varTest1(xn, { sigma0: 2 })
      if (cn.pValue < 0.05) count.chiN++
      if (cn.ci[0] <= 4 && 4 <= cn.ci[1]) count.cover++
      if (varTest1(xt, { sigma0: Math.SQRT2 }).pValue < 0.05) count.chiT++
      if (varTest1(xn, { sigma0: 2, method: 'bonett' }).pValue < 0.05) count.bonN++
      if (varTest1(xt, { sigma0: Math.SQRT2, method: 'bonett' }).pValue < 0.05) count.bonT++
    }
    expect(count.chiN / reps).toBeGreaterThan(0.035)
    expect(count.chiN / reps).toBeLessThan(0.065)
    expect(count.cover / reps).toBeGreaterThan(0.935)
    expect(count.cover / reps).toBeLessThan(0.965)
    expect(count.chiT / reps).toBeGreaterThan(0.14) // χ² breaks down with heavy tails (≈ 0.19 here)
    expect(count.bonN / reps).toBeGreaterThan(0.025)
    expect(count.bonN / reps).toBeLessThan(0.075)
    expect(count.bonT / reps).toBeLessThan(0.1) // ≈ 0.075 at n = 40 (kurtosis-estimate bias), → 0.05 as n grows
  })
})

describe('2.5 correlation with inference', () => {
  it('Pearson r, p, Fisher-z CI and Spearman match scipy pearsonr / spearmanr', () => {
    const c = ref.corr
    const p = corrTest(c.a, c.b)
    close(p.estimate, c.pearson.r)
    close(p.pValue, c.pearson.p)
    close(p.ci[0], c.pearson.ci95[0], 8)
    close(p.ci[1], c.pearson.ci95[1], 8)
    close(corrTest(c.a, c.b, { alternative: 'less' }).pValue, c.pearson.less)
    close(corrTest(c.a, c.b, { alternative: 'greater' }).pValue, c.pearson.greater)
    const s = corrTest(c.a, c.b, { method: 'spearman' })
    close(s.estimate, c.spearman.r)
    close(s.pValue, c.spearman.p)
    const t = corrTest(ref.corr_ties.a, ref.corr_ties.b, { method: 'spearman' })
    close(t.estimate, ref.corr_ties.r)
    close(t.pValue, ref.corr_ties.p)
    expect(() => corrTest([1, 2], [1, 2])).toThrow(RangeError)
    expect(() => corrTest([1, 2, 3], [1, 2])).toThrow(RangeError)
    // nulls are dropped pairwise
    const d = corrTest([1, null, 2, 3, 4], [2, 5, 4, 6, 8])
    expect(d.n).toBe(4)
  })

  it('size under independence and Fisher-z coverage at ρ = 0.5 (Monte-Carlo, seeded)', () => {
    const g = rng(505)
    const reps = 3000
    const n = 30
    let rejP = 0
    let rejS = 0
    let cover = 0
    const rho = 0.5
    for (let i = 0; i < reps; i++) {
      const x = sample(n, g.normal)
      const y = sample(n, g.normal)
      if (corrTest(x, y).pValue < 0.05) rejP++
      if (corrTest(x, y, { method: 'spearman' }).pValue < 0.05) rejS++
      const yc = x.map((v, j) => rho * v + Math.sqrt(1 - rho * rho) * y[j]!)
      const r = corrTest(x, yc)
      if (r.ci[0] <= rho && rho <= r.ci[1]) cover++
    }
    expect(rejP / reps).toBeGreaterThan(0.035)
    expect(rejP / reps).toBeLessThan(0.065)
    expect(rejS / reps).toBeGreaterThan(0.03)
    expect(rejS / reps).toBeLessThan(0.07)
    expect(cover / reps).toBeGreaterThan(0.93)
    expect(cover / reps).toBeLessThan(0.97)
  })
})

describe('2.6 outlier tests', () => {
  it('Grubbs matches the closed form on the NIST example', () => {
    const c = ref.grubbs
    const r = grubbs(c.x)
    close(r.statistic, c.G)
    close(r.critical, c.crit, 8)
    close(r.pValue, c.p, 8)
    expect(r.significant).toBe(true)
    expect(r.outlier.value).toBe(245.57)
    expect(r.outlier.index).toBe(7)
    const mn = grubbs(c.x, { alternative: 'min' })
    expect(mn.outlier.value).toBe(199.31)
    expect(mn.significant).toBe(false)
    expect(() => grubbs([1, 2])).toThrow(RangeError)
    expect(() => grubbs([3, 3, 3, 3])).toThrow(RangeError)
  })

  it('Dixon critical values agree with the Dean–Dixon / Rorabacher tables', () => {
    // two-sided 95 % (Rorabacher 1991): r10 n = 3..7, r11 n = 8..10, r21 n = 11..13, r22 n = 14..
    const twoSided: Record<number, number> = { 3: 0.97, 4: 0.829, 5: 0.71, 6: 0.625, 7: 0.568, 8: 0.615, 9: 0.57, 10: 0.534, 11: 0.625, 12: 0.592, 13: 0.565, 14: 0.59, 15: 0.568, 20: 0.491, 30: 0.42 }
    for (const [n, crit] of Object.entries(twoSided)) {
      const r = dixon(Array.from({ length: Number(n) }, (_, i) => i * 1.0 + Math.sin(i)), {})
      expect(Math.abs(r.critical - crit)).toBeLessThan(0.02)
    }
    // one-sided 95 % (Dean & Dixon 1951 r10): n = 3..7
    const oneSided: Record<number, number> = { 3: 0.941, 4: 0.765, 5: 0.642, 6: 0.56, 7: 0.507 }
    for (const [n, crit] of Object.entries(oneSided)) {
      const r = dixon(Array.from({ length: Number(n) }, (_, i) => i + Math.cos(i)), { alternative: 'max' })
      expect(Math.abs(r.critical - crit)).toBeLessThan(0.02)
    }
    // ratio definition r10 = (x₂ − x₁)/(xₙ − x₁) for n ≤ 7
    const r = dixon([1, 2, 3, 10], { alternative: 'max' })
    close(r.statistic, 7 / 9)
    expect(r.outlier.value).toBe(10)
    expect(r.significant).toBe(true)
    expect(() => dixon([1, 2])).toThrow(RangeError)
    expect(() => dixon(Array.from({ length: 31 }, (_, i) => i))).toThrow(RangeError)
  })

  it('both tests hold their size on normal data and detect a 4σ outlier (Monte-Carlo, seeded)', () => {
    const g = rng(606)
    const reps = 3000
    let rejG = 0
    let rejD = 0
    let powG = 0
    let powD = 0
    for (let i = 0; i < reps; i++) {
      const x = sample(12, g.normal)
      if (grubbs(x).significant) rejG++
      if (dixon(x).significant) rejD++
      const y = x.slice()
      y[3] = 5
      if (grubbs(y).significant) powG++
      if (dixon(y).significant) powD++
    }
    expect(rejG / reps).toBeGreaterThan(0.03)
    expect(rejG / reps).toBeLessThan(0.065)
    expect(rejD / reps).toBeGreaterThan(0.035)
    expect(rejD / reps).toBeLessThan(0.065)
    expect(powG / reps).toBeGreaterThan(0.85) // ≈ 0.93
    expect(powD / reps).toBeGreaterThan(0.7) // ≈ 0.79 (Dixon r21 uses x₃ − x₁, less sensitive than Grubbs)
  })
})

describe('DataFrame methods (Tier 2 basic)', () => {
  const df = DataFrame.fromColumns({
    x: ref.corr.a,
    y: ref.corr.b,
    g: ref.corr.a.map((_: number, i: number) => (i % 2 ? 'a' : 'b')),
  })
  it('ztest / varTest / corrTest / outlierTest delegate to the functions', async () => {
    close(df.ztest('x', { sigma: 3, mu: 6 }).pValue, ztest1(ref.corr.a, { sigma: 3, mu: 6 }).pValue)
    close(df.varTest('x', { sigma0: 3 }).pValue, varTest1(ref.corr.a, { sigma0: 3 }).pValue)
    close(df.corrTest('x', 'y').estimate, ref.corr.pearson.r)
    close(df.outlierTest('x').statistic, grubbs(ref.corr.a).statistic)
    expect(df.outlierTest('x', { method: 'dixon' }).test).toBe('Dixon')
    const lazy = await df.lazy().corrTest('x', 'y', { method: 'spearman' })
    close(lazy.estimate, ref.corr.spearman.r)
  })
})
