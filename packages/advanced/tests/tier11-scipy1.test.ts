import { describe, expect, it } from 'vitest'
import {
  ansariBradley,
  bootstrap,
  bowker,
  brunnerMunzel,
  cochranQ,
  cohensD,
  corrTest,
  cramerVonMises,
  dagostinoK2,
  fligner,
  gaussianKde,
  glassDelta,
  gumbel,
  hedgesG,
  hypergeometric,
  invgauss,
  jarqueBera,
  mcnemar,
  negativeBinomial,
  normalityTest,
  partialCorr,
  permutationTest,
  dist,
} from '@columna/advanced'

describe('corrTest kendall + partialCorr', () => {
  it('kendall tau is finite and in [-1,1]', () => {
    const x = [1, 2, 3, 4, 5, 6, 7, 8]
    const y = [1, 3, 2, 5, 4, 7, 6, 8]
    const r = corrTest(x, y, { method: 'kendall' })
    expect(r.test).toMatch(/kendall/i)
    expect(Math.abs(r.estimate)).toBeLessThanOrEqual(1)
    expect(r.pValue).toBeGreaterThan(0)
    expect(r.pValue).toBeLessThanOrEqual(1)
  })

  it('partialCorr reduces association after controlling', () => {
    const n = 40
    const z = Array.from({ length: n }, (_, i) => i)
    const x = z.map((v) => v + (v % 3) * 0.1)
    const y = z.map((v) => v * 0.9 + (v % 5) * 0.05)
    const full = corrTest(x, y)
    const part = partialCorr(x, y, [z])
    expect(Math.abs(part.estimate)).toBeLessThan(Math.abs(full.estimate))
  })
})

describe('normality extras', () => {
  const x = Array.from({ length: 40 }, (_, i) => {
    const u = ((i * 17 + 3) % 40) / 40
    return Math.sqrt(-2 * Math.log(Math.max(1e-6, u))) * Math.cos(2 * Math.PI * (((i * 13) % 40) / 40))
  })

  it('jarqueBera / dagostino / cvm via helpers and normalityTest', () => {
    expect(Number.isFinite(jarqueBera(x).statistic)).toBe(true)
    expect(Number.isFinite(dagostinoK2(x).statistic)).toBe(true)
    expect(cramerVonMises(x).statistic).toBeGreaterThan(0)
    expect(normalityTest(x, 'jarque-bera').test).toBe('Jarque-Bera')
    expect(normalityTest(x, 'dagostino').test).toBe("D'Agostino K²")
    expect(normalityTest(x, 'cramer-von-mises').test).toBe('Cramer-von-Mises')
  })
})

describe('nonparametric scale + tables', () => {
  it('fligner / ansari / brunner', () => {
    const a = [1, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.03]
    const b = [-2, 8, 0, 12, -5, 15, 20, -10]
    expect(fligner([a, b]).statistic).toBeGreaterThan(0)
    expect(ansariBradley(a, b).statistic).toBeGreaterThan(0)
    expect(Number.isFinite(brunnerMunzel(a, b).statistic)).toBe(true)
  })

  it('mcnemar / cochranQ / bowker', () => {
    const m = mcnemar([
      [10, 8],
      [1, 20],
    ])
    expect(m.pValue).toBeLessThan(0.05)
    expect(m.statistic).toBeGreaterThanOrEqual(0)
    const q = cochranQ([
      [1, 0, 1],
      [1, 1, 0],
      [0, 0, 1],
      [1, 0, 0],
      [1, 1, 1],
      [0, 1, 0],
    ])
    expect(q.statistic).toBeGreaterThanOrEqual(0)
    const bw = bowker([
      [5, 2, 1],
      [0, 4, 3],
      [1, 0, 6],
    ])
    expect(bw.df).toBe(3)
  })
})

describe('effect sizes + kde + resample', () => {
  it('cohensD / hedgesG / glassDelta', () => {
    const a = [1, 2, 3, 4, 5]
    const b = [3, 4, 5, 6, 7]
    expect(cohensD(a, b).estimate).toBeCloseTo(-1.2649, 3)
    expect(Math.abs(hedgesG(a, b).estimate)).toBeLessThan(Math.abs(cohensD(a, b).estimate) + 0.2)
    expect(glassDelta(a, b).estimate).not.toBeNaN()
  })

  it('gaussianKde integrates roughly to 1', () => {
    const kde = gaussianKde([0, 0.5, 1, 1.5, 2, 2.5, 3])
    const g = kde.grid(200)
    let area = 0
    const dx = g.x[1]! - g.x[0]!
    for (const d of g.density) area += d * dx
    expect(area).toBeGreaterThan(0.85)
    expect(area).toBeLessThan(1.15)
  })

  it('bootstrap CI covers mean; permutation detects shift', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    const boot = bootstrap(data, (s) => {
      let m = 0
      for (let i = 0; i < s.length; i++) m += s[i]!
      return m / s.length
    }, { nBoot: 200, seed: 1 })
    expect(boot.ci[0]).toBeLessThan(5.5)
    expect(boot.ci[1]).toBeGreaterThan(5.5)
    const perm = permutationTest([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], { nPerm: 500, seed: 2 })
    expect(perm.pValue).toBeLessThan(0.05)
  })
})

describe('distributions wave1', () => {
  it('negativeBinomial / hypergeometric pmf sum ≈ 1', () => {
    const nb = negativeBinomial(5, 0.4)
    let s = 0
    for (let k = 0; k < 80; k++) s += nb.pmf(k)
    expect(s).toBeGreaterThan(0.99)
    expect(nb.mean).toBeCloseTo((5 * 0.6) / 0.4, 10)

    const hg = hypergeometric(20, 7, 5)
    let sh = 0
    for (let k = 0; k <= 5; k++) sh += hg.pmf(k)
    expect(sh).toBeCloseTo(1, 10)
  })

  it('gumbel / pareto / invgauss cdf monotonic', () => {
    const g = gumbel(0, 1)
    expect(g.cdf(0)).toBeCloseTo(Math.exp(-1), 8)
    expect(g.ppf(0.5)).toBeCloseTo(-Math.log(Math.log(2)), 6)

    const p = dist.pareto(3, 1)
    expect(p.cdf(1)).toBe(0)
    expect(p.sf(2)).toBeCloseTo((1 / 2) ** 3, 10)

    const ig = invgauss(1, 2)
    expect(ig.mean).toBe(1)
    expect(ig.cdf(1)).toBeGreaterThan(0.4)
    expect(ig.cdf(1)).toBeLessThan(0.7)
  })
})
