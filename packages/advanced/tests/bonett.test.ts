import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import { bartlett, bonett, bonett2, equalVariances } from '@columna/advanced'

// Seeded generators: mulberry32 → uniform, Box–Muller → normal, t(3) = N / √(χ²₃/3)
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
  return { u, normal, t3 }
}

describe("Bonett's test for equal variances", () => {
  it('hand check: k = 2 reduces to Layard-type z² with kurtosis-adjusted weights', () => {
    const a = [5.1, 4.9, 6.2, 5.8, 6.0, 5.5, 5.3, 6.4, 5.9, 5.0]
    const b = [4.8, 5.2, 5.0, 4.6, 5.1, 4.9, 5.3, 4.7]
    const r = bonett({ a, b })
    expect(r.df).toBe(1)
    expect(r.kurtosis).toBeGreaterThan(1)
    // recompute with the documented formula
    const va = r.groups[0]!.variance
    const vb = r.groups[1]!.variance
    const wa = 9 / (r.kurtosis - 7 / 10)
    const wb = 7 / (r.kurtosis - 5 / 8)
    const cbar = (wa * Math.log(va) + wb * Math.log(vb)) / (wa + wb)
    const T = wa * (Math.log(va) - cbar) ** 2 + wb * (Math.log(vb) - cbar) ** 2
    expect(r.statistic).toBeCloseTo(T, 12)
    // equivalently z² = (ln va − ln vb)² / (1/wa + 1/wb)
    expect(r.statistic).toBeCloseTo((Math.log(va) - Math.log(vb)) ** 2 / (1 / wa + 1 / wb), 12)
    expect(equalVariances({ a, b }, 'bonett').test).toBe('Bonett')
    expect(() => bonett({ a: [1, 2, 3] })).toThrow(RangeError)
  })

  it('keeps its size under H0 for normal data and, unlike Bartlett, for heavy tails (Monte-Carlo, seeded)', () => {
    const reps = 3000
    const n = 30
    const g = rng(2024)
    let rejNormal = 0
    let rejBonettT3 = 0
    let rejBartlettT3 = 0
    for (let r = 0; r < reps; r++) {
      const normalGroups = [0, 1, 2].map(() => Array.from({ length: n }, () => 10 + 2 * g.normal()))
      if (bonett(normalGroups).pValue < 0.05) rejNormal++
      const heavy = [0, 1, 2].map(() => Array.from({ length: n }, () => g.t3()))
      if (bonett(heavy).pValue < 0.05) rejBonettT3++
      if (bartlett(heavy).pValue < 0.05) rejBartlettT3++
    }
    // nominal 5 % (binomial sd ≈ 0.4 % at 3000 reps): Bonett is slightly conservative-to-nominal in small samples
    expect(rejNormal / reps).toBeGreaterThan(0.025)
    expect(rejNormal / reps).toBeLessThan(0.07)
    expect(rejBonettT3 / reps).toBeLessThan(0.1)
    // Bartlett collapses under kurtosis: rejects far too often
    expect(rejBartlettT3 / reps).toBeGreaterThan(0.2)
    expect(rejBonettT3).toBeLessThan(rejBartlettT3 / 2)
  })

  it('two-sample version: CI for the ratio and p-value agree; power against a real difference', () => {
    const g = rng(7)
    const a = Array.from({ length: 25 }, () => 3 * g.normal())
    const b = Array.from({ length: 20 }, () => 1 * g.normal())
    const r = bonett2(a, b)
    expect(r.test).toBe('Bonett')
    expect(r.ratio).toBeCloseTo(r.groups[0]!.variance / r.groups[1]!.variance, 12)
    expect(r.ci![0]).toBeLessThan(r.ratio!)
    expect(r.ci![1]).toBeGreaterThan(r.ratio!)
    expect(r.pValue).toBeLessThan(0.01) // variance ratio ≈ 9
    expect(r.ci![0]).toBeGreaterThan(1)
    // consistency: at confidence 1 − p the interval touches 1
    const atP = bonett2(a, b, { confidence: 1 - r.pValue })
    expect(Math.min(Math.abs(atP.ci![0] - 1), Math.abs(atP.ci![1] - 1))).toBeLessThan(1e-6)
    // equal variances → p not small, interval covers 1
    const c = Array.from({ length: 25 }, () => 3 * g.normal())
    const same = bonett2(a, c)
    expect(same.ci![0]).toBeLessThan(1)
    expect(same.ci![1]).toBeGreaterThan(1)
    expect(same.pValue).toBeGreaterThan(0.05)
    expect(() => bonett2([1], [1, 2])).toThrow(RangeError)
  })

  it('DataFrame method dispatch', async () => {
    const g = rng(11)
    const rows = ['x', 'y', 'z'].flatMap((k, i) => Array.from({ length: 15 }, () => ({ v: (i + 1) * g.normal(), k })))
    const df = DataFrame.fromRows(rows)
    const r = df.equalVariances('v', 'k', 'bonett')
    expect(r.test).toBe('Bonett')
    expect(r.df).toBe(2)
    expect(r.pValue).toBeLessThan(0.05) // sds 1, 2, 3
    expect((await df.lazy().equalVariances('v', 'k', 'bonett')).statistic).toBeCloseTo(r.statistic, 12)
  })
})
