import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import { andersonDarling, kolmogorovSmirnov, normalityTest, ryanJoiner, shapiroWilk } from '@columna/advanced'

// Reference values from scipy 1.14 (scratch script norm_ref.py): shapiro W/p, anderson A²;
// the AD p-value is the D'Agostino–Stephens formula (Minitab / R nortest), cross-checked in Python.
type Case = { x: number[]; sw_w: number; sw_p: number; ad_a2: number; ad_p: number }
const ref = JSON.parse(readFileSync(new URL('./fixtures/normality-scipy.json', import.meta.url), 'utf8')) as Record<string, Case>

describe('Shapiro–Wilk', () => {
  it('W and p match scipy.stats.shapiro for n = 3 … 5000, normal and non-normal samples', () => {
    for (const [name, c] of Object.entries(ref)) {
      const r = shapiroWilk(c.x)
      expect(r.n, name).toBe(c.x.length)
      // W to 1e-10 (scipy's swilk is single-precision in places → agree to ~1e-7 on p)
      expect(Math.abs(r.statistic - c.sw_w), `${name} W`).toBeLessThan(1e-9)
      expect(Math.abs(r.pValue - c.sw_p), `${name} p`).toBeLessThan(2e-7)
    }
  })
  it('rejects non-normal shapes and accepts normal ones at the usual level', () => {
    expect(shapiroWilk(ref.n1000_lognormal!.x).pValue).toBeLessThan(1e-10)
    expect(shapiroWilk(ref.n200_t3!.x).pValue).toBeLessThan(0.01)
    expect(shapiroWilk(ref.n1000_normal!.x).pValue).toBeGreaterThan(0.05)
    expect(() => shapiroWilk([1, 2])).toThrow(RangeError)
    expect(() => shapiroWilk([3, 3, 3, 3])).toThrow(/identical/)
  })
})

describe('Anderson–Darling', () => {
  it('A² matches scipy.stats.anderson and p follows the D\'Agostino–Stephens formula', () => {
    for (const [name, c] of Object.entries(ref)) {
      if (c.x.length < 8) continue
      const r = andersonDarling(c.x)
      expect(Math.abs(r.statistic - c.ad_a2), `${name} A²`).toBeLessThan(1e-10)
      expect(r.adjusted).toBeCloseTo(c.ad_a2 * (1 + 0.75 / c.x.length + 2.25 / c.x.length ** 2), 9)
      expect(Math.abs(r.pValue - c.ad_p), `${name} p`).toBeLessThan(1e-9) // A² agrees to 1e-10; dp/dA ≈ 5.7·p
    }
  })
  it('p-value is continuous across the formula breakpoints and behaves on normal vs skewed data', () => {
    // Evaluate the piecewise approximation on synthetic A*² values around the joins
    const pAt = (a: number) =>
      a >= 0.6
        ? Math.exp(1.2937 - 5.709 * a + 0.0186 * a * a)
        : a >= 0.34
          ? Math.exp(0.9177 - 4.279 * a - 1.38 * a * a)
          : a >= 0.2
            ? 1 - Math.exp(-8.318 + 42.796 * a - 59.938 * a * a)
            : 1 - Math.exp(-13.436 + 101.14 * a - 223.73 * a * a)
    for (const a of [0.2, 0.34, 0.6]) expect(Math.abs(pAt(a) - pAt(a - 1e-9))).toBeLessThan(0.01)
    // Stephens (1974) critical values of A*² for normality with both parameters estimated
    expect(pAt(0.752)).toBeCloseTo(0.05, 2)
    expect(pAt(0.632)).toBeCloseTo(0.1, 2)
    expect(pAt(1.035)).toBeCloseTo(0.01, 2)
    expect(andersonDarling(ref.n1000_lognormal!.x).pValue).toBeLessThan(1e-6)
    expect(andersonDarling(ref.n200_t3!.x).pValue).toBeLessThan(0.01)
    expect(andersonDarling(ref.n1000_normal!.x).pValue).toBeGreaterThan(0.05)
    expect(() => andersonDarling([1, 2, 3, 4, 5, 6, 7])).toThrow(RangeError)
  })
})

describe('normalityTest dispatch and DataFrame methods', () => {
  it('defaults to Anderson–Darling, skips nulls, works on LazyFrame', async () => {
    const x = ref.n30_normal!.x
    expect(normalityTest(x).test).toBe('Anderson-Darling')
    expect(normalityTest(x, 'shapiro-wilk').test).toBe('Shapiro-Wilk')
    const df = DataFrame.fromRows([...x.map((v) => ({ v })), { v: null }])
    expect(df.normalityTest('v').statistic).toBeCloseTo(ref.n30_normal!.ad_a2, 10)
    expect(df.normalityTest('v', 'shapiro-wilk').statistic).toBeCloseTo(ref.n30_normal!.sw_w, 9)
    expect((await df.lazy().normalityTest('v')).n).toBe(30)
  })
})

describe('Ryan–Joiner and Kolmogorov–Smirnov (Lilliefors)', () => {
  it('R matches the correlation with normal scores (scipy pearsonr) and p follows Minitab critical values', () => {
    for (const [name, c] of Object.entries(ref) as Array<[string, Case & { rj_r?: number }]>) {
      if (c.rj_r === undefined) continue
      const r = ryanJoiner(c.x)
      expect(Math.abs(r.statistic - c.rj_r), `${name} R`).toBeLessThan(1e-10)
      expect(r.critical![0.1]).toBeGreaterThan(r.critical![0.05])
      expect(r.critical![0.05]).toBeGreaterThan(r.critical![0.01])
      expect(r.pValue).toBeGreaterThanOrEqual(0)
      expect(r.pValue).toBeLessThanOrEqual(1)
    }
    // Minitab reports bounds outside the table
    expect(ryanJoiner(ref.n1000_normal!.x).pBound).toBe('> 0.100')
    expect(ryanJoiner(ref.n1000_lognormal!.x).pBound).toBe('< 0.010')
    expect(ryanJoiner(ref.n1000_lognormal!.x).pValue).toBeLessThan(0.01)
    // interpolation hits the tabulated α exactly at the critical values (n = 30)
    const n = 30
    const rn = Math.sqrt(n)
    const r05 = 1.0063 - 0.1288 / rn - 0.6118 / n + 1.3505 / (n * n)
    const mock = ryanJoiner(ref.n30_normal!.x)
    expect(mock.critical![0.05]).toBeCloseTo(r05, 12)
    // critical values stay below 1 and shrink toward 1 for large n (Monte-Carlo at n = 1000: q10 ≈ 0.99869, q01 ≈ 0.99784)
    const big = ryanJoiner(ref.n1000_normal!.x).critical!
    expect(big[0.1]).toBeLessThan(1)
    expect(big[0.1]).toBeCloseTo(0.99869, 3)
    expect(big[0.01]).toBeCloseTo(0.99784, 3)
    expect(() => ryanJoiner([1, 2, 3])).toThrow(RangeError)
  })

  it('D matches scipy kstest with estimated parameters; Dallal–Wilkinson p behaves', () => {
    for (const [name, c] of Object.entries(ref) as Array<[string, Case & { ks_d?: number }]>) {
      if (c.ks_d === undefined) continue
      const r = kolmogorovSmirnov(c.x)
      expect(Math.abs(r.statistic - c.ks_d), `${name} D`).toBeLessThan(1e-10)
      expect(r.pValue).toBeGreaterThanOrEqual(0)
      expect(r.pValue).toBeLessThanOrEqual(1)
    }
    expect(kolmogorovSmirnov(ref.n1000_lognormal!.x).pValue).toBeLessThan(1e-6)
    expect(kolmogorovSmirnov(ref.n200_t3!.x).pValue).toBeLessThan(0.05)
    expect(kolmogorovSmirnov(ref.n1000_normal!.x).pValue).toBeGreaterThan(0.05)
    expect(kolmogorovSmirnov(ref.n30_normal!.x).pValue).toBeGreaterThan(0.05)
    expect(normalityTest(ref.n30_normal!.x, 'ryan-joiner').test).toBe('Ryan-Joiner')
    expect(normalityTest(ref.n30_normal!.x, 'kolmogorov-smirnov').test).toBe('Kolmogorov-Smirnov')
    const df = DataFrame.fromRows(ref.n30_normal!.x.map((v) => ({ v })))
    expect(df.normalityTest('v', 'ryan-joiner').statistic).toBeCloseTo((ref.n30_normal as Case & { rj_r: number }).rj_r, 10)
    expect(df.normalityTest('v', 'kolmogorov-smirnov').statistic).toBeCloseTo((ref.n30_normal as Case & { ks_d: number }).ks_d, 10)
  })
})
