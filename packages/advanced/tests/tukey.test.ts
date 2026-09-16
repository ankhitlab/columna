import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import { bartlett, equalVariances, levene, ptukey, qtukey, tukeyHSD, varTest2 } from '@columna/advanced'

// Reference values from scipy 1.14 (scratch script tukey_ref.py)
const ref = JSON.parse(readFileSync(new URL('./fixtures/tukey-scipy.json', import.meta.url), 'utf8'))
const close = (got: number, want: number, digits = 12) => expect(got).toBeCloseTo(want, digits)
const groups = ref.groups as Record<string, number[]>

describe('studentized range distribution', () => {
  it('ptukey / qtukey match scipy.stats.studentized_range on k = 2 … 20, df = 5 … 10000', () => {
    let worstCdf = 0
    let worstPpf = 0
    for (const r of ref.srange as Array<{ k: number; df: number; q?: number; cdf?: number; p?: number; ppf?: number }>) {
      if (r.q !== undefined) {
        const got = ptukey(r.q, r.k, r.df)
        const e = Math.abs(got - r.cdf!)
        if (e > 1e-7) throw new Error(`ptukey(${r.q}, ${r.k}, ${r.df}) = ${got}, scipy ${r.cdf}`)
        worstCdf = Math.max(worstCdf, e)
      } else {
        const got = qtukey(r.p!, r.k, r.df)
        const e = Math.abs(got - r.ppf!) / r.ppf!
        if (e > 2e-6) throw new Error(`qtukey(${r.p}, ${r.k}, ${r.df}) = ${got}, scipy ${r.ppf}`)
        worstPpf = Math.max(worstPpf, e)
        // and it inverts our own cdf
        close(ptukey(got, r.k, r.df), r.p!, 9)
      }
    }
    expect(worstCdf).toBeLessThan(1e-7) // Copenhaver–Holland quadrature is accurate to ~1e-7, same as R
    expect(worstPpf).toBeLessThan(2e-6)
  })
  it('textbook critical values and edge cases', () => {
    // q(0.95; k=3, df=21) ≈ 3.565, q(0.95; k=4, df=27) ≈ 3.87 (Tukey tables)
    close(qtukey(0.95, 3, 21), 3.565, 2)
    close(qtukey(0.95, 4, 27), 3.87, 2)
    expect(ptukey(0, 3, 10)).toBe(0)
    expect(ptukey(Infinity, 3, 10)).toBe(1)
    expect(ptukey(2, 1, 10)).toBeNaN()
    expect(qtukey(0, 3, 10)).toBe(0)
    expect(qtukey(1, 3, 10)).toBe(Infinity)
  })
})

describe('Tukey HSD', () => {
  it('pairwise differences, p-values and simultaneous CIs match scipy tukey_hsd', () => {
    const t = tukeyHSD(groups)
    expect(t.comparisons.length).toBe(6)
    for (const r of ref.tukey as Array<{ a: string; b: string; diff: number; p: number; lo: number; hi: number }>) {
      const c = t.comparisons.find((x) => x.a === r.a && x.b === r.b)!
      close(c.diff, r.diff)
      close(c.pValue, r.p, 7)
      close(c.ci[0], r.lo, 6)
      close(c.ci[1], r.hi, 6)
      expect(c.significant).toBe(r.p < 0.05)
      expect(c.significant).toBe(r.lo > 0 || r.hi < 0)
    }
    expect(t.df).toBe(27)
    expect(t.groups.map((g) => g.name)).toEqual(['a', 'b', 'c', 'd'])
    expect(t.qCritical).toBeCloseTo(qtukey(0.95, 4, 27), 12)
  })
  it('grouping letters: groups sharing a letter are not significantly different', () => {
    const t = tukeyHSD(groups)
    const letters = Object.fromEntries(t.groups.map((g) => [g.name, g.letters]))
    for (const c of t.comparisons) {
      const shared = [...letters[c.a]!].some((l) => letters[c.b]!.includes(l))
      expect(shared, `${c.a} vs ${c.b}`).toBe(!c.significant)
    }
    // clearly separated groups get distinct letters
    const sep = tukeyHSD({ lo: [1, 1.1, 0.9, 1.05], mid: [5, 5.1, 4.9, 5.05], hi: [9, 9.1, 8.9, 9.05] })
    expect(sep.groups.map((g) => g.letters)).toEqual(['C', 'B', 'A'])
    expect(sep.comparisons.every((c) => c.significant)).toBe(true)
    expect(() => tukeyHSD({ a: [1, 2] })).toThrow(RangeError)
    expect(() => tukeyHSD(groups, { alpha: 1.5 })).toThrow(RangeError)
  })
})

describe('equal variances', () => {
  it('Bartlett and Levene (median / mean) match scipy', () => {
    const b = bartlett(groups)
    close(b.statistic, ref.bartlett.stat)
    close(b.pValue, ref.bartlett.p)
    expect(b.df).toBe(3)
    const lm = levene(groups)
    expect(lm.test).toBe('Brown-Forsythe')
    close(lm.statistic, ref.levene_median.stat)
    close(lm.pValue, ref.levene_median.p)
    expect(lm.df).toEqual([3, 27])
    const lmean = levene(groups, { center: 'mean' })
    expect(lmean.test).toBe('Levene')
    close(lmean.statistic, ref.levene_mean.stat)
    close(lmean.pValue, ref.levene_mean.p)
    expect(equalVariances(groups).test).toBe('Brown-Forsythe')
    expect(equalVariances(groups, 'bartlett').test).toBe('Bartlett')
    expect(() => bartlett({ a: [1, 2, 3] })).toThrow(RangeError)
  })
  it('two-sample F-test with CI for the variance ratio', () => {
    const f = varTest2(groups.a!, groups.b!)
    close(f.ratio!, ref.ftest.ratio)
    close(f.pValue, ref.ftest.p, 11)
    close(f.ci![0], ref.ftest.lo, 9)
    close(f.ci![1], ref.ftest.hi, 9)
    expect(f.df).toEqual([9, 7])
    expect(varTest2(groups.a!, groups.b!, { alternative: 'greater' }).ci![1]).toBe(Infinity)
  })
  it('DataFrame / LazyFrame methods', async () => {
    const df = DataFrame.fromRows(Object.entries(groups).flatMap(([g, xs]) => xs.map((x) => ({ x, g }))))
    close(df.equalVariances('x', 'g').statistic, ref.levene_median.stat)
    close(df.equalVariances('x', 'g', 'bartlett').statistic, ref.bartlett.stat)
    const t = df.tukey('x', 'g')
    close(t.comparisons[0]!.pValue, ref.tukey[0].p, 7)
    close((await df.lazy().tukey('x', 'g', { alpha: 0.1 })).alpha, 0.1)
    close((await df.lazy().equalVariances('x', 'g')).pValue, ref.levene_median.p)
  })
})
