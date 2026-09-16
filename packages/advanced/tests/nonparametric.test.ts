import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import { kruskal, mannWhitney } from '@columna/advanced'

// Reference values from scipy 1.14 (scratch script np_ref.py)
const ref = JSON.parse(readFileSync(new URL('./fixtures/nonparametric-scipy.json', import.meta.url), 'utf8'))
const close = (got: number, want: number, digits = 12) => expect(got).toBeCloseTo(want, digits)
const alts = ['two-sided', 'less', 'greater'] as const

describe('Mann–Whitney', () => {
  it('exact and asymptotic p-values match scipy mannwhitneyu (no ties)', () => {
    const { a, b } = ref.mw_noties
    for (const alt of alts) {
      const ex = mannWhitney(a, b, { alternative: alt }) // auto → exact (no ties, 80 pairs)
      expect(ex.method).toBe('exact')
      expect(ex.ties).toBe(false)
      close(ex.statistic, ref.mw_noties.exact[alt].U)
      close(ex.pValue, ref.mw_noties.exact[alt].p)
      const as = mannWhitney(a, b, { alternative: alt, method: 'asymptotic' })
      close(as.pValue, ref.mw_noties.asymptotic[alt].p)
    }
    expect(mannWhitney(a, b).W).toBe(mannWhitney(a, b).statistic + (10 * 11) / 2)
  })
  it('tie-adjusted asymptotic p-values match scipy; exact refuses ties', () => {
    const { a, b } = ref.mw_ties
    for (const alt of alts) {
      const r = mannWhitney(a, b, { alternative: alt })
      expect(r.method).toBe('asymptotic')
      expect(r.ties).toBe(true)
      close(r.statistic, ref.mw_ties.asymptotic[alt].U)
      close(r.pValue, ref.mw_ties.asymptotic[alt].p)
    }
    expect(() => mannWhitney(a, b, { method: 'exact' })).toThrow(/ties/)
  })
  it('Hodges–Lehmann estimate and Minitab-style CI for η₁ − η₂', () => {
    const { a, b } = ref.mw_noties
    const r = mannWhitney(a, b)
    close(r.estimate, ref.hl.estimate)
    close(r.ci[0], ref.hl.lo)
    close(r.ci[1], ref.hl.hi)
    close(r.confidence, ref.hl.achieved, 10)
    expect(r.confidence).toBeGreaterThan(0.94)
    expect(r.medians).toEqual([5.65, 4.975])
    const g = mannWhitney(a, b, { alternative: 'greater' })
    expect(g.ci[1]).toBe(Infinity)
    expect(g.ci[0]).toBeGreaterThan(0) // a is clearly above b
    expect(() => mannWhitney([], [1])).toThrow(RangeError)
  })
})

describe('Kruskal–Wallis', () => {
  it('H and p match scipy kruskal, with and without ties', () => {
    const k = kruskal(ref.kw.groups)
    close(k.statistic, ref.kw.H)
    close(k.pValue, ref.kw.p)
    expect(k.df).toBe(2)
    expect(k.n).toBe(18)
    expect(k.statistic).toBeCloseTo(k.hUnadjusted, 12) // no ties → no adjustment
    const t = kruskal(ref.kw_ties.groups)
    close(t.statistic, ref.kw_ties.H)
    close(t.pValue, ref.kw_ties.p)
    expect(t.statistic).toBeGreaterThan(t.hUnadjusted)
    // Minitab's per-group output: medians, average ranks, z-values (sum of z weighted by n is ≈ 0)
    expect(k.groups.map((g) => g.name)).toEqual(['g1', 'g2', 'g3'])
    expect(k.groups.map((g) => g.median)).toEqual([5.65, 5, 6.3])
    const weighted = k.groups.reduce((s, g) => s + g.n * (g.avgRank - 9.5), 0)
    expect(Math.abs(weighted)).toBeLessThan(1e-9)
    expect(k.groups[2]!.z).toBeGreaterThan(2) // g3 ranks highest
    expect(() => kruskal({ a: [1, 2] })).toThrow(RangeError)
  })
})

describe('DataFrame / LazyFrame methods', () => {
  it('mannWhitney by a 2-level column and kruskal by a k-level column', async () => {
    const { a, b } = ref.mw_noties
    const df = DataFrame.fromRows([...a.map((x: number) => ({ x, g: 'A' })), ...b.map((x: number) => ({ x, g: 'B' }))])
    const r = df.mannWhitney('x', 'g')
    close(r.statistic, ref.mw_noties.exact['two-sided'].U)
    expect((await df.lazy().mannWhitney('x', 'g', { alternative: 'greater' })).alternative).toBe('greater')
    const kw = DataFrame.fromRows(Object.entries(ref.kw.groups as Record<string, number[]>).flatMap(([g, xs]) => xs.map((x) => ({ x, g }))))
    close(kw.kruskal('x', 'g').statistic, ref.kw.H)
    close((await kw.lazy().kruskal('x', 'g')).pValue, ref.kw.p)
    expect(() => kw.mannWhitney('x', 'g')).toThrow(/2 levels/)
  })
})
