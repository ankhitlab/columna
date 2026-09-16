import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import { dunnett, fisherLSD, hsuMCB, pdunnett, qdunnett } from '@columna/advanced'

// References: Fisher LSD computed directly in Python (pooled MS_within + t), Dunnett from scipy.stats.dunnett
// (QMC integration → ~1e-4 noise), plus Dunnett's (1955) tabulated critical values.
const ref = JSON.parse(readFileSync(new URL('./fixtures/multcomp-scipy.json', import.meta.url), 'utf8'))
const groups = ref.groups as Record<string, number[]>
const close = (got: number, want: number, digits = 12) => expect(got).toBeCloseTo(want, digits)

describe('Fisher LSD', () => {
  it('matches pairwise t-tests on the pooled error term', () => {
    const f = fisherLSD(groups)
    expect(f.df).toBe(ref.fisher.df)
    close(f.msWithin, ref.fisher.msw)
    close(f.tCritical, ref.fisher.tcrit, 9)
    for (const r of ref.fisher.comparisons as Array<{ a: string; b: string; diff: number; t: number; p: number; lo: number; hi: number }>) {
      const c = f.comparisons.find((x) => x.a === r.a && x.b === r.b)!
      close(c.diff, r.diff)
      close(c.t, r.t)
      close(c.pValue, r.p)
      close(c.ci[0], r.lo, 9)
      close(c.ci[1], r.hi, 9)
      expect(c.significant).toBe(r.p < 0.05)
    }
    // family error rate for 6 comparisons at 5 %
    close(f.familyAlpha, 1 - 0.95 ** 6, 12)
    // letters consistent with significance; LSD is less conservative than Tukey → at least as many differences
    const letters = Object.fromEntries(f.groups.map((g) => [g.name, g.letters]))
    for (const c of f.comparisons) {
      const shared = [...letters[c.a]!].some((l) => letters[c.b]!.includes(l))
      expect(shared, `${c.a} vs ${c.b}`).toBe(!c.significant)
    }
  })
})

describe("Dunnett's distribution", () => {
  it('reproduces the classical table (k−1 = 3, df = 20, equal n): 2.54 two-sided, 2.19 one-sided', () => {
    const lambdas = [Math.SQRT1_2, Math.SQRT1_2, Math.SQRT1_2] // equal n → ρ = ½
    expect(qdunnett(0.95, lambdas, 20, true)).toBeCloseTo(ref.table.two_sided_05, 2)
    expect(qdunnett(0.95, lambdas, 20, false)).toBeCloseTo(ref.table.one_sided_05, 2)
    // with a single treatment it is just Student's t
    const d = qdunnett(0.95, [Math.SQRT1_2], 12, true)
    expect(d).toBeCloseTo(2.178812829667228, 6) // t(0.975, 12)
    expect(qdunnett(0.95, [Math.SQRT1_2], 12, false)).toBeCloseTo(1.782287555649159, 6) // t(0.95, 12)
    expect(pdunnett(0, lambdas, 20)).toBe(0)
    expect(pdunnett(Infinity, lambdas, 20)).toBe(1)
    // monotone and consistent inverse
    for (const c of [1, 2, 3]) expect(pdunnett(qdunnett(pdunnett(c, lambdas, 20), lambdas, 20), lambdas, 20)).toBeCloseTo(pdunnett(c, lambdas, 20), 8)
  })
})

describe('Dunnett comparisons', () => {
  const treatments = ref.dunnett.treatments as string[]
  for (const alt of ['two-sided', 'greater', 'less'] as const) {
    it(`${alt}: statistics, adjusted p-values and CIs match scipy.stats.dunnett`, () => {
      const r = ref.dunnett[alt]
      const d = dunnett(groups, { control: ref.dunnett.control, alternative: alt })
      expect(d.control).toBe('b')
      expect(d.comparisons.map((c) => c.group)).toEqual(treatments)
      d.comparisons.forEach((c, i) => {
        close(c.t, r.stat[i])
        // scipy integrates by QMC: agreement to ~2e-3 in p, ~5e-3 in interval ends
        expect(Math.abs(c.pValue - r.p[i]), `p ${c.group}`).toBeLessThan(2e-3)
        if (r.lo[i] !== null) expect(Math.abs(c.ci[0] - r.lo[i]), `lo ${c.group}`).toBeLessThan(5e-3)
        else expect(c.ci[0]).toBe(-Infinity)
        if (r.hi[i] !== null) expect(Math.abs(c.ci[1] - r.hi[i]), `hi ${c.group}`).toBeLessThan(5e-3)
        else expect(c.ci[1]).toBe(Infinity)
        expect(c.significant).toBe(alt === 'two-sided' ? c.ci[0] > 0 || c.ci[1] < 0 : alt === 'greater' ? c.ci[0] > 0 : c.ci[1] < 0)
      })
    })
  }
  it('errors and DataFrame methods', async () => {
    expect(() => dunnett(groups, { control: 'zzz' })).toThrow(/control group/)
    const df = DataFrame.fromRows(Object.entries(groups).flatMap(([g, xs]) => xs.map((x) => ({ x, g }))))
    const d = df.dunnett('x', 'g', { control: 'b' })
    close(d.comparisons[0]!.t, ref.dunnett['two-sided'].stat[0])
    const f = df.fisher('x', 'g')
    close(f.comparisons[0]!.t, ref.fisher.comparisons[0].t)
    expect((await df.lazy().dunnett('x', 'g', { control: 'b', alpha: 0.1 })).alpha).toBe(0.1)
    expect((await df.lazy().fisher('x', 'g')).test).toBe('Fisher LSD')
  })
})

describe("Hsu's MCB", () => {
  const g = ref.groups as Record<string, number[]>
  it('constrained intervals: equal-n case reduces to one-sided Dunnett half-widths; structure holds', () => {
    // equal n → one shared d = one-sided Dunnett quantile with k−1 = 3, df = 20 (table: 2.19)
    const eq = { a: [10, 12, 11, 13, 12, 11], b: [14, 15, 13, 16, 15, 14], c: [9, 8, 10, 9, 8, 10], d: [12, 13, 12, 14, 13, 12] }
    const h = hsuMCB(eq)
    expect(h.df).toBe(20)
    expect(new Set(h.groups.map((x) => x.dCritical.toFixed(9))).size).toBe(1)
    expect(h.groups[0]!.dCritical).toBeCloseTo(2.19, 2)
    const D = h.groups[0]!.dCritical * Math.sqrt(h.msWithin * (2 / 6))
    for (const x of h.groups) {
      // every interval touches 0 at one end and contains it
      expect(Math.min(x.ci[0], x.ci[1])).toBeLessThanOrEqual(0)
      expect(Math.max(x.ci[0], x.ci[1])).toBeGreaterThanOrEqual(0)
      expect(x.ci[0] === 0 || x.ci[1] === 0).toBe(true)
      const diff = x.diffToBest
      expect(x.ci[0]).toBeCloseTo(Math.min(0, diff - D), 10)
      expect(x.ci[1]).toBeCloseTo(Math.max(0, diff + D), 10)
    }
    // b is clearly the largest: it is the only candidate and significantly best; c is excluded
    const byName = Object.fromEntries(h.groups.map((x) => [x.name, x]))
    expect(byName.b!.isBest).toBe(true)
    expect(byName.b!.ci[0]).toBe(0)
    expect(byName.b!.canBeBest).toBe(true)
    expect(byName.c!.canBeBest).toBe(false)
    expect(byName.c!.ci[1]).toBe(0)
    expect(h.groups.filter((x) => x.canBeBest).length).toBeGreaterThanOrEqual(1)
    // smallest is best: c wins
    const hs = hsuMCB(eq, { best: 'smallest' })
    const s = Object.fromEntries(hs.groups.map((x) => [x.name, x]))
    expect(s.c!.isBest).toBe(true)
    expect(s.c!.ci[1]).toBe(0)
    expect(s.b!.canBeBest).toBe(false)
    expect(s.a!.diffToBest).toBeCloseTo(11.5 - 9, 12)
  })
  it('unbalanced groups: per-group d from Dunnett with group i as control; consistent with dunnett()', () => {
    const h = hsuMCB(g)
    // group 'b' as control in dunnett (one-sided, greater) must use the same critical value as Hsu's d for 'b'
    const d = dunnett(g, { control: 'b', alternative: 'greater' })
    const hb = h.groups.find((x) => x.name === 'b')!
    expect(hb.dCritical).toBeCloseTo(d.tCritical, 8)
    // intervals always contain 0; they touch 0 exactly when the group is significantly best or significantly not best
    for (const x of h.groups) {
      expect(x.ci[0]).toBeLessThanOrEqual(0)
      expect(x.ci[1]).toBeGreaterThanOrEqual(0)
      expect(x.ci[0] === 0 || x.ci[1] === 0).toBe(x.isBest || !x.canBeBest)
    }
    // overlapping means here: every group remains a candidate, none is significantly best
    expect(h.groups.every((x) => x.canBeBest && !x.isBest)).toBe(true)
    // candidates never include a group whose interval excludes the possibility of being best
    for (const x of h.groups) expect(x.canBeBest).toBe(x.ci[1] > 0)
    expect(() => hsuMCB({ a: [1, 2] })).toThrow(RangeError)
    expect(() => hsuMCB(g, { alpha: 0 })).toThrow(RangeError)
  })
  it('DataFrame / LazyFrame', async () => {
    const df = DataFrame.fromRows(Object.entries(g).flatMap(([k, xs]) => xs.map((x) => ({ x, k }))))
    const h = df.hsu('x', 'k', { best: 'largest' })
    expect(h.groups.map((x) => x.name)).toEqual(['a', 'b', 'c', 'd'])
    expect((await df.lazy().hsu('x', 'k', { alpha: 0.1 })).alpha).toBe(0.1)
  })
})
