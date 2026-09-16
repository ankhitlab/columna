import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import { friedman, moodMedian, runsTest, signTest, ttest1, wilcoxonSigned } from '@columna/advanced'

// Reference values from scipy 1.14 (tests/refs/tier2_ref.py)
const ref = JSON.parse(readFileSync(new URL('./fixtures/tier2-scipy.json', import.meta.url), 'utf8'))
const close = (got: number, want: number, digits = 10) => expect(got).toBeCloseTo(want, digits)
const alts = ['two-sided', 'less', 'greater'] as const

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
const sample = (n: number, g: () => number) => Array.from({ length: n }, g)

describe('2.7 sign test', () => {
  it('exact binomial p-values match scipy binomtest; counts and estimate', () => {
    const c = ref.sign
    for (const alt of alts) {
      const r = signTest(c.x, { median: c.median, alternative: alt })
      close(r.pValue, c.p[alt], 12)
      expect(r.above).toBe(c.above)
      expect(r.below).toBe(c.below)
      expect(r.equal).toBe(0)
    }
    const r = signTest(c.x, { median: c.median })
    const sorted = [...c.x].sort((a: number, b: number) => a - b)
    close(r.estimate, (sorted[6] + sorted[7]) / 2, 12)
    // values equal to the hypothesized median are dropped
    const z = signTest([1, 2, 2, 3, 5, 7, 2], { median: 2 })
    expect(z.equal).toBe(3)
    expect(z.n).toBe(7) // n counts all observations; the test uses below + above = 4
    expect(z.below + z.above).toBe(4)
    expect(signTest([2, 2], { median: 2 }).pValue).toBe(1) // all ties → no evidence
    expect(() => signTest([null, undefined], { median: 2 })).toThrow(RangeError)
  })

  it('achievable intervals bracket the requested confidence and the interpolated CI lies between them', () => {
    const r = signTest(ref.sign.x, { median: 1.5 })
    expect(r.achievable.length).toBe(2)
    const [wide, narrow] = r.achievable
    expect(wide!.confidence).toBeGreaterThanOrEqual(0.95)
    expect(narrow!.confidence).toBeLessThan(0.95)
    expect(wide!.position + 1).toBe(narrow!.position)
    expect(r.ci[0]).toBeGreaterThanOrEqual(wide!.ci[0])
    expect(r.ci[0]).toBeLessThanOrEqual(narrow!.ci[0])
    expect(r.ci[1]).toBeLessThanOrEqual(wide!.ci[1])
    expect(r.ci[1]).toBeGreaterThanOrEqual(narrow!.ci[1])
    // n = 14: the 95 % achievable interval is [x(3), x(12)] with confidence 1 − 2·P(Bin(14, ½) ≤ 2) = 0.9871
    close(wide!.confidence, 1 - 2 * (1 + 14 + 91) / 2 ** 14, 12)
    expect(wide!.position).toBe(3)
  })

  it('exact size ≤ α on any continuous distribution, interpolated CI covers ≈ 95 % (Monte-Carlo, seeded)', () => {
    const g = rng(707)
    const reps = 3000
    let rej = 0
    let cover = 0
    let rejT = 0
    for (let i = 0; i < reps; i++) {
      const x = sample(20, g.t3)
      const r = signTest(x)
      if (r.pValue < 0.05) rej++
      if (r.ci[0] <= 0 && 0 <= r.ci[1]) cover++
      if (ttest1(x).pValue < 0.05) rejT++
    }
    expect(rej / reps).toBeLessThanOrEqual(0.055)
    expect(rej / reps).toBeGreaterThan(0.015)
    expect(cover / reps).toBeGreaterThan(0.93)
    expect(cover / reps).toBeLessThan(0.98)
    expect(rejT / reps).toBeLessThan(0.07) // t stays valid too (symmetric), just for reference
  })
})

describe('2.7 Wilcoxon signed-rank', () => {
  it('exact and asymptotic p-values match scipy wilcoxon (no ties)', () => {
    const c = ref.wilcoxon_exact
    for (const alt of alts) {
      const r = wilcoxonSigned(c.x, { median: c.median, alternative: alt })
      expect(r.method).toBe('exact')
      close(r.statistic, c.W, 12)
      close(r.pValue, c.p[alt], 12)
      const a = wilcoxonSigned(c.x, { median: c.median, alternative: alt, method: 'asymptotic' })
      close(a.pValue, ref.wilcoxon_approx.p[alt], 10)
    }
  })

  it('ties and zeros: tie-corrected normal approximation matches scipy (zero_method wilcox)', () => {
    const c = ref.wilcoxon_ties
    for (const alt of alts) {
      const r = wilcoxonSigned(c.x, { median: c.median, alternative: alt })
      expect(r.method).toBe('asymptotic')
      expect(r.n).toBe(c.nNonzero)
      expect(r.nTotal).toBe(c.x.length)
      close(r.pValue, c.p[alt], 10)
    }
    expect(() => wilcoxonSigned(c.x, { median: c.median, method: 'exact' })).toThrow(RangeError)
  })

  it('Walsh-average estimate and Minitab-style CI', () => {
    const c = ref.wilcoxon_ci
    const r = wilcoxonSigned(ref.wilcoxon_exact.x, { median: 1.5 })
    close(r.estimate, c.estimate, 12)
    close(r.ci[0], c.ci[0], 12)
    close(r.ci[1], c.ci[1], 12)
    const g = wilcoxonSigned(ref.wilcoxon_exact.x, { median: 1.5, alternative: 'greater' })
    expect(g.ci[1]).toBe(Infinity)
    expect(g.ci[0]).toBeLessThan(r.estimate)
  })

  it('exact size ≈ α, CI coverage ≈ 95 %, more powerful than the sign test under a normal shift (Monte-Carlo, seeded)', () => {
    const g = rng(808)
    const reps = 3000
    let rej = 0
    let cover = 0
    let powW = 0
    let powS = 0
    for (let i = 0; i < reps; i++) {
      const x = sample(15, g.normal)
      const r = wilcoxonSigned(x)
      if (r.pValue < 0.05) rej++
      if (r.ci[0] <= 0 && 0 <= r.ci[1]) cover++
      const y = x.map((v) => v + 0.6)
      if (wilcoxonSigned(y).pValue < 0.05) powW++
      if (signTest(y).pValue < 0.05) powS++
    }
    expect(rej / reps).toBeGreaterThan(0.025)
    expect(rej / reps).toBeLessThanOrEqual(0.055)
    expect(cover / reps).toBeGreaterThan(0.93)
    expect(cover / reps).toBeLessThan(0.98)
    expect(powW).toBeGreaterThan(powS * 1.15)
    expect(powW / reps).toBeGreaterThan(0.5)
  })
})

describe("2.8 Mood's median test", () => {
  it('matches scipy median_test(ties="below", correction=False)', () => {
    const c = ref.mood
    const r = moodMedian(c.groups)
    close(r.statistic, c.stat, 10)
    close(r.pValue, c.p, 10)
    close(r.grandMedian, c.median, 12)
    expect(r.df).toBe(2)
    expect(r.groups.map((g) => g.nAbove)).toEqual(c.table[0])
    expect(r.groups.map((g) => g.nBelowOrEqual)).toEqual(c.table[1])
    const r2 = moodMedian([c.groups.g1, c.groups.g2])
    close(r2.statistic, ref.mood2.stat, 10)
    close(r2.pValue, ref.mood2.p, 10)
    expect(() => moodMedian({ a: [1, 2, 3] })).toThrow(RangeError)
  })

  it('size near α with 3 groups of 20 and power under a location shift (Monte-Carlo, seeded)', () => {
    const g = rng(909)
    const reps = 3000
    let rej = 0
    let pow = 0
    for (let i = 0; i < reps; i++) {
      const a = sample(20, g.t3)
      const b = sample(20, g.t3)
      const c = sample(20, g.t3)
      if (moodMedian([a, b, c]).pValue < 0.05) rej++
      if (moodMedian([a, b, c.map((v) => v + 1.2)]).pValue < 0.05) pow++
    }
    expect(rej / reps).toBeGreaterThan(0.02)
    expect(rej / reps).toBeLessThan(0.065)
    expect(pow / reps).toBeGreaterThan(0.7)
  })
})

describe('2.8 Friedman', () => {
  it('matches scipy friedmanchisquare with and without ties', () => {
    const c = ref.friedman
    const r = friedman(c.table, ['A', 'B', 'C', 'D'])
    close(r.statistic, c.stat, 10)
    close(r.pValue, c.p, 10)
    expect(r.df).toBe(3)
    expect(r.blocks).toBe(6)
    expect(r.treatments.map((t) => t.name)).toEqual(['A', 'B', 'C', 'D'])
    close(r.treatments.reduce((s, t) => s + t.sumRanks, 0), 6 * 10, 12)
    const t = friedman(ref.friedman_ties.table)
    close(t.statistic, ref.friedman_ties.stat, 10)
    close(t.pValue, ref.friedman_ties.p, 10)
    expect(t.sUnadjusted).toBeLessThan(t.statistic)
    expect(() => friedman([[1, 2]])).toThrow(RangeError)
    expect(() => friedman([[1, 2], [1]])).toThrow(RangeError)
  })

  it('size near α (12 blocks × 4 treatments) and power for a treatment effect (Monte-Carlo, seeded)', () => {
    const g = rng(1010)
    const reps = 3000
    let rej = 0
    let pow = 0
    for (let i = 0; i < reps; i++) {
      const blockEffect = sample(12, () => 3 * g.normal())
      const table = blockEffect.map((b) => [b + g.normal(), b + g.normal(), b + g.normal(), b + g.normal()])
      if (friedman(table).pValue < 0.05) rej++
      const shifted = table.map((row) => [row[0]!, row[1]!, row[2]!, row[3]! + 1.5])
      if (friedman(shifted).pValue < 0.05) pow++
    }
    expect(rej / reps).toBeGreaterThan(0.03)
    expect(rej / reps).toBeLessThan(0.07)
    expect(pow / reps).toBeGreaterThan(0.7)
  })
})

describe('2.8 runs test', () => {
  it('matches the normal-approximation formula about the mean', () => {
    const c = ref.runs
    const r = runsTest(c.x)
    expect(r.runs).toBe(c.runs)
    expect(r.nAbove).toBe(c.n1)
    expect(r.nBelow).toBe(c.n2)
    close(r.k, c.k, 12)
    close(r.expected, c.expected, 12)
    close(r.statistic, c.z, 10)
    close(r.pValue, c.p, 10)
    // explicit k; values equal to k count as below
    const s = runsTest([1, 3, 1, 3, 3, 1, 1, 3], { k: 3 })
    expect(s.nAbove).toBe(0)
    expect(Number.isNaN(s.statistic)).toBe(true)
    expect(s.pValue).toBe(1)
    const alt = runsTest([1, 3, 1, 3, 1, 3, 1, 3], { k: 2 })
    expect(alt.runs).toBe(8)
    close(alt.expected, 5, 12)
    close(alt.pValue, 0.021946771003246907, 10) // z = 3/√(768/448)
    const cc = runsTest(c.x, { correction: true })
    expect(Math.abs(cc.statistic)).toBeLessThan(Math.abs(r.statistic))
    expect(() => runsTest([1])).toThrow(RangeError)
  })

  it('size near α on iid data and power against AR(1) dependence (Monte-Carlo, seeded)', () => {
    const g = rng(1111)
    const reps = 3000
    let rej = 0
    let pow = 0
    for (let i = 0; i < reps; i++) {
      const x = sample(40, g.normal)
      if (runsTest(x).pValue < 0.05) rej++
      const y: number[] = [g.normal()]
      for (let t = 1; t < 40; t++) y.push(0.7 * y[t - 1]! + g.normal())
      if (runsTest(y).pValue < 0.05) pow++
    }
    expect(rej / reps).toBeGreaterThan(0.03)
    expect(rej / reps).toBeLessThan(0.07)
    expect(pow / reps).toBeGreaterThan(0.8)
  })
})

describe('DataFrame methods (Tier 2 nonparametric)', () => {
  it('signTest / wilcoxon / mood / friedman / runsTest', async () => {
    const df = DataFrame.fromColumns({ x: ref.wilcoxon_exact.x })
    close(df.signTest('x', { median: 1.5 }).pValue, ref.sign.p['two-sided'], 12)
    close(df.wilcoxon('x', { median: 1.5 }).pValue, ref.wilcoxon_exact.p['two-sided'], 12)
    expect(df.runsTest('x').runs).toBe(runsTest(ref.wilcoxon_exact.x).runs)
    const m = ref.mood.groups as Record<string, number[]>
    const long = DataFrame.fromColumns({
      y: [...m.g1!, ...m.g2!, ...m.g3!],
      g: [...m.g1!.map(() => 'g1'), ...m.g2!.map(() => 'g2'), ...m.g3!.map(() => 'g3')],
    })
    close(long.mood('y', 'g').statistic, ref.mood.stat, 10)
    close((await long.lazy().mood('y', 'g')).pValue, ref.mood.p, 10)
    // friedman from long format: block × treatment
    const table = ref.friedman.table as number[][]
    const rows = table.flatMap((row, b) => row.map((v, t) => ({ v, b: `blk${b}`, t: 'ABCD'[t]! })))
    const fr = DataFrame.fromColumns({ v: rows.map((r) => r.v), b: rows.map((r) => r.b), t: rows.map((r) => r.t) })
    const f = fr.friedman('v', 't', 'b')
    close(f.statistic, ref.friedman.stat, 10)
    expect(f.treatments.map((t) => t.name)).toEqual(['A', 'B', 'C', 'D'])
    const dup = DataFrame.fromColumns({ v: [1, 2, 3], b: ['a', 'a', 'b'], t: ['x', 'x', 'y'] })
    expect(() => dup.friedman('v', 't', 'b')).toThrow(RangeError)
  })
})
