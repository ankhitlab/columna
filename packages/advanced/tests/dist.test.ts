import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { dist, erf, erfc, lgamma } from '@columna/advanced'

// Reference values from scipy 1.14 (see scratch script dist_ref.py); regenerate if the grid changes.
type Row = Record<string, number>
const ref = JSON.parse(readFileSync(new URL('./fixtures/dist-scipy.json', import.meta.url), 'utf8')) as Record<string, Row[]>

/** Relative error with an absolute floor so that values near 0 (deep tails) are judged sensibly. */
const relErr = (got: number, want: number) => {
  if (Math.abs(want) < 1e-15) return Math.abs(got - want) // quantile at the median is 0 up to noise
  if (!Number.isFinite(want)) return got === want ? 0 : Infinity
  return Math.abs(got - want) / Math.abs(want)
}

function checkGrid(name: string, rows: Row[], make: (r: Row) => ReturnType<typeof dist.normal>, tolCdf: number, tolPpf: number) {
  let worstCdf = 0
  let worstPpf = 0
  for (const r of rows) {
    const d = make(r)
    if ('x' in r) {
      for (const fn of ['pdf', 'cdf', 'sf'] as const) {
        const got = d[fn](r.x!)
        const want = r[fn]!
        // scipy underflows to 0 in the far tails; anything below 1e-300 is "zero" for both
        if (want < 1e-300 && got < 1e-300) continue
        const e = relErr(got, want)
        if (e > tolCdf) throw new Error(`${name} ${fn}(${r.x}) [${JSON.stringify(r)}]: got ${got}, want ${want}, rel ${e}`)
        worstCdf = Math.max(worstCdf, e)
      }
    } else {
      for (const fn of ['ppf', 'isf'] as const) {
        const got = d[fn](r.p!)
        const want = r[fn]!
        // scipy's own inverses are only ~1e-11 accurate (and off by 1e-3 in 1e-15 tails for F), so
        // compare loosely against them away from the extreme tails …
        const e = r.p! >= 1e-8 && r.p! <= 1 - 1e-8 ? relErr(got, want) : 0
        if (e > tolPpf) throw new Error(`${name} ${fn}(${r.p}) [${JSON.stringify(r)}]: got ${got}, want ${want}, rel ${e}`)
        // … and strictly via the round trip against our own (scipy-verified) cdf / sf, always on the
        // well-conditioned side: the near tail with the exact 1 − p above the median
        const pp = r.p!
        const useCdf = (fn === 'ppf') === pp <= 0.5
        const back = useCdf ? d.cdf(got) : d.sf(got)
        const rt = relErr(back, pp <= 0.5 ? pp : 1 - pp)
        if (rt > 1e-12) throw new Error(`${name} round trip ${fn}(${r.p}) → ${got} → ${back} [${JSON.stringify(r)}], rel ${rt}`)
        worstPpf = Math.max(worstPpf, rt)
      }
    }
  }
  return { worstCdf, worstPpf }
}

describe('dist: special functions', () => {
  it('lgamma / erf / erfc match scipy', () => {
    for (const r of ref.special!) {
      const got = r.fn === 'lgamma' ? lgamma(r.x!) : r.fn === 'erf' ? erf(r.x!) : erfc(r.x!)
      expect(relErr(got, r.v!)).toBeLessThan(1e-13)
    }
    expect(lgamma(1)).toBeCloseTo(0, 14)
    expect(lgamma(5)).toBeCloseTo(Math.log(24), 13)
  })
})

describe('dist: normal', () => {
  it('pdf/cdf/sf/ppf/isf match scipy on a 3-parameter grid incl. far tails', () => {
    const w = checkGrid('normal', ref.normal!, (r) => dist.normal(r.mu, r.sd), 1e-13, 1e-9)
    expect(w.worstCdf).toBeLessThan(1e-13)
    expect(w.worstPpf).toBeLessThan(1e-12)
  })
  it('classic values and round trips', () => {
    const n = dist.normal()
    expect(n.cdf(1.959963984540054)).toBeCloseTo(0.975, 14)
    expect(n.ppf(0.975)).toBeCloseTo(1.959963984540054, 13)
    expect(n.sf(6)).toBeCloseTo(9.865876450376946e-10, 20)
    expect(n.ppf(0)).toBe(-Infinity)
    expect(n.ppf(1)).toBe(Infinity)
    expect(n.ppf(1.5)).toBeNaN()
    for (const p of [1e-12, 0.001, 0.2, 0.5, 0.8, 0.999, 1 - 1e-12]) expect(n.cdf(n.ppf(p))).toBeCloseTo(p, 13)
    expect(Array.from(dist.normal(10, 2).map('cdf', [8, 10, 12]))).toEqual([n.cdf(-1), 0.5, n.cdf(1)])
    expect(() => dist.normal(0, 0)).toThrow(RangeError)
  })
})

describe('dist: t', () => {
  it('matches scipy for df 0.5 … 1000', () => {
    const w = checkGrid('t', ref.t!, (r) => dist.t(r.df!), 1e-11, 1e-9) // df = 1000 → betainc(500, ½) is ~1e-12
    expect(w.worstCdf).toBeLessThan(1e-11)
    expect(w.worstPpf).toBeLessThan(1e-12)
  })
  it('symmetry, limits and moments', () => {
    const t10 = dist.t(10)
    expect(t10.cdf(-2) + t10.cdf(2)).toBeCloseTo(1, 15)
    expect(t10.ppf(0.5)).toBe(0)
    expect(t10.ppf(0.975)).toBeCloseTo(2.2281388519649385, 9)
    expect(t10.cdf(t10.ppf(0.975))).toBeCloseTo(0.975, 15)
    expect(dist.t(1).cdf(1)).toBeCloseTo(0.75, 14) // Cauchy
    expect(dist.t(1e6).cdf(1.96)).toBeCloseTo(dist.normal().cdf(1.96), 6)
    expect(t10.variance).toBeCloseTo(10 / 8, 15)
    expect(dist.t(1).mean).toBeNaN()
  })
})

describe('dist: chi2', () => {
  it('matches scipy for k 0.5 … 500', () => {
    const w = checkGrid('chi2', ref.chi2!, (r) => dist.chi2(r.k!), 1e-12, 1e-9)
    expect(w.worstCdf).toBeLessThan(1e-12)
    expect(w.worstPpf).toBeLessThan(1e-12)
  })
  it('critical values used in tables', () => {
    expect(dist.chi2(5).sf(11.07)).toBeCloseTo(0.05000961862240545, 13)
    expect(dist.chi2(1).ppf(0.95)).toBeCloseTo(3.841458820694124, 12)
    expect(dist.chi2(2).pdf(0)).toBe(0.5)
    expect(dist.chi2(4).mean).toBe(4)
    expect(dist.chi2(4).variance).toBe(8)
  })
})

describe('dist: F', () => {
  it('matches scipy for 9 (d1, d2) pairs', () => {
    const w = checkGrid('f', ref.f!, (r) => dist.f(r.d1!, r.d2!), 1e-12, 1e-9)
    expect(w.worstCdf).toBeLessThan(1e-12)
    expect(w.worstPpf).toBeLessThan(1e-12)
  })
  it('critical values and relations', () => {
    expect(dist.f(3, 10).cdf(3.708)).toBeCloseTo(0.9499912183954575, 13)
    expect(dist.f(3, 10).ppf(0.95)).toBeCloseTo(3.7082648189181947, 9)
    expect(dist.f(3, 10).cdf(dist.f(3, 10).ppf(0.95))).toBeCloseTo(0.95, 15)
    // F(1, ν) = t(ν)²: P(F ≤ x) = P(|T| ≤ √x)
    expect(dist.f(1, 12).cdf(4)).toBeCloseTo(dist.t(12).cdf(2) - dist.t(12).cdf(-2), 13)
    // F(d1, d2) and 1/F(d2, d1)
    expect(dist.f(4, 7).cdf(2)).toBeCloseTo(dist.f(7, 4).sf(0.5), 13)
    expect(dist.f(5, 10).mean).toBeCloseTo(10 / 8, 15)
  })
})
