/**
 * Power and Sample Size (Minitab Stat › Power and Sample Size): 1-Sample Z, 1-Sample t, 2-Sample t,
 * Paired t, 1 Proportion, 2 Proportions, One-Way ANOVA, 1 Variance, 2 Variances.
 *
 * Give any two of {effect, n, power} and the third is solved: power by the noncentral t / F (or the
 * normal / χ² / F for the z, variance tests), sample size as the smallest integer n reaching the target
 * power, effect by bisection. `alpha` defaults to 0.05, alternatives to two-sided.
 */
import { chi2 as chi2Dist, f as fDist, ncfCdf, nctCdf, normal, t as tDist } from './dist.js'
import type { Alternative } from './tests.js'

const STD = normal()

export type PowerTest = '1-sample z' | '1-sample t' | '2-sample t' | 'paired t' | '1 proportion' | '2 proportions' | 'one-way anova' | '1 variance' | '2 variances'

export interface PowerOptions {
  test: PowerTest
  /**
   * Effect size in the test's natural units: difference in means (z / t tests; with `sigma`),
   * alternative proportion (1 proportion; with `p0`) or second proportion (2 proportions; with `p1`),
   * maximum difference between group means (ANOVA; with `sigma`, `groups`), or the ratio σ/σ₀ (1 variance)
   * / σ₁/σ₂ (2 variances). Omit to solve for it.
   */
  effect?: number
  /** Sample size per group. Omit to solve for it. */
  n?: number
  /** Target power. Omit to compute it. */
  power?: number
  alpha?: number
  alternative?: Alternative
  /** Standard deviation for z / t / ANOVA (default 1 → effect in σ units). */
  sigma?: number
  /** 1 proportion: hypothesized p₀. 2 proportions: first-group proportion p₁. */
  p0?: number
  p1?: number
  /** One-way ANOVA: number of groups. */
  groups?: number
  /** Upper bound when solving for n (default 1e6). */
  maxN?: number
}

export interface PowerResult {
  test: PowerTest
  effect: number
  n: number
  power: number
  alpha: number
  alternative: Alternative
  /** Which quantity was solved for. */
  solvedFor: 'power' | 'n' | 'effect'
}

function alt(o: PowerOptions): Alternative {
  const a = o.alternative ?? 'two-sided'
  if (a !== 'two-sided' && a !== 'less' && a !== 'greater') throw new RangeError(`alternative must be 'two-sided' | 'less' | 'greater', got ${String(a)}`)
  return a
}

/** Power of a z-type test with shift δ (in se units): P(reject) under the alternative. */
function zPower(delta: number, alpha: number, alternative: Alternative): number {
  if (alternative === 'greater') return STD.sf(STD.ppf(1 - alpha) - delta)
  if (alternative === 'less') return STD.cdf(-STD.ppf(1 - alpha) - delta)
  const c = STD.ppf(1 - alpha / 2)
  return STD.sf(c - delta) + STD.cdf(-c - delta)
}

/** Power of a t-test with noncentrality δ and df. */
function tPower(delta: number, df: number, alpha: number, alternative: Alternative): number {
  const d = tDist(df)
  if (alternative === 'greater') return 1 - nctCdf(d.ppf(1 - alpha), df, delta)
  if (alternative === 'less') return nctCdf(-d.ppf(1 - alpha), df, delta)
  const c = d.ppf(1 - alpha / 2)
  return 1 - nctCdf(c, df, delta) + nctCdf(-c, df, delta)
}

/** Power as a function of (effect, n) for each test. */
function powerOf(o: PowerOptions, effect: number, n: number): number {
  const alpha = o.alpha ?? 0.05
  const a = alt(o)
  const sigma = o.sigma ?? 1
  switch (o.test) {
    case '1-sample z':
      return zPower((effect / sigma) * Math.sqrt(n), alpha, a)
    case '1-sample t':
    case 'paired t':
      return tPower((effect / sigma) * Math.sqrt(n), n - 1, alpha, a)
    case '2-sample t':
      return tPower((effect / sigma) * Math.sqrt(n / 2), 2 * n - 2, alpha, a)
    case '1 proportion': {
      const p0 = o.p0
      if (!(p0 !== undefined && p0 > 0 && p0 < 1)) throw new RangeError('1 proportion power needs p0 in (0, 1)')
      if (!(effect > 0 && effect < 1)) throw new RangeError('1 proportion: effect is the alternative proportion in (0, 1)')
      // normal approximation with the null se in the critical value and the alternative se in the power
      const se0 = Math.sqrt((p0 * (1 - p0)) / n)
      const se1 = Math.sqrt((effect * (1 - effect)) / n)
      const shift = (effect - p0) / se1
      const ratio = se0 / se1
      if (a === 'greater') return STD.sf(STD.ppf(1 - alpha) * ratio - shift)
      if (a === 'less') return STD.cdf(-STD.ppf(1 - alpha) * ratio - shift)
      const c = STD.ppf(1 - alpha / 2) * ratio
      return STD.sf(c - shift) + STD.cdf(-c - shift)
    }
    case '2 proportions': {
      const p1 = o.p1
      if (!(p1 !== undefined && p1 > 0 && p1 < 1)) throw new RangeError('2 proportions power needs p1 in (0, 1)')
      if (!(effect > 0 && effect < 1)) throw new RangeError('2 proportions: effect is the second proportion in (0, 1)')
      const p2 = effect
      const pbar = (p1 + p2) / 2
      const se0 = Math.sqrt((2 * pbar * (1 - pbar)) / n)
      const se1 = Math.sqrt((p1 * (1 - p1)) / n + (p2 * (1 - p2)) / n)
      const shift = (p2 - p1) / se1
      const ratio = se0 / se1
      if (a === 'greater') return STD.sf(STD.ppf(1 - alpha) * ratio - shift)
      if (a === 'less') return STD.cdf(-STD.ppf(1 - alpha) * ratio - shift)
      const c = STD.ppf(1 - alpha / 2) * ratio
      return STD.sf(c - shift) + STD.cdf(-c - shift)
    }
    case 'one-way anova': {
      const k = o.groups
      if (!(k !== undefined && k >= 2)) throw new RangeError('one-way anova power needs groups ≥ 2')
      // Minitab: maximum difference between means → λ = n·Δ²/(2σ²) (two extreme means, others at the grand mean)
      const lambda = (n * effect * effect) / (2 * sigma * sigma)
      const d1 = k - 1
      const d2 = k * (n - 1)
      return 1 - ncfCdf(fDist(d1, d2).ppf(1 - alpha), d1, d2, lambda)
    }
    case '1 variance': {
      // H0 σ = σ₀, alternative σ = effect·σ₀ (ratio); (n−1)s²/σ₀² ~ ratio²·χ²(n−1)
      if (!(effect > 0)) throw new RangeError('1 variance: effect is the ratio σ/σ₀ > 0')
      const df = n - 1
      const d = chi2Dist(df)
      const r2 = effect * effect
      if (a === 'greater') return d.sf(d.ppf(1 - alpha) / r2)
      if (a === 'less') return d.cdf(d.ppf(alpha) / r2)
      return d.sf(d.ppf(1 - alpha / 2) / r2) + d.cdf(d.ppf(alpha / 2) / r2)
    }
    case '2 variances': {
      if (!(effect > 0)) throw new RangeError('2 variances: effect is the ratio σ₁/σ₂ > 0')
      const df = n - 1
      const d = fDist(df, df)
      const r2 = effect * effect
      if (a === 'greater') return d.sf(d.ppf(1 - alpha) / r2)
      if (a === 'less') return d.cdf(d.ppf(alpha) / r2)
      return d.sf(d.ppf(1 - alpha / 2) / r2) + d.cdf(d.ppf(alpha / 2) / r2)
    }
  }
}

function minN(o: PowerOptions): number {
  return o.test === '1-sample z' ? 1 : o.test === 'one-way anova' || o.test === '1 variance' || o.test === '2 variances' || o.test === '2-sample t' ? 2 : 2
}

/**
 * Solve the power equation. Exactly one of `effect`, `n`, `power` must be omitted.
 *   power({ test: '2-sample t', effect: 1, n: 20, sigma: 1 })            → power
 *   power({ test: '1-sample t', effect: 0.5, power: 0.9 })              → n
 *   power({ test: 'one-way anova', groups: 4, n: 10, power: 0.8 })      → effect (max difference)
 */
export function power(o: PowerOptions): PowerResult {
  const alpha = o.alpha ?? 0.05
  if (!(alpha > 0 && alpha < 1)) throw new RangeError(`alpha must be in (0, 1), got ${alpha}`)
  const a = alt(o)
  const missing = (['effect', 'n', 'power'] as const).filter((k) => o[k] === undefined)
  if (missing.length !== 1) throw new RangeError(`power: give exactly two of effect, n, power (missing: ${missing.join(', ') || 'none'})`)
  const solvedFor = missing[0]!
  if (o.n !== undefined && !(Number.isInteger(o.n) && o.n >= minN(o))) throw new RangeError(`power: n must be an integer ≥ ${minN(o)}, got ${o.n}`)
  if (o.power !== undefined && !(o.power > 0 && o.power < 1)) throw new RangeError(`power: target power must be in (0, 1), got ${o.power}`)

  if (solvedFor === 'power') {
    return { test: o.test, effect: o.effect!, n: o.n!, power: powerOf(o, o.effect!, o.n!), alpha, alternative: a, solvedFor }
  }
  if (solvedFor === 'n') {
    const target = o.power!
    const maxN = o.maxN ?? 1e6
    let lo = minN(o)
    if (powerOf(o, o.effect!, lo) >= target) return { test: o.test, effect: o.effect!, n: lo, power: powerOf(o, o.effect!, lo), alpha, alternative: a, solvedFor }
    let hi = lo
    while (hi < maxN && powerOf(o, o.effect!, hi) < target) {
      lo = hi
      hi = Math.min(maxN, hi * 2)
    }
    if (powerOf(o, o.effect!, hi) < target) throw new RangeError(`power: target ${target} not reachable with n ≤ ${maxN}`)
    while (hi - lo > 1) {
      const mid = Math.floor((lo + hi) / 2)
      if (powerOf(o, o.effect!, mid) >= target) hi = mid
      else lo = mid
    }
    return { test: o.test, effect: o.effect!, n: hi, power: powerOf(o, o.effect!, hi), alpha, alternative: a, solvedFor }
  }
  // solve for effect: bisection on the monotone branch away from the null
  const target = o.power!
  const n = o.n!
  let lo: number
  let hi: number
  const ratioTest = o.test === '1 variance' || o.test === '2 variances'
  const propTest = o.test === '1 proportion' || o.test === '2 proportions'
  const nullValue = ratioTest ? 1 : propTest ? (o.test === '1 proportion' ? o.p0! : o.p1!) : 0
  const upward = a !== 'less'
  if (propTest) {
    lo = nullValue
    hi = upward ? 1 - 1e-9 : 1e-9
  } else if (ratioTest) {
    lo = 1
    hi = upward ? 1e6 : 1e-6
  } else {
    lo = 0
    hi = upward ? 1 : -1
    while (powerOf(o, hi, n) < target && Math.abs(hi) < 1e6) hi *= 2
  }
  for (let i = 0; i < 200; i++) {
    const mid = ratioTest ? Math.sqrt(lo * hi) : 0.5 * (lo + hi)
    if (powerOf(o, mid, n) < target) lo = mid
    else hi = mid
    if (Math.abs(hi - lo) < 1e-10 * Math.max(1, Math.abs(hi))) break
  }
  const effect = ratioTest ? Math.sqrt(lo * hi) : 0.5 * (lo + hi)
  return { test: o.test, effect, n, power: powerOf(o, effect, n), alpha, alternative: a, solvedFor }
}
