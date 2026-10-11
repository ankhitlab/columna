/**
 * Power and Sample Size (Minitab Stat › Power and Sample Size): 1-Sample Z, 1-Sample t, 2-Sample t,
 * Paired t, 1 Proportion, 2 Proportions, One-Way ANOVA, 1 Variance, 2 Variances.
 *
 * Give any two of {effect, n, power} and the third is solved: power by the noncentral t / F (or the
 * normal / χ² / F for the z, variance tests), sample size as the smallest integer n reaching the target
 * power, effect by bisection. `alpha` defaults to 0.05, alternatives to two-sided.
 */
import { chi2 as chi2Dist, f as fDist, ncfCdf, nctCdf, ncChi2Cdf, normal, t as tDist } from './dist.js'
import type { Alternative } from './tests.js'

const STD = normal()

export type PowerTest = '1-sample z' | '1-sample t' | '2-sample t' | 'paired t' | '1 proportion' | '2 proportions' | 'one-way anova' | '1 variance' | '2 variances' | 'chi-square'

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
  /** n2 = ceil(n * allocation); two independent samples only. Default 1. */
  allocation?: number
  /** Degrees of freedom for chi-square GOF / independence planning. */
  df?: number
  /** Branch for a two-sided effect search; not the alternative hypothesis. */
  direction?: 'increase' | 'decrease'
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
  n2?: number
  totalN: number
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
  if (alternative === 'greater') return nctCdf(-d.ppf(1 - alpha), df, -delta)
  if (alternative === 'less') return nctCdf(-d.ppf(1 - alpha), df, delta)
  const c = d.ppf(1 - alpha / 2)
  return nctCdf(-c, df, -delta) + nctCdf(-c, df, delta)
}

/** Power as a function of (effect, n) for each test. */
function powerOf(o: PowerOptions, effect: number, n: number): number {
  const alpha = o.alpha ?? 0.05
  const a = alt(o)
  const sigma = o.sigma ?? 1
  const n2 = Math.ceil(n * (o.allocation ?? 1))
  switch (o.test) {
    case '1-sample z':
      return zPower((effect / sigma) * Math.sqrt(n), alpha, a)
    case '1-sample t':
    case 'paired t':
      return tPower((effect / sigma) * Math.sqrt(n), n - 1, alpha, a)
    case '2-sample t':
      return tPower((effect / sigma) / Math.sqrt(1 / n + 1 / n2), n + n2 - 2, alpha, a)
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
      const pbar = (n * p1 + n2 * p2) / (n + n2)
      const se0 = Math.sqrt(pbar * (1 - pbar) * (1 / n + 1 / n2))
      const se1 = Math.sqrt((p1 * (1 - p1)) / n + (p2 * (1 - p2)) / n2)
      // Direction agrees with propTest2: first minus second.
      const shift = (p1 - p2) / se1
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
    case 'chi-square': {
      const df = o.df!
      return 1 - ncChi2Cdf(chi2Dist(df).isf(alpha), df, n * effect * effect)
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
      const d = fDist(df, n2 - 1)
      const r2 = effect * effect
      if (a === 'greater') return d.sf(d.ppf(1 - alpha) / r2)
      if (a === 'less') return d.cdf(d.ppf(alpha) / r2)
      return d.sf(d.ppf(1 - alpha / 2) / r2) + d.cdf(d.ppf(alpha / 2) / r2)
    }
  }
}

/** Supported workbench planning tests. Proportions use normal approximations, NOT exact binomial/Fisher power. */
export const POWER_TESTS: PowerTest[] = ['1-sample z', '1-sample t', '2-sample t', 'paired t', '1 proportion', '2 proportions', 'one-way anova', '1 variance', '2 variances', 'chi-square']
const twoSamples = (o: PowerOptions) => ['2-sample t', '2 proportions', '2 variances'].includes(o.test)
function minN(o: PowerOptions): number {
  const base = ['1-sample z', '1 proportion', '2 proportions', 'chi-square'].includes(o.test) ? 1 : 2
  return twoSamples(o) ? Math.max(base, Math.floor((base - 1) / (o.allocation ?? 1)) + 1) : base
}
function validated(o: PowerOptions): void {
  if (!POWER_TESTS.includes(o.test)) throw new RangeError('power: unknown test')
  alt(o)
  if (!((o.alpha ?? .05) > 0 && (o.alpha ?? .05) < 1)) throw new RangeError('power: alpha must be in (0, 1)')
  if (!(Number.isFinite(o.sigma ?? 1) && (o.sigma ?? 1) > 0)) throw new RangeError('power: sigma must be finite and positive')
  if (!(Number.isFinite(o.allocation ?? 1) && (o.allocation ?? 1) > 0)) throw new RangeError('power: allocation must be finite and positive')
  if (o.direction !== undefined && !['increase', 'decrease'].includes(o.direction)) throw new RangeError('power: unknown effect direction')
  if (o.test === '1 proportion' && !(o.p0! > 0 && o.p0! < 1)) throw new RangeError('power: p0 must be in (0, 1)')
  if (o.test === '2 proportions' && !(o.p1! > 0 && o.p1! < 1)) throw new RangeError('power: p1 must be in (0, 1)')
  if (o.test === 'one-way anova' && !(Number.isSafeInteger(o.groups) && o.groups! >= 2)) throw new RangeError('power: integer groups >= 2 required')
  if (o.test === 'chi-square' && !(Number.isSafeInteger(o.df) && o.df! >= 1)) throw new RangeError('power: positive integer df required')
  if (['one-way anova', 'chi-square'].includes(o.test) && alt(o) !== 'two-sided') throw new RangeError('power: this omnibus test has no directional alternative')
  if (o.effect !== undefined) {
    if (!Number.isFinite(o.effect)) throw new RangeError('power: effect must be finite')
    if (['1 proportion', '2 proportions'].includes(o.test) && !(o.effect > 0 && o.effect < 1)) throw new RangeError('power: alternative proportion must be in (0, 1)')
    if (['1 variance', '2 variances'].includes(o.test) && !(o.effect > 0)) throw new RangeError('power: sigma ratio must be positive')
    if (['chi-square', 'one-way anova'].includes(o.test) && o.effect < 0) throw new RangeError('power: effect cannot be negative')
  }
  if (o.n !== undefined && !(Number.isSafeInteger(o.n) && o.n >= minN(o))) throw new RangeError(`power: n must be an integer >= ${minN(o)}`)
  if (o.power !== undefined && !(o.power > 0 && o.power < 1)) throw new RangeError('power: target must be in (0, 1)')
  if (o.maxN !== undefined && !(Number.isSafeInteger(o.maxN) && o.maxN >= minN(o))) throw new RangeError('power: maxN must be a valid integer search budget')
}
/** Solve for exactly one of power, n, effect. n is the first-group size, not total N. */
export function power(o: PowerOptions): PowerResult {
  validated(o)
  const missing = (['effect', 'n', 'power'] as const).filter(k => o[k] === undefined)
  if (missing.length !== 1) throw new RangeError('power: give exactly two of effect, n, power')
  const solvedFor = missing[0]!, alpha = o.alpha ?? .05, alternative = alt(o)
  const evaluate = (effect: number, n: number) => {
    if (twoSamples(o) && !Number.isSafeInteger(Math.ceil(n * (o.allocation ?? 1)))) throw new RangeError('power: second-group sample size is not safely representable')
    const p = powerOf(o, effect, n)
    if (!Number.isFinite(p) || p < -1e-7 || p > 1 + 1e-7) throw new RangeError('power: numerical probability failed; change the parameter range')
    return Math.min(1, Math.max(0, p))
  }
  const result = (effect: number, n: number): PowerResult => {
    const n2 = twoSamples(o) ? Math.ceil(n * (o.allocation ?? 1)) : undefined
    const totalN = n2 !== undefined ? n + n2 : o.test === 'one-way anova' ? n * o.groups! : n
    if (!Number.isSafeInteger(totalN)) throw new RangeError('power: total N is not safely representable')
    return {test:o.test, effect, n, n2, totalN, power:evaluate(effect,n), alpha, alternative, solvedFor}
  }
  if (solvedFor === 'power') return result(o.effect!, o.n!)
  const target = o.power!
  const nullEffect = o.test === '1 proportion' ? o.p0! : o.test === '2 proportions' ? o.p1! : ['1 variance','2 variances'].includes(o.test) ? 1 : 0
  if (solvedFor === 'n') {
    const effect = o.effect!, signed = (o.test === '2 proportions' ? -1 : 1) * (effect-nullEffect)
    if (target > alpha && (signed === 0 || alternative === 'greater' && signed < 0 || alternative === 'less' && signed > 0)) throw new RangeError('power: effect must be away from H0 in the chosen alternative direction to solve n')
    let lo = minN(o), hi = lo
    if (evaluate(effect,lo) >= target) return result(effect,lo)
    const maxN = o.maxN ?? 1e6
    while (hi < maxN && evaluate(effect,hi) < target) {lo=hi;hi=Math.min(maxN,hi*2)}
    if (evaluate(effect,hi) < target) throw new RangeError(`power: target not reached within search budget n <= ${maxN}; increase maxN or the effect`)
    while (hi-lo>1) {const mid=Math.floor((hi+lo)/2);if(evaluate(effect,mid)>=target)hi=mid;else lo=mid}
    return result(effect,hi)
  }
  const n=o.n!, ratio=['1 variance','2 variances'].includes(o.test), prop=['1 proportion','2 proportions'].includes(o.test)
  if (target <= evaluate(nullEffect,n)+1e-12) throw new RangeError('power: effect search requires target power greater than power under H0')
  let upward=alternative==='less'?false:alternative==='greater'?true:o.direction!=='decrease'
  if (o.test==='2 proportions' && alternative!=='two-sided') upward=!upward
  if (['chi-square','one-way anova'].includes(o.test)) upward=true
  let lo=nullEffect, hi=prop?(upward?1-1e-10:1e-10):ratio?(upward?2:.5):(upward?1:-1)*(o.sigma??1)
  let iterations=0
  while(!prop&&evaluate(hi,n)<target&&iterations++<60)hi=ratio?(upward?hi*2:hi/2):hi*2
  if(evaluate(hi,n)<target)throw new RangeError('power: target cannot be reached on the selected effect branch')
  for(let i=0;i<100;i++){
    const mid=ratio?Math.sqrt(lo*hi):(lo+hi)/2
    if(evaluate(mid,n)<target)lo=mid;else hi=mid
    if(Math.abs(hi-lo)<=1e-11*Math.max(1,Math.abs(hi)))break
  }
  return result(hi,n)
}

/** Plot-ready power curve, evaluated by the same Columna power engine. No observed/post-hoc power. */
export function powerStudy(options: PowerOptions) {
  const solution=power(options), lo=minN(options), top=Math.max(lo+1,Math.ceil(solution.n*1.8))
  const ns=[...new Set([lo,solution.n,...Array.from({length:41},(_,i)=>Math.max(lo,Math.round(lo+(top-lo)*i/40)))])].sort((a,b)=>a-b)
  const curve=ns.map(n=>power({...options,effect:solution.effect,n,power:undefined}))
  const previous=solution.solvedFor==='n'&&solution.n>lo?power({...options,effect:solution.effect,n:solution.n-1,power:undefined}):undefined
  return {solution,curve,previous}
}
