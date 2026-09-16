/**
 * Equivalence tests (Minitab Stat › Equivalence Tests): two one-sided t-tests (TOST) against the
 * equivalence limits [lower, upper] for a one-sample mean, a two-sample difference (Welch or pooled)
 * and a paired difference. Equivalence is claimed when both one-sided tests reject, i.e. when the
 * 100(1 − 2α)% confidence interval lies inside the limits.
 */
import { t as tDist } from './dist.js'
import { cleanNumbers } from './tests.js'

export interface TostResult {
  test: 'equivalence (one-sample)' | 'equivalence (two-sample, Welch)' | 'equivalence (two-sample, pooled)' | 'equivalence (paired)'
  /** Mean, or difference of means (test − reference). */
  estimate: number
  se: number
  df: number
  limits: [number, number]
  /** t statistics against the lower and upper limits. */
  tLower: number
  tUpper: number
  /** One-sided p-values; `pValue` is their maximum (the TOST p-value). */
  pLower: number
  pUpper: number
  pValue: number
  alpha: number
  /** 100(1 − 2α)% confidence interval for the estimate (Minitab's equivalence CI). */
  ci: [number, number]
  equivalent: boolean
  n: number
}

function moments(v: Float64Array) {
  const n = v.length
  let s = 0
  for (let i = 0; i < n; i++) s += v[i]!
  const mean = s / n
  let m2 = 0
  for (let i = 0; i < n; i++) m2 += (v[i]! - mean) ** 2
  const variance = n > 1 ? m2 / (n - 1) : NaN
  return { n, mean, variance, sd: Math.sqrt(variance) }
}

function checkLimits(limits: [number, number], alpha: number): void {
  if (!(limits[0] < limits[1])) throw new RangeError(`equivalence limits must satisfy lower < upper, got [${limits[0]}, ${limits[1]}]`)
  if (!(alpha > 0 && alpha < 0.5)) throw new RangeError(`alpha must be in (0, 0.5), got ${alpha}`)
}

function tost(test: TostResult['test'], estimate: number, se: number, df: number, limits: [number, number], alpha: number, n: number): TostResult {
  const d = tDist(df)
  const tLower = (estimate - limits[0]) / se
  const tUpper = (estimate - limits[1]) / se
  const pLower = d.sf(tLower) // H0: mean ≤ lower  vs  H1: mean > lower
  const pUpper = d.cdf(tUpper) // H0: mean ≥ upper  vs  H1: mean < upper
  const h = d.ppf(1 - alpha) * se
  const ci: [number, number] = [estimate - h, estimate + h]
  const pValue = Math.max(pLower, pUpper)
  return { test, estimate, se, df, limits, tLower, tUpper, pLower, pUpper, pValue, alpha, ci, equivalent: pValue < alpha, n }
}

/** One-sample equivalence test of the mean against [lower, upper] (Minitab 1-Sample Equivalence). */
export function tost1(x: ArrayLike<number | null | undefined>, options: { limits: [number, number]; alpha?: number }): TostResult {
  const alpha = options.alpha ?? 0.05
  checkLimits(options.limits, alpha)
  const m = moments(cleanNumbers(x))
  if (m.n < 2) throw new RangeError(`tost1 needs at least 2 observations, got ${m.n}`)
  return tost('equivalence (one-sample)', m.mean, m.sd / Math.sqrt(m.n), m.n - 1, options.limits, alpha, m.n)
}

/** Two-sample equivalence test of mean(test) − mean(reference) (Minitab 2-Sample Equivalence; Welch by default). */
export function tost2(
  test: ArrayLike<number | null | undefined>,
  reference: ArrayLike<number | null | undefined>,
  options: { limits: [number, number]; alpha?: number; equalVar?: boolean },
): TostResult {
  const alpha = options.alpha ?? 0.05
  checkLimits(options.limits, alpha)
  const x = moments(cleanNumbers(test))
  const y = moments(cleanNumbers(reference))
  if (x.n < 2 || y.n < 2) throw new RangeError(`tost2 needs at least 2 observations per sample, got ${x.n} and ${y.n}`)
  let se: number
  let df: number
  if (options.equalVar) {
    const sp2 = ((x.n - 1) * x.variance + (y.n - 1) * y.variance) / (x.n + y.n - 2)
    se = Math.sqrt(sp2 * (1 / x.n + 1 / y.n))
    df = x.n + y.n - 2
  } else {
    const vx = x.variance / x.n
    const vy = y.variance / y.n
    se = Math.sqrt(vx + vy)
    df = ((vx + vy) * (vx + vy)) / ((vx * vx) / (x.n - 1) + (vy * vy) / (y.n - 1))
  }
  return tost(options.equalVar ? 'equivalence (two-sample, pooled)' : 'equivalence (two-sample, Welch)', x.mean - y.mean, se, df, options.limits, alpha, x.n + y.n)
}

/** Paired equivalence test on test − reference (Minitab Paired Equivalence). */
export function tostPaired(
  test: ArrayLike<number | null | undefined>,
  reference: ArrayLike<number | null | undefined>,
  options: { limits: [number, number]; alpha?: number },
): TostResult {
  if (test.length !== reference.length) throw new RangeError(`tostPaired needs equal-length samples, got ${test.length} and ${reference.length}`)
  const d: number[] = []
  for (let i = 0; i < test.length; i++) {
    const u = test[i]
    const w = reference[i]
    if (typeof u === 'number' && typeof w === 'number' && Number.isFinite(u) && Number.isFinite(w)) d.push(u - w)
  }
  const r = tost1(d, options)
  return { ...r, test: 'equivalence (paired)' }
}
