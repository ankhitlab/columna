/**
 * Normality tests — the three in Minitab's Normality Test dialog plus Shapiro–Wilk:
 *   Anderson–Darling   (Minitab's default; p-value by D'Agostino & Stephens 1986)
 *   Ryan–Joiner        (correlation with normal scores; Minitab's critical-value approximations)
 *   Kolmogorov–Smirnov (Lilliefors form — parameters estimated; p-value by Dallal & Wilkinson 1986)
 *   Shapiro–Wilk       (Royston's AS R94, the algorithm behind R `shapiro.test` and scipy `shapiro`)
 */
import { chi2 as chi2Dist, normal } from './dist.js'
import { cleanNumbers } from './tests.js'

export type NormalityMethod =
  | 'anderson-darling'
  | 'ryan-joiner'
  | 'kolmogorov-smirnov'
  | 'shapiro-wilk'
  | 'jarque-bera'
  | 'dagostino'
  | 'cramer-von-mises'

export interface NormalityResult {
  test:
    | 'Anderson-Darling'
    | 'Ryan-Joiner'
    | 'Kolmogorov-Smirnov'
    | 'Shapiro-Wilk'
    | 'Jarque-Bera'
    | "D'Agostino K²"
    | 'Cramer-von-Mises'
  /** A² (unadjusted, as scipy reports it), R (Ryan–Joiner), D (Kolmogorov–Smirnov) or W. */
  statistic: number
  /** Anderson–Darling only: A*² = A²·(1 + 0.75/n + 2.25/n²), the statistic the p-value is based on. */
  adjusted?: number
  pValue: number
  /**
   * Ryan–Joiner only: Minitab reports the p-value as a bound outside the tabulated range
   * ('> 0.100' / '< 0.010'); `pValue` then carries an extrapolated point estimate.
   */
  pBound?: '> 0.100' | '< 0.010'
  /** Ryan–Joiner only: critical values of R at α = 0.10, 0.05, 0.01 for this n. */
  critical?: { 0.1: number; 0.05: number; 0.01: number }
  n: number
  mean: number
  sd: number
}

const STD_NORMAL = normal()

function sortedClean(x: ArrayLike<number | null | undefined>, minN: number, name: string): { v: Float64Array; mean: number; sd: number } {
  const v = cleanNumbers(x).slice().sort()
  const n = v.length
  if (n < minN) throw new RangeError(`${name} needs at least ${minN} observations, got ${n}`)
  let s = 0
  for (let i = 0; i < n; i++) s += v[i]!
  const mean = s / n
  let m2 = 0
  for (let i = 0; i < n; i++) m2 += (v[i]! - mean) ** 2
  const sd = Math.sqrt(m2 / (n - 1))
  if (!(sd > 0)) throw new RangeError(`${name}: all observations are identical`)
  return { v, mean, sd }
}

/**
 * Anderson–Darling test for normality with mean and sd estimated from the sample.
 * Statistic A² = −n − (1/n)·Σ(2i−1)[ln F(z₍ᵢ₎) + ln(1 − F(z₍ₙ₊₁₋ᵢ₎))]; the p-value uses the
 * D'Agostino & Stephens (1986) approximation on A*² = A²(1 + 0.75/n + 2.25/n²) — the same as Minitab.
 */
export function andersonDarling(x: ArrayLike<number | null | undefined>): NormalityResult {
  const { v, mean, sd } = sortedClean(x, 8, 'andersonDarling')
  const n = v.length
  // ln F and ln(1 − F) evaluated on the near tail so extreme z keep precision
  const logCdf = new Float64Array(n)
  const logSf = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const z = (v[i]! - mean) / sd
    logCdf[i] = Math.log(STD_NORMAL.cdf(z))
    logSf[i] = Math.log(STD_NORMAL.sf(z))
  }
  let s = 0
  for (let i = 0; i < n; i++) s += (2 * i + 1) * (logCdf[i]! + logSf[n - 1 - i]!)
  const a2 = -n - s / n
  const adjusted = a2 * (1 + 0.75 / n + 2.25 / (n * n))
  let p: number
  if (adjusted >= 0.6) p = Math.exp(1.2937 - 5.709 * adjusted + 0.0186 * adjusted * adjusted)
  else if (adjusted >= 0.34) p = Math.exp(0.9177 - 4.279 * adjusted - 1.38 * adjusted * adjusted)
  else if (adjusted >= 0.2) p = 1 - Math.exp(-8.318 + 42.796 * adjusted - 59.938 * adjusted * adjusted)
  else p = 1 - Math.exp(-13.436 + 101.14 * adjusted - 223.73 * adjusted * adjusted)
  return { test: 'Anderson-Darling', statistic: a2, adjusted, pValue: Math.min(1, Math.max(0, p)), n, mean, sd }
}

// Royston (1992) polynomial coefficients (AS R94), as in R's swilk.c
const C1 = [0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056]
const C2 = [0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633]
const C3 = [0.544, -0.39978, 0.025054, -6.714e-4]
const C4 = [1.3822, -0.77857, 0.062767, -0.0020322]
const C5 = [-1.5861, -0.31082, -0.083751, 0.0038915]
const C6 = [-0.4803, -0.082676, 0.0030302]
const G = [-2.273, 0.459]
const poly = (c: number[], x: number) => c.reduce((acc, ci, i) => acc + ci * x ** i, 0)

/**
 * Shapiro–Wilk W test (Royston's approximation, valid for 3 ≤ n ≤ 5000; larger samples are accepted
 * but the p-value approximation degrades, like scipy). Values match R `shapiro.test` / scipy `shapiro`.
 */
export function shapiroWilk(x: ArrayLike<number | null | undefined>): NormalityResult {
  const { v, mean, sd } = sortedClean(x, 3, 'shapiroWilk')
  const n = v.length
  // normal scores m_i and the coefficient vector a
  const m = new Float64Array(n)
  let mm = 0
  for (let i = 0; i < n; i++) {
    m[i] = STD_NORMAL.ppf((i + 1 - 0.375) / (n + 0.25))
    mm += m[i]! * m[i]!
  }
  const a = new Float64Array(n)
  if (n === 3) {
    a[0] = -Math.SQRT1_2
    a[2] = Math.SQRT1_2
  } else {
    const u = 1 / Math.sqrt(n)
    const rsn = 1 / Math.sqrt(mm)
    const an = poly(C1, u) + m[n - 1]! * rsn
    let eps: number
    if (n > 5) {
      const an1 = poly(C2, u) + m[n - 2]! * rsn
      eps = (mm - 2 * m[n - 1]! ** 2 - 2 * m[n - 2]! ** 2) / (1 - 2 * an * an - 2 * an1 * an1)
      a[n - 1] = an
      a[0] = -an
      a[n - 2] = an1
      a[1] = -an1
      const se = Math.sqrt(eps)
      for (let i = 2; i < n - 2; i++) a[i] = m[i]! / se
    } else {
      eps = (mm - 2 * m[n - 1]! ** 2) / (1 - 2 * an * an)
      a[n - 1] = an
      a[0] = -an
      const se = Math.sqrt(eps)
      for (let i = 1; i < n - 1; i++) a[i] = m[i]! / se
    }
  }
  let num = 0
  let ssq = 0
  for (let i = 0; i < n; i++) {
    num += a[i]! * v[i]!
    ssq += (v[i]! - mean) ** 2
  }
  const w = Math.min(1, (num * num) / ssq)
  let p: number
  if (n === 3) {
    p = Math.max(0, (6 / Math.PI) * (Math.asin(Math.sqrt(w)) - Math.asin(Math.sqrt(0.75))))
  } else if (n <= 11) {
    const gamma = poly(G, n)
    const mu = poly(C3, n)
    const sigma = Math.exp(poly(C4, n))
    const y = -Math.log(gamma - Math.log1p(-w))
    p = STD_NORMAL.sf((y - mu) / sigma)
  } else {
    const ln = Math.log(n)
    const mu = poly(C5, ln)
    const sigma = Math.exp(poly(C6, ln))
    p = STD_NORMAL.sf((Math.log1p(-w) - mu) / sigma)
  }
  return { test: 'Shapiro-Wilk', statistic: w, pValue: p, n, mean, sd }
}

/**
 * Ryan–Joiner test: R = correlation between the ordered data and the normal scores
 * b₍ᵢ₎ = Φ⁻¹((i − 3/8)/(n + 1/4)). The p-value is derived from Minitab's critical-value
 * approximations (Ryan & Joiner 1976) at α = 0.10, 0.05, 0.01 by log-linear interpolation in α;
 * outside that range Minitab only reports a bound, which `pBound` reproduces.
 */
/**
 * Critical values of the Ryan–Joiner R. Minitab's published approximations (Ryan & Joiner 1976)
 * hold up to n ≈ 100 and exceed 1 beyond n ≈ 300, so above 100 the tail 1 − R is continued with a
 * power law (1 − R ∝ n^−0.91) anchored at n = 100 — within ~5 % of Monte-Carlo quantiles at n = 300…5000.
 */
function rjCritical(n: number): { 0.1: number; 0.05: number; 0.01: number } {
  const approx = (m: number) => {
    const rn = Math.sqrt(m)
    return {
      0.1: 1.0071 - 0.1371 / rn - 0.3682 / m + 0.778 / (m * m),
      0.05: 1.0063 - 0.1288 / rn - 0.6118 / m + 1.3505 / (m * m),
      0.01: 0.9963 - 0.0211 / rn - 1.4106 / m + 3.1791 / (m * m),
    }
  }
  if (n <= 100) return approx(n)
  const base = approx(100)
  const scale = Math.pow(n / 100, -0.91)
  return { 0.1: 1 - (1 - base[0.1]) * scale, 0.05: 1 - (1 - base[0.05]) * scale, 0.01: 1 - (1 - base[0.01]) * scale }
}

export function ryanJoiner(x: ArrayLike<number | null | undefined>): NormalityResult {
  const { v, mean, sd } = sortedClean(x, 4, 'ryanJoiner')
  const n = v.length
  let sxb = 0
  let sbb = 0
  for (let i = 0; i < n; i++) {
    const b = STD_NORMAL.ppf((i + 1 - 0.375) / (n + 0.25))
    sxb += (v[i]! - mean) * b
    sbb += b * b
  }
  const r = sxb / Math.sqrt(sbb * (n - 1) * sd * sd)
  const critical = rjCritical(n)
  // log(α) is close to linear in R between the tabulated points; extrapolate with the nearest segment
  const interp = (r0: number, a0: number, r1: number, a1: number) =>
    Math.exp(Math.log(a0) + ((r - r0) * (Math.log(a1) - Math.log(a0))) / (r1 - r0))
  let p: number
  let pBound: NormalityResult['pBound']
  if (r >= critical[0.1]) {
    p = Math.min(1, interp(critical[0.05], 0.05, critical[0.1], 0.1))
    pBound = '> 0.100'
  } else if (r >= critical[0.05]) p = interp(critical[0.05], 0.05, critical[0.1], 0.1)
  else if (r >= critical[0.01]) p = interp(critical[0.01], 0.01, critical[0.05], 0.05)
  else {
    p = Math.max(0, interp(critical[0.01], 0.01, critical[0.05], 0.05))
    pBound = '< 0.010'
  }
  return { test: 'Ryan-Joiner', statistic: r, pValue: p, pBound, critical, n, mean, sd }
}

/**
 * Kolmogorov–Smirnov test for normality with mean and sd estimated from the sample (Lilliefors).
 * D = max over the sample of |F̂(x) − Φ(z)| using both step sides; the p-value follows the
 * Dallal & Wilkinson (1986) approximation as implemented in R's nortest::lillie.test (Minitab's
 * KS normality test uses the same estimated-parameter statistic).
 */
export function kolmogorovSmirnov(x: ArrayLike<number | null | undefined>): NormalityResult {
  const { v, mean, sd } = sortedClean(x, 4, 'kolmogorovSmirnov')
  const n = v.length
  let d = 0
  for (let i = 0; i < n; i++) {
    const f = STD_NORMAL.cdf((v[i]! - mean) / sd)
    const dPlus = (i + 1) / n - f
    const dMinus = f - i / n
    if (dPlus > d) d = dPlus
    if (dMinus > d) d = dMinus
  }
  // Dallal–Wilkinson: samples above 100 are mapped onto n = 100
  let kd = d
  let nd = n
  if (n > 100) {
    kd = d * Math.pow(n / 100, 0.49)
    nd = 100
  }
  let p = Math.exp(
    -7.01256 * kd * kd * (nd + 2.78019) + 2.99587 * kd * Math.sqrt(nd + 2.78019) - 0.122119 + 0.974598 / Math.sqrt(nd) + 1.67997 / nd,
  )
  if (p > 0.1) {
    // Stephens' polynomial for the upper range, on the modified statistic
    const kk = (Math.sqrt(n) - 0.01 + 0.85 / Math.sqrt(n)) * d
    if (kk <= 0.302) p = 1
    else if (kk <= 0.5) p = 2.76773 - 19.828315 * kk + 80.709644 * kk ** 2 - 138.55152 * kk ** 3 + 81.218 * kk ** 4
    else if (kk <= 0.9) p = -4.901232 + 40.662806 * kk - 97.490286 * kk ** 2 + 94.029866 * kk ** 3 - 32.355711 * kk ** 4
    else if (kk <= 1.31) p = 6.198765 - 19.558097 * kk + 23.186922 * kk ** 2 - 12.193892 * kk ** 3 + 2.334262 * kk ** 4
    else p = 0
  }
  return { test: 'Kolmogorov-Smirnov', statistic: d, pValue: Math.min(1, Math.max(0, p)), n, mean, sd }
}

function skewKurt(v: Float64Array): { mean: number; sd: number; skew: number; kurt: number } {
  const n = v.length
  let s = 0
  for (let i = 0; i < n; i++) s += v[i]!
  const mean = s / n
  let m2 = 0
  let m3 = 0
  let m4 = 0
  for (let i = 0; i < n; i++) {
    const d = v[i]! - mean
    const d2 = d * d
    m2 += d2
    m3 += d2 * d
    m4 += d2 * d2
  }
  const sd = Math.sqrt(m2 / n)
  const skew = sd > 0 ? m3 / n / sd ** 3 : 0
  const kurt = sd > 0 ? m4 / n / sd ** 4 - 3 : 0
  return { mean, sd: Math.sqrt(m2 / Math.max(1, n - 1)), skew, kurt }
}

/** Jarque–Bera normality test (scipy jarque_bera). */
export function jarqueBera(x: ArrayLike<number | null | undefined>): NormalityResult {
  const v = cleanNumbers(x)
  const n = v.length
  if (n < 8) throw new RangeError(`jarqueBera needs at least 8 observations, got ${n}`)
  const { mean, sd, skew, kurt } = skewKurt(v)
  const jb = (n / 6) * (skew * skew + 0.25 * kurt * kurt)
  const pValue = chi2Dist(2).sf(jb)
  return { test: 'Jarque-Bera', statistic: jb, pValue, n, mean, sd }
}

/** D'Agostino–Pearson K² omnibus normality test (scipy normaltest). */
export function dagostinoK2(x: ArrayLike<number | null | undefined>): NormalityResult {
  const v = cleanNumbers(x)
  const n = v.length
  if (n < 20) throw new RangeError(`dagostinoK2 needs at least 20 observations, got ${n}`)
  const { mean, sd, skew, kurt } = skewKurt(v)
  // Transform skewness (D'Agostino 1970)
  const y = skew * Math.sqrt(((n + 1) * (n + 3)) / (6 * (n - 2)))
  const beta2 = (3 * (n * n + 27 * n - 70) * (n + 1) * (n + 3)) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
  const w2 = -1 + Math.sqrt(2 * (beta2 - 1))
  const delta = 1 / Math.sqrt(Math.log(Math.sqrt(w2)))
  const alpha = Math.sqrt(2 / (w2 - 1))
  const zSkew = delta * Math.log(y / alpha + Math.sqrt((y / alpha) ** 2 + 1))
  // Kurtosis transform (Anscombe–Glynn)
  const E = (3 * (n - 1)) / (n + 1)
  const varK = (24 * n * (n - 2) * (n - 3)) / ((n + 1) ** 2 * (n + 3) * (n + 5))
  const xk = (kurt - E) / Math.sqrt(varK)
  const beta1 = (6 * (n * n - 5 * n + 2)) / ((n + 7) * (n + 9)) * Math.sqrt((6 * (n + 3) * (n + 5)) / (n * (n - 2) * (n - 3)))
  const A = 6 + 8 / beta1 * (2 / beta1 + Math.sqrt(1 + 4 / (beta1 * beta1)))
  const zKurtRaw = (1 - 2 / A) / (1 + xk * Math.sqrt(2 / (A - 4)))
  const zKurt =
    zKurtRaw > 0
      ? Math.sqrt((9 * A) / 2) * (1 - 2 / (9 * A) - Math.pow(zKurtRaw, 1 / 3))
      : xk
  const k2 = zSkew * zSkew + zKurt * zKurt
  const pValue = chi2Dist(2).sf(Math.max(0, k2))
  return { test: "D'Agostino K²", statistic: Number.isFinite(k2) ? k2 : 0, pValue: Number.isFinite(pValue) ? pValue : 1, n, mean, sd }
}

/** Cramér–von Mises normality test (estimated parameters; Stephens modification). */
export function cramerVonMises(x: ArrayLike<number | null | undefined>): NormalityResult {
  const { v, mean, sd } = sortedClean(x, 8, 'cramerVonMises')
  const n = v.length
  let w2 = 1 / (12 * n)
  for (let i = 0; i < n; i++) {
    const zi = STD_NORMAL.cdf((v[i]! - mean) / sd)
    w2 += ((2 * i + 1) / (2 * n) - zi) ** 2
  }
  // Stephens modified W²*
  const wStar = w2 * (1 + 0.5 / n)
  // Approximate p via Stephens table interpolation
  let p: number
  if (wStar < 0.0275) p = 1 - Math.exp(-13.953 + 775.5 * wStar - 12542 * wStar * wStar)
  else if (wStar < 0.051) p = 1 - Math.exp(-5.903 + 179.55 * wStar - 1515 * wStar * wStar)
  else if (wStar < 0.092) p = Math.exp(0.9206 - 70.54 * wStar + 108.4 * wStar * wStar)
  else p = Math.exp(0.718 - 29.04 * wStar)
  return { test: 'Cramer-von-Mises', statistic: w2, adjusted: wStar, pValue: Math.min(1, Math.max(0, p)), n, mean, sd }
}

/** Dispatch by name; default Anderson–Darling (Minitab's default). */
export function normalityTest(x: ArrayLike<number | null | undefined>, method: NormalityMethod = 'anderson-darling'): NormalityResult {
  switch (method) {
    case 'shapiro-wilk':
      return shapiroWilk(x)
    case 'ryan-joiner':
      return ryanJoiner(x)
    case 'kolmogorov-smirnov':
      return kolmogorovSmirnov(x)
    case 'jarque-bera':
      return jarqueBera(x)
    case 'dagostino':
      return dagostinoK2(x)
    case 'cramer-von-mises':
      return cramerVonMises(x)
    default:
      return andersonDarling(x)
  }
}
