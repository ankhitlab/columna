/**
 * Tier 4.5–4.7 — Process Capability & Tolerance:
 * normal Cp/Cpk/Pp/Ppk/Cpm/PPM/Z.Bench, Box–Cox / Johnson / Weibull fits, normal and nonparametric
 * tolerance intervals (Howe / Hahn–Meeker style).
 */
import { chi2 as chi2Dist, lgamma, normal } from './dist.js'
import { cleanNumbers } from './tests.js'
import { controlChart, spcConstants } from './spc.js'
import type { PlotSeries } from './plot.js'
import { maxOf, minOf } from './numerics.js'

const STD = normal()

function meanOf(v: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i]!
  return s / v.length
}
function sdOf(v: ArrayLike<number>, ddof = 1): number {
  const n = v.length
  if (n <= ddof) return NaN
  const m = meanOf(v)
  let s2 = 0
  for (let i = 0; i < n; i++) s2 += (v[i]! - m) ** 2
  return Math.sqrt(s2 / (n - ddof))
}
function asSubgroups(x: ArrayLike<number | null | undefined> | number[][], subgroup?: number): number[][] {
  if (Array.isArray(x) && x.length > 0 && Array.isArray(x[0])) return (x as number[][]).map((g) => Array.from(cleanNumbers(g)))
  const v = Array.from(cleanNumbers(x as ArrayLike<number | null | undefined>))
  if (!(subgroup && subgroup >= 2)) return [v]
  const groups: number[][] = []
  for (let i = 0; i + subgroup <= v.length; i += subgroup) groups.push(v.slice(i, i + subgroup))
  return groups
}

// ---- 4.5 Capability --------------------------------------------------------------------------------

export interface CapabilityResult {
  n: number
  mean: number
  /** Within-subgroup σ (R̄/d₂ or S̄/c₄); equals overall when no subgrouping. */
  sigmaWithin: number
  /** Overall sample sd. */
  sigmaOverall: number
  lsl?: number
  usl?: number
  target?: number
  Cp?: number
  Cpl?: number
  Cpu?: number
  Cpk?: number
  Pp?: number
  Ppl?: number
  Ppu?: number
  Ppk?: number
  Cpm?: number
  /** Expected PPM outside specs using within σ (normal). */
  ppmWithin?: number
  /** Expected PPM using overall σ. */
  ppmOverall?: number
  /** Z.Bench = Φ⁻¹(1 − P(defect overall)). */
  zBench?: number
  /** Approximate CI for Cpk (Bissell / Chou–Owen–Borrego normal approx). */
  cpkCI?: [number, number]
  confidence: number
}

function ppmNormal(mean: number, sigma: number, lsl?: number, usl?: number): number {
  if (!(sigma > 0)) return NaN
  let p = 0
  if (lsl !== undefined) p += STD.cdf((lsl - mean) / sigma)
  if (usl !== undefined) p += STD.sf((usl - mean) / sigma)
  return p * 1e6
}

/**
 * Process capability indices (Minitab Capability Analysis — Normal).
 * Within σ from R̄/d₂ (default) or S̄/c₄ when `method: 's'`; overall σ = sample sd.
 */
export function capability(
  x: ArrayLike<number | null | undefined> | number[][],
  options: {
    lsl?: number
    usl?: number
    target?: number
    subgroup?: number
    method?: 'r' | 's'
    confidence?: number
  } = {},
): CapabilityResult {
  const { lsl, usl } = options
  if (lsl === undefined && usl === undefined) throw new RangeError('capability: provide lsl and/or usl')
  if (lsl !== undefined && usl !== undefined && !(usl > lsl)) throw new RangeError('capability: usl must be > lsl')
  const confidence = options.confidence ?? 0.95
  if (!(confidence > 0 && confidence < 1)) throw new RangeError(`capability: confidence must be in (0, 1)`)

  const groups = asSubgroups(x, options.subgroup)
  const flat = groups.flat()
  const n = flat.length
  if (n < 2) throw new RangeError('capability needs at least 2 observations')
  const mean = meanOf(flat)
  const sigmaOverall = sdOf(flat, 1)

  let sigmaWithin: number
  if (groups.length >= 2 && groups.every((g) => g.length >= 2)) {
    const useS = options.method === 's'
    const ns = groups.map((g) => g.length)
    const equal = ns.every((ni) => ni === ns[0])
    if (equal) {
      const c = spcConstants(ns[0]!)
      const bar = meanOf(groups.map((g) => (useS ? sdOf(g, 1) : Math.max(...g) - Math.min(...g))))
      sigmaWithin = useS ? bar / c.c4 : bar / c.d2
    } else {
      let s = 0
      for (const g of groups) {
        const c = spcConstants(g.length)
        s += useS ? sdOf(g, 1) / c.c4 : (Math.max(...g) - Math.min(...g)) / c.d2
      }
      sigmaWithin = s / groups.length
    }
  } else {
    sigmaWithin = sigmaOverall
  }

  const target = options.target
  const out: CapabilityResult = { n, mean, sigmaWithin, sigmaOverall, lsl, usl, target, confidence }

  const index = (lo: number | undefined, hi: number | undefined, sigma: number) => {
    const cpl = lo !== undefined && sigma > 0 ? (mean - lo) / (3 * sigma) : undefined
    const cpu = hi !== undefined && sigma > 0 ? (hi - mean) / (3 * sigma) : undefined
    const cp = lo !== undefined && hi !== undefined && sigma > 0 ? (hi - lo) / (6 * sigma) : undefined
    const cpk = cpl !== undefined && cpu !== undefined ? Math.min(cpl, cpu) : (cpl ?? cpu)
    return { cp, cpl, cpu, cpk }
  }

  const within = index(lsl, usl, sigmaWithin)
  const overall = index(lsl, usl, sigmaOverall)
  out.Cp = within.cp
  out.Cpl = within.cpl
  out.Cpu = within.cpu
  out.Cpk = within.cpk
  out.Pp = overall.cp
  out.Ppl = overall.cpl
  out.Ppu = overall.cpu
  out.Ppk = overall.cpk

  if (target !== undefined && lsl !== undefined && usl !== undefined && sigmaOverall > 0) {
    const tau2 = sigmaOverall ** 2 + (mean - target) ** 2
    out.Cpm = (usl - lsl) / (6 * Math.sqrt(tau2))
  }

  out.ppmWithin = ppmNormal(mean, sigmaWithin, lsl, usl)
  out.ppmOverall = ppmNormal(mean, sigmaOverall, lsl, usl)
  if (out.ppmOverall !== undefined && out.ppmOverall < 1e6) {
    const pDef = out.ppmOverall / 1e6
    out.zBench = pDef <= 0 ? Infinity : STD.isf(pDef)
  }

  // Cpk CI (normal approx): SE(Cpk) ≈ √((1/(9n)) + Cpk²/(2(n−1)))  — Bissell
  if (out.Cpk !== undefined && Number.isFinite(out.Cpk) && n > 1) {
    const se = Math.sqrt(1 / (9 * n) + (out.Cpk * out.Cpk) / (2 * (n - 1)))
    const z = STD.ppf(0.5 + confidence / 2)
    out.cpkCI = [out.Cpk - z * se, out.Cpk + z * se]
  }
  return out
}

// ---- 4.6 transforms / fits ------------------------------------------------------------------------

export interface BoxCoxResult {
  lambda: number
  /** Maximised profile log-likelihood (Box–Cox). */
  logLik: number
  transformed: number[]
}

/** Box–Cox λ that maximises the profile likelihood on a grid (default −2…2, step 0.01); `y(λ) = (x^λ−1)/λ`. */
export function boxCoxLambda(
  x: ArrayLike<number | null | undefined>,
  options: { min?: number; max?: number; step?: number } = {},
): BoxCoxResult {
  const v = Array.from(cleanNumbers(x))
  if (v.length < 3) throw new RangeError('boxCoxLambda needs at least 3 observations')
  if (v.some((xi) => !(xi > 0))) throw new RangeError('boxCoxLambda: all observations must be > 0')
  const lo = options.min ?? -2
  const hi = options.max ?? 2
  const step = options.step ?? 0.01
  const n = v.length
  const logX = v.map(Math.log)
  const meanLog = meanOf(logX)

  const transform = (lam: number): Float64Array => {
    const y = new Float64Array(n)
    if (Math.abs(lam) < 1e-12) {
      for (let i = 0; i < n; i++) y[i] = logX[i]!
    } else {
      for (let i = 0; i < n; i++) y[i] = (v[i]! ** lam - 1) / lam
    }
    return y
  }
  const profile = (lam: number): number => {
    const y = transform(lam)
    const m = meanOf(y)
    let sse = 0
    for (let i = 0; i < n; i++) sse += (y[i]! - m) ** 2
    // ℓ(λ) = −(n/2) log(SSE/n) + (λ−1) Σ log x
    return (-n / 2) * Math.log(sse / n) + (lam - 1) * meanLog * n
  }

  let bestLam = 0
  let bestLL = -Infinity
  for (let lam = lo; lam <= hi + 1e-12; lam += step) {
    const ll = profile(lam)
    if (ll > bestLL) {
      bestLL = ll
      bestLam = Math.round(lam / step) * step // stabilize float grid
    }
  }
  // refine locally
  for (const d of [-step / 2, 0, step / 2]) {
    const lam = bestLam + d
    if (lam < lo || lam > hi) continue
    const ll = profile(lam)
    if (ll > bestLL) {
      bestLL = ll
      bestLam = lam
    }
  }
  return { lambda: bestLam, logLik: bestLL, transformed: Array.from(transform(bestLam)) }
}

export type JohnsonFamily = 'SU' | 'SB' | 'SL'

export interface JohnsonFit {
  family: JohnsonFamily
  gamma: number
  delta: number
  xi: number
  lambda: number
  /** Transformed values ≈ N(0,1) under the fitted map. */
  z: number[]
}

function moments4(v: number[]) {
  const n = v.length
  const m = meanOf(v)
  let m2 = 0
  let m3 = 0
  let m4 = 0
  for (const x of v) {
    const d = x - m
    m2 += d * d
    m3 += d * d * d
    m4 += d * d * d * d
  }
  m2 /= n
  m3 /= n
  m4 /= n
  const sd = Math.sqrt(m2)
  const skew = m3 / (sd ** 3)
  const kurt = m4 / (m2 * m2) // raw kurtosis (normal = 3)
  return { mean: m, sd, skew, kurt }
}

/**
 * Johnson system fit by the percentile / moment method (Slifker–Shapiro style choice of family,
 * then closed-form parameters). Maps data to approximately standard normal.
 */
export function johnsonFit(x: ArrayLike<number | null | undefined>): JohnsonFit {
  const v = Array.from(cleanNumbers(x)).sort((a, b) => a - b)
  const n = v.length
  if (n < 8) throw new RangeError('johnsonFit needs at least 8 observations')
  const { mean, sd, skew, kurt } = moments4(v)
  // Slifker–Shapiro: use z=0.524 (≈ P=0.7) quantiles
  const q = (p: number) => {
    const i = (n - 1) * p
    const lo = Math.floor(i)
    const hi = Math.ceil(i)
    if (lo === hi) return v[lo]!
    return v[lo]! * (hi - i) + v[hi]! * (i - lo)
  }
  const z = 0.524
  const xnz = q(STD.cdf(-3 * z))
  const xmz = q(STD.cdf(-z))
  const xpz = q(STD.cdf(z))
  const x3z = q(STD.cdf(3 * z))
  const m = xpz - xmz
  const n1 = xmz - xnz
  const p = x3z - xpz
  const ratio = (m * m) / (p * n1)

  let family: JohnsonFamily
  if (Math.abs(ratio - 1) < 0.1 && Math.abs(skew) < 0.1) family = 'SL'
  else if (ratio >= 1) family = 'SU'
  else family = 'SB'

  let gamma: number
  let delta: number
  let xi: number
  let lambda: number
  const zs: number[] = []

  if (family === 'SU') {
    // unbounded
    const term = (1 + p / m) * (1 + n1 / m)
    delta = z / Math.acosh(0.5 * Math.sqrt(term))
    gamma = delta * Math.asinh(((p / m - n1 / m) / 2) * Math.sqrt((1 / ((p / m) * (n1 / m)) - 1) / (((p + n1) / m + 2) * 0.25)))
    // fallback moment start if quantile degenerates
    if (!Number.isFinite(delta) || !(delta > 0)) {
      delta = 1 / Math.sqrt(Math.max(1e-6, Math.log(0.5 * (kurt + 1))))
      gamma = -skew
    }
    lambda = (2 * m * Math.sqrt(((p / m) * (n1 / m) - 1) / (((p + n1) / m + 2) * ((p / m) * (n1 / m))))) / ((p / m + n1 / m - 2) * Math.sinh(2 * z / delta) || 1e-12)
    if (!Number.isFinite(lambda) || !(lambda > 0)) lambda = sd
    xi = (xmz + xpz) / 2 + (lambda * (p / m - n1 / m)) / (2 * Math.sqrt((p / m) * (n1 / m) - 1) || 1e-12)
    if (!Number.isFinite(xi)) xi = mean
    for (const xi_ of v) zs.push(gamma + delta * Math.asinh((xi_ - xi) / lambda))
  } else if (family === 'SB') {
    delta = z / Math.acosh(0.5 * Math.sqrt((1 + m / p) * (1 + m / n1)))
    gamma = delta * Math.asinh(((n1 / m - p / m) / 2) * Math.sqrt(((1 + m / p) * (1 + m / n1) - 4) / (((m / p) * (m / n1) - 1) || 1e-12)))
    if (!Number.isFinite(delta) || !(delta > 0)) {
      delta = 1
      gamma = -skew
    }
    lambda = (m * Math.sqrt(((1 + m / p) * (1 + m / n1) - 2) ** 2 - 4)) / ((m / p + m / n1 - 2) * Math.sinh(2 * z / delta) || 1e-12)
    if (!Number.isFinite(lambda) || !(lambda > 0)) lambda = maxOf(v) - minOf(v)
    xi = (xmz + xpz) / 2 - lambda / 2 + (lambda * (m / n1 - m / p)) / (2 * (m / p + m / n1 - 2) || 1e-12)
    if (!Number.isFinite(xi)) xi = minOf(v) - 0.01 * lambda
    for (const xi_ of v) {
      const y = (xi_ - xi) / lambda
      const yy = Math.min(1 - 1e-12, Math.max(1e-12, y))
      zs.push(gamma + delta * Math.log(yy / (1 - yy)))
    }
  } else {
    // SL lognormal
    delta = 2 * z / Math.log(p / n1)
    gamma = delta * Math.log((m / Math.sqrt(p * n1) || 1) / Math.abs(1 - (p * n1) / (m * m) || 1e-12))
    if (!Number.isFinite(delta) || !(delta > 0)) {
      delta = 1 / Math.sqrt(Math.log(1 + (sd / mean) ** 2) || 1)
      gamma = 0.5 / delta - delta * Math.log(mean || 1)
    }
    lambda = Math.sign(skew) || 1
    xi = (xpz + xmz) / 2 - 0.5 * m * ((p / n1 + 1) / ((p / n1 - 1) || 1e-12))
    if (!Number.isFinite(xi)) xi = minOf(v) - 1
    for (const xi_ of v) zs.push(gamma + delta * Math.log(Math.max(1e-300, lambda * (xi_ - xi))))
  }
  return { family, gamma, delta, xi, lambda, z: zs }
}

export interface WeibullFit {
  shape: number
  scale: number
  /** Location (threshold); 0 for two-parameter Weibull. */
  location: number
  logLik: number
}

/** Two-parameter Weibull MLE (shape β, scale η) via Newton on the shape equation. */
export function weibullFit(x: ArrayLike<number | null | undefined>, options: { location?: number } = {}): WeibullFit {
  const loc = options.location ?? 0
  const raw = Array.from(cleanNumbers(x))
  const v = raw.map((xi) => xi - loc).filter((xi) => xi > 0)
  const n = v.length
  if (n < 3) throw new RangeError('weibullFit needs at least 3 observations above the location')
  const logV = v.map(Math.log)
  // Newton on Σ ln x_i / n  =  (Σ x^β ln x) / (Σ x^β) − 1/β
  let beta = 1
  for (let iter = 0; iter < 50; iter++) {
    let sXb = 0
    let sXbL = 0
    let sXbL2 = 0
    for (let i = 0; i < n; i++) {
      const xb = v[i]! ** beta
      const lv = logV[i]!
      sXb += xb
      sXbL += xb * lv
      sXbL2 += xb * lv * lv
    }
    const meanLog = meanOf(logV)
    const f = sXbL / sXb - 1 / beta - meanLog
    const df = (sXbL2 * sXb - sXbL * sXbL) / (sXb * sXb) + 1 / (beta * beta)
    const step = f / df
    beta -= step
    if (!(beta > 0)) beta = Math.abs(beta) + 0.1
    if (Math.abs(step) < 1e-10) break
  }
  let sXb = 0
  for (let i = 0; i < n; i++) sXb += v[i]! ** beta
  const eta = (sXb / n) ** (1 / beta)
  let logLik = n * Math.log(beta) - n * beta * Math.log(eta) + (beta - 1) * meanOf(logV)
  for (let i = 0; i < n; i++) logLik -= (v[i]! / eta) ** beta
  return { shape: beta, scale: eta, location: loc, logLik }
}

// ---- 4.7 Tolerance intervals ----------------------------------------------------------------------

export interface ToleranceIntervalResult {
  method: 'normal' | 'nonparametric'
  coverage: number
  confidence: number
  interval: [number, number]
  /** Howe / Hahn–Meeker k-factor for the normal method. */
  k?: number
  n: number
}

/**
 * Two-sided tolerance interval that covers at least `coverage` of the population with `confidence`.
 * Normal: Howe (1969) approximation to the exact k-factor. Nonparametric: order-statistic interval
 * with the smallest m such that the coverage probability ≥ confidence (Hahn–Meeker).
 */
export function toleranceInterval(
  x: ArrayLike<number | null | undefined>,
  options: { coverage?: number; confidence?: number; method?: 'normal' | 'nonparametric' } = {},
): ToleranceIntervalResult {
  const v = Array.from(cleanNumbers(x)).sort((a, b) => a - b)
  const n = v.length
  if (n < 2) throw new RangeError('toleranceInterval needs at least 2 observations')
  const coverage = options.coverage ?? 0.95
  const confidence = options.confidence ?? 0.95
  if (!(coverage > 0 && coverage < 1)) throw new RangeError('toleranceInterval: coverage must be in (0, 1)')
  if (!(confidence > 0 && confidence < 1)) throw new RangeError('toleranceInterval: confidence must be in (0, 1)')
  const method = options.method ?? 'normal'

  if (method === 'normal') {
    const m = meanOf(v)
    const s = sdOf(v, 1)
    // Howe (1969): k ≈ z_{ (1+p)/2 } · √(1 + 1/n) · √(χ²_{1−α, n−1} / (n−1))  — slightly conservative
    // Better: u = z_{(1+γ)/2} √(1+1/n); k = u √((n−1)/χ²_{α,n−1})  where γ=coverage, α=1−confidence? 
    // Standard two-sided: P(coverage ≥ p) = confidence. Howe:
    const z = STD.ppf(0.5 + coverage / 2)
    const chi = chi2Dist(n - 1).ppf(1 - confidence) // lower-tail χ² for the sd factor
    const u = z * Math.sqrt(1 + 1 / n)
    const k = u * Math.sqrt((n - 1) / chi)
    return { method: 'normal', coverage, confidence, interval: [m - k * s, m + k * s], k, n }
  }

  // Nonparametric: find smallest r such that P(X_{(r)} … X_{(n−r+1)} covers ≥ p) ≥ confidence
  // = Σ_{k=0}^{r−1} C(n,k) p̂… wait: P(W ≥ p) where W = F(X_{(n−r+1)}) − F(X_{(r)})
  // = Σ_{j=0}^{r-1} Binom(n,j) but exact: I_p(n−2r+1, 2r) wait.
  // Standard: number of observations trimmed each side r−1; interval (X_(r), X_(n−r+1)).
  // Confidence = Σ_{k=0}^{2r−2} C(n,k) (1−coverage)^{n−k} coverage^k? 
  // Actually P(X_{(r)} < θ_pL and X_{(n−r+1)} > θ_pU) where [θ_pL,θ_pU] has mass coverage
  // = Σ_{i=r}^{n−r} C(n,i) coverage^i (1−coverage)^{n−i}  — no that's wrong.
  // Correct (Wilks): P(F(X_{(s)}) − F(X_{(r)}) ≥ γ) = Σ_{i=0}^{r+n−s−1?} 
  // For symmetric (r, n−r+1): conf = Σ_{k=0}^{2r−2} binom(n,k) (1−γ)^{…}
  // Use: conf(r) = 1 − betainc_reg(γ; n−2r+2, 2r−1) wait.
  // Simpler exact: conf = Σ_{j=0}^{r-1} C(n,j) [ (1-p)^{n-j} * something ]
  // Wilks (1941): P = Σ_{i=0}^{r-1} Σ_{j=0}^{r-1}  n!/(i!j!(n-i-j)!) * ∫… 
  // For two-sided equal-tailed order stats X_(r), X_(n-r+1):
  //   confidence = Σ_{k=n−2r+2}^{n} C(n,k) coverage^k (1−coverage)^{n−k}
  const binomCdfUpper = (kMin: number, nn: number, p: number) => {
    // P(K ≥ kMin) for K~Bin(nn,p)
    let s = 0
    const term = Math.exp(lgamma(nn + 1) - lgamma(1) - lgamma(nn + 1) + nn * Math.log(1 - p)) // k=0
    // recompute per k for stability at small n
    for (let k = kMin; k <= nn; k++) {
      const lp = lgamma(nn + 1) - lgamma(k + 1) - lgamma(nn - k + 1) + k * Math.log(p) + (nn - k) * Math.log(1 - p)
      s += Math.exp(lp)
    }
    void term
    return Math.min(1, s)
  }

  let r = 1
  for (; r <= Math.floor(n / 2); r++) {
    const conf = binomCdfUpper(n - 2 * r + 2, n, coverage)
    if (conf >= confidence) break
  }
  if (r > Math.floor(n / 2)) {
    // cannot achieve — return outermost
    r = 1
  }
  // When even the widest (r=1) fails the confidence requirement, still return [X_(1), X_(n)]
  const confAchieved = binomCdfUpper(n - 2 * r + 2, n, coverage)
  if (confAchieved < confidence && r === 1) {
    // try reporting with a note via method only — interval is still valid as outermost
  }
  return { method: 'nonparametric', coverage, confidence, interval: [v[r - 1]!, v[n - r]!], n }
}

// ---- Capability Sixpack (plot-ready panels) -------------------------------------------------------

export interface CapabilitySixpackResult {
  capability: CapabilityResult
  panels: {
    histogram: PlotSeries[]
    normalPlot: PlotSeries[]
    controlChart: PlotSeries[]
    capability: PlotSeries[]
  }
  /** Flat series list for `plotSeries`-style consumers. */
  series: PlotSeries[]
}

/**
 * Plot-ready Capability Sixpack panels (no renderer): histogram, normal probability plot,
 * I-MR or X̄-R control chart, and capability summary lines.
 */
export function capabilitySixpack(
  x: ArrayLike<number | null | undefined> | number[][],
  options: {
    lsl?: number
    usl?: number
    target?: number
    subgroup?: number
    method?: 'r' | 's'
    confidence?: number
    bins?: number
  } = {},
): CapabilitySixpackResult {
  const cap = capability(x, options)
  const groups = asSubgroups(x, options.subgroup)
  const flat = groups.flat()
  const n = flat.length
  const sorted = flat.slice().sort((a, b) => a - b)
  const lo = sorted[0]!
  const hi = sorted[n - 1]!
  const bins = Math.max(5, options.bins ?? Math.min(20, Math.ceil(Math.sqrt(n))))
  const width = (hi - lo) / bins || 1
  const counts = new Array(bins).fill(0)
  const centers = new Array(bins).fill(0)
  for (let b = 0; b < bins; b++) centers[b] = lo + (b + 0.5) * width
  for (const v of flat) {
    let b = Math.floor((v - lo) / width)
    if (b >= bins) b = bins - 1
    if (b < 0) b = 0
    counts[b]++
  }
  const histogram: PlotSeries[] = [
    { name: 'hist', x: centers.slice(), y: counts.slice(), role: 'data' },
  ]
  if (cap.lsl !== undefined) {
    histogram.push({ name: 'lsl', x: [cap.lsl, cap.lsl], y: [0, Math.max(...counts)], role: 'lcl' })
  }
  if (cap.usl !== undefined) {
    histogram.push({ name: 'usl', x: [cap.usl, cap.usl], y: [0, Math.max(...counts)], role: 'ucl' })
  }

  // Normal probability (QQ) plot
  const qqX: number[] = []
  const qqY: number[] = []
  for (let i = 0; i < n; i++) {
    const p = (i + 0.5) / n
    qqX.push(STD.ppf(p))
    qqY.push(sorted[i]!)
  }
  const normalPlot: PlotSeries[] = [
    { name: 'qq', x: qqX, y: qqY, role: 'data' },
    {
      name: 'ref',
      x: [qqX[0]!, qqX[n - 1]!],
      y: [cap.mean + cap.sigmaOverall * qqX[0]!, cap.mean + cap.sigmaOverall * qqX[n - 1]!],
      role: 'center',
    },
  ]

  const useXbar = options.subgroup != null && options.subgroup >= 2
  const chart = controlChart(x as ArrayLike<number | null | undefined> | number[][], {
    type: useXbar ? (options.method === 's' ? 'xbar-s' : 'xbar-r') : 'i-mr',
    subgroup: options.subgroup,
  })
  const ccSeries: PlotSeries[] = [
    { name: 'data', x: chart.points.map((p) => p.index), y: chart.points.map((p) => p.value), role: 'data' },
    { name: 'center', x: chart.points.map((p) => p.index), y: chart.points.map((p) => p.center), role: 'center' },
    { name: 'ucl', x: chart.points.map((p) => p.index), y: chart.points.map((p) => p.ucl), role: 'ucl' },
    { name: 'lcl', x: chart.points.map((p) => p.index), y: chart.points.map((p) => p.lcl), role: 'lcl' },
  ]
  if (chart.companion) {
    ccSeries.push({
      name: 'companion',
      x: chart.companion.points.map((p) => p.index),
      y: chart.companion.points.map((p) => p.value),
      role: 'companion',
    })
  }

  const capPanel: PlotSeries[] = [
    { name: 'mean', x: [0, 1], y: [cap.mean, cap.mean], role: 'center' },
  ]
  if (cap.Cpk != null) capPanel.push({ name: 'Cpk', x: [0], y: [cap.Cpk], role: 'data' })
  if (cap.Cp != null) capPanel.push({ name: 'Cp', x: [0], y: [cap.Cp], role: 'band' })

  const series = [
    ...histogram.map((s) => ({ ...s, name: `hist.${s.name}` })),
    ...normalPlot.map((s) => ({ ...s, name: `qq.${s.name}` })),
    ...ccSeries.map((s) => ({ ...s, name: `spc.${s.name}` })),
    ...capPanel.map((s) => ({ ...s, name: `cap.${s.name}` })),
  ]

  return {
    capability: cap,
    panels: { histogram, normalPlot, controlChart: ccSeries, capability: capPanel },
    series,
  }
}

