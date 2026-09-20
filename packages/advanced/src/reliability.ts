/**
 * Tier 5.3 — Reliability / Survival:
 * Weibull / lognormal / exponential MLE with right (and interval) censoring, LSXY (probability-plot)
 * fits, Kaplan–Meier, warranty prediction.
 */
import { chi2 as chi2Dist, normal } from './dist.js'
import { lstsq, matrix } from './linalg.js'
import { weibullFit } from './capability.js'
import { nelderMead } from './optim.js'

const STD = normal()

export type CensorCode = 0 | 1 | 2 | 3 // 0 exact, 1 right, 2 left, 3 interval (use time2)

export interface SurvivalObs {
  time: number
  /** Upper end for interval censoring. */
  time2?: number
  /** 0 = failure / exact, 1 = right-censored, 2 = left-censored, 3 = interval. */
  censor?: CensorCode
}

function asObs(time: ArrayLike<number | null | undefined>, censor?: ArrayLike<number | null | undefined>, time2?: ArrayLike<number | null | undefined>): SurvivalObs[] {
  const t = Array.from(time)
  const out: SurvivalObs[] = []
  for (let i = 0; i < t.length; i++) {
    const ti = t[i]
    if (typeof ti !== 'number' || !Number.isFinite(ti) || ti < 0) continue
    const c = censor ? Number(censor[i] ?? 0) : 0
    out.push({ time: ti, time2: time2 ? Number(time2[i]) : undefined, censor: (c as CensorCode) || 0 })
  }
  if (out.length < 3) throw new RangeError('reliability: need at least 3 observations')
  return out
}

export interface ParametricSurvival {
  distribution: 'weibull' | 'lognormal' | 'exponential'
  /** Shape (Weibull β) or σ (lognormal) or unused. */
  shape?: number
  /** Scale η (Weibull) / exp(μ) (lognormal) / mean (exponential). */
  scale: number
  location?: number
  logLik: number
  n: number
  nFailures: number
  /** Percentiles t_p for requested p. */
  percentiles?: Array<{ p: number; time: number }>
  method: 'MLE' | 'LSXY'
  /** Parameter standard errors (MLE only), keyed by name. */
  se?: Record<string, number>
  /** Wald confidence intervals (MLE only). */
  ci?: Record<string, [number, number]>
  /** Variance–covariance of the parameter vector in `paramNames` order. */
  vcov?: number[][]
  /** Names matching rows/cols of `vcov`. */
  paramNames?: string[]
  confidence?: number
}

/** Numerical Hessian of f at x (central differences); returns H such that H_ij ≈ ∂²f/∂x_i∂x_j. */
function numericalHessian(f: (x: number[]) => number, x: number[], eps = 1e-5): number[][] {
  const k = x.length
  const H = Array.from({ length: k }, () => new Array(k).fill(0))
  const f0 = f(x)
  for (let i = 0; i < k; i++) {
    const hi = Math.max(eps, eps * Math.abs(x[i]!))
    const xp = x.slice()
    const xm = x.slice()
    xp[i]! += hi
    xm[i]! -= hi
    H[i]![i] = (f(xp) - 2 * f0 + f(xm)) / (hi * hi)
    for (let j = i + 1; j < k; j++) {
      const hj = Math.max(eps, eps * Math.abs(x[j]!))
      const xpp = x.slice()
      const xpm = x.slice()
      const xmp = x.slice()
      const xmm = x.slice()
      xpp[i]! += hi
      xpp[j]! += hj
      xpm[i]! += hi
      xpm[j]! -= hj
      xmp[i]! -= hi
      xmp[j]! += hj
      xmm[i]! -= hi
      xmm[j]! -= hj
      const hij = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * hi * hj)
      H[i]![j] = hij
      H[j]![i] = hij
    }
  }
  return H
}

function invertSymmetric(H: number[][]): number[][] | null {
  const k = H.length
  const A = H.map((row) => row.slice())
  const I = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => (i === j ? 1 : 0)))
  for (let col = 0; col < k; col++) {
    let piv = col
    for (let r = col + 1; r < k; r++) if (Math.abs(A[r]![col]!) > Math.abs(A[piv]![col]!)) piv = r
    if (Math.abs(A[piv]![col]!) < 1e-14) return null
    ;[A[col], A[piv]] = [A[piv]!, A[col]!]
    ;[I[col], I[piv]] = [I[piv]!, I[col]!]
    const d = A[col]![col]!
    for (let j = 0; j < k; j++) {
      A[col]![j]! /= d
      I[col]![j]! /= d
    }
    for (let r = 0; r < k; r++) {
      if (r === col) continue
      const f = A[r]![col]!
      for (let j = 0; j < k; j++) {
        A[r]![j]! -= f * A[col]![j]!
        I[r]![j]! -= f * I[col]![j]!
      }
    }
  }
  return I
}

function attachMleErrors(
  fit: ParametricSurvival,
  paramNames: string[],
  params: number[],
  logLikFn: (p: number[]) => number,
  confidence: number,
): ParametricSurvival {
  // Hessian of logLik; information = −H; vcov = (−H)⁻¹
  const H = numericalHessian(logLikFn, params)
  const negH = H.map((row) => row.map((v) => -v))
  const vcov = invertSymmetric(negH)
  if (!vcov) return { ...fit, confidence }
  const z = STD.ppf(0.5 + confidence / 2)
  const se: Record<string, number> = {}
  const ci: Record<string, [number, number]> = {}
  for (let i = 0; i < paramNames.length; i++) {
    const s = Math.sqrt(Math.max(0, vcov[i]![i]!))
    se[paramNames[i]!] = s
    ci[paramNames[i]!] = [params[i]! - z * s, params[i]! + z * s]
  }
  return { ...fit, se, ci, vcov, paramNames, confidence }
}

function weibullLogLik(obs: SurvivalObs[], beta: number, eta: number): number {
  let ll = 0
  for (const o of obs) {
    const c = o.censor ?? 0
    if (c === 0) {
      // f(t) = (β/η)(t/η)^{β−1} exp(−(t/η)^β)
      ll += Math.log(beta / eta) + (beta - 1) * Math.log(o.time / eta) - (o.time / eta) ** beta
    } else if (c === 1) {
      ll += -((o.time / eta) ** beta) // S(t)
    } else if (c === 2) {
      // F(t) = 1 − S(t)
      const F = 1 - Math.exp(-((o.time / eta) ** beta))
      ll += Math.log(Math.max(F, 1e-300))
    } else if (c === 3 && o.time2 !== undefined) {
      const S1 = Math.exp(-((o.time / eta) ** beta))
      const S2 = Math.exp(-((o.time2 / eta) ** beta))
      ll += Math.log(Math.max(S1 - S2, 1e-300))
    }
  }
  return ll
}

function fitWeibullMle(obs: SurvivalObs[], confidence = 0.95): ParametricSurvival {
  const failures = obs.filter((o) => (o.censor ?? 0) === 0).map((o) => o.time)
  let beta = 1.2
  let eta = meanOf(failures.length ? failures : obs.map((o) => o.time)) || 1
  if (failures.length >= 3) {
    try {
      const w = weibullFit(failures)
      beta = w.shape
      eta = w.scale
    } catch {
      /* keep starts */
    }
  }
  let best = weibullLogLik(obs, beta, eta)
  for (let pass = 0; pass < 50; pass++) {
    let improved = false
    for (const [dBeta, dEta] of [
      [0.05, 0],
      [-0.05, 0],
      [0.01, 0],
      [-0.01, 0],
      [0, 0.05 * eta],
      [0, -0.05 * eta],
      [0, 0.01 * eta],
      [0, -0.01 * eta],
      [0.02, 0.02 * eta],
      [-0.02, -0.02 * eta],
    ] as const) {
      const b2 = Math.max(0.05, beta + dBeta)
      const e2 = Math.max(1e-8, eta + dEta)
      const ll = weibullLogLik(obs, b2, e2)
      if (ll > best) {
        best = ll
        beta = b2
        eta = e2
        improved = true
      }
    }
    if (!improved) break
  }
  {
    // polish the coarse pattern search to full precision
    const nm = nelderMead((t) => -weibullLogLik(obs, Math.exp(t[0]!), Math.exp(t[1]!)), [Math.log(beta), Math.log(eta)], { step: 0.02 })
    if (-nm.value >= best) {
      best = -nm.value
      beta = Math.exp(nm.x[0]!)
      eta = Math.exp(nm.x[1]!)
    }
  }
  const fit: ParametricSurvival = {
    distribution: 'weibull',
    shape: beta,
    scale: eta,
    logLik: best,
    n: obs.length,
    nFailures: obs.filter((o) => (o.censor ?? 0) === 0).length,
    method: 'MLE',
  }
  return attachMleErrors(fit, ['shape', 'scale'], [beta, eta], (p) => weibullLogLik(obs, p[0]!, p[1]!), confidence)
}

function meanOf(v: number[]): number {
  return v.reduce((s, x) => s + x, 0) / (v.length || 1)
}

function exponentialLogLik(obs: SurvivalObs[], scale: number): number {
  // scale = MTTF = 1/λ
  if (!(scale > 0)) return -Infinity
  let ll = 0
  for (const o of obs) {
    const c = o.censor ?? 0
    if (c === 0) ll += -Math.log(scale) - o.time / scale
    else if (c === 1) ll += -o.time / scale
    else if (c === 2) {
      const F = 1 - Math.exp(-o.time / scale)
      ll += Math.log(Math.max(F, 1e-300))
    } else if (c === 3 && o.time2 !== undefined) {
      const S1 = Math.exp(-o.time / scale)
      const S2 = Math.exp(-o.time2 / scale)
      ll += Math.log(Math.max(S1 - S2, 1e-300))
    }
  }
  return ll
}

function fitExponentialMle(obs: SurvivalObs[], confidence = 0.95): ParametricSurvival {
  // Start from exact+right closed form when possible, then refine with full likelihood
  let totalTime = 0
  let failures = 0
  for (const o of obs) {
    const c = o.censor ?? 0
    if (c === 0 || c === 1) totalTime += o.time
    if (c === 0) failures++
    if (c === 3 && o.time2 !== undefined) totalTime += 0.5 * (o.time + o.time2)
  }
  if (failures < 1 && !obs.some((o) => (o.censor ?? 0) === 2 || (o.censor ?? 0) === 3)) {
    throw new RangeError('exponential MLE: need at least one failure')
  }
  let scale = failures > 0 ? totalTime / failures : meanOf(obs.map((o) => o.time)) || 1
  let best = exponentialLogLik(obs, scale)
  for (let pass = 0; pass < 40; pass++) {
    let improved = false
    for (const step of [0.1, 0.02, 0.005].map((f) => f * scale)) {
      for (const s of [scale + step, scale - step]) {
        if (!(s > 0)) continue
        const ll = exponentialLogLik(obs, s)
        if (ll > best) {
          best = ll
          scale = s
          improved = true
        }
      }
    }
    if (!improved) break
  }
  {
    const nm = nelderMead((t) => -exponentialLogLik(obs, Math.exp(t[0]!)), [Math.log(scale)], { step: 0.02 })
    if (-nm.value >= best) {
      best = -nm.value
      scale = Math.exp(nm.x[0]!)
    }
  }
  const nFailures = obs.filter((o) => (o.censor ?? 0) === 0).length
  const fit: ParametricSurvival = {
    distribution: 'exponential',
    scale,
    logLik: best,
    n: obs.length,
    nFailures,
    method: 'MLE',
  }
  return attachMleErrors(fit, ['scale'], [scale], (p) => exponentialLogLik(obs, p[0]!), confidence)
}

function lognormalLogLik(obs: SurvivalObs[], mu: number, sigma: number): number {
  if (!(sigma > 0)) return -Infinity
  let L = 0
  for (const o of obs) {
    const c = o.censor ?? 0
    if (c === 3 && o.time2 !== undefined) {
      const z1 = (Math.log(Math.max(o.time, 1e-300)) - mu) / sigma
      const z2 = (Math.log(Math.max(o.time2, 1e-300)) - mu) / sigma
      L += Math.log(Math.max(STD.cdf(z2) - STD.cdf(z1), 1e-300))
      continue
    }
    const z = (Math.log(Math.max(o.time, 1e-300)) - mu) / sigma
    if (c === 0) L += -Math.log(o.time * sigma * Math.sqrt(2 * Math.PI)) - 0.5 * z * z
    else if (c === 1) L += Math.log(Math.max(STD.sf(z), 1e-300))
    else if (c === 2) L += Math.log(Math.max(STD.cdf(z), 1e-300))
  }
  return L
}

function fitLognormalMle(obs: SurvivalObs[], confidence = 0.95): ParametricSurvival {
  const failures = obs.filter((o) => (o.censor ?? 0) === 0).map((o) => o.time).filter((t) => t > 0)
  if (failures.length < 2) throw new RangeError('lognormal MLE: need ≥ 2 failures')
  const logs = failures.map(Math.log)
  let mu = meanOf(logs)
  let sigma = Math.sqrt(logs.reduce((s, x) => s + (x - mu) ** 2, 0) / (logs.length - 1))
  let best = lognormalLogLik(obs, mu, sigma)
  for (let pass = 0; pass < 40; pass++) {
    let improved = false
    for (const [dm, ds] of [
      [0.05, 0],
      [-0.05, 0],
      [0, 0.05],
      [0, -0.05],
      [0.01, 0.01],
      [-0.01, -0.01],
    ] as const) {
      const m2 = mu + dm
      const s2 = Math.max(0.05, sigma + ds)
      const L = lognormalLogLik(obs, m2, s2)
      if (L > best) {
        best = L
        mu = m2
        sigma = s2
        improved = true
      }
    }
    if (!improved) break
  }
  {
    const nm = nelderMead((t) => -lognormalLogLik(obs, t[0]!, Math.exp(t[1]!)), [mu, Math.log(sigma)], { step: 0.02 })
    if (-nm.value >= best) {
      best = -nm.value
      mu = nm.x[0]!
      sigma = Math.exp(nm.x[1]!)
    }
  }
  const fit: ParametricSurvival = {
    distribution: 'lognormal',
    shape: sigma,
    scale: Math.exp(mu),
    location: mu,
    logLik: best,
    n: obs.length,
    nFailures: failures.length,
    method: 'MLE',
  }
  return attachMleErrors(fit, ['location', 'shape'], [mu, sigma], (p) => lognormalLogLik(obs, p[0]!, p[1]!), confidence)
}

/** Median-rank (Bernard) plotting positions for uncensored / right-censored samples. */
function medianRanks(obs: SurvivalObs[]): Array<{ time: number; F: number }> {
  const sorted = obs.slice().sort((a, b) => a.time - b.time)
  const n = sorted.length
  const points: Array<{ time: number; F: number }> = []
  let i = 0
  for (let k = 0; k < n; k++) {
    if ((sorted[k]!.censor ?? 0) !== 0) continue
    i++
    // Kaplan–Meier-ish rank adjustment for right censoring: use order among failures with reverse rank
    const F = (i - 0.3) / (n + 0.4)
    points.push({ time: sorted[k]!.time, F })
  }
  return points
}

function fitLsxy(obs: SurvivalObs[], distribution: 'weibull' | 'lognormal' | 'exponential'): ParametricSurvival {
  const pts = medianRanks(obs).filter((p) => p.time > 0 && p.F > 0 && p.F < 1)
  if (pts.length < 2) throw new RangeError('LSXY: need at least 2 failures')
  let x: number[]
  let y: number[]
  if (distribution === 'weibull') {
    x = pts.map((p) => Math.log(p.time))
    y = pts.map((p) => Math.log(-Math.log(1 - p.F)))
  } else if (distribution === 'lognormal') {
    x = pts.map((p) => Math.log(p.time))
    y = pts.map((p) => STD.ppf(p.F))
  } else {
    x = pts.map((p) => p.time)
    y = pts.map((p) => -Math.log(1 - p.F))
  }
  const X = matrix(x.length, 2)
  for (let i = 0; i < x.length; i++) {
    X.data[i * 2] = 1
    X.data[i * 2 + 1] = x[i]!
  }
  const { coef } = lstsq(X, y)
  if (distribution === 'weibull') {
    const beta = coef[1]!
    const eta = Math.exp(-coef[0]! / beta)
    return { distribution, shape: beta, scale: eta, logLik: NaN, n: obs.length, nFailures: pts.length, method: 'LSXY' }
  }
  if (distribution === 'lognormal') {
    const sigma = 1 / coef[1]!
    const mu = -coef[0]! * sigma
    return { distribution, shape: sigma, scale: Math.exp(mu), location: mu, logLik: NaN, n: obs.length, nFailures: pts.length, method: 'LSXY' }
  }
  const rate = coef[1]!
  return { distribution: 'exponential', scale: 1 / rate, logLik: NaN, n: obs.length, nFailures: pts.length, method: 'LSXY' }
}

/**
 * Parametric reliability fit (Weibull / lognormal / exponential) by MLE or LSXY.
 * `censor`: 0 exact, 1 right-censored (default 0).
 */
export function reliabilityFit(
  time: ArrayLike<number | null | undefined>,
  options: {
    distribution?: 'weibull' | 'lognormal' | 'exponential'
    method?: 'MLE' | 'LSXY'
    censor?: ArrayLike<number | null | undefined>
    time2?: ArrayLike<number | null | undefined>
    percentiles?: number[]
    confidence?: number
  } = {},
): ParametricSurvival {
  const obs = asObs(time, options.censor, options.time2)
  const dist = options.distribution ?? 'weibull'
  const method = options.method ?? 'MLE'
  const confidence = options.confidence ?? 0.95
  let fit =
    method === 'LSXY'
      ? fitLsxy(obs, dist)
      : dist === 'weibull'
        ? fitWeibullMle(obs, confidence)
        : dist === 'lognormal'
          ? fitLognormalMle(obs, confidence)
          : fitExponentialMle(obs, confidence)
  const pct = options.percentiles ?? [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]
  fit = {
    ...fit,
    percentiles: pct.map((p) => ({
      p,
      time:
        fit.distribution === 'weibull'
          ? fit.scale * (-Math.log(1 - p)) ** (1 / (fit.shape ?? 1))
          : fit.distribution === 'lognormal'
            ? Math.exp((fit.location ?? Math.log(fit.scale)) + (fit.shape ?? 1) * STD.ppf(p))
            : -fit.scale * Math.log(1 - p),
    })),
  }
  return fit
}

export interface KaplanMeierPoint {
  time: number
  survival: number
  /** Greenwood SE. */
  se: number
  atRisk: number
  events: number
  lower: number
  upper: number
}

export interface KaplanMeierResult {
  curve: KaplanMeierPoint[]
  n: number
  nEvents: number
}

/** Kaplan–Meier product-limit estimator with Greenwood variance and log-log CI. */
export function kaplanMeier(
  time: ArrayLike<number | null | undefined>,
  options: { censor?: ArrayLike<number | null | undefined>; confidence?: number } = {},
): KaplanMeierResult {
  const obs = asObs(time, options.censor)
  const conf = options.confidence ?? 0.95
  const z = STD.ppf(0.5 + conf / 2)
  const times = [...new Set(obs.map((o) => o.time))].sort((a, b) => a - b)
  const eventsAt = new Map<number, number>()
  const censoredAt = new Map<number, number>()
  for (const o of obs) {
    if ((o.censor ?? 0) === 0) eventsAt.set(o.time, (eventsAt.get(o.time) ?? 0) + 1)
    else if ((o.censor ?? 0) === 1) censoredAt.set(o.time, (censoredAt.get(o.time) ?? 0) + 1)
  }
  let atRisk = obs.length
  let S = 1
  let sumVar = 0
  const curve: KaplanMeierPoint[] = []
  let nEvents = 0
  for (const t of times) {
    const events = eventsAt.get(t) ?? 0
    const censored = censoredAt.get(t) ?? 0
    if (events > 0 && atRisk > 0) {
      S *= 1 - events / atRisk
      sumVar += events / (atRisk * (atRisk - events || 1e-12))
      nEvents += events
      const se = S * Math.sqrt(sumVar)
      // log-log CI
      let lower = 0
      let upper = 1
      if (S > 0 && S < 1 && se > 0) {
        const theta = Math.log(-Math.log(S))
        const seTheta = Math.sqrt(sumVar) / (Math.log(S) || 1e-12) // d/dS log(-log S) = 1/(S log S) * S... wait
        // better: se of log(-log S) ≈ se / (S · |log S|)
        const seLL = se / (S * Math.abs(Math.log(S)))
        lower = Math.exp(-Math.exp(theta + z * seLL))
        upper = Math.exp(-Math.exp(theta - z * seLL))
        void seTheta
      }
      curve.push({ time: t, survival: S, se, atRisk, events, lower, upper })
    }
    atRisk -= events + censored
  }
  return { curve, n: obs.length, nEvents }
}

export interface ProbabilityPlotData {
  distribution: string
  /** x plotting positions (transformed time). */
  x: number[]
  /** y plotting positions (transformed F). */
  y: number[]
  /** Fitted line y = a + b x. */
  intercept: number
  slope: number
  times: number[]
  F: number[]
}

/** Coordinates for a reliability probability plot (Weibull / lognormal / exponential). */
export function probabilityPlot(
  time: ArrayLike<number | null | undefined>,
  options: { distribution?: 'weibull' | 'lognormal' | 'exponential'; censor?: ArrayLike<number | null | undefined> } = {},
): ProbabilityPlotData {
  const obs = asObs(time, options.censor)
  const dist = options.distribution ?? 'weibull'
  const pts = medianRanks(obs).filter((p) => p.time > 0 && p.F > 0 && p.F < 1)
  let x: number[]
  let y: number[]
  if (dist === 'weibull') {
    x = pts.map((p) => Math.log(p.time))
    y = pts.map((p) => Math.log(-Math.log(1 - p.F)))
  } else if (dist === 'lognormal') {
    x = pts.map((p) => Math.log(p.time))
    y = pts.map((p) => STD.ppf(p.F))
  } else {
    x = pts.map((p) => p.time)
    y = pts.map((p) => -Math.log(1 - p.F))
  }
  const X = matrix(x.length, 2)
  for (let i = 0; i < x.length; i++) {
    X.data[i * 2] = 1
    X.data[i * 2 + 1] = x[i]!
  }
  const { coef } = lstsq(X, y)
  return { distribution: dist, x, y, intercept: coef[0]!, slope: coef[1]!, times: pts.map((p) => p.time), F: pts.map((p) => p.F) }
}

export interface WarrantyResult {
  distribution: string
  /** Expected failures in [0, warranty] for `n` units. */
  expectedFailures: number
  /** Survival probability at warranty length. */
  reliability: number
  warranty: number
  nUnits: number
}

/** Warranty prediction from a fitted parametric model. */
export function warrantyPrediction(fit: ParametricSurvival, options: { warranty: number; nUnits?: number }): WarrantyResult {
  const w = options.warranty
  const nUnits = options.nUnits ?? 1
  if (!(w > 0)) throw new RangeError('warrantyPrediction: warranty must be > 0')
  let R: number
  if (fit.distribution === 'weibull') R = Math.exp(-((w / fit.scale) ** (fit.shape ?? 1)))
  else if (fit.distribution === 'lognormal') {
    const mu = fit.location ?? Math.log(fit.scale)
    R = STD.sf((Math.log(w) - mu) / (fit.shape ?? 1))
  } else R = Math.exp(-w / fit.scale)
  return { distribution: fit.distribution, expectedFailures: nUnits * (1 - R), reliability: R, warranty: w, nUnits }
}

export interface LogRankResult {
  test: 'log-rank' | 'wilcoxon' | 'tarone-ware'
  statistic: number
  df: number
  pValue: number
  groups: string[]
  /** Observed events per group. */
  observed: number[]
  /** Expected events under H0 (Mantel–Haenszel). */
  expected: number[]
  n: number
  nEvents: number
  weight: 'logrank' | 'wilcoxon' | 'tarone-ware'
  stratified: boolean
}

function accumulateLogRank(
  rows: Array<{ time: number; group: string; event: boolean }>,
  groups: string[],
  weight: 'logrank' | 'wilcoxon' | 'tarone-ware',
): { O: number[]; E: number[]; V: number[][] } {
  const K = groups.length
  const gIndex = new Map(groups.map((g, i) => [g, i]))
  const O = new Array(K).fill(0)
  const E = new Array(K).fill(0)
  const V = Array.from({ length: K }, () => new Array(K).fill(0))
  // single ascending sweep: at-risk counters decrease as rows leave the risk set
  const sorted = rows.slice().sort((a, b) => a.time - b.time)
  const atRisk = new Array<number>(K).fill(0)
  for (const r of sorted) atRisk[gIndex.get(r.group)!]!++
  let nRisk = sorted.length
  let idx = 0
  while (idx < sorted.length) {
    const t = sorted[idx]!.time
    const deaths = new Array<number>(K).fill(0)
    let d = 0
    let leaving = 0
    const leavingByGroup = new Array<number>(K).fill(0)
    let end = idx
    while (end < sorted.length && sorted[end]!.time === t) {
      const r = sorted[end]!
      const j = gIndex.get(r.group)!
      if (r.event) {
        deaths[j]!++
        d++
      }
      leavingByGroup[j]!++
      leaving++
      end++
    }
    const nRiskNow = nRisk
    const atRiskNow = atRisk.slice()
    for (let j = 0; j < K; j++) atRisk[j]! -= leavingByGroup[j]!
    nRisk -= leaving
    idx = end
    if (d < 1 || nRiskNow < 2) continue
    {
      const nRisk = nRiskNow
      const atRisk = atRiskNow
    const w =
      weight === 'wilcoxon' ? nRisk : weight === 'tarone-ware' ? Math.sqrt(nRisk) : 1
    for (let j = 0; j < K; j++) {
      O[j] += w * deaths[j]!
      E[j] += w * (atRisk[j]! * d) / nRisk
    }
    for (let j = 0; j < K; j++) {
      for (let l = 0; l < K; l++) {
        const nj = atRisk[j]!
        const nl = atRisk[l]!
        const delta = j === l ? 1 : 0
        V[j]![l]! +=
          w * w * ((d * (nRisk - d)) / (nRisk * (nRisk - 1) || 1)) * (delta * nj - (nj * nl) / nRisk)
      }
    }
    }
  }
  return { O, E, V }
}

function chi2FromOE(O: number[], E: number[], V: number[][]): { statistic: number; df: number } {
  const K = O.length
  const df = K - 1
  let statistic = 0
  if (K === 2) {
    const diff = O[0]! - E[0]!
    const v = Math.max(V[0]![0]!, 1e-300)
    statistic = (diff * diff) / v
  } else {
    const dim = K - 1
    const A = Array.from({ length: dim }, (_, i) => V[i]!.slice(0, dim))
    const b = Array.from({ length: dim }, (_, i) => O[i]! - E[i]!)
    const Aug = A.map((row, i) => row.concat([b[i]!]))
    for (let col = 0; col < dim; col++) {
      let piv = col
      for (let r = col + 1; r < dim; r++) if (Math.abs(Aug[r]![col]!) > Math.abs(Aug[piv]![col]!)) piv = r
      ;[Aug[col], Aug[piv]] = [Aug[piv]!, Aug[col]!]
      const d0 = Aug[col]![col]!
      if (Math.abs(d0) < 1e-14) continue
      for (let j = col; j <= dim; j++) Aug[col]![j]! /= d0
      for (let r = 0; r < dim; r++) {
        if (r === col) continue
        const f = Aug[r]![col]!
        for (let j = col; j <= dim; j++) Aug[r]![j]! -= f * Aug[col]![j]!
      }
    }
    const x = Aug.map((row) => row[dim]!)
    for (let i = 0; i < dim; i++) statistic += b[i]! * x[i]!
  }
  return { statistic: Math.max(0, statistic), df }
}

/**
 * Mantel–Haenszel / Wilcoxon / Tarone–Ware test comparing survival across 2+ groups.
 * `censor`: 0 = event, 1 = right-censored (default 0). Optional `strata` for stratified test.
 */
export function logRank(
  time: ArrayLike<number | null | undefined>,
  group: ArrayLike<string | number | null | undefined>,
  options: {
    censor?: ArrayLike<number | null | undefined>
    weight?: 'logrank' | 'wilcoxon' | 'tarone-ware'
    strata?: ArrayLike<string | number | null | undefined>
  } = {},
): LogRankResult {
  const weight = options.weight ?? 'logrank'
  const tArr = Array.from(time)
  const gArr = Array.from(group)
  if (tArr.length !== gArr.length) throw new RangeError('logRank: time/group length mismatch')
  const rows: Array<{ time: number; group: string; event: boolean; stratum: string }> = []
  for (let i = 0; i < tArr.length; i++) {
    const ti = tArr[i]
    const gi = gArr[i]
    if (typeof ti !== 'number' || !Number.isFinite(ti) || ti < 0) continue
    if (gi === null || gi === undefined || gi === '') continue
    const c = options.censor ? Number(options.censor[i] ?? 0) : 0
    const st = options.strata ? String(options.strata[i] ?? '') : ''
    rows.push({ time: ti, group: String(gi), event: c === 0, stratum: st })
  }
  if (rows.length < 4) throw new RangeError('logRank: need at least 4 valid observations')
  const groups = [...new Set(rows.map((r) => r.group))].sort()
  const K = groups.length
  if (K < 2) throw new RangeError('logRank: need at least 2 groups')

  const strataKeys = [...new Set(rows.map((r) => r.stratum))]
  const O = new Array(K).fill(0)
  const E = new Array(K).fill(0)
  const V = Array.from({ length: K }, () => new Array(K).fill(0))
  for (const sk of strataKeys) {
    const sub = rows.filter((r) => r.stratum === sk)
    if (sub.length < 2) continue
    const acc = accumulateLogRank(sub, groups, weight)
    for (let j = 0; j < K; j++) {
      O[j] += acc.O[j]!
      E[j] += acc.E[j]!
      for (let l = 0; l < K; l++) V[j]![l]! += acc.V[j]![l]!
    }
  }

  const { statistic, df } = chi2FromOE(O, E, V)
  const pValue = chi2Dist(df).sf(statistic)
  const testName = weight === 'wilcoxon' ? 'wilcoxon' : weight === 'tarone-ware' ? 'tarone-ware' : 'log-rank'
  return {
    test: testName,
    statistic,
    df,
    pValue,
    groups,
    observed: O,
    expected: E,
    n: rows.length,
    nEvents: O.reduce((a, b) => a + b, 0),
    weight,
    stratified: strataKeys.length > 1 || (strataKeys.length === 1 && strataKeys[0] !== ''),
  }
}

export interface CoxPHResult {
  coefficients: number[]
  se: number[]
  /** Model-based SE (always present); equals `se` unless cluster robust was requested. */
  seModel?: number[]
  z: number[]
  pValue: number[]
  ci: Array<[number, number]>
  hr: number[]
  logLik: number
  n: number
  nEvents: number
  names: string[]
  iterations: number
  ties: 'breslow'
  stratified: boolean
  robust?: boolean
  causeSpecific?: boolean
  /** True when frailty uses bands / AR(1) process / counting-process start. */
  timeVaryingFrailty?: boolean
  frailtyVar?: number
  /** AR(1) correlation across ordered frailty bands when `process: 'ar1'`. */
  frailtyRho?: number
  frailty?: Array<{ group: string; blup: number }>
  frailtyByBand?: Array<{ band: string; group: string; blup: number }>
}

type CoxObs = {
  start: number
  stop: number
  event: boolean
  x: number[]
  stratum: string
  frailtyGroup: string
  frailtyBand: string
  cluster: string
  offset: number
  id: number
}

function solveNewton(negH: number[][], score: number[]): number[] {
  const dim = score.length
  const Aug = negH.map((row, i) => row.concat([score[i]!]))
  for (let col = 0; col < dim; col++) {
    let piv = col
    for (let r = col + 1; r < dim; r++) if (Math.abs(Aug[r]![col]!) > Math.abs(Aug[piv]![col]!)) piv = r
    ;[Aug[col], Aug[piv]] = [Aug[piv]!, Aug[col]!]
    const d0 = Aug[col]![col]!
    if (Math.abs(d0) < 1e-14) continue
    for (let j = col; j <= dim; j++) Aug[col]![j]! /= d0
    for (let r = 0; r < dim; r++) {
      if (r === col) continue
      const f = Aug[r]![col]!
      for (let j = col; j <= dim; j++) Aug[r]![j]! -= f * Aug[col]![j]!
    }
  }
  return Aug.map((row) => row[dim]!)
}

function invertInfo(negH: number[][]): number[][] {
  const p = negH.length
  const A = negH.map((r) => r.slice())
  const I = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => (i === j ? 1 : 0)))
  for (let col = 0; col < p; col++) {
    let piv = col
    for (let r = col + 1; r < p; r++) if (Math.abs(A[r]![col]!) > Math.abs(A[piv]![col]!)) piv = r
    ;[A[col], A[piv]] = [A[piv]!, A[col]!]
    ;[I[col], I[piv]] = [I[piv]!, I[col]!]
    const d0 = A[col]![col]!
    if (Math.abs(d0) < 1e-14) continue
    for (let j = 0; j < p; j++) {
      A[col]![j]! /= d0
      I[col]![j]! /= d0
    }
    for (let r = 0; r < p; r++) {
      if (r === col) continue
      const f = A[r]![col]!
      for (let j = 0; j < p; j++) {
        A[r]![j]! -= f * A[col]![j]!
        I[r]![j]! -= f * I[col]![j]!
      }
    }
  }
  return I
}

function coxPartialLL(obs: CoxObs[], beta: number[], p: number): { ll: number; score: number[]; hess: number[][] } {
  const strata = [...new Set(obs.map((o) => o.stratum))]
  let ll = 0
  const score = new Array(p).fill(0)
  const hess = Array.from({ length: p }, () => new Array(p).fill(0))

  for (const st of strata) {
    const sub = obs.filter((o) => o.stratum === st)
    const m = sub.length
    // per-row weights exp(x'β + offset) once per evaluation
    const w = new Float64Array(m)
    const xb = new Float64Array(m)
    for (let i = 0; i < m; i++) {
      let v = sub[i]!.offset
      for (let j = 0; j < p; j++) v += beta[j]! * sub[i]!.x[j]!
      xb[i] = v
      w[i] = Math.exp(Math.max(-50, Math.min(50, v)))
    }
    // events grouped by time, processed in descending time so risk sets grow monotonically:
    // rows enter when stop ≥ t (sorted by stop desc) and leave when start ≥ t (sorted by start desc)
    const dyingAt = new Map<number, number[]>()
    for (let i = 0; i < m; i++) if (sub[i]!.event) {
      const list = dyingAt.get(sub[i]!.stop) ?? []
      list.push(i)
      dyingAt.set(sub[i]!.stop, list)
    }
    const eventTimes = [...dyingAt.keys()].sort((a, b) => b - a)
    const byStop = Array.from({ length: m }, (_, i) => i).sort((a, b) => sub[b]!.stop - sub[a]!.stop)
    const byStart = Array.from({ length: m }, (_, i) => i).sort((a, b) => sub[b]!.start - sub[a]!.start)
    let ps = 0
    let pt = 0
    let S0 = 0
    const S1 = new Float64Array(p)
    const S2 = new Float64Array(p * p)
    const addRow = (i: number, sign: number) => {
      const wi = sign * w[i]!
      const x = sub[i]!.x
      S0 += wi
      for (let j = 0; j < p; j++) {
        S1[j] = S1[j]! + wi * x[j]!
        for (let k = 0; k < p; k++) S2[j * p + k] = S2[j * p + k]! + wi * x[j]! * x[k]!
      }
    }
    for (const t of eventTimes) {
      while (ps < m && sub[byStop[ps]!]!.stop >= t) addRow(byStop[ps++]!, 1)
      while (pt < m && sub[byStart[pt]!]!.start >= t) addRow(byStart[pt++]!, -1)
      const dying = dyingAt.get(t)!
      if (!(S0 > 0)) continue
      const d = dying.length
      for (const di of dying) ll += xb[di]!
      ll -= d * Math.log(S0)
      for (let j = 0; j < p; j++) {
        const mean = S1[j]! / S0
        for (const di of dying) score[j]! += sub[di]!.x[j]! - mean
        for (let k = 0; k < p; k++) {
          const cov = S2[j * p + k]! / S0 - mean * (S1[k]! / S0)
          hess[j]![k]! -= d * cov
        }
      }
    }
  }
  return { ll, score, hess }
}

function fitCoxNewton(
  obs: CoxObs[],
  p: number,
  maxIter: number,
): { beta: number[]; logLik: number; iterations: number; hess: number[][] } {
  const beta = new Array(p).fill(0)
  let iter = 0
  for (iter = 0; iter < maxIter; iter++) {
    const ev = coxPartialLL(obs, beta, p)
    const negH = ev.hess.map((row) => row.map((v) => -v))
    const delta = solveNewton(negH, ev.score)
    let maxStep = 0
    for (let j = 0; j < p; j++) {
      const step = Math.max(-5, Math.min(5, delta[j]!))
      beta[j]! += step
      maxStep = Math.max(maxStep, Math.abs(step))
    }
    if (maxStep < 1e-8) break
  }
  const final = coxPartialLL(obs, beta, p)
  return { beta, logLik: final.ll, iterations: iter + 1, hess: final.hess }
}

/**
 * Cox proportional hazards (Breslow ties).
 * - `censor`: 0 = event, 1 = right-censored
 * - `start`: optional counting-process left endpoint → risk set is (start, stop]
 * - `strata`: stratified baselines, shared β
 * - `frailty: { group }`: shared gamma frailty (EM / penalized offset)
 * - `cluster`: cluster-robust (sandwich) SE from score residuals
 * - `eventType` + `cause`: cause-specific competing risks (other events censored)
 */
export function coxPH(
  time: ArrayLike<number | null | undefined>,
  X: ArrayLike<ArrayLike<number>>,
  options: {
    censor?: ArrayLike<number | null | undefined>
    start?: ArrayLike<number | null | undefined>
    strata?: ArrayLike<string | number | null | undefined>
    frailty?: {
      group: ArrayLike<string | number>
      /** Piecewise time bands (per row); shared gamma frailty estimated within each band×group. */
      bands?: ArrayLike<string | number>
      /** `piecewise` (default) or AR(1) shrinkage across ordered bands. */
      process?: 'piecewise' | 'ar1'
      distribution?: 'gamma'
    }
    cluster?: ArrayLike<string | number | null | undefined>
    /** Competing-risk cause labels per row; used with `cause`. */
    eventType?: ArrayLike<string | number | null | undefined>
    /** Cause of interest for cause-specific Cox (other events treated as censored). */
    cause?: string | number
    names?: string[]
    confidence?: number
    maxIter?: number
  } = {},
): CoxPHResult {
  const tArr = Array.from(time)
  const rowsX = Array.from(X).map((r) => Array.from(r))
  if (tArr.length !== rowsX.length) throw new RangeError('coxPH: time/X length mismatch')
  const causeSpecific = options.eventType != null && options.cause != null
  const causeKey = options.cause != null ? String(options.cause) : ''
  const obs: CoxObs[] = []
  for (let i = 0; i < tArr.length; i++) {
    const stop = tArr[i]
    if (typeof stop !== 'number' || !Number.isFinite(stop) || stop < 0) continue
    const startRaw = options.start ? Number(options.start[i] ?? 0) : 0
    const start = Number.isFinite(startRaw) ? startRaw : 0
    if (start >= stop && options.start) continue
    let c = options.censor ? Number(options.censor[i] ?? 0) : 0
    if (causeSpecific) {
      const et = options.eventType![i]
      if (et === null || et === undefined || et === '') {
        // keep censor from options (or treat as censored)
        if (!options.censor) c = 1
      } else if (String(et) === causeKey) {
        c = 0
      } else {
        // competing event → censored at this time
        c = 1
      }
    }
    const stratum = options.strata != null ? String(options.strata[i] ?? '') : ''
    const frailtyGroup = options.frailty ? String(options.frailty.group[i] ?? '') : ''
    const frailtyBand =
      options.frailty?.bands != null ? String(options.frailty.bands[i] ?? '') : ''
    const cluster = options.cluster != null ? String(options.cluster[i] ?? i) : String(i)
    obs.push({
      start,
      stop,
      event: c === 0,
      x: rowsX[i]!,
      stratum,
      frailtyGroup,
      frailtyBand,
      cluster,
      offset: 0,
      id: i,
    })
  }
  if (obs.length < 4) throw new RangeError('coxPH: need at least 4 observations')
  const p = obs[0]!.x.length
  if (p < 1) throw new RangeError('coxPH: need at least one covariate')
  const names = options.names ?? Array.from({ length: p }, (_, j) => `X${j + 1}`)
  const confidence = options.confidence ?? 0.95
  const maxIter = options.maxIter ?? 50
  const zCrit = STD.ppf(0.5 + confidence / 2)
  const stratified = options.strata != null
  const hasFrailtyBands = options.frailty?.bands != null
  const frailtyProcess = options.frailty?.process ?? 'piecewise'
  if (frailtyProcess === 'ar1' && !hasFrailtyBands) {
    throw new RangeError("coxPH frailty: process 'ar1' requires bands")
  }

  let frailtyVar: number | undefined
  let frailtyRho: number | undefined
  let frailtyOut: Array<{ group: string; blup: number }> | undefined
  let frailtyByBand: Array<{ band: string; group: string; blup: number }> | undefined

  if (options.frailty) {
    // When bands present, frailty units are group×band
    const unitKey = (o: CoxObs) => (hasFrailtyBands ? `${o.frailtyGroup}::${o.frailtyBand}` : o.frailtyGroup)
    const groups = [...new Set(obs.map(unitKey))]
    const g = groups.length
    if (g < 2) throw new RangeError('coxPH frailty: need ≥2 groups (or group×band units)')
    // Gamma frailty EM: w_i = (1/θ + d_i) / (1/θ + Λ_i); θ = frailty variance
    let theta = 0.5
    const wMap = new Map(groups.map((k) => [k, 1]))
    let fit = fitCoxNewton(obs, p, Math.min(20, maxIter))
    for (let em = 0; em < 25; em++) {
      for (const o of obs) o.offset = Math.log(Math.max(1e-8, wMap.get(unitKey(o))!))
      fit = fitCoxNewton(obs, p, Math.min(15, maxIter))
      const Lambda = new Map(groups.map((k) => [k, 0]))
      const deaths = new Map(groups.map((k) => [k, 0]))
      const strataKeys = [...new Set(obs.map((o) => o.stratum))]
      for (const st of strataKeys) {
        const sub = obs.filter((o) => o.stratum === st)
        const eventTimes = [...new Set(sub.filter((o) => o.event).map((o) => o.stop))].sort((a, b) => a - b)
        for (const t of eventTimes) {
          const dying = sub.filter((o) => o.event && o.stop === t)
          const risk = sub.filter((o) => o.start < t && o.stop >= t)
          let S0 = 0
          const riskW = new Map<string, number>()
          for (const r of risk) {
            let xb = r.offset
            for (let j = 0; j < p; j++) xb += fit.beta[j]! * r.x[j]!
            const w = Math.exp(Math.max(-50, Math.min(50, xb)))
            S0 += w
            const uk = unitKey(r)
            riskW.set(uk, (riskW.get(uk) ?? 0) + w)
          }
          if (!(S0 > 0)) continue
          const d = dying.length
          for (const di of dying) deaths.set(unitKey(di), (deaths.get(unitKey(di)) ?? 0) + 1)
          for (const [gk, rw] of riskW) {
            Lambda.set(gk, (Lambda.get(gk) ?? 0) + (d * rw) / S0)
          }
        }
      }
      const ws: number[] = []
      for (const gk of groups) {
        const di = deaths.get(gk) ?? 0
        const Li = Lambda.get(gk) ?? 0
        const wi = (1 / theta + di) / (1 / theta + Li)
        wMap.set(gk, Math.max(1e-6, wi))
        ws.push(wi)
      }
      // AR(1) shrinkage across ordered bands within each group
      if (frailtyProcess === 'ar1' && hasFrailtyBands) {
        const bandOrder = [...new Set(obs.map((o) => o.frailtyBand))]
        // Prefer numeric sort when labels parse as numbers
        const allNum = bandOrder.every((b) => Number.isFinite(Number(b)))
        bandOrder.sort((a, b) => (allNum ? Number(a) - Number(b) : a.localeCompare(b)))
        const groupIds = [...new Set(obs.map((o) => o.frailtyGroup))]
        // MoM ρ from adjacent log-BLUPs
        const pairs: Array<[number, number]> = []
        for (const gk of groupIds) {
          const seq: number[] = []
          for (const b of bandOrder) {
            const key = `${gk}::${b}`
            if (wMap.has(key)) seq.push(Math.log(Math.max(1e-8, wMap.get(key)!)))
          }
          for (let i = 1; i < seq.length; i++) pairs.push([seq[i - 1]!, seq[i]!])
        }
        let rho = 0.5
        if (pairs.length >= 2) {
          const mx = pairs.reduce((a, p0) => a + p0[0], 0) / pairs.length
          const my = pairs.reduce((a, p0) => a + p0[1], 0) / pairs.length
          let num = 0
          let dx = 0
          let dy = 0
          for (const [a, b] of pairs) {
            num += (a - mx) * (b - my)
            dx += (a - mx) ** 2
            dy += (b - my) ** 2
          }
          rho = Math.max(-0.95, Math.min(0.95, num / Math.sqrt(Math.max(1e-12, dx * dy))))
        }
        frailtyRho = rho
        for (const gk of groupIds) {
          const keysB = bandOrder.map((b) => `${gk}::${b}`).filter((k) => wMap.has(k))
          if (keysB.length < 2) continue
          const raw = keysB.map((k) => wMap.get(k)!)
          const shrunk = raw.slice()
          for (let i = 1; i < shrunk.length; i++) {
            shrunk[i] = Math.max(1e-6, rho * shrunk[i - 1]! + (1 - Math.abs(rho)) * raw[i]!)
          }
          for (let i = 0; i < keysB.length; i++) wMap.set(keysB[i]!, shrunk[i]!)
        }
        // refresh ws for theta update
        ws.length = 0
        for (const gk of groups) ws.push(wMap.get(gk)!)
      }
      const meanW = ws.reduce((a, b) => a + b, 0) / ws.length
      let v = 0
      for (const wi of ws) v += (wi - meanW) ** 2
      v /= Math.max(1, ws.length - 1)
      theta = Math.max(1e-4, 0.7 * theta + 0.3 * v)
    }
    frailtyVar = theta
    if (hasFrailtyBands) {
      frailtyByBand = groups.map((key) => {
        const [group, band] = key.split('::')
        return { band: band ?? '', group: group ?? key, blup: wMap.get(key)! }
      })
      const gMap = new Map<string, number[]>()
      for (const row of frailtyByBand) {
        if (!gMap.has(row.group)) gMap.set(row.group, [])
        gMap.get(row.group)!.push(row.blup)
      }
      frailtyOut = [...gMap.entries()].map(([group, vals]) => ({
        group,
        blup: vals.reduce((a, b) => a + b, 0) / vals.length,
      }))
    } else {
      frailtyOut = groups.map((group) => ({ group, blup: wMap.get(group)! }))
    }
    for (const o of obs) o.offset = Math.log(Math.max(1e-8, wMap.get(unitKey(o))!))
  }

  const fit = fitCoxNewton(obs, p, maxIter)
  const negH = fit.hess.map((row) => row.map((v) => -v))
  const I = invertInfo(negH)
  const seModel = I.map((row, i) => Math.sqrt(Math.max(0, row[i]!)))
  let se = seModel.slice()
  let robust = false

  if (options.cluster != null) {
    // Score residuals → cluster sandwich
    const U = obs.map(() => new Array(p).fill(0))
    const strataKeys = [...new Set(obs.map((o) => o.stratum))]
    for (const st of strataKeys) {
      const sub = obs.filter((o) => o.stratum === st)
      const eventTimes = [...new Set(sub.filter((o) => o.event).map((o) => o.stop))].sort((a, b) => a - b)
      for (const t of eventTimes) {
        const dying = sub.filter((o) => o.event && o.stop === t)
        const risk = sub.filter((o) => o.start < t && o.stop >= t)
        let S0 = 0
        const S1 = new Array(p).fill(0)
        const weights: number[] = []
        for (const r of risk) {
          let xb = r.offset
          for (let j = 0; j < p; j++) xb += fit.beta[j]! * r.x[j]!
          const w = Math.exp(Math.max(-50, Math.min(50, xb)))
          weights.push(w)
          S0 += w
          for (let j = 0; j < p; j++) S1[j]! += w * r.x[j]!
        }
        if (!(S0 > 0)) continue
        const mean = S1.map((s) => s / S0)
        const d = dying.length
        for (const di of dying) {
          const ii = obs.indexOf(di)
          for (let j = 0; j < p; j++) U[ii]![j]! += di.x[j]! - mean[j]!
        }
        for (let ri = 0; ri < risk.length; ri++) {
          const r = risk[ri]!
          const ii = obs.indexOf(r)
          const w = weights[ri]!
          for (let j = 0; j < p; j++) U[ii]![j]! -= (d * w * (r.x[j]! - mean[j]!)) / S0
        }
      }
    }
    const clusters = [...new Set(obs.map((o) => o.cluster))]
    const meat = Array.from({ length: p }, () => new Array(p).fill(0))
    for (const ck of clusters) {
      const sumU = new Array(p).fill(0)
      for (let i = 0; i < obs.length; i++) {
        if (obs[i]!.cluster !== ck) continue
        for (let j = 0; j < p; j++) sumU[j]! += U[i]![j]!
      }
      for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) meat[a]![b]! += sumU[a]! * sumU[b]!
    }
    // Sandwich = I^{-1} Meat I^{-1}
    const bread = I
    const tmp = Array.from({ length: p }, () => new Array(p).fill(0))
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) {
      let s = 0
      for (let k = 0; k < p; k++) s += bread[a]![k]! * meat[k]![b]!
      tmp[a]![b] = s
    }
    const sand = Array.from({ length: p }, () => new Array(p).fill(0))
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) {
      let s = 0
      for (let k = 0; k < p; k++) s += tmp[a]![k]! * bread[k]![b]!
      sand[a]![b] = s
    }
    se = sand.map((row, i) => Math.sqrt(Math.max(0, row[i]!)))
    robust = true
  }

  const z = fit.beta.map((b, i) => (se[i]! > 0 ? b / se[i]! : NaN))
  const pValue = z.map((zi) => (Number.isFinite(zi) ? 2 * STD.sf(Math.abs(zi)) : NaN))
  const ci = fit.beta.map((b, i) => [b - zCrit * se[i]!, b + zCrit * se[i]!] as [number, number])
  const hr = fit.beta.map((b) => Math.exp(b))
  const nEvents = obs.filter((o) => o.event).length

  return {
    coefficients: fit.beta,
    se,
    seModel,
    z,
    pValue,
    ci,
    hr,
    logLik: fit.logLik,
    n: obs.length,
    nEvents,
    names,
    iterations: fit.iterations,
    ties: 'breslow',
    stratified,
    robust,
    causeSpecific: causeSpecific || undefined,
    timeVaryingFrailty:
      options.frailty != null &&
      (options.start != null || hasFrailtyBands || frailtyProcess === 'ar1')
        ? true
        : undefined,
    frailtyVar,
    frailtyRho,
    frailty: frailtyOut,
    frailtyByBand,
  }
}

export interface FineGrayResult {
  coefficients: number[]
  se: number[]
  z: number[]
  pValue: number[]
  ci: Array<[number, number]>
  hr: number[]
  logLik: number
  n: number
  nEvents: number
  names: string[]
  cause: string
  /** Cumulative incidence function points for the cause of interest. */
  cif: Array<{ time: number; cif: number }>
  iterations: number
}

/**
 * Fine–Gray subdistribution hazard model for competing risks.
 * Other causes remain in the risk set with IPCW-style weights (simplified Breslow FG).
 */
export function fineGray(
  time: ArrayLike<number | null | undefined>,
  X: ArrayLike<ArrayLike<number>>,
  options: {
    eventType: ArrayLike<string | number | null | undefined>
    cause: string | number
    censor?: ArrayLike<number | null | undefined>
    names?: string[]
    confidence?: number
    maxIter?: number
  },
): FineGrayResult {
  const tArr = Array.from(time)
  const rowsX = Array.from(X).map((r) => Array.from(r))
  const etArr = Array.from(options.eventType)
  if (tArr.length !== rowsX.length || tArr.length !== etArr.length) {
    throw new RangeError('fineGray: time/X/eventType length mismatch')
  }
  const causeKey = String(options.cause)
  const confidence = options.confidence ?? 0.95
  const maxIter = options.maxIter ?? 40
  const zCrit = STD.ppf(0.5 + confidence / 2)

  type FGObs = { time: number; causeEvent: boolean; otherEvent: boolean; x: number[]; weight: number }
  const raw: FGObs[] = []
  for (let i = 0; i < tArr.length; i++) {
    const ti = tArr[i]
    if (typeof ti !== 'number' || !Number.isFinite(ti) || ti < 0) continue
    const c = options.censor ? Number(options.censor[i] ?? 0) : 0
    const et = etArr[i]
    const isCens = c !== 0 || et === null || et === undefined || et === ''
    const causeEvent = !isCens && String(et) === causeKey
    const otherEvent = !isCens && String(et) !== causeKey
    raw.push({ time: ti, causeEvent, otherEvent, x: rowsX[i]!, weight: 1 })
  }
  if (raw.length < 4) throw new RangeError('fineGray: need ≥4 observations')
  const p = raw[0]!.x.length
  if (p < 1) throw new RangeError('fineGray: need ≥1 covariate')
  const names = options.names ?? Array.from({ length: p }, (_, j) => `X${j + 1}`)

  // KM censoring distribution for IPCW of competing events (treat cause+other as events for G)
  const timesAsc = raw.map((o) => o.time).sort((a, b) => a - b)
  const uniqueTimes = [...new Set(timesAsc)]
  // Simplified FG: weight for subjects with other events after their event time stays 1 until end
  // (classical Fine–Gray keeps them with weight decreasing by censoring KM — use Ĝ from censoring only)
  const censTimes = raw.filter((o) => !o.causeEvent && !o.otherEvent).map((o) => o.time)
  // Build censoring KM S_c
  const allT = [...new Set(raw.map((o) => o.time))].sort((a, b) => a - b)
  const Sc = new Map<number, number>()
  const countAt = new Map<number, number>()
  const censAt = new Map<number, number>()
  const causeAt = new Map<number, number>()
  const otherAt = new Map<number, number>()
  for (const o of raw) {
    countAt.set(o.time, (countAt.get(o.time) ?? 0) + 1)
    if (!o.causeEvent && !o.otherEvent) censAt.set(o.time, (censAt.get(o.time) ?? 0) + 1)
    if (o.causeEvent) causeAt.set(o.time, (causeAt.get(o.time) ?? 0) + 1)
    if (o.otherEvent) otherAt.set(o.time, (otherAt.get(o.time) ?? 0) + 1)
  }
  let atRisk = raw.length
  let surv = 1
  for (const t of allT) {
    const dC = censAt.get(t) ?? 0
    if (atRisk > 0 && dC > 0) surv *= 1 - dC / atRisk
    Sc.set(t, surv)
    atRisk -= countAt.get(t) ?? 0
  }

  // Expand: for Fine–Gray score, risk set includes cause-free subjects + competing with weight Sc(t)/Sc(T_i)
  const fgPartial = (beta: number[]) => {
    let ll = 0
    const score = new Array(p).fill(0)
    const hess = Array.from({ length: p }, () => new Array(p).fill(0))
    const eventTimes = [...new Set(raw.filter((o) => o.causeEvent).map((o) => o.time))].sort((a, b) => a - b)
    for (const t of eventTimes) {
      const dying = raw.filter((o) => o.causeEvent && o.time === t)
      // risk: not yet failed from cause; competing stay with weight
      const risk: Array<{ o: FGObs; w: number }> = []
      for (const o of raw) {
        if (o.causeEvent && o.time < t) continue
        if (o.causeEvent && o.time === t) {
          risk.push({ o, w: 1 })
          continue
        }
        if (o.otherEvent && o.time < t) {
          const scT = Sc.get(t) ?? 1
          const scTi = Sc.get(o.time) ?? 1
          const w = scTi > 1e-12 ? Math.min(1, scT / scTi) : 0
          if (w > 0) risk.push({ o, w })
          continue
        }
        if (!o.causeEvent && !o.otherEvent && o.time >= t) risk.push({ o, w: 1 })
        if (!o.causeEvent && !o.otherEvent && o.time < t) continue
        if (o.otherEvent && o.time >= t) risk.push({ o, w: 1 })
      }
      if (dying.length < 1 || risk.length < 1) continue
      let S0 = 0
      const S1 = new Array(p).fill(0)
      const S2 = Array.from({ length: p }, () => new Array(p).fill(0))
      for (const { o, w } of risk) {
        let xb = 0
        for (let j = 0; j < p; j++) xb += beta[j]! * o.x[j]!
        const ew = w * Math.exp(Math.max(-50, Math.min(50, xb)))
        S0 += ew
        for (let j = 0; j < p; j++) {
          S1[j]! += ew * o.x[j]!
          for (let k = 0; k < p; k++) S2[j]![k]! += ew * o.x[j]! * o.x[k]!
        }
      }
      if (!(S0 > 0)) continue
      const d = dying.length
      for (const di of dying) {
        let xb = 0
        for (let j = 0; j < p; j++) xb += beta[j]! * di.x[j]!
        ll += xb
      }
      ll -= d * Math.log(S0)
      for (let j = 0; j < p; j++) {
        const mean = S1[j]! / S0
        for (const di of dying) score[j]! += di.x[j]! - mean
        for (let k = 0; k < p; k++) {
          const cov = S2[j]![k]! / S0 - (S1[j]! / S0) * (S1[k]! / S0)
          hess[j]![k]! -= d * cov
        }
      }
    }
    return { ll, score, hess }
  }

  const beta = new Array(p).fill(0)
  let iter = 0
  for (iter = 0; iter < maxIter; iter++) {
    const ev = fgPartial(beta)
    const negH = ev.hess.map((row) => row.map((v) => -v))
    const delta = solveNewton(negH, ev.score)
    let maxStep = 0
    for (let j = 0; j < p; j++) {
      const step = Math.max(-5, Math.min(5, delta[j]!))
      beta[j]! += step
      maxStep = Math.max(maxStep, Math.abs(step))
    }
    if (maxStep < 1e-8) break
  }
  const final = fgPartial(beta)
  const I = invertInfo(final.hess.map((row) => row.map((v) => -v)))
  const se = I.map((row, i) => Math.sqrt(Math.max(0, row[i]!)))
  const z = beta.map((b, i) => (se[i]! > 0 ? b / se[i]! : NaN))
  const pValue = z.map((zi) => (Number.isFinite(zi) ? 2 * STD.sf(Math.abs(zi)) : NaN))
  const ci = beta.map((b, i) => [b - zCrit * se[i]!, b + zCrit * se[i]!] as [number, number])
  const hr = beta.map((b) => Math.exp(b))

  // CIF via Aalen–Johansen lite (cause-specific cumulative incidence without covariates for baseline)
  const cif: Array<{ time: number; cif: number }> = []
  let sAll = 1
  let cifCum = 0
  let nRisk = raw.length
  for (const t of uniqueTimes) {
    const dCause = causeAt.get(t) ?? 0
    const dOther = otherAt.get(t) ?? 0
    const d = dCause + dOther
    if (nRisk > 0 && dCause > 0) {
      cifCum += sAll * (dCause / nRisk)
      cif.push({ time: t, cif: Math.min(1, cifCum) })
    }
    if (nRisk > 0 && d > 0) sAll *= 1 - d / nRisk
    nRisk -= countAt.get(t) ?? 0
  }

  void censTimes
  return {
    coefficients: beta,
    se,
    z,
    pValue,
    ci,
    hr,
    logLik: final.ll,
    n: raw.length,
    nEvents: raw.filter((o) => o.causeEvent).length,
    names,
    cause: causeKey,
    cif,
    iterations: iter + 1,
  }
}