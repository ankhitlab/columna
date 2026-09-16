/**
 * Life-data extensions (Minitab Stat › Reliability/Survival): Regression with Life Data (accelerated
 * failure-time models with right / left / interval censoring), Accelerated Life Testing (Arrhenius,
 * inverse power, exponential relations with use-condition predictions), Test Plans (demonstration and
 * estimation), Repairable Systems (power-law NHPP / Crow–AMSAA with trend tests), and Probit Analysis
 * (dose–response percentiles with Fieller intervals and natural response).
 */
import { chi2 as chi2Dist, logistic as logisticDist, normal, smallestExtremeValue } from './dist.js'
import { glm } from './glm.js'
import { matmul, matrix, matvec, transpose, type Matrix } from './linalg.js'
import { newtonMax } from './optim.js'
import { completeRows, designMatrix, toColumns, type Column, type Predictors } from './regression.js'

const STD = normal()

export type LifeDistribution = 'weibull' | 'lognormal' | 'loglogistic' | 'exponential' | 'smallest-extreme-value' | 'normal' | 'logistic'

/** Location-scale family of the (log) response for each life distribution. */
function family(d: LifeDistribution): { base: 'sev' | 'normal' | 'logistic'; logScale: boolean; fixedSigma?: number } {
  switch (d) {
    case 'weibull':
      return { base: 'sev', logScale: true }
    case 'exponential':
      return { base: 'sev', logScale: true, fixedSigma: 1 }
    case 'lognormal':
      return { base: 'normal', logScale: true }
    case 'loglogistic':
      return { base: 'logistic', logScale: true }
    case 'smallest-extreme-value':
      return { base: 'sev', logScale: false }
    case 'normal':
      return { base: 'normal', logScale: false }
    case 'logistic':
      return { base: 'logistic', logScale: false }
  }
}

const baseDist = { sev: smallestExtremeValue(), normal: STD, logistic: logisticDist() }

export interface LifeCoefficient {
  name: string
  coef: number
  se: number
  z: number
  pValue: number
  ci: [number, number]
}

export interface LifeRegressionResult {
  test: 'regression with life data'
  distribution: LifeDistribution
  /** Regression coefficients on the (log) location: intercept first. */
  coefficients: LifeCoefficient[]
  /** Scale σ (Weibull shape = 1/σ) with SE and CI. */
  scale: { estimate: number; se: number; ci: [number, number] }
  shape?: number
  logLik: number
  n: number
  nFailures: number
  aic: number
  bic: number
  covariance: Matrix
  converged: boolean
  iterations: number
  /** Standardized residuals (on the location-scale) and Cox–Snell residuals. */
  residuals: { standardized: Float64Array; coxSnell: Float64Array }
  /** Percentiles at a covariate setting: t_p with delta-method CI. */
  percentile(p: number, x: ArrayLike<number>, options?: { confidence?: number }): { p: number; time: number; se: number; ci: [number, number] }
  /** Survival probability at a time and covariate setting. */
  survival(time: number, x: ArrayLike<number>): number
}

interface Obs {
  y: number
  y2?: number
  censor: 0 | 1 | 2 | 3
}

/**
 * Regression with Life Data (Minitab): log(T) = β₀ + x'β + σ·ε with ε from the smallest-extreme-value
 * (Weibull / exponential), normal (lognormal) or logistic (loglogistic) distribution; also the
 * location-scale (non-log) variants. Censoring: 0 exact, 1 right, 2 left, 3 interval (`time2` upper end).
 */
export function lifeRegression(
  time: Column,
  X: Predictors,
  options: { distribution?: LifeDistribution; censor?: Column; time2?: Column; names?: string[]; confidence?: number; intercept?: boolean } = {},
): LifeRegressionResult {
  const distribution = options.distribution ?? 'weibull'
  const fam = family(distribution)
  const confidence = options.confidence ?? 0.95
  const { names: xnames, cols } = toColumns(X, options.names)
  const { keep } = completeRows(time, cols)
  const n = keep.length
  const D = designMatrix(cols, keep, options.intercept ?? true)
  const p = D.cols
  // fit on standardized predictors (Newton is scale-sensitive), then map back to the original scale
  const hasIntercept = options.intercept ?? true
  const colMean = new Float64Array(p)
  const colSd = new Float64Array(p).fill(1)
  const Ds = matrix(D.rows, p, Float64Array.from(D.data))
  for (let j = hasIntercept ? 1 : 0; j < p; j++) {
    let m = 0
    for (let i = 0; i < D.rows; i++) m += D.data[i * p + j]!
    m /= D.rows
    let v = 0
    for (let i = 0; i < D.rows; i++) v += (D.data[i * p + j]! - m) ** 2
    const sd = Math.sqrt(v / Math.max(1, D.rows - 1)) || 1
    colMean[j] = hasIntercept ? m : 0
    colSd[j] = sd
    for (let i = 0; i < D.rows; i++) Ds.data[i * p + j] = (D.data[i * p + j]! - colMean[j]!) / sd
  }
  const obs: Obs[] = keep.map((i) => {
    const t = time[i] as number
    const c = (options.censor ? Number(options.censor[i] ?? 0) : 0) as 0 | 1 | 2 | 3
    const t2 = options.time2 ? Number(options.time2[i]) : undefined
    if (fam.logScale && !(t > 0)) throw new RangeError('lifeRegression: times must be positive for log-location-scale distributions')
    if (c === 3 && !(t2 !== undefined && t2 > t)) throw new RangeError('lifeRegression: interval-censored rows need time2 > time')
    return { y: fam.logScale ? Math.log(t) : t, y2: t2 !== undefined ? (fam.logScale ? Math.log(t2) : t2) : undefined, censor: c }
  })
  const nFailures = obs.filter((o) => o.censor === 0).length
  const base = baseDist[fam.base]
  const k = fam.fixedSigma ? p : p + 1
  const loglik = (theta: Float64Array): number => {
    const sigma = fam.fixedSigma ?? Math.exp(theta[p]!)
    let ll = 0
    for (let i = 0; i < n; i++) {
      let mu = 0
      for (let j = 0; j < p; j++) mu += Ds.data[i * p + j]! * theta[j]!
      const o = obs[i]!
      const z = (o.y - mu) / sigma
      if (o.censor === 0) ll += Math.log(base.pdf(z)) - Math.log(sigma)
      else if (o.censor === 1) ll += Math.log(base.sf(z))
      else if (o.censor === 2) ll += Math.log(base.cdf(z))
      else ll += Math.log(Math.max(1e-300, base.sf(z) - base.sf((o.y2! - mu) / sigma)))
    }
    return ll
  }
  // analytic score: ψ(z) = f′(z)/f(z) for exact rows; hazard-type ratios for censored rows
  const psi = fam.base === 'sev' ? (z: number) => 1 - Math.exp(z) : fam.base === 'normal' ? (z: number) => -z : (z: number) => 1 - 2 * base.cdf(z)
  const grad = (theta: Float64Array): Float64Array => {
    const sigma = fam.fixedSigma ?? Math.exp(theta[p]!)
    const g = new Float64Array(k)
    for (let i = 0; i < n; i++) {
      let mu = 0
      for (let j = 0; j < p; j++) mu += Ds.data[i * p + j]! * theta[j]!
      const o = obs[i]!
      const z = (o.y - mu) / sigma
      let dMu: number // ∂ℓ/∂μ
      let dLs: number // ∂ℓ/∂log σ
      if (o.censor === 0) {
        const ps = psi(z)
        dMu = -ps / sigma
        dLs = -z * ps - 1
      } else if (o.censor === 1) {
        const h = base.pdf(z) / Math.max(1e-300, base.sf(z))
        dMu = h / sigma
        dLs = z * h
      } else if (o.censor === 2) {
        const h = base.pdf(z) / Math.max(1e-300, base.cdf(z))
        dMu = -h / sigma
        dLs = -z * h
      } else {
        const z2 = (o.y2! - mu) / sigma
        const den = Math.max(1e-300, base.sf(z) - base.sf(z2))
        dMu = (base.pdf(z) - base.pdf(z2)) / (sigma * den)
        dLs = (z * base.pdf(z) - z2 * base.pdf(z2)) / den
      }
      for (let j = 0; j < p; j++) g[j] = g[j]! + dMu * Ds.data[i * p + j]!
      if (!fam.fixedSigma) g[p] = g[p]! + dLs
    }
    return g
  }
  // start: least squares on exact/right times, σ from residual sd
  const start = new Float64Array(k)
  let ybar = 0
  for (const o of obs) ybar += o.y
  ybar /= n
  if (options.intercept ?? true) start[0] = ybar
  let sd = 0
  for (const o of obs) sd += (o.y - ybar) ** 2
  sd = Math.sqrt(sd / Math.max(1, n - 1)) || 1
  if (!fam.fixedSigma) start[p] = Math.log(sd)
  const fitStd = newtonMax(loglik, start, { grad })
  // θ_orig = T θ_std: b_j = b_j' / s_j, b_0 = b_0' − Σ b_j' m_j / s_j; log σ unchanged
  const T = matrix(k, k)
  for (let a = 0; a < k; a++) T.data[a * k + a] = 1
  for (let j = hasIntercept ? 1 : 0; j < p; j++) {
    T.data[j * k + j] = 1 / colSd[j]!
    if (hasIntercept) T.data[0 * k + j] = -colMean[j]! / colSd[j]!
  }
  const thetaOrig = matvec(T, fitStd.theta)
  const covOrig = matmul(matmul(T, fitStd.covariance), transpose(T))
  const fit = { ...fitStd, theta: thetaOrig, covariance: covOrig }
  const sigma = fam.fixedSigma ?? Math.exp(fit.theta[p]!)
  const zc = STD.ppf(0.5 + confidence / 2)
  const names = [...((options.intercept ?? true) ? ['Intercept'] : []), ...xnames]
  const coefficients: LifeCoefficient[] = names.map((name, j) => {
    const b = fit.theta[j]!
    const se = Math.sqrt(Math.max(0, fit.covariance.data[j * k + j]!))
    const z = b / se
    return { name, coef: b, se, z, pValue: Math.min(1, 2 * STD.sf(Math.abs(z))), ci: [b - zc * se, b + zc * se] }
  })
  const seLogSigma = fam.fixedSigma ? 0 : Math.sqrt(Math.max(0, fit.covariance.data[p * k + p]!))
  const scale = { estimate: sigma, se: sigma * seLogSigma, ci: [sigma * Math.exp(-zc * seLogSigma), sigma * Math.exp(zc * seLogSigma)] as [number, number] }
  const standardized = new Float64Array(n)
  const coxSnell = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let mu = 0
    for (let j = 0; j < p; j++) mu += D.data[i * p + j]! * fit.theta[j]!
    const z = (obs[i]!.y - mu) / sigma
    standardized[i] = z
    coxSnell[i] = -Math.log(base.sf(z))
  }
  const rowOf = (x: ArrayLike<number>): Float64Array => {
    if (x.length !== xnames.length) throw new RangeError(`expected ${xnames.length} predictor values, got ${x.length}`)
    const v = new Float64Array(p)
    let j = 0
    if (options.intercept ?? true) v[j++] = 1
    for (let c = 0; c < x.length; c++) v[j++] = x[c]!
    return v
  }
  const percentile = (q: number, x: ArrayLike<number>, o: { confidence?: number } = {}) => {
    const v = rowOf(x)
    let mu = 0
    for (let j = 0; j < p; j++) mu += v[j]! * fit.theta[j]!
    const zq = base.ppf(q)
    const yq = mu + sigma * zq
    // gradient wrt (β, log σ)
    const g = new Float64Array(k)
    for (let j = 0; j < p; j++) g[j] = v[j]!
    if (!fam.fixedSigma) g[p] = sigma * zq
    let varY = 0
    for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) varY += g[a]! * fit.covariance.data[a * k + b]! * g[b]!
    const seY = Math.sqrt(Math.max(0, varY))
    const c = STD.ppf(0.5 + (o.confidence ?? confidence) / 2)
    const tr = (y: number) => (fam.logScale ? Math.exp(y) : y)
    return { p: q, time: tr(yq), se: fam.logScale ? tr(yq) * seY : seY, ci: [tr(yq - c * seY), tr(yq + c * seY)] as [number, number] }
  }
  const survival = (t: number, x: ArrayLike<number>) => {
    const v = rowOf(x)
    let mu = 0
    for (let j = 0; j < p; j++) mu += v[j]! * fit.theta[j]!
    return base.sf(((fam.logScale ? Math.log(t) : t) - mu) / sigma)
  }
  return {
    test: 'regression with life data',
    distribution,
    coefficients,
    scale,
    shape: distribution === 'weibull' ? 1 / sigma : undefined,
    logLik: fit.logLik,
    n,
    nFailures,
    aic: -2 * fit.logLik + 2 * k,
    bic: -2 * fit.logLik + k * Math.log(n),
    covariance: fit.covariance,
    converged: fit.converged,
    iterations: fit.iterations,
    residuals: { standardized, coxSnell },
    percentile,
    survival,
  }
}

export type AccelerationRelation = 'arrhenius' | 'inverse-power' | 'exponential' | 'linear'

export interface AltResult extends LifeRegressionResult {
  relation: AccelerationRelation
  /** Transformed stress used as the regressor (e.g. 11604.83 / (T °C + 273.15) for Arrhenius). */
  transform(stress: number): number
  /** Acceleration factor between two stress levels. */
  accelerationFactor(stress: number, useStress: number): number
  /** Percentiles at the use stress (when given): p → time with CI. */
  usePercentiles?: Array<{ p: number; time: number; ci: [number, number] }>
}

/**
 * Accelerated Life Testing (Minitab): life regression on a transformed stress —
 * Arrhenius (11604.83 / K), inverse power (ln stress), exponential / linear (stress) — with percentiles
 * at the use condition. Temperatures for Arrhenius are in °C (converted to Kelvin).
 */
export function altRegression(
  time: Column,
  stress: Column,
  options: { relation?: AccelerationRelation; distribution?: LifeDistribution; censor?: Column; time2?: Column; useStress?: number; percentiles?: number[]; confidence?: number } = {},
): AltResult {
  const relation = options.relation ?? 'arrhenius'
  const transform = (s: number) => (relation === 'arrhenius' ? 11604.83 / (s + 273.15) : relation === 'inverse-power' ? Math.log(s) : s)
  const xs = Array.from({ length: stress.length }, (_, i) => (typeof stress[i] === 'number' && Number.isFinite(stress[i] as number) ? transform(stress[i] as number) : null))
  const fit = lifeRegression(time, { stress: xs }, { distribution: options.distribution, censor: options.censor, time2: options.time2, confidence: options.confidence })
  const beta = fit.coefficients[1]!.coef
  const fam = family(fit.distribution)
  const accelerationFactor = (s: number, use: number) => (fam.logScale ? Math.exp(beta * (transform(use) - transform(s))) : NaN)
  let usePercentiles: AltResult['usePercentiles']
  if (options.useStress !== undefined) {
    const u = transform(options.useStress)
    usePercentiles = (options.percentiles ?? [0.01, 0.05, 0.1, 0.5]).map((p) => {
      const r = fit.percentile(p, [u])
      return { p, time: r.time, ci: r.ci }
    })
  }
  return { ...fit, relation, transform, accelerationFactor, usePercentiles }
}

export interface DemonstrationPlan {
  /** Required sample size for a zero-failure (or `allowedFailures`) demonstration. */
  sampleSize: number
  /** Or, for a fixed sample size: the required test duration per unit. */
  testTime?: number
  reliability: number
  confidence: number
  shape: number
  allowedFailures: number
}

/**
 * Demonstration Test Plan (Minitab): Weibull with known shape β — to demonstrate reliability R at
 * time t₀ with confidence C using a test of duration T per unit and c allowed failures, the sample size
 * solves Σ_{i≤c} C(n,i) qⁱ (1−q)^{n−i} = 1 − C with q = 1 − R^{(T/t₀)^β}. Give `sampleSize` instead
 * to solve for the test time.
 */
export function demonstrationTestPlan(options: {
  reliability: number
  time: number
  confidence?: number
  shape?: number
  testTime?: number
  sampleSize?: number
  allowedFailures?: number
}): DemonstrationPlan {
  const { reliability, time } = options
  if (!(reliability > 0 && reliability < 1)) throw new RangeError('demonstrationTestPlan: reliability in (0, 1)')
  const confidence = options.confidence ?? 0.95
  const shape = options.shape ?? 1
  const c = options.allowedFailures ?? 0
  const qOf = (T: number) => 1 - reliability ** ((T / time) ** shape)
  const passProb = (n: number, q: number) => {
    // P(≤ c failures)
    let s = 0
    for (let i = 0; i <= c; i++) s += Math.exp(lchoose(n, i) + i * Math.log(q) + (n - i) * Math.log(1 - q))
    return s
  }
  if (options.sampleSize !== undefined) {
    const n = options.sampleSize
    // solve T: passProb(n, q(T)) = 1 − C
    let lo = 1e-9
    let hi = time
    while (passProb(n, qOf(hi)) > 1 - confidence && hi < 1e9) hi *= 2
    for (let i = 0; i < 200; i++) {
      const mid = 0.5 * (lo + hi)
      if (passProb(n, qOf(mid)) > 1 - confidence) lo = mid
      else hi = mid
    }
    return { sampleSize: n, testTime: hi, reliability, confidence, shape, allowedFailures: c }
  }
  const T = options.testTime ?? time
  const q = qOf(T)
  let n = c + 1
  while (passProb(n, q) > 1 - confidence && n < 1e7) n++
  return { sampleSize: n, testTime: T, reliability, confidence, shape, allowedFailures: c }
}

function lchoose(n: number, k: number): number {
  let s = 0
  for (let i = 1; i <= k; i++) s += Math.log(n - k + i) - Math.log(i)
  return s
}

export interface EstimationPlan {
  sampleSize: number
  /** Expected number of failures. */
  expectedFailures: number
  /** Achieved relative half-width (upper / estimate ratio) of the CI for the target percentile. */
  precision: number
  percentile: number
  confidence: number
}

/**
 * Estimation Test Plan (Minitab, Weibull with planning values): sample size so that the two-sided
 * confidence interval for the p-th percentile has ratio upper/lower ≤ `ratio`, under type I censoring
 * at `censorTime` — via the large-sample variance of the log percentile from the expected Fisher
 * information (computed by Monte-Carlo-free numerical integration of the censored likelihood).
 */
export function estimationTestPlan(options: { shape: number; scale: number; percentile?: number; ratio: number; censorTime: number; confidence?: number }): EstimationPlan {
  const { shape, scale, censorTime, ratio } = options
  const p = options.percentile ?? 0.1
  const confidence = options.confidence ?? 0.95
  if (!(ratio > 1)) throw new RangeError('estimationTestPlan: ratio must be > 1')
  // Fisher information per unit for (μ = ln η, log σ) of the SEV model with type I censoring at ln(censorTime)
  const mu = Math.log(scale)
  const sigma = 1 / shape
  const zc = (Math.log(censorTime) - mu) / sigma
  const sev = smallestExtremeValue()
  // numerical integration on z ∈ (−∞, zc]
  const grid = 4000
  const lo = -12
  const h = (zc - lo) / grid
  let i11 = 0
  let i12 = 0
  let i22 = 0
  for (let g = 0; g < grid; g++) {
    const z = lo + (g + 0.5) * h
    const f = sev.pdf(z)
    // score components for exact observations: ∂ℓ/∂μ = (e^z − 1)/σ, ∂ℓ/∂logσ = z(e^z − 1) − 1
    const ez = Math.exp(z)
    const sMu = (ez - 1) / sigma
    const sLs = z * (ez - 1) - 1
    i11 += sMu * sMu * f * h
    i12 += sMu * sLs * f * h
    i22 += sLs * sLs * f * h
  }
  // censored contribution: P(z > zc) with score of log S(z) = −e^z: ∂/∂μ = e^{zc}/σ, ∂/∂logσ = zc e^{zc}
  const S = sev.sf(zc)
  const ezc = Math.exp(zc)
  i11 += ((ezc / sigma) ** 2) * S
  i12 += (ezc / sigma) * (zc * ezc) * S
  i22 += (zc * ezc) ** 2 * S
  const det = i11 * i22 - i12 * i12
  const inv = [[i22 / det, -i12 / det], [-i12 / det, i11 / det]]
  // log percentile y_p = μ + σ z_p; gradient (1, σ z_p)
  const zp = sev.ppf(p)
  const gvec = [1, sigma * zp]
  const varPerUnit = gvec[0]! * gvec[0]! * inv[0]![0]! + 2 * gvec[0]! * gvec[1]! * inv[0]![1]! + gvec[1]! * gvec[1]! * inv[1]![1]!
  const z = STD.ppf(0.5 + confidence / 2)
  // ratio upper/lower = exp(2 z se) with se = sqrt(var/n)
  const n = Math.ceil((2 * z) ** 2 * varPerUnit / Math.log(ratio) ** 2)
  const se = Math.sqrt(varPerUnit / n)
  return { sampleSize: n, expectedFailures: n * (1 - S), precision: Math.exp(2 * z * se), percentile: p, confidence }
}

export interface NhppResult {
  test: 'power-law NHPP (Crow–AMSAA)'
  /** Shape β (< 1 improving, > 1 deteriorating) and scale λ of the intensity λβt^{β−1}. */
  shape: number
  scale: number
  se: { shape: number; scale: number }
  ci: { shape: [number, number]; scale: [number, number] }
  n: number
  systems: number
  /** Total observation time. */
  totalTime: number
  /** Instantaneous MTBF at the end of observation. */
  mtbf: number
  /** Trend tests: Laplace / MIL-HDBK-189 (H0: HPP). */
  trend: { laplace: { statistic: number; pValue: number }; milHdbk: { statistic: number; df: number; pValue: number } }
  /** Total time on test plot data. */
  ttt: Array<{ u: number; scaled: number }>
  /** Expected cumulative failures at t. */
  cumulative(t: number): number
}

/**
 * Repairable Systems (Minitab Parametric Growth Curve): power-law NHPP fitted by MLE to failure times
 * of one or more systems (each observed on [0, T_k], time-truncated), with Laplace and MIL-HDBK-189
 * trend tests and TTT-plot data.
 */
export function powerLawNHPP(
  times: ArrayLike<number> | ArrayLike<ArrayLike<number>>,
  options: { endTime: number | ArrayLike<number>; confidence?: number } = { endTime: NaN },
): NhppResult {
  const systems: number[][] = times.length === 0 || typeof (times as ArrayLike<unknown>)[0] === 'number' ? [Array.from(times as ArrayLike<number>)] : Array.from(times as ArrayLike<ArrayLike<number>>).map((sys) => Array.from(sys))
  const ends = typeof options.endTime === 'number' ? systems.map(() => options.endTime as number) : Array.from(options.endTime as ArrayLike<number>)
  if (ends.length !== systems.length) throw new RangeError('powerLawNHPP: one end time per system')
  const confidence = options.confidence ?? 0.95
  let n = 0
  let sumLog = 0
  let totalTime = 0
  systems.forEach((s, k) => {
    const T = ends[k]!
    if (!(T > 0)) throw new RangeError('powerLawNHPP: end times must be positive')
    for (const t of s) {
      if (!(t > 0 && t <= T)) throw new RangeError('powerLawNHPP: failure times must lie in (0, endTime]')
      n++
      sumLog += Math.log(T / t)
    }
    totalTime += T
  })
  if (n < 2) throw new RangeError('powerLawNHPP needs at least 2 failures')
  // MLE (time truncated): β = n / Σ ln(T_k/t_ki), λ = n / Σ T_k^β
  const shape = n / sumLog
  let sumTb = 0
  for (const T of ends) sumTb += T ** shape
  const scale = n / sumTb
  const seShape = shape / Math.sqrt(n)
  const z = STD.ppf(0.5 + confidence / 2)
  // observed information at the MLE in θ = (log β, log λ), closed form:
  //   ℓ = n(log λ + log β) + (β − 1) Σ log tᵢ − λ Σ Tₖ^β
  let sumLogT = 0
  for (const sys of systems) for (const t of sys) sumLogT += Math.log(t)
  let A1 = 0 // Σ T^β log T
  let A2 = 0 // Σ T^β (log T)²
  for (const T of ends) {
    const tb = T ** shape
    const lt = Math.log(T)
    A1 += tb * lt
    A2 += tb * lt * lt
  }
  const hbb = shape * sumLogT - scale * (shape * A1 + shape * shape * A2)
  const hbl = -scale * shape * A1
  const hll = -scale * sumTb
  const det = hbb * hll - hbl * hbl
  const fit = { covariance: { rows: 2, cols: 2, data: Float64Array.from([-hll / det, hbl / det, hbl / det, -hbb / det]) } }
  const seLogShape = Math.sqrt(Math.max(0, fit.covariance.data[0]!))
  const seLogScale = Math.sqrt(Math.max(0, fit.covariance.data[3]!))
  // Laplace trend test (time truncated, pooled): U = (Σ t − n T̄/2) / (T̄ √(n/12)) with pooled scaling per system
  let sumT = 0
  systems.forEach((sys, k) => {
    for (const t of sys) sumT += t / ends[k]!
  })
  const laplaceZ = (sumT - n / 2) / Math.sqrt(n / 12)
  const mil = 2 * sumLog // χ²(2n) under HPP
  const ttt: NhppResult['ttt'] = []
  if (systems.length === 1) {
    const s = systems[0]!.slice().sort((a, b) => a - b)
    const T = ends[0]!
    s.forEach((t, i) => ttt.push({ u: (i + 1) / n, scaled: t / T }))
  }
  return {
    test: 'power-law NHPP (Crow–AMSAA)',
    shape,
    scale,
    se: { shape: seShape, scale: scale * seLogScale },
    ci: { shape: [shape * Math.exp(-z * seLogShape), shape * Math.exp(z * seLogShape)], scale: [scale * Math.exp(-z * seLogScale), scale * Math.exp(z * seLogScale)] },
    n,
    systems: systems.length,
    totalTime,
    mtbf: 1 / (scale * shape * (totalTime / systems.length) ** (shape - 1)),
    trend: {
      laplace: { statistic: laplaceZ, pValue: Math.min(1, 2 * STD.sf(Math.abs(laplaceZ))) },
      milHdbk: { statistic: mil, df: 2 * n, pValue: Math.min(1, 2 * Math.min(chi2Dist(2 * n).cdf(mil), chi2Dist(2 * n).sf(mil))) },
    },
    ttt,
    cumulative: (t: number) => scale * t ** shape,
  }
}

export interface ProbitResult {
  test: 'probit analysis'
  distribution: 'normal' | 'logistic'
  coefficients: LifeCoefficient[]
  naturalResponse: number
  /** Percentiles of the tolerance distribution (ED/LD p) with Fieller confidence limits. */
  percentiles: Array<{ p: number; stress: number; se: number; ci: [number, number] }>
  /** Mean and sd (normal) / location and scale (logistic) of the tolerance distribution. */
  location: number
  scale: number
  deviance: number
  pearsonChi2: number
  df: number
  pValue: number
  n: number
  logLik: number
}

/**
 * Probit Analysis (Minitab): events / trials at stress levels with a normal (probit) or logistic
 * (logit) tolerance distribution and an optional natural response rate (adjusted by Abbott's formula);
 * reports percentiles (e.g. ED50) with Fieller-type intervals.
 */
export function probitAnalysis(
  events: Column,
  trials: Column,
  stress: Column,
  options: { distribution?: 'normal' | 'logistic'; naturalResponse?: number; percentiles?: number[]; confidence?: number; logStress?: boolean } = {},
): ProbitResult {
  const distribution = options.distribution ?? 'normal'
  const confidence = options.confidence ?? 0.95
  const nr = options.naturalResponse ?? 0
  if (!(nr >= 0 && nr < 1)) throw new RangeError('probitAnalysis: naturalResponse in [0, 1)')
  const x = Array.from({ length: stress.length }, (_, i) => (typeof stress[i] === 'number' ? (options.logStress ? Math.log10(stress[i] as number) : (stress[i] as number)) : null))
  // Abbott-adjusted events: p' = (p − nr) / (1 − nr)
  const ev = Array.from({ length: events.length }, (_, i) => {
    const e = events[i]
    const t = trials[i]
    if (typeof e !== 'number' || typeof t !== 'number') return null
    return Math.max(0, Math.min(t, Math.round(((e / t - nr) / (1 - nr)) * t)))
  })
  const fit = glm(ev, { stress: x }, { family: 'binomial', link: distribution === 'normal' ? 'probit' : 'logit', trials, confidence })
  const b0 = fit.coefficients[0]!.coef
  const b1 = fit.coefficients[1]!.coef
  const v = fit.covariance
  const base = distribution === 'normal' ? STD : logisticDist()
  const zc = STD.ppf(0.5 + confidence / 2)
  const percentiles = (options.percentiles ?? [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]).map((p) => {
    const zp = base.ppf(p)
    const xp = (zp - b0) / b1
    // Fieller: g = z² v11 / b1²
    const v00 = v.data[0]!
    const v01 = v.data[1]!
    const v11 = v.data[3]!
    const g = (zc * zc * v11) / (b1 * b1)
    const varX = (v00 + 2 * xp * v01 + xp * xp * v11) / (b1 * b1)
    const se = Math.sqrt(Math.max(0, varX))
    let ci: [number, number]
    if (g < 1) {
      const center = xp + (g / (1 - g)) * (xp + v01 / v11)
      const halfw = (zc / (Math.abs(b1) * (1 - g))) * Math.sqrt(Math.max(0, v00 + 2 * xp * v01 + xp * xp * v11 - g * (v00 - (v01 * v01) / v11)))
      ci = [center - halfw, center + halfw]
    } else ci = [-Infinity, Infinity]
    const back = (u: number) => (options.logStress ? 10 ** u : u)
    return { p, stress: back(xp), se: options.logStress ? NaN : se, ci: [back(ci[0]), back(ci[1])] as [number, number] }
  })
  return {
    test: 'probit analysis',
    distribution,
    coefficients: fit.coefficients.map((c) => ({ name: c.name, coef: c.coef, se: c.se, z: c.z, pValue: c.pValue, ci: c.ci })),
    naturalResponse: nr,
    percentiles,
    location: -b0 / b1,
    scale: 1 / b1,
    deviance: fit.deviance,
    pearsonChi2: fit.pearsonChi2,
    df: fit.dfResidual,
    pValue: fit.goodnessOfFit.pearson.pValue,
    n: fit.n,
    logLik: fit.logLik,
  }
}

