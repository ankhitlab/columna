/**
 * Generalized linear models by IRLS (Minitab Binary Logistic / Poisson Regression, plus gaussian and
 * gamma families), ordinal logistic regression (proportional odds, cumulative logit) and nominal
 * (multinomial) logistic regression by Newton–Raphson. Output follows Minitab: coefficient table with
 * z / p / CI and odds (rate) ratios, deviance and Pearson goodness-of-fit, the G test of all slopes,
 * Hosmer–Lemeshow for binary logistic, log-likelihood and information criteria.
 */
import { chi2 as chi2Dist, lgamma, normal, t as tDist } from './dist.js'
import { inverse, lstsq, matrix, matvec, type Matrix } from './linalg.js'
import { completeRows, designMatrix, toColumns, type Column, type Predictors } from './regression.js'

const STD = normal()

export type Family = 'binomial' | 'poisson' | 'gaussian' | 'gamma'
export type Link = 'logit' | 'probit' | 'cloglog' | 'log' | 'identity' | 'inverse'

export interface GlmOptions {
  family: Family
  link?: Link
  intercept?: boolean
  names?: string[]
  confidence?: number
  /** Binomial: number of trials per row; the response is then the number of events (Minitab's event/trial form), or the observed proportion when it is non-integer. */
  trials?: Column
  /** Offset added to the linear predictor (e.g. log exposure for Poisson rates). */
  offset?: Column
  weights?: Column
  maxIter?: number
  tol?: number
}

export interface GlmCoefficient {
  name: string
  coef: number
  se: number
  z: number
  pValue: number
  ci: [number, number]
  /** exp(coef) with CI — odds ratio (logit) or rate ratio (log link); absent for other links. */
  ratio?: number
  ratioCI?: [number, number]
  vif?: number
  aliased?: boolean
}

export interface GlmResult {
  test: 'glm'
  family: Family
  link: Link
  n: number
  p: number
  coefficients: GlmCoefficient[]
  names: string[]
  deviance: number
  nullDeviance: number
  dfResidual: number
  dfNull: number
  pearsonChi2: number
  /** Estimated dispersion (1 for binomial / poisson). */
  dispersion: number
  logLik: number
  aic: number
  aicc: number
  bic: number
  /** Deviance-based test that all slopes are zero (Minitab "G"). */
  gTest: { statistic: number; df: number; pValue: number }
  /** Goodness-of-fit tests (deviance, Pearson; Hosmer–Lemeshow for binary logistic). */
  goodnessOfFit: { deviance: { statistic: number; df: number; pValue: number }; pearson: { statistic: number; df: number; pValue: number }; hosmerLemeshow?: { statistic: number; df: number; pValue: number; groups: Array<{ observed: number; expected: number; n: number }> } }
  /** Deviance R² and adjusted (Minitab: 1 − dev/nullDev). */
  r2deviance: number
  r2devianceAdj: number
  fitted: Float64Array
  linearPredictor: Float64Array
  residuals: { deviance: Float64Array; pearson: Float64Array; standardizedPearson: Float64Array; response: Float64Array }
  leverage: Float64Array
  cooksD: Float64Array
  iterations: number
  converged: boolean
  omitted: number[]
  covariance: Matrix
  confidence: number
  predict(x: ArrayLike<number> | ArrayLike<number>[], options?: { type?: 'response' | 'link'; confidence?: number }): Array<{ fit: number; se: number; ci: [number, number] }>
}

function defaultLink(f: Family): Link {
  return f === 'binomial' ? 'logit' : f === 'poisson' ? 'log' : f === 'gamma' ? 'log' : 'identity'
}

const links: Record<Link, { g: (mu: number) => number; inv: (eta: number) => number; dmu: (eta: number) => number }> = {
  logit: { g: (m) => Math.log(m / (1 - m)), inv: (e) => 1 / (1 + Math.exp(-e)), dmu: (e) => { const m = 1 / (1 + Math.exp(-e)); return m * (1 - m) } },
  probit: { g: (m) => STD.ppf(m), inv: (e) => STD.cdf(e), dmu: (e) => STD.pdf(e) },
  cloglog: { g: (m) => Math.log(-Math.log(1 - m)), inv: (e) => 1 - Math.exp(-Math.exp(e)), dmu: (e) => Math.exp(e - Math.exp(e)) },
  log: { g: Math.log, inv: Math.exp, dmu: Math.exp },
  identity: { g: (m) => m, inv: (e) => e, dmu: () => 1 },
  inverse: { g: (m) => 1 / m, inv: (e) => 1 / e, dmu: (e) => -1 / (e * e) },
}

function variance(f: Family, mu: number): number {
  return f === 'binomial' ? mu * (1 - mu) : f === 'poisson' ? mu : f === 'gamma' ? mu * mu : 1
}

/** Unit deviance d(y, μ) (already multiplied by the binomial trials via prior weights). */
function unitDeviance(f: Family, y: number, mu: number): number {
  switch (f) {
    case 'binomial': {
      const a = y > 0 ? y * Math.log(y / mu) : 0
      const b = y < 1 ? (1 - y) * Math.log((1 - y) / (1 - mu)) : 0
      return 2 * (a + b)
    }
    case 'poisson':
      return 2 * ((y > 0 ? y * Math.log(y / mu) : 0) - (y - mu))
    case 'gamma':
      return 2 * (-Math.log(y / mu) + (y - mu) / mu)
    default:
      return (y - mu) ** 2
  }
}

function clampMu(f: Family, link: Link, mu: number): number {
  if (f === 'binomial') return Math.min(1 - 1e-10, Math.max(1e-10, mu))
  if (f === 'poisson' || f === 'gamma' || link === 'log') return Math.max(1e-10, mu)
  return mu
}

interface Irls {
  coef: Float64Array
  eta: Float64Array
  mu: Float64Array
  xtwxInv: Matrix
  leverage: Float64Array
  deviance: number
  iterations: number
  converged: boolean
  rank: number
  dependent: number[]
}

function irls(X: Matrix, y: Float64Array, w: Float64Array, offset: Float64Array, family: Family, link: Link, maxIter: number, tol: number): Irls {
  const n = X.rows
  const p = X.cols
  const L = links[link]
  // starting values from the response
  const mu = new Float64Array(n)
  const eta = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let m = y[i]!
    if (family === 'binomial') m = (w[i]! * y[i]! + 0.5) / (w[i]! + 1)
    else if (family === 'poisson' || family === 'gamma') m = Math.max(y[i]!, 0.1)
    mu[i] = m
    eta[i] = L.g(m)
  }
  let coef = new Float64Array(p)
  let prevCoef: Float64Array | undefined
  let dev = Infinity
  let iterations = 0
  let converged = false
  let last: ReturnType<typeof lstsq> | undefined
  const Xw = matrix(n, p)
  const z = new Float64Array(n)
  const sw = new Float64Array(n)
  for (let it = 1; it <= maxIter; it++) {
    iterations = it
    for (let i = 0; i < n; i++) {
      const d = L.dmu(eta[i]!)
      const v = variance(family, mu[i]!)
      const wi = (w[i]! * d * d) / v
      sw[i] = Math.sqrt(Math.max(wi, 0))
      z[i] = (eta[i]! - offset[i]! + (y[i]! - mu[i]!) / d) * sw[i]!
      for (let j = 0; j < p; j++) Xw.data[i * p + j] = X.data[i * p + j]! * sw[i]!
    }
    last = lstsq(Xw, z)
    coef = Float64Array.from(last.coef)
    const evalDev = (): number => {
      let d = 0
      const etaNew = matvec(X, coef)
      for (let i = 0; i < n; i++) {
        eta[i] = etaNew[i]! + offset[i]!
        mu[i] = clampMu(family, link, L.inv(eta[i]!))
        d += w[i]! * unitDeviance(family, y[i]!, mu[i]!)
      }
      return d
    }
    let newDev = evalDev()
    // step-halving toward the previous iterate if the deviance increased
    for (let half = 0; prevCoef && newDev > dev + 1e-12 && half < 10; half++) {
      for (let j = 0; j < p; j++) coef[j] = 0.5 * (coef[j]! + prevCoef[j]!)
      newDev = evalDev()
    }
    prevCoef = coef
    if (Math.abs(newDev - dev) < tol * (Math.abs(newDev) + 0.1)) {
      dev = newDev
      converged = true
      break
    }
    dev = newDev
  }
  const lev = last!.leverage
  return { coef, eta, mu, xtwxInv: last!.xtxInv, leverage: lev, deviance: dev, iterations, converged, rank: last!.rank, dependent: last!.dependent }
}

/**
 * Fit a GLM.
 *   glm(y01, { x1, x2 }, { family: 'binomial' })                       // binary logistic
 *   glm(events, X, { family: 'binomial', trials })                     // events / trials
 *   glm(counts, X, { family: 'poisson', offset: exposure.map(Math.log) })
 */
export function glm(y: Column, X: Predictors, options: GlmOptions): GlmResult {
  const family = options.family
  if (!['binomial', 'poisson', 'gaussian', 'gamma'].includes(family)) throw new RangeError(`glm: unknown family "${String(family)}"`)
  const link = options.link ?? defaultLink(family)
  if (!(link in links)) throw new RangeError(`glm: unknown link "${String(link)}"`)
  if (family === 'binomial' && !['logit', 'probit', 'cloglog'].includes(link)) throw new RangeError(`glm: binomial needs logit / probit / cloglog, got ${link}`)
  const intercept = options.intercept ?? true
  const confidence = options.confidence ?? 0.95
  const { names: xnames, cols } = toColumns(X, options.names)
  const extra: Column[] = []
  if (options.trials) extra.push(options.trials)
  if (options.offset) extra.push(options.offset)
  const { keep, omitted } = completeRows(y, [...cols, ...extra], options.weights)
  const n = keep.length
  const design = designMatrix(cols, keep, intercept)
  const pAll = design.cols
  if (n <= pAll) throw new RangeError(`glm needs more observations than coefficients (n = ${n}, p = ${pAll})`)
  const yv = new Float64Array(n)
  const w = new Float64Array(n)
  const offset = new Float64Array(n)
  // with trials: events unless some response is a non-integer in [0, 1] (then proportions)
  const proportions = !!options.trials && keep.some((r) => !Number.isInteger(y[r] as number)) && keep.every((r) => (y[r] as number) >= 0 && (y[r] as number) <= 1)
  for (let i = 0; i < n; i++) {
    const r = keep[i]!
    let yi = y[r] as number
    let wi = options.weights ? (options.weights[r] as number) : 1
    if (family === 'binomial') {
      if (options.trials) {
        const m = options.trials[r] as number
        if (!(m > 0) || !Number.isInteger(m)) throw new RangeError(`glm: trials must be positive integers (row ${r})`)
        if (!proportions) {
          if (!Number.isInteger(yi) || yi > m) throw new RangeError(`glm: events must be integers ≤ trials (row ${r})`)
          yi /= m
        }
        wi *= m
      }
      if (!(yi >= 0 && yi <= 1)) throw new RangeError(`glm: binomial response must be in [0, 1] or events ≤ trials (row ${r})`)
    } else if ((family === 'poisson' && yi < 0) || (family === 'gamma' && yi <= 0)) throw new RangeError(`glm: invalid response for ${family} (row ${r})`)
    yv[i] = yi
    w[i] = wi
    offset[i] = options.offset ? (options.offset[r] as number) : 0
  }
  const maxIter = options.maxIter ?? 50
  const tol = options.tol ?? 1e-10
  const fit = irls(design, yv, w, offset, family, link, maxIter, tol)
  const p = fit.rank
  const dfResidual = n - p
  // null model (constant only, same offset)
  let nullDeviance: number
  if (intercept) {
    const one = matrix(n, 1, new Float64Array(n).fill(1))
    nullDeviance = irls(one, yv, w, offset, family, link, maxIter, tol).deviance
  } else {
    let d = 0
    for (let i = 0; i < n; i++) d += w[i]! * unitDeviance(family, yv[i]!, clampMu(family, link, links[link].inv(offset[i]!)))
    nullDeviance = d
  }
  const dfNull = n - (intercept ? 1 : 0)
  // Pearson χ², residuals
  const devRes = new Float64Array(n)
  const pearson = new Float64Array(n)
  const stdPearson = new Float64Array(n)
  const response = new Float64Array(n)
  let pearsonChi2 = 0
  for (let i = 0; i < n; i++) {
    const mu = fit.mu[i]!
    const r = yv[i]! - mu
    response[i] = r
    const pr = (r * Math.sqrt(w[i]!)) / Math.sqrt(variance(family, mu))
    pearson[i] = pr
    pearsonChi2 += pr * pr
    devRes[i] = Math.sign(r) * Math.sqrt(Math.max(0, w[i]! * unitDeviance(family, yv[i]!, mu)))
  }
  const estimateDispersion = family === 'gaussian' || family === 'gamma'
  const dispersion = estimateDispersion ? pearsonChi2 / dfResidual : 1
  const cooks = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const h = fit.leverage[i]!
    stdPearson[i] = pearson[i]! / Math.sqrt(dispersion * (1 - h))
    cooks[i] = (stdPearson[i]! ** 2 * h) / (p * (1 - h))
  }
  // log-likelihood
  let logLik = 0
  for (let i = 0; i < n; i++) {
    const mu = fit.mu[i]!
    const yi = yv[i]!
    if (family === 'binomial') {
      const m = w[i]!
      const k = yi * m
      logLik += lgamma(m + 1) - lgamma(k + 1) - lgamma(m - k + 1) + k * Math.log(mu) + (m - k) * Math.log(1 - mu)
    } else if (family === 'poisson') logLik += w[i]! * (yi * Math.log(mu) - mu - lgamma(yi + 1))
    else if (family === 'gaussian') logLik += -0.5 * (Math.log(2 * Math.PI * (fit.deviance / n)) + (w[i]! * (yi - mu) ** 2) / (fit.deviance / n))
    else {
      const a = 1 / dispersion
      logLik += w[i]! * (a * Math.log((a * yi) / mu) - (a * yi) / mu - Math.log(yi) - lgamma(a))
    }
  }
  const k = p + (estimateDispersion ? 1 : 0)
  const aic = -2 * logLik + 2 * k
  const aicc = n - k - 1 > 0 ? aic + (2 * k * (k + 1)) / (n - k - 1) : Infinity
  const bic = -2 * logLik + k * Math.log(n)
  // coefficient table
  const covariance = matrix(pAll, pAll, Float64Array.from(fit.xtwxInv.data, (v) => v * dispersion))
  const useT = estimateDispersion
  const crit = useT ? tDist(dfResidual).ppf(0.5 + confidence / 2) : STD.ppf(0.5 + confidence / 2)
  const pOf = (z: number) => (useT ? Math.min(1, 2 * tDist(dfResidual).sf(Math.abs(z))) : Math.min(1, 2 * STD.sf(Math.abs(z))))
  const aliased = new Set(fit.dependent)
  const names = intercept ? ['Constant', ...xnames] : [...xnames]
  const ratioLink = link === 'logit' || link === 'log'
  const coefficients: GlmCoefficient[] = names.map((name, j) => {
    if (aliased.has(j)) return { name, coef: 0, se: NaN, z: NaN, pValue: NaN, ci: [NaN, NaN], aliased: true }
    const se = Math.sqrt(covariance.data[j * pAll + j]!)
    const b = fit.coef[j]!
    const z = b / se
    const ci: [number, number] = [b - crit * se, b + crit * se]
    const row: GlmCoefficient = { name, coef: b, se, z, pValue: pOf(z), ci }
    if (ratioLink && !(intercept && j === 0)) {
      row.ratio = Math.exp(b)
      row.ratioCI = [Math.exp(ci[0]), Math.exp(ci[1])]
    }
    return row
  })
  // VIF on the (unweighted) design, as Minitab reports for logistic
  if (intercept && cols.length > 1) {
    for (let j = 1; j < pAll; j++) {
      if (aliased.has(j)) continue
      const others = matrix(n, pAll - 1)
      const xj = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        let c2 = 0
        for (let c = 0; c < pAll; c++) if (c !== j) others.data[i * (pAll - 1) + c2++] = design.data[i * pAll + c]!
        xj[i] = design.data[i * pAll + j]!
      }
      const f2 = lstsq(others, xj)
      let mean = 0
      for (let i = 0; i < n; i++) mean += xj[i]!
      mean /= n
      let tot = 0
      for (let i = 0; i < n; i++) tot += (xj[i]! - mean) ** 2
      coefficients[j]!.vif = tot > 0 ? tot / f2.sse : Infinity
    }
  }
  const gStat = nullDeviance - fit.deviance
  const gDf = dfNull - dfResidual
  const gTest = { statistic: gStat, df: gDf, pValue: gDf > 0 ? chi2Dist(gDf).sf(gStat / dispersion) : NaN }
  const goodnessOfFit: GlmResult['goodnessOfFit'] = {
    deviance: { statistic: fit.deviance, df: dfResidual, pValue: chi2Dist(dfResidual).sf(fit.deviance) },
    pearson: { statistic: pearsonChi2, df: dfResidual, pValue: chi2Dist(dfResidual).sf(pearsonChi2) },
  }
  if (family === 'binomial') {
    // Hosmer–Lemeshow: 10 groups of (roughly) equal size by fitted probability
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => fit.mu[a]! - fit.mu[b]!)
    const G = Math.min(10, n)
    const groups: Array<{ observed: number; expected: number; n: number }> = []
    let stat = 0
    for (let g = 0; g < G; g++) {
      const lo = Math.floor((g * n) / G)
      const hi = Math.floor(((g + 1) * n) / G)
      let obs = 0
      let exp = 0
      let m = 0
      for (let q = lo; q < hi; q++) {
        const i = order[q]!
        obs += yv[i]! * w[i]!
        exp += fit.mu[i]! * w[i]!
        m += w[i]!
      }
      if (m === 0) continue
      groups.push({ observed: obs, expected: exp, n: m })
      const denom = exp * (1 - exp / m)
      if (denom > 0) stat += (obs - exp) ** 2 / denom
    }
    const df = Math.max(1, groups.length - 2)
    goodnessOfFit.hosmerLemeshow = { statistic: stat, df, pValue: chi2Dist(df).sf(stat), groups }
  }
  const r2deviance = 1 - fit.deviance / nullDeviance
  const r2adj = 1 - ((fit.deviance / dfResidual) * dfNull) / nullDeviance
  const predict: GlmResult['predict'] = (x, o = {}) => {
    const type = o.type ?? 'response'
    const conf = o.confidence ?? confidence
    const c = useT ? tDist(dfResidual).ppf(0.5 + conf / 2) : STD.ppf(0.5 + conf / 2)
    const rows: ArrayLike<number>[] = typeof (x as ArrayLike<number>)[0] === 'number' || x.length === 0 ? [x as ArrayLike<number>] : (x as ArrayLike<number>[])
    return rows.map((row) => {
      if (row.length !== xnames.length) throw new RangeError(`predict: expected ${xnames.length} predictor values, got ${row.length}`)
      const v = new Float64Array(pAll)
      let j = 0
      if (intercept) v[j++] = 1
      for (let c2 = 0; c2 < row.length; c2++) v[j++] = row[c2]!
      let eta = 0
      for (let c2 = 0; c2 < pAll; c2++) eta += v[c2]! * fit.coef[c2]!
      let q = 0
      for (let a = 0; a < pAll; a++) for (let b = 0; b < pAll; b++) q += v[a]! * covariance.data[a * pAll + b]! * v[b]!
      const se = Math.sqrt(Math.max(0, q))
      if (type === 'link') return { fit: eta, se, ci: [eta - c * se, eta + c * se] }
      const inv = links[link].inv
      const d = links[link].dmu(eta)
      return { fit: inv(eta), se: Math.abs(d) * se, ci: [inv(eta - c * se), inv(eta + c * se)] }
    })
  }
  return {
    test: 'glm',
    family,
    link,
    n,
    p,
    coefficients,
    names: xnames,
    deviance: fit.deviance,
    nullDeviance,
    dfResidual,
    dfNull,
    pearsonChi2,
    dispersion,
    logLik,
    aic,
    aicc,
    bic,
    gTest,
    goodnessOfFit,
    r2deviance,
    r2devianceAdj: r2adj,
    fitted: fit.mu,
    linearPredictor: fit.eta,
    residuals: { deviance: devRes, pearson, standardizedPearson: stdPearson, response },
    leverage: fit.leverage,
    cooksD: cooks,
    iterations: fit.iterations,
    converged: fit.converged,
    omitted,
    covariance,
    confidence,
    predict,
  }
}

/** Binary Logistic Regression (Minitab): glm with the binomial family; `link` logit (default), probit or cloglog. */
export function logit(y: Column, X: Predictors, options: Omit<GlmOptions, 'family'> = {}): GlmResult {
  return glm(y, X, { ...options, family: 'binomial' })
}

/** Poisson Regression (Minitab): glm with the poisson family and log link; pass `offset` = log(exposure) for rates. */
export function poissonRegression(y: Column, X: Predictors, options: Omit<GlmOptions, 'family'> = {}): GlmResult {
  return glm(y, X, { ...options, family: 'poisson' })
}

// ---- Newton–Raphson for the ordinal / nominal models -------------------------------------------------------

interface NewtonResult {
  theta: Float64Array
  logLik: number
  hessian: Matrix
  iterations: number
  converged: boolean
}

/** Maximize ℓ(θ) with analytic gradient; Hessian by Richardson-extrapolated central differences of the gradient. */
function newton(loglik: (t: Float64Array) => number, grad: (t: Float64Array) => Float64Array, theta0: Float64Array, hessian?: (t: Float64Array) => Matrix, maxIter = 100, tol = 1e-10): NewtonResult {
  const k = theta0.length
  const numHessian = (t: Float64Array): Matrix => {
    const H = matrix(k, k)
    for (let j = 0; j < k; j++) {
      const h = 1e-4 * Math.max(1, Math.abs(t[j]!))
      const g = (step: number) => {
        const tt = Float64Array.from(t)
        tt[j] = tt[j]! + step
        return grad(tt)
      }
      const g1 = g(h)
      const g2 = g(-h)
      const g3 = g(2 * h)
      const g4 = g(-2 * h)
      for (let i = 0; i < k; i++) {
        const d1 = (g1[i]! - g2[i]!) / (2 * h)
        const d2 = (g3[i]! - g4[i]!) / (4 * h)
        H.data[i * k + j] = (4 * d1 - d2) / 3
      }
    }
    for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) {
      const m = 0.5 * (H.data[i * k + j]! + H.data[j * k + i]!)
      H.data[i * k + j] = m
      H.data[j * k + i] = m
    }
    return H
  }
  const hess = hessian ?? numHessian
  let theta = Float64Array.from(theta0)
  let ll = loglik(theta)
  let converged = false
  let it = 0
  for (it = 1; it <= maxIter; it++) {
    const g = grad(theta)
    const H = hess(theta)
    let step: Float64Array
    try {
      const Hinv = inverse(H)
      step = matvec(Hinv, g)
      for (let j = 0; j < k; j++) step[j] = -step[j]!
    } catch {
      step = Float64Array.from(g, (v) => 0.01 * v)
    }
    // ensure ascent direction
    let dot = 0
    for (let j = 0; j < k; j++) dot += step[j]! * g[j]!
    if (dot < 0) for (let j = 0; j < k; j++) step[j] = 0.01 * g[j]!
    let alpha = 1
    let next = new Float64Array(k)
    let llNext = -Infinity
    for (let half = 0; half < 30; half++) {
      for (let j = 0; j < k; j++) next[j] = theta[j]! + alpha * step[j]!
      llNext = loglik(next)
      if (Number.isFinite(llNext) && llNext >= ll - 1e-12) break
      alpha /= 2
    }
    const gmax = Math.max(...Array.from(g, Math.abs))
    theta = next
    const improved = llNext - ll
    ll = llNext
    if (Math.abs(improved) < tol * (Math.abs(ll) + 1) && gmax < 1e-6 * (Math.abs(ll) + 1)) {
      converged = true
      break
    }
    next = new Float64Array(k)
  }
  return { theta, logLik: ll, hessian: hess(theta), iterations: it, converged }
}

export interface OrdinalCoefficient {
  name: string
  coef: number
  se: number
  z: number
  pValue: number
  ci: [number, number]
  oddsRatio?: number
  oddsRatioCI?: [number, number]
}

export interface OrdinalResult {
  test: 'ordinal logistic'
  /** Sorted response levels (ascending). */
  levels: string[]
  n: number
  /** Threshold constants θ₁ … θ_{K−1} (Minitab: Const(1), Const(2), …). */
  thresholds: OrdinalCoefficient[]
  coefficients: OrdinalCoefficient[]
  logLik: number
  nullLogLik: number
  gTest: { statistic: number; df: number; pValue: number }
  goodnessOfFit: { pearson: { statistic: number; df: number; pValue: number }; deviance: { statistic: number; df: number; pValue: number } }
  aic: number
  bic: number
  /** Fitted category probabilities per row (n × K). */
  probabilities: Float64Array[]
  iterations: number
  converged: boolean
  omitted: number[]
  /** Predicted probabilities for new predictor rows. */
  predict(x: ArrayLike<number> | ArrayLike<number>[]): number[][]
}

function levelsOf(y: ArrayLike<unknown>, keep: number[]): { levels: string[]; codes: Int32Array } {
  const numeric = keep.every((i) => typeof y[i] === 'number')
  const set = new Set<string>()
  for (const i of keep) set.add(String(y[i]))
  const levels = [...set].sort(numeric ? (a, b) => Number(a) - Number(b) : undefined)
  const idx = new Map(levels.map((l, i) => [l, i]))
  return { levels, codes: Int32Array.from(keep, (i) => idx.get(String(y[i]))!) }
}

/**
 * Ordinal Logistic Regression (Minitab, proportional odds): logit P(Y ≤ k) = θ_k + x'β for the
 * sorted levels k = 1 … K − 1 (Minitab's sign convention: a positive coefficient shifts probability
 * toward the lower categories).
 */
export function ologit(y: ArrayLike<number | string | null | undefined>, X: Predictors, options: { names?: string[]; confidence?: number; maxIter?: number } = {}): OrdinalResult {
  const confidence = options.confidence ?? 0.95
  const { names: xnames, cols } = toColumns(X, options.names)
  const n0 = y.length
  const keep: number[] = []
  const omitted: number[] = []
  for (let i = 0; i < n0; i++) {
    const ok = y[i] !== null && y[i] !== undefined && cols.every((c) => typeof c[i] === 'number' && Number.isFinite(c[i] as number))
    ;(ok ? keep : omitted).push(i)
  }
  const n = keep.length
  const { levels, codes } = levelsOf(y, keep)
  const K = levels.length
  if (K < 3) throw new RangeError(`ologit needs at least 3 response levels, got ${K}`)
  const D = designMatrix(cols, keep, false)
  const p = D.cols
  const nTheta = K - 1 + p
  // parameterization: θ_1, then positive increments (log) — keep simple: θ_k directly, with ordering enforced by the likelihood
  const cum = (t: Float64Array, i: number, k: number) => {
    if (k < 0) return 0
    if (k >= K - 1) return 1
    let e = t[k]!
    for (let j = 0; j < p; j++) e += t[K - 1 + j]! * D.data[i * p + j]!
    return 1 / (1 + Math.exp(-e))
  }
  const loglik = (t: Float64Array) => {
    for (let k = 1; k < K - 1; k++) if (!(t[k]! > t[k - 1]!)) return -Infinity
    let s = 0
    for (let i = 0; i < n; i++) {
      const c = codes[i]!
      const pr = cum(t, i, c) - cum(t, i, c - 1)
      if (!(pr > 0)) return -Infinity
      s += Math.log(pr)
    }
    return s
  }
  const grad = (t: Float64Array) => {
    const g = new Float64Array(nTheta)
    for (let i = 0; i < n; i++) {
      const c = codes[i]!
      const gk = cum(t, i, c)
      const gk1 = cum(t, i, c - 1)
      const pr = gk - gk1
      const dk = c < K - 1 ? gk * (1 - gk) : 0
      const dk1 = c > 0 ? gk1 * (1 - gk1) : 0
      if (c < K - 1) g[c] = g[c]! + dk / pr
      if (c > 0) g[c - 1] = g[c - 1]! - dk1 / pr
      const dbeta = (dk - dk1) / pr
      for (let j = 0; j < p; j++) g[K - 1 + j] = g[K - 1 + j]! + dbeta * D.data[i * p + j]!
    }
    return g
  }
  // start: thresholds from marginal cumulative proportions, β = 0
  const theta0 = new Float64Array(nTheta)
  const counts = new Float64Array(K)
  for (let i = 0; i < n; i++) counts[codes[i]!] = counts[codes[i]!]! + 1
  let acc = 0
  for (let k = 0; k < K - 1; k++) {
    acc += counts[k]!
    const q = Math.min(1 - 1e-6, Math.max(1e-6, acc / n))
    theta0[k] = Math.log(q / (1 - q))
  }
  const fit = newton(loglik, grad, theta0, undefined, options.maxIter ?? 100)
  const cov = inverse(matrix(nTheta, nTheta, Float64Array.from(fit.hessian.data, (v) => -v)))
  const crit = STD.ppf(0.5 + confidence / 2)
  const row = (name: string, j: number, or: boolean): OrdinalCoefficient => {
    const b = fit.theta[j]!
    const se = Math.sqrt(Math.max(0, cov.data[j * nTheta + j]!))
    const z = b / se
    const r: OrdinalCoefficient = { name, coef: b, se, z, pValue: Math.min(1, 2 * STD.sf(Math.abs(z))), ci: [b - crit * se, b + crit * se] }
    if (or) {
      r.oddsRatio = Math.exp(b)
      r.oddsRatioCI = [Math.exp(r.ci[0]), Math.exp(r.ci[1])]
    }
    return r
  }
  const thresholds = Array.from({ length: K - 1 }, (_, k) => row(`Const(${k + 1})`, k, false))
  const coefficients = xnames.map((nm, j) => row(nm, K - 1 + j, true))
  // null model: thresholds only
  const nullFit = newton(
    (t) => loglik(Float64Array.from([...t, ...new Array(p).fill(0)])),
    (t) => grad(Float64Array.from([...t, ...new Array(p).fill(0)])).subarray(0, K - 1),
    theta0.subarray(0, K - 1),
  )
  const g = 2 * (fit.logLik - nullFit.logLik)
  const probabilities = Array.from({ length: n }, (_, i) => {
    const pr = new Float64Array(K)
    for (let k = 0; k < K; k++) pr[k] = cum(fit.theta, i, k) - cum(fit.theta, i, k - 1)
    return pr
  })
  // Pearson and deviance goodness-of-fit on the distinct covariate patterns
  const patterns = new Map<string, { obs: Float64Array; exp: Float64Array; m: number }>()
  for (let i = 0; i < n; i++) {
    const key = Array.from({ length: p }, (_, j) => D.data[i * p + j]).join('|')
    let pt = patterns.get(key)
    if (!pt) patterns.set(key, (pt = { obs: new Float64Array(K), exp: new Float64Array(K), m: 0 }))
    pt.obs[codes[i]!] = pt.obs[codes[i]!]! + 1
    for (let k = 0; k < K; k++) pt.exp[k] = pt.exp[k]! + probabilities[i]![k]!
    pt.m++
  }
  let pearsonStat = 0
  let devStat = 0
  for (const pt of patterns.values()) {
    for (let k = 0; k < K; k++) {
      const e = pt.exp[k]!
      const o = pt.obs[k]!
      if (e > 0) pearsonStat += (o - e) ** 2 / e
      if (o > 0) devStat += 2 * o * Math.log(o / e)
    }
  }
  const gofDf = patterns.size * (K - 1) - nTheta
  const goodnessOfFit = {
    pearson: { statistic: pearsonStat, df: gofDf, pValue: gofDf > 0 ? chi2Dist(gofDf).sf(pearsonStat) : NaN },
    deviance: { statistic: devStat, df: gofDf, pValue: gofDf > 0 ? chi2Dist(gofDf).sf(devStat) : NaN },
  }
  const predict = (x: ArrayLike<number> | ArrayLike<number>[]): number[][] => {
    const rows: ArrayLike<number>[] = typeof (x as ArrayLike<number>)[0] === 'number' || x.length === 0 ? [x as ArrayLike<number>] : (x as ArrayLike<number>[])
    return rows.map((r) => {
      if (r.length !== p) throw new RangeError(`predict: expected ${p} predictor values, got ${r.length}`)
      let xb = 0
      for (let j = 0; j < p; j++) xb += fit.theta[K - 1 + j]! * r[j]!
      const cumk = (k: number) => (k < 0 ? 0 : k >= K - 1 ? 1 : 1 / (1 + Math.exp(-(fit.theta[k]! + xb))))
      return Array.from({ length: K }, (_, k) => cumk(k) - cumk(k - 1))
    })
  }
  return {
    test: 'ordinal logistic',
    levels,
    n,
    thresholds,
    coefficients,
    logLik: fit.logLik,
    nullLogLik: nullFit.logLik,
    gTest: { statistic: g, df: p, pValue: chi2Dist(p).sf(g) },
    goodnessOfFit,
    aic: -2 * fit.logLik + 2 * nTheta,
    bic: -2 * fit.logLik + nTheta * Math.log(n),
    probabilities,
    iterations: fit.iterations,
    converged: fit.converged,
    omitted,
    predict,
  }
}

export interface NominalResult {
  test: 'nominal logistic'
  levels: string[]
  reference: string
  n: number
  /** One coefficient table per non-reference level (logit of level vs reference). */
  equations: Array<{ level: string; coefficients: OrdinalCoefficient[] }>
  logLik: number
  nullLogLik: number
  gTest: { statistic: number; df: number; pValue: number }
  goodnessOfFit: { pearson: { statistic: number; df: number; pValue: number }; deviance: { statistic: number; df: number; pValue: number } }
  aic: number
  bic: number
  probabilities: Float64Array[]
  iterations: number
  converged: boolean
  omitted: number[]
  predict(x: ArrayLike<number> | ArrayLike<number>[]): number[][]
}

/**
 * Nominal (multinomial) Logistic Regression (Minitab): for each non-reference level j,
 * log[P(Y = j)/P(Y = ref)] = α_j + x'β_j. Reference defaults to the first sorted level.
 */
export function mlogit(y: ArrayLike<number | string | null | undefined>, X: Predictors, options: { names?: string[]; reference?: string | number; confidence?: number; maxIter?: number } = {}): NominalResult {
  const confidence = options.confidence ?? 0.95
  const { names: xnames, cols } = toColumns(X, options.names)
  const keep: number[] = []
  const omitted: number[] = []
  for (let i = 0; i < y.length; i++) {
    const ok = y[i] !== null && y[i] !== undefined && cols.every((c) => typeof c[i] === 'number' && Number.isFinite(c[i] as number))
    ;(ok ? keep : omitted).push(i)
  }
  const n = keep.length
  const lv = levelsOf(y, keep)
  const K = lv.levels.length
  if (K < 2) throw new RangeError('mlogit needs at least 2 response levels')
  const refName = options.reference === undefined ? lv.levels[0]! : String(options.reference)
  const refIdx = lv.levels.indexOf(refName)
  if (refIdx < 0) throw new RangeError(`mlogit: reference level "${refName}" not found`)
  const nonRef = lv.levels.map((_, i) => i).filter((i) => i !== refIdx)
  const D = designMatrix(cols, keep, true)
  const p = D.cols
  const nTheta = (K - 1) * p
  const probs = (t: Float64Array, i: number): Float64Array => {
    const out = new Float64Array(K)
    let denom = 1
    for (let q = 0; q < K - 1; q++) {
      let e = 0
      for (let j = 0; j < p; j++) e += t[q * p + j]! * D.data[i * p + j]!
      const ex = Math.exp(e)
      out[nonRef[q]!] = ex
      denom += ex
    }
    out[refIdx] = 1
    for (let k = 0; k < K; k++) out[k] = out[k]! / denom
    return out
  }
  const codes = lv.codes
  const loglik = (t: Float64Array) => {
    let s = 0
    for (let i = 0; i < n; i++) s += Math.log(probs(t, i)[codes[i]!]!)
    return s
  }
  const grad = (t: Float64Array) => {
    const g = new Float64Array(nTheta)
    for (let i = 0; i < n; i++) {
      const pr = probs(t, i)
      for (let q = 0; q < K - 1; q++) {
        const ind = codes[i] === nonRef[q] ? 1 : 0
        const d = ind - pr[nonRef[q]!]!
        for (let j = 0; j < p; j++) g[q * p + j] = g[q * p + j]! + d * D.data[i * p + j]!
      }
    }
    return g
  }
  const hessian = (t: Float64Array) => {
    const H = matrix(nTheta, nTheta)
    for (let i = 0; i < n; i++) {
      const pr = probs(t, i)
      for (let q = 0; q < K - 1; q++) {
        for (let r = 0; r < K - 1; r++) {
          const pq = pr[nonRef[q]!]!
          const prr = pr[nonRef[r]!]!
          const w = -(q === r ? pq * (1 - pq) : -pq * prr)
          for (let a = 0; a < p; a++) {
            const xa = D.data[i * p + a]! * w
            if (xa === 0) continue
            for (let b = 0; b < p; b++) H.data[(q * p + a) * nTheta + r * p + b] += xa * D.data[i * p + b]!
          }
        }
      }
    }
    return H
  }
  const fit = newton(loglik, grad, new Float64Array(nTheta), hessian, options.maxIter ?? 100)
  const cov = inverse(matrix(nTheta, nTheta, Float64Array.from(fit.hessian.data, (v) => -v)))
  const crit = STD.ppf(0.5 + confidence / 2)
  const equations = nonRef.map((lvl, q) => ({
    level: lv.levels[lvl]!,
    coefficients: ['Constant', ...xnames].map((name, j) => {
      const idx = q * p + j
      const b = fit.theta[idx]!
      const se = Math.sqrt(Math.max(0, cov.data[idx * nTheta + idx]!))
      const z = b / se
      const r: OrdinalCoefficient = { name, coef: b, se, z, pValue: Math.min(1, 2 * STD.sf(Math.abs(z))), ci: [b - crit * se, b + crit * se] }
      if (j > 0) {
        r.oddsRatio = Math.exp(b)
        r.oddsRatioCI = [Math.exp(r.ci[0]), Math.exp(r.ci[1])]
      }
      return r
    }),
  }))
  // null: constants only → log-likelihood of the marginal proportions
  const counts = new Float64Array(K)
  for (let i = 0; i < n; i++) counts[codes[i]!] = counts[codes[i]!]! + 1
  let nullLogLik = 0
  for (let k = 0; k < K; k++) if (counts[k]! > 0) nullLogLik += counts[k]! * Math.log(counts[k]! / n)
  const g = 2 * (fit.logLik - nullLogLik)
  const gdf = (K - 1) * (p - 1)
  const probabilities = Array.from({ length: n }, (_, i) => probs(fit.theta, i))
  const patterns = new Map<string, { obs: Float64Array; exp: Float64Array }>()
  for (let i = 0; i < n; i++) {
    const key = Array.from({ length: p }, (_, j) => D.data[i * p + j]).join('|')
    let pt = patterns.get(key)
    if (!pt) patterns.set(key, (pt = { obs: new Float64Array(K), exp: new Float64Array(K) }))
    pt.obs[codes[i]!] = pt.obs[codes[i]!]! + 1
    for (let k = 0; k < K; k++) pt.exp[k] = pt.exp[k]! + probabilities[i]![k]!
  }
  let pearsonStat = 0
  let devStat = 0
  for (const pt of patterns.values()) for (let k = 0; k < K; k++) {
    const e = pt.exp[k]!
    const o = pt.obs[k]!
    if (e > 0) pearsonStat += (o - e) ** 2 / e
    if (o > 0) devStat += 2 * o * Math.log(o / e)
  }
  const gofDf = patterns.size * (K - 1) - nTheta
  const predict = (x: ArrayLike<number> | ArrayLike<number>[]): number[][] => {
    const rows: ArrayLike<number>[] = typeof (x as ArrayLike<number>)[0] === 'number' || x.length === 0 ? [x as ArrayLike<number>] : (x as ArrayLike<number>[])
    return rows.map((r) => {
      if (r.length !== p - 1) throw new RangeError(`predict: expected ${p - 1} predictor values, got ${r.length}`)
      const out = new Array<number>(K).fill(0)
      let denom = 1
      for (let q = 0; q < K - 1; q++) {
        let e = fit.theta[q * p]!
        for (let j = 1; j < p; j++) e += fit.theta[q * p + j]! * r[j - 1]!
        out[nonRef[q]!] = Math.exp(e)
        denom += out[nonRef[q]!]!
      }
      out[refIdx] = 1
      return out.map((v) => v / denom)
    })
  }
  return {
    test: 'nominal logistic',
    levels: lv.levels,
    reference: refName,
    n,
    equations,
    logLik: fit.logLik,
    nullLogLik,
    gTest: { statistic: g, df: gdf, pValue: chi2Dist(gdf).sf(g) },
    goodnessOfFit: {
      pearson: { statistic: pearsonStat, df: gofDf, pValue: gofDf > 0 ? chi2Dist(gofDf).sf(pearsonStat) : NaN },
      deviance: { statistic: devStat, df: gofDf, pValue: gofDf > 0 ? chi2Dist(gofDf).sf(devStat) : NaN },
    },
    aic: -2 * fit.logLik + 2 * nTheta,
    bic: -2 * fit.logLik + nTheta * Math.log(n),
    probabilities,
    iterations: fit.iterations,
    converged: fit.converged,
    omitted,
    predict,
  }
}
