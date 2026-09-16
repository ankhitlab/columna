/**
 * Nonlinear regression (Minitab Stat › Regression › Nonlinear Regression) by Levenberg–Marquardt with
 * a finite-difference (or user-supplied) Jacobian: parameter estimates with SE, t, p and confidence
 * intervals from the linearized (JᵀJ)⁻¹, S, lack-of-fit-free ANOVA, iteration history and prediction.
 */
import { t as tDist } from './dist.js'
import { cholesky, choleskySolve, inverse, matrix, type Matrix } from './linalg.js'

export type NlsModel = (x: number | number[], params: number[]) => number

export interface NlsOptions {
  /** Starting values (required). */
  start: number[]
  names?: string[]
  confidence?: number
  maxIter?: number
  /** Relative convergence tolerance on SSE (default 1e-10). */
  tol?: number
  /** Analytic Jacobian: partial derivatives of the model at (x, params), one per parameter. */
  jacobian?: (x: number | number[], params: number[]) => number[]
  /** Optional observation weights. */
  weights?: ArrayLike<number>
  /** Parameter bounds [lower, upper] per parameter (projected). */
  bounds?: Array<[number, number]>
}

export interface NlsParameter {
  name: string
  estimate: number
  se: number
  t: number
  pValue: number
  ci: [number, number]
}

export interface NlsResult {
  test: 'nonlinear regression'
  parameters: NlsParameter[]
  n: number
  sse: number
  s: number
  df: number
  /** 1 − SSE / SST (a pseudo-R² for nonlinear models). */
  r2: number
  iterations: number
  converged: boolean
  history: Array<{ iteration: number; sse: number; params: number[] }>
  fitted: Float64Array
  residuals: Float64Array
  /** Correlation matrix of the parameter estimates. */
  correlation: Matrix
  covariance: Matrix
  predict(x: number | number[] | Array<number | number[]>, options?: { confidence?: number }): Array<{ fit: number; se: number; ci: [number, number]; pi: [number, number] }>
}

function numericJacobian(model: NlsModel, x: number | number[], params: number[]): number[] {
  const out = new Array<number>(params.length)
  for (let j = 0; j < params.length; j++) {
    const h = 1e-5 * Math.abs(params[j]!) || 1e-8 // relative central-difference step
    const p1 = params.slice()
    const p2 = params.slice()
    p1[j] = p1[j]! + h
    p2[j] = p2[j]! - h
    out[j] = (model(x, p1) - model(x, p2)) / (2 * h)
  }
  return out
}

/**
 * Fit y ≈ model(x, θ) by least squares.
 *   nls((x, [a, b]) => a * (1 - Math.exp(-b * x)), x, y, { start: [500, 1e-4], names: ['a', 'b'] })
 */
export function nls(model: NlsModel, x: ArrayLike<number | number[]>, y: ArrayLike<number | null | undefined>, options: NlsOptions): NlsResult {
  if (x.length !== y.length) throw new RangeError(`nls: x has ${x.length} rows, y has ${y.length}`)
  const start = options.start
  if (!start?.length) throw new RangeError('nls: start values are required')
  const k = start.length
  const names = options.names ?? start.map((_, i) => `θ${i + 1}`)
  const confidence = options.confidence ?? 0.95
  const keep: number[] = []
  for (let i = 0; i < y.length; i++) if (typeof y[i] === 'number' && Number.isFinite(y[i] as number)) keep.push(i)
  const n = keep.length
  if (n <= k) throw new RangeError(`nls needs more observations than parameters (n = ${n}, k = ${k})`)
  const xs = keep.map((i) => x[i]!)
  const ys = Float64Array.from(keep, (i) => y[i] as number)
  const w = Float64Array.from(keep, (i) => (options.weights ? options.weights[i]! : 1))
  const jac = options.jacobian ?? ((xi: number | number[], p: number[]) => numericJacobian(model, xi, p))
  const bounds = options.bounds
  const clamp = (p: number[]) => (bounds ? p.map((v, j) => Math.min(bounds[j]![1], Math.max(bounds[j]![0], v))) : p)
  const residualsAt = (p: number[]): { r: Float64Array; sse: number } => {
    const r = new Float64Array(n)
    let sse = 0
    for (let i = 0; i < n; i++) {
      const v = model(xs[i]!, p)
      if (!Number.isFinite(v)) return { r, sse: Infinity }
      r[i] = ys[i]! - v
      sse += w[i]! * r[i]! ** 2
    }
    return { r, sse }
  }
  let params = clamp(start.slice())
  let cur = residualsAt(params)
  if (!Number.isFinite(cur.sse)) throw new RangeError('nls: model is not finite at the starting values')
  let lambda = 1e-3
  const maxIter = options.maxIter ?? 200
  const tol = options.tol ?? 1e-10
  const history: NlsResult['history'] = [{ iteration: 0, sse: cur.sse, params: params.slice() }]
  let converged = false
  let iterations = 0
  const J = matrix(n, k)
  for (let it = 1; it <= maxIter; it++) {
    iterations = it
    for (let i = 0; i < n; i++) {
      const row = jac(xs[i]!, params)
      for (let j = 0; j < k; j++) J.data[i * k + j] = row[j]!
    }
    // normal equations with weights: (JᵀWJ + λ diag) δ = JᵀW r
    const A = matrix(k, k)
    const g = new Float64Array(k)
    for (let i = 0; i < n; i++) {
      const wi = w[i]!
      for (let a = 0; a < k; a++) {
        const ja = J.data[i * k + a]! * wi
        g[a] = g[a]! + ja * cur.r[i]!
        for (let b = a; b < k; b++) A.data[a * k + b] += ja * J.data[i * k + b]!
      }
    }
    for (let a = 0; a < k; a++) for (let b = 0; b < a; b++) A.data[a * k + b] = A.data[b * k + a]!
    let accepted = false
    for (let tries = 0; tries < 30; tries++) {
      const M = matrix(k, k, Float64Array.from(A.data))
      for (let a = 0; a < k; a++) M.data[a * k + a] = M.data[a * k + a]! * (1 + lambda) + 1e-300
      const L = cholesky(M)
      if (!L) {
        lambda *= 10
        continue
      }
      const delta = choleskySolve(L, g)
      const next = clamp(params.map((v, j) => v + delta[j]!))
      const trial = residualsAt(next)
      if (trial.sse < cur.sse) {
        const rel = (cur.sse - trial.sse) / (cur.sse + 1e-300)
        params = next
        cur = trial
        lambda = Math.max(lambda / 10, 1e-12)
        accepted = true
        history.push({ iteration: it, sse: cur.sse, params: params.slice() })
        if (rel < tol) converged = true
        break
      }
      lambda *= 10
    }
    if (!accepted) {
      converged = true // no improving step exists at this precision
      break
    }
    if (converged) break
  }
  // final Jacobian and covariance
  for (let i = 0; i < n; i++) {
    const row = jac(xs[i]!, params)
    for (let j = 0; j < k; j++) J.data[i * k + j] = row[j]!
  }
  const A = matrix(k, k)
  for (let i = 0; i < n; i++) for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) A.data[a * k + b] += w[i]! * J.data[i * k + a]! * J.data[i * k + b]!
  const df = n - k
  const mse = cur.sse / df
  let cov: Matrix
  try {
    cov = inverse(A)
  } catch {
    cov = matrix(k, k, new Float64Array(k * k).fill(NaN))
  }
  for (let i = 0; i < k * k; i++) cov.data[i] = cov.data[i]! * mse
  const td = tDist(df)
  const tc = td.ppf(0.5 + confidence / 2)
  const parameters: NlsParameter[] = params.map((est, j) => {
    const se = Math.sqrt(cov.data[j * k + j]!)
    const t = est / se
    return { name: names[j]!, estimate: est, se, t, pValue: Math.min(1, 2 * td.sf(Math.abs(t))), ci: [est - tc * se, est + tc * se] }
  })
  const correlation = matrix(k, k)
  for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) correlation.data[a * k + b] = cov.data[a * k + b]! / Math.sqrt(cov.data[a * k + a]! * cov.data[b * k + b]!)
  const fitted = new Float64Array(n)
  for (let i = 0; i < n; i++) fitted[i] = ys[i]! - cur.r[i]!
  let ybar = 0
  for (let i = 0; i < n; i++) ybar += ys[i]!
  ybar /= n
  let sst = 0
  for (let i = 0; i < n; i++) sst += w[i]! * (ys[i]! - ybar) ** 2
  const predict: NlsResult['predict'] = (xNew, o = {}) => {
    const conf = o.confidence ?? confidence
    const c = td.ppf(0.5 + conf / 2)
    const multi = Array.isArray(xs[0])
    let rows: Array<number | number[]>
    if (typeof xNew === 'number') rows = [xNew]
    else if (multi) rows = Array.isArray(xNew[0]) ? (xNew as number[][]) : [xNew as number[]]
    else rows = xNew as number[]
    return rows.map((xi) => {
      const fit = model(xi, params)
      const gr = jac(xi, params)
      let q = 0
      for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) q += gr[a]! * cov.data[a * k + b]! * gr[b]!
      const se = Math.sqrt(Math.max(0, q))
      const sp = Math.sqrt(se * se + mse)
      return { fit, se, ci: [fit - c * se, fit + c * se], pi: [fit - c * sp, fit + c * sp] }
    })
  }
  return {
    test: 'nonlinear regression',
    parameters,
    n,
    sse: cur.sse,
    s: Math.sqrt(mse),
    df,
    r2: sst > 0 ? 1 - cur.sse / sst : NaN,
    iterations,
    converged,
    history,
    fitted,
    residuals: cur.r,
    correlation,
    covariance: cov,
    predict,
  }
}
