/**
 * Small shared optimizers: Nelder–Mead simplex (derivative-free) and Newton–Raphson maximisation with a
 * numerical Hessian, used by the life-data, stability and repairable-systems modules.
 */
import { inverse, matrix, matvec, type Matrix } from './linalg.js'

export interface NelderMeadResult {
  x: Float64Array
  value: number
  iterations: number
  converged: boolean
}

/** Minimize f by Nelder–Mead. `step` sets the initial simplex size per coordinate. */
export function nelderMead(
  f: (x: Float64Array) => number,
  x0: ArrayLike<number>,
  options: { maxIter?: number; tol?: number; step?: number | ArrayLike<number> } = {},
): NelderMeadResult {
  const n = x0.length
  const maxIter = options.maxIter ?? 2000 * n
  const tol = options.tol ?? 1e-10
  const stepOf = (i: number) => (typeof options.step === 'number' ? options.step : options.step ? options.step[i]! : Math.max(0.1 * Math.abs(x0[i]!), 0.05))
  const pts: Float64Array[] = [Float64Array.from(x0)]
  for (let i = 0; i < n; i++) {
    const p = Float64Array.from(x0)
    p[i] = p[i]! + stepOf(i)
    pts.push(p)
  }
  const vals = pts.map((p) => {
    const v = f(p)
    return Number.isFinite(v) ? v : Infinity
  })
  let it = 0
  let converged = false
  const order = () => {
    const idx = pts.map((_, i) => i).sort((a, b) => vals[a]! - vals[b]!)
    const np = idx.map((i) => pts[i]!)
    const nv = idx.map((i) => vals[i]!)
    for (let i = 0; i <= n; i++) {
      pts[i] = np[i]!
      vals[i] = nv[i]!
    }
  }
  const evalAt = (p: Float64Array) => {
    const v = f(p)
    return Number.isFinite(v) ? v : Infinity
  }
  for (it = 1; it <= maxIter; it++) {
    order()
    const best = vals[0]!
    const worst = vals[n]!
    if (Math.abs(worst - best) <= tol * (Math.abs(best) + Math.abs(worst) + 1e-12)) {
      let spread = 0
      for (let i = 1; i <= n; i++) for (let j = 0; j < n; j++) spread = Math.max(spread, Math.abs(pts[i]![j]! - pts[0]![j]!))
      if (spread <= 1e-8 * (1 + Math.max(...Array.from(pts[0]!, Math.abs)))) {
        converged = true
        break
      }
    }
    const centroid = new Float64Array(n)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) centroid[j] = centroid[j]! + pts[i]![j]! / n
    const reflect = (coef: number) => Float64Array.from(centroid, (c, j) => c + coef * (pts[n]![j]! - c))
    const xr = reflect(-1)
    const fr = evalAt(xr)
    if (fr < vals[0]!) {
      const xe = reflect(-2)
      const fe = evalAt(xe)
      if (fe < fr) {
        pts[n] = xe
        vals[n] = fe
      } else {
        pts[n] = xr
        vals[n] = fr
      }
    } else if (fr < vals[n - 1]!) {
      pts[n] = xr
      vals[n] = fr
    } else {
      const outside = fr < vals[n]!
      const xc = reflect(outside ? -0.5 : 0.5)
      const fc = evalAt(xc)
      if (fc < (outside ? fr : vals[n]!)) {
        pts[n] = xc
        vals[n] = fc
      } else {
        for (let i = 1; i <= n; i++) {
          pts[i] = Float64Array.from(pts[i]!, (v, j) => pts[0]![j]! + 0.5 * (v - pts[0]![j]!))
          vals[i] = evalAt(pts[i]!)
        }
      }
    }
  }
  order()
  return { x: pts[0]!, value: vals[0]!, iterations: it, converged }
}

/** Numerical gradient by central differences. */
export function numGradient(f: (x: Float64Array) => number, x: Float64Array): Float64Array {
  const g = new Float64Array(x.length)
  for (let j = 0; j < x.length; j++) {
    const h = 1e-5 * Math.max(1, Math.abs(x[j]!))
    const a = Float64Array.from(x)
    const b = Float64Array.from(x)
    a[j] = a[j]! + h
    b[j] = b[j]! - h
    g[j] = (f(a) - f(b)) / (2 * h)
  }
  return g
}

/** Numerical Hessian by Richardson-extrapolated central differences of a gradient. */
export function numHessian(grad: (x: Float64Array) => Float64Array, x: Float64Array): Matrix {
  const k = x.length
  const H = matrix(k, k)
  for (let j = 0; j < k; j++) {
    const h = 1e-4 * Math.max(1, Math.abs(x[j]!))
    const g = (step: number) => {
      const t = Float64Array.from(x)
      t[j] = t[j]! + step
      return grad(t)
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

export interface NewtonResult {
  theta: Float64Array
  logLik: number
  /** Observed-information covariance (−H)⁻¹. */
  covariance: Matrix
  hessian: Matrix
  iterations: number
  converged: boolean
}

/**
 * Maximize a log-likelihood by Newton–Raphson with step halving. `grad` defaults to numerical differences;
 * the Hessian is always numerical (from the gradient). Falls back to Nelder–Mead polishing when Newton stalls.
 */
export function newtonMax(
  loglik: (t: Float64Array) => number,
  theta0: ArrayLike<number>,
  options: { grad?: (t: Float64Array) => Float64Array; maxIter?: number; tol?: number } = {},
): NewtonResult {
  const grad = options.grad ?? ((t: Float64Array) => numGradient(loglik, t))
  const maxIter = options.maxIter ?? 200
  const tol = options.tol ?? 1e-10
  let theta: Float64Array = Float64Array.from(theta0)
  let ll = loglik(theta)
  if (!Number.isFinite(ll)) {
    const nm = nelderMead((t) => -loglik(t), theta)
    theta = nm.x
    ll = -nm.value
  }
  let converged = false
  let it = 0
  const k = theta.length
  for (it = 1; it <= maxIter; it++) {
    const g = grad(theta)
    const H = numHessian(grad, theta)
    let step: Float64Array
    try {
      step = matvec(inverse(H), g)
      for (let j = 0; j < k; j++) step[j] = -step[j]!
    } catch {
      step = Float64Array.from(g, (v) => 0.01 * v)
    }
    let dot = 0
    for (let j = 0; j < k; j++) dot += step[j]! * g[j]!
    if (!(dot > 0)) step = Float64Array.from(g, (v) => 0.01 * v)
    let alpha = 1
    let next = theta
    let llNext = -Infinity
    for (let half = 0; half < 40; half++) {
      next = Float64Array.from(theta, (v, j) => v + alpha * step[j]!)
      llNext = loglik(next)
      if (Number.isFinite(llNext) && llNext >= ll - 1e-12) break
      alpha /= 2
    }
    const gmax = Math.max(...Array.from(g, Math.abs))
    const improved = llNext - ll
    theta = next
    ll = llNext
    if (Math.abs(improved) < tol * (Math.abs(ll) + 1) && gmax < 1e-5 * (Math.abs(ll) + 1)) {
      converged = true
      break
    }
  }
  if (!converged) {
    const nm = nelderMead((t) => -loglik(t), theta, { step: 0.01 })
    if (-nm.value > ll) {
      theta = nm.x
      ll = -nm.value
    }
    converged = nm.converged
  }
  const H = numHessian(grad, theta)
  let covariance: Matrix
  try {
    covariance = inverse(matrix(k, k, Float64Array.from(H.data, (v) => -v)))
  } catch {
    covariance = matrix(k, k, new Float64Array(k * k).fill(NaN))
  }
  return { theta, logLik: ll, covariance, hessian: H, iterations: it, converged }
}
