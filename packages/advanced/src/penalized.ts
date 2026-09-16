/**
 * Penalized and quantile regression (ridge / lasso / elastic net / QR).
 */
import { lstsq, matrix, type Matrix } from './linalg.js'

export interface PenalizedResult {
  method: 'ridge' | 'lasso' | 'elasticNet' | 'quantile'
  coef: number[]
  intercept: number
  alpha: number
  l1Ratio?: number
  fitted: number[]
  residual: number[]
  n: number
  p: number
}

function design(
  y: ArrayLike<number | null | undefined>,
  X: ArrayLike<ArrayLike<number>>,
): { yy: Float64Array; Xm: Matrix; n: number; p: number } {
  const rows = Array.from(X).map((r) => Array.from(r))
  const n = Math.min(y.length, rows.length)
  const p = rows[0]?.length ?? 0
  const yy = new Float64Array(n)
  const Xm = matrix(n, p)
  for (let i = 0; i < n; i++) {
    const yi = y[i]
    if (typeof yi !== 'number' || !Number.isFinite(yi)) throw new RangeError('penalized: non-finite y')
    yy[i] = yi
    for (let j = 0; j < p; j++) Xm.data[i * p + j] = rows[i]![j]!
  }
  return { yy, Xm, n, p }
}

function centerScale(Xm: Matrix, yy: Float64Array) {
  const { n, p } = { n: Xm.rows, p: Xm.cols }
  const xMean = new Array(p).fill(0)
  const xSd = new Array(p).fill(1)
  let yMean = 0
  for (let i = 0; i < n; i++) yMean += yy[i]!
  yMean /= n
  for (let j = 0; j < p; j++) {
    let s = 0
    for (let i = 0; i < n; i++) s += Xm.data[i * p + j]!
    xMean[j] = s / n
    let m2 = 0
    for (let i = 0; i < n; i++) {
      const v = Xm.data[i * p + j]! - xMean[j]!
      m2 += v * v
    }
    xSd[j] = Math.sqrt(m2 / n) || 1
  }
  const Z = matrix(n, p)
  const yc = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    yc[i] = yy[i]! - yMean
    for (let j = 0; j < p; j++) Z.data[i * p + j] = (Xm.data[i * p + j]! - xMean[j]!) / xSd[j]!
  }
  return { Z, yc, yMean, xMean, xSd }
}

/** Ridge regression via augmented least squares. */
export function ridge(
  y: ArrayLike<number | null | undefined>,
  X: ArrayLike<ArrayLike<number>>,
  options: { alpha?: number } = {},
): PenalizedResult {
  const alpha = options.alpha ?? 1
  const { yy, Xm, n, p } = design(y, X)
  const { Z, yc, yMean, xMean, xSd } = centerScale(Xm, yy)
  const aug = matrix(n + p, p)
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) aug.data[i * p + j] = Z.data[i * p + j]!
  const sqrtA = Math.sqrt(alpha)
  for (let j = 0; j < p; j++) aug.data[(n + j) * p + j] = sqrtA
  const yAug = new Float64Array(n + p)
  for (let i = 0; i < n; i++) yAug[i] = yc[i]!
  const fit = lstsq(aug, yAug)
  const coefStd = Array.from(fit.coef)
  const coef = coefStd.map((c, j) => c / xSd[j]!)
  let intercept = yMean
  for (let j = 0; j < p; j++) intercept -= coef[j]! * xMean[j]!
  const fitted = new Array(n)
  const residual = new Array(n)
  for (let i = 0; i < n; i++) {
    let pred = intercept
    for (let j = 0; j < p; j++) pred += coef[j]! * Xm.data[i * p + j]!
    fitted[i] = pred
    residual[i] = yy[i]! - pred
  }
  return { method: 'ridge', coef, intercept, alpha, fitted, residual, n, p }
}

/** Coordinate-descent lasso / elastic net (standardized predictors). */
export function elasticNet(
  y: ArrayLike<number | null | undefined>,
  X: ArrayLike<ArrayLike<number>>,
  options: { alpha?: number; l1Ratio?: number; maxIter?: number; tol?: number } = {},
): PenalizedResult {
  const alpha = options.alpha ?? 1
  const l1Ratio = options.l1Ratio ?? 0.5
  const maxIter = options.maxIter ?? 1000
  const tol = options.tol ?? 1e-6
  const { yy, Xm, n, p } = design(y, X)
  const { Z, yc, yMean, xMean, xSd } = centerScale(Xm, yy)
  const beta = new Array(p).fill(0)
  const r = Float64Array.from(yc)
  const soft = (z: number, lam: number) => (z > lam ? z - lam : z < -lam ? z + lam : 0)
  const lam1 = alpha * l1Ratio
  const lam2 = alpha * (1 - l1Ratio)
  for (let iter = 0; iter < maxIter; iter++) {
    let maxDelta = 0
    for (let j = 0; j < p; j++) {
      // partial residual
      for (let i = 0; i < n; i++) r[i]! += beta[j]! * Z.data[i * p + j]!
      let rho = 0
      let zj2 = 0
      for (let i = 0; i < n; i++) {
        rho += Z.data[i * p + j]! * r[i]!
        zj2 += Z.data[i * p + j]! ** 2
      }
      rho /= n
      zj2 /= n
      const newB = soft(rho, lam1) / (zj2 + lam2)
      const delta = newB - beta[j]!
      maxDelta = Math.max(maxDelta, Math.abs(delta))
      beta[j] = newB
      for (let i = 0; i < n; i++) r[i]! -= beta[j]! * Z.data[i * p + j]!
    }
    if (maxDelta < tol) break
  }
  const coef = beta.map((c, j) => c / xSd[j]!)
  let intercept = yMean
  for (let j = 0; j < p; j++) intercept -= coef[j]! * xMean[j]!
  const fitted = new Array(n)
  const residual = new Array(n)
  for (let i = 0; i < n; i++) {
    let pred = intercept
    for (let j = 0; j < p; j++) pred += coef[j]! * Xm.data[i * p + j]!
    fitted[i] = pred
    residual[i] = yy[i]! - pred
  }
  return {
    method: l1Ratio >= 1 - 1e-12 ? 'lasso' : 'elasticNet',
    coef,
    intercept,
    alpha,
    l1Ratio,
    fitted,
    residual,
    n,
    p,
  }
}

export function lasso(
  y: ArrayLike<number | null | undefined>,
  X: ArrayLike<ArrayLike<number>>,
  options: { alpha?: number; maxIter?: number; tol?: number } = {},
): PenalizedResult {
  return elasticNet(y, X, { ...options, l1Ratio: 1 })
}

/** Quantile regression via iteratively reweighted least squares (check function). */
export function quantileRegression(
  y: ArrayLike<number | null | undefined>,
  X: ArrayLike<ArrayLike<number>>,
  options: { tau?: number; maxIter?: number } = {},
): PenalizedResult {
  const tau = options.tau ?? 0.5
  const maxIter = options.maxIter ?? 40
  const { yy, Xm, n, p } = design(y, X)
  // add intercept column
  const Xi = matrix(n, p + 1)
  for (let i = 0; i < n; i++) {
    Xi.data[i * (p + 1)] = 1
    for (let j = 0; j < p; j++) Xi.data[i * (p + 1) + 1 + j] = Xm.data[i * p + j]!
  }
  let beta = new Array(p + 1).fill(0)
  for (let iter = 0; iter < maxIter; iter++) {
    const fitted = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let s = 0
      for (let j = 0; j < p + 1; j++) s += beta[j]! * Xi.data[i * (p + 1) + j]!
      fitted[i] = s
    }
    const W = matrix(n, p + 1)
    const z = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      const e = yy[i]! - fitted[i]!
      const w = 1 / Math.max(1e-4, Math.abs(e))
      const adj = fitted[i]! + ((e >= 0 ? tau : tau - 1) / w)
      z[i] = adj * Math.sqrt(w)
      for (let j = 0; j < p + 1; j++) W.data[i * (p + 1) + j] = Xi.data[i * (p + 1) + j]! * Math.sqrt(w)
    }
    beta = Array.from(lstsq(W, z).coef)
  }
  const intercept = beta[0]!
  const coef = beta.slice(1)
  const fitted = new Array(n)
  const residual = new Array(n)
  for (let i = 0; i < n; i++) {
    let pred = intercept
    for (let j = 0; j < p; j++) pred += coef[j]! * Xm.data[i * p + j]!
    fitted[i] = pred
    residual[i] = yy[i]! - pred
  }
  return { method: 'quantile', coef, intercept, alpha: tau, fitted, residual, n, p }
}
