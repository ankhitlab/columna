/**
 * Orthogonal (Deming) regression and Partial Least Squares (Minitab Stat › Regression › Orthogonal
 * Regression / Partial Least Squares). PLS uses NIPALS with optional standardization, reports R²X / R²Y
 * per component, leave-one-out cross-validated predicted R² (Minitab's default for choosing the number
 * of components), scores, loadings, weights and regression coefficients on the original scale.
 */
import { t as tDist } from './dist.js'
import { fromColumns, inverse, matmul, matrix, transpose, type Matrix } from './linalg.js'
import { completeRows, toColumns, type Column, type Predictors } from './regression.js'

export interface OrthogonalResult {
  test: 'orthogonal regression'
  /** Ratio of error variances Var(ε_y) / Var(ε_x) used in the fit. */
  errorVarianceRatio: number
  intercept: { estimate: number; se: number; ci: [number, number]; z: number; pValue: number }
  slope: { estimate: number; se: number; ci: [number, number]; z: number; pValue: number }
  n: number
  /** Orthogonal-residual sum of squares (weighted perpendicular distances). */
  sse: number
  fitted: Float64Array
  residuals: Float64Array
  /** Test of slope = 1 and intercept = 0 (method-comparison hypotheses). */
  slopeEqualsOne: { z: number; pValue: number }
  interceptEqualsZero: { z: number; pValue: number }
  confidence: number
}

function demingSlope(x: Float64Array, y: Float64Array, delta: number): { b1: number; b0: number } {
  const n = x.length
  let mx = 0
  let my = 0
  for (let i = 0; i < n; i++) {
    mx += x[i]!
    my += y[i]!
  }
  mx /= n
  my /= n
  let sxx = 0
  let syy = 0
  let sxy = 0
  for (let i = 0; i < n; i++) {
    sxx += (x[i]! - mx) ** 2
    syy += (y[i]! - my) ** 2
    sxy += (x[i]! - mx) * (y[i]! - my)
  }
  const b1 = sxy === 0 ? (syy > delta * sxx ? Infinity : 0) : (syy - delta * sxx + Math.sqrt((syy - delta * sxx) ** 2 + 4 * delta * sxy * sxy)) / (2 * sxy)
  return { b1, b0: my - b1 * mx }
}

/**
 * Orthogonal regression of y on x with known error-variance ratio δ = Var(ε_y)/Var(ε_x) (Deming; δ = 1
 * is total least squares). Standard errors are jackknife estimates (Linnet 1990), intervals use t(n − 2).
 */
export function orthogonalRegression(x: Column, y: Column, options: { errorVarianceRatio?: number; confidence?: number } = {}): OrthogonalResult {
  const delta = options.errorVarianceRatio ?? 1
  if (!(delta > 0)) throw new RangeError('errorVarianceRatio must be > 0')
  const confidence = options.confidence ?? 0.95
  const { keep } = completeRows(y, [x])
  const n = keep.length
  if (n < 3) throw new RangeError(`orthogonalRegression needs at least 3 complete pairs, got ${n}`)
  const xv = Float64Array.from(keep, (i) => x[i] as number)
  const yv = Float64Array.from(keep, (i) => y[i] as number)
  const full = demingSlope(xv, yv, delta)
  // jackknife: leave-one-out sums updated in O(1) per point (O(n) total)
  const b1s = new Float64Array(n)
  const b0s = new Float64Array(n)
  let mx = 0
  let my = 0
  for (let i = 0; i < n; i++) {
    mx += xv[i]!
    my += yv[i]!
  }
  mx /= n
  my /= n
  let sxx = 0
  let syy = 0
  let sxy = 0
  for (let i = 0; i < n; i++) {
    sxx += (xv[i]! - mx) ** 2
    syy += (yv[i]! - my) ** 2
    sxy += (xv[i]! - mx) * (yv[i]! - my)
  }
  const m1 = n - 1
  for (let i = 0; i < n; i++) {
    const dx = xv[i]! - mx
    const dy = yv[i]! - my
    // centered sums without point i: S − d² · n/(n−1)
    const f = n / m1
    const sxxI = sxx - dx * dx * f
    const syyI = syy - dy * dy * f
    const sxyI = sxy - dx * dy * f
    const b1 = sxyI === 0 ? (syyI > delta * sxxI ? Infinity : 0) : (syyI - delta * sxxI + Math.sqrt((syyI - delta * sxxI) ** 2 + 4 * delta * sxyI * sxyI)) / (2 * sxyI)
    const mxI = (n * mx - xv[i]!) / m1
    const myI = (n * my - yv[i]!) / m1
    b1s[i] = b1
    b0s[i] = myI - b1 * mxI
  }
  const jackSe = (v: Float64Array) => {
    let m = 0
    for (let i = 0; i < n; i++) m += v[i]!
    m /= n
    let s = 0
    for (let i = 0; i < n; i++) s += (v[i]! - m) ** 2
    return Math.sqrt(((n - 1) / n) * s)
  }
  const se1 = jackSe(b1s)
  const se0 = jackSe(b0s)
  const td = tDist(n - 2)
  const tc = td.ppf(0.5 + confidence / 2)
  const fitted = new Float64Array(n)
  const residuals = new Float64Array(n)
  let sse = 0
  for (let i = 0; i < n; i++) {
    fitted[i] = full.b0 + full.b1 * xv[i]!
    residuals[i] = yv[i]! - fitted[i]!
    // perpendicular (weighted) distance²: r² / (δ + b1²)·δ
    sse += (delta * residuals[i]! ** 2) / (delta + full.b1 * full.b1)
  }
  const pOf = (z: number) => Math.min(1, 2 * td.sf(Math.abs(z)))
  return {
    test: 'orthogonal regression',
    errorVarianceRatio: delta,
    intercept: { estimate: full.b0, se: se0, ci: [full.b0 - tc * se0, full.b0 + tc * se0], z: full.b0 / se0, pValue: pOf(full.b0 / se0) },
    slope: { estimate: full.b1, se: se1, ci: [full.b1 - tc * se1, full.b1 + tc * se1], z: full.b1 / se1, pValue: pOf(full.b1 / se1) },
    n,
    sse,
    fitted,
    residuals,
    slopeEqualsOne: { z: (full.b1 - 1) / se1, pValue: pOf((full.b1 - 1) / se1) },
    interceptEqualsZero: { z: full.b0 / se0, pValue: pOf(full.b0 / se0) },
    confidence,
  }
}

export interface PlsOptions {
  /** Number of components to fit (default min(p, n − 1, 10)). */
  components?: number
  /** Standardize predictors and responses (default false: center only). */
  standardize?: boolean
  /** Leave-one-out cross-validation for predicted R² (default true). */
  crossValidate?: boolean
  names?: string[]
  responseNames?: string[]
}

export interface PlsComponentRow {
  components: number
  /** Cumulative R² of X explained. */
  r2x: number
  /** Cumulative R² of Y (all responses pooled). */
  r2y: number
  /** Cross-validated predicted R² (when enabled). */
  r2pred?: number
  press?: number
}

export interface PlsResult {
  test: 'partial least squares'
  n: number
  p: number
  /** Number of responses. */
  m: number
  /** Components fitted. */
  components: number
  /** Components selected (max predicted R², or all when not cross-validated). */
  selected: number
  summary: PlsComponentRow[]
  /** Regression coefficients on the original scale for the selected components: [response][0 = constant, 1..p]. */
  coefficients: Array<{ response: string; constant: number; coef: number[] }>
  /** Standardized coefficients (when standardize) or centered-scale coefficients: [response][p]. */
  names: string[]
  responseNames: string[]
  scores: Matrix
  xWeights: Matrix
  xLoadings: Matrix
  yLoadings: Matrix
  fitted: Float64Array[]
  residuals: Float64Array[]
  /** Leverage of each observation in the selected model (from the scores). */
  leverage: Float64Array
  predict(x: ArrayLike<number> | ArrayLike<number>[], components?: number): number[][]
}

/** NIPALS PLS on already centered/scaled matrices; returns weights, loadings, scores per component. */
function nipals(X0: Matrix, Y0: Matrix, A: number) {
  const n = X0.rows
  const p = X0.cols
  const m = Y0.cols
  const X = matrix(n, p, Float64Array.from(X0.data))
  const Y = matrix(n, m, Float64Array.from(Y0.data))
  const W = matrix(p, A)
  const P = matrix(p, A)
  const Q = matrix(m, A)
  const T = matrix(n, A)
  const ssx: number[] = []
  const ssy: number[] = []
  for (let a = 0; a < A; a++) {
    // u = column of Y with the largest variance
    let best = 0
    let bestVar = -1
    for (let j = 0; j < m; j++) {
      let s = 0
      for (let i = 0; i < n; i++) s += Y.data[i * m + j]! ** 2
      if (s > bestVar) {
        bestVar = s
        best = j
      }
    }
    const u = new Float64Array(n)
    for (let i = 0; i < n; i++) u[i] = Y.data[i * m + best]!
    const w = new Float64Array(p)
    const t = new Float64Array(n)
    const q = new Float64Array(m)
    let tOld: Float64Array | null = null
    for (let iter = 0; iter < 500; iter++) {
      let uu = 0
      for (let i = 0; i < n; i++) uu += u[i]! * u[i]!
      if (uu === 0) break
      w.fill(0)
      for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) w[j] = w[j]! + X.data[i * p + j]! * u[i]!
      let wn = 0
      for (let j = 0; j < p; j++) wn += w[j]! ** 2
      wn = Math.sqrt(wn)
      if (wn === 0) break
      for (let j = 0; j < p; j++) w[j] = w[j]! / wn
      let tt = 0
      for (let i = 0; i < n; i++) {
        let s = 0
        for (let j = 0; j < p; j++) s += X.data[i * p + j]! * w[j]!
        t[i] = s
        tt += s * s
      }
      q.fill(0)
      for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) q[j] = q[j]! + Y.data[i * m + j]! * t[i]!
      for (let j = 0; j < m; j++) q[j] = q[j]! / tt
      if (m === 1) break
      let qq = 0
      for (let j = 0; j < m; j++) qq += q[j]! ** 2
      for (let i = 0; i < n; i++) {
        let s = 0
        for (let j = 0; j < m; j++) s += Y.data[i * m + j]! * q[j]!
        u[i] = s / qq
      }
      if (tOld) {
        let d = 0
        let s = 0
        for (let i = 0; i < n; i++) {
          d += (t[i]! - tOld[i]!) ** 2
          s += t[i]! ** 2
        }
        if (d <= 1e-24 * s) break
      }
      tOld = Float64Array.from(t)
    }
    let tt = 0
    for (let i = 0; i < n; i++) tt += t[i]! ** 2
    const pl = new Float64Array(p)
    for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) pl[j] = pl[j]! + X.data[i * p + j]! * t[i]!
    for (let j = 0; j < p; j++) pl[j] = pl[j]! / tt
    let sx = 0
    let sy = 0
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < p; j++) {
        const d = t[i]! * pl[j]!
        X.data[i * p + j] = X.data[i * p + j]! - d
        sx += d * d
      }
      for (let j = 0; j < m; j++) {
        const d = t[i]! * q[j]!
        Y.data[i * m + j] = Y.data[i * m + j]! - d
        sy += d * d
      }
    }
    ssx.push(sx)
    ssy.push(sy)
    for (let j = 0; j < p; j++) {
      W.data[j * A + a] = w[j]!
      P.data[j * A + a] = pl[j]!
    }
    for (let j = 0; j < m; j++) Q.data[j * A + a] = q[j]!
    for (let i = 0; i < n; i++) T.data[i * A + a] = t[i]!
  }
  return { W, P, Q, T, ssx, ssy }
}

/** B (p × m) = W (PᵀW)⁻¹ Qᵀ using the first `a` components. */
function plsCoefficients(W: Matrix, P: Matrix, Q: Matrix, a: number): Matrix {
  const p = W.rows
  const m = Q.rows
  const Wa = matrix(p, a)
  const Pa = matrix(p, a)
  const Qa = matrix(m, a)
  for (let j = 0; j < p; j++) for (let k = 0; k < a; k++) {
    Wa.data[j * a + k] = W.data[j * W.cols + k]!
    Pa.data[j * a + k] = P.data[j * P.cols + k]!
  }
  for (let j = 0; j < m; j++) for (let k = 0; k < a; k++) Qa.data[j * a + k] = Q.data[j * Q.cols + k]!
  const PtW = matmul(transpose(Pa), Wa)
  return matmul(matmul(Wa, inverse(PtW)), transpose(Qa))
}

/**
 * Partial least squares regression of one or more responses on the predictors.
 *   pls(y, { x1, x2, x3 }, { components: 2, standardize: true })
 */
export function pls(y: Column | Column[], X: Predictors, options: PlsOptions = {}): PlsResult {
  const ys: Column[] = Array.isArray(y) && y.length && (Array.isArray(y[0]) || ArrayBuffer.isView(y[0])) ? (y as Column[]) : [y as Column]
  const { names, cols } = toColumns(X, options.names)
  const responseNames = ys.map((_, i) => options.responseNames?.[i] ?? (ys.length === 1 ? 'y' : `y${i + 1}`))
  const { keep } = completeRows(ys[0]!, [...cols, ...ys.slice(1)])
  const n = keep.length
  const p = cols.length
  const m = ys.length
  const A = Math.max(1, Math.min(options.components ?? Math.min(p, n - 1, 10), p, n - 1))
  const standardize = options.standardize ?? false
  const Xraw = fromColumns(cols.map((c) => Float64Array.from(keep, (i) => c[i] as number)))
  const Yraw = fromColumns(ys.map((c) => Float64Array.from(keep, (i) => c[i] as number)))
  const prep = (M: Matrix) => {
    const mean = new Float64Array(M.cols)
    const scale = new Float64Array(M.cols).fill(1)
    for (let j = 0; j < M.cols; j++) {
      let s = 0
      for (let i = 0; i < M.rows; i++) s += M.data[i * M.cols + j]!
      mean[j] = s / M.rows
      if (standardize) {
        let v = 0
        for (let i = 0; i < M.rows; i++) v += (M.data[i * M.cols + j]! - mean[j]!) ** 2
        scale[j] = Math.sqrt(v / (M.rows - 1)) || 1
      }
    }
    const out = matrix(M.rows, M.cols)
    for (let i = 0; i < M.rows; i++) for (let j = 0; j < M.cols; j++) out.data[i * M.cols + j] = (M.data[i * M.cols + j]! - mean[j]!) / scale[j]!
    return { out, mean, scale }
  }
  const px = prep(Xraw)
  const py = prep(Yraw)
  const fit = nipals(px.out, py.out, A)
  let ssxTotal = 0
  for (const v of px.out.data) ssxTotal += v * v
  let ssyTotal = 0
  for (const v of py.out.data) ssyTotal += v * v
  const summary: PlsComponentRow[] = []
  let cx = 0
  let cy = 0
  for (let a = 0; a < A; a++) {
    cx += fit.ssx[a]!
    cy += fit.ssy[a]!
    summary.push({ components: a + 1, r2x: cx / ssxTotal, r2y: cy / ssyTotal })
  }
  // leave-one-out cross-validation on the (centered/scaled) data, PRESS on the original response scale
  const crossValidate = options.crossValidate ?? true
  if (crossValidate && n > A + 1) {
    const press = new Float64Array(A)
    for (let leave = 0; leave < n; leave++) {
      const Xi = matrix(n - 1, p)
      const Yi = matrix(n - 1, m)
      let r = 0
      for (let i = 0; i < n; i++) {
        if (i === leave) continue
        for (let j = 0; j < p; j++) Xi.data[r * p + j] = Xraw.data[i * p + j]!
        for (let j = 0; j < m; j++) Yi.data[r * m + j] = Yraw.data[i * m + j]!
        r++
      }
      const pxi = prep(Xi)
      const pyi = prep(Yi)
      const fi = nipals(pxi.out, pyi.out, A)
      const xs = new Float64Array(p)
      for (let j = 0; j < p; j++) xs[j] = (Xraw.data[leave * p + j]! - pxi.mean[j]!) / pxi.scale[j]!
      for (let a = 1; a <= A; a++) {
        const B = plsCoefficients(fi.W, fi.P, fi.Q, a)
        for (let k = 0; k < m; k++) {
          let s = 0
          for (let j = 0; j < p; j++) s += xs[j]! * B.data[j * m + k]!
          const pred = s * pyi.scale[k]! + pyi.mean[k]!
          press[a - 1] = press[a - 1]! + (Yraw.data[leave * m + k]! - pred) ** 2
        }
      }
    }
    let sstY = 0
    for (let k = 0; k < m; k++) {
      let mean = 0
      for (let i = 0; i < n; i++) mean += Yraw.data[i * m + k]!
      mean /= n
      for (let i = 0; i < n; i++) sstY += (Yraw.data[i * m + k]! - mean) ** 2
    }
    for (let a = 0; a < A; a++) {
      summary[a]!.press = press[a]!
      summary[a]!.r2pred = 1 - press[a]! / sstY
    }
  }
  let selected = A
  if (summary[0]!.r2pred !== undefined) {
    let best = -Infinity
    for (const row of summary) if (row.r2pred! > best + 1e-12) {
      best = row.r2pred!
      selected = row.components
    }
  }
  const coefFor = (a: number) => {
    const B = plsCoefficients(fit.W, fit.P, fit.Q, a)
    return Array.from({ length: m }, (_, k) => {
      const coef = Array.from({ length: p }, (_, j) => (B.data[j * m + k]! * py.scale[k]!) / px.scale[j]!)
      let c = py.mean[k]!
      for (let j = 0; j < p; j++) c -= coef[j]! * px.mean[j]!
      return { response: responseNames[k]!, constant: c, coef }
    })
  }
  const coefficients = coefFor(selected)
  const fitted = Array.from({ length: m }, () => new Float64Array(n))
  const residuals = Array.from({ length: m }, () => new Float64Array(n))
  for (let k = 0; k < m; k++) {
    for (let i = 0; i < n; i++) {
      let s = coefficients[k]!.constant
      for (let j = 0; j < p; j++) s += coefficients[k]!.coef[j]! * Xraw.data[i * p + j]!
      fitted[k]![i] = s
      residuals[k]![i] = Yraw.data[i * m + k]! - s
    }
  }
  // leverage from scores of the selected components: 1/n + tᵢ (TᵀT)⁻¹ tᵢ
  const leverage = new Float64Array(n)
  const TtT = matrix(selected, selected)
  for (let i = 0; i < n; i++) for (let a = 0; a < selected; a++) for (let b = 0; b < selected; b++) TtT.data[a * selected + b] += fit.T.data[i * A + a]! * fit.T.data[i * A + b]!
  const TtTinv = inverse(TtT)
  for (let i = 0; i < n; i++) {
    let h = 1 / n
    for (let a = 0; a < selected; a++) for (let b = 0; b < selected; b++) h += fit.T.data[i * A + a]! * TtTinv.data[a * selected + b]! * fit.T.data[i * A + b]!
    leverage[i] = h
  }
  const predict = (x: ArrayLike<number> | ArrayLike<number>[], components = selected): number[][] => {
    const rows: ArrayLike<number>[] = typeof (x as ArrayLike<number>)[0] === 'number' || x.length === 0 ? [x as ArrayLike<number>] : (x as ArrayLike<number>[])
    const cf = components === selected ? coefficients : coefFor(components)
    return rows.map((row) => {
      if (row.length !== p) throw new RangeError(`predict: expected ${p} predictor values, got ${row.length}`)
      return cf.map((c) => {
        let s = c.constant
        for (let j = 0; j < p; j++) s += c.coef[j]! * row[j]!
        return s
      })
    })
  }
  return {
    test: 'partial least squares',
    n,
    p,
    m,
    components: A,
    selected,
    summary,
    coefficients,
    names,
    responseNames,
    scores: fit.T,
    xWeights: fit.W,
    xLoadings: fit.P,
    yLoadings: fit.Q,
    fitted,
    residuals,
    leverage,
    predict,
  }
}
