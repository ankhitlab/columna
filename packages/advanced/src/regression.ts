/**
 * Ordinary least squares with Minitab's Regression output: coefficient table (SE, t, p, CI, VIF),
 * S / R² / R²(adj) / R²(pred), ANOVA with sequential and adjusted SS, residual diagnostics
 * (standardized and deleted-t residuals, leverage, Cook's D, DFFITS), unusual observations,
 * Durbin–Watson, information criteria and prediction with confidence / prediction intervals.
 */
import { f as fDist, t as tDist } from './dist.js'
import { fromColumns, lstsq, matrix, matvec, type Matrix } from './linalg.js'

export type Column = ArrayLike<number | null | undefined>
/** Predictors as named columns, an array of columns, or an array of rows. */
export type Predictors = Record<string, Column> | Column[]

export interface OlsOptions {
  intercept?: boolean
  /** Names for array-form predictors (default x1, x2, …). */
  names?: string[]
  confidence?: number
  /** Observation weights (weighted least squares). */
  weights?: Column
}

export interface Coefficient {
  name: string
  coef: number
  se: number
  t: number
  pValue: number
  ci: [number, number]
  /** Variance inflation factor (undefined for the constant). */
  vif?: number
  /** True when the column was dropped as linearly dependent. */
  aliased?: boolean
}

export interface AnovaRow {
  source: string
  df: number
  ss: number
  ms: number
  f?: number
  pValue?: number
}

export interface TermRow {
  name: string
  df: number
  seqSS: number
  adjSS: number
  adjMS: number
  f: number
  pValue: number
}

export interface Prediction {
  fit: number
  se: number
  ci: [number, number]
  pi: [number, number]
}

export interface OlsResult {
  test: 'regression'
  n: number
  /** Number of estimated (non-aliased) coefficients including the constant. */
  p: number
  coefficients: Coefficient[]
  names: string[]
  intercept: boolean
  s: number
  r2: number
  r2adj: number
  r2pred: number
  press: number
  logLik: number
  aic: number
  aicc: number
  bic: number
  anova: { regression: AnovaRow; error: AnovaRow; total: AnovaRow }
  terms: TermRow[]
  fitted: Float64Array
  residuals: Float64Array
  standardizedResiduals: Float64Array
  /** Externally studentized (deleted-t) residuals. */
  studentizedResiduals: Float64Array
  leverage: Float64Array
  cooksD: Float64Array
  dffits: Float64Array
  durbinWatson: number
  /** Minitab's unusual observations: R = large standardized residual (|r| > 2), X = high leverage (h > 3p/n). */
  unusual: Array<{ index: number; y: number; fitted: number; residual: number; stdResidual: number; leverage: number; flags: string }>
  /** Row indices (of the input) dropped for missing values. */
  omitted: number[]
  confidence: number
  /** Predict at new rows (arrays of predictor values in `names` order, without the constant). */
  predict(x: ArrayLike<number> | ArrayLike<number>[], options?: { confidence?: number }): Prediction[]
  /** Coefficient covariance matrix (p × p, in coefficient order). */
  covariance: Matrix
  /** Design matrix used for the fit (after dropping missing rows). */
  design: Matrix
  y: Float64Array
}

/** Normalize predictors into named numeric columns; returns null entries for missing values. */
export function toColumns(X: Predictors, names?: string[]): { names: string[]; cols: Column[] } {
  if (Array.isArray(X)) {
    const cols = X as Column[]
    return { names: cols.map((_, i) => names?.[i] ?? `x${i + 1}`), cols }
  }
  const entries = Object.entries(X)
  return { names: entries.map(([k], i) => names?.[i] ?? k), cols: entries.map(([, v]) => v) }
}

function isNum(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v)
}

/** Listwise-complete rows of y and the predictor columns. */
export function completeRows(y: Column, cols: Column[], weights?: Column): { keep: number[]; omitted: number[] } {
  const n = y.length
  for (const c of cols) if (c.length !== n) throw new RangeError(`predictor length ${c.length} ≠ response length ${n}`)
  if (weights && weights.length !== n) throw new RangeError('weights length must match the response')
  const keep: number[] = []
  const omitted: number[] = []
  for (let i = 0; i < n; i++) {
    let ok = isNum(y[i]) && (!weights || (isNum(weights[i]) && (weights[i] as number) >= 0))
    for (let j = 0; ok && j < cols.length; j++) if (!isNum(cols[j]![i])) ok = false
    ;(ok ? keep : omitted).push(i)
  }
  return { keep, omitted }
}

/** Build the design matrix (with optional constant column first). */
export function designMatrix(cols: Column[], keep: number[], intercept: boolean): Matrix {
  const n = keep.length
  const p = cols.length + (intercept ? 1 : 0)
  const X = matrix(n, p)
  for (let i = 0; i < n; i++) {
    const r = keep[i]!
    let j = 0
    if (intercept) X.data[i * p + j++] = 1
    for (const c of cols) X.data[i * p + j++] = c[r] as number
  }
  return X
}

/**
 * Variance inflation factors from the full fit: 1/(1 − R²ⱼ) = (XᵀX)⁻¹ⱼⱼ · Σ(xⱼ − x̄ⱼ)² (exact identity for a
 * design with a constant column c; for weighted designs the centering uses the scaled constant column).
 */
export function vifFromFit(X: Matrix, xtxInv: Matrix, constantCol: number, aliased: Set<number>): Float64Array {
  const n = X.rows
  const p = X.cols
  const out = new Float64Array(p).fill(NaN)
  let cc = 0
  for (let i = 0; i < n; i++) cc += X.data[i * p + constantCol]! ** 2
  for (let j = 0; j < p; j++) {
    if (j === constantCol || aliased.has(j)) continue
    let xx = 0
    let xc = 0
    for (let i = 0; i < n; i++) {
      const v = X.data[i * p + j]!
      xx += v * v
      xc += v * X.data[i * p + constantCol]!
    }
    const centered = xx - (xc * xc) / cc
    out[j] = centered > 0 ? xtxInv.data[j * p + j]! * centered : Infinity
  }
  return out
}

/**
 * Ordinary (or weighted) least squares of `y` on `X`.
 *   ols(y, { x1, x2 })                      // named columns
 *   ols(y, [x1, x2], { names: ['a', 'b'] })
 */
export function ols(y: Column, X: Predictors, options: OlsOptions = {}): OlsResult {
  const intercept = options.intercept ?? true
  const confidence = options.confidence ?? 0.95
  if (!(confidence > 0 && confidence < 1)) throw new RangeError(`confidence must be in (0, 1), got ${confidence}`)
  const { names: xnames, cols } = toColumns(X, options.names)
  const { keep, omitted } = completeRows(y, cols, options.weights)
  const n = keep.length
  const design = designMatrix(cols, keep, intercept)
  const yv = Float64Array.from(keep, (i) => y[i] as number)
  const names = intercept ? ['Constant', ...xnames] : [...xnames]
  const pAll = design.cols
  if (n <= pAll) throw new RangeError(`ols needs more observations than coefficients (n = ${n}, p = ${pAll})`)
  // weighted LS: scale rows by √w
  let Xf = design
  let yf = yv
  let w: Float64Array | undefined
  if (options.weights) {
    w = Float64Array.from(keep, (i) => options.weights![i] as number)
    Xf = matrix(n, pAll, Float64Array.from(design.data))
    yf = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      const sw = Math.sqrt(w[i]!)
      yf[i] = yv[i]! * sw
      for (let j = 0; j < pAll; j++) Xf.data[i * pAll + j] = Xf.data[i * pAll + j]! * sw
    }
  }
  const fit = lstsq(Xf, yf)
  const p = fit.rank
  const dfe = n - p
  const sse = fit.sse
  const mse = sse / dfe
  const s = Math.sqrt(mse)
  // totals (about the weighted mean when there is a constant)
  let sst = 0
  if (intercept) {
    let sw = 0
    let sy = 0
    for (let i = 0; i < n; i++) {
      const wi = w ? w[i]! : 1
      sw += wi
      sy += wi * yv[i]!
    }
    const ybar = sy / sw
    for (let i = 0; i < n; i++) sst += (w ? w[i]! : 1) * (yv[i]! - ybar) ** 2
  } else for (let i = 0; i < n; i++) sst += (w ? w[i]! : 1) * yv[i]! ** 2
  const ssr = sst - sse
  const dfr = p - (intercept ? 1 : 0)
  const dft = n - (intercept ? 1 : 0)
  const fitted = w ? matvec(design, fit.coef) : fit.fitted
  const residuals = new Float64Array(n)
  for (let i = 0; i < n; i++) residuals[i] = yv[i]! - fitted[i]!
  // residual diagnostics (on the weighted scale when weighted)
  const lev = fit.leverage
  const std = new Float64Array(n)
  const stud = new Float64Array(n)
  const cook = new Float64Array(n)
  const dffits = new Float64Array(n)
  let press = 0
  for (let i = 0; i < n; i++) {
    const e = fit.residuals[i]!
    const h = lev[i]!
    const denom = 1 - h
    press += denom > 1e-12 ? (e / denom) ** 2 : 0
    const r = denom > 1e-12 ? e / (s * Math.sqrt(denom)) : NaN
    std[i] = r
    const sMinus2 = dfe > 1 ? (sse - (e * e) / denom) / (dfe - 1) : NaN
    stud[i] = denom > 1e-12 && sMinus2 > 0 ? e / (Math.sqrt(sMinus2) * Math.sqrt(denom)) : NaN
    cook[i] = (r * r * h) / (p * denom)
    dffits[i] = stud[i]! * Math.sqrt(h / denom)
  }
  const r2 = sst > 0 ? 1 - sse / sst : NaN
  const r2adj = sst > 0 ? 1 - (sse / dfe) / (sst / dft) : NaN
  const r2pred = sst > 0 ? 1 - press / sst : NaN
  // coefficient table
  const td = tDist(dfe)
  const tcrit = td.ppf(0.5 + confidence / 2)
  const covariance = matrix(pAll, pAll, Float64Array.from(fit.xtxInv.data, (v) => v * mse))
  const aliased = new Set(fit.dependent)
  const coefficients: Coefficient[] = names.map((name, j) => {
    if (aliased.has(j)) return { name, coef: 0, se: NaN, t: NaN, pValue: NaN, ci: [NaN, NaN], aliased: true }
    const se = Math.sqrt(covariance.data[j * pAll + j]!)
    const tt = fit.coef[j]! / se
    return { name, coef: fit.coef[j]!, se, t: tt, pValue: Math.min(1, 2 * td.sf(Math.abs(tt))), ci: [fit.coef[j]! - tcrit * se, fit.coef[j]! + tcrit * se] }
  })
  // VIF: 1 / (1 − R²_j) from the full fit (needs a constant)
  if (intercept && cols.length > 1) {
    const vif = vifFromFit(Xf, fit.xtxInv, 0, aliased)
    for (let j = 1; j < pAll; j++) if (!aliased.has(j)) coefficients[j]!.vif = vif[j]!
  } else if (intercept) for (let j = 1; j < pAll; j++) if (!aliased.has(j)) coefficients[j]!.vif = 1
  // sequential SS: (Qᵀy)ₖ² from the QR of the full design (columns in order); refit only when rank-deficient
  const terms: TermRow[] = []
  let prevSse = sst
  const startCol = intercept ? 1 : 0
  const fullRank = fit.dependent.length === 0
  for (let k = startCol; k < pAll; k++) {
    let seqSS: number
    if (fullRank) seqSS = fit.qty[k]! ** 2
    else {
      let sseK: number
      if (k === pAll - 1) sseK = sse
      else {
        const sub = matrix(n, k + 1)
        for (let i = 0; i < n; i++) for (let c = 0; c <= k; c++) sub.data[i * (k + 1) + c] = Xf.data[i * pAll + c]!
        sseK = lstsq(sub, yf).sse
      }
      seqSS = prevSse - sseK
      prevSse = sseK
    }
    const c = coefficients[k]!
    const adjSS = c.aliased ? 0 : c.t * c.t * mse
    const df = c.aliased ? 0 : 1
    terms.push({ name: c.name, df, seqSS, adjSS, adjMS: df ? adjSS : NaN, f: df ? adjSS / mse : NaN, pValue: df ? fDist(1, dfe).sf(adjSS / mse) : NaN })
  }
  const fStat = dfr > 0 ? ssr / dfr / mse : NaN
  const anova = {
    regression: { source: 'Regression', df: dfr, ss: ssr, ms: dfr > 0 ? ssr / dfr : NaN, f: fStat, pValue: dfr > 0 ? fDist(dfr, dfe).sf(fStat) : NaN },
    error: { source: 'Error', df: dfe, ss: sse, ms: mse },
    total: { source: 'Total', df: dft, ss: sst, ms: NaN },
  }
  let dwNum = 0
  for (let i = 1; i < n; i++) dwNum += (fit.residuals[i]! - fit.residuals[i - 1]!) ** 2
  const durbinWatson = dwNum / sse
  const unusual: OlsResult['unusual'] = []
  const levCut = (3 * p) / n
  for (let i = 0; i < n; i++) {
    let flags = ''
    if (Math.abs(std[i]!) > 2) flags += 'R'
    if (lev[i]! > levCut || lev[i]! > 0.99) flags += 'X'
    if (flags) unusual.push({ index: keep[i]!, y: yv[i]!, fitted: fitted[i]!, residual: residuals[i]!, stdResidual: std[i]!, leverage: lev[i]!, flags })
  }
  // Gaussian log-likelihood with MLE variance
  const logLik = -0.5 * n * (Math.log((2 * Math.PI * sse) / n) + 1)
  const k = p + 1
  const aic = -2 * logLik + 2 * k
  const aicc = n - k - 1 > 0 ? aic + (2 * k * (k + 1)) / (n - k - 1) : Infinity
  const bic = -2 * logLik + k * Math.log(n)

  const predict = (x: ArrayLike<number> | ArrayLike<number>[], o: { confidence?: number } = {}): Prediction[] => {
    const conf = o.confidence ?? confidence
    const tc = td.ppf(0.5 + conf / 2)
    const rows: ArrayLike<number>[] = typeof (x as ArrayLike<number>)[0] === 'number' || x.length === 0 ? [x as ArrayLike<number>] : (x as ArrayLike<number>[])
    return rows.map((row) => {
      if (row.length !== xnames.length) throw new RangeError(`predict: expected ${xnames.length} predictor values, got ${row.length}`)
      const v = new Float64Array(pAll)
      let j = 0
      if (intercept) v[j++] = 1
      for (let c = 0; c < row.length; c++) v[j++] = row[c]!
      let fitv = 0
      for (let c = 0; c < pAll; c++) fitv += v[c]! * fit.coef[c]!
      let q = 0
      for (let a = 0; a < pAll; a++) for (let b = 0; b < pAll; b++) q += v[a]! * covariance.data[a * pAll + b]! * v[b]!
      const se = Math.sqrt(Math.max(0, q))
      const sePred = Math.sqrt(se * se + mse)
      return { fit: fitv, se, ci: [fitv - tc * se, fitv + tc * se], pi: [fitv - tc * sePred, fitv + tc * sePred] }
    })
  }
  return {
    test: 'regression',
    n,
    p,
    coefficients,
    names: xnames,
    intercept,
    s,
    r2,
    r2adj,
    r2pred,
    press,
    logLik,
    aic,
    aicc,
    bic,
    anova,
    terms,
    fitted,
    residuals,
    standardizedResiduals: std,
    studentizedResiduals: stud,
    leverage: lev,
    cooksD: cook,
    dffits,
    durbinWatson,
    unusual,
    omitted,
    confidence,
    predict,
    covariance,
    design,
    y: yv,
  }
}

/**
 * Fitted Line Plot (Minitab): polynomial regression of y on x of degree 1–3, with the fitted curve
 * as a function. `transform` applies log10 to x and/or y before fitting (Minitab's options).
 */
export function fittedLine(
  x: Column,
  y: Column,
  options: { degree?: 1 | 2 | 3; logX?: boolean; logY?: boolean; confidence?: number } = {},
): OlsResult & { degree: number; curve(x: number): number; equation: string } {
  const degree = options.degree ?? 1
  if (![1, 2, 3].includes(degree)) throw new RangeError('fittedLine: degree must be 1, 2 or 3')
  const tx = (v: number) => (options.logX ? Math.log10(v) : v)
  const ty = (v: number) => (options.logY ? Math.log10(v) : v)
  const xs = Array.from(x, (v) => (isNum(v) && (!options.logX || v > 0) ? tx(v) : null))
  const ys = Array.from(y, (v) => (isNum(v) && (!options.logY || v > 0) ? ty(v) : null))
  const cols: Column[] = []
  const names: string[] = []
  const base = options.logX ? 'log10(x)' : 'x'
  for (let d = 1; d <= degree; d++) {
    cols.push(xs.map((v) => (v === null ? null : v ** d)))
    names.push(d === 1 ? base : `${base}^${d}`)
  }
  const r = ols(ys, cols, { names, confidence: options.confidence })
  const b = r.coefficients.map((c) => c.coef)
  const curve = (v: number) => {
    const u = tx(v)
    let s = b[0]!
    for (let d = 1; d <= degree; d++) s += b[d]! * u ** d
    return options.logY ? 10 ** s : s
  }
  const fmt = (v: number) => (Math.abs(v) < 1e-4 || Math.abs(v) >= 1e6 ? v.toExponential(4) : v.toFixed(4))
  let equation = `${options.logY ? 'log10(y)' : 'y'} = ${fmt(b[0]!)}`
  for (let d = 1; d <= degree; d++) equation += ` ${b[d]! < 0 ? '−' : '+'} ${fmt(Math.abs(b[d]!))} ${names[d - 1]}`
  return { ...r, degree, curve, equation }
}
