/**
 * Tier 5.1 — Time Series (Minitab Stat › Time Series):
 * Trend Analysis, classical Decomposition, SES / DES / Holt–Winters (optimised α,β,γ),
 * ACF / PACF / CCF with Bartlett bands, Ljung–Box, ARIMA via CSS (conditional sum of squares).
 */
import { chi2 as chi2Dist, normal } from './dist.js'
import { cleanNumbers } from './tests.js'
import { lstsq, matrix } from './linalg.js'
import { maxOf, minOf } from './numerics.js'

const STD = normal()

function meanOf(v: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i]!
  return s / v.length
}
function sseOf(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += (a[i]! - b[i]!) ** 2
  return s
}
function mapeOf(actual: ArrayLike<number>, fitted: ArrayLike<number>): number {
  let s = 0
  let n = 0
  for (let i = 0; i < actual.length; i++) {
    if (actual[i]! !== 0) {
      s += Math.abs((actual[i]! - fitted[i]!) / actual[i]!)
      n++
    }
  }
  return n ? (100 * s) / n : NaN
}

// ---- Trend Analysis -------------------------------------------------------------------------------

export type TrendModel = 'linear' | 'quadratic' | 'exponential' | 's-curve'

export interface TrendResult {
  model: TrendModel
  /** Fitted values over the observed times 1…n. */
  fitted: number[]
  /** Forecasts for the next `horizon` periods. */
  forecast: number[]
  parameters: Record<string, number>
  equation: string
  mape: number
  mad: number
  msd: number
  r2: number
}

/** Trend Analysis: linear / quadratic / exponential / Pearl–Reed S-curve on yₜ vs t = 1…n. */
export function trendAnalysis(
  y: ArrayLike<number | null | undefined>,
  options: { model?: TrendModel; horizon?: number } = {},
): TrendResult {
  const v = Array.from(cleanNumbers(y))
  const n = v.length
  if (n < 3) throw new RangeError('trendAnalysis needs at least 3 observations')
  const model = options.model ?? 'linear'
  const horizon = options.horizon ?? 0
  const t = Array.from({ length: n }, (_, i) => i + 1)
  const sst = (() => {
    const m = meanOf(v)
    let s = 0
    for (const yi of v) s += (yi - m) ** 2
    return s
  })()

  const fitAt = (tt: number, params: number[]): number => {
    if (model === 'linear') return params[0]! + params[1]! * tt
    if (model === 'quadratic') return params[0]! + params[1]! * tt + params[2]! * tt * tt
    if (model === 'exponential') return params[0]! * Math.exp(params[1]! * tt)
    // S-curve (logistic / Pearl–Reed): Ŷ = 10^a / (b + t^c)  — Minitab form; use Ŷ = a / (1 + b · exp(−c t))
    return params[0]! / (1 + params[1]! * Math.exp(-params[2]! * tt))
  }

  let params: number[]
  let equation: string
  if (model === 'linear') {
    const X = matrix(n, 2)
    for (let i = 0; i < n; i++) {
      X.data[i * 2] = 1
      X.data[i * 2 + 1] = t[i]!
    }
    const { coef } = lstsq(X, v)
    params = [coef[0]!, coef[1]!]
    equation = `Yt = ${params[0]!.toPrecision(6)} + ${params[1]!.toPrecision(6)}·t`
  } else if (model === 'quadratic') {
    const X = matrix(n, 3)
    for (let i = 0; i < n; i++) {
      X.data[i * 3] = 1
      X.data[i * 3 + 1] = t[i]!
      X.data[i * 3 + 2] = t[i]! * t[i]!
    }
    const { coef } = lstsq(X, v)
    params = [coef[0]!, coef[1]!, coef[2]!]
    equation = `Yt = ${params[0]!.toPrecision(6)} + ${params[1]!.toPrecision(6)}·t + ${params[2]!.toPrecision(6)}·t²`
  } else if (model === 'exponential') {
    // log-linear: ln y = ln a + b t  (requires y > 0)
    if (v.some((yi) => !(yi > 0))) throw new RangeError('trendAnalysis exponential: all y must be > 0')
    const ly = v.map(Math.log)
    const X = matrix(n, 2)
    for (let i = 0; i < n; i++) {
      X.data[i * 2] = 1
      X.data[i * 2 + 1] = t[i]!
    }
    const { coef } = lstsq(X, ly)
    params = [Math.exp(coef[0]!), coef[1]!]
    equation = `Yt = ${params[0]!.toPrecision(6)} · exp(${params[1]!.toPrecision(6)}·t)`
  } else {
    // S-curve via nls-ish: start from linearised logit of scaled y
    const ymin = minOf(v)
    const ymax = maxOf(v)
    const a0 = ymax * 1.1 || 1
    const scaled = v.map((yi) => Math.min(0.99, Math.max(0.01, yi / a0)))
    const z = scaled.map((p) => Math.log(1 / p - 1))
    const X = matrix(n, 2)
    for (let i = 0; i < n; i++) {
      X.data[i * 2] = 1
      X.data[i * 2 + 1] = -t[i]!
    }
    const { coef } = lstsq(X, z)
    // z ≈ ln b − c t  → b = exp(coef0), c = coef1
    let a = a0
    let b = Math.exp(coef[0]!)
    let c = coef[1]!
    // refine a,b,c by a few Gauss–Newton steps
    for (let iter = 0; iter < 30; iter++) {
      const J = matrix(n, 3)
      const r = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        const tt = t[i]!
        const den = 1 + b * Math.exp(-c * tt)
        const fit = a / den
        r[i] = v[i]! - fit
        const e = Math.exp(-c * tt)
        J.data[i * 3] = 1 / den
        J.data[i * 3 + 1] = (-a * e) / (den * den)
        J.data[i * 3 + 2] = (a * b * tt * e) / (den * den)
      }
      const step = lstsq(J, r).coef
      a += step[0]!
      b += step[1]!
      c += step[2]!
      if (step.every((s) => Math.abs(s) < 1e-10)) break
    }
    params = [a, b, c]
    equation = `Yt = ${a.toPrecision(6)} / (1 + ${b.toPrecision(6)}·exp(−${c.toPrecision(6)}·t))`
    void ymin
  }

  const fitted = t.map((tt) => fitAt(tt, params))
  const forecast = Array.from({ length: horizon }, (_, i) => fitAt(n + i + 1, params))
  const sse = sseOf(v, fitted)
  let mad = 0
  for (let i = 0; i < n; i++) mad += Math.abs(v[i]! - fitted[i]!)
  mad /= n
  const names: Record<string, number> =
    model === 'linear'
      ? { intercept: params[0]!, slope: params[1]! }
      : model === 'quadratic'
        ? { intercept: params[0]!, linear: params[1]!, quadratic: params[2]! }
        : model === 'exponential'
          ? { a: params[0]!, b: params[1]! }
          : { a: params[0]!, b: params[1]!, c: params[2]! }
  return {
    model,
    fitted,
    forecast,
    parameters: names,
    equation,
    mape: mapeOf(v, fitted),
    mad,
    msd: sse / n,
    r2: sst > 0 ? 1 - sse / sst : NaN,
  }
}

// ---- Classical decomposition ----------------------------------------------------------------------

export interface DecompositionResult {
  method: 'additive' | 'multiplicative'
  seasonLength: number
  trend: number[]
  seasonal: number[]
  residual: number[]
  /** Seasonal indices (length = seasonLength), mean-centred (add) or mean-1 (mult). */
  seasonalIndices: number[]
}

/** Classical seasonal decomposition with centred moving average trend. */
export function decompose(
  y: ArrayLike<number | null | undefined>,
  options: { seasonLength: number; method?: 'additive' | 'multiplicative' },
): DecompositionResult {
  const v = Array.from(cleanNumbers(y))
  const n = v.length
  const s = options.seasonLength
  if (!(s >= 2 && Number.isInteger(s))) throw new RangeError('decompose: seasonLength must be an integer ≥ 2')
  if (n < 2 * s) throw new RangeError('decompose: need at least 2 full seasons')
  const method = options.method ?? 'additive'
  if (method === 'multiplicative' && v.some((yi) => !(yi > 0))) throw new RangeError('decompose multiplicative: y must be > 0')

  // centred MA of length s
  const trend = new Array<number>(n).fill(NaN)
  const half = Math.floor(s / 2)
  for (let i = half; i < n - half + (s % 2 === 0 ? 0 : 1); i++) {
    if (s % 2 === 1) {
      let sum = 0
      for (let k = -half; k <= half; k++) sum += v[i + k]!
      trend[i] = sum / s
    } else {
      // even: average of two MA(s)
      let s1 = 0
      let s2 = 0
      for (let k = -half; k < half; k++) s1 += v[i + k]!
      for (let k = -half + 1; k <= half; k++) s2 += v[i + k]!
      trend[i] = 0.5 * (s1 / s + s2 / s)
    }
  }

  const det = v.map((yi, i) => (Number.isFinite(trend[i]!) ? (method === 'additive' ? yi - trend[i]! : yi / trend[i]!) : NaN))
  const seasonalIndices = new Array(s).fill(0)
  const counts = new Array(s).fill(0)
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(det[i]!)) continue
    const k = i % s
    seasonalIndices[k] += det[i]!
    counts[k]++
  }
  for (let k = 0; k < s; k++) seasonalIndices[k] /= counts[k] || 1
  // normalise
  if (method === 'additive') {
    const m = meanOf(seasonalIndices)
    for (let k = 0; k < s; k++) seasonalIndices[k] -= m
  } else {
    const m = meanOf(seasonalIndices)
    for (let k = 0; k < s; k++) seasonalIndices[k] /= m || 1
  }
  const seasonal = Array.from({ length: n }, (_, i) => seasonalIndices[i % s]!)
  const residual = v.map((yi, i) => (method === 'additive' ? yi - (trend[i]! || 0) - seasonal[i]! : yi / ((trend[i]! || 1) * seasonal[i]!)))
  return { method, seasonLength: s, trend, seasonal, residual, seasonalIndices }
}

// ---- STL (Cleveland LOESS) ------------------------------------------------------------------------

export interface StlResult {
  seasonLength: number
  /** When multi-season MVP used, the list of periods. */
  periods?: number[]
  /** Per-period seasonal components when `periods` has length > 1. */
  seasonals?: number[][]
  trend: number[]
  seasonal: number[]
  residual: number[]
  robust: boolean
}

/** Locally weighted regression (LOESS) at integer positions 0..n-1. */
function loessSmooth(
  y: number[],
  span: number,
  degree: 0 | 1 = 1,
  weights?: number[],
): number[] {
  const n = y.length
  const q = Math.max(degree + 1, Math.min(n, Math.ceil(span * n)))
  const out = new Array(n)
  for (let i = 0; i < n; i++) {
    // q nearest neighbours on an equally spaced grid form the window [lo, hi]: expand from i, taking the
    // closer side first (left on ties, as a stable sort by distance would) — O(q) instead of an O(n log n) sort
    let lo = i
    let hi = i
    let count = 1
    while (count < q) {
      const dl = lo > 0 ? i - lo + 1 : Infinity
      const dr = hi < n - 1 ? hi + 1 - i : Infinity
      if (dl <= dr) lo--
      else hi++
      count++
    }
    const dmax = Math.max(i - lo, hi - i) || 1
    const ww: number[] = []
    const xx: number[] = []
    const yy: number[] = []
    for (let j = lo; j <= hi; j++) {
      const d = Math.abs(j - i)
      const u = d / dmax
      const tricube = u < 1 ? (1 - u * u * u) ** 3 : 0
      const rw = weights ? weights[j]! : 1
      const w = Math.max(tricube * rw, 0)
      if (w <= 0 && degree === 1) continue
      ww.push(w)
      xx.push(j)
      yy.push(y[j]!)
    }
    if (xx.length < degree + 1) {
      // fallback mean
      let s = 0
      let c = 0
      for (let k = 0; k < ww.length; k++) {
        s += ww[k]! * yy[k]!
        c += ww[k]!
      }
      out[i] = c > 0 ? s / c : y[i]!
      continue
    }
    if (degree === 0) {
      let s = 0
      let c = 0
      for (let k = 0; k < ww.length; k++) {
        s += ww[k]! * yy[k]!
        c += ww[k]!
      }
      out[i] = s / c
    } else {
      // weighted linear: [1, x] β
      let S0 = 0
      let S1 = 0
      let S2 = 0
      let T0 = 0
      let T1 = 0
      for (let k = 0; k < ww.length; k++) {
        const w = ww[k]!
        const x = xx[k]!
        const yi = yy[k]!
        S0 += w
        S1 += w * x
        S2 += w * x * x
        T0 += w * yi
        T1 += w * x * yi
      }
      const det = S0 * S2 - S1 * S1
      if (Math.abs(det) < 1e-14) {
        out[i] = T0 / S0
      } else {
        const b0 = (S2 * T0 - S1 * T1) / det
        const b1 = (S0 * T1 - S1 * T0) / det
        out[i] = b0 + b1 * i
      }
    }
  }
  return out
}

function movingAverage(y: number[], len: number): number[] {
  const n = y.length
  const out = new Array(n).fill(NaN)
  if (len < 1) return out
  let sum = 0
  for (let i = 0; i < n; i++) {
    sum += y[i]!
    if (i >= len) sum -= y[i - len]!
    if (i >= len - 1) out[i - Math.floor((len - 1) / 2)] = sum / len
  }
  // fill edges by nearest
  let first = 0
  while (first < n && !Number.isFinite(out[first]!)) first++
  let last = n - 1
  while (last >= 0 && !Number.isFinite(out[last]!)) last--
  for (let i = 0; i < first; i++) out[i] = out[first]!
  for (let i = last + 1; i < n; i++) out[i] = out[last]!
  for (let i = 0; i < n; i++) if (!Number.isFinite(out[i]!)) out[i] = y[i]!
  return out
}

/**
 * Cleveland STL: Seasonal-Trend decomposition using LOESS.
 */
export function stl(
  y: ArrayLike<number | null | undefined>,
  options: {
    seasonLength?: number
    /** Multi-season STL MVP: extract each period sequentially and sum seasonals. */
    periods?: number[]
    seasonalSpan?: number
    trendSpan?: number
    robust?: boolean
    inner?: number
    outer?: number
  },
): StlResult {
  const periods =
    options.periods && options.periods.length > 0
      ? options.periods
      : options.seasonLength != null
        ? [options.seasonLength]
        : null
  if (!periods) throw new RangeError('stl: provide seasonLength or periods')
  if (periods.length === 1) {
    return stlOnce(y, { ...options, seasonLength: periods[0]! })
  }
  // Sequential multi-season: peel each seasonal, keep last trend
  let rem = Array.from(cleanNumbers(y))
  const seasonals: number[][] = []
  let last: StlResult | null = null
  for (const s of periods) {
    last = stlOnce(rem, { ...options, seasonLength: s })
    seasonals.push(last.seasonal)
    rem = rem.map((v, i) => v - last!.seasonal[i]!)
  }
  const seasonal = seasonals[0]!.map((_, i) => seasonals.reduce((acc, c) => acc + c[i]!, 0))
  const trend = last!.trend
  const residual = Array.from(cleanNumbers(y)).map((v, i) => v - trend[i]! - seasonal[i]!)
  return {
    seasonLength: periods[0]!,
    periods: [...periods],
    seasonals,
    trend,
    seasonal,
    residual,
    robust: options.robust ?? false,
  }
}

function stlOnce(
  y: ArrayLike<number | null | undefined>,
  options: {
    seasonLength: number
    seasonalSpan?: number
    trendSpan?: number
    robust?: boolean
    inner?: number
    outer?: number
  },
): StlResult {
  const v = Array.from(cleanNumbers(y))
  const n = v.length
  const s = options.seasonLength
  if (!(s >= 2 && Number.isInteger(s))) throw new RangeError('stl: seasonLength must be an integer ≥ 2')
  if (n < 2 * s) throw new RangeError('stl: need at least 2 full seasons')
  const robust = options.robust ?? false
  const nInner = options.inner ?? (robust ? 1 : 2)
  const nOuter = options.outer ?? (robust ? 5 : 0)
  // Default spans (Cleveland-style heuristics)
  const seasonalSpan = options.seasonalSpan ?? Math.min(0.9, Math.max(0.5, 5 / Math.ceil(n / s)))
  const trendSpan = options.trendSpan ?? Math.min(0.75, Math.max(0.2, (1.5 * s) / n))

  const seasonal = new Array(n).fill(0)
  let trend = new Array(n).fill(0)
  let residual = v.slice()
  const weights = new Array(n).fill(1)

  const runInner = () => {
    // Detrend
    const detrended = v.map((yi, i) => yi - trend[i]!)
    // Seasonal subseries LOESS
    const C = new Array(n).fill(0)
    for (let j = 0; j < s; j++) {
      const idx: number[] = []
      const vals: number[] = []
      const wts: number[] = []
      for (let i = j; i < n; i += s) {
        idx.push(i)
        vals.push(detrended[i]!)
        wts.push(weights[i]!)
      }
      if (vals.length < 2) {
        for (const i of idx) C[i] = vals[0] ?? 0
        continue
      }
      // smooth on cycle index 0..m-1 then map back
      const sm = loessSmooth(vals, Math.min(1, Math.max(seasonalSpan, 3 / vals.length)), 1, wts)
      for (let k = 0; k < idx.length; k++) C[idx[k]!] = sm[k]!
    }
    // Low-pass of C: MA of length s, then s, then 3, then LOESS
    let low = movingAverage(C, s)
    low = movingAverage(low, s)
    low = movingAverage(low, 3)
    low = loessSmooth(low, Math.min(1, Math.max(trendSpan, 0.2)), 1, weights)
    for (let i = 0; i < n; i++) seasonal[i] = C[i]! - low[i]!
    // Move period means into trend later: centre each full season block so Σ_season ≈ 0
    for (let start = 0; start + s <= n; start += s) {
      let m = 0
      for (let j = 0; j < s; j++) m += seasonal[start + j]!
      m /= s
      for (let j = 0; j < s; j++) seasonal[start + j]! -= m
    }
    // leftover partial period: centre against global seasonal mean of complete indices
    if (n % s !== 0) {
      const remStart = n - (n % s)
      let m = 0
      let c = 0
      for (let i = remStart; i < n; i++) {
        m += seasonal[i]!
        c++
      }
      m /= c || 1
      for (let i = remStart; i < n; i++) seasonal[i]! -= m
    }
    // Deseasonalize and trend
    const deseas = v.map((yi, i) => yi - seasonal[i]!)
    trend = loessSmooth(deseas, Math.min(1, Math.max(trendSpan, 0.15)), 1, weights)
    residual = v.map((yi, i) => yi - trend[i]! - seasonal[i]!)
  }

  // Initial trend = 0
  for (let outer = 0; outer <= nOuter; outer++) {
    for (let inner = 0; inner < nInner; inner++) runInner()
    if (!robust || outer === nOuter) break
    // Robustness weights from remainder
    const abs = residual.map(Math.abs)
    const sorted = abs.slice().sort((a, b) => a - b)
    const med = sorted[Math.floor(sorted.length / 2)]! || 1
    const h = 6 * med || 1
    for (let i = 0; i < n; i++) {
      const u = Math.abs(residual[i]!) / h
      weights[i] = u < 1 ? (1 - u * u) ** 2 : 0
    }
  }

  return { seasonLength: s, trend, seasonal, residual, robust }
}

// ---- Exponential smoothing ------------------------------------------------------------------------

export interface EtsResult {
  method: 'ses' | 'des' | 'winters-add' | 'winters-mul'
  alpha: number
  beta?: number
  gamma?: number
  level: number[]
  trend?: number[]
  seasonal?: number[]
  fitted: number[]
  forecast: number[]
  /** Approximate prediction intervals (Hyndman-style). */
  forecastLower?: number[]
  forecastUpper?: number[]
  confidence?: number
  sigma?: number
  sse: number
  mape: number
}

function sesFit(y: number[], alpha: number): { fitted: number[]; level: number[]; sse: number; last: number } {
  const n = y.length
  const level = new Array(n)
  const fitted = new Array(n)
  level[0] = y[0]!
  fitted[0] = y[0]!
  let sse = 0
  for (let t = 1; t < n; t++) {
    fitted[t] = level[t - 1]!
    sse += (y[t]! - fitted[t]!) ** 2
    level[t] = alpha * y[t]! + (1 - alpha) * level[t - 1]!
  }
  return { fitted, level, sse, last: level[n - 1]! }
}

function desFit(y: number[], alpha: number, beta: number): { fitted: number[]; level: number[]; trend: number[]; sse: number } {
  const n = y.length
  const level = new Array(n)
  const trend = new Array(n)
  const fitted = new Array(n)
  level[0] = y[0]!
  trend[0] = y[1]! - y[0]!
  fitted[0] = y[0]!
  let sse = 0
  for (let t = 1; t < n; t++) {
    fitted[t] = level[t - 1]! + trend[t - 1]!
    sse += (y[t]! - fitted[t]!) ** 2
    level[t] = alpha * y[t]! + (1 - alpha) * (level[t - 1]! + trend[t - 1]!)
    trend[t] = beta * (level[t]! - level[t - 1]!) + (1 - beta) * trend[t - 1]!
  }
  return { fitted, level, trend, sse }
}

function wintersFit(
  y: number[],
  alpha: number,
  beta: number,
  gamma: number,
  s: number,
  multiplicative: boolean,
): { fitted: number[]; level: number[]; trend: number[]; seasonal: number[]; sse: number } {
  const n = y.length
  const level = new Array(n).fill(0)
  const trend = new Array(n).fill(0)
  const seasonal = new Array(n).fill(0)
  const fitted = new Array(n).fill(0)
  // init seasonal from first season ratios / diffs
  const mean0 = meanOf(y.slice(0, s))
  for (let i = 0; i < s; i++) seasonal[i] = multiplicative ? y[i]! / mean0 : y[i]! - mean0
  level[s - 1] = mean0
  trend[s - 1] = (meanOf(y.slice(s, 2 * s)) - mean0) / s
  let sse = 0
  for (let t = 0; t < s; t++) fitted[t] = y[t]!
  for (let t = s; t < n; t++) {
    const sLag = seasonal[t - s]!
    fitted[t] = multiplicative ? (level[t - 1]! + trend[t - 1]!) * sLag : level[t - 1]! + trend[t - 1]! + sLag
    sse += (y[t]! - fitted[t]!) ** 2
    if (multiplicative) {
      level[t] = alpha * (y[t]! / sLag) + (1 - alpha) * (level[t - 1]! + trend[t - 1]!)
      trend[t] = beta * (level[t]! - level[t - 1]!) + (1 - beta) * trend[t - 1]!
      seasonal[t] = gamma * (y[t]! / level[t]!) + (1 - gamma) * sLag
    } else {
      level[t] = alpha * (y[t]! - sLag) + (1 - alpha) * (level[t - 1]! + trend[t - 1]!)
      trend[t] = beta * (level[t]! - level[t - 1]!) + (1 - beta) * trend[t - 1]!
      seasonal[t] = gamma * (y[t]! - level[t]!) + (1 - gamma) * sLag
    }
  }
  return { fitted, level, trend, seasonal, sse }
}

function gridOpt(fn: (p: number[]) => number, lows: number[], highs: number[], steps = 8): number[] {
  const dim = lows.length
  let best = lows.slice()
  let bestVal = Infinity
  const recurse = (d: number, cur: number[]) => {
    if (d === dim) {
      const v = fn(cur)
      if (v < bestVal) {
        bestVal = v
        best = cur.slice()
      }
      return
    }
    for (let i = 0; i <= steps; i++) {
      cur[d] = lows[d]! + ((highs[d]! - lows[d]!) * i) / steps
      recurse(d + 1, cur)
    }
  }
  recurse(0, new Array(dim))
  // local refine
  for (let pass = 0; pass < 3; pass++) {
    for (let d = 0; d < dim; d++) {
      const span = (highs[d]! - lows[d]!) / (steps * (pass + 2))
      for (const delta of [-span, 0, span]) {
        const trial = best.slice()
        trial[d] = Math.min(highs[d]!, Math.max(lows[d]!, best[d]! + delta))
        const v = fn(trial)
        if (v < bestVal) {
          bestVal = v
          best = trial
        }
      }
    }
  }
  return best
}

/**
 * Exponential smoothing: SES, Holt DES, or Holt–Winters (additive / multiplicative).
 * Smoothing parameters are optimised by SSE grid search unless supplied.
 * Forecast intervals use Hyndman approximations (σ from in-sample one-step SSE).
 */
export function ets(
  y: ArrayLike<number | null | undefined>,
  options: {
    method?: 'ses' | 'des' | 'winters-add' | 'winters-mul'
    seasonLength?: number
    alpha?: number
    beta?: number
    gamma?: number
    horizon?: number
    confidence?: number
  } = {},
): EtsResult {
  const v = Array.from(cleanNumbers(y))
  if (v.length < 3) throw new RangeError('ets needs at least 3 observations')
  const method = options.method ?? 'ses'
  const horizon = options.horizon ?? 0
  const confidence = options.confidence ?? 0.95
  const z = STD.ppf(0.5 + confidence / 2)

  const withPi = (forecast: number[], sigma: number, varFactor: (h: number) => number, base: Omit<EtsResult, 'forecast' | 'forecastLower' | 'forecastUpper' | 'confidence' | 'sigma' | 'sse' | 'mape'> & { sse: number; mape: number }): EtsResult => {
    const forecastLower = forecast.map((f, h) => f - z * sigma * Math.sqrt(varFactor(h + 1)))
    const forecastUpper = forecast.map((f, h) => f + z * sigma * Math.sqrt(varFactor(h + 1)))
    return { ...base, forecast, forecastLower, forecastUpper, confidence, sigma, sse: base.sse, mape: base.mape }
  }

  if (method === 'ses') {
    const alpha = options.alpha ?? gridOpt((p) => sesFit(v, p[0]!).sse, [0.01], [0.99])[0]!
    const fit = sesFit(v, alpha)
    const nEff = Math.max(1, v.length - 1)
    const sigma = Math.sqrt(fit.sse / nEff)
    const forecast = Array.from({ length: horizon }, () => fit.last)
    // SES: Var(e_{n+h}) = σ² (1 + (h−1) α²)
    return withPi(forecast, sigma, (h) => 1 + (h - 1) * alpha * alpha, {
      method,
      alpha,
      level: fit.level,
      fitted: fit.fitted,
      sse: fit.sse,
      mape: mapeOf(v, fit.fitted),
    })
  }
  if (method === 'des') {
    const [alpha, beta] =
      options.alpha !== undefined && options.beta !== undefined
        ? [options.alpha, options.beta]
        : (gridOpt((p) => desFit(v, p[0]!, p[1]!).sse, [0.01, 0.01], [0.99, 0.99]) as [number, number])
    const fit = desFit(v, alpha, beta)
    const n = v.length
    const nEff = Math.max(1, n - 2)
    const sigma = Math.sqrt(fit.sse / nEff)
    const forecast = Array.from({ length: horizon }, (_, h) => fit.level[n - 1]! + (h + 1) * fit.trend[n - 1]!)
    // Holt: approximate Var ≈ σ² (1 + (h−1)(α² + αβh + …)) — use Hyndman additive trend form
    return withPi(
      forecast,
      sigma,
      (h) => {
        const c = alpha + 0.5 * (h - 1) * beta
        return 1 + (h - 1) * c * c
      },
      { method, alpha, beta, level: fit.level, trend: fit.trend, fitted: fit.fitted, sse: fit.sse, mape: mapeOf(v, fit.fitted) },
    )
  }
  const s = options.seasonLength
  if (!(s && s >= 2)) throw new RangeError('ets winters: provide seasonLength ≥ 2')
  const mul = method === 'winters-mul'
  const [alpha, beta, gamma] =
    options.alpha !== undefined && options.beta !== undefined && options.gamma !== undefined
      ? [options.alpha, options.beta, options.gamma]
      : gridOpt((p) => wintersFit(v, p[0]!, p[1]!, p[2]!, s, mul).sse, [0.01, 0.01, 0.01], [0.99, 0.99, 0.99], 5)
  const fit = wintersFit(v, alpha, beta, gamma, s, mul)
  const n = v.length
  const nEff = Math.max(1, n - s)
  const sigma = Math.sqrt(fit.sse / nEff)
  const forecast = Array.from({ length: horizon }, (_, h) => {
    const seas = fit.seasonal[n - s + (h % s)]!
    return mul ? (fit.level[n - 1]! + (h + 1) * fit.trend[n - 1]!) * seas : fit.level[n - 1]! + (h + 1) * fit.trend[n - 1]! + seas
  })
  return withPi(
    forecast,
    sigma,
    (h) => {
      const k = Math.floor((h - 1) / s)
      const c = alpha + 0.5 * (h - 1) * beta
      return 1 + (h - 1) * c * c + k * gamma * gamma * (1 - alpha) * (1 - alpha)
    },
    {
      method,
      alpha,
      beta,
      gamma,
      level: fit.level,
      trend: fit.trend,
      seasonal: fit.seasonal,
      fitted: fit.fitted,
      sse: fit.sse,
      mape: mapeOf(v.slice(s), fit.fitted.slice(s)),
    },
  )
}

// ---- ACF / PACF / CCF / Ljung–Box -----------------------------------------------------------------

export interface AcfResult {
  lag: number[]
  acf: number[]
  /** Bartlett ±z · √((1+2Σρ²)/n) for lag > 0; lag 0 band = 0. */
  lower: number[]
  upper: number[]
  n: number
}

/** Sample ACF with Bartlett confidence bands (default 95 %). */
export function acf(y: ArrayLike<number | null | undefined>, options: { maxLag?: number; confidence?: number } = {}): AcfResult {
  const v = Array.from(cleanNumbers(y))
  const n = v.length
  if (n < 3) throw new RangeError('acf needs at least 3 observations')
  const maxLag = options.maxLag ?? Math.min(40, Math.floor(n / 2))
  const conf = options.confidence ?? 0.95
  const z = STD.ppf(0.5 + conf / 2)
  const m = meanOf(v)
  let c0 = 0
  for (const yi of v) c0 += (yi - m) ** 2
  const lag: number[] = []
  const rho: number[] = []
  for (let k = 0; k <= maxLag; k++) {
    let ck = 0
    for (let t = k; t < n; t++) ck += (v[t]! - m) * (v[t - k]! - m)
    lag.push(k)
    rho.push(ck / c0)
  }
  const lower: number[] = [0]
  const upper: number[] = [0]
  let sumSq = 0
  for (let k = 1; k <= maxLag; k++) {
    sumSq += rho[k]! ** 2
    const se = Math.sqrt((1 + 2 * sumSq) / n) // Bartlett; for white noise ≈ 1/√n after lag 0
    // Minitab / statsmodels use √(1/n) for the band under white-noise null by default for plot
    const sePlot = Math.sqrt(1 / n)
    void se
    lower.push(-z * sePlot)
    upper.push(z * sePlot)
  }
  return { lag, acf: rho, lower, upper, n }
}

export interface PacfResult {
  lag: number[]
  pacf: number[]
  lower: number[]
  upper: number[]
  n: number
}

/** PACF via Durbin–Levinson; bands ±z/√n. */
export function pacf(y: ArrayLike<number | null | undefined>, options: { maxLag?: number; confidence?: number } = {}): PacfResult {
  const a = acf(y, options)
  const maxLag = a.lag[a.lag.length - 1]!
  const phi: number[] = [1] // φ_{00}
  const pacfVals = [1]
  const phiPrev: number[] = []
  for (let k = 1; k <= maxLag; k++) {
    let num = a.acf[k]!
    for (let j = 1; j < k; j++) num -= phiPrev[j - 1]! * a.acf[k - j]!
    let den = 1
    for (let j = 1; j < k; j++) den -= phiPrev[j - 1]! * a.acf[j]!
    const phikk = num / den
    const phiNew = new Array(k)
    for (let j = 1; j < k; j++) phiNew[j - 1] = phiPrev[j - 1]! - phikk * phiPrev[k - j - 1]!
    phiNew[k - 1] = phikk
    phiPrev.length = 0
    phiPrev.push(...phiNew)
    pacfVals.push(phikk)
    void phi
  }
  const z = STD.ppf(0.5 + (options.confidence ?? 0.95) / 2)
  const se = z / Math.sqrt(a.n)
  return {
    lag: a.lag,
    pacf: pacfVals,
    lower: a.lag.map((k) => (k === 0 ? 0 : -se)),
    upper: a.lag.map((k) => (k === 0 ? 0 : se)),
    n: a.n,
  }
}

export interface CcfResult {
  lag: number[]
  ccf: number[]
  lower: number[]
  upper: number[]
  n: number
}

/** Cross-correlation function of x and y for lags −maxLag…+maxLag. */
export function ccf(
  x: ArrayLike<number | null | undefined>,
  y: ArrayLike<number | null | undefined>,
  options: { maxLag?: number; confidence?: number } = {},
): CcfResult {
  const xs = Array.from(cleanNumbers(x))
  const ys = Array.from(cleanNumbers(y))
  const n = Math.min(xs.length, ys.length)
  if (n < 3) throw new RangeError('ccf needs at least 3 paired observations')
  const maxLag = options.maxLag ?? Math.min(20, Math.floor(n / 3))
  const mx = meanOf(xs.slice(0, n))
  const my = meanOf(ys.slice(0, n))
  let sxx = 0
  let syy = 0
  for (let i = 0; i < n; i++) {
    sxx += (xs[i]! - mx) ** 2
    syy += (ys[i]! - my) ** 2
  }
  const denom = Math.sqrt(sxx * syy)
  const lag: number[] = []
  const rho: number[] = []
  for (let k = -maxLag; k <= maxLag; k++) {
    let s = 0
    if (k >= 0) for (let t = k; t < n; t++) s += (xs[t]! - mx) * (ys[t - k]! - my)
    else for (let t = -k; t < n; t++) s += (xs[t + k]! - mx) * (ys[t]! - my)
    lag.push(k)
    rho.push(s / denom)
  }
  const z = STD.ppf(0.5 + (options.confidence ?? 0.95) / 2)
  const se = z / Math.sqrt(n)
  return { lag, ccf: rho, lower: lag.map(() => -se), upper: lag.map(() => se), n }
}

export interface TransferIdentifyResult {
  delay: number
  omega: number
  delta: number
  ccf: CcfResult
}

/**
 * Box–Jenkins-lite transfer ID: prewhiten x via AR, filter y the same way,
 * pick delay from CCF peak; suggest omega/delta ∈ {0,1,2} from lag pattern.
 */
export function transferIdentify(
  y: ArrayLike<number | null | undefined>,
  x: ArrayLike<number | null | undefined>,
  options: { maxDelay?: number; arOrder?: number } = {},
): TransferIdentifyResult {
  const ys = Array.from(cleanNumbers(y))
  const xs = Array.from(cleanNumbers(x))
  const n = Math.min(ys.length, xs.length)
  if (n < 12) throw new RangeError('transferIdentify: need at least 12 paired observations')
  const maxDelay = Math.max(1, Math.min(options.maxDelay ?? 12, Math.floor(n / 4)))
  const arOrder = Math.max(1, Math.min(options.arOrder ?? 2, 5))

  // Yule–Walker AR on x
  const a = acf(xs.slice(0, n), { maxLag: arOrder })
  const ar = new Array(arOrder).fill(0)
  const R = matrix(arOrder, arOrder)
  const r = new Float64Array(arOrder)
  for (let i = 0; i < arOrder; i++) {
    r[i] = a.acf[i + 1]!
    for (let j = 0; j < arOrder; j++) R.data[i * arOrder + j] = a.acf[Math.abs(i - j)]!
  }
  try {
    const sol = Array.from(lstsq(R, r).coef)
    for (let i = 0; i < arOrder; i++) ar[i] = Math.max(-0.95, Math.min(0.95, sol[i]!))
  } catch {
    /* zeros */
  }

  const filterAr = (z: number[]) => {
    const e = new Array(n).fill(0)
    for (let t = 0; t < n; t++) {
      let pred = 0
      for (let i = 0; i < arOrder; i++) if (t - 1 - i >= 0) pred += ar[i]! * z[t - 1 - i]!
      e[t] = z[t]! - pred
    }
    return e
  }
  const ex = filterAr(xs.slice(0, n))
  const ey = filterAr(ys.slice(0, n))
  const cc = ccf(ex, ey, { maxLag: maxDelay })

  // Peak at lag k ≥ 0 means y correlates with x lagged by k (x leads y)
  let bestLag = 0
  let bestAbs = -1
  for (let i = 0; i < cc.lag.length; i++) {
    const k = cc.lag[i]!
    if (k < 0) continue
    const abs = Math.abs(cc.ccf[i]!)
    if (abs > bestAbs) {
      bestAbs = abs
      bestLag = k
    }
  }

  // omega/delta heuristics from |ccf| decay after the peak
  const rhoAt = (lag: number) => {
    const i = cc.lag.indexOf(lag)
    return i >= 0 ? Math.abs(cc.ccf[i]!) : 0
  }
  const r0 = rhoAt(bestLag)
  const r1 = rhoAt(bestLag + 1)
  const r2 = rhoAt(bestLag + 2)
  let omega = 0
  let delta = 0
  if (r1 > 0.4 * r0) omega = 1
  if (r2 > 0.35 * r0 && omega >= 1) omega = 2
  // slow decay → suggest delta
  if (r1 > 0.55 * r0 && r2 > 0.4 * r0) delta = 1
  if (delta === 1 && r2 > 0.5 * r1) delta = 2

  return {
    delay: bestLag,
    omega: Math.min(2, omega),
    delta: Math.min(2, delta),
    ccf: cc,
  }
}

export interface LjungBoxResult {
  statistic: number
  df: number
  pValue: number
  lags: number
}

/** Ljung–Box portmanteau test on residual autocorrelations. */
export function ljungBox(y: ArrayLike<number | null | undefined>, options: { lags?: number; df?: number } = {}): LjungBoxResult {
  const a = acf(y, { maxLag: options.lags ?? 10 })
  const h = options.lags ?? 10
  const n = a.n
  let q = 0
  for (let k = 1; k <= h; k++) q += (a.acf[k]! * a.acf[k]!) / (n - k)
  q *= n * (n + 2)
  const df = Math.max(1, h - (options.df ?? 0))
  return { statistic: q, df, pValue: chi2Dist(df).sf(q), lags: h }
}

// ---- ARIMA / SARIMA / ARIMAX (CSS / CSS-ML) --------------------------------------------------------

export interface ArimaResult {
  order: { p: number; d: number; q: number; P?: number; D?: number; Q?: number; period?: number }
  method: 'CSS' | 'CSS-ML' | 'ML'
  ar: number[]
  ma: number[]
  sar: number[]
  sma: number[]
  intercept: number
  xregCoef?: number[]
  /** SE of [ar…, ma…, sar…, sma…, intercept?, xreg…] */
  se: number[]
  z: number[]
  pValue: number[]
  ci: Array<[number, number]>
  sigma2: number
  fitted: number[]
  residuals: number[]
  forecast: number[]
  forecastLower?: number[]
  forecastUpper?: number[]
  confidence?: number
  aic: number
  bic: number
  logLik: number
  sse: number
}

function diff(y: number[], d: number): number[] {
  let cur = y.slice()
  for (let k = 0; k < d; k++) {
    const next: number[] = []
    for (let i = 1; i < cur.length; i++) next.push(cur[i]! - cur[i - 1]!)
    cur = next
  }
  return cur
}

function seasonalDiff(y: number[], D: number, period: number): number[] {
  let cur = y.slice()
  for (let k = 0; k < D; k++) {
    const next: number[] = []
    for (let i = period; i < cur.length; i++) next.push(cur[i]! - cur[i - period]!)
    cur = next
  }
  return cur
}

function undiff(diffed: number[], history: number[], d: number): number[] {
  if (d === 0) return diffed
  if (d === 1) {
    let last = history[history.length - 1]!
    return diffed.map((delta) => {
      last += delta
      return last
    })
  }
  const y0 = history[history.length - 2]!
  const y1 = history[history.length - 1]!
  let prev = y1
  let prevDiff = y1 - y0
  const out: number[] = []
  for (const d2 of diffed) {
    prevDiff += d2
    prev += prevDiff
    out.push(prev)
  }
  return out
}

function undiffSeasonal(diffed: number[], history: number[], D: number, period: number): number[] {
  if (D === 0) return diffed
  const buf = history.slice()
  const out: number[] = []
  for (const delta of diffed) {
    const next = buf[buf.length - period]! + delta
    out.push(next)
    buf.push(next)
  }
  return out
}

/** Stationarity/invertibility projection for AR/MA of order ≤ 2. */
function projectArMa(coefs: number[], kind: 'ar' | 'ma'): number[] {
  const c = coefs.slice()
  if (c.length === 1) {
    c[0] = Math.max(-0.99, Math.min(0.99, c[0]!))
  } else if (c.length === 2) {
    let a = c[0]!
    let b = c[1]!
    b = Math.max(-0.99, Math.min(0.99, b))
    if (a + b >= 1) a = 0.99 - b
    if (b - a >= 1) a = b - 0.99
    c[0] = a
    c[1] = b
  } else {
    for (let i = 0; i < c.length; i++) c[i] = Math.max(-0.99, Math.min(0.99, c[i]!))
  }
  void kind
  return c
}

function armaPsi(ar: number[], ma: number[], h: number): number[] {
  const psi = new Array(h).fill(0)
  psi[0] = 1
  for (let j = 1; j < h; j++) {
    let s = j <= ma.length ? ma[j - 1]! : 0
    for (let i = 1; i <= ar.length; i++) if (j - i >= 0) s += ar[i - 1]! * psi[j - i]!
    psi[j] = s
  }
  return psi
}

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

function invertSym(H: number[][]): number[][] | null {
  const k = H.length
  const A = H.map((r) => r.slice())
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

function diffColumns(X: number[][], d: number, D: number, period: number): number[][] {
  let cols = X.map((c) => c.slice())
  if (D > 0) cols = cols.map((c) => seasonalDiff(c, D, period))
  if (d > 0) cols = cols.map((c) => diff(c, d))
  return cols
}

function polyMulAR(a: number[], b: number[]): number[] {
  // Multiply (1 - a1 B - …)(1 - b1 B - …) → return coeffs of B^1, B^2, …
  const A = [1, ...a.map((v) => -v)]
  const B = [1, ...b.map((v) => -v)]
  const out = new Array(A.length + B.length - 1).fill(0)
  for (let i = 0; i < A.length; i++) for (let j = 0; j < B.length; j++) out[i + j]! += A[i]! * B[j]!
  // convert back to AR coeffs (negated beyond constant)
  return out.slice(1).map((v) => -v)
}

function seasonalExpand(ar: number[], ma: number[], sar: number[], sma: number[], period: number): { ar: number[]; ma: number[] } {
  if (!(period >= 2) || (sar.length === 0 && sma.length === 0)) return { ar, ma }
  const sarPad: number[] = []
  for (let i = 0; i < sar.length; i++) {
    while (sarPad.length < (i + 1) * period - 1) sarPad.push(0)
    sarPad.push(sar[i]!)
  }
  const smaPad: number[] = []
  for (let i = 0; i < sma.length; i++) {
    while (smaPad.length < (i + 1) * period - 1) smaPad.push(0)
    smaPad.push(sma[i]!)
  }
  return {
    ar: sarPad.length ? polyMulAR(ar, sarPad) : ar.slice(),
    ma: smaPad.length ? polyMulAR(ma, smaPad) : ma.slice(),
  }
}

/**
 * Exact Gaussian log-likelihood for ARMA via Kalman filter (Harvey state-space).
 * Seasonal SARMA supported by expanding to equivalent ARMA.
 */
function kalmanArmaLikelihood(
  y: number[],
  ar: number[],
  ma: number[],
  mu: number,
  xreg?: number[][],
  beta?: number[],
): { logLik: number; sigma2: number; fitted: number[]; resid: number[]; sse: number } {
  const n = y.length
  const p = ar.length
  const q = ma.length
  const r = Math.max(p, q + 1, 1)
  const mX = beta?.length ?? 0
  const meanAt = (t: number) => {
    let m0 = mu
    for (let j = 0; j < mX; j++) m0 += (beta?.[j] ?? 0) * (xreg?.[t]?.[j] ?? 0)
    return m0
  }
  // Companion T
  const T = Array.from({ length: r }, () => new Array(r).fill(0))
  for (let j = 0; j < p; j++) T[0]![j] = ar[j]!
  for (let i = 1; i < r; i++) T[i]![i - 1] = 1
  // R: [1, θ1, …] with MA coeffs (pad)
  const R = new Array(r).fill(0)
  R[0] = 1
  for (let j = 0; j < q; j++) R[j + 1] = ma[j]!
  // Stationary P0 via doubling iterations on vec(P) = T P T' + R R'
  let P = Array.from({ length: r }, () => new Array(r).fill(0))
  for (let it = 0; it < 80; it++) {
    const TP = Array.from({ length: r }, () => new Array(r).fill(0))
    for (let i = 0; i < r; i++) for (let j = 0; j < r; j++) {
      let s = 0
      for (let k = 0; k < r; k++) s += T[i]![k]! * P[k]![j]!
      TP[i]![j] = s
    }
    const TPT = Array.from({ length: r }, () => new Array(r).fill(0))
    for (let i = 0; i < r; i++) for (let j = 0; j < r; j++) {
      let s = 0
      for (let k = 0; k < r; k++) s += TP[i]![k]! * T[j]![k]!
      TPT[i]![j] = s + R[i]! * R[j]!
    }
    let diff = 0
    for (let i = 0; i < r; i++) for (let j = 0; j < r; j++) {
      diff = Math.max(diff, Math.abs(TPT[i]![j]! - P[i]![j]!))
      P[i]![j] = TPT[i]![j]!
    }
    if (diff < 1e-12) break
  }
  let a = new Array(r).fill(0)
  const fitted = new Array(n)
  const resid = new Array(n)
  let sumV2F = 0
  let sumLogF = 0
  for (let t = 0; t < n; t++) {
    const mt = meanAt(t)
    const yt = y[t]! - mt
    // F = Z P Z' + 1  (H=1 for unit innovation scale; Z=[1,0,…])
    const F = Math.max(1e-10, P[0]![0]! + 1)
    const v = yt - a[0]!
    fitted[t] = mt + a[0]!
    resid[t] = v
    sumV2F += (v * v) / F
    sumLogF += Math.log(F)
    // K = T P Z' / F
    const PZ0 = P.map((row) => row[0]!)
    const K = new Array(r)
    for (let i = 0; i < r; i++) {
      let s = 0
      for (let k = 0; k < r; k++) s += T[i]![k]! * PZ0[k]!
      K[i] = s / F
    }
    // a <- T a + K v
    const Ta = new Array(r).fill(0)
    for (let i = 0; i < r; i++) {
      let s = 0
      for (let k = 0; k < r; k++) s += T[i]![k]! * a[k]!
      Ta[i] = s + K[i]! * v
    }
    a = Ta
    // P <- T P T' + R R' - K F K'
    const TP = Array.from({ length: r }, () => new Array(r).fill(0))
    for (let i = 0; i < r; i++) for (let j = 0; j < r; j++) {
      let s = 0
      for (let k = 0; k < r; k++) s += T[i]![k]! * P[k]![j]!
      TP[i]![j] = s
    }
    const Pn = Array.from({ length: r }, () => new Array(r).fill(0))
    for (let i = 0; i < r; i++) for (let j = 0; j < r; j++) {
      let s = 0
      for (let k = 0; k < r; k++) s += TP[i]![k]! * T[j]![k]!
      Pn[i]![j] = s + R[i]! * R[j]! - K[i]! * F * K[j]!
    }
    P = Pn
  }
  const sigma2 = sumV2F / n
  const logLik = -0.5 * (n * Math.log(2 * Math.PI) + n * Math.log(Math.max(sigma2, 1e-300)) + sumLogF + sumV2F / Math.max(sigma2, 1e-300))
  let sse = 0
  for (const e of resid) sse += e * e
  return { logLik, sigma2, fitted, resid, sse }
}

/**
 * ARIMA / SARIMA / ARIMAX by CSS, CSS-ML, or exact Gaussian ML (Kalman; nonseasonal, no xreg).
 * Seasonal via `seasonal: { P, D, Q, period }`; exogenous via `xreg` (+ `xregFuture` for forecasts).
 */
export function arima(
  y: ArrayLike<number | null | undefined>,
  options: {
    p?: number
    d?: number
    q?: number
    includeMean?: boolean
    horizon?: number
    method?: 'CSS' | 'CSS-ML' | 'ML'
    confidence?: number
    seasonal?: { P?: number; D?: number; Q?: number; period: number }
    xreg?: ArrayLike<ArrayLike<number>>
    xregFuture?: ArrayLike<ArrayLike<number>>
    /** Transfer-function inputs: ω(B)BᵇX / (1−δ(B)); noise remains ARMA. */
    transfer?: Array<{
      x: ArrayLike<number>
      delay?: number
      omega?: number
      /** Denominator order δ(B), ≤2; filtered via recursive AR on lagged input. */
      delta?: number
      xFuture?: ArrayLike<number>
    }>
  } = {},
): ArimaResult {
  const raw = Array.from(cleanNumbers(y))
  const p = options.p ?? 1
  const d = options.d ?? 0
  const q = options.q ?? 0
  const P = options.seasonal?.P ?? 0
  const D = options.seasonal?.D ?? 0
  const Q = options.seasonal?.Q ?? 0
  const period = options.seasonal?.period ?? 0
  const method = options.method ?? 'CSS-ML'
  const horizon = options.horizon ?? 0
  const confidence = options.confidence ?? 0.95
  const includeMean = options.includeMean ?? d + D === 0

  if (p < 0 || q < 0 || d < 0 || d > 2) throw new RangeError('arima: invalid order')
  if (p > 5 || q > 5 || P > 5 || Q > 5) throw new RangeError('arima: p,q,P,Q must be ≤ 5')
  if (D > 1) throw new RangeError('arima: seasonal D must be ≤ 1')
  if ((P > 0 || D > 0 || Q > 0) && !(period >= 2)) throw new RangeError('arima: seasonal period must be ≥ 2')
  // ML + transfer: Kalman observation mean = μ + xreg β (transfer columns folded into xreg)

  let xregRaw: number[][] = []
  if (options.xreg) {
    xregRaw = Array.from(options.xreg).map((r) => Array.from(r))
    if (xregRaw.length !== raw.length) throw new RangeError('arima: xreg rows must match y length')
  }
  // Transfer-function: ω(B)B^b X, optional δ(B) recursive filter
  const transferFutureCols: number[][] = []
  if (options.transfer) {
    for (const tf of options.transfer) {
      const xv = Array.from(tf.x)
      if (xv.length !== raw.length) throw new RangeError('arima: transfer.x length must match y')
      const delay = tf.delay ?? 0
      const om = Math.min(2, Math.max(0, Math.floor(tf.omega ?? 0)))
      const deltaOrder = Math.min(2, Math.max(0, Math.floor(tf.delta ?? 0)))
      const fut = tf.xFuture ? Array.from(tf.xFuture) : []
      if (deltaOrder > 0) {
        // Build u_t = sum lagged x, then v_t = u_t + Σ δ_k v_{t-k} with δ from Yule–Walker of u
        const u = xv.map((_, i) => {
          let s = 0
          let c = 0
          for (let L = 0; L <= om; L++) {
            const j = i - delay - L
            if (j >= 0) {
              s += xv[j]!
              c++
            }
          }
          return c > 0 ? s / c : 0
        })
        const arD = new Array(deltaOrder).fill(0)
        if (u.length > deltaOrder + 2) {
          const a = acf(u, { maxLag: deltaOrder })
          const R = matrix(deltaOrder, deltaOrder)
          const rv = new Float64Array(deltaOrder)
          for (let i = 0; i < deltaOrder; i++) {
            rv[i] = a.acf[i + 1]!
            for (let j = 0; j < deltaOrder; j++) R.data[i * deltaOrder + j] = a.acf[Math.abs(i - j)]!
          }
          try {
            const sol = Array.from(lstsq(R, rv).coef)
            for (let i = 0; i < deltaOrder; i++) arD[i] = Math.max(-0.95, Math.min(0.95, sol[i]!))
          } catch {
            /* zeros */
          }
        }
        const v = new Array(u.length).fill(0)
        for (let t = 0; t < u.length; t++) {
          let vt = u[t]!
          for (let k = 1; k <= deltaOrder; k++) if (t - k >= 0) vt += arD[k - 1]! * v[t - k]!
          v[t] = vt
        }
        xregRaw = xregRaw.length ? xregRaw.map((row, i) => [...row, v[i]!]) : v.map((vi) => [vi])
        if (horizon > 0) {
          const uExt = [...u]
          const vExt = [...v]
          const fcol = Array.from({ length: horizon }, (_, h) => {
            let s = 0
            let c = 0
            for (let L = 0; L <= om; L++) {
              const j = raw.length + h - delay - L
              let xj = 0
              if (j >= 0 && j < raw.length) xj = xv[j]!
              else {
                const fj = j - raw.length
                xj = fj >= 0 && fj < fut.length ? fut[fj]! : 0
              }
              s += xj
              c++
            }
            const ut = c > 0 ? s / c : 0
            uExt.push(ut)
            let vt = ut
            const t = uExt.length - 1
            for (let k = 1; k <= deltaOrder; k++) if (t - k >= 0) vt += arD[k - 1]! * vExt[t - k]!
            vExt.push(vt)
            return vt
          })
          transferFutureCols.push(fcol)
        }
      } else {
        for (let L = 0; L <= om; L++) {
          const col = xv.map((_, i) => {
            const j = i - delay - L
            return j >= 0 ? xv[j]! : 0
          })
          xregRaw = xregRaw.length
            ? xregRaw.map((row, i) => [...row, col[i]!])
            : col.map((v) => [v])
          if (horizon > 0) {
            const fcol = Array.from({ length: horizon }, (_, h) => {
              const j = raw.length + h - delay - L
              if (j >= 0 && j < raw.length) return xv[j]!
              const fj = j - raw.length
              return fj >= 0 && fj < fut.length ? fut[fj]! : 0
            })
            transferFutureCols.push(fcol)
          }
        }
      }
    }
  }
  const m = xregRaw[0]?.length ?? 0
  let xregFutureRows: number[][] | undefined
  if (horizon > 0 && m > 0) {
    const hasXreg = !!options.xreg
    if (hasXreg && (!options.xregFuture || Array.from(options.xregFuture).length !== horizon)) {
      throw new RangeError('arima: xregFuture with horizon rows required when forecasting with xreg')
    }
    const base = options.xregFuture
      ? Array.from(options.xregFuture).map((r) => Array.from(r))
      : Array.from({ length: horizon }, () => [] as number[])
    xregFutureRows = base.map((row, h) => [...row, ...transferFutureCols.map((c) => c[h]!)])
  }

  // Differencing: seasonal then regular
  const afterSeas = D > 0 ? seasonalDiff(raw, D, period) : raw.slice()
  const seasDropped = raw.length - afterSeas.length
  const z = d > 0 ? diff(afterSeas, d) : afterSeas.slice()
  const n = z.length
  const burn = Math.max(p, q, P * (period || 0), Q * (period || 0))
  if (n < burn + 3) throw new RangeError('arima: series too short for the requested order')

  let xregZ: number[][] = []
  if (m > 0) {
    // column-major then transpose to row-major aligned with z
    const cols = Array.from({ length: m }, (_, j) => xregRaw.map((r) => r[j]!))
    const colsD = diffColumns(cols, d, D, period || 0)
    const nX = colsD[0]!.length
    // align to end of z (differencing drops from start)
    const offset = n - nX
    xregZ = Array.from({ length: n }, (_, i) => {
      const ii = i - Math.max(0, offset)
      if (ii < 0 || ii >= nX) return new Array(m).fill(0)
      return colsD.map((c) => c[ii]!)
    })
  }

  // Yule–Walker start for nonseasonal AR
  const a = acf(z, { maxLag: Math.max(p, 1) })
  const ar0 = new Array(p).fill(0)
  if (p > 0) {
    const R = matrix(p, p)
    const r = new Float64Array(p)
    for (let i = 0; i < p; i++) {
      r[i] = a.acf[i + 1]!
      for (let j = 0; j < p; j++) R.data[i * p + j] = a.acf[Math.abs(i - j)]!
    }
    try {
      const sol = lstsq(R, r).coef
      for (let i = 0; i < p; i++) ar0[i] = sol[i]!
    } catch {
      /* zeros */
    }
  }
  const ma0 = new Array(q).fill(0)
  const sar0 = new Array(P).fill(0)
  const sma0 = new Array(Q).fill(0)
  // OLS start for mean + xreg
  let mu0 = includeMean ? meanOf(z) : 0
  let beta0 = new Array(m).fill(0)
  if (m > 0) {
    const cols = includeMean ? m + 1 : m
    const M = matrix(n, cols)
    for (let i = 0; i < n; i++) {
      let c = 0
      if (includeMean) {
        M.data[i * cols] = 1
        c = 1
      }
      for (let j = 0; j < m; j++) M.data[i * cols + c + j] = xregZ[i]![j]!
    }
    try {
      const sol = Array.from(lstsq(M, z).coef)
      if (includeMean) {
        mu0 = sol[0]!
        beta0 = sol.slice(1)
      } else beta0 = sol
    } catch {
      /* keep defaults */
    }
  }

  // param layout: ar | ma | sar | sma | mu? | beta
  const unpack = (par: number[]) => {
    let o = 0
    const ar = par.slice(o, o + p)
    o += p
    const ma = par.slice(o, o + q)
    o += q
    const sar = par.slice(o, o + P)
    o += P
    const sma = par.slice(o, o + Q)
    o += Q
    const mu = includeMean ? par[o++]! : 0
    const beta = par.slice(o, o + m)
    return { ar, ma, sar, sma, mu, beta }
  }

  function projectParams(par: number[]): number[] {
    const u = unpack(par)
    const ar = projectArMa(u.ar, 'ar')
    const ma = projectArMa(u.ma, 'ma')
    const sar = projectArMa(u.sar, 'ar')
    const sma = projectArMa(u.sma, 'ma')
    return [...ar, ...ma, ...sar, ...sma, ...(includeMean ? [u.mu] : []), ...u.beta]
  }

  let params = projectParams([...ar0, ...ma0, ...sar0, ...sma0, ...(includeMean ? [mu0] : []), ...beta0])
  const nPar = params.length

  const meanAt = (t: number, mu: number, beta: number[], xrow?: number[]) => {
    let m0 = mu
    for (let j = 0; j < m; j++) m0 += beta[j]! * (xrow?.[j] ?? xregZ[t]?.[j] ?? 0)
    return m0
  }

  const cssEval = (par: number[]) => {
    const pr = projectParams(par)
    const { ar, ma, sar, sma, mu, beta } = unpack(pr)
    const e = new Array(n).fill(0)
    const fitted = new Array(n).fill(0)
    let sse = 0
    for (let t = 0; t < n; t++) {
      const mt = meanAt(t, mu, beta)
      let pred = mt
      for (let i = 0; i < p; i++) if (t - 1 - i >= 0) pred += ar[i]! * (z[t - 1 - i]! - meanAt(t - 1 - i, mu, beta))
      for (let i = 0; i < P; i++) {
        const lag = (i + 1) * period
        if (t - lag >= 0) pred += sar[i]! * (z[t - lag]! - meanAt(t - lag, mu, beta))
      }
      for (let i = 0; i < q; i++) if (t - 1 - i >= 0) pred += ma[i]! * e[t - 1 - i]!
      for (let i = 0; i < Q; i++) {
        const lag = (i + 1) * period
        if (t - lag >= 0) pred += sma[i]! * e[t - lag]!
      }
      fitted[t] = pred
      e[t] = z[t]! - pred
      if (t >= burn) sse += e[t]! * e[t]!
    }
    const nEff = Math.max(1, n - burn)
    const sigma2 = sse / nEff
    const logLik = -0.5 * nEff * (Math.log(2 * Math.PI) + Math.log(Math.max(sigma2, 1e-300)) + 1)
    return { sse, resid: e, fitted, sigma2, logLik, nEff, params: pr, ar, ma, sar, sma, mu, beta }
  }

  const mlEval = (par: number[]) => {
    const pr = projectParams(par)
    const { ar, ma, sar, sma, mu, beta } = unpack(pr)
    const exp = seasonalExpand(ar, ma, sar, sma, period || 0)
    // Cap expanded order for Kalman state size
    if (exp.ar.length > 24 || exp.ma.length > 24) {
      const r = cssEval(par)
      return { ...r, sar, sma }
    }
    const k = kalmanArmaLikelihood(z, exp.ar, exp.ma, mu, m > 0 ? xregZ : undefined, m > 0 ? beta : undefined)
    return {
      sse: k.sse,
      resid: k.resid,
      fitted: k.fitted,
      sigma2: k.sigma2,
      logLik: k.logLik,
      nEff: n,
      params: pr,
      ar,
      ma,
      sar,
      sma,
      mu,
      beta,
    }
  }

  const fitEval = method === 'ML' ? mlEval : cssEval

  const objective = (par: number[]) => {
    const r = fitEval(par)
    return method === 'CSS' ? r.sse : -r.logLik
  }

  const dim = nPar
  const simplex: number[][] = [params.slice()]
  for (let i = 0; i < dim; i++) {
    const v = params.slice()
    v[i]! += Math.max(0.05, 0.1 * Math.abs(v[i]! || 0.1))
    simplex.push(projectParams(v))
  }
  const vals = simplex.map(objective)
  for (let iter = 0; iter < 220; iter++) {
    const order = vals.map((v, i) => i).sort((i, j) => vals[i]! - vals[j]!)
    const bestI = order[0]!
    const worstI = order[dim]!
    const secondI = order[dim - 1]!
    const centroid = new Array(dim).fill(0)
    for (let i = 0; i < dim; i++) {
      const idx = order[i]!
      for (let j = 0; j < dim; j++) centroid[j]! += simplex[idx]![j]!
    }
    for (let j = 0; j < dim; j++) centroid[j]! /= dim
    const reflect = projectParams(centroid.map((c, j) => c + (c - simplex[worstI]![j]!)))
    const fr = objective(reflect)
    if (fr < vals[bestI]!) {
      const expand = projectParams(centroid.map((c, j) => c + 2 * (reflect[j]! - c)))
      const fe = objective(expand)
      if (fe < fr) {
        simplex[worstI] = expand
        vals[worstI] = fe
      } else {
        simplex[worstI] = reflect
        vals[worstI] = fr
      }
    } else if (fr < vals[secondI]!) {
      simplex[worstI] = reflect
      vals[worstI] = fr
    } else {
      const contract = projectParams(centroid.map((c, j) => c + 0.5 * (simplex[worstI]![j]! - c)))
      const fc = objective(contract)
      if (fc < vals[worstI]!) {
        simplex[worstI] = contract
        vals[worstI] = fc
      } else {
        for (let i = 0; i <= dim; i++) {
          if (i === bestI) continue
          simplex[i] = projectParams(simplex[i]!.map((v, j) => 0.5 * (v + simplex[bestI]![j]!)))
          vals[i] = objective(simplex[i]!)
        }
      }
    }
    if (Math.max(...vals) - Math.min(...vals) < 1e-10) break
  }
  const bestIdx = vals.indexOf(Math.min(...vals))
  params = simplex[bestIdx]!
  const best = fitEval(params)

  const se = new Array(nPar).fill(NaN)
  const zStats = new Array(nPar).fill(NaN)
  const pValues = new Array(nPar).fill(NaN)
  const ci: Array<[number, number]> = Array.from({ length: nPar }, () => [NaN, NaN])
  const zCrit = STD.ppf(0.5 + confidence / 2)
  if ((method === 'CSS-ML' || method === 'ML') && nPar > 0) {
    const H = numericalHessian((par) => fitEval(par).logLik, best.params)
    const negH = H.map((row) => row.map((v) => -v))
    const vcov = invertSym(negH)
    if (vcov) {
      for (let i = 0; i < nPar; i++) {
        se[i] = Math.sqrt(Math.max(0, vcov[i]![i]!))
        zStats[i] = se[i]! > 0 ? best.params[i]! / se[i]! : NaN
        pValues[i] = Number.isFinite(zStats[i]!) ? 2 * STD.sf(Math.abs(zStats[i]!)) : NaN
        ci[i] = [best.params[i]! - zCrit * se[i]!, best.params[i]! + zCrit * se[i]!]
      }
    }
  }

  const k = nPar + 1
  const aic = -2 * best.logLik + 2 * k
  const bic = -2 * best.logLik + k * Math.log(best.nEff)

  const xFuture = xregFutureRows ?? []
  // Future xreg on differenced scale: for d=D=0 use as-is; else approximate with raw future (MVP)
  const xregFutureZ = xFuture

  const zF: number[] = []
  const eExt = best.resid.slice()
  const zExt = z.slice()
  for (let h = 0; h < horizon; h++) {
    const t = zExt.length
    const xrow = m > 0 ? xregFutureZ[h] : undefined
    const mt = meanAt(t, best.mu, best.beta, xrow)
    let pred = mt
    for (let i = 0; i < p; i++) {
      const lag = t - 1 - i
      const mlx = meanAt(lag, best.mu, best.beta, lag < n ? undefined : xregFutureZ[lag - n])
      pred += best.ar[i]! * (zExt[lag]! - mlx)
    }
    for (let i = 0; i < P; i++) {
      const lag = t - (i + 1) * period
      if (lag >= 0) {
        const mlx = meanAt(lag, best.mu, best.beta, lag < n ? undefined : xregFutureZ[lag - n])
        pred += best.sar[i]! * (zExt[lag]! - mlx)
      }
    }
    for (let i = 0; i < q; i++) pred += best.ma[i]! * (eExt[t - 1 - i] ?? 0)
    for (let i = 0; i < Q; i++) {
      const lag = t - (i + 1) * period
      if (lag >= 0) pred += best.sma[i]! * (eExt[lag] ?? 0)
    }
    zF.push(pred)
    zExt.push(pred)
    eExt.push(0)
  }

  // Integrate forecasts: first regular undiff onto afterSeas scale, then seasonal undiff onto raw
  let forecast: number[]
  if (d === 0 && D === 0) forecast = zF
  else if (D === 0) forecast = undiff(zF, raw, d)
  else if (d === 0) forecast = undiffSeasonal(zF, raw, D, period)
  else {
    const mid = undiff(zF, afterSeas, d)
    forecast = undiffSeasonal(mid, raw, D, period)
  }

  const sigma = Math.sqrt(best.sigma2)
  // Approximate ψ using nonseasonal ARMA only (seasonal contribution omitted in PI MVP)
  const psi = armaPsi(best.ar, best.ma, Math.max(1, horizon))
  let forecastLower: number[] | undefined
  let forecastUpper: number[] | undefined
  if (horizon > 0) {
    forecastLower = []
    forecastUpper = []
    for (let h = 0; h < horizon; h++) {
      let sumPsi = 0
      for (let j = 0; j <= h; j++) sumPsi += psi[j]! * psi[j]!
      const half = zCrit * sigma * Math.sqrt(sumPsi)
      forecastLower.push(forecast[h]! - half)
      forecastUpper.push(forecast[h]! + half)
    }
  }

  void seasDropped
  return {
    order: {
      p,
      d,
      q,
      ...(options.seasonal ? { P, D, Q, period } : {}),
    },
    method,
    ar: best.ar,
    ma: best.ma,
    sar: best.sar,
    sma: best.sma,
    intercept: best.mu,
    xregCoef: m > 0 ? best.beta : undefined,
    se,
    z: zStats,
    pValue: pValues,
    ci,
    sigma2: best.sigma2,
    fitted: best.fitted,
    residuals: best.resid,
    forecast,
    forecastLower,
    forecastUpper,
    confidence,
    aic,
    bic,
    logLik: best.logLik,
    sse: best.sse,
  }
}

function seriesVariance(y: number[]): number {
  const m = meanOf(y)
  let s = 0
  for (const v of y) s += (v - m) ** 2
  return s / Math.max(1, y.length - 1)
}

function chooseDiffOrder(y: number[], maxD: number): number {
  let cur = y.slice()
  let bestD = 0
  let bestVar = seriesVariance(cur)
  for (let d = 1; d <= maxD; d++) {
    if (cur.length < 8) break
    const next: number[] = []
    for (let i = 1; i < cur.length; i++) next.push(cur[i]! - cur[i - 1]!)
    const v = seriesVariance(next)
    if (v < bestVar * 0.95) {
      bestVar = v
      bestD = d
      cur = next
    } else break
  }
  return bestD
}

/**
 * Hyndman–Khandakar-style stepwise Auto-ARIMA (CSS-ML fits via `arima`).
 */
export function autoArima(
  y: ArrayLike<number | null | undefined>,
  options: {
    seasonalPeriod?: number
    maxP?: number
    maxQ?: number
    maxD?: number
    maxPD?: number
    maxPQ?: number
    information?: 'aic' | 'bic'
    includeMean?: boolean
    horizon?: number
  } = {},
): ArimaResult {
  const raw = Array.from(cleanNumbers(y))
  const period = options.seasonalPeriod
  const maxP = options.maxP ?? 3
  const maxQ = options.maxQ ?? 3
  const maxD = options.maxD ?? 2
  const maxPD = options.maxPD ?? (period ? 1 : 0)
  const maxPQ = options.maxPQ ?? (period ? 1 : 0)
  const info = options.information ?? 'aic'
  const d = chooseDiffOrder(raw, Math.min(2, maxD))
  let D = 0
  if (period && period >= 2) {
    // seasonal diff if variance drops
    const seas = seasonalDiff(raw, 1, period)
    if (seas.length > 10 && seriesVariance(seas) < seriesVariance(raw) * 0.9) D = Math.min(1, maxPD)
  }

  type Ord = { p: number; d: number; q: number; P: number; D: number; Q: number }
  const tryFit = (o: Ord): ArimaResult | null => {
    try {
      return arima(raw, {
        p: o.p,
        d: o.d,
        q: o.q,
        includeMean: options.includeMean,
        horizon: options.horizon,
        seasonal: period && (o.P > 0 || o.D > 0 || o.Q > 0 || D > 0)
          ? { P: o.P, D: o.D, Q: o.Q, period }
          : undefined,
      })
    } catch {
      return null
    }
  }
  const score = (r: ArimaResult) => (info === 'bic' ? r.bic : r.aic)

  let best: ArimaResult | null = null
  let bestOrd: Ord = { p: 0, d, q: 0, P: 0, D, Q: 0 }
  // initial candidates
  const seeds: Ord[] = [
    { p: 0, d, q: 0, P: 0, D, Q: 0 },
    { p: 1, d, q: 0, P: 0, D, Q: 0 },
    { p: 0, d, q: 1, P: 0, D, Q: 0 },
    { p: 1, d, q: 1, P: 0, D, Q: 0 },
    { p: 2, d, q: 2, P: 0, D, Q: 0 },
  ]
  if (period) {
    seeds.push(
      { p: 0, d, q: 0, P: 1, D, Q: 0 },
      { p: 0, d, q: 0, P: 0, D, Q: 1 },
      { p: 1, d, q: 1, P: 1, D, Q: 1 },
    )
  }
  for (const o of seeds) {
    const fit = tryFit(o)
    if (!fit) continue
    if (!best || score(fit) < score(best)) {
      best = fit
      bestOrd = o
    }
  }
  if (!best) throw new RangeError('autoArima: failed to fit any candidate model')

  // Stepwise neighbors
  let improved = true
  let steps = 0
  while (improved && steps < 40) {
    improved = false
    steps++
    const neighbors: Ord[] = []
    const { p, q, P, Q } = bestOrd
    for (const dp of [-1, 0, 1]) for (const dq of [-1, 0, 1]) {
      if (dp === 0 && dq === 0) continue
      neighbors.push({ p: p + dp, d, q: q + dq, P, D, Q })
    }
    if (period) {
      for (const dP of [-1, 0, 1]) for (const dQ of [-1, 0, 1]) {
        if (dP === 0 && dQ === 0) continue
        neighbors.push({ p, d, q, P: P + dP, D, Q: Q + dQ })
      }
    }
    // also allow p±1,q∓1 swaps
    neighbors.push({ p: p + 1, d, q: Math.max(0, q - 1), P, D, Q })
    neighbors.push({ p: Math.max(0, p - 1), d, q: q + 1, P, D, Q })

    for (const o of neighbors) {
      if (o.p < 0 || o.q < 0 || o.P < 0 || o.Q < 0) continue
      if (o.p > maxP || o.q > maxQ || o.P > maxPQ || o.Q > maxPQ) continue
      const fit = tryFit(o)
      if (!fit) continue
      if (score(fit) < score(best!) - 1e-6) {
        best = fit
        bestOrd = o
        improved = true
      }
    }
  }
  return best!
}
