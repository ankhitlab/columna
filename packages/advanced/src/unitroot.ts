/**
 * Unit-root tests: Augmented Dickey–Fuller and KPSS (MVP).
 */
import { normal } from './dist.js'
import { cleanNumbers } from './tests.js'
import { lstsq, matrix } from './linalg.js'

export interface UnitRootResult {
  test: 'ADF' | 'KPSS'
  statistic: number
  pValue: number
  lags: number
  n: number
  regression: 'c' | 'ct' | 'n'
}

const STD = normal()

/** MacKinnon approx p-value for ADF τ (constant, no trend) — rough MVP. */
function adfPValue(tau: number, regression: 'c' | 'ct' | 'n'): number {
  // Critical values ≈ (1%, 5%, 10%) for large n
  const crit =
    regression === 'ct'
      ? { c01: -3.96, c05: -3.41, c10: -3.13 }
      : regression === 'n'
        ? { c01: -2.58, c05: -1.95, c10: -1.61 }
        : { c01: -3.43, c05: -2.86, c10: -2.57 }
  // interpolate on normal scale of |tau| relative to c05
  if (tau <= crit.c01) return 0.001
  if (tau >= crit.c10) {
    // surface toward 1 as tau → 0+
    return Math.min(0.99, STD.sf(Math.abs(tau) * 0.5))
  }
  if (tau <= crit.c05) {
    const t = (crit.c05 - tau) / (crit.c05 - crit.c01)
    return 0.05 - t * 0.04
  }
  const t = (crit.c10 - tau) / (crit.c10 - crit.c05)
  return 0.1 - t * 0.05
}

/**
 * Augmented Dickey–Fuller test of H0: unit root.
 * Δy_t = α + β t + γ y_{t-1} + Σ φ_i Δy_{t-i} + e
 */
export function adfTest(
  y: ArrayLike<number | null | undefined>,
  options: { lags?: number; regression?: 'c' | 'ct' | 'n' } = {},
): UnitRootResult {
  const v = Array.from(cleanNumbers(y))
  const n0 = v.length
  if (n0 < 8) throw new RangeError('adfTest needs ≥8 observations')
  const regression = options.regression ?? 'c'
  const maxLag = options.lags ?? Math.max(0, Math.floor(Math.pow(n0 - 1, 1 / 3)))
  const dy: number[] = []
  for (let t = 1; t < n0; t++) dy.push(v[t]! - v[t - 1]!)
  // effective sample starts at maxLag+1 in differenced index
  const rows: number[][] = []
  const yy: number[] = []
  for (let t = maxLag; t < dy.length; t++) {
    const row: number[] = []
    if (regression === 'c' || regression === 'ct') row.push(1)
    if (regression === 'ct') row.push(t + 1) // trend on level time
    row.push(v[t]!) // y_{t} is lag of level for Δy_t (index t in dy corresponds to time t+1)
    for (let L = 1; L <= maxLag; L++) row.push(dy[t - L]!)
    rows.push(row)
    yy.push(dy[t]!)
  }
  const n = rows.length
  const p = rows[0]!.length
  const X = matrix(n, p)
  const yv = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    yv[i] = yy[i]!
    for (let j = 0; j < p; j++) X.data[i * p + j] = rows[i]![j]!
  }
  const fit = lstsq(X, yv)
  const gammaIdx = regression === 'ct' ? 2 : regression === 'c' ? 1 : 0
  const gamma = fit.coef[gammaIdx]!
  const se = Math.sqrt(Math.max(1e-300, fit.xtxInv.data[gammaIdx * p + gammaIdx]! * (fit.sse / Math.max(1, n - p))))
  const statistic = gamma / se
  return {
    test: 'ADF',
    statistic,
    pValue: adfPValue(statistic, regression),
    lags: maxLag,
    n,
    regression,
  }
}

/**
 * KPSS test of H0: level/trend stationarity (Kwiatkowski–Phillips–Schmidt–Shin).
 */
export function kpssTest(
  y: ArrayLike<number | null | undefined>,
  options: { regression?: 'c' | 'ct'; lags?: number } = {},
): UnitRootResult {
  const v = Array.from(cleanNumbers(y))
  const n = v.length
  if (n < 8) throw new RangeError('kpssTest needs ≥8 observations')
  const regression = options.regression ?? 'c'
  // residuals from constant or constant+trend
  const resid = new Float64Array(n)
  if (regression === 'c') {
    let m = 0
    for (const x of v) m += x
    m /= n
    for (let i = 0; i < n; i++) resid[i] = v[i]! - m
  } else {
    // OLS on [1, t]
    let s1 = 0
    let st = 0
    let stt = 0
    let sy = 0
    let sty = 0
    for (let i = 0; i < n; i++) {
      const t = i + 1
      s1++
      st += t
      stt += t * t
      sy += v[i]!
      sty += t * v[i]!
    }
    const det = s1 * stt - st * st
    const a = (sy * stt - st * sty) / det
    const b = (s1 * sty - st * sy) / det
    for (let i = 0; i < n; i++) resid[i] = v[i]! - a - b * (i + 1)
  }
  // partial sums
  const S = new Float64Array(n)
  S[0] = resid[0]!
  for (let i = 1; i < n; i++) S[i] = S[i - 1]! + resid[i]!
  let eta = 0
  for (let i = 0; i < n; i++) eta += S[i]! * S[i]!
  eta /= n * n
  // long-run variance with Newey–West
  const lags = options.lags ?? Math.max(0, Math.floor(3 * Math.sqrt(n) / 13))
  let s2 = 0
  for (let i = 0; i < n; i++) s2 += resid[i]! * resid[i]!
  s2 /= n
  for (let L = 1; L <= lags; L++) {
    let gamma = 0
    for (let i = L; i < n; i++) gamma += resid[i]! * resid[i - L]!
    gamma /= n
    const w = 1 - L / (lags + 1)
    s2 += 2 * w * gamma
  }
  s2 = Math.max(1e-12, s2)
  const statistic = eta / s2
  // asymptotic critical values (level / trend)
  const crit = regression === 'ct' ? { c10: 0.119, c05: 0.146, c01: 0.216 } : { c10: 0.347, c05: 0.463, c01: 0.739 }
  let pValue: number
  if (statistic <= crit.c10) pValue = 0.2
  else if (statistic <= crit.c05) pValue = 0.1 - 0.05 * ((statistic - crit.c10) / (crit.c05 - crit.c10))
  else if (statistic <= crit.c01) pValue = 0.05 - 0.04 * ((statistic - crit.c05) / (crit.c01 - crit.c05))
  else pValue = Math.max(0.001, 0.01 * (crit.c01 / statistic))
  return {
    test: 'KPSS',
    statistic,
    pValue,
    lags,
    n,
    regression,
  }
}
