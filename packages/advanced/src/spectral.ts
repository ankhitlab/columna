/**
 * Spectral analysis (Minitab Time Series › Spectral Analysis / white-noise diagnostics): raw and
 * smoothed periodogram (modified Daniell spans), split-cosine taper, dominant frequencies, and the
 * cumulative periodogram with Bartlett's Kolmogorov–Smirnov test for white noise.
 */
import { cleanNumbers } from './tests.js'
import { dft } from './fft.js'

export interface PeriodogramResult {
  /** Fourier frequencies k/n for k = 1 … ⌊n/2⌋ (cycles per observation). */
  frequency: number[]
  period: number[]
  /** Two-sided periodogram I(f) = |Σ xₜ e^{−2πift}|² / n. */
  power: number[]
  /** One-sided spectral density estimate (2·I(f) except at the Nyquist frequency). */
  spectrum: number[]
  /** Frequencies ordered by power (top 5). */
  dominant: Array<{ frequency: number; period: number; power: number }>
  n: number
  detrend: 'none' | 'mean' | 'linear'
  spans?: number[]
}

/**
 * Periodogram of a series. `detrend` removes the mean (default) or a linear trend; `taper` is the
 * split-cosine proportion (0–0.5); `spans` applies modified Daniell smoothing (odd spans, e.g. [3, 3]).
 */
export function periodogram(
  x: ArrayLike<number | null | undefined>,
  options: { detrend?: 'none' | 'mean' | 'linear'; taper?: number; spans?: number[] } = {},
): PeriodogramResult {
  const v = Array.from(cleanNumbers(x))
  const n = v.length
  if (n < 4) throw new RangeError('periodogram needs at least 4 observations')
  const detrend = options.detrend ?? 'mean'
  if (detrend === 'mean') {
    const m = v.reduce((a, b) => a + b, 0) / n
    for (let i = 0; i < n; i++) v[i] = v[i]! - m
  } else if (detrend === 'linear') {
    const tm = (n - 1) / 2
    let sxy = 0
    let sxx = 0
    const m = v.reduce((a, b) => a + b, 0) / n
    for (let i = 0; i < n; i++) {
      sxy += (i - tm) * (v[i]! - m)
      sxx += (i - tm) ** 2
    }
    const b = sxy / sxx
    for (let i = 0; i < n; i++) v[i] = v[i]! - m - b * (i - tm)
  }
  const taper = options.taper ?? 0
  if (taper > 0) {
    if (taper > 0.5) throw new RangeError('periodogram: taper must be ≤ 0.5')
    const m = Math.floor(n * taper)
    for (let i = 0; i < m; i++) {
      const w = 0.5 * (1 - Math.cos((Math.PI * (i + 0.5)) / m))
      v[i] = v[i]! * w
      v[n - 1 - i] = v[n - 1 - i]! * w
    }
  }
  const half = Math.floor(n / 2)
  const frequency: number[] = []
  const power: number[] = []
  const X = dft(v)
  for (let k = 1; k <= half; k++) {
    frequency.push(k / n)
    power.push((X.re[k]! ** 2 + X.im[k]! ** 2) / n)
  }
  let smoothed = power.slice()
  if (options.spans?.length) {
    for (const span of options.spans) {
      if (!(Number.isInteger(span) && span >= 3 && span % 2 === 1)) throw new RangeError('periodogram: spans must be odd integers ≥ 3')
      const h = (span - 1) / 2
      // modified Daniell weights: 1/(2h) interior, 1/(4h) at the ends
      const out = smoothed.map((_, i) => {
        let s = 0
        for (let j = -h; j <= h; j++) {
          const w = Math.abs(j) === h ? 1 / (4 * h) : 1 / (2 * h)
          // reflect at the boundaries
          let idx = i + j
          if (idx < 0) idx = -idx - 1
          if (idx >= smoothed.length) idx = 2 * smoothed.length - idx - 1
          s += w * smoothed[Math.max(0, Math.min(smoothed.length - 1, idx))]!
        }
        return s
      })
      smoothed = out
    }
  }
  const spectrum = smoothed.map((p, i) => (n % 2 === 0 && i === half - 1 ? p : 2 * p))
  const dominant = frequency
    .map((f, i) => ({ frequency: f, period: 1 / f, power: smoothed[i]! }))
    .sort((a, b) => b.power - a.power)
    .slice(0, 5)
  return { frequency, period: frequency.map((f) => 1 / f), power: smoothed, spectrum, dominant, n, detrend, spans: options.spans }
}

export interface CumulativePeriodogramResult {
  frequency: number[]
  /** Normalized cumulative periodogram C(f_k). */
  cumulative: number[]
  /** Bartlett's Kolmogorov–Smirnov statistic: max |C − k/m| and its critical bands at α = 0.05 / 0.01. */
  statistic: number
  critical: { 0.05: number; 0.01: number }
  /** Approximate p-value from the Kolmogorov distribution. */
  pValue: number
  whiteNoise: boolean
}

/** Cumulative periodogram with Bartlett's test for white noise (Box–Jenkins). */
export function cumulativePeriodogram(x: ArrayLike<number | null | undefined>, options: { alpha?: number } = {}): CumulativePeriodogramResult {
  const pg = periodogram(x, { detrend: 'mean' })
  const m = pg.power.length
  const total = pg.power.reduce((a, b) => a + b, 0)
  let acc = 0
  const cumulative = pg.power.map((p) => {
    acc += p
    return acc / total
  })
  // compare with the uniform line k/m
  let d = 0
  for (let k = 0; k < m; k++) d = Math.max(d, Math.abs(cumulative[k]! - (k + 1) / m), Math.abs(cumulative[k]! - k / m))
  const q = Math.sqrt(m - 1)
  const critical = { 0.05: 1.358 / q, 0.01: 1.628 / q }
  // Kolmogorov distribution tail: P(K > z) = 2 Σ (−1)^{j−1} exp(−2 j² z²)
  const z = d * q
  let p = 0
  for (let j = 1; j <= 100; j++) p += 2 * (j % 2 ? 1 : -1) * Math.exp(-2 * j * j * z * z)
  const pValue = Math.max(0, Math.min(1, p))
  const alpha = options.alpha ?? 0.05
  return { frequency: pg.frequency, cumulative, statistic: d, critical, pValue, whiteNoise: pValue >= alpha }
}
