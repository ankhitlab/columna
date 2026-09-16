/**
 * Signal utilities: Welch PSD and Savitzky–Golay filter.
 */
import { cleanNumbers } from './tests.js'

export interface WelchPsdResult {
  frequencies: number[]
  power: number[]
  nfft: number
  fs: number
}

/** Welch periodogram (Hann window, 50% overlap MVP). */
export function welchPsd(
  x: ArrayLike<number | null | undefined>,
  options: { fs?: number; nperseg?: number } = {},
): WelchPsdResult {
  const v = Array.from(cleanNumbers(x))
  const n = v.length
  const fs = options.fs ?? 1
  const nperseg = Math.min(n, options.nperseg ?? Math.min(256, Math.max(8, 1 << Math.floor(Math.log2(n)))))
  const noverlap = Math.floor(nperseg / 2)
  const step = Math.max(1, nperseg - noverlap)
  const window = Float64Array.from({ length: nperseg }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (nperseg - 1)))
  let wss = 0
  for (const w of window) wss += w * w
  const nfft = nperseg
  const half = Math.floor(nfft / 2) + 1
  const acc = new Float64Array(half)
  let nSeg = 0
  for (let start = 0; start + nperseg <= n; start += step) {
    const seg = new Float64Array(nfft)
    for (let i = 0; i < nperseg; i++) seg[i] = v[start + i]! * window[i]!
    // DFT (real) — O(n²) MVP
    for (let k = 0; k < half; k++) {
      let re = 0
      let im = 0
      for (let t = 0; t < nfft; t++) {
        const ang = (-2 * Math.PI * k * t) / nfft
        re += seg[t]! * Math.cos(ang)
        im += seg[t]! * Math.sin(ang)
      }
      const p = (re * re + im * im) / (fs * wss)
      acc[k]! += k === 0 || k === half - 1 ? p : 2 * p
    }
    nSeg++
  }
  if (nSeg === 0) throw new RangeError('welchPsd: series shorter than nperseg')
  const frequencies = Array.from({ length: half }, (_, k) => (k * fs) / nfft)
  const power = Array.from(acc, (p) => p / nSeg)
  return { frequencies, power, nfft, fs }
}

/** Savitzky–Golay smoothing filter. */
export function savitzkyGolay(
  x: ArrayLike<number | null | undefined>,
  options: { windowLength?: number; polyOrder?: number } = {},
): number[] {
  const v = Array.from(cleanNumbers(x))
  const n = v.length
  let wl = options.windowLength ?? 5
  if (wl % 2 === 0) wl++
  const poly = options.polyOrder ?? 2
  if (wl <= poly) throw new RangeError('savitzkyGolay: windowLength must be > polyOrder')
  if (n < wl) throw new RangeError('savitzkyGolay: series shorter than window')
  const half = (wl - 1) / 2
  // build local design and solve for center coefficient row via normal equations once
  const coeffs = sgCoeffs(half, poly)
  const out = new Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = -half; j <= half; j++) {
      const idx = Math.min(n - 1, Math.max(0, i + j))
      s += coeffs[j + half]! * v[idx]!
    }
    out[i] = s
  }
  return out
}

function sgCoeffs(half: number, poly: number): Float64Array {
  const wl = 2 * half + 1
  const A = Array.from({ length: wl }, (_, i) => {
    const x = i - half
    const row = new Array(poly + 1)
    let p = 1
    for (let k = 0; k <= poly; k++) {
      row[k] = p
      p *= x
    }
    return row
  })
  // (AᵀA)^{-1} Aᵀ — want row that extracts value (poly coef 0) at center
  const ata = Array.from({ length: poly + 1 }, () => new Array(poly + 1).fill(0))
  for (let i = 0; i <= poly; i++) {
    for (let j = 0; j <= poly; j++) {
      let s = 0
      for (let r = 0; r < wl; r++) s += A[r]![i]! * A[r]![j]!
      ata[i]![j] = s
    }
  }
  // invert ata (small)
  const inv = invertSmall(ata)
  const coeffs = new Float64Array(wl)
  for (let r = 0; r < wl; r++) {
    let s = 0
    for (let k = 0; k <= poly; k++) s += inv[0]![k]! * A[r]![k]!
    coeffs[r] = s
  }
  return coeffs
}

function invertSmall(M: number[][]): number[][] {
  const n = M.length
  const a = M.map((row) => row.slice())
  const inv = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))
  for (let col = 0; col < n; col++) {
    let piv = col
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r]![col]!) > Math.abs(a[piv]![col]!)) piv = r
    ;[a[col], a[piv]] = [a[piv]!, a[col]!]
    ;[inv[col], inv[piv]] = [inv[piv]!, inv[col]!]
    const div = a[col]![col]!
    for (let j = 0; j < n; j++) {
      a[col]![j]! /= div
      inv[col]![j]! /= div
    }
    for (let r = 0; r < n; r++) {
      if (r === col) continue
      const f = a[r]![col]!
      for (let j = 0; j < n; j++) {
        a[r]![j]! -= f * a[col]![j]!
        inv[r]![j]! -= f * inv[col]![j]!
      }
    }
  }
  return inv
}
