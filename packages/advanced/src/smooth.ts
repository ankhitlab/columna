/**
 * Smoothing: LOWESS and isotonic regression.
 */
import { cleanNumbers } from './tests.js'

/**
 * Cleveland LOWESS (degree-1 tricube, optional robust iterations).
 *
 * Sorted by x; each local fit uses a contiguous window of `q = ⌈frac·n⌉`
 * nearest neighbours. Optional `delta` (default 1% of x-range) skips fits for
 * nearly-tied x and interpolates — same idea as R `lowess` / statsmodels.
 * Complexity ≈ O(n · q · iterations / skip-factor), not O(n² log n).
 */
export function lowess(
  x: ArrayLike<number | null | undefined>,
  y: ArrayLike<number | null | undefined>,
  options: { frac?: number; it?: number; delta?: number } = {},
): { x: number[]; y: number[]; fitted: number[] } {
  const xs: number[] = []
  const ys: number[] = []
  const n0 = Math.min(x.length, y.length)
  for (let i = 0; i < n0; i++) {
    const xi = x[i]
    const yi = y[i]
    if (typeof xi === 'number' && Number.isFinite(xi) && typeof yi === 'number' && Number.isFinite(yi)) {
      xs.push(xi)
      ys.push(yi)
    }
  }
  const n = xs.length
  if (n < 3) throw new RangeError('lowess needs ≥3 points')
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => xs[a]! - xs[b]!)
  const X = new Float64Array(n)
  const Y = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    X[i] = xs[order[i]!]!
    Y[i] = ys[order[i]!]!
  }
  const frac = options.frac ?? 2 / 3
  const it = options.it ?? 3
  const q = Math.max(2, Math.min(n, Math.ceil(frac * n)))
  const xRange = X[n - 1]! - X[0]!
  const delta = options.delta ?? (xRange > 0 ? 0.01 * xRange : 0)
  const fitted = new Float64Array(n)
  const weights = new Float64Array(n)
  weights.fill(1)
  const resid = new Float64Array(n)
  // Indices that receive a full local regression (others interpolated)
  const fitAt = new Int32Array(n)
  let nFit = 0
  fitAt[nFit++] = 0
  if (delta > 0) {
    let last = 0
    for (let i = 1; i < n - 1; i++) {
      if (X[i]! - X[last]! > delta) {
        fitAt[nFit++] = i
        last = i
      }
    }
    fitAt[nFit++] = n - 1
  } else {
    for (let i = 1; i < n; i++) fitAt[nFit++] = i
  }

  const localFit = (i: number, lo: number, hi: number) => {
    const xi = X[i]!
    const h = Math.max(xi - X[lo]!, X[hi]! - xi) || 1
    const invH = 1 / h
    let sw = 0
    let swx = 0
    let swy = 0
    let swxx = 0
    let swxy = 0
    for (let j = lo; j <= hi; j++) {
      const rw = weights[j]!
      if (rw === 0) continue
      const u = Math.abs(X[j]! - xi) * invH
      if (u >= 1) continue
      // tricube: (1 - u³)³
      const u3 = u * u * u
      const t = 1 - u3
      const w = t * t * t * rw
      const xj = X[j]!
      const yj = Y[j]!
      sw += w
      swx += w * xj
      swy += w * yj
      swxx += w * xj * xj
      swxy += w * xj * yj
    }
    if (sw === 0) return Y[i]!
    const det = sw * swxx - swx * swx
    if (Math.abs(det) < 1e-14 * Math.max(1, sw * swxx)) return swy / sw
    const a = (swy * swxx - swx * swxy) / det
    const b = (sw * swxy - swx * swy) / det
    return a + b * xi
  }

  for (let iter = 0; iter <= it; iter++) {
    let lo = 0
    let hi = Math.min(n - 1, q - 1)
    for (let k = 0; k < nFit; k++) {
      const i = fitAt[k]!
      const xi = X[i]!
      // Monotone slide: window of size q minimizing radius around xi
      while (hi < n - 1 && X[hi + 1]! - xi < xi - X[lo]!) {
        lo++
        hi++
      }
      // Left edge / short series corrections
      while (lo > 0 && hi - lo + 1 < q) lo--
      while (hi < n - 1 && hi - lo + 1 < q) hi++
      while (hi - lo + 1 > q) {
        if (xi - X[lo]! > X[hi]! - xi) lo++
        else hi--
      }
      fitted[i] = localFit(i, lo, hi)
    }
    // Linear interpolate skipped points between fitted anchors
    if (nFit < n) {
      for (let k = 0; k < nFit - 1; k++) {
        const i0 = fitAt[k]!
        const i1 = fitAt[k + 1]!
        if (i1 <= i0 + 1) continue
        const x0 = X[i0]!
        const x1 = X[i1]!
        const y0 = fitted[i0]!
        const y1 = fitted[i1]!
        const dx = x1 - x0
        if (dx === 0) {
          for (let i = i0 + 1; i < i1; i++) fitted[i] = y0
        } else {
          const inv = 1 / dx
          for (let i = i0 + 1; i < i1; i++) {
            const t = (X[i]! - x0) * inv
            fitted[i] = y0 + t * (y1 - y0)
          }
        }
      }
    }
    if (iter === it) break
    for (let i = 0; i < n; i++) resid[i] = Math.abs(Y[i]! - fitted[i]!)
    const mad = selectMedian(resid)
    const scale = 6 * (mad || 1)
    const invScale = 1 / scale
    for (let i = 0; i < n; i++) {
      const u = resid[i]! * invScale
      weights[i] = u < 1 ? (1 - u * u) * (1 - u * u) : 0
    }
  }
  return { x: Array.from(X), y: Array.from(Y), fitted: Array.from(fitted) }
}

/** Quickselect median (O(n) average). */
function selectMedian(v: Float64Array): number {
  const n = v.length
  const copy = Float64Array.from(v)
  const k = n >> 1
  let lo = 0
  let hi = n - 1
  while (lo < hi) {
    const pivot = copy[(lo + hi) >> 1]!
    let i = lo
    let j = hi
    while (i <= j) {
      while (copy[i]! < pivot) i++
      while (copy[j]! > pivot) j--
      if (i <= j) {
        const t = copy[i]!
        copy[i] = copy[j]!
        copy[j] = t
        i++
        j--
      }
    }
    if (k <= j) hi = j
    else if (k >= i) lo = i
    else break
  }
  return copy[k]!
}

/** Pool-adjacent-violators isotonic (nondecreasing) regression. */
export function isotonicRegression(
  y: ArrayLike<number | null | undefined>,
  options: { increasing?: boolean } = {},
): { fitted: number[]; y: number[] } {
  const v = Array.from(cleanNumbers(y))
  const n = v.length
  if (n < 1) throw new RangeError('isotonicRegression: empty')
  const increasing = options.increasing !== false
  const yWork = increasing ? v.slice() : v.map((x) => -x)
  // PAV
  const level: number[] = []
  const weight: number[] = []
  const fitted = new Array(n)
  for (let i = 0; i < n; i++) {
    level.push(yWork[i]!)
    weight.push(1)
    while (level.length >= 2 && level[level.length - 2]! > level[level.length - 1]!) {
      const y2 = level.pop()!
      const w2 = weight.pop()!
      const y1 = level.pop()!
      const w1 = weight.pop()!
      level.push((y1 * w1 + y2 * w2) / (w1 + w2))
      weight.push(w1 + w2)
    }
  }
  // expand blocks
  let idx = 0
  for (let b = 0; b < level.length; b++) {
    for (let t = 0; t < weight[b]!; t++) fitted[idx++] = level[b]!
  }
  if (!increasing) for (let i = 0; i < n; i++) fitted[i] = -fitted[i]!
  return { fitted, y: v }
}
