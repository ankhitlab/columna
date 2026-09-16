/**
 * Gaussian kernel density estimation (scipy.stats.gaussian_kde MVP).
 */
import { cleanNumbers } from './tests.js'

export interface GaussianKdeResult {
  n: number
  bandwidth: number
  /** Evaluate density at points. */
  evaluate(x: ArrayLike<number>): Float64Array
  /** Convenience grid over [min−3h, max+3h]. */
  grid(points?: number): { x: number[]; density: number[] }
}

/** Scott / Silverman bandwidth or fixed numeric h. */
export function gaussianKde(
  data: ArrayLike<number | null | undefined>,
  options: { bandwidth?: 'scott' | 'silverman' | number } = {},
): GaussianKdeResult {
  const v = cleanNumbers(data)
  const n = v.length
  if (n < 2) throw new RangeError('gaussianKde needs ≥2 observations')
  let mean = 0
  for (let i = 0; i < n; i++) mean += v[i]!
  mean /= n
  let m2 = 0
  for (let i = 0; i < n; i++) m2 += (v[i]! - mean) ** 2
  const sd = Math.sqrt(m2 / (n - 1))
  const sorted = Array.from(v).sort((a, b) => a - b)
  const q1 = sorted[Math.floor(0.25 * (n - 1))]!
  const q3 = sorted[Math.floor(0.75 * (n - 1))]!
  const iqr = q3 - q1
  const sigma = Math.min(sd, iqr / 1.34) || sd || 1
  let h: number
  if (typeof options.bandwidth === 'number') h = options.bandwidth
  else if (options.bandwidth === 'silverman') h = 0.9 * sigma * n ** (-0.2)
  else h = n ** (-1 / 5) * sigma // Scott
  if (!(h > 0)) h = 1
  const invH = 1 / h
  const invSqrt2pi = 1 / Math.sqrt(2 * Math.PI)
  const evaluate = (xs: ArrayLike<number>) => {
    const out = new Float64Array(xs.length)
    for (let t = 0; t < xs.length; t++) {
      let s = 0
      const xt = xs[t]!
      for (let i = 0; i < n; i++) {
        const u = (xt - v[i]!) * invH
        s += Math.exp(-0.5 * u * u)
      }
      out[t] = (s * invSqrt2pi * invH) / n
    }
    return out
  }
  return {
    n,
    bandwidth: h,
    evaluate,
    grid(points = 100) {
      let lo = Infinity
      let hi = -Infinity
      for (let i = 0; i < n; i++) {
        if (v[i]! < lo) lo = v[i]!
        if (v[i]! > hi) hi = v[i]!
      }
      lo -= 3 * h
      hi += 3 * h
      const x = Array.from({ length: points }, (_, i) => lo + ((hi - lo) * i) / (points - 1))
      return { x, density: Array.from(evaluate(x)) }
    },
  }
}
