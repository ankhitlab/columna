/**
 * Numerical utilities: Brent root finding and quadrature.
 */

/** Brent–Dekker root finder on [a,b] with f(a)·f(b) < 0. */
export function brentq(
  f: (x: number) => number,
  a: number,
  b: number,
  options: { xtol?: number; maxIter?: number } = {},
): number {
  const xtol = options.xtol ?? 1e-12
  const maxIter = options.maxIter ?? 100
  let fa = f(a)
  let fb = f(b)
  if (!Number.isFinite(fa) || !Number.isFinite(fb)) throw new RangeError('brentq: f returned non-finite')
  if (fa * fb > 0) throw new RangeError('brentq: root not bracketed')
  if (Math.abs(fa) < Math.abs(fb)) {
    ;[a, b] = [b, a]
    ;[fa, fb] = [fb, fa]
  }
  let c = a
  let fc = fa
  let d = b - a
  let e = d
  for (let iter = 0; iter < maxIter; iter++) {
    if (fb === 0 || Math.abs(b - a) < xtol) return b
    if (Math.abs(fa) < Math.abs(fb)) {
      ;[a, b] = [b, a]
      ;[fa, fb] = [fb, fa]
      c = a
      fc = fa
    }
    const tol1 = 2 * Number.EPSILON * Math.abs(b) + 0.5 * xtol
    const xm = 0.5 * (c - b)
    if (Math.abs(xm) <= tol1 || fb === 0) return b
    let p = 0
    let q = 0
    if (Math.abs(e) >= tol1 && Math.abs(fa) > Math.abs(fb)) {
      // inverse quadratic / secant
      const s = fb / fa
      if (a === c) {
        p = 2 * xm * s
        q = 1 - s
      } else {
        const r = fb / fc
        const t = fa / fc
        p = s * (2 * xm * t * (t - r) - (b - a) * (r - 1))
        q = (t - 1) * (r - 1) * (s - 1)
      }
      if (p > 0) q = -q
      p = Math.abs(p)
      const min1 = 3 * xm * q - Math.abs(tol1 * q)
      const min2 = Math.abs(e * q)
      if (2 * p < Math.min(min1, min2)) {
        e = d
        d = p / q
      } else {
        d = xm
        e = d
      }
    } else {
      d = xm
      e = d
    }
    a = b
    fa = fb
    b += Math.abs(d) > tol1 ? d : (xm > 0 ? tol1 : -tol1)
    fb = f(b)
    if (fb * fc > 0) {
      c = a
      fc = fa
      d = b - a
      e = d
    }
  }
  return b
}

/** Composite trapezoidal rule. */
/** Min / max without spread (Math.min(...x) overflows the argument stack past ~120k values). */
export function minOf(x: ArrayLike<number>): number {
  let m = Infinity
  for (let i = 0; i < x.length; i++) if (x[i]! < m) m = x[i]!
  return m
}
export function maxOf(x: ArrayLike<number>): number {
  let m = -Infinity
  for (let i = 0; i < x.length; i++) if (x[i]! > m) m = x[i]!
  return m
}

/** Trapezoidal rule (`scipy.integrate.trapezoid`): `x` = sample positions (same length as `y`) or spacing `dx` (default 1). */
export function trapz(y: ArrayLike<number>, x?: ArrayLike<number> | number): number {
  const n = y.length
  if (n < 2) return 0
  let s = 0
  if (x === undefined || typeof x === 'number') {
    const dx = x ?? 1
    for (let i = 1; i < n; i++) s += 0.5 * (Number(y[i - 1]) + Number(y[i]))
    return s * dx
  }
  if (x.length !== n) throw new RangeError(`trapz: x has ${x.length} samples, y has ${n}`)
  for (let i = 1; i < n; i++) {
    const dx = Number(x[i]) - Number(x[i - 1])
    s += 0.5 * (Number(y[i - 1]) + Number(y[i])) * dx
  }
  return s
}

/**
 * Composite Simpson's rule — the same algorithm as `scipy.integrate.simpson` (SciPy ≥ 1.11).
 *
 * `x` is either sample positions (same length as `y`, strictly monotonic — decreasing integrates with the
 * sign of the direction, as in SciPy) or a uniform spacing `dx` (default 1).
 *
 * - Odd number of samples (even number of intervals): every pair of intervals `[x₀, x₁, x₂]` is integrated
 *   by the parabola through its three points, with the weights for unequal widths `h₀, h₁`:
 *   `(h₀+h₁)/6 · [ (2 − h₁/h₀)·y₀ + (h₀+h₁)²/(h₀h₁)·y₁ + (2 − h₀/h₁)·y₂ ]`. Exact for cubics on any grid.
 * - Even number of samples: Simpson over the first N − 1 samples plus Cartwright's correction for the last
 *   interval (the parabola through the last three points integrated over the last interval only):
 *   `α·y_{N−1} + β·y_{N−2} − η·y_{N−3}` with `α = (2h₁² + 3h₀h₁)/(6(h₀+h₁))`, `β = (h₁² + 3h₀h₁)/(6h₀)`,
 *   `η = h₁³/(6h₀(h₀+h₁))`. Exact for quadratics.
 * - Two samples: the trapezoid (the only rule two points define). Fewer: 0.
 */
export function simpson(y: ArrayLike<number>, x?: ArrayLike<number> | number): number {
  const n = y.length
  if (n < 2) return 0
  const ys = new Float64Array(n)
  for (let i = 0; i < n; i++) ys[i] = Number(y[i])
  let xs: Float64Array | null = null
  let dx = 1
  if (typeof x === 'number') {
    if (!Number.isFinite(x) || x === 0) throw new RangeError(`simpson: dx must be finite and non-zero (got ${x})`)
    dx = x
  } else if (x !== undefined) {
    if (x.length !== n) throw new RangeError(`simpson: x has ${x.length} samples, y has ${n}`)
    xs = new Float64Array(n)
    for (let i = 0; i < n; i++) xs[i] = Number(x[i])
    const dir = Math.sign(xs[1]! - xs[0]!)
    for (let i = 1; i < n; i++) {
      const d = xs[i]! - xs[i - 1]!
      if (!Number.isFinite(d) || Math.sign(d) !== dir || d === 0) {
        throw new RangeError('simpson: x must be finite and strictly monotonic (a repeated or reversed sample makes an interval of width ≤ 0)')
      }
    }
  }
  const h = (i: number): number => (xs ? xs[i + 1]! - xs[i]! : dx)
  if (n === 2) return 0.5 * h(0) * (ys[0]! + ys[1]!)

  // pairs of intervals over samples [0, last]
  const pairs = (last: number): number => {
    let s = 0
    for (let i = 0; i + 2 <= last; i += 2) {
      const h0 = h(i)
      const h1 = h(i + 1)
      const hsum = h0 + h1
      s += (hsum / 6) * ((2 - h1 / h0) * ys[i]! + ((hsum * hsum) / (h0 * h1)) * ys[i + 1]! + (2 - h0 / h1) * ys[i + 2]!)
    }
    return s
  }
  if (n % 2 === 1) return pairs(n - 1)
  const h0 = h(n - 3)
  const h1 = h(n - 2)
  const alpha = (2 * h1 * h1 + 3 * h0 * h1) / (6 * (h0 + h1))
  const beta = (h1 * h1 + 3 * h0 * h1) / (6 * h0)
  const eta = (h1 * h1 * h1) / (6 * h0 * (h0 + h1))
  return pairs(n - 2) + alpha * ys[n - 1]! + beta * ys[n - 2]! - eta * ys[n - 3]!
}
