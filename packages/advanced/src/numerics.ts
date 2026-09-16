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

export function trapz(y: ArrayLike<number>, x?: ArrayLike<number>): number {
  const n = y.length
  if (n < 2) return 0
  let s = 0
  if (!x) {
    for (let i = 1; i < n; i++) s += 0.5 * (Number(y[i - 1]) + Number(y[i]))
    return s
  }
  for (let i = 1; i < n; i++) {
    const dx = Number(x[i]) - Number(x[i - 1])
    s += 0.5 * (Number(y[i - 1]) + Number(y[i])) * dx
  }
  return s
}

/** Composite Simpson rule (even n-1 preferred; falls back with trap on last). */
export function simpson(y: ArrayLike<number>, x?: ArrayLike<number>): number {
  const n = y.length
  if (n < 2) return 0
  if (!x) {
    if (n % 2 === 0) {
      // odd number of intervals — simpson on n-1 then trap last
      return simpson(Array.from({ length: n - 1 }, (_, i) => Number(y[i]))) + 0.5 * (Number(y[n - 2]) + Number(y[n - 1]))
    }
    let s = Number(y[0]) + Number(y[n - 1])
    for (let i = 1; i < n - 1; i++) s += (i % 2 === 0 ? 2 : 4) * Number(y[i])
    return s / 3
  }
  // uneven spacing: pairwise parabolic / trap fallback
  return trapz(y, x)
}
