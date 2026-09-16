/**
 * Random data generation (Minitab Calc › Random Data): seeded (mulberry32) generators for the common
 * distributions, sampling from columns, and patterned data.
 */
import { binomial as binomialDist, chi2 as chi2Dist, f as fDist, poisson as poissonDist, t as tDist } from './dist.js'

export interface Random {
  /** Uniform(0, 1). */
  uniform(n: number, lo?: number, hi?: number): Float64Array
  normal(n: number, mean?: number, sd?: number): Float64Array
  lognormal(n: number, mu?: number, sigma?: number): Float64Array
  exponential(n: number, scale?: number): Float64Array
  weibull(n: number, shape: number, scale: number): Float64Array
  gamma(n: number, shape: number, scale?: number): Float64Array
  beta(n: number, a: number, b: number): Float64Array
  t(n: number, df: number): Float64Array
  chi2(n: number, df: number): Float64Array
  f(n: number, d1: number, d2: number): Float64Array
  binomial(n: number, trials: number, p: number): Int32Array
  bernoulli(n: number, p: number): Int32Array
  poisson(n: number, lambda: number): Int32Array
  /** Integers in [lo, hi] inclusive. */
  integer(n: number, lo: number, hi: number): Int32Array
  /** Sample rows from a column, with or without replacement. */
  sample<T>(column: ArrayLike<T>, n: number, options?: { replace?: boolean }): T[]
  /** Shuffle a copy of the column. */
  shuffle<T>(column: ArrayLike<T>): T[]
  /** Raw uniform draw. */
  next(): number
}

/** Seeded generator. Same seed → same sequence (mulberry32 + Box–Muller / Marsaglia–Tsang). */
export function random(seed = 12345): Random {
  let a = seed >>> 0
  const u = () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296
  }
  const normal1 = () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u())
  const gamma1 = (shape: number): number => {
    if (shape < 1) return gamma1(shape + 1) * u() ** (1 / shape)
    const d = shape - 1 / 3
    const c = 1 / Math.sqrt(9 * d)
    for (;;) {
      let x: number
      let v: number
      do {
        x = normal1()
        v = 1 + c * x
      } while (v <= 0)
      v = v * v * v
      const uu = u()
      if (uu < 1 - 0.0331 * x ** 4) return d * v
      if (Math.log(uu) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v
    }
  }
  const fill = (n: number, g: () => number) => {
    const out = new Float64Array(n)
    for (let i = 0; i < n; i++) out[i] = g()
    return out
  }
  const fillInt = (n: number, g: () => number) => {
    const out = new Int32Array(n)
    for (let i = 0; i < n; i++) out[i] = g()
    return out
  }
  const poisson1 = (lambda: number) => {
    if (lambda < 30) {
      const L = Math.exp(-lambda)
      let k = 0
      let p = 1
      do {
        k++
        p *= u()
      } while (p > L)
      return k - 1
    }
    return poissonDist(lambda).ppf(u())
  }
  return {
    next: u,
    uniform: (n, lo = 0, hi = 1) => fill(n, () => lo + (hi - lo) * u()),
    normal: (n, mean = 0, sd = 1) => fill(n, () => mean + sd * normal1()),
    lognormal: (n, mu = 0, sigma = 1) => fill(n, () => Math.exp(mu + sigma * normal1())),
    exponential: (n, scale = 1) => fill(n, () => -scale * Math.log(u())),
    weibull: (n, shape, scale) => fill(n, () => scale * (-Math.log(u())) ** (1 / shape)),
    gamma: (n, shape, scale = 1) => fill(n, () => scale * gamma1(shape)),
    beta: (n, aa, bb) =>
      fill(n, () => {
        const x = gamma1(aa)
        const y = gamma1(bb)
        return x / (x + y)
      }),
    t: (n, df) => {
      const d = tDist(df)
      return fill(n, () => d.ppf(u()))
    },
    chi2: (n, df) => {
      const d = chi2Dist(df)
      return fill(n, () => d.ppf(u()))
    },
    f: (n, d1, d2) => {
      const d = fDist(d1, d2)
      return fill(n, () => d.ppf(u()))
    },
    binomial: (n, trials, p) => {
      if (trials <= 200) return fillInt(n, () => {
        let k = 0
        for (let i = 0; i < trials; i++) if (u() < p) k++
        return k
      })
      const d = binomialDist(trials, p)
      return fillInt(n, () => d.ppf(u()))
    },
    bernoulli: (n, p) => fillInt(n, () => (u() < p ? 1 : 0)),
    poisson: (n, lambda) => fillInt(n, () => poisson1(lambda)),
    integer: (n, lo, hi) => fillInt(n, () => lo + Math.floor(u() * (hi - lo + 1))),
    sample: <T>(column: ArrayLike<T>, n: number, options: { replace?: boolean } = {}) => {
      const m = column.length
      if (options.replace) return Array.from({ length: n }, () => column[Math.floor(u() * m)]!)
      if (n > m) throw new RangeError('sample without replacement: n exceeds the column length')
      const idx = Array.from({ length: m }, (_, i) => i)
      for (let i = 0; i < n; i++) {
        const j = i + Math.floor(u() * (m - i))
        ;[idx[i], idx[j]] = [idx[j]!, idx[i]!]
      }
      return idx.slice(0, n).map((i) => column[i]!)
    },
    shuffle: <T>(column: ArrayLike<T>) => {
      const out = Array.from(column)
      for (let i = out.length - 1; i > 0; i--) {
        const j = Math.floor(u() * (i + 1))
        ;[out[i], out[j]] = [out[j]!, out[i]!]
      }
      return out
    },
  }
}

/** Patterned data (Minitab Make Patterned Data): from…to by step, each value repeated `repeat` times, whole sequence repeated `times`. */
export function patterned(from: number, to: number, options: { step?: number; repeat?: number; times?: number } = {}): number[] {
  const step = options.step ?? 1
  if (!(step > 0)) throw new RangeError('patterned: step must be > 0')
  const repeat = options.repeat ?? 1
  const times = options.times ?? 1
  const base: number[] = []
  const dir = to >= from ? 1 : -1
  for (let v = from; dir > 0 ? v <= to + 1e-12 : v >= to - 1e-12; v += dir * step) for (let r = 0; r < repeat; r++) base.push(Number(v.toFixed(12)))
  const out: number[] = []
  for (let t = 0; t < times; t++) out.push(...base)
  return out
}
