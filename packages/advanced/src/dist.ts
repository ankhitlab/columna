/**
 * Probability distributions for statistical tests: normal, Student t, chi-square, F.
 *
 * Each factory returns a frozen distribution with `pdf`, `cdf`, `sf` (= 1 − cdf, accurate in the
 * upper tail), `ppf` (inverse cdf) and `isf` (inverse sf), plus `mean` / `variance`. Everything is
 * built on three special functions — `lgamma` (Lanczos), the regularized incomplete gamma
 * (series + Lentz continued fraction) and the regularized incomplete beta (Lentz) — so the cdf values
 * agree with scipy to ~1e-14 relative in the body and ~1e-12 deep in the tails. Inverses use
 * Acklam's rational approximation refined by Halley steps (normal) or bracketed bisection followed
 * by Newton polish (t, chi², F).
 *
 *   dist.normal().cdf(1.96)            // 0.9750021…
 *   dist.t(10).ppf(0.975)              // 2.2281388…
 *   dist.chi2(5).sf(11.07)             // p-value 0.0500096…
 *   dist.f(3, 10).cdf(3.708)           // 0.9499912…
 */

export interface Distribution {
  readonly name: string
  /** Probability density at x. */
  pdf(x: number): number
  /** P(X ≤ x). */
  cdf(x: number): number
  /** P(X > x) — use for p-values; accurate where 1 − cdf would cancel. */
  sf(x: number): number
  /** Inverse cdf: smallest x with P(X ≤ x) ≥ p. */
  ppf(p: number): number
  /** Inverse survival: x with P(X > x) = p. */
  isf(p: number): number
  readonly mean: number
  readonly variance: number
  /** Apply one of the functions to every element of a typed / plain array. */
  map(fn: 'pdf' | 'cdf' | 'sf' | 'ppf' | 'isf', xs: ArrayLike<number>): Float64Array
}

// ---- special functions --------------------------------------------------------------------------

const EPS = 1e-16
const FPMIN = Number.MIN_VALUE / EPS
const MAX_ITER = 1000
/**
 * Iteration budget for the incomplete gamma / beta series and continued fractions. Near the transition
 * x ≈ a the series' ratio is x / (a + i) ≈ 1 and convergence needs O(√a) terms: at a = 10⁵ about 3 500,
 * which a fixed cap of 1 000 silently truncated (Poisson sf at λ = 10⁵ was off by 2·10⁻⁷).
 */
const iterBudget = (scale: number) => Math.max(MAX_ITER, Math.ceil(60 * Math.sqrt(Math.max(1, scale))) + 200)
const LN_SQRT_2PI = 0.9189385332046727 // ln √(2π)
const SQRT2 = Math.SQRT2

// Lanczos approximation, g = 7, n = 9 (relative error ≈ 1e-15)
const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
]

/** ln Γ(x) for x > 0 (reflection for x < 0.5). */
export function lgamma(x: number): number {
  if (Number.isNaN(x)) return NaN
  if (x <= 0 && Number.isInteger(x)) return Infinity
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x)
  x -= 1
  let a = LANCZOS[0]!
  const t = x + 7.5
  for (let i = 1; i < 9; i++) a += LANCZOS[i]! / (x + i)
  return LN_SQRT_2PI + (x + 0.5) * Math.log(t) - t + Math.log(a)
}

/** Regularized lower incomplete gamma P(a, x). */
export function gammainc(a: number, x: number): number {
  if (!(a > 0) || Number.isNaN(x)) return NaN
  if (x <= 0) return 0
  if (x === Infinity) return 1
  return x < a + 1 ? gammaSeries(a, x) : 1 - gammaCF(a, x)
}

/** Regularized upper incomplete gamma Q(a, x) = 1 − P(a, x), accurate for large x. */
export function gammaincc(a: number, x: number): number {
  if (!(a > 0) || Number.isNaN(x)) return NaN
  if (x <= 0) return 1
  if (x === Infinity) return 0
  return x < a + 1 ? 1 - gammaSeries(a, x) : gammaCF(a, x)
}

function gammaSeries(a: number, x: number): number {
  let ap = a
  let sum = 1 / a
  let del = sum
  const budget = iterBudget(a)
  for (let i = 0; i < budget; i++) {
    ap += 1
    del *= x / ap
    sum += del
    if (Math.abs(del) < Math.abs(sum) * EPS) break
  }
  return sum * Math.exp(-x + a * Math.log(x) - lgamma(a))
}

function gammaCF(a: number, x: number): number {
  // modified Lentz for the continued fraction of Q(a, x)
  let b = x + 1 - a
  let c = 1 / FPMIN
  let d = 1 / b
  let h = d
  const budget = iterBudget(a)
  for (let i = 1; i <= budget; i++) {
    const an = -i * (i - a)
    b += 2
    d = an * d + b
    if (Math.abs(d) < FPMIN) d = FPMIN
    c = b + an / c
    if (Math.abs(c) < FPMIN) c = FPMIN
    d = 1 / d
    const del = d * c
    h *= del
    if (Math.abs(del - 1) < EPS) break
  }
  return Math.exp(-x + a * Math.log(x) - lgamma(a)) * h
}

/** Regularized incomplete beta I_x(a, b). */
export function betainc(a: number, b: number, x: number): number {
  if (!(a > 0) || !(b > 0) || Number.isNaN(x)) return NaN
  if (x <= 0) return 0
  if (x >= 1) return 1
  const lbt = lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log1p(-x)
  if (x < (a + 1) / (a + b + 2)) return (Math.exp(lbt) * betaCF(a, b, x)) / a
  return 1 - (Math.exp(lbt) * betaCF(b, a, 1 - x)) / b
}

function betaCF(a: number, b: number, x: number): number {
  const qab = a + b
  const qap = a + 1
  const qam = a - 1
  let c = 1
  let d = 1 - (qab * x) / qap
  if (Math.abs(d) < FPMIN) d = FPMIN
  d = 1 / d
  let h = d
  for (let m = 1; m <= MAX_ITER; m++) {
    const m2 = 2 * m
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2))
    d = 1 + aa * d
    if (Math.abs(d) < FPMIN) d = FPMIN
    c = 1 + aa / c
    if (Math.abs(c) < FPMIN) c = FPMIN
    d = 1 / d
    h *= d * c
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2))
    d = 1 + aa * d
    if (Math.abs(d) < FPMIN) d = FPMIN
    c = 1 + aa / c
    if (Math.abs(c) < FPMIN) c = FPMIN
    d = 1 / d
    const del = d * c
    h *= del
    if (Math.abs(del - 1) < EPS) break
  }
  return h
}

/** Error function. */
export function erf(x: number): number {
  if (Number.isNaN(x)) return NaN
  const p = gammainc(0.5, x * x)
  return x >= 0 ? p : -p
}

/** Complementary error function, accurate for large x (no cancellation). */
export function erfc(x: number): number {
  if (Number.isNaN(x)) return NaN
  if (x >= 0) return gammaincc(0.5, x * x)
  return 2 - gammaincc(0.5, x * x)
}

// ---- inverse helpers ----------------------------------------------------------------------------

// Acklam's algorithm for the standard normal quantile (rel. error 1.15e-9), then Halley refinement.
const ACK_A = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239]
const ACK_B = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1]
const ACK_C = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
const ACK_D = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416]

function stdNormalPpf(p: number): number {
  if (Number.isNaN(p) || p < 0 || p > 1) return NaN
  if (p === 0) return -Infinity
  if (p === 1) return Infinity
  const plow = 0.02425
  let x: number
  if (p < plow) {
    const q = Math.sqrt(-2 * Math.log(p))
    x =
      (((((ACK_C[0]! * q + ACK_C[1]!) * q + ACK_C[2]!) * q + ACK_C[3]!) * q + ACK_C[4]!) * q + ACK_C[5]!) /
      ((((ACK_D[0]! * q + ACK_D[1]!) * q + ACK_D[2]!) * q + ACK_D[3]!) * q + 1)
  } else if (p <= 1 - plow) {
    const q = p - 0.5
    const r = q * q
    x =
      ((((((ACK_A[0]! * r + ACK_A[1]!) * r + ACK_A[2]!) * r + ACK_A[3]!) * r + ACK_A[4]!) * r + ACK_A[5]!) * q) /
      (((((ACK_B[0]! * r + ACK_B[1]!) * r + ACK_B[2]!) * r + ACK_B[3]!) * r + ACK_B[4]!) * r + 1)
  } else {
    const q = Math.sqrt(-2 * Math.log1p(-p))
    x =
      -(((((ACK_C[0]! * q + ACK_C[1]!) * q + ACK_C[2]!) * q + ACK_C[3]!) * q + ACK_C[4]!) * q + ACK_C[5]!) /
      ((((ACK_D[0]! * q + ACK_D[1]!) * q + ACK_D[2]!) * q + ACK_D[3]!) * q + 1)
  }
  // Two Halley steps against the accurate erfc-based cdf → full double precision
  for (let k = 0; k < 2; k++) {
    // residual evaluated on the nearer tail: cdf(x) − p for p < ½, (1 − p) − sf(x) otherwise (1 − p is exact there)
    const e = p < 0.5 ? 0.5 * erfc(-x / SQRT2) - p : 1 - p - 0.5 * erfc(x / SQRT2)
    const u = e * Math.sqrt(2 * Math.PI) * Math.exp((x * x) / 2)
    x -= u / (1 + (x * u) / 2)
  }
  return x
}

/**
 * Generic quantile on (0, ∞): find a bracket [a, 2a] by doubling / halving from `hi`, bisect
 * geometrically (so quantiles like 1e-60 are resolved to relative precision), then polish with
 * Newton steps guarded to the bracket. `fn` is the cdf (increasing) or the sf (decreasing,
 * `increasing=false`); inverting the sf directly keeps full precision for tiny tail probabilities.
 */
function invert(
  fn: (x: number) => number,
  pdf: (x: number) => number,
  target: number,
  increasing: boolean,
  hi: number,
): number {
  if (Number.isNaN(target) || target < 0 || target > 1) return NaN
  const left = (x: number) => (increasing ? fn(x) < target : fn(x) > target) // x is left of the root
  let a: number
  let b: number
  if (left(hi)) {
    a = hi
    b = hi * 2
    let grow = 0
    while (left(b) && grow++ < 1100) {
      a = b
      b *= 2
    }
    if (left(b)) return Infinity
  } else {
    b = hi
    a = hi / 2
    let shrink = 0
    while (!left(a) && shrink++ < 1100) {
      b = a
      a /= 2
    }
    if (!left(a)) return 0
  }
  // coarse geometric bisection (to ~1e-4 relative), then bracket-guarded Newton to machine precision
  for (let i = 0; i < 120; i++) {
    const m = Math.sqrt(a * b)
    if (left(m)) a = m
    else b = m
    if (b - a <= 1e-4 * b) break
  }
  let x = Math.sqrt(a * b)
  for (let k = 0; k < 40; k++) {
    const fx = fn(x) - target
    const d = pdf(x)
    // keep the bracket consistent with the sign of the residual
    if (increasing ? fx < 0 : fx > 0) a = x
    else b = x
    let nx = !(d > 0) || !Number.isFinite(d) ? NaN : x - fx / (increasing ? d : -d)
    if (!(nx > a && nx < b)) nx = Math.sqrt(a * b) // Newton left the bracket: bisect instead
    const done = Math.abs(nx - x) <= 2e-16 * Math.abs(nx) || b - a <= 4e-16 * b
    x = nx
    if (done) break
  }
  return x
}

/**
 * Quantile for a distribution on [0, ∞): invert the cdf for p ≤ ½ and the sf (with the exact 1 − p)
 * above, so neither side suffers 1 − tiny cancellation. `upper` flips the roles for isf.
 */
function quantile(
  primary: (x: number) => number,
  complement: (x: number) => number,
  pdf: (x: number) => number,
  p: number,
  hi: number,
  upper = false,
): number {
  if (Number.isNaN(p) || p < 0 || p > 1) return NaN
  if (p === 0) return upper ? Infinity : 0
  if (p === 1) return upper ? 0 : Infinity
  // primary is increasing for ppf (cdf) and decreasing for isf (sf)
  return p <= 0.5 ? invert(primary, pdf, p, !upper, hi) : invert(complement, pdf, 1 - p, upper, hi)
}

function makeMap(self: Distribution) {
  return (fn: 'pdf' | 'cdf' | 'sf' | 'ppf' | 'isf', xs: ArrayLike<number>): Float64Array => {
    const f = self[fn]
    const out = new Float64Array(xs.length)
    for (let i = 0; i < xs.length; i++) out[i] = f(xs[i]!)
    return out
  }
}

// ---- distributions --------------------------------------------------------------------------------

/** Normal N(mean, sd²). */
export function normal(mean = 0, sd = 1): Distribution {
  if (!(sd > 0)) throw new RangeError(`normal: sd must be > 0 (got ${sd})`)
  const z = (x: number) => (x - mean) / sd
  const d: Distribution = {
    name: `normal(${mean}, ${sd})`,
    mean,
    variance: sd * sd,
    pdf: (x) => Math.exp(-0.5 * z(x) * z(x)) / (sd * Math.sqrt(2 * Math.PI)),
    cdf: (x) => 0.5 * erfc(-z(x) / SQRT2),
    sf: (x) => 0.5 * erfc(z(x) / SQRT2),
    ppf: (p) => mean + sd * stdNormalPpf(p),
    isf: (p) => mean - sd * stdNormalPpf(p),
    map: null as unknown as Distribution['map'],
  }
  d.map = makeMap(d)
  return Object.freeze(d)
}

/** Student's t with `df` degrees of freedom. */
export function t(df: number): Distribution {
  if (!(df > 0)) throw new RangeError(`t: df must be > 0 (got ${df})`)
  const lnorm = lgamma((df + 1) / 2) - lgamma(df / 2) - 0.5 * Math.log(df * Math.PI)
  const pdf = (x: number) => Math.exp(lnorm - ((df + 1) / 2) * Math.log1p((x * x) / df))
  // P(T > |x|) = ½ · I_{df/(df+x²)}(df/2, ½)
  const tail = (x: number) => 0.5 * betainc(df / 2, 0.5, df / (df + x * x))
  const cdf = (x: number) => (Number.isNaN(x) ? NaN : x === Infinity ? 1 : x === -Infinity ? 0 : x >= 0 ? 1 - tail(x) : tail(x))
  const sf = (x: number) => (Number.isNaN(x) ? NaN : x === Infinity ? 0 : x === -Infinity ? 1 : x >= 0 ? tail(x) : 1 - tail(x))
  // x > 0 with P(T > x) = q for q ≤ ½; both quantile functions map onto it by symmetry, so the
  // tail probability is never formed as 1 − p except where that subtraction is exact (p ≥ ½).
  const upper = (q: number) => (q === 0.5 ? 0 : invert(tail, pdf, q, false, 2))
  const ppf = (p: number): number => {
    if (Number.isNaN(p) || p < 0 || p > 1) return NaN
    if (p === 0) return -Infinity
    if (p === 1) return Infinity
    return p < 0.5 ? -upper(p) : upper(1 - p)
  }
  const isf = (p: number): number => {
    if (Number.isNaN(p) || p < 0 || p > 1) return NaN
    if (p === 0) return Infinity
    if (p === 1) return -Infinity
    return p < 0.5 ? upper(p) : -upper(1 - p)
  }
  const d: Distribution = {
    name: `t(${df})`,
    mean: df > 1 ? 0 : NaN,
    variance: df > 2 ? df / (df - 2) : df > 1 ? Infinity : NaN,
    pdf,
    cdf,
    sf,
    ppf,
    isf,
    map: null as unknown as Distribution['map'],
  }
  d.map = makeMap(d)
  return Object.freeze(d)
}

/** Chi-square with `k` degrees of freedom. */
export function chi2(k: number): Distribution {
  if (!(k > 0)) throw new RangeError(`chi2: df must be > 0 (got ${k})`)
  const half = k / 2
  const lnorm = -half * Math.LN2 - lgamma(half)
  const pdf = (x: number) => (x < 0 ? 0 : x === 0 ? (k === 2 ? 0.5 : k < 2 ? Infinity : 0) : Math.exp(lnorm + (half - 1) * Math.log(x) - x / 2))
  const cdf = (x: number) => (x <= 0 ? 0 : gammainc(half, x / 2))
  const sf = (x: number) => (x <= 0 ? 1 : gammaincc(half, x / 2))
  const ppf = (p: number) => quantile(cdf, sf, pdf, p, Math.max(1, k))
  const d: Distribution = {
    name: `chi2(${k})`,
    mean: k,
    variance: 2 * k,
    pdf,
    cdf,
    sf,
    ppf,
    isf: (p) => quantile(sf, cdf, pdf, p, Math.max(1, k), true),
    map: null as unknown as Distribution['map'],
  }
  d.map = makeMap(d)
  return Object.freeze(d)
}

/** Fisher–Snedecor F with `d1` (numerator) and `d2` (denominator) degrees of freedom. */
export function f(d1: number, d2: number): Distribution {
  if (!(d1 > 0) || !(d2 > 0)) throw new RangeError(`f: degrees of freedom must be > 0 (got ${d1}, ${d2})`)
  const a = d1 / 2
  const b = d2 / 2
  const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
  const pdf = (x: number) => {
    if (x < 0) return 0
    if (x === 0) return d1 === 2 ? 1 : d1 < 2 ? Infinity : 0
    // ½·[d1 ln(d1 x) + d2 ln d2 − (d1+d2) ln(d1 x + d2)] − ln x − ln B(a, b)
    const lx = a * Math.log(d1 * x) + b * Math.log(d2) - (a + b) * Math.log(d1 * x + d2) - Math.log(x) - lbeta
    return Math.exp(lx)
  }
  const cdf = (x: number) => (x <= 0 ? 0 : x === Infinity ? 1 : betainc(a, b, (d1 * x) / (d1 * x + d2)))
  const sf = (x: number) => (x <= 0 ? 1 : x === Infinity ? 0 : betainc(b, a, d2 / (d2 + d1 * x)))
  const ppf = (p: number) => quantile(cdf, sf, pdf, p, 1)
  const d: Distribution = {
    name: `f(${d1}, ${d2})`,
    mean: d2 > 2 ? d2 / (d2 - 2) : NaN,
    variance: d2 > 4 ? (2 * d2 * d2 * (d1 + d2 - 2)) / (d1 * (d2 - 2) * (d2 - 2) * (d2 - 4)) : NaN,
    pdf,
    cdf,
    sf,
    ppf,
    isf: (p) => quantile(sf, cdf, pdf, p, 1, true),
    map: null as unknown as Distribution['map'],
  }
  d.map = makeMap(d)
  return Object.freeze(d)
}

/** Beta(a, b) on (0, 1). */
export function beta(a: number, b: number): Distribution {
  if (!(a > 0) || !(b > 0)) throw new RangeError(`beta: shape parameters must be > 0 (got ${a}, ${b})`)
  const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
  const pdf = (x: number) => (x < 0 || x > 1 ? 0 : x === 0 ? (a < 1 ? Infinity : a === 1 ? Math.exp(-lbeta) : 0) : x === 1 ? (b < 1 ? Infinity : b === 1 ? Math.exp(-lbeta) : 0) : Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lbeta))
  const cdf = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : betainc(a, b, x))
  const sf = (x: number) => (x <= 0 ? 1 : x >= 1 ? 0 : betainc(b, a, 1 - x))
  const ppf = (q: number) => {
    if (Number.isNaN(q) || q < 0 || q > 1) return NaN
    if (q === 0) return 0
    if (q === 1) return 1
    // upper tail: invert the mirrored distribution so both tails get the log-space treatment
    if (q > 0.5) return 1 - betaLowerPpf(b, a, 1 - q, lbeta)
    return betaLowerPpf(a, b, q, lbeta)
  }
  const d: Distribution = {
    name: `beta(${a}, ${b})`,
    mean: a / (a + b),
    variance: (a * b) / ((a + b) ** 2 * (a + b + 1)),
    pdf,
    cdf,
    sf,
    ppf,
    isf: (q) => ppf(1 - q),
    map: null as unknown as Distribution['map'],
  }
  d.map = makeMap(d)
  return Object.freeze(d)
}

/**
 * Lower-tail beta quantile. Bisection on log x reaches quantiles like 1e-115 (tiny shapes), which a linear
 * bisection on [0, 1] cannot: its resolution floor is ~1e-16 and it returned that instead of the answer.
 */
function betaLowerPpf(a: number, b: number, q: number, lbeta: number): number {
  const cdf = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : betainc(a, b, x))
  // leading-order guess x0 = (q · a · B(a, b))^(1/a); bracket it geometrically in log space
  let lgx = (Math.log(q) + Math.log(a) + lbeta) / a
  if (!Number.isFinite(lgx)) lgx = -1
  let lo = Math.min(lgx - 2, -1)
  let hi = 0
  for (let i = 0; i < 2000 && cdf(Math.exp(lo)) >= q; i++) lo -= 2
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi)
    if (cdf(Math.exp(mid)) < q) lo = mid
    else hi = mid
    if (hi - lo < 1e-15) break
  }
  let x = Math.exp(0.5 * (lo + hi))
  // Newton polish on x with the pdf (guarded to the bracket)
  const pdf = (v: number) => Math.exp((a - 1) * Math.log(v) + (b - 1) * Math.log1p(-v) - lbeta)
  for (let k = 0; k < 30; k++) {
    const fx = cdf(x) - q
    const d = pdf(x)
    if (!(d > 0) || !Number.isFinite(d)) break
    const nx = x - fx / d
    if (!(nx > Math.exp(lo) && nx < Math.exp(hi))) break
    if (Math.abs(nx - x) <= 1e-16 * Math.abs(x)) {
      x = nx
      break
    }
    x = nx
  }
  return x
}

/** Gamma(shape k, scale θ). */
export function gamma(shape: number, scale = 1): Distribution {
  if (!(shape > 0) || !(scale > 0)) throw new RangeError(`gamma: shape and scale must be > 0 (got ${shape}, ${scale})`)
  const lg = lgamma(shape)
  const pdf = (x: number) => (x < 0 ? 0 : x === 0 ? (shape < 1 ? Infinity : shape === 1 ? 1 / scale : 0) : Math.exp((shape - 1) * Math.log(x / scale) - x / scale - lg) / scale)
  const cdf = (x: number) => (x <= 0 ? 0 : gammainc(shape, x / scale))
  const sf = (x: number) => (x <= 0 ? 1 : gammaincc(shape, x / scale))
  const ppf = (q: number) => quantile(cdf, sf, pdf, q, shape * scale + 3 * Math.sqrt(shape) * scale)
  const d: Distribution = {
    name: `gamma(${shape}, ${scale})`,
    mean: shape * scale,
    variance: shape * scale * scale,
    pdf,
    cdf,
    sf,
    ppf,
    isf: (q) => quantile(sf, cdf, pdf, q, shape * scale + 3 * Math.sqrt(shape) * scale, true),
    map: null as unknown as Distribution['map'],
  }
  d.map = makeMap(d)
  return Object.freeze(d)
}

/** Weibull(shape β, scale η) with optional threshold. */
export function weibull(shape: number, scale: number, threshold = 0): Distribution {
  if (!(shape > 0) || !(scale > 0)) throw new RangeError(`weibull: shape and scale must be > 0 (got ${shape}, ${scale})`)
  const z = (x: number) => (x - threshold) / scale
  const pdf = (x: number) => (x <= threshold ? (x === threshold && shape === 1 ? 1 / scale : 0) : (shape / scale) * z(x) ** (shape - 1) * Math.exp(-(z(x) ** shape)))
  const cdf = (x: number) => (x <= threshold ? 0 : -Math.expm1(-(z(x) ** shape)))
  const sf = (x: number) => (x <= threshold ? 1 : Math.exp(-(z(x) ** shape)))
  const ppf = (q: number) => (Number.isNaN(q) || q < 0 || q > 1 ? NaN : threshold + scale * (-Math.log1p(-q)) ** (1 / shape))
  const g1 = Math.exp(lgamma(1 + 1 / shape))
  const g2 = Math.exp(lgamma(1 + 2 / shape))
  const d: Distribution = {
    name: `weibull(${shape}, ${scale}${threshold ? `, ${threshold}` : ''})`,
    mean: threshold + scale * g1,
    variance: scale * scale * (g2 - g1 * g1),
    pdf,
    cdf,
    sf,
    ppf,
    isf: (q) => (Number.isNaN(q) || q < 0 || q > 1 ? NaN : threshold + scale * (-Math.log(q)) ** (1 / shape)),
    map: null as unknown as Distribution['map'],
  }
  d.map = makeMap(d)
  return Object.freeze(d)
}

/** Lognormal: ln X ~ N(mu, sigma²), with optional threshold. */
export function lognormal(mu = 0, sigma = 1, threshold = 0): Distribution {
  if (!(sigma > 0)) throw new RangeError(`lognormal: sigma must be > 0 (got ${sigma})`)
  const base = normal(mu, sigma)
  const pdf = (x: number) => (x <= threshold ? 0 : base.pdf(Math.log(x - threshold)) / (x - threshold))
  const cdf = (x: number) => (x <= threshold ? 0 : base.cdf(Math.log(x - threshold)))
  const sf = (x: number) => (x <= threshold ? 1 : base.sf(Math.log(x - threshold)))
  const d: Distribution = {
    name: `lognormal(${mu}, ${sigma}${threshold ? `, ${threshold}` : ''})`,
    mean: threshold + Math.exp(mu + (sigma * sigma) / 2),
    variance: (Math.exp(sigma * sigma) - 1) * Math.exp(2 * mu + sigma * sigma),
    pdf,
    cdf,
    sf,
    ppf: (q) => threshold + Math.exp(base.ppf(q)),
    isf: (q) => threshold + Math.exp(base.isf(q)),
    map: null as unknown as Distribution['map'],
  }
  d.map = makeMap(d)
  return Object.freeze(d)
}

/** Exponential with scale (mean) θ and optional threshold. */
export function exponential(scale = 1, threshold = 0): Distribution {
  return weibull(1, scale, threshold)
}

/** Logistic(location, scale). */
export function logistic(location = 0, scale = 1): Distribution {
  if (!(scale > 0)) throw new RangeError(`logistic: scale must be > 0 (got ${scale})`)
  const z = (x: number) => (x - location) / scale
  const pdf = (x: number) => {
    const e = Math.exp(-Math.abs(z(x)))
    return e / (scale * (1 + e) ** 2)
  }
  const cdf = (x: number) => 1 / (1 + Math.exp(-z(x)))
  const sf = (x: number) => 1 / (1 + Math.exp(z(x)))
  const d: Distribution = {
    name: `logistic(${location}, ${scale})`,
    mean: location,
    variance: (scale * scale * Math.PI * Math.PI) / 3,
    pdf,
    cdf,
    sf,
    ppf: (q) => location + scale * Math.log(q / (1 - q)),
    isf: (q) => location - scale * Math.log(q / (1 - q)),
    map: null as unknown as Distribution['map'],
  }
  d.map = makeMap(d)
  return Object.freeze(d)
}

/** Smallest extreme value (Gumbel minimum) with location μ and scale σ — the log of a Weibull. */
export function smallestExtremeValue(location = 0, scale = 1): Distribution {
  if (!(scale > 0)) throw new RangeError(`smallestExtremeValue: scale must be > 0 (got ${scale})`)
  const z = (x: number) => (x - location) / scale
  const pdf = (x: number) => Math.exp(z(x) - Math.exp(z(x))) / scale
  const cdf = (x: number) => -Math.expm1(-Math.exp(z(x)))
  const sf = (x: number) => Math.exp(-Math.exp(z(x)))
  const EULER = 0.5772156649015329
  const d: Distribution = {
    name: `sev(${location}, ${scale})`,
    mean: location - EULER * scale,
    variance: (scale * scale * Math.PI * Math.PI) / 6,
    pdf,
    cdf,
    sf,
    ppf: (q) => location + scale * Math.log(-Math.log1p(-q)),
    isf: (q) => location + scale * Math.log(-Math.log(q)),
    map: null as unknown as Distribution['map'],
  }
  d.map = makeMap(d)
  return Object.freeze(d)
}

/** Gumbel (maximum / Type I extreme value), scipy gumbel_r. */
export function gumbel(location = 0, scale = 1): Distribution {
  if (!(scale > 0)) throw new RangeError(`gumbel: scale must be > 0`)
  const z = (x: number) => (x - location) / scale
  const EULER = 0.5772156649015329
  const d: Distribution = {
    name: `gumbel(${location}, ${scale})`,
    mean: location + EULER * scale,
    variance: (scale * scale * Math.PI * Math.PI) / 6,
    pdf: (x) => Math.exp(-(z(x) + Math.exp(-z(x)))) / scale,
    cdf: (x) => Math.exp(-Math.exp(-z(x))),
    sf: (x) => -Math.expm1(-Math.exp(-z(x))),
    ppf: (q) => location - scale * Math.log(-Math.log(q)),
    isf: (q) => location - scale * Math.log(-Math.log1p(-q)),
    map: null as unknown as Distribution['map'],
  }
  d.map = makeMap(d)
  return Object.freeze(d)
}

/** Pareto Type I with shape b and scale xm (scipy pareto). */
export function pareto(b: number, scale = 1): Distribution {
  if (!(b > 0) || !(scale > 0)) throw new RangeError('pareto: b and scale must be > 0')
  const d: Distribution = {
    name: `pareto(${b}, ${scale})`,
    mean: b > 1 ? (b * scale) / (b - 1) : Infinity,
    variance: b > 2 ? (b * scale * scale) / ((b - 1) ** 2 * (b - 2)) : Infinity,
    pdf: (x) => (x < scale ? 0 : (b * scale ** b) / x ** (b + 1)),
    cdf: (x) => (x < scale ? 0 : 1 - (scale / x) ** b),
    sf: (x) => (x < scale ? 1 : (scale / x) ** b),
    ppf: (q) => scale / (1 - q) ** (1 / b),
    isf: (q) => scale / q ** (1 / b),
    map: null as unknown as Distribution['map'],
  }
  d.map = makeMap(d)
  return Object.freeze(d)
}

/** Inverse Gaussian (Wald) with mean μ and shape λ (scipy invgauss; parameterization μ, λ). */
export function invgauss(mu: number, lambda = 1): Distribution {
  if (!(mu > 0) || !(lambda > 0)) throw new RangeError('invgauss: mu and lambda must be > 0')
  const STD = normal()
  const pdf = (x: number) => {
    if (!(x > 0)) return 0
    return Math.sqrt(lambda / (2 * Math.PI * x ** 3)) * Math.exp((-lambda * (x - mu) ** 2) / (2 * mu * mu * x))
  }
  const cdf = (x: number) => {
    if (!(x > 0)) return 0
    const a = Math.sqrt(lambda / x) * (x / mu - 1)
    const b = Math.sqrt(lambda / x) * (x / mu + 1)
    return STD.cdf(a) + Math.exp((2 * lambda) / mu) * STD.cdf(-b)
  }
  const sf = (x: number) => 1 - cdf(x)
  // PPF via bisection
  const ppf = (q: number) => {
    if (q <= 0) return 0
    if (q >= 1) return Infinity
    let lo = 1e-12
    let hi = mu * 20
    while (cdf(hi) < q) hi *= 2
    for (let i = 0; i < 80; i++) {
      const mid = 0.5 * (lo + hi)
      if (cdf(mid) < q) lo = mid
      else hi = mid
    }
    return 0.5 * (lo + hi)
  }
  const d: Distribution = {
    name: `invgauss(${mu}, ${lambda})`,
    mean: mu,
    variance: (mu ** 3) / lambda,
    pdf,
    cdf,
    sf,
    ppf,
    isf: (q) => ppf(1 - q),
    map: null as unknown as Distribution['map'],
  }
  d.map = makeMap(d)
  return Object.freeze(d)
}

// ---- discrete distributions -----------------------------------------------------------------------

export interface DiscreteDistribution {
  readonly name: string
  /** P(X = k). */
  pmf(k: number): number
  /** P(X ≤ k). */
  cdf(k: number): number
  /** P(X > k). */
  sf(k: number): number
  /** Smallest k with P(X ≤ k) ≥ p. */
  ppf(p: number): number
  readonly mean: number
  readonly variance: number
}

/** Binomial(n, p). cdf via the regularized incomplete beta: P(X ≤ k) = I_{1−p}(n − k, k + 1). */
export function binomial(n: number, p: number): DiscreteDistribution {
  if (!(Number.isInteger(n) && n >= 0)) throw new RangeError(`binomial: n must be a non-negative integer (got ${n})`)
  if (!(p >= 0 && p <= 1)) throw new RangeError(`binomial: p must be in [0, 1] (got ${p})`)
  const lnC = (k: number) => lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
  const pmf = (k: number) => {
    if (!Number.isInteger(k) || k < 0 || k > n) return 0
    if (p === 0) return k === 0 ? 1 : 0
    if (p === 1) return k === n ? 1 : 0
    return Math.exp(lnC(k) + k * Math.log(p) + (n - k) * Math.log1p(-p))
  }
  const cdf = (k: number) => {
    k = Math.floor(k)
    if (k < 0) return 0
    if (k >= n) return 1
    if (p === 0) return 1
    if (p === 1) return 0
    return betainc(n - k, k + 1, 1 - p)
  }
  const sf = (k: number) => {
    k = Math.floor(k)
    if (k < 0) return 1
    if (k >= n) return 0
    if (p === 0) return 0
    if (p === 1) return 1
    return betainc(k + 1, n - k, p)
  }
  const ppf = (q: number) => {
    if (Number.isNaN(q) || q < 0 || q > 1) return NaN
    if (q === 0) return 0
    if (q === 1) return n
    let k = Math.max(0, Math.min(n, Math.round(n * p + Math.sqrt(n * p * (1 - p)) * stdNormalPpf(q))))
    while (k > 0 && cdf(k - 1) >= q) k--
    while (k < n && cdf(k) < q) k++
    return k
  }
  return Object.freeze({ name: `binomial(${n}, ${p})`, pmf, cdf, sf, ppf, mean: n * p, variance: n * p * (1 - p) })
}

/** Poisson(λ). cdf via the regularized upper incomplete gamma: P(X ≤ k) = Q(k + 1, λ). */
export function poisson(lambda: number): DiscreteDistribution {
  if (!(lambda >= 0)) throw new RangeError(`poisson: lambda must be ≥ 0 (got ${lambda})`)
  const pmf = (k: number) => {
    if (!Number.isInteger(k) || k < 0) return 0
    if (lambda === 0) return k === 0 ? 1 : 0
    return Math.exp(k * Math.log(lambda) - lambda - lgamma(k + 1))
  }
  const cdf = (k: number) => {
    k = Math.floor(k)
    if (k < 0) return 0
    if (lambda === 0) return 1
    return gammaincc(k + 1, lambda)
  }
  const sf = (k: number) => {
    k = Math.floor(k)
    if (k < 0) return 1
    if (lambda === 0) return 0
    return gammainc(k + 1, lambda)
  }
  const ppf = (q: number) => {
    if (Number.isNaN(q) || q < 0 || q > 1) return NaN
    if (q === 0) return 0
    if (q === 1) return Infinity
    let k = Math.max(0, Math.round(lambda + Math.sqrt(lambda) * stdNormalPpf(q)))
    while (k > 0 && cdf(k - 1) >= q) k--
    while (cdf(k) < q) k++
    return k
  }
  return Object.freeze({ name: `poisson(${lambda})`, pmf, cdf, sf, ppf, mean: lambda, variance: lambda })
}

/** Hypergeometric pmf: P(X = k) drawing n from N with K successes (log-gamma based). */
export function hypergeomPmf(k: number, N: number, K: number, n: number): number {
  if (k < Math.max(0, n + K - N) || k > Math.min(K, n)) return 0
  const lnC = (a: number, b: number) => lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)
  return Math.exp(lnC(K, k) + lnC(N - K, n - k) - lnC(N, n))
}

/** Hypergeometric(N, K, n) as DiscreteDistribution — population N, K success states, draw n. */
export function hypergeometric(N: number, K: number, n: number): DiscreteDistribution {
  if (![N, K, n].every((v) => Number.isInteger(v) && v >= 0)) throw new RangeError('hypergeometric: N,K,n must be non-negative integers')
  if (K > N || n > N) throw new RangeError('hypergeometric: K and n must be ≤ N')
  const lo = Math.max(0, n - (N - K))
  const hi = Math.min(n, K)
  const pmf = (k: number) => hypergeomPmf(k, N, K, n)
  const cdf = (k: number) => {
    k = Math.floor(k)
    if (k < lo) return 0
    if (k >= hi) return 1
    let s = 0
    for (let i = lo; i <= k; i++) s += pmf(i)
    return s
  }
  const sf = (k: number) => 1 - cdf(k)
  const ppf = (p: number) => {
    if (Number.isNaN(p) || p < 0 || p > 1) return NaN
    if (p === 0) return lo
    if (p === 1) return hi
    let s = 0
    for (let k = lo; k <= hi; k++) {
      s += pmf(k)
      if (s >= p) return k
    }
    return hi
  }
  return Object.freeze({
    name: `hypergeometric(${N},${K},${n})`,
    pmf,
    cdf,
    sf,
    ppf,
    mean: (n * K) / N,
    variance: N > 1 ? n * (K / N) * (1 - K / N) * ((N - n) / (N - 1)) : 0,
  })
}

/** Negative binomial (number of failures before `n` successes), scipy nbinom(n, p). */
export function negativeBinomial(n: number, p: number): DiscreteDistribution {
  if (!(n > 0)) throw new RangeError('negativeBinomial: n (successes) must be > 0')
  if (!(p > 0 && p <= 1)) throw new RangeError('negativeBinomial: p must be in (0,1]')
  const lnC = (k: number) => lgamma(k + n) - lgamma(n) - lgamma(k + 1)
  const pmf = (k: number) => {
    if (!Number.isInteger(k) || k < 0) return 0
    if (p === 1) return k === 0 ? 1 : 0
    return Math.exp(lnC(k) + n * Math.log(p) + k * Math.log1p(-p))
  }
  const cdf = (k: number) => {
    k = Math.floor(k)
    if (k < 0) return 0
    if (p === 1) return 1
    // P(X≤k) = I_p(n, k+1)
    return betainc(n, k + 1, p)
  }
  const sf = (k: number) => 1 - cdf(k)
  const ppf = (q: number) => {
    if (Number.isNaN(q) || q < 0 || q > 1) return NaN
    if (q === 0) return 0
    if (q === 1) return Infinity
    let lo = 0
    let hi = Math.max(10, Math.ceil((n * (1 - p)) / p * 4))
    while (cdf(hi) < q) hi *= 2
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (cdf(mid) < q) lo = mid + 1
      else hi = mid
    }
    return lo
  }
  return Object.freeze({
    name: `nbinom(${n}, ${p})`,
    pmf,
    cdf,
    sf,
    ppf,
    mean: (n * (1 - p)) / p,
    variance: (n * (1 - p)) / (p * p),
  })
}

// ---- noncentral distributions -----------------------------------------------------------------------

/**
 * Cdf of the noncentral t (AS 243, Lenth 1989): P(T ≤ x | df, δ) as a series of incomplete beta
 * terms with Poisson weights; x < 0 by the symmetry T(δ) ~ −T(−δ). Used for the power of t-tests
 * (Minitab Power and Sample Size).
 */
export function nctCdf(x: number, df: number, delta: number): number {
  if (Number.isNaN(x) || !(df > 0)) return NaN
  if (x === Infinity) return 1
  if (x === -Infinity) return 0
  if (delta === 0) return t(df).cdf(x)
  const negative = x < 0
  const tt = negative ? -x : x
  const del = negative ? -delta : delta
  let tnc = 0
  if (tt > 0) {
    const x2 = (tt * tt) / (tt * tt + df)
    const lambda = (del * del) / 2
    let p = 0.5 * Math.exp(-lambda)
    let q = Math.sqrt(2 / Math.PI) * p * del
    let s = 0.5 - p
    let a = 0.5
    const b = 0.5 * df
    const rxb = Math.pow(1 - x2, b)
    const albeta = 0.5 * Math.log(Math.PI) + lgamma(b) - lgamma(0.5 + b)
    let xodd = betainc(a, b, x2)
    let godd = 2 * rxb * Math.exp(a * Math.log(x2) - albeta)
    let xeven = 1 - rxb
    let geven = b * x2 * rxb
    tnc = p * xodd + q * xeven
    for (let it = 1; it <= 2000; it++) {
      a += 1
      xodd -= godd
      xeven -= geven
      godd *= (x2 * (a + b - 1)) / a
      geven *= (x2 * (a + b - 0.5)) / (a + 0.5)
      p *= lambda / it
      q *= lambda / (it + 0.5)
      s -= p
      tnc += p * xodd + q * xeven
      const errbd = 2 * s * (xodd - godd)
      if (Math.abs(errbd) < 1e-14) break
    }
  }
  tnc += normal().cdf(-del)
  const result = negative ? 1 - tnc : tnc
  return Math.min(1, Math.max(0, result))
}

/** Poisson(λ/2)-weighted mixture Σⱼ wⱼ·term(j), summed outward from the modal weight. */
function poissonMixture(lambda: number, term: (j: number) => number): number {
  const half = lambda / 2
  const j0 = Math.floor(half)
  const logW = (j: number) => -half + j * Math.log(half) - lgamma(j + 1)
  let total = 0
  for (let j = j0; j >= 0; j--) {
    const w = Math.exp(logW(j))
    if (w < 1e-18 && j < j0) break
    total += w * term(j)
  }
  for (let j = j0 + 1; j < j0 + 100000; j++) {
    const w = Math.exp(logW(j))
    if (w < 1e-18) break
    total += w * term(j)
  }
  return Math.min(1, Math.max(0, total))
}

/** Cdf of the noncentral F(d1, d2, λ): Σⱼ Poisson(λ/2)ⱼ · I_y(d1/2 + j, d2/2), y = d1x/(d1x + d2). */
export function ncfCdf(x: number, d1: number, d2: number, lambda: number): number {
  if (Number.isNaN(x) || !(d1 > 0) || !(d2 > 0) || !(lambda >= 0)) return NaN
  if (x <= 0) return 0
  if (x === Infinity) return 1
  if (lambda === 0) return f(d1, d2).cdf(x)
  const y = (d1 * x) / (d1 * x + d2)
  return poissonMixture(lambda, (j) => betainc(d1 / 2 + j, d2 / 2, y))
}

/** Cdf of the noncentral χ²(k, λ): Σⱼ Poisson(λ/2)ⱼ · P(k/2 + j, x/2). */
export function ncChi2Cdf(x: number, k: number, lambda: number): number {
  if (Number.isNaN(x) || !(k > 0) || !(lambda >= 0)) return NaN
  if (x <= 0) return 0
  if (x === Infinity) return 1
  if (lambda === 0) return chi2(k).cdf(x)
  return poissonMixture(lambda, (j) => gammainc(k / 2 + j, x / 2))
}

export const dist = {
  normal,
  t,
  chi2,
  f,
  beta,
  gamma,
  weibull,
  lognormal,
  exponential,
  logistic,
  smallestExtremeValue,
  gumbel,
  pareto,
  invgauss,
  binomial,
  poisson,
  negativeBinomial,
  hypergeometric,
  hypergeomPmf,
  nctCdf,
  ncfCdf,
  ncChi2Cdf,
  lgamma,
  gammainc,
  gammaincc,
  betainc,
  erf,
  erfc,
}
export type Dist = typeof dist
