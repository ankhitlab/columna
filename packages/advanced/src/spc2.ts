/**
 * Rare-event and multivariate control charts (Minitab Stat › Control Charts › Rare Event Charts and
 * Multivariate Charts): G chart (opportunities between events), T chart (time between events),
 * Hotelling T² (individuals and subgroups, phase I / II limits), MEWMA and the generalized variance chart.
 */
import { beta as betaDist, f as fDist, normal } from './dist.js'
import { cholesky, choleskySolve, inverse, matrix, type Matrix } from './linalg.js'
import { weibullFit } from './capability.js'
import { cleanNumbers } from './tests.js'

const STD = normal()

export interface RareEventPoint {
  index: number
  value: number
  beyondLimits: boolean
}

export interface RareEventChartResult {
  type: 'g' | 't'
  center: number
  ucl: number
  lcl: number
  /** Geometric p̂ (G chart) or Weibull shape / scale (T chart). */
  parameters: Record<string, number>
  points: RareEventPoint[]
  outOfControl: number[]
  /** Benneyan's test: probability limits at these tail probabilities. */
  tails: [number, number]
}

/**
 * G chart (Minitab): number of opportunities between rare events modelled as geometric with
 * p̂ = 1/(ḡ + 1) (or Benneyan's MVUE p̂ = (n − 1)/(n(ḡ + 1) − 1) with `estimator: 'mvue'`).
 * Probability limits at the 0.00135 / 0.99865 quantiles (default) or 3σ limits with `limits: 'sigma'`.
 */
export function gChart(
  x: ArrayLike<number | null | undefined>,
  options: { estimator?: 'mle' | 'mvue'; limits?: 'probability' | 'sigma'; alpha?: number; p?: number } = {},
): RareEventChartResult {
  const v = cleanNumbers(x)
  const n = v.length
  if (n < 2) throw new RangeError('gChart needs at least 2 counts')
  for (const g of v) if (!(Number.isInteger(g) && g >= 0)) throw new RangeError('gChart: counts must be non-negative integers')
  let gbar = 0
  for (const g of v) gbar += g
  gbar /= n
  const p = options.p ?? (options.estimator === 'mvue' ? (n - 1) / (n * (gbar + 1) - 1) : 1 / (gbar + 1))
  const alpha = options.alpha ?? 0.0027
  let ucl: number
  let lcl: number
  const center = (1 - p) / p
  if (options.limits === 'sigma') {
    const sigma = Math.sqrt((1 - p) / (p * p))
    ucl = center + 3 * sigma
    lcl = Math.max(0, center - 3 * sigma)
  } else {
    // geometric (failures before the first success) quantiles
    const q = (prob: number) => Math.max(0, Math.ceil(Math.log(1 - prob) / Math.log(1 - p) - 1))
    ucl = q(1 - alpha / 2)
    lcl = q(alpha / 2)
  }
  const points: RareEventPoint[] = Array.from(v, (g, i) => ({ index: i, value: g, beyondLimits: g > ucl || g < lcl }))
  return { type: 'g', center, ucl, lcl, parameters: { p, gbar }, points, outOfControl: points.filter((pt) => pt.beyondLimits).map((pt) => pt.index), tails: [alpha / 2, 1 - alpha / 2] }
}

/**
 * T chart (Minitab): time between events fitted with a Weibull (default) or exponential distribution;
 * center line at the median, probability limits at the 0.00135 / 0.99865 quantiles.
 */
export function tChart(x: ArrayLike<number | null | undefined>, options: { distribution?: 'weibull' | 'exponential'; alpha?: number } = {}): RareEventChartResult {
  const v = cleanNumbers(x)
  const n = v.length
  if (n < 3) throw new RangeError('tChart needs at least 3 times')
  for (const t of v) if (!(t > 0)) throw new RangeError('tChart: times must be positive')
  const alpha = options.alpha ?? 0.0027
  let shape: number
  let scale: number
  if (options.distribution === 'exponential') {
    shape = 1
    let s = 0
    for (const t of v) s += t
    scale = s / n
  } else {
    const w = weibullFit(v)
    shape = w.shape
    scale = w.scale
  }
  const q = (prob: number) => scale * (-Math.log(1 - prob)) ** (1 / shape)
  const ucl = q(1 - alpha / 2)
  const lcl = q(alpha / 2)
  const center = q(0.5)
  const points: RareEventPoint[] = Array.from(v, (t, i) => ({ index: i, value: t, beyondLimits: t > ucl || t < lcl }))
  return { type: 't', center, ucl, lcl, parameters: { shape, scale }, points, outOfControl: points.filter((pt) => pt.beyondLimits).map((pt) => pt.index), tails: [alpha / 2, 1 - alpha / 2] }
}

// ---- multivariate ----------------------------------------------------------------------------------------------

export interface T2ChartResult {
  type: 'T2'
  /** Number of variables p, subgroup size m (1 for individuals), number of (sub)groups k. */
  p: number
  m: number
  k: number
  phase: 1 | 2
  ucl: number
  mean: number[]
  covariance: Matrix
  t2: number[]
  outOfControl: number[]
  alpha: number
  /** Decomposition of each out-of-control T² into per-variable contributions (T² − T²₋ⱼ). */
  contributions: Array<{ index: number; byVariable: number[] }>
}

function meanCov(rows: number[][]): { mean: number[]; cov: Matrix } {
  const n = rows.length
  const p = rows[0]!.length
  const mean = new Array<number>(p).fill(0)
  for (const r of rows) for (let j = 0; j < p; j++) mean[j]! += r[j]! / n
  const cov = matrix(p, p)
  for (const r of rows) for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) cov.data[a * p + b] += ((r[a]! - mean[a]!) * (r[b]! - mean[b]!)) / (n - 1)
  return { mean, cov }
}

function quadForm(v: number[], L: Matrix): number {
  const y = choleskySolve(L, v)
  let s = 0
  for (let j = 0; j < v.length; j++) s += v[j]! * y[j]!
  return s
}

/**
 * Hotelling T² chart (Minitab Multivariate Charts › T²): rows of observations (optionally in subgroups of
 * size m via `subgroup`). Phase I (limits from the same data) uses the Beta distribution for individuals
 * and the F distribution for subgroups; phase II (`phase: 2`) uses F limits for future observations.
 */
export function t2Chart(
  data: ArrayLike<ArrayLike<number>>,
  options: { subgroup?: ArrayLike<unknown>; phase?: 1 | 2; alpha?: number; mean?: number[]; covariance?: Matrix } = {},
): T2ChartResult {
  const rows = Array.from(data).map((r) => Array.from(r))
  const n = rows.length
  if (n < 3) throw new RangeError('t2Chart needs at least 3 observations')
  const p = rows[0]!.length
  const alpha = options.alpha ?? 0.00135
  const phase = options.phase ?? 1
  let groups: number[][][]
  if (options.subgroup) {
    const map = new Map<string, number[][]>()
    for (let i = 0; i < n; i++) {
      const k = String(options.subgroup[i])
      let g = map.get(k)
      if (!g) map.set(k, (g = []))
      g.push(rows[i]!)
    }
    groups = [...map.values()]
  } else groups = rows.map((r) => [r])
  const k = groups.length
  const m = groups[0]!.length
  if (groups.some((g) => g.length !== m)) throw new RangeError('t2Chart: subgroups must have equal size')
  let mean: number[]
  let cov: Matrix
  if (m === 1) {
    const mc = options.mean && options.covariance ? { mean: options.mean, cov: options.covariance } : meanCov(rows)
    mean = mc.mean
    cov = mc.cov
  } else {
    // grand mean and pooled within-subgroup covariance
    mean = new Array<number>(p).fill(0)
    cov = matrix(p, p)
    for (const g of groups) {
      const mc = meanCov(g)
      for (let j = 0; j < p; j++) mean[j]! += mc.mean[j]! / k
      for (let i = 0; i < p * p; i++) cov.data[i] += mc.cov.data[i]! / k
    }
    if (options.mean && options.covariance) {
      mean = options.mean
      cov = options.covariance
    }
  }
  const L = cholesky(cov)
  if (!L) throw new RangeError('t2Chart: covariance matrix is not positive definite')
  const t2 = groups.map((g) => {
    const gm = new Array<number>(p).fill(0)
    for (const r of g) for (let j = 0; j < p; j++) gm[j]! += r[j]! / m
    const d = gm.map((v, j) => v - mean[j]!)
    return m * quadForm(d, L)
  })
  let ucl: number
  if (m === 1) {
    ucl = phase === 1 ? ((k - 1) ** 2 / k) * betaDist(p / 2, (k - p - 1) / 2).ppf(1 - alpha) : ((p * (k + 1) * (k - 1)) / (k * (k - p))) * fDist(p, k - p).ppf(1 - alpha)
  } else {
    const df2 = k * m - k - p + 1
    ucl = phase === 1 ? ((p * (m - 1) * (k - 1)) / df2) * fDist(p, df2).ppf(1 - alpha) : ((p * (k + 1) * (m - 1)) / df2) * fDist(p, df2).ppf(1 - alpha)
  }
  const outOfControl = t2.map((v, i) => (v > ucl ? i : -1)).filter((i) => i >= 0)
  // contributions by dropping each variable (Runger–Alt–Montgomery decomposition)
  const contributions = outOfControl.map((i) => {
    const g = groups[i]!
    const gm = new Array<number>(p).fill(0)
    for (const r of g) for (let j = 0; j < p; j++) gm[j]! += r[j]! / m
    const byVariable = Array.from({ length: p }, (_, drop) => {
      const keep = Array.from({ length: p }, (_, j) => j).filter((j) => j !== drop)
      if (!keep.length) return t2[i]!
      const sub = matrix(keep.length, keep.length)
      keep.forEach((a, ai) => keep.forEach((b, bi) => (sub.data[ai * keep.length + bi] = cov.data[a * p + b]!)))
      const d = keep.map((j) => gm[j]! - mean[j]!)
      const Ls = cholesky(sub)
      return t2[i]! - (Ls ? m * quadForm(d, Ls) : 0)
    })
    return { index: i, byVariable }
  })
  return { type: 'T2', p, m, k, phase, ucl, mean, covariance: cov, t2, outOfControl, alpha, contributions }
}

export interface MewmaResult {
  type: 'MEWMA'
  lambda: number
  h: number
  t2: number[]
  z: number[][]
  outOfControl: number[]
  mean: number[]
  covariance: Matrix
  /** In-control ARL targeted when h was derived by simulation. */
  arl0?: number
}

/** Seeded in-control ARL of a MEWMA scheme (multivariate standard normal), used to calibrate h. */
function mewmaArl(p: number, lambda: number, h: number, runs: number, seed: number): number {
  let a = seed >>> 0
  const u = () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296
  }
  const normalRv = () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u())
  let total = 0
  const maxRun = 5000
  for (let r = 0; r < runs; r++) {
    const z = new Float64Array(p)
    let i = 0
    for (i = 1; i <= maxRun; i++) {
      let q = 0
      for (let j = 0; j < p; j++) {
        z[j] = lambda * normalRv() + (1 - lambda) * z[j]!
        q += z[j]! * z[j]!
      }
      const varZ = (lambda / (2 - lambda)) * (1 - (1 - lambda) ** (2 * i))
      if (q / varZ > h) break
    }
    total += i
  }
  return total / runs
}

/**
 * MEWMA chart (Lowry et al. 1992): Zᵢ = λxᵢ + (1 − λ)Zᵢ₋₁, T²ᵢ = Zᵢᵀ Σ_Zᵢ⁻¹ Zᵢ with the exact
 * time-varying covariance. `h` defaults to the value giving in-control ARL₀ = 200 (Minitab's default),
 * found by seeded simulation when not supplied.
 */
export function mewma(
  data: ArrayLike<ArrayLike<number>>,
  options: { lambda?: number; h?: number; arl0?: number; mean?: number[]; covariance?: Matrix; phase1?: number } = {},
): MewmaResult {
  const rows = Array.from(data).map((r) => Array.from(r))
  const n = rows.length
  const p = rows[0]!.length
  const lambda = options.lambda ?? 0.1
  if (!(lambda > 0 && lambda <= 1)) throw new RangeError('mewma: lambda must be in (0, 1]')
  const base = options.phase1 ? rows.slice(0, options.phase1) : rows
  const mc = options.mean && options.covariance ? { mean: options.mean, cov: options.covariance } : meanCov(base)
  const Sinv = inverse(mc.cov)
  let h = options.h
  let arl0: number | undefined
  if (h === undefined) {
    arl0 = options.arl0 ?? 200
    // bisection on h for ARL0 (coarse but deterministic)
    let lo = 1
    let hi = 60
    for (let it = 0; it < 14; it++) {
      const mid = 0.5 * (lo + hi)
      const arl = mewmaArl(p, lambda, mid, 400, 0x51ab + it)
      if (arl < arl0) lo = mid
      else hi = mid
    }
    h = 0.5 * (lo + hi)
  }
  const z = new Array<number>(p).fill(0)
  const t2: number[] = []
  const zs: number[][] = []
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) z[j] = lambda * (rows[i]![j]! - mc.mean[j]!) + (1 - lambda) * z[j]!
    const varZ = (lambda / (2 - lambda)) * (1 - (1 - lambda) ** (2 * (i + 1)))
    let q = 0
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) q += z[a]! * Sinv.data[a * p + b]! * z[b]!
    t2.push(q / varZ)
    zs.push(z.slice())
  }
  return { type: 'MEWMA', lambda, h, t2, z: zs, outOfControl: t2.map((v, i) => (v > h! ? i : -1)).filter((i) => i >= 0), mean: mc.mean, covariance: mc.cov, arl0 }
}

export interface GeneralizedVarianceResult {
  type: 'generalized variance'
  /** |S| per subgroup. */
  values: number[]
  center: number
  ucl: number
  lcl: number
  outOfControl: number[]
}

/** Generalized variance chart: |Sᵢ| of each subgroup with limits |Σ̂|(b₁ ± 3√b₂) (Alt 1985). */
export function generalizedVarianceChart(data: ArrayLike<ArrayLike<number>>, subgroup: ArrayLike<unknown>): GeneralizedVarianceResult {
  const rows = Array.from(data).map((r) => Array.from(r))
  const map = new Map<string, number[][]>()
  for (let i = 0; i < rows.length; i++) {
    const k = String(subgroup[i])
    let g = map.get(k)
    if (!g) map.set(k, (g = []))
    g.push(rows[i]!)
  }
  const groups = [...map.values()]
  const m = groups[0]!.length
  const p = rows[0]!.length
  if (m <= p) throw new RangeError('generalizedVarianceChart: subgroup size must exceed the number of variables')
  const det = (M: Matrix) => {
    const L = cholesky(M)
    if (!L) return 0
    let d = 1
    for (let i = 0; i < M.rows; i++) d *= L.data[i * M.rows + i]! ** 2
    return d
  }
  const values = groups.map((g) => det(meanCov(g).cov))
  // pooled estimate of |Σ| from the average covariance
  const pooled = matrix(p, p)
  for (const g of groups) {
    const c = meanCov(g).cov
    for (let i = 0; i < p * p; i++) pooled.data[i] += c.data[i]! / groups.length
  }
  let b1 = 1
  let prodA = 1
  let prodB = 1
  for (let i = 1; i <= p; i++) {
    b1 *= (m - i) / (m - 1)
    prodA *= (m - i + 2) / (m - 1)
    prodB *= (m - i) / (m - 1)
  }
  const b2 = b1 * (prodA - prodB)
  const sigmaDet = det(pooled)
  const center = sigmaDet
  const ucl = sigmaDet * (1 + 3 * Math.sqrt(b2) / b1)
  const lcl = Math.max(0, sigmaDet * (1 - 3 * Math.sqrt(b2) / b1))
  return { type: 'generalized variance', values, center, ucl, lcl, outOfControl: values.map((v, i) => (v > ucl || v < lcl ? i : -1)).filter((i) => i >= 0) }
}

void STD
