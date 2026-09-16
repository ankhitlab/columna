/**
 * Tier 4.11 — Quality Tools extras:
 * Pareto, Run-chart tests (clustering / mixtures / oscillation / trends), Multi-Vari summary,
 * symmetry test, Individual Distribution Identification (Anderson–Darling for common families).
 */
import { andersonDarling } from './normality.js'
import { boxCoxLambda, weibullFit } from './capability.js'
import { chi2 as chi2Dist, normal } from './dist.js'
import { cleanNumbers } from './tests.js'
import { runsTest } from './nonparametric2.js'
import { maxOf, minOf } from './numerics.js'

const STD = normal()

function meanOf(v: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i]!
  return s / v.length
}
function sdOf(v: ArrayLike<number>): number {
  const n = v.length
  const m = meanOf(v)
  let s2 = 0
  for (let i = 0; i < n; i++) s2 += (v[i]! - m) ** 2
  return Math.sqrt(s2 / (n - 1))
}

// ---- Pareto ---------------------------------------------------------------------------------------

export interface ParetoItem {
  category: string
  count: number
  pct: number
  cumulativePct: number
}

export interface ParetoResult {
  items: ParetoItem[]
  total: number
}

/** Pareto chart data: categories sorted by count descending with cumulative percent. */
export function pareto(categories: ArrayLike<string | number | null | undefined>, options: { weights?: ArrayLike<number | null | undefined> } = {}): ParetoResult {
  const n = categories.length
  const weights = options.weights
  const map = new Map<string, number>()
  let total = 0
  for (let i = 0; i < n; i++) {
    const c = categories[i]
    if (c == null) continue
    const w = weights ? Number(weights[i]) : 1
    if (!Number.isFinite(w) || w < 0) continue
    const k = String(c)
    map.set(k, (map.get(k) ?? 0) + w)
    total += w
  }
  if (total <= 0) throw new RangeError('pareto: no observations')
  const items = [...map.entries()]
    .map(([category, count]) => ({ category, count, pct: (100 * count) / total, cumulativePct: 0 }))
    .sort((a, b) => b.count - a.count || (a.category < b.category ? -1 : 1))
  let cum = 0
  for (const it of items) {
    cum += it.pct
    it.cumulativePct = cum
  }
  return { items, total }
}

// ---- Run chart ------------------------------------------------------------------------------------

export interface RunChartResult {
  n: number
  median: number
  nRuns: number
  /** Clustering: too few runs about the median. */
  clustering: { pValue: number; reject: boolean }
  /** Mixtures: too many runs. */
  mixtures: { pValue: number; reject: boolean }
  /** Oscillation: too many crossings / alternating pattern (same as mixtures for median runs). */
  oscillation: { pValue: number; reject: boolean }
  /** Trends: too few up/down runs in consecutive differences. */
  trends: { pValue: number; reject: boolean }
}

function medianOf(sorted: number[]): number {
  const n = sorted.length
  return n % 2 ? sorted[(n - 1) / 2]! : 0.5 * (sorted[n / 2 - 1]! + sorted[n / 2]!)
}

/**
 * Minitab Run Chart tests about the sample median: clustering, mixtures, oscillation, trends.
 * Uses the exact / normal runs distribution (same core as `runsTest`).
 */
export function runChart(x: ArrayLike<number | null | undefined>, options: { alpha?: number } = {}): RunChartResult {
  const v = Array.from(cleanNumbers(x))
  const n = v.length
  if (n < 10) throw new RangeError('runChart needs at least 10 observations')
  const alpha = options.alpha ?? 0.05
  const med = medianOf(v.slice().sort((a, b) => a - b))
  // dichotomize about median; drop ties at median (Minitab)
  const binary: number[] = []
  for (const xi of v) {
    if (xi > med) binary.push(1)
    else if (xi < med) binary.push(0)
  }
  const nb = binary.length
  if (nb < 10) throw new RangeError('runChart: too many ties at the median')
  const rt = runsTest(binary.map((b) => (b ? 1 : -1)), { k: 0, correction: false })
  // mixtures / oscillation: upper-tail (too many runs); clustering / trends: lower-tail
  const n1 = binary.filter((b) => b === 1).length
  const n2 = nb - n1
  const mu = (2 * n1 * n2) / nb + 1
  const sigma = Math.sqrt((2 * n1 * n2 * (2 * n1 * n2 - nb)) / (nb * nb * (nb - 1)))
  const z = (rt.runs - mu) / sigma
  const pLower = STD.cdf(z)
  const pUpper = STD.sf(z)

  // Trends: runs of successive differences (up/down)
  const diffs: number[] = []
  for (let i = 1; i < n; i++) {
    const d = v[i]! - v[i - 1]!
    if (d !== 0) diffs.push(d > 0 ? 1 : -1)
  }
  const trendRuns = runsTest(diffs, { k: 0, correction: false })
  const nUp = diffs.filter((d) => d > 0).length
  const nDown = diffs.length - nUp
  const muT = (2 * nUp * nDown) / diffs.length + 1
  const sigT = Math.sqrt((2 * nUp * nDown * (2 * nUp * nDown - diffs.length)) / (diffs.length * diffs.length * (diffs.length - 1)))
  const zT = (trendRuns.runs - muT) / sigT
  const pTrend = STD.cdf(zT)

  return {
    n,
    median: med,
    nRuns: rt.runs,
    clustering: { pValue: pLower, reject: pLower < alpha },
    mixtures: { pValue: pUpper, reject: pUpper < alpha },
    oscillation: { pValue: pUpper, reject: pUpper < alpha },
    trends: { pValue: pTrend, reject: pTrend < alpha },
  }
}

// ---- Multi-vari -----------------------------------------------------------------------------------

export interface MultiVariLevel {
  factor: string
  level: string
  n: number
  mean: number
  min: number
  max: number
  range: number
}

export interface MultiVariResult {
  levels: MultiVariLevel[]
  /** Nested means table for up to 3 factors (factor order as given). */
  cells: Array<{ keys: string[]; n: number; mean: number; min: number; max: number }>
}

/** Multi-Vari chart summary: means / min / max by one to three nested factors. */
export function multiVari(
  measurement: ArrayLike<number | null | undefined>,
  factors: ArrayLike<string | number | null>[],
): MultiVariResult {
  if (factors.length < 1 || factors.length > 3) throw new RangeError('multiVari: provide 1 to 3 factor columns')
  const y = Array.from(measurement)
  const n = y.length
  for (const f of factors) if (f.length !== n) throw new RangeError('multiVari: factor length mismatch')
  const levels: MultiVariLevel[] = []
  for (let fi = 0; fi < factors.length; fi++) {
    const map = new Map<string, number[]>()
    for (let i = 0; i < n; i++) {
      const yi = y[i]
      const fv = factors[fi]![i]
      if (typeof yi !== 'number' || !Number.isFinite(yi) || fv == null) continue
      const k = String(fv)
      let arr = map.get(k)
      if (!arr) map.set(k, (arr = []))
      arr.push(yi)
    }
    for (const [level, arr] of [...map.entries()].sort(([a], [b]) => (a < b ? -1 : 1))) {
      const mn = minOf(arr)
      const mx = maxOf(arr)
      levels.push({ factor: `F${fi + 1}`, level, n: arr.length, mean: meanOf(arr), min: mn, max: mx, range: mx - mn })
    }
  }
  const cellsMap = new Map<string, number[]>()
  for (let i = 0; i < n; i++) {
    const yi = y[i]
    if (typeof yi !== 'number' || !Number.isFinite(yi)) continue
    const keys: string[] = []
    let ok = true
    for (const f of factors) {
      if (f[i] == null) {
        ok = false
        break
      }
      keys.push(String(f[i]))
    }
    if (!ok) continue
    const k = keys.join('\0')
    let arr = cellsMap.get(k)
    if (!arr) cellsMap.set(k, (arr = []))
    arr.push(yi)
  }
  const cells = [...cellsMap.entries()].map(([k, arr]) => {
    const mn = minOf(arr)
    const mx = maxOf(arr)
    return { keys: k.split('\0'), n: arr.length, mean: meanOf(arr), min: mn, max: mx }
  })
  return { levels, cells }
}

// ---- Symmetry -------------------------------------------------------------------------------------

export interface SymmetryResult {
  /** Mira / triples symmetry statistic (approximate). */
  statistic: number
  pValue: number
  n: number
  method: 'triples'
}

/**
 * Triples test of symmetry about the sample median (Randles et al. / Minitab Symmetry Test).
 * Counts right-skewed vs left-skewed triples; normal approximation for large n.
 */
export function symmetryTest(x: ArrayLike<number | null | undefined>): SymmetryResult {
  const v = Array.from(cleanNumbers(x))
  const n = v.length
  if (n < 8) throw new RangeError('symmetryTest needs at least 8 observations')
  // For n large, full O(n³) is expensive — subsample systematically when n > 40
  const idx = n <= 40 ? v.map((_, i) => i) : Array.from({ length: 40 }, (_, i) => Math.floor((i * (n - 1)) / 39))
  const sample = idx.map((i) => v[i]!)
  const m = sample.length
  let pos = 0
  let neg = 0
  for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) for (let k = j + 1; k < m; k++) {
    const a = sample[i]!
    const b = sample[j]!
    const c = sample[k]!
    const mid = a + b + c - Math.min(a, b, c) - Math.max(a, b, c)
    const mean3 = (a + b + c) / 3
    if (mean3 > mid) pos++
    else if (mean3 < mid) neg++
  }
  const total = pos + neg
  if (total === 0) return { statistic: 0, pValue: 1, n, method: 'triples' }
  const hat = (pos - neg) / total
  // Var under symmetry ≈ 1/(9n) order; use binomial SE on (pos/(pos+neg) − ½)
  const se = Math.sqrt(0.25 / total)
  const z = (pos / total - 0.5) / se
  void hat
  return { statistic: z, pValue: 2 * STD.sf(Math.abs(z)), n, method: 'triples' }
}

// ---- Individual Distribution Identification -------------------------------------------------------

export type IdDistribution =
  | 'normal'
  | 'lognormal'
  | 'lognormal-3'
  | 'exponential'
  | 'exponential-2'
  | 'weibull'
  | 'weibull-3'
  | 'smallest-extreme'
  | 'largest-extreme'
  | 'gamma'
  | 'logistic'
  | 'loglogistic'
  | 'box-cox'
  | 'uniform'

export interface IdFit {
  distribution: IdDistribution
  /** Anderson–Darling A² on the probability-integral-transformed scale (or Box–Cox normal). */
  ad: number
  pValue: number
  params: Record<string, number>
}

export interface IdResult {
  fits: IdFit[]
  /** Best (lowest AD) among successfully fitted families. */
  best: IdFit
}

function adOnUniforms(u: number[]): { ad: number; pValue: number } {
  // Anderson–Darling against Uniform(0,1) / known continuous CDF — Stephens
  const v = u.slice().sort((a, b) => a - b).map((x) => Math.min(1 - 1e-15, Math.max(1e-15, x)))
  const n = v.length
  let s = 0
  for (let i = 0; i < n; i++) s += (2 * i + 1) * (Math.log(v[i]!) + Math.log(1 - v[n - 1 - i]!))
  const a2 = -n - s / n
  // Stephens modified for known parameters: A* = A² (1 + 0.6/n) roughly; use D'Agostino table-ish
  const adjusted = a2 * (1 + 0.6 / n)
  let p: number
  if (adjusted >= 0.6) p = Math.exp(1.2937 - 5.709 * adjusted + 0.0186 * adjusted * adjusted)
  else if (adjusted >= 0.34) p = Math.exp(0.9177 - 4.279 * adjusted - 1.38 * adjusted * adjusted)
  else if (adjusted >= 0.2) p = 1 - Math.exp(-8.318 + 42.796 * adjusted - 59.938 * adjusted * adjusted)
  else p = 1 - Math.exp(-13.436 + 101.14 * adjusted - 223.73 * adjusted * adjusted)
  return { ad: a2, pValue: Math.min(1, Math.max(0, p)) }
}

function gammaPpfInvCdf(x: number, shape: number, scale: number): number {
  // lower incomplete gamma ratio P(shape, x/scale)
  // use series / continued fraction via chi2 relation: F_gamma(x;a,θ) = F_χ²(2x/θ; 2a)
  return chi2Dist(2 * shape).cdf((2 * x) / scale)
}

/**
 * Individual Distribution Identification: fit common Minitab families and rank by Anderson–Darling.
 * Covers the usual 14-ish candidates; three-parameter versions use a simple threshold = min − ε.
 */
export function individualDistributionID(x: ArrayLike<number | null | undefined>): IdResult {
  const raw = Array.from(cleanNumbers(x))
  const n = raw.length
  if (n < 8) throw new RangeError('individualDistributionID needs at least 8 observations')
  const fits: IdFit[] = []
  const push = (distribution: IdDistribution, params: Record<string, number>, u: number[]) => {
    const { ad, pValue } = adOnUniforms(u)
    fits.push({ distribution, ad, pValue, params })
  }

  // Normal
  {
    const m = meanOf(raw)
    const s = sdOf(raw)
    push('normal', { mean: m, sd: s }, raw.map((xi) => STD.cdf((xi - m) / s)))
  }

  // Lognormal (2p)
  if (raw.every((xi) => xi > 0)) {
    const logs = raw.map(Math.log)
    const m = meanOf(logs)
    const s = sdOf(logs)
    push('lognormal', { location: m, scale: s }, raw.map((xi) => STD.cdf((Math.log(xi) - m) / s)))
  }

  // Lognormal 3p
  {
    const loc = minOf(raw) - 1e-6 * (maxOf(raw) - minOf(raw) || 1)
    const shifted = raw.map((xi) => xi - loc)
    if (shifted.every((xi) => xi > 0)) {
      const logs = shifted.map(Math.log)
      const m = meanOf(logs)
      const s = sdOf(logs)
      push('lognormal-3', { threshold: loc, location: m, scale: s }, shifted.map((xi) => STD.cdf((Math.log(xi) - m) / s)))
    }
  }

  // Exponential (1p, scale = mean)
  if (raw.every((xi) => xi >= 0)) {
    const scale = meanOf(raw)
    push('exponential', { scale }, raw.map((xi) => 1 - Math.exp(-xi / scale)))
  }

  // Exponential 2p
  {
    const loc = minOf(raw)
    const shifted = raw.map((xi) => xi - loc)
    const scale = meanOf(shifted)
    if (scale > 0) push('exponential-2', { threshold: loc, scale }, shifted.map((xi) => 1 - Math.exp(-xi / scale)))
  }

  // Weibull 2p / 3p
  if (raw.every((xi) => xi > 0)) {
    try {
      const w = weibullFit(raw)
      push(
        'weibull',
        { shape: w.shape, scale: w.scale },
        raw.map((xi) => 1 - Math.exp(-((xi / w.scale) ** w.shape))),
      )
    } catch {
      /* skip */
    }
  }
  {
    const loc = minOf(raw) - 1e-9
    try {
      const w = weibullFit(raw, { location: loc })
      push(
        'weibull-3',
        { threshold: loc, shape: w.shape, scale: w.scale },
        raw.map((xi) => 1 - Math.exp(-(((xi - loc) / w.scale) ** w.shape))),
      )
    } catch {
      /* skip */
    }
  }

  // Smallest / largest extreme value (Gumbel)
  {
    // μ, β by moments: β = s √6 / π, μ = mean − 0.57721 β
    const s = sdOf(raw)
    const beta = (s * Math.sqrt(6)) / Math.PI
    const mu = meanOf(raw) - 0.5772156649 * beta
    push(
      'smallest-extreme',
      { location: mu, scale: beta },
      raw.map((xi) => 1 - Math.exp(-Math.exp((xi - mu) / beta))),
    )
    push(
      'largest-extreme',
      { location: mu, scale: beta },
      raw.map((xi) => Math.exp(-Math.exp(-(xi - mu) / beta))),
    )
  }

  // Gamma (2p MOM)
  if (raw.every((xi) => xi > 0)) {
    const m = meanOf(raw)
    const s2 = sdOf(raw) ** 2
    const shape = (m * m) / s2
    const scale = s2 / m
    if (shape > 0 && scale > 0) {
      push(
        'gamma',
        { shape, scale },
        raw.map((xi) => gammaPpfInvCdf(xi, shape, scale)),
      )
    }
  }

  // Logistic
  {
    const s = sdOf(raw)
    const scale = (s * Math.sqrt(3)) / Math.PI
    const loc = meanOf(raw)
    push(
      'logistic',
      { location: loc, scale },
      raw.map((xi) => 1 / (1 + Math.exp(-(xi - loc) / scale))),
    )
  }

  // Loglogistic
  if (raw.every((xi) => xi > 0)) {
    const logs = raw.map(Math.log)
    const s = sdOf(logs)
    const scale = (s * Math.sqrt(3)) / Math.PI
    const loc = meanOf(logs)
    push(
      'loglogistic',
      { location: loc, scale },
      raw.map((xi) => 1 / (1 + Math.exp(-(Math.log(xi) - loc) / scale))),
    )
  }

  // Box–Cox → normal
  if (raw.every((xi) => xi > 0)) {
    try {
      const bc = boxCoxLambda(raw)
      const m = meanOf(bc.transformed)
      const s = sdOf(bc.transformed)
      const adNorm = andersonDarling(bc.transformed)
      fits.push({ distribution: 'box-cox', ad: adNorm.statistic, pValue: adNorm.pValue, params: { lambda: bc.lambda, mean: m, sd: s } })
    } catch {
      /* skip */
    }
  }

  // Uniform
  {
    const a = minOf(raw)
    const b = maxOf(raw)
    if (b > a) push('uniform', { min: a, max: b }, raw.map((xi) => (xi - a) / (b - a)))
  }

  fits.sort((p, q) => p.ad - q.ad)
  return { fits, best: fits[0]! }
}
