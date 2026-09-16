/**
 * Rank-based tests, Minitab's Nonparametrics menu: Mann–Whitney (two samples) and Kruskal–Wallis
 * (k samples). Statistics and p-values agree with scipy (`mannwhitneyu`, `kruskal`); the
 * Hodges–Lehmann estimate and the confidence interval for η₁ − η₂ follow Minitab's output.
 */
import { chi2 as chi2Dist, normal, t as tDist } from './dist.js'
import { cleanNumbers, type Alternative } from './tests.js'
import { kthPairSum } from './kth.js'

const STD = normal()

/** Average ranks (ties share the mean rank) and the tie correction term Σ(t³ − t). */
function rankAverage(values: Float64Array): { ranks: Float64Array; tieSum: number; ties: boolean } {
  const n = values.length
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => values[a]! - values[b]!)
  const ranks = new Float64Array(n)
  let tieSum = 0
  let ties = false
  for (let i = 0; i < n; ) {
    let j = i
    while (j + 1 < n && values[order[j + 1]!] === values[order[i]!]) j++
    const t = j - i + 1
    if (t > 1) {
      ties = true
      tieSum += t * t * t - t
    }
    const r = (i + 1 + j + 1) / 2
    for (let p = i; p <= j; p++) ranks[order[p]!] = r
    i = j + 1
  }
  return { ranks, tieSum, ties }
}

function median(sorted: ArrayLike<number>, len = sorted.length): number {
  if (len === 0) return NaN
  const h = len >> 1
  return len % 2 ? sorted[h]! : (sorted[h - 1]! + sorted[h]!) / 2
}

export interface MannWhitneyResult {
  test: 'Mann-Whitney'
  /** U for the first sample (scipy `mannwhitneyu` statistic). */
  statistic: number
  /** Wilcoxon rank-sum W of the first sample (what Minitab prints). */
  W: number
  /** Hodges–Lehmann estimate of η₁ − η₂: median of all pairwise differences. */
  estimate: number
  /** Confidence interval for η₁ − η₂ from the ordered pairwise differences (Minitab's method). */
  ci: [number, number]
  /** Achieved confidence of `ci` (the discrete order statistics cannot hit `confidence` exactly). */
  confidence: number
  pValue: number
  alternative: Alternative
  method: 'exact' | 'asymptotic'
  ties: boolean
  n1: number
  n2: number
  medians: [number, number]
}

/**
 * Exact null distribution of U for sample sizes m, n without ties: f(u; m, n) = number of
 * arrangements with U = u, via f(u; m, n) = f(u − n; m − 1, n) + f(u; m, n − 1) (largest observation
 * is an X → it beats all n Y's; or a Y → contributes nothing).
 */
function exactUCounts(m: number, n: number): Float64Array {
  const table: Float64Array[][] = []
  for (let i = 0; i <= m; i++) {
    const row: Float64Array[] = []
    for (let j = 0; j <= n; j++) {
      const f = new Float64Array(i * j + 1)
      if (i === 0 || j === 0) f[0] = 1
      else {
        const fx = table[i - 1]![j]! // largest is X: shift by j
        const fy = row[j - 1]! // largest is Y
        for (let u = 0; u <= i * j; u++) {
          let v = 0
          if (u - j >= 0 && u - j < fx.length) v += fx[u - j]!
          if (u < fy.length) v += fy[u]!
          f[u] = v
        }
      }
      row.push(f)
    }
    table.push(row)
  }
  return table[m]![n]!
}

/**
 * Mann–Whitney test that the two populations have the same location (H0: η₁ = η₂).
 * p-value: normal approximation with continuity correction and tie adjustment (Minitab), or the exact
 * permutation distribution when there are no ties and n₁·n₂ ≤ 2500 (`method: 'auto'`, like scipy).
 */
export function mannWhitney(
  a: ArrayLike<number | null | undefined>,
  b: ArrayLike<number | null | undefined>,
  options: { alternative?: Alternative; confidence?: number; method?: 'auto' | 'exact' | 'asymptotic' } = {},
): MannWhitneyResult {
  const alternative = options.alternative ?? 'two-sided'
  const confidence = options.confidence ?? 0.95
  if (!(confidence > 0 && confidence < 1)) throw new RangeError(`confidence must be in (0, 1), got ${confidence}`)
  const x = cleanNumbers(a)
  const y = cleanNumbers(b)
  const n1 = x.length
  const n2 = y.length
  if (n1 < 1 || n2 < 1) throw new RangeError(`mannWhitney needs at least 1 observation per sample, got ${n1} and ${n2}`)
  const all = new Float64Array(n1 + n2)
  all.set(x)
  all.set(y, n1)
  const { ranks, tieSum, ties } = rankAverage(all)
  let W = 0
  for (let i = 0; i < n1; i++) W += ranks[i]!
  const U1 = W - (n1 * (n1 + 1)) / 2
  const N = n1 + n2

  let method: 'exact' | 'asymptotic' = options.method === 'exact' ? 'exact' : 'asymptotic'
  if ((options.method ?? 'auto') === 'auto' && !ties && n1 * n2 <= 2500) method = 'exact'
  if (method === 'exact' && ties) throw new RangeError('mannWhitney: exact p-value is not defined with ties; use method "asymptotic"')

  let pValue: number
  if (method === 'exact') {
    const counts = exactUCounts(n1, n2)
    let total = 0
    for (let u = 0; u < counts.length; u++) total += counts[u]!
    const cdf = (u: number) => {
      let s = 0
      for (let k = 0; k <= Math.min(u, counts.length - 1); k++) s += counts[k]!
      return s / total
    }
    const sf = (u: number) => 1 - cdf(u - 1) // P(U ≥ u)
    if (alternative === 'greater') pValue = sf(U1)
    else if (alternative === 'less') pValue = cdf(U1)
    else pValue = Math.min(1, 2 * Math.min(cdf(U1), sf(U1)))
  } else {
    const mu = (n1 * n2) / 2
    const sigma = Math.sqrt(((n1 * n2) / 12) * (N + 1 - tieSum / (N * (N - 1))))
    // continuity-corrected z for the direction of the alternative
    const z = (u: number, dir: 1 | -1) => (u - mu - dir * 0.5) / sigma
    if (alternative === 'greater') pValue = STD.sf(z(U1, 1))
    else if (alternative === 'less') pValue = STD.cdf(z(U1, -1))
    else pValue = Math.min(1, 2 * STD.sf(Math.abs(U1 - mu) - 0.5 > 0 ? (Math.abs(U1 - mu) - 0.5) / sigma : 0))
  }

  // Hodges–Lehmann estimate and CI from the ordered pairwise differences xᵢ − yⱼ, via k-th order
  // statistics of the implicit difference set (O(n log n) instead of materialising n₁n₂ values)
  const xa = x.slice().sort()
  const yNeg = Float64Array.from(y, (v) => -v).sort()
  const m = n1 * n2
  const diffAt = (kk: number) => kthPairSum(xa, yNeg, kk) // kk-th smallest difference, 1-indexed
  const estimate = m % 2 ? diffAt((m + 1) / 2) : 0.5 * (diffAt(m / 2) + diffAt(m / 2 + 1))
  const sd = Math.sqrt((n1 * n2 * (N + 1)) / 12)
  let ci: [number, number]
  let achieved: number
  if (alternative === 'two-sided') {
    const z = STD.ppf(0.5 + confidence / 2)
    let kk = Math.floor(m / 2 - z * sd) // Minitab: k-th smallest and k-th largest difference
    if (kk < 1) kk = 1
    ci = [diffAt(kk), diffAt(m - kk + 1)]
    // achieved confidence: P(kk ≤ U ≤ m − kk) under the normal approximation with continuity
    achieved = 1 - 2 * STD.sf((m / 2 - kk + 0.5) / sd)
  } else {
    const z = STD.ppf(confidence)
    let kk = Math.floor(m / 2 - z * sd)
    if (kk < 1) kk = 1
    achieved = 1 - STD.sf((m / 2 - kk + 0.5) / sd)
    ci = alternative === 'greater' ? [diffAt(kk), Infinity] : [-Infinity, diffAt(m - kk + 1)]
  }
  const xs = x.slice().sort()
  const ys = y.slice().sort()
  return {
    test: 'Mann-Whitney',
    statistic: U1,
    W,
    estimate,
    ci,
    confidence: Math.min(1, Math.max(0, achieved)),
    pValue,
    alternative,
    method,
    ties,
    n1,
    n2,
    medians: [median(xs), median(ys)],
  }
}

export interface KruskalResult {
  test: 'Kruskal-Wallis'
  /** H adjusted for ties (what Minitab and scipy report). */
  statistic: number
  hUnadjusted: number
  df: number
  pValue: number
  n: number
  groups: Array<{ name: string; n: number; median: number; avgRank: number; z: number }>
}

/** Kruskal–Wallis test of equal medians across k groups (H, tie-adjusted; χ² with k − 1 df). */
export function kruskal(
  groups: Record<string, ArrayLike<number | null | undefined>> | Array<ArrayLike<number | null | undefined>>,
): KruskalResult {
  const entries = Array.isArray(groups) ? groups.map((g, i) => [String(i), g] as const) : Object.entries(groups)
  const gs = entries.map(([name, g]) => ({ name, v: cleanNumbers(g) })).filter((g) => g.v.length > 0)
  const k = gs.length
  if (k < 2) throw new RangeError(`kruskal needs at least 2 non-empty groups, got ${k}`)
  const N = gs.reduce((s, g) => s + g.v.length, 0)
  const all = new Float64Array(N)
  let off = 0
  for (const g of gs) {
    all.set(g.v, off)
    off += g.v.length
  }
  const { ranks, tieSum } = rankAverage(all)
  let h = 0
  off = 0
  const out: KruskalResult['groups'] = []
  for (const g of gs) {
    const n = g.v.length
    let R = 0
    for (let i = 0; i < n; i++) R += ranks[off + i]!
    off += n
    h += (R * R) / n
    const avgRank = R / n
    const z = (avgRank - (N + 1) / 2) / Math.sqrt(((N + 1) * (N - n)) / (12 * n))
    out.push({ name: g.name, n, median: median(g.v.slice().sort()), avgRank, z })
  }
  const hUnadjusted = (12 / (N * (N + 1))) * h - 3 * (N + 1)
  const correction = 1 - tieSum / (N * N * N - N)
  const statistic = correction > 0 ? hUnadjusted / correction : NaN
  const df = k - 1
  return { test: 'Kruskal-Wallis', statistic, hUnadjusted, df, pValue: chi2Dist(df).sf(statistic), n: N, groups: out }
}

export interface TwoSampleScaleResult {
  test: 'Fligner-Killeen' | 'Ansari-Bradley' | 'Brunner-Munzel' | 'Mood two-sample' | 'Epps-Singleton'
  statistic: number
  pValue: number
  n1: number
  n2: number
}

function asGroups(groups: Record<string, ArrayLike<number | null | undefined>> | Array<ArrayLike<number | null | undefined>>): Float64Array[] {
  const arr = Array.isArray(groups) ? groups : Object.values(groups)
  return arr.map((g) => cleanNumbers(g))
}

/** Fligner–Killeen test of equal variances (scipy fligner). */
export function fligner(
  groups: Record<string, ArrayLike<number | null | undefined>> | Array<ArrayLike<number | null | undefined>>,
): TwoSampleScaleResult {
  const gs = asGroups(groups)
  if (gs.length < 2) throw new RangeError('fligner: need ≥2 groups')
  const medians = gs.map((g) => {
    const s = Array.from(g).sort((a, b) => a - b)
    const m = s.length
    return m % 2 ? s[(m - 1) / 2]! : 0.5 * (s[m / 2 - 1]! + s[m / 2]!)
  })
  const all: { a: number; gi: number }[] = []
  for (let gi = 0; gi < gs.length; gi++) {
    for (const x of gs[gi]!) all.push({ a: Math.abs(x - medians[gi]!), gi })
  }
  all.sort((u, v) => u.a - v.a)
  const N = all.length
  // normal scores of ranks
  const scores = new Array(N)
  for (let i = 0; i < N; ) {
    let j = i
    while (j + 1 < N && all[j + 1]!.a === all[i]!.a) j++
    const r = (i + 1 + j + 1) / 2
    const sc = STD.ppf(0.5 + r / (2 * (N + 1)))
    for (let k = i; k <= j; k++) scores[k] = sc
    i = j + 1
  }
  const meanScore = scores.reduce((a, b) => a + b!, 0) / N
  let ss = 0
  for (const s of scores) ss += (s! - meanScore) ** 2
  const varScore = ss / (N - 1)
  let X2 = 0
  for (let gi = 0; gi < gs.length; gi++) {
    let sum = 0
    let n = 0
    for (let i = 0; i < N; i++) if (all[i]!.gi === gi) {
      sum += scores[i]!
      n++
    }
    X2 += n * ((sum / n - meanScore) ** 2)
  }
  X2 /= varScore
  const df = gs.length - 1
  const pValue = chi2Dist(df).sf(X2)
  return { test: 'Fligner-Killeen', statistic: X2, pValue, n1: gs[0]!.length, n2: gs[1]!.length }
}

/** Ansari–Bradley test for equal scale (two samples; scipy ansari). */
export function ansariBradley(
  a: ArrayLike<number | null | undefined>,
  b: ArrayLike<number | null | undefined>,
): TwoSampleScaleResult {
  const x = cleanNumbers(a)
  const y = cleanNumbers(b)
  const n1 = x.length
  const n2 = y.length
  if (n1 < 2 || n2 < 2) throw new RangeError('ansariBradley: need ≥2 per sample')
  const all = [...Array.from(x).map((v) => ({ v, g: 0 })), ...Array.from(y).map((v) => ({ v, g: 1 }))]
  all.sort((u, v) => u.v - v.v)
  const N = all.length
  const ranks = new Array(N)
  for (let i = 0; i < N; ) {
    let j = i
    while (j + 1 < N && all[j + 1]!.v === all[i]!.v) j++
    const r = (i + 1 + j + 1) / 2
    for (let k = i; k <= j; k++) ranks[k] = r
    i = j + 1
  }
  // Ansari scores: min(rank, N+1-rank)
  let AB = 0
  for (let i = 0; i < N; i++) if (all[i]!.g === 0) AB += Math.min(ranks[i]!, N + 1 - ranks[i]!)
  const mu = n1 * (N + 1) / 4
  const varAB =
    N % 2 === 0
      ? (n1 * n2 * (N + 2) * (N - 2)) / (48 * (N - 1))
      : (n1 * n2 * (N + 1) * (3 + N ** 2)) / (48 * N ** 2)
  const z = (AB - mu) / Math.sqrt(Math.max(1e-12, varAB))
  const pValue = Math.min(1, 2 * STD.sf(Math.abs(z)))
  return { test: 'Ansari-Bradley', statistic: AB, pValue, n1, n2 }
}

/** Brunner–Munzel test (scipy brunnermunzel) — stochastic equality. */
export function brunnerMunzel(
  a: ArrayLike<number | null | undefined>,
  b: ArrayLike<number | null | undefined>,
): TwoSampleScaleResult {
  const x = cleanNumbers(a)
  const y = cleanNumbers(b)
  const n1 = x.length
  const n2 = y.length
  if (n1 < 2 || n2 < 2) throw new RangeError('brunnerMunzel: need ≥2 per sample')
  const all = [...Array.from(x).map((v) => ({ v, g: 0 })), ...Array.from(y).map((v) => ({ v, g: 1 }))]
  all.sort((u, v) => u.v - v.v)
  const N = all.length
  const ranks = new Array(N)
  for (let i = 0; i < N; ) {
    let j = i
    while (j + 1 < N && all[j + 1]!.v === all[i]!.v) j++
    const r = (i + 1 + j + 1) / 2
    for (let k = i; k <= j; k++) ranks[k] = r
    i = j + 1
  }
  let r1 = 0
  let r2 = 0
  const mid1: number[] = []
  const mid2: number[] = []
  // within-group ranks for variance
  const xSorted = Array.from(x).sort((a, b) => a - b)
  const ySorted = Array.from(y).sort((a, b) => a - b)
  const rankIn = (sorted: number[], val: number) => {
    let lo = 0
    let hi = sorted.length
    while (lo < hi) {
      const m = (lo + hi) >> 1
      if (sorted[m]! < val) lo = m + 1
      else hi = m
    }
    let j = lo
    while (j + 1 < sorted.length && sorted[j + 1] === val) j++
    return (lo + 1 + j + 1) / 2
  }
  for (let i = 0; i < N; i++) {
    if (all[i]!.g === 0) {
      r1 += ranks[i]!
      mid1.push(ranks[i]! - rankIn(xSorted, all[i]!.v))
    } else {
      r2 += ranks[i]!
      mid2.push(ranks[i]! - rankIn(ySorted, all[i]!.v))
    }
  }
  const m1 = r1 / n1
  const m2 = r2 / n2
  const pHat = (m1 - (n1 + 1) / 2) / n2
  let s1 = 0
  let s2 = 0
  const mean1 = mid1.reduce((a, b) => a + b, 0) / n1
  const mean2 = mid2.reduce((a, b) => a + b, 0) / n2
  for (const v of mid1) s1 += (v - mean1) ** 2
  for (const v of mid2) s2 += (v - mean2) ** 2
  s1 /= n1 - 1
  s2 /= n2 - 1
  const se = Math.sqrt(s1 / n2 ** 2 / n1 + s2 / n1 ** 2 / n2)
  const statistic = (pHat - 0.5) / Math.max(1e-12, se)
  const df =
    se ** 4 /
    ((s1 / n2 ** 2 / n1) ** 2 / (n1 - 1) + (s2 / n1 ** 2 / n2) ** 2 / (n2 - 1))
  const pValue = Math.min(1, 2 * tDist(Math.max(1, df)).sf(Math.abs(statistic)))
  return { test: 'Brunner-Munzel', statistic, pValue, n1, n2 }
}

/** Mood two-sample scale test (scipy mood). */
export function moodTwoSample(
  a: ArrayLike<number | null | undefined>,
  b: ArrayLike<number | null | undefined>,
): TwoSampleScaleResult {
  const x = cleanNumbers(a)
  const y = cleanNumbers(b)
  const n1 = x.length
  const n2 = y.length
  const all = [...Array.from(x).map((v) => ({ v, g: 0 })), ...Array.from(y).map((v) => ({ v, g: 1 }))]
  all.sort((u, v) => u.v - v.v)
  const N = all.length
  const ranks = new Array(N)
  for (let i = 0; i < N; ) {
    let j = i
    while (j + 1 < N && all[j + 1]!.v === all[i]!.v) j++
    const r = (i + 1 + j + 1) / 2
    for (let k = i; k <= j; k++) ranks[k] = r
    i = j + 1
  }
  const mid = (N + 1) / 2
  let M = 0
  for (let i = 0; i < N; i++) {
    if (all[i]!.g === 0) M += (ranks[i]! - mid) ** 2
  }
  const mu = (n1 * (N * N - 1)) / 12
  const varM = (n1 * n2 * (N + 1) * (N + 2) * (N - 2)) / 180
  const z = (M - mu) / Math.sqrt(Math.max(1e-12, varM))
  const pValue = Math.min(1, 2 * STD.sf(Math.abs(z)))
  return { test: 'Mood two-sample', statistic: M, pValue, n1, n2 }
}

/**
 * Epps–Singleton two-sample test (characteristic function; scipy epps_singleton_2samp MVP).
 * Uses t = {0.4, 0.8} on standardized pooled data.
 */
export function eppsSingleton(
  a: ArrayLike<number | null | undefined>,
  b: ArrayLike<number | null | undefined>,
): TwoSampleScaleResult {
  const x = Array.from(cleanNumbers(a))
  const y = Array.from(cleanNumbers(b))
  const n1 = x.length
  const n2 = y.length
  if (n1 < 5 || n2 < 5) throw new RangeError('eppsSingleton needs ≥5 per sample')
  const pool = x.concat(y)
  const N = pool.length
  let mean = 0
  for (const v of pool) mean += v
  mean /= N
  let m2 = 0
  for (const v of pool) m2 += (v - mean) ** 2
  const sd = Math.sqrt(m2 / (N - 1)) || 1
  const xs = x.map((v) => (v - mean) / sd)
  const ys = y.map((v) => (v - mean) / sd)
  const ts = [0.4, 0.8]
  // g = (mean cos tx, mean sin tx, mean cos 2tx? — use two t's → 4-vector)
  const moment = (sample: number[], t: number) => {
    let c = 0
    let s = 0
    for (const v of sample) {
      c += Math.cos(t * v)
      s += Math.sin(t * v)
    }
    return [c / sample.length, s / sample.length] as const
  }
  const gx: number[] = []
  const gy: number[] = []
  for (const t of ts) {
    const mx = moment(xs, t)
    const my = moment(ys, t)
    gx.push(mx[0], mx[1])
    gy.push(my[0], my[1])
  }
  const diff = gx.map((v, i) => v - gy[i]!)
  // Covariance of CF moments (diagonal approx MVP)
  const covDiag = new Array(4).fill(0)
  for (let k = 0; k < 4; k++) {
    const t = ts[Math.floor(k / 2)]!
    const useCos = k % 2 === 0
    let s1 = 0
    let s2 = 0
    for (const v of xs) {
      const u = useCos ? Math.cos(t * v) : Math.sin(t * v)
      s1 += (u - gx[k]!) ** 2
    }
    for (const v of ys) {
      const u = useCos ? Math.cos(t * v) : Math.sin(t * v)
      s2 += (u - gy[k]!) ** 2
    }
    covDiag[k] = s1 / (n1 * (n1 - 1)) + s2 / (n2 * (n2 - 1))
  }
  let W = 0
  for (let k = 0; k < 4; k++) W += (diff[k]! * diff[k]!) / Math.max(1e-12, covDiag[k]!)
  const pValue = chi2Dist(4).sf(W)
  return { test: 'Epps-Singleton', statistic: W, pValue, n1, n2 }
}
