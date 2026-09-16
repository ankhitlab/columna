/**
 * Tier 2 — the rest of Minitab's Nonparametrics menu: 1-Sample Sign, 1-Sample Wilcoxon,
 * Mood's Median, Friedman, Runs Test.
 */
import { binomial, chi2 as chi2Dist, normal } from './dist.js'
import { cleanNumbers, type Alternative } from './tests.js'
import { kthWalshAverage } from './kth.js'

const STD = normal()

function checkAlt(a: Alternative | undefined): Alternative {
  const alt = a ?? 'two-sided'
  if (alt !== 'two-sided' && alt !== 'less' && alt !== 'greater') throw new RangeError(`alternative must be 'two-sided' | 'less' | 'greater', got ${String(alt)}`)
  return alt
}
function median(sorted: ArrayLike<number>): number {
  const n = sorted.length
  if (n === 0) return NaN
  const h = n >> 1
  return n % 2 ? sorted[h]! : (sorted[h - 1]! + sorted[h]!) / 2
}

// ---- 1-Sample Sign --------------------------------------------------------------------------------------

export interface SignTestResult {
  test: '1-sample sign'
  /** Sample median. */
  estimate: number
  /** Counts below / equal / above the hypothesized median. */
  below: number
  equal: number
  above: number
  pValue: number
  alternative: Alternative
  /**
   * Minitab-style confidence interval: the two achievable intervals from order statistics that bracket
   * the requested confidence, and the nonlinear interpolation (Hettmansperger & Sheather 1986) between them.
   */
  ci: [number, number]
  confidence: number
  achievable: Array<{ confidence: number; ci: [number, number]; position: number }>
  /** Number of observations (including those equal to the hypothesized median). */
  n: number
}

/**
 * 1-Sample Sign test of H0: median = eta (Minitab). Exact binomial p-value on the signs (ties with eta
 * are dropped); CI by order statistics with Minitab's nonlinear interpolation.
 */
export function signTest(
  x: ArrayLike<number | null | undefined>,
  options: { median?: number; alternative?: Alternative; confidence?: number } = {},
): SignTestResult {
  const eta = options.median ?? 0
  const alternative = checkAlt(options.alternative)
  const confidence = options.confidence ?? 0.95
  if (!(confidence > 0 && confidence < 1)) throw new RangeError(`confidence must be in (0, 1), got ${confidence}`)
  const v = cleanNumbers(x)
  const n = v.length
  if (n < 1) throw new RangeError('signTest needs at least 1 observation')
  let below = 0
  let above = 0
  let equal = 0
  for (let i = 0; i < n; i++) {
    if (v[i]! < eta) below++
    else if (v[i]! > eta) above++
    else equal++
  }
  const m = below + above
  let pValue: number
  if (m === 0) pValue = 1
  else {
    const d = binomial(m, 0.5)
    if (alternative === 'greater') pValue = d.sf(above - 1)
    else if (alternative === 'less') pValue = d.cdf(above)
    else pValue = Math.min(1, 2 * Math.min(d.cdf(above), d.sf(above - 1)))
  }
  // CI from order statistics: [x(d+1), x(n−d)] (1-indexed) has confidence 1 − 2·P(Bin(n, ½) ≤ d).
  // Minitab reports the achievable interval just above the requested confidence (d*), the one just
  // below (d* + 1) and the nonlinear interpolation between them (Hettmansperger & Sheather 1986).
  const sorted = v.slice().sort()
  const bin = binomial(n, 0.5)
  const confAt = (dd: number) => 1 - 2 * bin.cdf(dd)
  const dMax = Math.floor((n - 1) / 2)
  let dStar = 0
  while (dStar + 1 <= dMax && confAt(dStar + 1) >= confidence) dStar++
  const interval = (dd: number): [number, number] => [sorted[dd]!, sorted[n - 1 - dd]!]
  const achievable: SignTestResult['achievable'] = [{ confidence: confAt(dStar), ci: interval(dStar), position: dStar + 1 }]
  let ci: [number, number] = interval(dStar)
  if (dStar + 1 <= dMax) {
    const narrow = { confidence: confAt(dStar + 1), ci: interval(dStar + 1), position: dStar + 2 }
    achievable.push(narrow)
    if (achievable[0]!.confidence > confidence && narrow.confidence < confidence) {
      const dd = dStar
      const I = (achievable[0]!.confidence - confidence) / (achievable[0]!.confidence - narrow.confidence)
      const lambda = ((n - dd) * I) / (dd + (n - 2 * dd) * I)
      ci = [(1 - lambda) * ci[0] + lambda * narrow.ci[0], (1 - lambda) * ci[1] + lambda * narrow.ci[1]]
    }
  }
  return { test: '1-sample sign', estimate: median(sorted), below, equal, above, pValue, alternative, ci, confidence, achievable, n }
}

// ---- 1-Sample Wilcoxon --------------------------------------------------------------------------------------

export interface WilcoxonResult {
  test: '1-sample Wilcoxon'
  /** Sum of positive signed ranks W⁺ (what Minitab prints). */
  statistic: number
  /** Hodges–Lehmann estimate: median of the Walsh averages. */
  estimate: number
  pValue: number
  method: 'exact' | 'asymptotic'
  alternative: Alternative
  ci: [number, number]
  confidence: number
  /** Observations used (zeros dropped). */
  n: number
  nTotal: number
}

/** Exact null distribution of W⁺ for n signed ranks without ties: counts per value 0..n(n+1)/2. */
function exactSignedRankCounts(n: number): Float64Array {
  const max = (n * (n + 1)) / 2
  let f = new Float64Array(max + 1)
  f[0] = 1
  for (let k = 1; k <= n; k++) {
    const g = new Float64Array(max + 1)
    for (let w = 0; w <= max; w++) {
      if (f[w]) {
        g[w] += f[w]!
        g[w + k] += f[w]!
      }
    }
    f = g
  }
  return f
}

/**
 * 1-Sample Wilcoxon signed-rank test of H0: median = eta (Minitab). Zeros are dropped, ties get average
 * ranks; exact p-value when there are no ties among |x − eta| and n ≤ 50, else normal approximation
 * with continuity and tie corrections. Estimate and CI from the Walsh averages (Minitab's method).
 */
export function wilcoxonSigned(
  x: ArrayLike<number | null | undefined>,
  options: { median?: number; alternative?: Alternative; confidence?: number; method?: 'auto' | 'exact' | 'asymptotic' } = {},
): WilcoxonResult {
  const eta = options.median ?? 0
  const alternative = checkAlt(options.alternative)
  const confidence = options.confidence ?? 0.95
  if (!(confidence > 0 && confidence < 1)) throw new RangeError(`confidence must be in (0, 1), got ${confidence}`)
  const all = cleanNumbers(x)
  const nTotal = all.length
  const d = Array.from(all, (v) => v - eta).filter((v) => v !== 0)
  const n = d.length
  if (n < 1) throw new RangeError('wilcoxonSigned needs at least 1 observation different from the hypothesized median')
  const abs = d.map(Math.abs)
  const order = abs.map((_, i) => i).sort((a, b) => abs[a]! - abs[b]!)
  const ranks = new Float64Array(n)
  let tieSum = 0
  let ties = false
  for (let i = 0; i < n; ) {
    let j = i
    while (j + 1 < n && abs[order[j + 1]!] === abs[order[i]!]) j++
    const t = j - i + 1
    if (t > 1) {
      ties = true
      tieSum += t * t * t - t
    }
    const r = (i + 1 + j + 1) / 2
    for (let p = i; p <= j; p++) ranks[order[p]!] = r
    i = j + 1
  }
  let wPlus = 0
  for (let i = 0; i < n; i++) if (d[i]! > 0) wPlus += ranks[i]!
  let method: 'exact' | 'asymptotic' = options.method === 'exact' ? 'exact' : 'asymptotic'
  if ((options.method ?? 'auto') === 'auto' && !ties && n <= 50) method = 'exact'
  if (method === 'exact' && ties) throw new RangeError('wilcoxonSigned: exact p-value is not defined with tied |x − median|; use method "asymptotic"')
  let pValue: number
  if (method === 'exact') {
    const counts = exactSignedRankCounts(n)
    const total = Math.pow(2, n)
    const cdf = (w: number) => {
      let s = 0
      for (let k = 0; k <= Math.min(w, counts.length - 1); k++) s += counts[k]!
      return s / total
    }
    const sf = (w: number) => 1 - cdf(w - 1)
    if (alternative === 'greater') pValue = sf(wPlus)
    else if (alternative === 'less') pValue = cdf(wPlus)
    else pValue = Math.min(1, 2 * Math.min(cdf(wPlus), sf(wPlus)))
  } else {
    const mu = (n * (n + 1)) / 4
    const sigma = Math.sqrt((n * (n + 1) * (2 * n + 1)) / 24 - tieSum / 48)
    if (alternative === 'greater') pValue = STD.sf((wPlus - mu - 0.5) / sigma)
    else if (alternative === 'less') pValue = STD.cdf((wPlus - mu + 0.5) / sigma)
    else pValue = Math.min(1, 2 * STD.sf(Math.max(0, Math.abs(wPlus - mu) - 0.5) / sigma))
  }
  // Walsh averages of the full (zero-included) sample for the estimate and CI — k-th order statistics
  // of the implicit set (O(n log n) instead of materialising n(n+1)/2 values)
  const sortedAll = all.slice().sort()
  const m = sortedAll.length
  const M = (m * (m + 1)) / 2
  const walshAt = (kk: number) => kthWalshAverage(sortedAll, kk)
  const estimate = M % 2 ? walshAt((M + 1) / 2) : 0.5 * (walshAt(M / 2) + walshAt(M / 2 + 1))
  const sd = Math.sqrt((m * (m + 1) * (2 * m + 1)) / 24)
  const muW = (m * (m + 1)) / 4
  let ci: [number, number]
  if (alternative === 'two-sided') {
    let kk = Math.floor(muW - STD.ppf(0.5 + confidence / 2) * sd) // Minitab: k-th Walsh averages from each end
    if (kk < 1) kk = 1
    ci = [walshAt(kk), walshAt(M - kk + 1)]
  } else {
    let kk = Math.floor(muW - STD.ppf(confidence) * sd)
    if (kk < 1) kk = 1
    ci = alternative === 'greater' ? [walshAt(kk), Infinity] : [-Infinity, walshAt(M - kk + 1)]
  }
  return { test: '1-sample Wilcoxon', statistic: wPlus, estimate, pValue, method, alternative, ci, confidence, n, nTotal }
}

// ---- Mood's median test -------------------------------------------------------------------------------------

export interface MoodResult {
  test: "Mood's median"
  statistic: number
  df: number
  pValue: number
  /** Overall (grand) median. */
  grandMedian: number
  n: number
  groups: Array<{ name: string; n: number; median: number; nBelowOrEqual: number; nAbove: number; q1: number; q3: number }>
}

function quantile7(sorted: ArrayLike<number>, q: number): number {
  const n = sorted.length
  if (n === 0) return NaN
  const pos = (n - 1) * q
  const lo = Math.floor(pos)
  const hi = Math.ceil(pos)
  return lo === hi ? sorted[lo]! : sorted[lo]! * (hi - pos) + sorted[hi]! * (pos - lo)
}

/**
 * Mood's median test (Minitab): χ² test on the k×2 table of counts ≤ grand median / > grand median
 * (scipy `median_test(ties='below')`).
 */
export function moodMedian(
  groups: Record<string, ArrayLike<number | null | undefined>> | Array<ArrayLike<number | null | undefined>>,
): MoodResult {
  const entries = Array.isArray(groups) ? groups.map((g, i) => [String(i), g] as const) : Object.entries(groups)
  const gs = entries.map(([name, g]) => ({ name, v: cleanNumbers(g).slice().sort() })).filter((g) => g.v.length > 0)
  const k = gs.length
  if (k < 2) throw new RangeError(`moodMedian needs at least 2 non-empty groups, got ${k}`)
  const all = Float64Array.from(gs.flatMap((g) => Array.from(g.v))).sort()
  const grand = median(all)
  const N = all.length
  let totalAbove = 0
  const rows = gs.map((g) => {
    let above = 0
    for (let i = 0; i < g.v.length; i++) if (g.v[i]! > grand) above++
    totalAbove += above
    return { name: g.name, n: g.v.length, median: median(g.v), nBelowOrEqual: g.v.length - above, nAbove: above, q1: quantile7(g.v, 0.25), q3: quantile7(g.v, 0.75) }
  })
  const pAbove = totalAbove / N
  let statistic = 0
  for (const r of rows) {
    const eAbove = r.n * pAbove
    const eBelow = r.n - eAbove
    if (eAbove > 0) statistic += (r.nAbove - eAbove) ** 2 / eAbove
    if (eBelow > 0) statistic += (r.nBelowOrEqual - eBelow) ** 2 / eBelow
  }
  const df = k - 1
  return { test: "Mood's median", statistic, df, pValue: chi2Dist(df).sf(statistic), grandMedian: grand, n: N, groups: rows }
}

// ---- Friedman -----------------------------------------------------------------------------------------------

export interface FriedmanResult {
  test: 'Friedman'
  /** S adjusted for ties (Minitab), equal to scipy `friedmanchisquare`. */
  statistic: number
  sUnadjusted: number
  df: number
  pValue: number
  blocks: number
  treatments: Array<{ name: string; n: number; median: number; sumRanks: number }>
}

/**
 * Friedman test for a randomized block design (Minitab): rows = blocks, columns = treatments; ranks
 * within each block, S = 12/(bk(k+1)) Σ Rⱼ² − 3b(k+1), divided by the tie correction 1 − Σ(t³−t)/(bk(k²−1)).
 */
export function friedman(table: number[][], treatmentNames?: string[]): FriedmanResult {
  const b = table.length
  const k = b ? table[0]!.length : 0
  if (b < 2 || k < 2) throw new RangeError(`friedman needs at least 2 blocks and 2 treatments, got ${b}×${k}`)
  const sumRanks = new Float64Array(k)
  let tieSum = 0
  for (const row of table) {
    if (row.length !== k) throw new RangeError('friedman: ragged table')
    const order = row.map((_, i) => i).sort((p, q) => row[p]! - row[q]!)
    for (let i = 0; i < k; ) {
      let j = i
      while (j + 1 < k && row[order[j + 1]!] === row[order[i]!]) j++
      const t = j - i + 1
      if (t > 1) tieSum += t * t * t - t
      const r = (i + 1 + j + 1) / 2
      for (let p = i; p <= j; p++) sumRanks[order[p]!] += r
      i = j + 1
    }
  }
  let s = 0
  for (let j = 0; j < k; j++) s += sumRanks[j]! * sumRanks[j]!
  const sUnadjusted = (12 / (b * k * (k + 1))) * s - 3 * b * (k + 1)
  const correction = 1 - tieSum / (b * k * (k * k - 1))
  const statistic = correction > 0 ? sUnadjusted / correction : NaN
  const df = k - 1
  const treatments = Array.from({ length: k }, (_, j) => {
    const col = Float64Array.from(table, (row) => row[j]!).sort()
    return { name: treatmentNames?.[j] ?? String(j), n: b, median: median(col), sumRanks: sumRanks[j]! }
  })
  return { test: 'Friedman', statistic, sUnadjusted, df, pValue: chi2Dist(df).sf(statistic), blocks: b, treatments }
}

// ---- Runs test ------------------------------------------------------------------------------------------------

export interface RunsTestResult {
  test: 'runs'
  /** Observed number of runs about the reference value. */
  runs: number
  expected: number
  statistic: number
  pValue: number
  /** Reference value (mean by default) and counts above / below it. */
  k: number
  nAbove: number
  nBelow: number
  n: number
}

/**
 * Runs test for randomness (Minitab): runs of observations above and below `k` (the mean by default;
 * values equal to k count as below), normal approximation with μ = 2n₁n₂/n + 1,
 * σ² = 2n₁n₂(2n₁n₂ − n)/(n²(n − 1)); `correction: true` applies the ±0.5 continuity correction.
 */
export function runsTest(x: ArrayLike<number | null | undefined>, options: { k?: number; correction?: boolean } = {}): RunsTestResult {
  const v = cleanNumbers(x)
  const n = v.length
  if (n < 2) throw new RangeError(`runsTest needs at least 2 observations, got ${n}`)
  let k = options.k
  if (k === undefined) {
    let s = 0
    for (let i = 0; i < n; i++) s += v[i]!
    k = s / n
  }
  let runs = 1
  let nAbove = 0
  let prev = v[0]! > k
  if (prev) nAbove++
  for (let i = 1; i < n; i++) {
    const cur = v[i]! > k
    if (cur) nAbove++
    if (cur !== prev) runs++
    prev = cur
  }
  const nBelow = n - nAbove
  const expected = (2 * nAbove * nBelow) / n + 1
  const variance = (2 * nAbove * nBelow * (2 * nAbove * nBelow - n)) / (n * n * (n - 1))
  let statistic: number
  let pValue: number
  if (!(variance > 0)) {
    statistic = NaN
    pValue = 1
  } else {
    const sigma = Math.sqrt(variance)
    let diff = runs - expected
    if (options.correction) diff = diff > 0.5 ? diff - 0.5 : diff < -0.5 ? diff + 0.5 : 0
    statistic = diff / sigma
    pValue = Math.min(1, 2 * STD.sf(Math.abs(statistic)))
  }
  return { test: 'runs', runs, expected, statistic, pValue, k, nAbove, nBelow, n }
}
