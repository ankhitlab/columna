/**
 * Ready-made hypothesis tests on plain numeric arrays (nulls / NaN are skipped): one-sample,
 * two-sample (Welch or pooled) and paired t-tests, one-way ANOVA, chi-square test of independence
 * on a contingency table and chi-square goodness of fit. DataFrame / LazyFrame expose the same
 * tests on columns (`df.ttest('x', { mu: 50 })`, `df.anova('x', 'group')`, `df.chi2test('a', 'b')`).
 *
 * Numbers match scipy.stats (`ttest_1samp`, `ttest_ind`, `ttest_rel`, `f_oneway`,
 * `chi2_contingency`, `chisquare`) and Minitab's Basic Statistics / ANOVA / Tables output.
 */
import { chi2 as chi2Dist, f as fDist, normal as normalDist, t as tDist } from './dist.js'

export type Alternative = 'two-sided' | 'less' | 'greater'

export interface TTestOptions {
  /** Hypothesized mean (one-sample / paired) or mean difference (two-sample). Default 0. */
  mu?: number
  alternative?: Alternative
  /** Confidence level for the interval, default 0.95. */
  confidence?: number
}

export interface TTestResult {
  test: 'one-sample t' | "two-sample t (Welch)" | 'two-sample t (pooled)' | 'paired t'
  /** Sample mean, or mean difference for two-sample / paired. */
  estimate: number
  /** Standard error of the estimate. */
  se: number
  statistic: number
  df: number
  pValue: number
  alternative: Alternative
  /** Confidence interval for the mean (difference); one-sided alternatives give a half-open interval. */
  ci: [number, number]
  confidence: number
  n: number
  /** Per-sample descriptives (one entry for one-sample / paired, two for two-sample). */
  samples: Array<{ n: number; mean: number; sd: number }>
}

export interface AnovaResult {
  test: 'one-way ANOVA'
  statistic: number
  pValue: number
  dfBetween: number
  dfWithin: number
  ssBetween: number
  ssWithin: number
  msBetween: number
  msWithin: number
  /** Proportion of variance explained, SS_between / SS_total. */
  etaSquared: number
  n: number
  groups: Array<{ name: string; n: number; mean: number; sd: number }>
}

export interface Chi2Result {
  test: 'chi-square independence' | 'chi-square goodness of fit'
  statistic: number
  df: number
  pValue: number
  /** Expected counts under H0 (matrix for independence, vector for goodness of fit). */
  expected: number[][] | number[]
  /** True when any expected count is below 5 — the chi-square approximation is then questionable. */
  lowExpected: boolean
  /** Yates continuity correction applied (2×2 tables only). */
  correction?: boolean
  /** Row / column labels when built from DataFrame columns. */
  rows?: string[]
  cols?: string[]
}

// ---- helpers --------------------------------------------------------------------------------------

/** Finite numbers only; nulls, undefined and NaN are dropped. */
export function cleanNumbers(xs: ArrayLike<number | null | undefined>): Float64Array {
  const out = new Float64Array(xs.length)
  let k = 0
  for (let i = 0; i < xs.length; i++) {
    const v = xs[i]
    if (typeof v === 'number' && Number.isFinite(v)) out[k++] = v
  }
  return out.subarray(0, k)
}

function moments(x: Float64Array): { n: number; mean: number; sd: number; variance: number } {
  const n = x.length
  if (n === 0) return { n: 0, mean: NaN, sd: NaN, variance: NaN }
  let s = 0
  for (let i = 0; i < n; i++) s += x[i]!
  const mean = s / n
  let m2 = 0
  for (let i = 0; i < n; i++) {
    const d = x[i]! - mean
    m2 += d * d
  }
  const variance = n > 1 ? m2 / (n - 1) : NaN
  return { n, mean, sd: Math.sqrt(variance), variance }
}

function tPValue(t: number, df: number, alternative: Alternative): number {
  const d = tDist(df)
  if (alternative === 'less') return d.cdf(t)
  if (alternative === 'greater') return d.sf(t)
  return 2 * d.sf(Math.abs(t))
}

function tInterval(estimate: number, se: number, df: number, confidence: number, alternative: Alternative): [number, number] {
  const d = tDist(df)
  if (alternative === 'less') return [-Infinity, estimate + d.ppf(confidence) * se]
  if (alternative === 'greater') return [estimate - d.ppf(confidence) * se, Infinity]
  const h = d.ppf(0.5 + confidence / 2) * se
  return [estimate - h, estimate + h]
}

function checkOptions(o: TTestOptions): Required<TTestOptions> {
  const confidence = o.confidence ?? 0.95
  if (!(confidence > 0 && confidence < 1)) throw new RangeError(`confidence must be in (0, 1), got ${confidence}`)
  const alternative = o.alternative ?? 'two-sided'
  if (alternative !== 'two-sided' && alternative !== 'less' && alternative !== 'greater') {
    throw new RangeError(`alternative must be 'two-sided' | 'less' | 'greater', got ${String(alternative)}`)
  }
  return { mu: o.mu ?? 0, alternative, confidence }
}

// ---- t-tests --------------------------------------------------------------------------------------

/** One-sample t-test of H0: mean = mu (scipy `ttest_1samp`, Minitab 1-Sample t). */
export function ttest1(x: ArrayLike<number | null | undefined>, options: TTestOptions = {}): TTestResult {
  const { mu, alternative, confidence } = checkOptions(options)
  const v = cleanNumbers(x)
  const m = moments(v)
  if (m.n < 2) throw new RangeError(`ttest1 needs at least 2 observations, got ${m.n}`)
  const se = m.sd / Math.sqrt(m.n)
  const df = m.n - 1
  const statistic = (m.mean - mu) / se
  return {
    test: 'one-sample t',
    estimate: m.mean,
    se,
    statistic,
    df,
    pValue: tPValue(statistic, df, alternative),
    alternative,
    ci: tInterval(m.mean, se, df, confidence, alternative),
    confidence,
    n: m.n,
    samples: [{ n: m.n, mean: m.mean, sd: m.sd }],
  }
}

/**
 * Two-sample t-test of H0: mean(a) − mean(b) = mu. Welch (unequal variances) by default like
 * Minitab; `equalVar: true` pools the variances (scipy `ttest_ind` default).
 */
export function ttest2(
  a: ArrayLike<number | null | undefined>,
  b: ArrayLike<number | null | undefined>,
  options: TTestOptions & { equalVar?: boolean } = {},
): TTestResult {
  const { mu, alternative, confidence } = checkOptions(options)
  const x = moments(cleanNumbers(a))
  const y = moments(cleanNumbers(b))
  if (x.n < 2 || y.n < 2) throw new RangeError(`ttest2 needs at least 2 observations per sample, got ${x.n} and ${y.n}`)
  const estimate = x.mean - y.mean
  let se: number
  let df: number
  if (options.equalVar) {
    const sp2 = ((x.n - 1) * x.variance + (y.n - 1) * y.variance) / (x.n + y.n - 2)
    se = Math.sqrt(sp2 * (1 / x.n + 1 / y.n))
    df = x.n + y.n - 2
  } else {
    const vx = x.variance / x.n
    const vy = y.variance / y.n
    se = Math.sqrt(vx + vy)
    df = ((vx + vy) * (vx + vy)) / ((vx * vx) / (x.n - 1) + (vy * vy) / (y.n - 1))
  }
  const statistic = (estimate - mu) / se
  return {
    test: options.equalVar ? 'two-sample t (pooled)' : 'two-sample t (Welch)',
    estimate,
    se,
    statistic,
    df,
    pValue: tPValue(statistic, df, alternative),
    alternative,
    ci: tInterval(estimate, se, df, confidence, alternative),
    confidence,
    n: x.n + y.n,
    samples: [
      { n: x.n, mean: x.mean, sd: x.sd },
      { n: y.n, mean: y.mean, sd: y.sd },
    ],
  }
}

/** Paired t-test on a − b (pairs with a missing value are dropped); scipy `ttest_rel`, Minitab Paired t. */
export function ttestPaired(
  a: ArrayLike<number | null | undefined>,
  b: ArrayLike<number | null | undefined>,
  options: TTestOptions = {},
): TTestResult {
  if (a.length !== b.length) throw new RangeError(`ttestPaired needs equal-length samples, got ${a.length} and ${b.length}`)
  const d: number[] = []
  for (let i = 0; i < a.length; i++) {
    const u = a[i]
    const v = b[i]
    if (typeof u === 'number' && typeof v === 'number' && Number.isFinite(u) && Number.isFinite(v)) d.push(u - v)
  }
  const r = ttest1(d, options)
  return { ...r, test: 'paired t' }
}

// ---- ANOVA ----------------------------------------------------------------------------------------

/** One-way ANOVA across named groups (scipy `f_oneway`, Minitab One-Way ANOVA). */
export function anova(groups: Record<string, ArrayLike<number | null | undefined>> | Array<ArrayLike<number | null | undefined>>): AnovaResult {
  const entries = Array.isArray(groups) ? groups.map((g, i) => [String(i), g] as const) : Object.entries(groups)
  const stats = entries
    .map(([name, g]) => ({ name, ...moments(cleanNumbers(g)) }))
    .filter((s) => s.n > 0)
  const k = stats.length
  if (k < 2) throw new RangeError(`anova needs at least 2 non-empty groups, got ${k}`)
  let n = 0
  let sum = 0
  for (const s of stats) {
    n += s.n
    sum += s.mean * s.n
  }
  if (n <= k) throw new RangeError(`anova needs more observations (${n}) than groups (${k})`)
  const grand = sum / n
  let ssBetween = 0
  let ssWithin = 0
  for (const s of stats) {
    ssBetween += s.n * (s.mean - grand) ** 2
    if (s.n > 1) ssWithin += (s.n - 1) * s.variance
  }
  const dfBetween = k - 1
  const dfWithin = n - k
  const msBetween = ssBetween / dfBetween
  const msWithin = ssWithin / dfWithin
  const statistic = msBetween / msWithin
  return {
    test: 'one-way ANOVA',
    statistic,
    pValue: fDist(dfBetween, dfWithin).sf(statistic),
    dfBetween,
    dfWithin,
    ssBetween,
    ssWithin,
    msBetween,
    msWithin,
    etaSquared: ssBetween / (ssBetween + ssWithin),
    n,
    groups: stats.map((s) => ({ name: s.name, n: s.n, mean: s.mean, sd: s.sd })),
  }
}

// ---- chi-square -----------------------------------------------------------------------------------

/**
 * Chi-square test of independence on an r×c contingency table of counts (scipy `chi2_contingency`
 * with `correction=false` by default, Minitab Chi-Square Test for Association).
 */
export function chi2test(observed: number[][], options: { correction?: boolean } = {}): Chi2Result {
  const r = observed.length
  const c = r ? observed[0]!.length : 0
  if (r < 2 || c < 2) throw new RangeError(`chi2test needs at least a 2×2 table, got ${r}×${c}`)
  const rowSum = observed.map((row) => {
    if (row.length !== c) throw new RangeError('chi2test: ragged table')
    return row.reduce((a, b) => a + b, 0)
  })
  const colSum = Array.from({ length: c }, (_, j) => observed.reduce((a, row) => a + row[j]!, 0))
  const total = rowSum.reduce((a, b) => a + b, 0)
  if (!(total > 0)) throw new RangeError('chi2test: table has no counts')
  const expected = rowSum.map((rs) => colSum.map((cs) => (rs * cs) / total))
  const correction = Boolean(options.correction) && r === 2 && c === 2
  let statistic = 0
  let lowExpected = false
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      const e = expected[i]![j]!
      if (e < 5) lowExpected = true
      if (e === 0) continue
      let d = Math.abs(observed[i]![j]! - e)
      if (correction) d = Math.max(0, d - 0.5)
      statistic += (d * d) / e
    }
  }
  const df = (r - 1) * (c - 1)
  return { test: 'chi-square independence', statistic, df, pValue: chi2Dist(df).sf(statistic), expected, lowExpected, correction }
}

/**
 * Chi-square goodness of fit (scipy `chisquare`, Minitab Chi-Square Goodness-of-Fit). `expected` may be
 * counts or probabilities (they are rescaled to the observed total); omitted → uniform. `ddof` reduces
 * the degrees of freedom by the number of parameters estimated from the data.
 */
export function chi2gof(observed: number[], expected?: number[], options: { ddof?: number } = {}): Chi2Result {
  const k = observed.length
  if (k < 2) throw new RangeError(`chi2gof needs at least 2 categories, got ${k}`)
  const total = observed.reduce((a, b) => a + b, 0)
  let exp: number[]
  if (!expected) exp = observed.map(() => total / k)
  else {
    if (expected.length !== k) throw new RangeError(`chi2gof: expected has ${expected.length} entries, observed ${k}`)
    const es = expected.reduce((a, b) => a + b, 0)
    exp = expected.map((e) => (e * total) / es)
  }
  let statistic = 0
  let lowExpected = false
  for (let i = 0; i < k; i++) {
    const e = exp[i]!
    if (e < 5) lowExpected = true
    if (e === 0) continue
    statistic += (observed[i]! - e) ** 2 / e
  }
  const df = k - 1 - (options.ddof ?? 0)
  if (df < 1) throw new RangeError(`chi2gof: degrees of freedom ${df} < 1`)
  return { test: 'chi-square goodness of fit', statistic, df, pValue: chi2Dist(df).sf(statistic), expected: exp, lowExpected }
}

/** Contingency table of two label arrays (row labels × column labels, sorted), for `chi2test`. */
export function crosstab(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
): { rows: string[]; cols: string[]; table: number[][] } {
  if (a.length !== b.length) throw new RangeError(`crosstab needs equal-length columns, got ${a.length} and ${b.length}`)
  const rowIdx = new Map<string, number>()
  const colIdx = new Map<string, number>()
  const counts = new Map<string, number>()
  for (let i = 0; i < a.length; i++) {
    const u = a[i]
    const v = b[i]
    if (u === null || u === undefined || v === null || v === undefined) continue
    const ku = String(u)
    const kv = String(v)
    if (!rowIdx.has(ku)) rowIdx.set(ku, rowIdx.size)
    if (!colIdx.has(kv)) colIdx.set(kv, colIdx.size)
    const key = ku + '\0' + kv
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  const rows = [...rowIdx.keys()].sort()
  const cols = [...colIdx.keys()].sort()
  const table = rows.map((r) => cols.map((c) => counts.get(r + '\0' + c) ?? 0))
  return { rows, cols, table }
}

export const stats = { ttest1, ttest2, ttestPaired, anova, chi2test, chi2gof, crosstab, levene, bartlett, bonett, bonett2, varTest2, equalVariances }

// ---- equal variances --------------------------------------------------------------------------------

export interface VarianceTestResult {
  test: 'Levene' | 'Brown-Forsythe' | 'Bartlett' | 'Bonett' | 'F-test (2 variances)'
  statistic: number
  pValue: number
  df: number | [number, number]
  n: number
  groups: Array<{ name: string; n: number; sd: number; variance: number }>
  /** F-test only: ratio s1²/s2² with its confidence interval. */
  ratio?: number
  ci?: [number, number]
}

function groupEntries(groups: Record<string, ArrayLike<number | null | undefined>> | Array<ArrayLike<number | null | undefined>>) {
  const entries = Array.isArray(groups) ? groups.map((g, i) => [String(i), g] as const) : Object.entries(groups)
  return entries.map(([name, g]) => ({ name, v: cleanNumbers(g) })).filter((e) => e.v.length > 0)
}

/**
 * Levene's test for equal variances: one-way ANOVA on |x − center|. `center: 'median'` (default,
 * Brown–Forsythe — what Minitab and scipy call "Levene") is robust to non-normality; 'mean' is the
 * original Levene statistic.
 */
export function levene(
  groups: Record<string, ArrayLike<number | null | undefined>> | Array<ArrayLike<number | null | undefined>>,
  options: { center?: 'median' | 'mean' } = {},
): VarianceTestResult {
  const center = options.center ?? 'median'
  const gs = groupEntries(groups)
  const k = gs.length
  if (k < 2) throw new RangeError(`levene needs at least 2 non-empty groups, got ${k}`)
  const dev: Record<string, number[]> = {}
  const summary: VarianceTestResult['groups'] = []
  for (const { name, v } of gs) {
    const m = moments(v)
    let c: number
    if (center === 'mean') c = m.mean
    else {
      const s = v.slice().sort()
      const h = s.length >> 1
      c = s.length % 2 ? s[h]! : (s[h - 1]! + s[h]!) / 2
    }
    dev[name] = Array.from(v, (x) => Math.abs(x - c))
    summary.push({ name, n: m.n, sd: m.sd, variance: m.variance })
  }
  const a = anova(dev)
  return {
    test: center === 'median' ? 'Brown-Forsythe' : 'Levene',
    statistic: a.statistic,
    pValue: a.pValue,
    df: [a.dfBetween, a.dfWithin],
    n: a.n,
    groups: summary,
  }
}

/** Bartlett's test for equal variances (assumes normality; scipy `bartlett`, Minitab "Bartlett's test"). */
export function bartlett(
  groups: Record<string, ArrayLike<number | null | undefined>> | Array<ArrayLike<number | null | undefined>>,
): VarianceTestResult {
  const gs = groupEntries(groups).filter((g) => g.v.length > 1)
  const k = gs.length
  if (k < 2) throw new RangeError(`bartlett needs at least 2 groups with ≥ 2 observations, got ${k}`)
  let N = 0
  let sumLog = 0
  let sumInv = 0
  let pooled = 0
  const summary: VarianceTestResult['groups'] = []
  for (const { name, v } of gs) {
    const m = moments(v)
    N += m.n
    pooled += (m.n - 1) * m.variance
    sumLog += (m.n - 1) * Math.log(m.variance)
    sumInv += 1 / (m.n - 1)
    summary.push({ name, n: m.n, sd: m.sd, variance: m.variance })
  }
  const sp2 = pooled / (N - k)
  const statistic = ((N - k) * Math.log(sp2) - sumLog) / (1 + (sumInv - 1 / (N - k)) / (3 * (k - 1)))
  return { test: 'Bartlett', statistic, pValue: chi2Dist(k - 1).sf(statistic), df: k - 1, n: N, groups: summary }
}

/** Two-sample F-test of σ₁² = σ₂² (Minitab 2 Variances, F method) with a confidence interval for the ratio. */
export function varTest2(
  a: ArrayLike<number | null | undefined>,
  b: ArrayLike<number | null | undefined>,
  options: { alternative?: Alternative; confidence?: number } = {},
): VarianceTestResult {
  const { alternative, confidence } = checkOptions(options)
  const x = moments(cleanNumbers(a))
  const y = moments(cleanNumbers(b))
  if (x.n < 2 || y.n < 2) throw new RangeError(`varTest2 needs at least 2 observations per sample, got ${x.n} and ${y.n}`)
  const ratio = x.variance / y.variance
  const d = fDist(x.n - 1, y.n - 1)
  const pValue = alternative === 'less' ? d.cdf(ratio) : alternative === 'greater' ? d.sf(ratio) : 2 * Math.min(d.cdf(ratio), d.sf(ratio))
  let ci: [number, number]
  if (alternative === 'less') ci = [0, ratio / d.ppf(1 - confidence)]
  else if (alternative === 'greater') ci = [ratio / d.ppf(confidence), Infinity]
  else ci = [ratio / d.ppf(0.5 + confidence / 2), ratio / d.ppf(0.5 - confidence / 2)]
  return {
    test: 'F-test (2 variances)',
    statistic: ratio,
    pValue: Math.min(1, pValue),
    df: [x.n - 1, y.n - 1],
    n: x.n + y.n,
    groups: [
      { name: '0', n: x.n, sd: x.sd, variance: x.variance },
      { name: '1', n: y.n, sd: y.sd, variance: y.variance },
    ],
    ratio,
    ci,
  }
}

/** Dispatch: Minitab's Test for Equal Variances — Levene (Brown–Forsythe) by default, Bartlett, or Bonett (multiple comparisons). */
export function equalVariances(
  groups: Record<string, ArrayLike<number | null | undefined>> | Array<ArrayLike<number | null | undefined>>,
  method: 'levene' | 'bartlett' | 'bonett' = 'levene',
): VarianceTestResult {
  return method === 'bartlett' ? bartlett(groups) : method === 'bonett' ? bonett(groups) : levene(groups)
}

// ---- Bonett's method ----------------------------------------------------------------------------------

/** Mean after trimming proportion p from each end (Bonett uses p = 1/(2√(n − 4)) for the kurtosis estimate). */
function trimmedMean(sorted: Float64Array, p: number): number {
  const n = sorted.length
  const g = Math.floor(n * p)
  let s = 0
  for (let i = g; i < n - g; i++) s += sorted[i]!
  return s / (n - 2 * g)
}

/** Pooled kurtosis estimate γ̂ = N·ΣΣ(x − m̃ᵢ)⁴ / [ΣΣ(x − m̃ᵢ)²]², m̃ᵢ trimmed means (Bonett 2006). */
function pooledKurtosis(samples: Float64Array[]): number {
  let N = 0
  let s4 = 0
  let s2 = 0
  for (const v of samples) {
    const n = v.length
    N += n
    const sorted = v.slice().sort()
    const m = n > 4 ? trimmedMean(sorted, 1 / (2 * Math.sqrt(n - 4))) : trimmedMean(sorted, 0)
    for (let i = 0; i < n; i++) {
      const d2 = (v[i]! - m) ** 2
      s2 += d2
      s4 += d2 * d2
    }
  }
  return (N * s4) / (s2 * s2)
}

/**
 * Bonett's test for equal variances (k ≥ 2 groups): Layard's statistic on ln sᵢ² with the
 * kurtosis-adjusted variances of Bonett (2006), T = Σ wᵢ (ln sᵢ² − c̄)², wᵢ = (nᵢ − 1)/(γ̂ − (nᵢ − 3)/nᵢ),
 * χ² with k − 1 df — robust to non-normality, the basis of Minitab's "multiple comparisons" test.
 */
export function bonett(
  groups: Record<string, ArrayLike<number | null | undefined>> | Array<ArrayLike<number | null | undefined>>,
): VarianceTestResult & { kurtosis: number } {
  const gs = groupEntries(groups).filter((g) => g.v.length > 1)
  const k = gs.length
  if (k < 2) throw new RangeError(`bonett needs at least 2 groups with ≥ 2 observations, got ${k}`)
  const kurt = pooledKurtosis(gs.map((g) => g.v))
  const summary: VarianceTestResult['groups'] = []
  let sw = 0
  let swl = 0
  const rows: Array<{ w: number; l: number }> = []
  let N = 0
  for (const { name, v } of gs) {
    const m = moments(v)
    N += m.n
    const w = (m.n - 1) / (kurt - (m.n - 3) / m.n)
    const l = Math.log(m.variance)
    rows.push({ w, l })
    sw += w
    swl += w * l
    summary.push({ name, n: m.n, sd: m.sd, variance: m.variance })
  }
  const cbar = swl / sw
  let statistic = 0
  for (const r of rows) statistic += r.w * (r.l - cbar) ** 2
  return { test: 'Bonett', statistic, pValue: chi2Dist(k - 1).sf(statistic), df: k - 1, n: N, groups: summary, kurtosis: kurt }
}

/**
 * Bonett's two-sample test of σ₁² = σ₂² with the small-sample correction c(α) (Minitab 2 Variances,
 * Bonett's method): CI for the ratio exp[ln(c·s₁²/s₂²) ± z_{α/2}·√(se₁² + se₂²)],
 * seᵢ² = (γ̂ − (nᵢ − 3)/nᵢ)/(nᵢ − 1), c = [n₁/(n₁ − z_{α/2})] / [n₂/(n₂ − z_{α/2})]. The p-value is the
 * smallest α at which the (1 − α) interval excludes 1, so p and CI always agree.
 */
export function bonett2(
  a: ArrayLike<number | null | undefined>,
  b: ArrayLike<number | null | undefined>,
  options: { confidence?: number } = {},
): VarianceTestResult & { kurtosis: number } {
  const confidence = options.confidence ?? 0.95
  if (!(confidence > 0 && confidence < 1)) throw new RangeError(`confidence must be in (0, 1), got ${confidence}`)
  const x = cleanNumbers(a)
  const y = cleanNumbers(b)
  const mx = moments(x)
  const my = moments(y)
  if (mx.n < 2 || my.n < 2) throw new RangeError(`bonett2 needs at least 2 observations per sample, got ${mx.n} and ${my.n}`)
  const kurt = pooledKurtosis([x, y])
  const se = Math.sqrt((kurt - (mx.n - 3) / mx.n) / (mx.n - 1) + (kurt - (my.n - 3) / my.n) / (my.n - 1))
  const lnRatio = Math.log(mx.variance / my.variance)
  const norm = normalDist()
  const bounds = (alpha: number): [number, number] => {
    const z = norm.ppf(1 - alpha / 2)
    const c = mx.n / (mx.n - z) / (my.n / (my.n - z))
    const center = Math.log(c) + lnRatio
    return [Math.exp(center - z * se), Math.exp(center + z * se)]
  }
  const ci = bounds(1 - confidence)
  // p-value: smallest α whose interval excludes 1 (monotone in α) — bisection on α ∈ (0, 1)
  const excludes = (alpha: number) => {
    const [lo, hi] = bounds(alpha)
    return lo > 1 || hi < 1
  }
  let pValue: number
  if (!excludes(1 - 1e-12)) pValue = 1
  else {
    let lo = 1e-16
    let hi = 1
    for (let i = 0; i < 100; i++) {
      const mid = 0.5 * (lo + hi)
      if (excludes(mid)) hi = mid
      else lo = mid
      if (hi - lo < 1e-12 * hi) break
    }
    pValue = hi
  }
  return {
    test: 'Bonett',
    statistic: lnRatio / se,
    pValue,
    df: 1,
    n: mx.n + my.n,
    groups: [
      { name: '0', n: mx.n, sd: mx.sd, variance: mx.variance },
      { name: '1', n: my.n, sd: my.sd, variance: my.variance },
    ],
    ratio: mx.variance / my.variance,
    ci,
    kurtosis: kurt,
  }
}
