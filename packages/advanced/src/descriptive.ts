/**
 * Minitab Basic Statistics › Display / Store Descriptive Statistics and Graphical Summary, the Poisson
 * goodness-of-fit test, and plot-ready data for Main Effects / Interaction / Interval plots, boxplots,
 * dotplots, empirical CDFs and cause-and-effect (fishbone) diagrams.
 */
import { chi2 as chi2Dist, poisson, t as tDist } from './dist.js'
import { andersonDarling, type NormalityResult } from './normality.js'
import { signTest } from './nonparametric2.js'
import { cleanNumbers } from './tests.js'

export interface DescriptiveStats {
  name?: string
  n: number
  /** Missing / non-numeric values. */
  nMissing: number
  mean: number
  seMean: number
  sd: number
  variance: number
  /** Coefficient of variation, 100·s/x̄ (%). */
  coefVar: number
  sum: number
  sumSquares: number
  min: number
  q1: number
  median: number
  q3: number
  max: number
  range: number
  iqr: number
  mode: number[]
  nMode: number
  skewness: number
  kurtosis: number
  /** Mean of the squared successive differences (Minitab MSSD). */
  mssd: number
  /** 5 % trimmed mean. */
  trimmedMean: number
}

function quantile6(sorted: Float64Array, p: number): number {
  // Minitab: position p(n + 1), linear interpolation (Hyndman–Fan type 6)
  const n = sorted.length
  const pos = p * (n + 1)
  if (pos <= 1) return sorted[0]!
  if (pos >= n) return sorted[n - 1]!
  const lo = Math.floor(pos)
  const frac = pos - lo
  return sorted[lo - 1]! + frac * (sorted[lo]! - sorted[lo - 1]!)
}

function quantile7(sorted: Float64Array, p: number): number {
  // pandas / numpy / polars default: position (n − 1)p, linear interpolation (Hyndman–Fan type 7)
  const n = sorted.length
  const pos = (n - 1) * Math.min(1, Math.max(0, p))
  const lo = Math.floor(pos)
  const hi = Math.ceil(pos)
  if (lo === hi) return sorted[lo]!
  return sorted[lo]! + (pos - lo) * (sorted[hi]! - sorted[lo]!)
}

/**
 * Sample quantiles of a numeric array (nulls / NaN skipped).
 *   quantile(x, 0.25)                        // Minitab: position p(n + 1), clamped, linear interpolation (type 6)
 *   quantile(x, [0.25, 0.5, 0.75])           // several at once
 *   quantile(x, 0.25, { method: 'linear' })  // pandas / numpy default (type 7) — position (n − 1)p
 * Q1 of {1, 2, 3, 4} is 1.25 (Minitab) versus 1.75 (pandas). Same definitions as
 * `col('x').quantile(p, 'minitab')` in the DataFrame engine, with Minitab as the default here.
 */
export function quantile(x: ArrayLike<number | null | undefined>, p: number, options?: { method?: 'minitab' | 'linear' }): number
export function quantile(x: ArrayLike<number | null | undefined>, p: number[], options?: { method?: 'minitab' | 'linear' }): number[]
export function quantile(
  x: ArrayLike<number | null | undefined>,
  p: number | number[],
  options: { method?: 'minitab' | 'linear' } = {},
): number | number[] {
  const vals: number[] = []
  for (let i = 0; i < x.length; i++) {
    const v = x[i]
    if (v != null && Number.isFinite(v)) vals.push(v)
  }
  const sorted = Float64Array.from(vals).sort()
  const f = options.method === 'linear' ? quantile7 : quantile6
  const one = (q: number) => {
    if (!(q >= 0 && q <= 1)) throw new RangeError(`quantile: p must be in [0, 1], got ${q}`)
    return sorted.length ? f(sorted, q) : NaN
  }
  return Array.isArray(p) ? p.map(one) : one(p)
}

function describeOne(x: ArrayLike<number | null | undefined>, name?: string): DescriptiveStats {
  const v = cleanNumbers(x)
  const n = v.length
  if (n === 0) throw new RangeError(`descriptiveStats: no numeric values${name ? ` in ${name}` : ''}`)
  const sorted = v.slice().sort()
  let sum = 0
  for (let i = 0; i < n; i++) sum += v[i]!
  const mean = sum / n
  let m2 = 0
  let m3 = 0
  let m4 = 0
  let sumSq = 0
  for (let i = 0; i < n; i++) {
    const d = v[i]! - mean
    m2 += d * d
    m3 += d * d * d
    m4 += d * d * d * d
    sumSq += v[i]! * v[i]!
  }
  const variance = n > 1 ? m2 / (n - 1) : NaN
  const sd = Math.sqrt(variance)
  const skewness = n > 2 && sd > 0 ? ((n / ((n - 1) * (n - 2))) * m3) / sd ** 3 : NaN
  const kurtosis = n > 3 && sd > 0 ? ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * (m4 / sd ** 4) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3)) : NaN
  let mssd = 0
  for (let i = 1; i < n; i++) mssd += (v[i]! - v[i - 1]!) ** 2
  mssd = n > 1 ? mssd / (2 * (n - 1)) : NaN
  // mode(s): most frequent values (Minitab lists up to 4) — one pass over the sorted values
  let nMode = 0
  let mode: number[] = []
  for (let i = 0; i < n; ) {
    let j = i
    while (j + 1 < n && sorted[j + 1] === sorted[i]) j++
    const c = j - i + 1
    if (c > nMode) {
      nMode = c
      mode = [sorted[i]!]
    } else if (c === nMode && mode.length < 4) mode.push(sorted[i]!)
    i = j + 1
  }
  if (nMode < 2) mode = []
  // 5 % trimmed mean: drop round(0.05 n) from each end
  const trim = Math.round(0.05 * n)
  let ts = 0
  for (let i = trim; i < n - trim; i++) ts += sorted[i]!
  const trimmedMean = ts / (n - 2 * trim)
  const q1 = quantile6(sorted, 0.25)
  const median = quantile6(sorted, 0.5)
  const q3 = quantile6(sorted, 0.75)
  return {
    name,
    n,
    nMissing: x.length - n,
    mean,
    seMean: sd / Math.sqrt(n),
    sd,
    variance,
    coefVar: mean !== 0 ? (100 * sd) / Math.abs(mean) : NaN,
    sum,
    sumSquares: sumSq,
    min: sorted[0]!,
    q1,
    median,
    q3,
    max: sorted[n - 1]!,
    range: sorted[n - 1]! - sorted[0]!,
    iqr: q3 - q1,
    mode,
    nMode: mode.length ? nMode : 0,
    skewness,
    kurtosis,
    mssd,
    trimmedMean,
  }
}

/**
 * Display Descriptive Statistics (Minitab): one row per variable, optionally split by the levels of `by`.
 *   descriptiveStats(x)                        → DescriptiveStats
 *   descriptiveStats({ a, b }, { by: group })  → DescriptiveStats[] (variable × level)
 */
export function descriptiveStats(x: ArrayLike<number | null | undefined>, options?: { by?: ArrayLike<unknown> }): DescriptiveStats | DescriptiveStats[]
export function descriptiveStats(x: Record<string, ArrayLike<number | null | undefined>>, options?: { by?: ArrayLike<unknown> }): DescriptiveStats[]
export function descriptiveStats(
  x: ArrayLike<number | null | undefined> | Record<string, ArrayLike<number | null | undefined>>,
  options: { by?: ArrayLike<unknown> } = {},
): DescriptiveStats | DescriptiveStats[] {
  const single = Array.isArray(x) || ArrayBuffer.isView(x)
  const vars: Array<[string | undefined, ArrayLike<number | null | undefined>]> = single ? [[undefined, x as ArrayLike<number | null | undefined>]] : Object.entries(x as Record<string, ArrayLike<number | null | undefined>>)
  if (!options.by) {
    if (single) return describeOne(vars[0]![1])
    return vars.map(([name, col]) => describeOne(col, name))
  }
  const by = options.by
  // one pass: row index lists per level
  const rowsOf = new Map<string, number[]>()
  for (let i = 0; i < by.length; i++) {
    const v = by[i]
    if (v === null || v === undefined) continue
    const key = String(v)
    let list = rowsOf.get(key)
    if (!list) rowsOf.set(key, (list = []))
    list.push(i)
  }
  const levels = [...rowsOf.keys()].sort()
  const out: DescriptiveStats[] = []
  for (const [name, col] of vars) {
    if (col.length !== by.length) throw new RangeError('descriptiveStats: by must have the same length as the data')
    for (const level of levels) {
      const idx = rowsOf.get(level)!
      const sub: Array<number | null | undefined> = new Array(idx.length)
      for (let k = 0; k < idx.length; k++) sub[k] = col[idx[k]!]
      out.push(describeOne(sub, name === undefined ? level : `${name}[${level}]`))
    }
  }
  return out
}

export interface GraphicalSummary {
  stats: DescriptiveStats
  normality: NormalityResult
  confidence: number
  ci: { mean: [number, number]; median: [number, number]; sd: [number, number] }
  /** Histogram bins (Minitab-style equal-width, ~√n bins) with counts. */
  histogram: Array<{ from: number; to: number; count: number }>
  boxplot: BoxplotStats
}

/** Graphical Summary (Minitab): descriptives, Anderson–Darling normality, CIs for mean / median / σ, histogram and boxplot data. */
export function graphicalSummary(x: ArrayLike<number | null | undefined>, options: { confidence?: number; bins?: number } = {}): GraphicalSummary {
  const confidence = options.confidence ?? 0.95
  const stats = describeOne(x)
  const v = cleanNumbers(x)
  const n = v.length
  if (n < 8) throw new RangeError(`graphicalSummary needs at least 8 observations, got ${n}`)
  const normality = andersonDarling(v)
  const tc = tDist(n - 1).ppf(0.5 + confidence / 2)
  const c2 = chi2Dist(n - 1)
  const ci = {
    mean: [stats.mean - tc * stats.seMean, stats.mean + tc * stats.seMean] as [number, number],
    median: signTest(v, { confidence }).ci,
    sd: [Math.sqrt(((n - 1) * stats.variance) / c2.ppf(0.5 + confidence / 2)), Math.sqrt(((n - 1) * stats.variance) / c2.ppf(0.5 - confidence / 2))] as [number, number],
  }
  const bins = options.bins ?? Math.max(5, Math.min(30, Math.ceil(Math.sqrt(n))))
  const width = stats.range / bins || 1
  const histogram = Array.from({ length: bins }, (_, i) => ({ from: stats.min + i * width, to: stats.min + (i + 1) * width, count: 0 }))
  for (let i = 0; i < n; i++) {
    const k = Math.min(bins - 1, Math.floor((v[i]! - stats.min) / width))
    histogram[k]!.count++
  }
  return { stats, normality, confidence, ci, histogram, boxplot: boxplotStats(v) as BoxplotStats }
}

export interface PoissonGofResult {
  test: 'Poisson goodness-of-fit'
  mean: number
  n: number
  statistic: number
  df: number
  pValue: number
  /** Categories after pooling (Minitab pools tail categories with expected < 5). */
  categories: Array<{ label: string; observed: number; expected: number; contribution: number }>
}

/**
 * Goodness-of-Fit Test for Poisson (Minitab): χ² of observed counts against Poisson(λ̂) with tail
 * categories pooled to expected ≥ `minExpected` (default 5); df = categories − 2 (one for λ̂).
 * Pass either raw counts per unit, or `{ values, frequencies }` for a frequency table.
 */
export function poissonGof(
  data: ArrayLike<number | null | undefined> | { values: ArrayLike<number>; frequencies: ArrayLike<number> },
  options: { minExpected?: number } = {},
): PoissonGofResult {
  const minExpected = options.minExpected ?? 5
  const freq = new Map<number, number>()
  let n = 0
  let sum = 0
  if ('frequencies' in (data as object)) {
    const d = data as { values: ArrayLike<number>; frequencies: ArrayLike<number> }
    for (let i = 0; i < d.values.length; i++) {
      const k = d.values[i]!
      const f = d.frequencies[i]!
      if (!(Number.isInteger(k) && k >= 0 && f >= 0)) throw new RangeError('poissonGof: values must be non-negative integers with non-negative frequencies')
      freq.set(k, (freq.get(k) ?? 0) + f)
      n += f
      sum += k * f
    }
  } else {
    const v = cleanNumbers(data as ArrayLike<number | null | undefined>)
    for (let i = 0; i < v.length; i++) {
      const k = v[i]!
      if (!(Number.isInteger(k) && k >= 0)) throw new RangeError('poissonGof: counts must be non-negative integers')
      freq.set(k, (freq.get(k) ?? 0) + 1)
      n++
      sum += k
    }
  }
  if (n < 2) throw new RangeError('poissonGof needs at least 2 observations')
  const mean = sum / n
  const d = poisson(mean)
  let kMax = -Infinity
  for (const k of freq.keys()) if (k > kMax) kMax = k
  // expected for 0..kMax, with the last category as "≥ kMax"
  const rows: Array<{ lo: number; hi: number; observed: number; expected: number }> = []
  for (let k = 0; k <= kMax; k++) rows.push({ lo: k, hi: k, observed: freq.get(k) ?? 0, expected: n * (k === kMax ? d.sf(kMax - 1) : d.pmf(k)) })
  // pool from the upper tail downward, then the lower tail upward
  const pooled: typeof rows = []
  let acc: (typeof rows)[number] | null = null
  for (const r of rows) {
    if (acc) {
      acc = { lo: acc.lo, hi: r.hi, observed: acc.observed + r.observed, expected: acc.expected + r.expected }
      if (acc.expected >= minExpected) {
        pooled.push(acc)
        acc = null
      }
    } else if (r.expected < minExpected) acc = { ...r }
    else pooled.push({ ...r })
  }
  if (acc) {
    if (pooled.length) {
      const last = pooled[pooled.length - 1]!
      pooled[pooled.length - 1] = { lo: last.lo, hi: acc.hi, observed: last.observed + acc.observed, expected: last.expected + acc.expected }
    } else pooled.push(acc)
  }
  let statistic = 0
  const categories = pooled.map((r) => {
    const contribution = r.expected > 0 ? (r.observed - r.expected) ** 2 / r.expected : 0
    statistic += contribution
    const label = r.lo === r.hi ? String(r.lo) : r.hi === kMax ? `>=${r.lo}` : `${r.lo}-${r.hi}`
    return { label, observed: r.observed, expected: r.expected, contribution }
  })
  const df = Math.max(1, categories.length - 2)
  return { test: 'Poisson goodness-of-fit', mean, n, statistic, df, pValue: categories.length > 2 ? chi2Dist(df).sf(statistic) : NaN, categories }
}

// ---- plot data --------------------------------------------------------------------------------------------

export interface BoxplotStats {
  name?: string
  n: number
  min: number
  q1: number
  median: number
  q3: number
  max: number
  mean: number
  /** Whisker ends: last observations within 1.5·IQR of the box. */
  whiskerLow: number
  whiskerHigh: number
  outliers: number[]
}

function boxOne(v: Float64Array, name?: string): BoxplotStats {
  const sorted = v.slice().sort()
  const n = sorted.length
  const q1 = quantile6(sorted, 0.25)
  const q3 = quantile6(sorted, 0.75)
  const iqr = q3 - q1
  const lo = q1 - 1.5 * iqr
  const hi = q3 + 1.5 * iqr
  let whiskerLow = Infinity
  let whiskerHigh = -Infinity
  const outliers: number[] = []
  let sum = 0
  for (let i = 0; i < n; i++) {
    const x = sorted[i]!
    sum += x
    if (x < lo || x > hi) outliers.push(x)
    else {
      whiskerLow = Math.min(whiskerLow, x)
      whiskerHigh = Math.max(whiskerHigh, x)
    }
  }
  return { name, n, min: sorted[0]!, q1, median: quantile6(sorted, 0.5), q3, max: sorted[n - 1]!, mean: sum / n, whiskerLow, whiskerHigh, outliers }
}

/** Boxplot statistics (Minitab quartiles, 1.5·IQR whiskers, outliers), optionally per level of `by`. */
export function boxplotStats(x: ArrayLike<number | null | undefined>, options: { by?: ArrayLike<unknown> } = {}): BoxplotStats | BoxplotStats[] {
  if (!options.by) {
    const v = cleanNumbers(x)
    if (!v.length) throw new RangeError('boxplotStats: no numeric values')
    return boxOne(v)
  }
  return groupedNumeric(x, options.by).map(({ level, values }) => boxOne(values, level))
}

function groupedNumeric(x: ArrayLike<number | null | undefined>, by: ArrayLike<unknown>): Array<{ level: string; values: Float64Array }> {
  if (x.length !== by.length) throw new RangeError('by must have the same length as the data')
  const groups = new Map<string, number[]>()
  for (let i = 0; i < x.length; i++) {
    const v = x[i]
    if (by[i] === null || by[i] === undefined || typeof v !== 'number' || !Number.isFinite(v)) continue
    const k = String(by[i])
    let arr = groups.get(k)
    if (!arr) groups.set(k, (arr = []))
    arr.push(v)
  }
  return [...groups.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)).map(([level, values]) => ({ level, values: Float64Array.from(values) }))
}

export interface IntervalPlotRow {
  level: string
  n: number
  mean: number
  se: number
  ci: [number, number]
}

/** Interval Plot data (Minitab): per-level means with t confidence intervals (pooled σ by default, as Minitab's default). */
export function intervalPlot(x: ArrayLike<number | null | undefined>, by: ArrayLike<unknown>, options: { confidence?: number; pooled?: boolean } = {}): IntervalPlotRow[] {
  const confidence = options.confidence ?? 0.95
  const pooled = options.pooled ?? true
  const groups = groupedNumeric(x, by)
  const stats = groups.map((g) => {
    const n = g.values.length
    let s = 0
    for (const v of g.values) s += v
    const mean = s / n
    let ss = 0
    for (const v of g.values) ss += (v - mean) ** 2
    return { level: g.level, n, mean, ss }
  })
  const N = stats.reduce((a, g) => a + g.n, 0)
  const dfPooled = N - stats.length
  const sPooled = Math.sqrt(stats.reduce((a, g) => a + g.ss, 0) / dfPooled)
  return stats.map((g) => {
    const sd = pooled ? sPooled : Math.sqrt(g.ss / (g.n - 1))
    const df = pooled ? dfPooled : g.n - 1
    const se = sd / Math.sqrt(g.n)
    const tc = df > 0 ? tDist(df).ppf(0.5 + confidence / 2) : NaN
    return { level: g.level, n: g.n, mean: g.mean, se, ci: [g.mean - tc * se, g.mean + tc * se] }
  })
}

export interface MainEffectsPlot {
  grandMean: number
  factors: Array<{ factor: string; levels: Array<{ level: string; n: number; mean: number }> }>
}

/** Main Effects Plot data (Minitab): level means per factor plus the grand-mean reference line. */
export function mainEffectsPlot(y: ArrayLike<number | null | undefined>, factors: Record<string, ArrayLike<unknown>>): MainEffectsPlot {
  const v = cleanNumbers(y)
  let s = 0
  for (const t of v) s += t
  const grandMean = s / v.length
  return {
    grandMean,
    factors: Object.entries(factors).map(([factor, col]) => ({
      factor,
      levels: groupedNumeric(y, col).map((g) => {
        let sum = 0
        for (const t of g.values) sum += t
        return { level: g.level, n: g.values.length, mean: sum / g.values.length }
      }),
    })),
  }
}

export interface InteractionPlot {
  a: string
  b: string
  aLevels: string[]
  bLevels: string[]
  /** Cell means: means[i][j] for a = aLevels[i], b = bLevels[j] (NaN for empty cells). */
  means: number[][]
  counts: number[][]
}

/** Interaction Plot data (Minitab): cell means of y for every a × b combination. */
export function interactionPlot(y: ArrayLike<number | null | undefined>, a: ArrayLike<unknown>, b: ArrayLike<unknown>, names: [string, string] = ['A', 'B']): InteractionPlot {
  if (y.length !== a.length || y.length !== b.length) throw new RangeError('interactionPlot: columns must have equal length')
  const aLevels = [...new Set(Array.from({ length: a.length }, (_, i) => a[i]).filter((v) => v != null).map(String))].sort()
  const bLevels = [...new Set(Array.from({ length: b.length }, (_, i) => b[i]).filter((v) => v != null).map(String))].sort()
  const sums = aLevels.map(() => bLevels.map(() => 0))
  const counts = aLevels.map(() => bLevels.map(() => 0))
  for (let i = 0; i < y.length; i++) {
    const v = y[i]
    if (a[i] == null || b[i] == null || typeof v !== 'number' || !Number.isFinite(v)) continue
    const r = aLevels.indexOf(String(a[i]))
    const c = bLevels.indexOf(String(b[i]))
    sums[r]![c]! += v
    counts[r]![c]!++
  }
  return { a: names[0], b: names[1], aLevels, bLevels, means: sums.map((row, r) => row.map((s, c) => (counts[r]![c]! ? s / counts[r]![c]! : NaN))), counts }
}

export interface EcdfResult {
  x: number[]
  /** F(x) = i / n at each sorted observation. */
  f: number[]
  n: number
}

/** Empirical CDF points. */
export function ecdf(x: ArrayLike<number | null | undefined>): EcdfResult {
  const v = cleanNumbers(x).slice().sort()
  const n = v.length
  return { x: Array.from(v), f: Array.from(v, (_, i) => (i + 1) / n), n }
}

export interface DotplotResult {
  binWidth: number
  bins: Array<{ center: number; count: number }>
}

/** Dotplot data: observations binned to `binWidth` (default range / 40) with stacked counts. */
export function dotplot(x: ArrayLike<number | null | undefined>, options: { binWidth?: number } = {}): DotplotResult {
  const v = cleanNumbers(x)
  if (!v.length) throw new RangeError('dotplot: no numeric values')
  let min = Infinity
  let max = -Infinity
  for (const t of v) {
    min = Math.min(min, t)
    max = Math.max(max, t)
  }
  const binWidth = options.binWidth ?? (max > min ? (max - min) / 40 : 1)
  const counts = new Map<number, number>()
  for (const t of v) {
    const k = Math.round((t - min) / binWidth)
    counts.set(k, (counts.get(k) ?? 0) + 1)
  }
  return { binWidth, bins: [...counts.entries()].sort(([a], [b]) => a - b).map(([k, count]) => ({ center: min + k * binWidth, count })) }
}

export interface FishboneNode {
  label: string
  children?: FishboneNode[]
}

export interface CauseAndEffect {
  effect: string
  categories: Array<{ label: string; side: 'top' | 'bottom'; x: number; causes: Array<{ label: string; x: number; y: number; subCauses: Array<{ label: string; x: number; y: number }> }> }>
  /** Simple SVG rendering of the diagram. */
  svg: string
}

/**
 * Cause-and-Effect (fishbone) diagram (Minitab): categories (default the 6 M's) with causes and sub-causes,
 * laid out on a normalized [0, 1] canvas and rendered to SVG.
 */
export function causeAndEffect(spec: { effect: string; categories: Record<string, Array<string | FishboneNode>> }, options: { width?: number; height?: number } = {}): CauseAndEffect {
  const names = Object.keys(spec.categories)
  if (!names.length) throw new RangeError('causeAndEffect needs at least one category')
  const width = options.width ?? 900
  const height = options.height ?? 480
  const top = names.filter((_, i) => i % 2 === 0)
  const bottom = names.filter((_, i) => i % 2 === 1)
  const spineY = 0.5
  const categories: CauseAndEffect['categories'] = []
  const place = (list: string[], side: 'top' | 'bottom') => {
    list.forEach((name, i) => {
      const x = 0.08 + ((i + 0.5) / list.length) * 0.72
      const nodes = spec.categories[name]!.map((c) => (typeof c === 'string' ? { label: c } : c))
      const causes = nodes.map((node, j) => {
        const frac = (j + 1) / (nodes.length + 1)
        const y = side === 'top' ? spineY - 0.38 * (1 - frac) - 0.02 : spineY + 0.38 * (1 - frac) + 0.02
        const cx = x + (side === 'top' ? 1 : 1) * 0.12 * (1 - frac)
        return { label: node.label, x: cx, y, subCauses: (node.children ?? []).map((s, k) => ({ label: s.label, x: cx + 0.05 * (k + 1), y: y + (side === 'top' ? -0.03 : 0.03) * (k + 1) })) }
      })
      categories.push({ label: name, side, x, causes })
    })
  }
  place(top, 'top')
  place(bottom, 'bottom')
  const X = (u: number) => (u * width).toFixed(1)
  const Y = (u: number) => (u * height).toFixed(1)
  const parts: string[] = [`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" font-family="sans-serif" font-size="12">`]
  parts.push(`<line x1="${X(0.05)}" y1="${Y(spineY)}" x2="${X(0.86)}" y2="${Y(spineY)}" stroke="currentColor" stroke-width="2"/>`)
  parts.push(`<rect x="${X(0.86)}" y="${Y(spineY - 0.06)}" width="${X(0.13)}" height="${Y(0.12)}" fill="none" stroke="currentColor"/>`)
  parts.push(`<text x="${X(0.925)}" y="${Y(spineY + 0.012)}" text-anchor="middle">${escapeXml(spec.effect)}</text>`)
  for (const c of categories) {
    const tipY = c.side === 'top' ? spineY - 0.42 : spineY + 0.42
    parts.push(`<line x1="${X(c.x - 0.12)}" y1="${Y(tipY)}" x2="${X(c.x)}" y2="${Y(spineY)}" stroke="currentColor"/>`)
    parts.push(`<text x="${X(c.x - 0.12)}" y="${Y(tipY + (c.side === 'top' ? -0.015 : 0.03))}" text-anchor="middle" font-weight="bold">${escapeXml(c.label)}</text>`)
    for (const cause of c.causes) {
      parts.push(`<text x="${X(cause.x - 0.12)}" y="${Y(cause.y)}" text-anchor="end">${escapeXml(cause.label)}</text>`)
      for (const s of cause.subCauses) parts.push(`<text x="${X(s.x - 0.12)}" y="${Y(s.y)}" text-anchor="end" font-size="10">${escapeXml(s.label)}</text>`)
    }
  }
  parts.push('</svg>')
  return { effect: spec.effect, categories, svg: parts.join('') }
}

function escapeXml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!)
}
