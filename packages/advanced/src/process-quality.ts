/**
 * Proposed Columna API extension: gap-aware process monitoring and capability.
 * Chart limits/statistics are delegated to spc.ts; capability indices to
 * capability.ts; exact intervals to basic.ts; descriptive estimates to
 * descriptive.ts. This module adds source alignment, a frozen baseline and
 * separately documented within-sigma estimators. No browser/UI dependencies.
 */
import { controlChart as coreChart, ewma, cusum, spcConstants, nelsonRules, type NelsonRule, type ControlChartResult, type ControlChartOptions } from './spc.js'
import { capability as coreCapability } from './capability.js'
import { descriptiveStats, histogramBins, type DescriptiveStats } from './descriptive.js'
import { proportionInterval, poissonRateInterval } from './basic.js'
import { normal, lgamma } from './dist.js'

export type QualityRaw = Array<number | number[] | null>
export type QualityType = 'i-mr' | 'xbar-r' | 'xbar-s' | 'p' | 'np' | 'c' | 'u' | 'laney-p' | 'laney-u' | 'ewma' | 'cusum'
export interface QualityOptions {
  type?: QualityType; rules?: NelsonRule[]; mu?: number; sigma?: number;
  center?: number; sizes?: Array<number | null>; baseline?: number;
  lambda?: number; L?: number; k?: number; h?: number
}
export interface WithinEstimate {
  sigma: number; df?: number; method: string; label: string; mrCount?: number
}
export interface QualityPoint {
  index: number; value: number | null; n?: number; center: number;
  lcl: number; ucl: number; zoneSigma: number; rules: NelsonRule[]; beyondLimits: boolean
}
export interface QualityChart {
  type: string; points: QualityPoint[]; center: number; sigma: number;
  processMean?: number; processSigma?: number; baseline?: number;
  estimator?: WithinEstimate | { label: string }; rules: NelsonRule[];
  signals: Array<{index: number; rules: NelsonRule[]}>; outOfControl: number[];
  companion?: QualityChart; rate?: number; sigmaZ?: number;
  lambda?: number; L?: number; k?: number; h?: number
}
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
export function qualityMoments(v: ArrayLike<number | null | undefined>) {
  const s = descriptiveStats(v) as DescriptiveStats
  return { ...s }
}
const average = (v: number[]) => qualityMoments(v).mean
export function qualityMovingRanges(raw: Array<number | null>): Array<number | null> {
  return raw.map((v, i) => i > 0 && finite(v) && finite(raw[i - 1]) ? Math.abs(v - (raw[i - 1] as number)) : null)
}
/** c4 for pooled residual DF; large-DF expansion avoids cancellation of log-gamma values. */
function pooledC4(df: number) {
  return df > 10000 ? 1 - 1 / (4 * df) + 1 / (32 * df * df) + 5 / (128 * df ** 3)
    : Math.sqrt(2 / df) * Math.exp(lgamma((df + 1) / 2) - lgamma(df / 2))
}
function rangeSize(n: number) {
  if (n > 25) throw new RangeError('Оценка R использует табличные d₂/d₃ для n ≤ 25. Для больших подгрупп выберите S или объединённое σ.')
}
/**
 * Within estimate: adjacent MR/d2, weighted R/d2 or S/c4, or pooled s/c4.
 * DF is a moment-matched approximation, not a claim of an exact chi-square law.
 * For overlapping MR, its variance includes the shared-observation covariance
 * under independent Gaussian observations (correlation of successive differences -1/2).
 */
export function withinEstimate(raw: QualityRaw, method = 'auto'): WithinEstimate {
  if (!Array.isArray(raw) || raw.some(v => v !== null && !finite(v) && (!Array.isArray(v) || v.some(x => !finite(x))))) throw new RangeError('Измерения должны быть конечными числами, полными подгруппами или null.')
  const grouped = raw.some(Array.isArray)
  if (grouped && raw.some(v => v !== null && !Array.isArray(v))) throw new RangeError('Нельзя смешивать отдельные измерения и подгруппы.')
  if (!grouped && !['auto', 'mr'].includes(method)) throw new RangeError('Для отдельных измерений используйте MR.')
  if (!grouped) {
    const mr = qualityMovingRanges(raw as Array<number | null>), good = mr.filter(finite)
    if (!good.length) throw new RangeError('Для σwithin нужны соседние заполненные строки; через пропуск MR не вычисляется.')
    const c = spcConstants(2), m = good.length
    let adjacent = 0
    for (let i = 1; i < mr.length; i++) if (finite(mr[i]) && finite(mr[i - 1])) adjacent++
    const covariance = 4 / Math.PI * (Math.sqrt(3) / 2 + Math.PI / 12 - 1)
    const relativeVariance = (m * c.d3 ** 2 + 2 * adjacent * covariance) / (m * m * c.d2 ** 2)
    return { sigma: average(good) / c.d2, df: 1 / (2 * relativeVariance), method: 'mr', label: 'MR̄ / d₂(2); DF учитывают перекрытие MR', mrCount: m }
  }
  const groups = raw.filter(Array.isArray) as number[][]
  if (groups.length < 2 || groups.some(g => g.length < 2 || g.some(v => !finite(v)))) throw new RangeError('Нужны минимум две полные подгруппы размера ≥ 2.')
  const stats = groups.map(qualityMoments), df = stats.reduce((s, g) => s + g.n - 1, 0)
  const chosen = method === 'auto' ? 'pooled' : method
  if (chosen === 'pooled') {
    const variance = stats.reduce((s, g) => s + (g.n - 1) * g.variance, 0) / df
    return { sigma: Math.sqrt(variance) / pooledC4(df), df, method: chosen, label: 'Объединённое s / c₄(DF + 1)' }
  }
  if (!['r', 's'].includes(chosen)) throw new RangeError('Выберите оценку R, S или pooled для подгрупп.')
  let sum = 0, weights = 0
  for (const g of stats) {
    if (chosen === 'r') rangeSize(g.n)
    const c = spcConstants(g.n), bias = chosen === 'r' ? c.d2 : c.c4
    const relvar = chosen === 'r' ? (c.d3 / c.d2) ** 2 : (1 - c.c4 * c.c4) / (c.c4 * c.c4)
    const weight = 1 / relvar
    sum += weight * (chosen === 'r' ? g.range : g.sd) / bias; weights += weight
  }
  return { sigma: sum / weights, df: weights / 2, method: chosen, label: chosen === 'r' ? 'R̄ / d₂; при разном n — взвешенная оценка' : 'S̄ / c₄; при разном n — взвешенная оценка' }
}
function enabledRules(rules: NelsonRule[] = [1]): NelsonRule[] {
  if (!Array.isArray(rules) || rules.some(r => !Number.isInteger(r) || r < 1 || r > 8)) throw new RangeError('Допустимы тесты 1–8.')
  return [...new Set(rules)].sort()
}
/** Reapply native Nelson rules after restoring gaps/source positions. */
function finish(chart: QualityChart, rules: NelsonRule[]): QualityChart {
  const z = chart.points.map(p => !finite(p.value) ? NaN : p.zoneSigma > 0 ? (p.value - p.center) / p.zoneSigma : 0)
  const fired = nelsonRules(z, 0, 1, rules.filter(r => r !== 1))
  chart.points.forEach((p, i) => {
    p.beyondLimits = finite(p.value) && (p.value > p.ucl || p.value < p.lcl)
    p.rules = finite(p.value) ? fired[i]! : []
    if (rules.includes(1) && p.beyondLimits) p.rules.unshift(1)
  })
  chart.rules = rules
  chart.signals = chart.points.filter(p => p.rules.length).map(p => ({index: p.index, rules: p.rules}))
  chart.outOfControl = chart.points.filter(p => p.beyondLimits).map(p => p.index)
  return chart
}
function aligned(result: ControlChartResult, keep: number[], length: number, rules: NelsonRule[]): QualityChart {
  const points: QualityPoint[] = Array.from({length}, (_, index) => ({index, value: null, center: result.center, lcl: result.lcl, ucl: result.ucl, zoneSigma: 0, rules: [], beyondLimits: false}))
  result.points.forEach((p, i) => {const index = keep[i]!; points[index] = {...p, index, value: finite(p.value) ? p.value : null, zoneSigma: p.zoneSigma ?? result.sigma}})
  return finish({type: result.type, center: result.center, sigma: result.sigma, points, rules, signals: [], outOfControl: []}, rules)
}
function segments(raw: Array<number | null>) {
  const result: Array<{values: number[]; positions: number[]}> = []; let group: {values: number[]; positions: number[]} | null = null
  raw.forEach((v, i) => {if (!finite(v)) {group = null; return} if (!group) {group = {values: [], positions: []}; result.push(group)} group.values.push(v); group.positions.push(i)})
  return result
}
/** All numeric charts use the actual native spc.ts functions, with a fixed baseline fit. */
export function processControlChart(raw: QualityRaw, options: QualityOptions = {}): QualityChart {
  if (!Array.isArray(raw)) throw new RangeError('Передайте массив измерений или подгрупп.')
  const type = options.type ?? 'i-mr', rules = enabledRules(options.rules), count = options.baseline ?? raw.length
  if (options.lambda !== undefined && (!finite(options.lambda) || !(options.lambda > 0 && options.lambda <= 1))) throw new RangeError('λ должна лежать в (0, 1].')
  if (options.L !== undefined && (!finite(options.L) || options.L <= 0)) throw new RangeError('L должно быть положительным конечным числом.')
  if (options.h !== undefined && (!finite(options.h) || options.h <= 0)) throw new RangeError('h должно быть положительным конечным числом.')
  if (options.k !== undefined && (!finite(options.k) || options.k < 0)) throw new RangeError('k должно быть неотрицательным конечным числом.')
  if (!Number.isSafeInteger(count) || count < 2 || count > raw.length) throw new RangeError('База: от 2 до числа исходных точек / подгрупп.')
  if (options.mu !== undefined && !finite(options.mu)) throw new RangeError('Историческое среднее должно быть конечным.')
  if (options.sigma !== undefined && (!finite(options.sigma) || options.sigma <= 0)) throw new RangeError('Историческое σ должно быть положительным.')
  if (['i-mr', 'xbar-r', 'xbar-s', 'ewma', 'cusum'].includes(type)) {
    const grouped = type === 'xbar-r' || type === 'xbar-s'
    if (raw.some(v => v !== null && (grouped ? !Array.isArray(v) : !finite(v)))) throw new RangeError('Структура измерений не соответствует типу карты.')
    const keep = raw.map((v, i) => v !== null ? i : -1).filter(i => i >= 0), baseKeep = keep.filter(i => i < count)
    if (baseKeep.length < 2) throw new RangeError('В базе нужно минимум две заполненные точки / подгруппы.')
    if (grouped && keep.some(i => (raw[i] as number[]).length < 2 || (raw[i] as number[]).some(v => !finite(v)))) throw new RangeError('Размер подгруппы должен быть ≥ 2.')
    if (type === 'xbar-r') keep.forEach(i => rangeSize((raw[i] as number[]).length))
    const baseline = raw.slice(0, count), flat = baseKeep.flatMap(i => Array.isArray(raw[i]) ? raw[i] as number[] : [raw[i] as number])
    const mean = options.mu ?? qualityMoments(flat).mean
    const estimate = options.sigma === undefined ? withinEstimate(baseline, type === 'xbar-r' ? 'r' : type === 'xbar-s' ? 's' : 'mr') : {sigma: options.sigma, method: 'historical', label: 'Заданное историческое σ'}
    const sigma = estimate.sigma
    let main: QualityChart
    if (type === 'ewma' || type === 'cusum') {
      if (!(sigma > 0)) throw new RangeError('Для EWMA / CUSUM требуется ненулевое σ.')
      const empty = (name: string): QualityChart => ({type: name, center: type === 'ewma' ? mean : 0, sigma, points: raw.map((_, index) => ({index, value: null, center: mean, lcl: NaN, ucl: NaN, zoneSigma: 0, rules: [], beyondLimits: false})), rules: [1], signals: [], outOfControl: []})
      main = empty(type === 'cusum' ? 'cusum-plus' : type)
      if (type === 'cusum') main.companion = empty('cusum-minus')
      for (const segment of segments(raw as Array<number | null>)) {
        if (type === 'ewma') {
          const r = ewma(segment.values, {mu: mean, sigma, lambda: options.lambda, L: options.L})
          r.z.forEach((v, i) => {const index = segment.positions[i]!; main.points[index] = {index, value: v, center: mean, ucl: r.ucl[i]!, lcl: r.lcl[i]!, zoneSigma: (r.ucl[i]! - mean) / r.L, rules: [], beyondLimits: false}})
          main.lambda = r.lambda; main.L = r.L
        } else {
          const r = cusum(segment.values, {mu: mean, sigma, k: options.k, h: options.h})
          r.cPlus.forEach((v, i) => {const index = segment.positions[i]!; main.points[index] = {index, value: v, center: 0, ucl: r.h, lcl: 0, zoneSigma: 0, rules: [], beyondLimits: false}; main.companion!.points[index] = {index, value: -r.cMinus[i]!, center: 0, ucl: 0, lcl: -r.h, zoneSigma: 0, rules: [], beyondLimits: false}})
          main.k = r.k; main.h = r.h
        }
      }
      finish(main, [1]); if (main.companion) finish(main.companion, [1])
    } else {
      const input = keep.map(i => raw[i]) as number[] | number[][]
      const native = coreChart(input, {type, mu: mean, sigma, rules: []})
      main = aligned(native, keep, raw.length, rules)
      if (native.companion) {
        if (grouped) main.companion = aligned(native.companion, keep, raw.length, [1])
        else {
          main.companion = aligned(native.companion, keep.slice(1), raw.length, [1])
          const mr = qualityMovingRanges(raw as Array<number | null>)
          main.companion.points.forEach((p, i) => {p.value = mr[i] ?? null})
          finish(main.companion, [1])
        }
      }
    }
    main.processMean = mean; main.processSigma = sigma; main.estimator = estimate; main.baseline = count
    return main
  }
  if (!['p', 'np', 'c', 'u', 'laney-p', 'laney-u'].includes(type)) throw new RangeError('Неизвестный тип карты.')
  if (raw.some(v => v !== null && !finite(v))) throw new RangeError('Счётчики должны быть конечными числами или null.')
  const x = raw as Array<number | null>, sizes = options.sizes
  if (type !== 'c' && (!sizes || sizes.length !== x.length)) throw new RangeError('Размеры / экспозиции должны совпадать с исходными строками.')
  const keep = x.map((v, i) => finite(v) ? i : -1).filter(i => i >= 0), bi = keep.filter(i => i < count)
  if (bi.length < 2) throw new RangeError('В базе нужно минимум две полные строки счётных данных.')
  const bin = ['p', 'np', 'laney-p'].includes(type)
  if (options.center !== undefined && (!finite(options.center) || options.center < 0 || bin && options.center > 1)) throw new RangeError('Некорректная историческая доля / интенсивность.')
  const baseType = type === 'laney-p' ? 'p' : type === 'laney-u' ? 'u' : type
  const center = options.center === undefined ? undefined : type === 'np' ? options.center * (sizes![bi[0]!] as number) : options.center
  const fit = coreChart(bi.map(i => x[i] as number), {type: baseType as ControlChartOptions['type'], sizes: sizes ? bi.map(i => sizes[i] as number) : undefined, center, rules: []})
  const rate = type === 'np' ? fit.center / (sizes![bi[0]!] as number) : fit.center
  let sigmaZ: number | undefined
  if (type.startsWith('laney')) {
    if (!(rate > 0) || bin && rate >= 1) throw new RangeError('Laney требует ненулевой интенсивности; для P′ доля должна быть строго между 0 и 1.')
    const z: Array<number | null> = Array.from({length: count}, () => null)
    fit.points.forEach((p, i) => {z[bi[i]!] = (p.value - p.center) / p.zoneSigma!})
    sigmaZ = withinEstimate(z).sigma
    if (!(sigmaZ > 0)) throw new RangeError('Laney: σz = 0; проверьте данные или используйте обычную P/U-карту.')
  }
  const nativeOptions: ControlChartOptions = {type: type as ControlChartOptions['type'], sizes: sizes ? keep.map(i => sizes[i] as number) : undefined, center: fit.center, sigmaZ, rules: []}
  const native = coreChart(keep.map(i => x[i] as number), nativeOptions)
  const main = aligned(native, keep, x.length, [1])
  main.rate = rate; main.sigmaZ = sigmaZ; main.baseline = count
  main.estimator = {label: options.center === undefined ? 'Оценка Columna по базовому периоду' : 'Заданный исторический уровень'}
  return main
}
export const binomialCI = proportionInterval
export const poissonCI = poissonRateInterval
export const processCapability = coreCapability
/** Aggregate attribute performance with a genuine library confidence interval. */
export function attributeCapability(counts: Array<number | null>, sizes: Array<number | null>, kind: 'binomial' | 'poisson', confidence = 0.95) {
  if (!['binomial', 'poisson'].includes(kind)) throw new RangeError('Выберите биномиальную или пуассоновскую модель.')
  const indices = counts.map((v, i) => finite(v) ? i : -1).filter(i => i >= 0)
  // Native control-chart validation also checks that the row-level binomial counts are feasible.
  const chart = processControlChart(counts, {type: kind === 'binomial' ? 'p' : 'u', sizes, rules: [1]})
  const events = indices.reduce((s, i) => s + counts[i]!, 0), exposure = indices.reduce((s, i) => s + sizes[i]!, 0)
  const ci = kind === 'binomial' ? proportionInterval(events, exposure, confidence) : poissonRateInterval(events, exposure, confidence)
  const rate = events / exposure
  return {eventsCount: events, exposure, rate, ci, confidence, ppm: kind === 'binomial' ? rate * 1e6 : undefined, yield: kind === 'binomial' ? 1 - rate : undefined, zBench: kind === 'binomial' ? normal().isf(rate) : undefined, plotted: indices.length, chart}
}
/** Plot-ready normal Q-Q coordinates, computed before any display-only downsampling. */
export function normalQQ(x: ArrayLike<number | null | undefined>): Array<[number, number]> {
  const sorted = Array.from(x).filter(finite).sort((a, b) => a - b), d = normal()
  return sorted.map((value, i) => [d.ppf((i + 1 - 0.375) / (sorted.length + 0.25)), value])
}
export function qualityHistogram(x: ArrayLike<number | null | undefined>) {
  return histogramBins(x, Math.max(5, Math.min(30, Math.ceil(Math.sqrt(Array.from(x).filter(finite).length)))))
}
