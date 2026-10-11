/** Proposed Columna extension: fitted-distribution capability, between/within and staged SPC.
 * MIT. Data-only APIs; no DOM, HTML or application-specific statistics.
 */
import { normal, gamma, weibull, lognormal, exponential, lgamma, type Distribution } from './dist.js'
import { descriptiveStats, quantile, histogramBins, type DescriptiveStats } from './descriptive.js'
import { withinEstimate, qualityMoments, qualityMovingRanges, processCapability, processControlChart, type QualityRaw, type QualityOptions, type QualityChart } from './process-quality.js'

const finite = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x)
const mean = (x: number[]) => x.reduce((s, v) => s + v, 0) / x.length
const stats = (x: ArrayLike<number | null>) => descriptiveStats(x) as DescriptiveStats
const STD = normal()
export type CapabilityDistribution = 'normal' | 'weibull' | 'lognormal' | 'gamma' | 'exponential'
export interface DistributionFit {
  distribution: CapabilityDistribution; parameters: Record<string, number>; logLik: number; aic: number; ad: number
  model: Distribution
}
function numericSample(raw: ArrayLike<number | null | undefined>, min = 8) {
  const x: number[] = []
  for (let i = 0; i < raw.length; i++) {
    const v = raw[i]; if (v == null) continue
    if (!finite(v)) throw new RangeError('Данные должны содержать конечные числа или пропуски.')
    x.push(v)
  }
  if (x.length < min || !(stats(x).sd > 0)) throw new RangeError(`Нужно не менее ${min} наблюдений с ненулевым разбросом.`)
  return x
}
function specifications(o: {lsl?: number; usl?: number}) {
  if (o.lsl === undefined && o.usl === undefined) throw new RangeError('Укажите LSL и/или USL.')
  if ([o.lsl, o.usl].some(v => v !== undefined && !finite(v))) throw new RangeError('Допуски должны быть конечными числами.')
  if (o.lsl !== undefined && o.usl !== undefined && !(o.usl > o.lsl)) throw new RangeError('USL должен быть больше LSL.')
}
/** AD statistic for a fitted CDF: NOT a distribution-free p-value. */
export function fittedAD(x: number[], model: Distribution): number {
  const sorted = x.slice().sort((a, b) => a - b), n = sorted.length
  let s = 0
  for (let i = 0; i < n; i++) s += (2 * i + 1) * (Math.log(Math.max(Number.MIN_VALUE, model.cdf(sorted[i]!))) + Math.log(Math.max(Number.MIN_VALUE, model.sf(sorted[n - 1 - i]!))))
  return Math.max(0, -n - s / n)
}
/** Uncensored maximum-likelihood fits. Positive distributions have a fixed zero threshold. */
export function fitCapabilityDistribution(raw: ArrayLike<number | null | undefined>, distribution: CapabilityDistribution): DistributionFit {
  const x = numericSample(raw), s = stats(x), n = x.length
  let model: Distribution, parameters: Record<string, number>, logLik: number, count = 2
  if (distribution === 'normal') {
    const sigma = s.sd * Math.sqrt((n - 1) / n)
    model = normal(s.mean, sigma); parameters = {mu: s.mean, sigma}; logLik = -n * Math.log(sigma * Math.sqrt(2 * Math.PI)) - n / 2
  } else {
    if (x.some(v => v <= 0)) throw new RangeError(`${distribution}: все измерения должны быть строго положительными (порог равен 0).`)
    const logs = x.map(Math.log), lm = mean(logs)
    if (distribution === 'gamma') {
      // Stable log(mean)-mean(log) even for very concentrated positive samples.
      const c = mean(x.map(v => {const r=(v-s.mean)/s.mean; return r-Math.log1p(r)}))
      if (!(c>0)) throw new RangeError('Gamma: положительная вариация не различима в double precision.')
      const logMinusDigamma=(k:number)=>{let z=k, rec=0;while(z<10){rec+=1/z;z++}const q=1/z,q2=q*q;return Math.log(k/z)+rec+q/2+q2/12-q2*q2/120+q2**3/252-q2**4/240+q2**5/132}
      let lo=1e-10,hi=Math.max(1,1/c);for(let i=0;i<150;i++){const mid=Math.sqrt(lo*hi);if(logMinusDigamma(mid)>c)lo=mid;else hi=mid;if((hi-lo)/hi<1e-12)break}
      const shape=Math.sqrt(lo*hi),scale=s.mean/shape
      if(!(shape<1e10))throw new RangeError('Gamma: форма выше численного бюджета 1e10; измените единицы или модель.')
      const core=shape>10?.5*Math.log(shape)-.5*Math.log(2*Math.PI)-1/(12*shape)+1/(360*shape**3)-1/(1260*shape**5):shape*Math.log(shape)-lgamma(shape)-shape
      logLik=n*(core-Math.log(s.mean)-(shape-1)*c);model=gamma(shape,scale);parameters={shape,scale}
    } else if (distribution === 'weibull') {
      // Zero-threshold uncensored MLE in log space: no x^shape overflow.
      const delta=logs.map(v=>v-lm),max=delta.reduce((a,b)=>Math.max(a,b),-Infinity)
      const score=(k:number)=>{let sw=0,sd=0;for(const d of delta){const w=Math.exp(k*(d-max));sw+=w;sd+=w*d}return 1/k-sd/sw}
      let lo=1e-8,hi=1;while(score(hi)>0&&hi<1e8)hi*=2
      if(score(hi)>0)throw new RangeError('Weibull: форма выше численного бюджета; проверьте разброс.')
      for(let i=0;i<150;i++){const mid=Math.sqrt(lo*hi);if(score(mid)>0)lo=mid;else hi=mid;if((hi-lo)/hi<1e-12)break}
      const shape=Math.sqrt(lo*hi),lmean=shape*max+Math.log(mean(delta.map(d=>Math.exp(shape*(d-max))))),logScale=lm+lmean/shape,scale=Math.exp(logScale)
      logLik=n*(Math.log(shape)-lm-lmean-1);model=weibull(shape,scale);parameters={shape,scale}
    } else if(distribution==='lognormal') {
      const sigma=Math.sqrt(mean(logs.map(v=>(v-lm)**2)));model=lognormal(lm,sigma);parameters={mu:lm,sigma};logLik=-n*(lm+Math.log(sigma*Math.sqrt(2*Math.PI))+.5)
    } else if(distribution==='exponential') {model=exponential(s.mean);parameters={scale:s.mean};count=1;logLik=-n*(Math.log(s.mean)+1)}
    else throw new RangeError('Неизвестное распределение.')
  }
  if (!finite(logLik!)) throw new RangeError('Нечисловое правдоподобие; проверьте масштаб данных.')
  return {distribution, parameters: parameters!, logLik: logLik!, aic: 2 * count - 2 * logLik!, ad: fittedAD(x, model!), model: model!}
}
export function nonnormalCapability(raw: ArrayLike<number | null | undefined>, options: {
  distribution?: CapabilityDistribution | 'auto' | 'empirical'; indices?: 'z-score' | 'iso'; lsl?: number; usl?: number; bins?: number
} = {}) {
  specifications(options)
  const x = numericSample(raw), n = x.length, s = stats(x), selected = options.distribution ?? 'auto', indices = selected === 'empirical' ? 'iso' : options.indices ?? 'z-score'
  if (!['iso','z-score'].includes(indices)) throw new RangeError('Индексы: z-score или iso.')
  const warnings: string[] = [], candidates: Array<{distribution: string; parameters?: Record<string, number>; aic?: number; logLik?: number; ad?: number; error?: string}> = []
  let fit: DistributionFit | undefined
  if (selected !== 'empirical') {
    for (const d of selected === 'auto' ? ['normal','weibull','lognormal','gamma','exponential'] : [selected]) {
      try {const f = fitCapabilityDistribution(x, d as CapabilityDistribution); const {model: _model, ...summary} = f; candidates.push(summary); if (!fit || f.aic < fit.aic) fit = f}
      catch (e) {if (selected !== 'auto') throw e; candidates.push({distribution: d, error: (e as Error).message})}
    }
    if (!fit) throw new RangeError('Ни одно из распределений не оценено.')
    candidates.sort((a, b) => (a.aic ?? Infinity) - (b.aic ?? Infinity))
    warnings.push('Подбор — минимум AIC среди перечисленных семейств на одной исходной шкале. Это не проверка пригодности распределения; AD приведён без p, поскольку параметры оценены из данных.')
  } else warnings.push('Непараметрический метод: эмпирические квантили Hyndman–Fan 6. Экстраполяция за выборочные экстремумы не выполняется. Ожидаемые модельные PPM не вычисляются.')
  const q = (p: number) => fit ? fit.model.ppf(p) : quantile(x, p), low = q(.00135), median = q(.5), high = q(.99865)
  if (![low, median, high].every(finite) || !(low < median && median < high)) throw new RangeError('Квантили вырождены или переполнены; индексы не определены.')
  const lsl = options.lsl, usl = options.usl
  const countBelow = lsl === undefined ? 0 : x.filter(v => v < lsl).length, countAbove = usl === undefined ? 0 : x.filter(v => v > usl).length
  const observed = {below: countBelow / n * 1e6, above: countAbove / n * 1e6, total: (countBelow + countAbove) / n * 1e6}
  const below = lsl === undefined ? 0 : fit?.model.cdf(lsl), above = usl === undefined ? 0 : fit?.model.sf(usl)
  let Pp: number | undefined, Ppl: number | undefined, Ppu: number | undefined
  if (indices === 'iso') {
    if (lsl !== undefined) Ppl = (median - lsl) / (median - low)
    if (usl !== undefined) Ppu = (usl - median) / (high - median)
    if (lsl !== undefined && usl !== undefined) Pp = (usl - lsl) / (high - low)
  } else {
    if (lsl !== undefined) Ppl = STD.isf(below!) / 3
    if (usl !== undefined) Ppu = STD.isf(above!) / 3
    if (Ppl !== undefined && Ppu !== undefined) Pp = (Ppl + Ppu) / 2
  }
  const Ppk = Ppl === undefined ? Ppu : Ppu === undefined ? Ppl : Math.min(Ppl, Ppu)
  const expected = fit ? {below: below! * 1e6, above: above! * 1e6, total: (below! + above!) * 1e6} : null
  const sorted = x.slice().sort((a, b) => a - b)
  const qq = fit ? sorted.map((v, i) => [q((i + .625) / (n + .25)), v]) : []
  const bins = histogramBins(x, options.bins ?? Math.max(5, Math.min(35, Math.ceil(Math.sqrt(n)))))
  const density = fit ? Array.from({length: 241}, (_, i) => {const v = s.min + (s.max - s.min) * i / 240; return [v, fit!.model.pdf(v)]}) : []
  warnings.push('Только Overall: Cp/Cpk и σwithin к этой модели не приписываются. Стабильность, независимость и физический смысл выбранного семейства проверяйте отдельно. Допуски не являются контрольными границами.')
  warnings.push('Доверительные интервалы ненормальных индексов не оценены. Бесконечный хвостовой индекс при нулевой модельной вероятности не является гарантией качества.')
  if (n < 100) warnings.push('N < 100: оценка крайних хвостов предварительная; PPM чувствительны к модели.')
  if (selected === 'empirical' && n < 741) warnings.push('N < 741: хвостовые эмпирические квантили 0,135% / 99,865% ограничены экстремумами выборки; устойчивость Pp/Ppk не подтверждена.')
  return {kind: 'nonnormal', n, omitted: raw.length - n, distribution: fit?.distribution ?? 'empirical', indices, parameters: fit?.parameters ?? {}, mean: s.mean, sd: s.sd, lsl, usl, quantiles: {low, median, high}, Pp, Ppl, Ppu, Ppk, zBench: expected ? STD.isf(expected.total / 1e6) : undefined, observed, expected, countBelow, countAbove, candidates, bins, density, qq, warnings}
}
/** Minitab-style short-term between estimate from adjacent subgroup means, not grand-mean ANOVA.
 * Equal-sized rational subgroups required so the sampling-variance subtraction is unambiguous.
 */
export function betweenWithinCapability(raw: QualityRaw, options: {lsl?: number; usl?: number; target?: number; withinMethod?: string; betweenMethod?: 'mr' | 'sample'; confidence?: number} = {}) {
  specifications(options)
  const groups = raw.filter(Array.isArray) as number[][]
  if (groups.length < 3 || raw.some(g => g !== null && (!Array.isArray(g) || g.length < 2 || g.some(v => !finite(v))))) throw new RangeError('B/W: нужно минимум три полные подгруппы размера ≥ 2.')
  const size = groups[0]!.length
  if (!groups.every(g => g.length === size)) throw new RangeError('B/W: в этой версии нужны одинаковые размеры подгрупп; переменный n не приводится молча к среднему.')
  const within = withinEstimate(raw, options.withinMethod ?? 'pooled'), means = raw.map(g => g === null ? null : mean(g as number[]))
  const method = options.betweenMethod ?? 'mr'
  if (!['mr','sample'].includes(method)) throw new RangeError('Оценка между подгруппами: mr или sample.')
  const sigmaMeans = method === 'mr' ? withinEstimate(means, 'mr').sigma : stats(means).sd
  const varianceBetweenRaw = sigmaMeans ** 2 - within.sigma ** 2 / size
  const sigmaBetween = Math.sqrt(Math.max(0, varianceBetweenRaw)), sigmaBW = Math.hypot(within.sigma, sigmaBetween)
  if (!(sigmaBW > 0)) throw new RangeError('B/W: нулевая дисперсия; проверьте измерительную систему.')
  const flat = groups.flat(), cap = processCapability(flat, {...options, sigmaWithin: sigmaBW})
  const control = processControlChart(raw, {type: 'xbar-s', rules: [1], sigma: within.sigma})
  const warnings = ['Cp/Cpk здесь основаны на σB/W = √(σwithin² + σbetween²), не на одном σwithin. Pp/Ppk — по общему выборочному s.', method === 'mr' ? 'σX̄ оценено по соседним MR средних / d₂(2); пропуски разрывают соседство. σbetween² = max(0, σX̄² − σwithin²/n).' : 'σX̄ оценено выборочным s средних; это альтернативный долгосрочный оцениватель, не стандартный MR-вариант.', 'Интервалы B/W-индексов не показаны: распределение объединённого оценивателя не заменено обычным χ². Контрольные карты X̄/S показаны с σwithin; сигналы могут отражать межпартийную вариацию.']
  if (varianceBetweenRaw < 0) warnings.push('Несмещённая разность оценок межгрупповой дисперсии отрицательна и усечена до нуля; это не доказывает отсутствия межгрупповой вариации.')
  return {kind: 'between-within', ...cap, intervals: {}, cpkCI: undefined, withinEstimator: within, sigmaMeans, sigmaBetween, sigmaBW, sigmaWithinOnly: within.sigma, varianceBetweenRaw, subgroupSize: size, groups: groups.length, means, control, betweenMethod: method, warnings}
}
export interface StageSpec {start: number; name?: string; baseline?: number; mu?: number; sigma?: number; center?: number}
/** Stage starts are zero-based input positions. No subgroup, MR, rule window or cumulative state crosses a stage. */
export function stagedControlChart(raw: QualityRaw, stages: StageSpec[], options: QualityOptions = {}) {
  if (!Array.isArray(stages) || !stages.length || stages[0]!.start !== 0) throw new RangeError('Стадии должны начинаться с первой точки.')
  if (stages.some((s, i) => !Number.isSafeInteger(s.start) || s.start < 0 || s.start >= raw.length || i > 0 && s.start <= stages[i - 1]!.start)) throw new RangeError('Начала стадий должны строго возрастать и лежать в данных.')
  if (options.baseline !== undefined || options.mu !== undefined || options.sigma !== undefined || options.center !== undefined) throw new RangeError('Для стадий задавайте базу/исторические параметры отдельно, не одновременно с общей базой.')
  const summaries: Array<Record<string, unknown>> = []
  let main: QualityChart | undefined
  for (let i = 0; i < stages.length; i++) {
    const stage = stages[i]!, end = stages[i + 1]?.start ?? raw.length, name = stage.name?.trim() || `Стадия ${i + 1}`
    let chart: QualityChart
    try {chart = processControlChart(raw.slice(stage.start, end), {...options, baseline: stage.baseline, mu: stage.mu, sigma: stage.sigma, center: stage.center, sizes: options.sizes?.slice(stage.start, end)})}
    catch (e) {throw new RangeError(`${name}: ${(e as Error).message}`)}
    const shift = (c: QualityChart) => ({...c, points: c.points.map(p => ({...p, index: p.index + stage.start, stage: i, stageName: name})), signals: c.signals.map(p => ({...p, index: p.index + stage.start})), outOfControl: c.outOfControl.map(j => j + stage.start)})
    summaries.push({name, start: stage.start, end: end - 1, n: chart.points.filter(p => p.value !== null).length, baseline: chart.baseline, mean: chart.processMean ?? chart.rate ?? chart.center, sigma: chart.processSigma ?? chart.sigma, sigmaZ: chart.sigmaZ, signals: chart.signals.length, estimator: chart.estimator?.label ?? '', historical: stage.mu !== undefined || stage.sigma !== undefined || stage.center !== undefined})
    const current = shift(chart), companion = chart.companion ? shift(chart.companion) : undefined
    if (!main) main = {...current, companion}
    else {main.points.push(...current.points); main.signals.push(...current.signals); main.outOfControl.push(...current.outOfControl); if (main.companion && companion) {main.companion.points.push(...companion.points); main.companion.signals.push(...companion.signals); main.companion.outOfControl.push(...companion.outOfControl)}}
  }
  return {...main!, baseline: undefined, stages: summaries, ...(main!.companion ? {companion: {...main!.companion, stages: summaries}} : {})}
}
