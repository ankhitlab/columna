/** Proposed Columna API: fixed-effects linear modelling and model-based comparisons.
 * All inference lives here, not in the application. Uses Columna QR, effects coding,
 * Student / F / studentized-range distributions and Games–Howell implementation.
 */
import { linearModel, linearModelVector, linearModelContrast, predictLinearModel, type Data, type LinearModelResult } from './lm.js'
import { t, f } from './dist.js'
import { groupingLetters, ptukey, qtukey, type GroupStats } from './tukey.js'
import { gamesHowell } from './multcomp.js'
import { descriptiveStats, type DescriptiveStats } from './descriptive.js'

export type ComparisonMethod = 'tukey' | 'bonferroni' | 'holm' | 'sidak' | 'fisher'
export interface LinearComparison {
  a: string; b: string; diff: number; se: number; statistic: number; df: number
  pRaw: number; pValue: number; ci: [number, number]; significant: boolean
}
export interface LinearMeansResult {
  factor: string; method: ComparisonMethod; confidence: number; at: Record<string, unknown>
  familySize: number; ciMethod: string
  means: Array<{ level: string; estimate: number; se: number; ci: [number, number]; n?: number; observed?: number; letters: string }>
  comparisons: LinearComparison[]
  warnings: string[]
}
const finite = Number.isFinite
function checkFit(fit: LinearModelResult) {
  const aliased = fit.coefficients.filter((c) => c.aliased).map((c) => c.name)
  if (aliased.length) throw new RangeError(`Модель неидентифицируема: линейно зависимые столбцы ${aliased.join(', ')}. Удалите дублирующие предикторы, лишние взаимодействия или заполните отсутствующие сочетания уровней.`)
  if (!(fit.anova.total.ss > 0) || !(fit.s > 0) || !finite(fit.s) || fit.anova.error.ss <= fit.anova.total.ss * 1e-26) throw new RangeError('Нулевая или практически нулевая остаточная дисперсия: стандартные ошибки и тесты модели ненадёжны.')
  if (fit.coefficients.some((c) => !finite(c.coef) || !finite(c.se))) throw new RangeError('Численная нестабильность модели: проверьте масштаб и коллинеарность предикторов.')
}

/** Adjust a complete family of p-values. Holm works under arbitrary dependence.
 * Sidak is labelled as an independence approximation for general model contrasts. */
export function adjustComparisonP(pvalues: number[], method: 'bonferroni' | 'holm' | 'sidak' | 'fisher'): number[] {
  if (!['bonferroni', 'holm', 'sidak', 'fisher'].includes(method)) throw new RangeError('Неизвестный метод корректировки p.')
  if (pvalues.some((p) => !(p >= 0 && p <= 1))) throw new RangeError('Некорректное исходное p-значение.')
  const m = pvalues.length
  if (method === 'holm') {
    const order = pvalues.map((p, i) => ({ p, i })).sort((a, b) => a.p - b.p)
    const out = new Array<number>(m); let previous = 0
    order.forEach(({ p, i }, rank) => { previous = Math.max(previous, Math.min(1, p * (m - rank))); out[i] = previous })
    return out
  }
  return pvalues.map((p) => method === 'fisher' ? p : method === 'sidak' ? -Math.expm1(m * Math.log1p(-p)) : Math.min(1, m * p))
}

/** All pairwise contrasts of equal-weight marginal means for ONE factor.
 * The joint covariance of the fitted means is retained in SE(diff), not replaced
 * with MSE(1/n_i+1/n_j). Numeric covariates default to complete-case means.
 * `at` fixes covariates or other factor levels and permits simple-effect comparisons.
 */
export function compareLinearMeans(fit: LinearModelResult, factor: string, options: { method?: ComparisonMethod; confidence?: number; at?: Record<string, unknown> } = {}): LinearMeansResult {
  checkFit(fit)
  const levels = fit.factors[factor]
  if (!levels) throw new RangeError('Для сравнений выберите категориальный фактор из модели.')
  const method = options.method ?? 'tukey', confidence = options.confidence ?? fit.confidence, at = { ...(options.at ?? {}) }
  if (!(confidence > 0 && confidence < 1) || !['tukey', 'bonferroni', 'holm', 'sidak', 'fisher'].includes(method)) throw new RangeError('Некорректный метод или доверительный уровень сравнений.')
  if (Object.hasOwn(at, factor)) throw new RangeError('Сравниваемый фактор нельзя одновременно зафиксировать в at.')
  const variables = new Set(fit.formula.terms.flatMap((term) => term.vars))
  for (const name of Object.keys(at)) if (!variables.has(name)) throw new RangeError(`Условие at не входит в модель: ${name}`)
  const vectors = levels.map((level) => linearModelVector(fit, { ...at, [factor]: level }, true))
  const means = levels.map((level, i) => {
    const pred = linearModelContrast(fit, vectors[i]!, confidence)
    const observed = fit.means[factor]!.find((m) => m.level === level)
    return { level, estimate: pred.fit, se: pred.se, ci: pred.ci, n: Object.keys(at).length ? undefined : observed?.n, observed: Object.keys(at).length ? undefined : observed?.mean, letters: '' }
  })
  const df = fit.anova.error.df, k = levels.length, m = k * (k - 1) / 2, alpha = 1 - confidence
  if (method === 'tukey' && df < 2) throw new RangeError('Tukey в Columna требует DF ошибки ≥ 2; используйте t-сравнения или увеличьте выборку.')
  const tail = method === 'bonferroni' || method === 'holm' ? alpha / m : method === 'sidak' ? -Math.expm1(Math.log1p(-alpha) / m) : alpha
  const critical = method === 'tukey' ? qtukey(confidence, k, df) / Math.SQRT2 : t(df).isf(tail / 2)
  if (!finite(critical)) throw new RangeError('Не удалось вычислить критическое значение сравнения.')
  const comparisons: LinearComparison[] = []
  for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) {
    const L = Float64Array.from(vectors[i]!, (x, c) => x - vectors[j]![c]!)
    const contrast = linearModelContrast(fit, L, confidence), diff = contrast.fit, se = contrast.se
    if (!(se > 0)) throw new RangeError('Стандартная ошибка контраста равна нулю; сравнение не определено.')
    const statistic = diff / se, pRaw = Math.min(1, 2 * t(df).sf(Math.abs(statistic)))
    const pValue = method === 'tukey' ? Math.min(1, Math.max(0, 1 - ptukey(Math.SQRT2 * Math.abs(statistic), k, df))) : pRaw
    const half = critical * se
    comparisons.push({ a: levels[i]!, b: levels[j]!, diff, se, statistic, df, pRaw, pValue, ci: [diff - half, diff + half], significant: false })
  }
  if (method !== 'tukey') adjustComparisonP(comparisons.map((p) => p.pRaw), method).forEach((p, i) => { comparisons[i]!.pValue = p })
  const differ = new Set<string>()
  comparisons.forEach((p) => { p.significant = p.pValue < alpha; if (p.significant) differ.add(`${p.a}\0${p.b}`).add(`${p.b}\0${p.a}`) })
  const letters = groupingLetters(means.map((r) => ({ name: r.level, mean: r.estimate, n: r.n ?? 0, m2: 0, sd: 0 })), differ)
  means.forEach((r) => { r.letters = letters.get(r.level) ?? '' })
  const warnings = [
    'Скорректированные средние: равные веса уровней остальных факторов; ковариаты фиксируются в среднем по полным строкам, если не задано at. Это не усреднение fitted внутри наблюдаемой группы.',
    'Поправка множественности относится только к выбранной семье парных сравнений. Повторные анализы и выбор модели по данным ею не защищены.',
    'Общая буква означает, что различие не выявлено выбранным тестом; она не доказывает эквивалентность.'
  ]
  if (method === 'holm') warnings.push('p скорректированы Holm; показаны консервативные одновременные интервалы Bonferroni, НЕ интервалы Holm. Решение по p и исключение нуля из интервала могут отличаться.')
  if (method === 'fisher') warnings.push('Fisher LSD не контролирует семейную ошибку: интервалы и p индивидуальные. Общий F-тест не является защитой всех попарных сравнений.')
  if (method === 'sidak') warnings.push('Šidák использует приближение независимости сравнений; при произвольной зависимости строгая гарантия семейной ошибки не заявляется. Bonferroni/Holm надёжнее в этом отношении.')
  if (method === 'tukey' && (Object.keys(fit.factors).length > 1 || Object.keys(fit.covariateMeans).length)) warnings.push('Tukey для скорректированных средних GLM: studentized-range приближение с модельным SE разности. В несбалансированных моделях с коррелированными оценками это не точный max-t тест.')
  if (fit.formula.terms.some((term) => term.vars.includes(factor) && new Set(term.vars).size > 1)) warnings.push('Фактор участвует во взаимодействии: маргинальное среднее может скрыть различия по условиям. Для простых эффектов зафиксируйте другой фактор в at.')
  return { factor, method, confidence, at, familySize: m, ciMethod: method === 'holm' ? 'bonferroni' : method, means, comparisons, warnings }
}

export interface LinearWorkbenchOptions {
  factors?: string[]; confidence?: number; ssType?: 'I' | 'III'
  comparison?: { factor: string; method?: ComparisonMethod; at?: Record<string, unknown> }
  predictionRows?: Array<Record<string, unknown>>
}
/** Common fixed-effects model fit for regression, ANOVA and ANCOVA. */
export function linearWorkbench(data: Data, formula: string, options: LinearWorkbenchOptions = {}) {
  const fit = linearModel(data, formula, options)
  checkFit(fit)
  const ssType = options.ssType ?? 'III'
  if (!['I', 'III'].includes(ssType)) throw new RangeError('Выберите суммы квадратов I или III.')
  if (!fit.formula.intercept && Object.keys(fit.factors).length) throw new RangeError('В этой реализации факторная GLM требует константу; для cell-means кодирования нужен отдельный патч.')
  const terms = fit.anova.terms.map((term) => {
    const ss = ssType === 'I' ? term.seqSS : term.adjSS, ms = ss / term.df, statistic = ms / fit.anova.error.ms
    return { ...term, ss, ms, f: statistic, pValue: f(term.df, fit.anova.error.df).sf(statistic) }
  })
  const warnings: string[] = []
  if (fit.coefficients.some((c) => (c.vif ?? 0) > 5)) warnings.push('Есть VIF > 5: оцените коллинеарность. VIF взаимодействий и эффект-кодированных столбцов зависит от параметризации и не является тестом значимости.')
  if (!finite(fit.r2pred)) warnings.push('R²(pred) не определён: хотя бы одно h близко к 1, исключение строки разрушает оценимость модели. Такая строка не исключается из PRESS молча.')
  if (fit.anova.error.df < 10) warnings.push('Мало остаточных степеней свободы: интервалы, диагностика и выводы модели требуют осторожности.')
  if (!fit.formula.intercept) warnings.push('Модель без константы: SS total и R² нецентрированные; не сравнивайте их напрямую с R² модели с константой.')
  // Diagnostics, with one-based ORIGINAL row indices. Never silently join residuals over gaps.
  const diagnostics = fit.keep.map((index, i) => {
    const h = fit.leverage[i]!, e = fit.residuals[i]!, df = fit.anova.error.df
    const sMinus2 = df > 1 && 1 - h > 1e-12 ? (fit.anova.error.ss - e * e / (1 - h)) / (df - 1) : NaN
    const deleted = sMinus2 > 0 ? e / Math.sqrt(sMinus2 * (1 - h)) : NaN
    let flags = Math.abs(fit.standardizedResiduals[i]!) > 2 ? 'R' : ''
    if (h > 3 * fit.coefficients.length / fit.n || h > .99) flags += 'X'
    if (fit.cooksD[i]! > 4 / fit.n) flags += 'D'
    return { row: index + 1, observed: fit.y[i]!, fitted: fit.fitted[i]!, residual: e, standardized: fit.standardizedResiduals[i]!, studentized: deleted, leverage: h, cooksD: fit.cooksD[i]!, flags }
  })
  const comparisons = options.comparison ? compareLinearMeans(fit, options.comparison.factor, { ...options.comparison, confidence: fit.confidence }) : undefined
  const predictionRows = options.predictionRows ?? []
  const predictions = predictLinearModel(fit, predictionRows).map((p, i) => ({ ...p, inputs: predictionRows[i]! }))
  for (const row of predictionRows) for (const variable of Object.keys(fit.covariateMeans)) {
    let min = Infinity, max = -Infinity
    for (const i of fit.keep) { const x = data[variable]![i] as number; min = Math.min(min, x); max = Math.max(max, x) }
    if ((row[variable] as number) < min || (row[variable] as number) > max) warnings.push(`Прогноз: ${variable} выходит за наблюдавшийся диапазон. Это экстраполяция; интервалы условны на принятую модель.`)
  }
  return { fit, ssType, terms, diagnostics, comparisons, predictions, warnings }
}

/** Welch one-way ANOVA, proposed API for the unequal-variance comparison workflow. */
export function welchOneWay(groups: Record<string, ArrayLike<number | null | undefined>>) {
  const stats = Object.entries(groups).map(([name, v]) => ({ ...(descriptiveStats(v) as DescriptiveStats), name }))
  if (stats.length < 2 || stats.some((s) => s.n < 2 || !(s.variance > 0))) throw new RangeError('Welch ANOVA: нужны минимум две группы, N ≥ 2 и ненулевая дисперсия в каждой.')
  const k = stats.length, weights = stats.map((s) => s.n / s.variance), sumWeights = weights.reduce((a, b) => a + b, 0)
  const mean = stats.reduce((a, s, i) => a + weights[i]! * s.mean, 0) / sumWeights
  const correction = stats.reduce((a, s, i) => a + (1 - weights[i]! / sumWeights) ** 2 / (s.n - 1), 0)
  const numerator = stats.reduce((a, s, i) => a + weights[i]! * (s.mean - mean) ** 2, 0) / (k - 1)
  const statistic = numerator / (1 + 2 * (k - 2) / (k * k - 1) * correction), df1 = k - 1, df2 = (k * k - 1) / (3 * correction)
  return { test: 'Welch ANOVA', statistic, df1, df2, pValue: f(df1, df2).sf(statistic) }
}

/** Games–Howell: use existing Columna implementation, but never downgrade failed range
 * probabilities to unadjusted t tests. Guards cover the existing range API's df domain. */
export function gamesHowellWorkbench(groups: Record<string, ArrayLike<number | null | undefined>>, confidence = .95) {
  if (!(confidence > 0 && confidence < 1)) throw new RangeError('Некорректный доверительный уровень.')
  const summaries = Object.values(groups).map((x) => descriptiveStats(x) as DescriptiveStats)
  if (summaries.some((s) => s.n < 3 || !(s.variance > 0))) throw new RangeError('Games–Howell: для этой версии Columna нужны N ≥ 3 и ненулевая дисперсия каждой группы (DF диапазона ≥ 2).')
  const result = gamesHowell(groups, { alpha: 1 - confidence })
  if (result.comparisons.some((p) => !(p.df >= 2) || !finite(p.pValue) || p.ci.some((x) => !finite(x)))) throw new RangeError('Games–Howell: численно не определено; результат не подменяется обычным t-тестом.')
  return { ...result, omnibus: welchOneWay(groups) }
}
