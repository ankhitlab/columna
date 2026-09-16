/**
 * Stability Study (Minitab Stat › Regression › Stability Study): response versus time with a batch factor.
 * Model selection as in Minitab / ICH Q1E: the batch × time interaction is dropped when p > `alphaPool`
 * (default 0.25), then the batch term; the shelf life is the earliest time at which the confidence bound
 * of the fitted mean crosses a specification limit (per batch when batch terms remain, overall = minimum).
 */
import { linearModel, type LinearModelResult } from './lm.js'
import { ols, type OlsResult } from './regression.js'

export interface StabilityOptions {
  /** Lower / upper specification limits (at least one). */
  lsl?: number
  usl?: number
  /** α for pooling decisions (default 0.25). */
  alphaPool?: number
  /** Confidence level for the shelf-life bound (default 0.95). One-sided with one limit, two-sided with both. */
  confidence?: number
  /** Search horizon for the shelf life (default 4 × the latest time). */
  maxTime?: number
}

export interface StabilityResult {
  test: 'stability study'
  /** Model selection steps: term tested, p-value, decision. */
  selection: Array<{ model: string; term: string; pValue: number; removed: boolean }>
  /** Final model: 'batch*time' | 'batch + time' | 'time'. */
  model: string
  fit: LinearModelResult
  batches: string[]
  /** Shelf life per batch (NaN when the bound never crosses within maxTime). */
  shelfLife: Array<{ batch: string; shelfLife: number; limit: 'lsl' | 'usl' | 'none'; slope: number; intercept: number }>
  /** Overall shelf life (minimum over batches). */
  overall: number
  confidence: number
  omitted: number[]
  /** Fitted mean and confidence bounds for a batch at a time. */
  predict(time: number, batch?: string): { fit: number; lower: number; upper: number }
}

export function stabilityStudy(
  response: ArrayLike<number | null | undefined>,
  time: ArrayLike<number | null | undefined>,
  batch: ArrayLike<unknown>,
  options: StabilityOptions,
): StabilityResult {
  if (options.lsl === undefined && options.usl === undefined) throw new RangeError('stabilityStudy needs lsl and/or usl')
  const alphaPool = options.alphaPool ?? 0.25
  const confidence = options.confidence ?? 0.95
  const data = { y: response, time, batch: Array.from({ length: batch.length }, (_, i) => (batch[i] == null ? null : String(batch[i]))) }
  const selection: StabilityResult['selection'] = []
  let formula = 'y ~ batch*time'
  let fit = linearModel(data, formula, { factors: ['batch'] })
  const termP = (f: LinearModelResult, name: string) => f.anova.terms.find((t) => t.term === name || t.term === name.split('*').reverse().join('*'))?.pValue ?? NaN
  const pInt = termP(fit, 'batch*time')
  const dropInt = !(pInt <= alphaPool)
  selection.push({ model: 'batch*time', term: 'batch*time', pValue: pInt, removed: dropInt })
  if (dropInt) {
    formula = 'y ~ batch + time'
    fit = linearModel(data, formula, { factors: ['batch'] })
    const pB = termP(fit, 'batch')
    const dropB = !(pB <= alphaPool)
    selection.push({ model: 'batch + time', term: 'batch', pValue: pB, removed: dropB })
    if (dropB) {
      formula = 'y ~ time'
      fit = linearModel(data, formula)
    }
  }
  const model = formula.slice(4)
  const batches = fit.factors.batch ?? [...new Set(data.batch.filter((b): b is string => b !== null))].sort()
  // refit the same design with ols to get predictions with covariance
  const X = fit.design
  const p = X.cols
  const cols = Array.from({ length: p - 1 }, (_, j) => Array.from({ length: X.rows }, (_, i) => X.data[i * p + j + 1]!))
  const o: OlsResult = ols(Array.from(fit.fitted, (_, i) => fit.residuals[i]! + fit.fitted[i]!), cols, { names: fit.columnNames.slice(1) })
  const m = batches.length
  const code = (b: string, level: string) => (b === level ? 1 : b === batches[m - 1] ? -1 : 0)
  const rowFor = (t: number, b: string | undefined): number[] =>
    fit.columnNames.slice(1).map((name) => {
      if (name === 'time') return t
      const mm = /^batch\[(.+)\]$/.exec(name)
      if (mm) return b === undefined ? 0 : code(b, mm[1]!)
      const mi = /^batch\[(.+)\]\*time$/.exec(name) ?? /^time\*batch\[(.+)\]$/.exec(name)
      if (mi) return b === undefined ? 0 : code(b, mi[1]!) * t
      throw new RangeError(`stabilityStudy: unexpected column ${name}`)
    })
  const twoSided = options.lsl !== undefined && options.usl !== undefined
  const conf = twoSided ? confidence : 2 * confidence - 1 // one-sided bound at `confidence`
  const predict = (t: number, b?: string) => {
    if (b !== undefined && !batches.includes(b)) throw new RangeError(`stabilityStudy: unknown batch ${b}`)
    const pr = o.predict(rowFor(t, model === 'time' ? undefined : b), { confidence: conf })[0]!
    return { fit: pr.fit, lower: pr.ci[0], upper: pr.ci[1] }
  }
  let tMax = 0
  for (let i = 0; i < time.length; i++) if (typeof time[i] === 'number') tMax = Math.max(tMax, time[i] as number)
  const horizon = options.maxTime ?? 4 * (tMax || 1)
  const shelfFor = (b: string | undefined) => {
    const crosses = (t: number): 'lsl' | 'usl' | null => {
      const pr = predict(t, b)
      if (options.lsl !== undefined && pr.lower < options.lsl) return 'lsl'
      if (options.usl !== undefined && pr.upper > options.usl) return 'usl'
      return null
    }
    const at0 = crosses(0)
    if (at0) return { shelfLife: 0, limit: at0 }
    if (!crosses(horizon)) return { shelfLife: NaN, limit: 'none' as const }
    let lo = 0
    let hi = horizon
    for (let i = 0; i < 100; i++) {
      const mid = 0.5 * (lo + hi)
      if (crosses(mid)) hi = mid
      else lo = mid
    }
    return { shelfLife: hi, limit: crosses(hi)! }
  }
  const slopeOf = (b: string | undefined) => {
    const p0 = predict(0, b).fit
    const p1 = predict(1, b).fit
    return { slope: p1 - p0, intercept: p0 }
  }
  const shelfLife = (model === 'time' ? [undefined] : batches).map((b) => {
    const s = shelfFor(b)
    return { batch: b ?? 'all', ...s, ...slopeOf(b) }
  })
  const finite = shelfLife.map((s) => s.shelfLife).filter((v) => Number.isFinite(v))
  return {
    test: 'stability study',
    selection,
    model,
    fit,
    batches,
    shelfLife,
    overall: finite.length ? finite.reduce((m, v) => (v < m ? v : m), Infinity) : NaN,
    confidence,
    omitted: fit.omitted,
    predict,
  }
}
