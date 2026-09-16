/**
 * Model selection (Minitab Regression › Stepwise / Best Subsets): forward selection, backward
 * elimination and stepwise with α-to-enter / α-to-remove, and exhaustive best-subsets ranked by R²
 * with R²(adj), R²(pred), Mallows' Cp and S.
 */
import { f as fDist } from './dist.js'
import { lstsq, matrix } from './linalg.js'
import { completeRows, designMatrix, ols, toColumns, type Column, type OlsResult, type Predictors } from './regression.js'

export interface StepwiseOptions {
  method?: 'stepwise' | 'forward' | 'backward'
  /** α to enter (default 0.15) and α to remove (default 0.15). */
  alphaIn?: number
  alphaOut?: number
  /** Predictors that must stay in the model. */
  include?: string[]
  names?: string[]
  maxSteps?: number
}

export interface StepwiseStep {
  step: number
  action: 'add' | 'remove'
  variable: string
  pValue: number
  variables: string[]
  s: number
  r2: number
  r2adj: number
  cp: number
}

export interface StepwiseResult {
  method: 'stepwise' | 'forward' | 'backward'
  steps: StepwiseStep[]
  selected: string[]
  model: OlsResult
  alphaIn: number
  alphaOut: number
}

interface Fit {
  sse: number
  p: number
  coefT: Float64Array
}

/** Fit y on the given subset (with constant) and return SSE and |t| per coefficient. */
function fitSubset(Xfull: Float64Array, n: number, pAll: number, subset: number[], y: Float64Array): Fit {
  const p = subset.length + 1
  const X = matrix(n, p)
  for (let i = 0; i < n; i++) {
    X.data[i * p] = 1
    for (let j = 0; j < subset.length; j++) X.data[i * p + j + 1] = Xfull[i * pAll + subset[j]!]!
  }
  const f = lstsq(X, y)
  const mse = f.sse / (n - f.rank)
  const coefT = new Float64Array(p)
  for (let j = 0; j < p; j++) {
    const v = f.xtxInv.data[j * p + j]! * mse
    coefT[j] = v > 0 ? f.coef[j]! / Math.sqrt(v) : 0
  }
  return { sse: f.sse, p: f.rank, coefT }
}

/**
 * Stepwise regression. Each step adds the predictor with the smallest p-value (< αIn) or removes
 * the one with the largest p-value (> αOut), reporting the model after every step.
 */
export function stepwise(y: Column, X: Predictors, options: StepwiseOptions = {}): StepwiseResult {
  const method = options.method ?? 'stepwise'
  const alphaIn = options.alphaIn ?? 0.15
  const alphaOut = options.alphaOut ?? 0.15
  if (method === 'stepwise' && alphaIn > alphaOut) throw new RangeError('stepwise: alphaIn must be ≤ alphaOut')
  const { names, cols } = toColumns(X, options.names)
  const { keep } = completeRows(y, cols)
  const n = keep.length
  const pAll = cols.length
  const D = designMatrix(cols, keep, false)
  const yv = Float64Array.from(keep, (i) => y[i] as number)
  const include = new Set((options.include ?? []).map((v) => {
    const j = names.indexOf(v)
    if (j < 0) throw new RangeError(`stepwise: unknown predictor "${v}"`)
    return j
  }))
  let ybar = 0
  for (let i = 0; i < n; i++) ybar += yv[i]!
  ybar /= n
  let sst = 0
  for (let i = 0; i < n; i++) sst += (yv[i]! - ybar) ** 2
  const fullMse = fitSubset(D.data, n, pAll, Array.from({ length: pAll }, (_, i) => i), yv)
  const mseFull = fullMse.sse / (n - fullMse.p)
  let current = method === 'backward' ? Array.from({ length: pAll }, (_, i) => i) : [...include]
  const steps: StepwiseStep[] = []
  const maxSteps = options.maxSteps ?? 4 * pAll + 4
  const summary = (subset: number[], fit: Fit) => {
    const dfe = n - fit.p
    const mse = fit.sse / dfe
    return { s: Math.sqrt(mse), r2: 1 - fit.sse / sst, r2adj: 1 - mse / (sst / (n - 1)), cp: fit.sse / mseFull - (n - 2 * fit.p) }
  }
  const pOfT = (t: number, dfe: number) => fDist(1, dfe).sf(t * t)
  for (let step = 1; step <= maxSteps; step++) {
    let changed = false
    // removal first (stepwise / backward)
    if (method !== 'forward' && current.length > 0) {
      const fit = fitSubset(D.data, n, pAll, current, yv)
      const dfe = n - fit.p
      let worst = -1
      let worstP = -1
      current.forEach((j, k) => {
        if (include.has(j)) return
        const pv = pOfT(fit.coefT[k + 1]!, dfe)
        if (pv > worstP) {
          worstP = pv
          worst = j
        }
      })
      if (worst >= 0 && worstP > alphaOut) {
        current = current.filter((j) => j !== worst)
        const f2 = fitSubset(D.data, n, pAll, current, yv)
        steps.push({ step, action: 'remove', variable: names[worst]!, pValue: worstP, variables: current.map((j) => names[j]!), ...summary(current, f2) })
        changed = true
        if (method === 'backward') continue
        step++
        if (step > maxSteps) break
      } else if (method === 'backward') break
    }
    if (method !== 'backward') {
      let best = -1
      let bestP = 2
      for (let j = 0; j < pAll; j++) {
        if (current.includes(j)) continue
        const trial = [...current, j]
        const fit = fitSubset(D.data, n, pAll, trial, yv)
        if (fit.p < trial.length + 1) continue // aliased
        const pv = pOfT(fit.coefT[trial.length]!, n - fit.p)
        if (pv < bestP) {
          bestP = pv
          best = j
        }
      }
      if (best >= 0 && bestP < alphaIn) {
        current = [...current, best]
        const f2 = fitSubset(D.data, n, pAll, current, yv)
        steps.push({ step, action: 'add', variable: names[best]!, pValue: bestP, variables: current.map((j) => names[j]!), ...summary(current, f2) })
        changed = true
      }
    }
    if (!changed) break
  }
  const selected = current.map((j) => names[j]!)
  const model = ols(y, current.map((j) => cols[j]!), { names: selected })
  return { method, steps, selected, model, alphaIn, alphaOut }
}

export interface Subset {
  size: number
  variables: string[]
  r2: number
  r2adj: number
  r2pred: number
  cp: number
  s: number
}

export interface BestSubsetsResult {
  subsets: Subset[]
  /** Full-model MSE used for Cp. */
  mseFull: number
  n: number
}

/**
 * Best Subsets (Minitab): every subset up to `maxK` predictors (default all), the `nBest` (default 2)
 * per size by R². Cp = SSE_p / MSE_full − (n − 2p).
 */
export function bestSubsets(y: Column, X: Predictors, options: { maxK?: number; nBest?: number; names?: string[]; include?: string[] } = {}): BestSubsetsResult {
  const { names, cols } = toColumns(X, options.names)
  const { keep } = completeRows(y, cols)
  const n = keep.length
  const pAll = cols.length
  if (pAll > 20) throw new RangeError('bestSubsets: at most 20 predictors (exhaustive search)')
  const D = designMatrix(cols, keep, false)
  const yv = Float64Array.from(keep, (i) => y[i] as number)
  const nBest = options.nBest ?? 2
  const maxK = options.maxK ?? pAll
  const include = (options.include ?? []).map((v) => {
    const j = names.indexOf(v)
    if (j < 0) throw new RangeError(`bestSubsets: unknown predictor "${v}"`)
    return j
  })
  let ybar = 0
  for (let i = 0; i < n; i++) ybar += yv[i]!
  ybar /= n
  let sst = 0
  for (let i = 0; i < n; i++) sst += (yv[i]! - ybar) ** 2
  const full = fitSubset(D.data, n, pAll, Array.from({ length: pAll }, (_, i) => i), yv)
  const mseFull = full.sse / (n - full.p)
  const bySize = new Map<number, Subset[]>()
  const total = 1 << pAll
  for (let mask = 1; mask < total; mask++) {
    const subset: number[] = []
    for (let j = 0; j < pAll; j++) if (mask & (1 << j)) subset.push(j)
    if (subset.length > maxK) continue
    if (include.some((j) => !subset.includes(j))) continue
    if (n <= subset.length + 1) continue
    const p = subset.length + 1
    const Xs = matrix(n, p)
    for (let i = 0; i < n; i++) {
      Xs.data[i * p] = 1
      for (let k = 0; k < subset.length; k++) Xs.data[i * p + k + 1] = D.data[i * pAll + subset[k]!]!
    }
    const f = lstsq(Xs, yv)
    if (f.rank < p) continue
    let press = 0
    for (let i = 0; i < n; i++) press += (f.residuals[i]! / (1 - f.leverage[i]!)) ** 2
    const dfe = n - p
    const mse = f.sse / dfe
    const row: Subset = { size: subset.length, variables: subset.map((j) => names[j]!), r2: 1 - f.sse / sst, r2adj: 1 - mse / (sst / (n - 1)), r2pred: 1 - press / sst, cp: f.sse / mseFull - (n - 2 * p), s: Math.sqrt(mse) }
    const list = bySize.get(subset.length) ?? []
    list.push(row)
    list.sort((a, b) => b.r2 - a.r2)
    if (list.length > nBest) list.length = nBest
    bySize.set(subset.length, list)
  }
  const subsets = [...bySize.keys()].sort((a, b) => a - b).flatMap((k) => bySize.get(k)!)
  return { subsets, mseFull, n }
}
