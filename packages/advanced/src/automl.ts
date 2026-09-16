/**
 * Model validation and automated model selection (Minitab Predictive Analytics › Automated Machine
 * Learning / model validation): k-fold cross-validation of the tree-based learners and of OLS, and
 * `autoModel`, which fits CART, Random Forests, TreeNet, MARS (regression) or CART / RF / TreeNet /
 * logistic (classification) and ranks them by cross-validated performance.
 */
import { cart, mars, randomForest, treeNet, type CartOptions } from './predictive.js'
import { logit } from './glm.js'
import { ols } from './regression.js'

export type ModelKind = 'cart' | 'random-forest' | 'treenet' | 'mars' | 'ols' | 'logistic'

export interface Metrics {
  /** Regression: R², RMSE, MAE. */
  r2?: number
  rmse?: number
  mae?: number
  /** Classification: accuracy, log-loss (when probabilities are available), per-class counts. */
  accuracy?: number
  logLoss?: number
  confusion?: Record<string, Record<string, number>>
}

export interface CvResult {
  model: ModelKind
  task: 'regression' | 'classification'
  folds: number
  /** Metrics on the pooled out-of-fold predictions. */
  metrics: Metrics
  /** Per-fold metrics. */
  perFold: Metrics[]
  /** Out-of-fold predictions in the original row order. */
  predictions: Array<number | string>
}

interface Fitted {
  predict(X: number[][]): Array<number | string>
  predictProba?(X: number[][]): Array<Record<string, number>>
}

function fitOne(kind: ModelKind, task: 'regression' | 'classification', X: number[][], y: Array<number | string>, opts: CartOptions & { nTrees?: number; learningRate?: number; maxTerms?: number }): Fitted {
  switch (kind) {
    case 'cart':
      return cart(X, y, { ...opts, task })
    case 'random-forest':
      return randomForest(X, y, { ...opts, task })
    case 'treenet':
      return treeNet(X, y, { task, nTrees: opts.nTrees, learningRate: opts.learningRate, maxDepth: opts.maxDepth, seed: opts.seed })
    case 'mars': {
      if (task !== 'regression') throw new RangeError('mars is regression-only')
      const m = mars(X, y as number[], { maxTerms: opts.maxTerms })
      return { predict: (Xn) => m.predict(Xn) }
    }
    case 'ols': {
      if (task !== 'regression') throw new RangeError('ols is regression-only')
      const p = X[0]!.length
      const cols = Array.from({ length: p }, (_, j) => X.map((r) => r[j]!))
      const o = ols(y as number[], cols)
      return { predict: (Xn) => Xn.map((r) => o.predict(r)[0]!.fit) }
    }
    case 'logistic': {
      if (task !== 'classification') throw new RangeError('logistic is classification-only')
      const classes = [...new Set(y.map(String))].sort()
      if (classes.length !== 2) throw new RangeError('logistic needs a binary response')
      const p = X[0]!.length
      const cols = Array.from({ length: p }, (_, j) => X.map((r) => r[j]!))
      const g = logit(y.map((v) => (String(v) === classes[1] ? 1 : 0)), cols)
      return {
        predict: (Xn) => Xn.map((r) => (g.predict(r)[0]!.fit >= 0.5 ? classes[1]! : classes[0]!)),
        predictProba: (Xn) => Xn.map((r) => {
          const p1 = g.predict(r)[0]!.fit
          return { [classes[0]!]: 1 - p1, [classes[1]!]: p1 }
        }),
      }
    }
  }
}

function metricsOf(task: 'regression' | 'classification', y: Array<number | string>, pred: Array<number | string>, proba?: Array<Record<string, number>>): Metrics {
  const n = y.length
  if (task === 'regression') {
    const yy = y as number[]
    const pp = pred as number[]
    const mean = yy.reduce((a, b) => a + b, 0) / n
    let sse = 0
    let sst = 0
    let mae = 0
    for (let i = 0; i < n; i++) {
      sse += (yy[i]! - pp[i]!) ** 2
      sst += (yy[i]! - mean) ** 2
      mae += Math.abs(yy[i]! - pp[i]!)
    }
    return { r2: sst > 0 ? 1 - sse / sst : NaN, rmse: Math.sqrt(sse / n), mae: mae / n }
  }
  let correct = 0
  let logLoss = 0
  const confusion: Record<string, Record<string, number>> = {}
  for (let i = 0; i < n; i++) {
    const a = String(y[i])
    const b = String(pred[i])
    if (a === b) correct++
    confusion[a] ??= {}
    confusion[a]![b] = (confusion[a]![b] ?? 0) + 1
    if (proba) logLoss -= Math.log(Math.max(1e-15, proba[i]![a] ?? 1e-15))
  }
  return { accuracy: correct / n, logLoss: proba ? logLoss / n : undefined, confusion }
}

function foldsOf(n: number, k: number, seed: number): number[] {
  let a = seed >>> 0
  const u = () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296
  }
  const idx = Array.from({ length: n }, (_, i) => i)
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(u() * (i + 1))
    ;[idx[i], idx[j]] = [idx[j]!, idx[i]!]
  }
  const fold = new Array<number>(n)
  idx.forEach((row, pos) => (fold[row] = pos % k))
  return fold
}

/**
 * k-fold cross-validation (default 5 folds, seeded shuffle) of one learner.
 *   crossValidate(X, y, { model: 'random-forest', folds: 5 })
 */
export function crossValidate(
  X: ArrayLike<ArrayLike<number>>,
  y: ArrayLike<number | string>,
  options: { model: ModelKind; task?: 'regression' | 'classification'; folds?: number; seed?: number } & CartOptions & { nTrees?: number; learningRate?: number; maxTerms?: number },
): CvResult {
  const rows = Array.from(X).map((r) => Array.from(r))
  const yy = Array.from(y)
  const n = rows.length
  const task = options.task ?? (typeof yy[0] === 'number' ? 'regression' : 'classification')
  const k = options.folds ?? 5
  if (n < 2 * k) throw new RangeError(`crossValidate: need at least ${2 * k} rows for ${k} folds`)
  const fold = foldsOf(n, k, options.seed ?? 7)
  const predictions = new Array<number | string>(n)
  const probas: Array<Record<string, number> | undefined> = new Array(n)
  const perFold: Metrics[] = []
  for (let f = 0; f < k; f++) {
    const train = rows.map((_, i) => i).filter((i) => fold[i] !== f)
    const test = rows.map((_, i) => i).filter((i) => fold[i] === f)
    const fit = fitOne(options.model, task, train.map((i) => rows[i]!), train.map((i) => yy[i]!), options)
    const Xt = test.map((i) => rows[i]!)
    const pr = fit.predict(Xt)
    const pp = fit.predictProba?.(Xt)
    test.forEach((i, j) => {
      predictions[i] = pr[j]!
      probas[i] = pp?.[j]
    })
    perFold.push(metricsOf(task, test.map((i) => yy[i]!), pr, pp))
  }
  const hasProba = probas.every((p) => p !== undefined)
  return { model: options.model, task, folds: k, metrics: metricsOf(task, yy, predictions, hasProba ? (probas as Array<Record<string, number>>) : undefined), perFold, predictions }
}

export interface AutoModelResult {
  task: 'regression' | 'classification'
  /** Candidates ranked by CV R² (regression) or accuracy (classification). */
  ranking: Array<{ model: ModelKind; metrics: Metrics }>
  best: ModelKind
  /** Best model refitted on all data. */
  fit: Fitted
  cv: CvResult[]
}

/**
 * Automated model selection (Minitab AutoML): cross-validates every applicable learner and refits the
 * best one on the full data. Regression candidates: ols, cart, random-forest, treenet, mars;
 * classification: logistic (binary), cart, random-forest, treenet.
 */
export function autoModel(
  X: ArrayLike<ArrayLike<number>>,
  y: ArrayLike<number | string>,
  options: { task?: 'regression' | 'classification'; folds?: number; seed?: number; models?: ModelKind[]; nTrees?: number } = {},
): AutoModelResult {
  const rows = Array.from(X).map((r) => Array.from(r))
  const yy = Array.from(y)
  const task = options.task ?? (typeof yy[0] === 'number' ? 'regression' : 'classification')
  const binary = task === 'classification' && new Set(yy.map(String)).size === 2
  const candidates: ModelKind[] = options.models ?? (task === 'regression' ? ['ols', 'cart', 'random-forest', 'treenet', 'mars'] : [...(binary ? ['logistic' as const] : []), 'cart', 'random-forest', 'treenet'])
  const cv = candidates.map((model) => crossValidate(rows, yy, { model, task, folds: options.folds, seed: options.seed, nTrees: options.nTrees }))
  const score = (m: Metrics) => (task === 'regression' ? m.r2 ?? -Infinity : m.accuracy ?? -Infinity)
  const ranking = cv.map((c) => ({ model: c.model, metrics: c.metrics })).sort((a, b) => score(b.metrics) - score(a.metrics))
  const best = ranking[0]!.model
  return { task, ranking, best, fit: fitOne(best, task, rows, yy, { nTrees: options.nTrees, seed: options.seed }), cv }
}
