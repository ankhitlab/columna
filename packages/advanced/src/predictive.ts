/**
 * Tier 5.5 — Predictive analytics:
 * CART (classification & regression trees), Random Forests (bagging),
 * TreeNet-style gradient boosting, and a basic MARS (hinge) model.
 */
import { lstsq, matrix } from './linalg.js'

export type TreeTask = 'regression' | 'classification'

interface Node {
  prediction: number | string
  /** Class probabilities for classification. */
  proba?: Record<string, number>
  feature?: number
  threshold?: number
  left?: Node
  right?: Node
  n: number
  impurity: number
}

export interface CartOptions {
  task?: TreeTask
  maxDepth?: number
  minSamplesLeaf?: number
  minSamplesSplit?: number
  /** Max features to consider at each split (default all; √p for RF). */
  maxFeatures?: number | 'sqrt' | 'log2'
  seed?: number
}

export interface CartResult {
  task: TreeTask
  tree: Node
  featureImportances: number[]
  predict(X: ArrayLike<ArrayLike<number>>): Array<number | string>
  predictProba?(X: ArrayLike<ArrayLike<number>>): Array<Record<string, number>>
}

function rng(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296
  }
}

function gini(y: ArrayLike<string | number>): number {
  const n = y.length
  const counts = new Map<string, number>()
  for (let i = 0; i < n; i++) {
    const k = String(y[i])
    counts.set(k, (counts.get(k) ?? 0) + 1)
  }
  let g = 1
  for (const c of counts.values()) g -= (c / n) ** 2
  return g
}

function mse(y: ArrayLike<number>): number {
  const n = y.length
  let m = 0
  for (let i = 0; i < n; i++) m += y[i]!
  m /= n
  let s = 0
  for (let i = 0; i < n; i++) s += (y[i]! - m) ** 2
  return s / n
}

function majority(y: ArrayLike<string | number>): { pred: string; proba: Record<string, number> } {
  const counts = new Map<string, number>()
  for (let i = 0; i < y.length; i++) {
    const k = String(y[i])
    counts.set(k, (counts.get(k) ?? 0) + 1)
  }
  let best = ''
  let bestC = -1
  const proba: Record<string, number> = {}
  for (const [k, c] of counts) {
    proba[k] = c / y.length
    if (c > bestC) {
      bestC = c
      best = k
    }
  }
  return { pred: best, proba }
}

function meanY(y: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < y.length; i++) s += y[i]!
  return s / y.length
}

function buildTree(
  X: number[][],
  y: Array<number | string>,
  task: TreeTask,
  depth: number,
  opts: Required<Pick<CartOptions, 'maxDepth' | 'minSamplesLeaf' | 'minSamplesSplit'>> & { maxFeatures: number; rnd: () => number },
  importances: Float64Array,
): Node {
  const n = y.length
  const impurity = task === 'regression' ? mse(y as number[]) : gini(y)
  const leafPred =
    task === 'regression'
      ? { prediction: meanY(y as number[]), proba: undefined as Record<string, number> | undefined }
      : (() => {
          const m = majority(y)
          return { prediction: m.pred, proba: m.proba }
        })()
  const node: Node = { prediction: leafPred.prediction, proba: leafPred.proba, n, impurity }
  if (depth >= opts.maxDepth || n < opts.minSamplesSplit || impurity === 0) return node

  const p = X[0]!.length
  const featIdx = Array.from({ length: p }, (_, i) => i)
  // shuffle and take maxFeatures
  for (let i = featIdx.length - 1; i > 0; i--) {
    const j = Math.floor(opts.rnd() * (i + 1))
    ;[featIdx[i], featIdx[j]] = [featIdx[j]!, featIdx[i]!]
  }
  const consider = featIdx.slice(0, opts.maxFeatures)

  let bestGain = 0
  let bestFeat = -1
  let bestThr = 0
  let bestLeft: number[] = []
  let bestRight: number[] = []

  for (const j of consider) {
    const vals = [...new Set(X.map((r) => r[j]!))].sort((a, b) => a - b)
    if (vals.length < 2) continue
    // try midpoints between unique values (cap at 32)
    const step = Math.max(1, Math.floor(vals.length / 32))
    for (let vi = 0; vi < vals.length - 1; vi += step) {
      const thr = 0.5 * (vals[vi]! + vals[vi + 1]!)
      const leftIdx: number[] = []
      const rightIdx: number[] = []
      for (let i = 0; i < n; i++) (X[i]![j]! <= thr ? leftIdx : rightIdx).push(i)
      if (leftIdx.length < opts.minSamplesLeaf || rightIdx.length < opts.minSamplesLeaf) continue
      const yL = leftIdx.map((i) => y[i]!)
      const yR = rightIdx.map((i) => y[i]!)
      const impL = task === 'regression' ? mse(yL as number[]) : gini(yL)
      const impR = task === 'regression' ? mse(yR as number[]) : gini(yR)
      const gain = impurity - (leftIdx.length / n) * impL - (rightIdx.length / n) * impR
      if (gain > bestGain) {
        bestGain = gain
        bestFeat = j
        bestThr = thr
        bestLeft = leftIdx
        bestRight = rightIdx
      }
    }
  }
  if (bestFeat < 0 || bestGain <= 0) return node
  importances[bestFeat]! += bestGain * n
  node.feature = bestFeat
  node.threshold = bestThr
  const XL = bestLeft.map((i) => X[i]!)
  const XR = bestRight.map((i) => X[i]!)
  const yL = bestLeft.map((i) => y[i]!)
  const yR = bestRight.map((i) => y[i]!)
  node.left = buildTree(XL, yL, task, depth + 1, opts, importances)
  node.right = buildTree(XR, yR, task, depth + 1, opts, importances)
  return node
}

function walk(node: Node, x: ArrayLike<number>): Node {
  if (node.feature === undefined || !node.left || !node.right) return node
  return x[node.feature]! <= node.threshold! ? walk(node.left, x) : walk(node.right, x)
}

/** CART decision / regression tree. */
export function cart(X: ArrayLike<ArrayLike<number>>, y: ArrayLike<number | string>, options: CartOptions = {}): CartResult {
  const rows = Array.from(X).map((r) => Array.from(r))
  const yy = Array.from(y)
  if (rows.length !== yy.length) throw new RangeError('cart: X/y length mismatch')
  if (rows.length < 2) throw new RangeError('cart: need at least 2 rows')
  const task: TreeTask = options.task ?? (typeof yy[0] === 'number' ? 'regression' : 'classification')
  const p = rows[0]!.length
  let maxFeatures = p
  if (options.maxFeatures === 'sqrt') maxFeatures = Math.max(1, Math.floor(Math.sqrt(p)))
  else if (options.maxFeatures === 'log2') maxFeatures = Math.max(1, Math.floor(Math.log2(p)))
  else if (typeof options.maxFeatures === 'number') maxFeatures = Math.min(p, options.maxFeatures)
  const importances = new Float64Array(p)
  const rnd = rng(options.seed ?? 1)
  const tree = buildTree(rows, yy, task, 0, {
    maxDepth: options.maxDepth ?? 8,
    minSamplesLeaf: options.minSamplesLeaf ?? 1,
    minSamplesSplit: options.minSamplesSplit ?? 2,
    maxFeatures,
    rnd,
  }, importances)
  const total = importances.reduce((a, b) => a + b, 0) || 1
  const featureImportances = Array.from(importances, (v) => v / total)
  const predict = (Xnew: ArrayLike<ArrayLike<number>>) => Array.from(Xnew).map((r) => walk(tree, Array.from(r)).prediction)
  const result: CartResult = { task, tree, featureImportances, predict }
  if (task === 'classification') {
    result.predictProba = (Xnew) =>
      Array.from(Xnew).map((r) => walk(tree, Array.from(r)).proba ?? { [String(walk(tree, Array.from(r)).prediction)]: 1 })
  }
  return result
}

export interface RandomForestResult {
  task: TreeTask
  nTrees: number
  featureImportances: number[]
  predict(X: ArrayLike<ArrayLike<number>>): Array<number | string>
  oobScore?: number
}

/** Random Forest via bagged CART trees. */
export function randomForest(
  X: ArrayLike<ArrayLike<number>>,
  y: ArrayLike<number | string>,
  options: CartOptions & { nTrees?: number } = {},
): RandomForestResult {
  const rows = Array.from(X).map((r) => Array.from(r))
  const yy = Array.from(y)
  const n = rows.length
  const task: TreeTask = options.task ?? (typeof yy[0] === 'number' ? 'regression' : 'classification')
  const nTrees = options.nTrees ?? 50
  const rnd = rng(options.seed ?? 42)
  const trees: CartResult[] = []
  const oobPreds: Array<Array<number | string>> = Array.from({ length: n }, () => [])
  const p = rows[0]!.length
  const imp = new Float64Array(p)
  for (let t = 0; t < nTrees; t++) {
    const idx: number[] = []
    const inBag = new Uint8Array(n)
    for (let i = 0; i < n; i++) {
      const j = Math.floor(rnd() * n)
      idx.push(j)
      inBag[j] = 1
    }
    const Xb = idx.map((i) => rows[i]!)
    const yb = idx.map((i) => yy[i]!)
    const tree = cart(Xb, yb, {
      ...options,
      task,
      maxFeatures: options.maxFeatures ?? 'sqrt',
      seed: Math.floor(rnd() * 1e9),
    })
    trees.push(tree)
    for (let j = 0; j < p; j++) imp[j]! += tree.featureImportances[j]!
    for (let i = 0; i < n; i++) if (!inBag[i]) oobPreds[i]!.push(tree.predict([rows[i]!])[0]!)
  }
  const total = imp.reduce((a, b) => a + b, 0) || 1
  const predict = (Xnew: ArrayLike<ArrayLike<number>>) => {
    const rowsN = Array.from(Xnew).map((r) => Array.from(r))
    return rowsN.map((r) => {
      const votes = trees.map((tr) => tr.predict([r])[0]!)
      if (task === 'regression') return (votes as number[]).reduce((a, b) => a + b, 0) / votes.length
      const counts = new Map<string, number>()
      for (const v of votes) counts.set(String(v), (counts.get(String(v)) ?? 0) + 1)
      let best = ''
      let bestC = -1
      for (const [k, c] of counts) if (c > bestC) {
        bestC = c
        best = k
      }
      return best
    })
  }
  let oobScore: number | undefined
  const oobReady = oobPreds.filter((p) => p.length > 0)
  if (oobReady.length > n * 0.3) {
    if (task === 'regression') {
      let ss = 0
      let sst = 0
      const m = meanY(yy as number[])
      let c = 0
      for (let i = 0; i < n; i++) {
        if (!oobPreds[i]!.length) continue
        const pred = (oobPreds[i] as number[]).reduce((a, b) => a + b, 0) / oobPreds[i]!.length
        ss += (pred - (yy[i] as number)) ** 2
        sst += ((yy[i] as number) - m) ** 2
        c++
      }
      oobScore = 1 - ss / (sst || 1)
      void c
    } else {
      let ok = 0
      let c = 0
      for (let i = 0; i < n; i++) {
        if (!oobPreds[i]!.length) continue
        const counts = new Map<string, number>()
        for (const v of oobPreds[i]!) counts.set(String(v), (counts.get(String(v)) ?? 0) + 1)
        let best = ''
        let bestC = -1
        for (const [k, cnt] of counts) if (cnt > bestC) {
          bestC = cnt
          best = k
        }
        if (best === String(yy[i])) ok++
        c++
      }
      oobScore = ok / c
    }
  }
  return { task, nTrees, featureImportances: Array.from(imp, (v) => v / total), predict, oobScore }
}

export interface TreeNetResult {
  task: 'regression' | 'classification'
  nTrees: number
  learningRate: number
  classes?: string[]
  predict(X: ArrayLike<ArrayLike<number>>): Array<number | string>
  predictProba?(X: ArrayLike<ArrayLike<number>>): Array<Record<string, number>>
  trainRmse?: number
  trainLogLoss?: number
  trainAccuracy?: number
}

function softmaxRows(F: number[][]): number[][] {
  return F.map((row) => {
    const m = Math.max(...row)
    const ex = row.map((v) => Math.exp(v - m))
    const s = ex.reduce((a, b) => a + b, 0)
    return ex.map((e) => e / s)
  })
}

/**
 * TreeNet-style gradient boosting.
 * - regression: squared-error residual trees
 * - classification: multinomial deviance (K=2 → Bernoulli) with one tree per class per stage
 */
export function treeNet(
  X: ArrayLike<ArrayLike<number>>,
  y: ArrayLike<number | string>,
  options: {
    task?: 'regression' | 'classification'
    nTrees?: number
    learningRate?: number
    maxDepth?: number
    seed?: number
  } = {},
): TreeNetResult {
  const rows = Array.from(X).map((r) => Array.from(r))
  const yy = Array.from(y)
  if (rows.length !== yy.length) throw new RangeError('treeNet: X/y length mismatch')
  const task = options.task ?? (typeof yy[0] === 'number' && yy.every((v) => typeof v === 'number') ? 'regression' : 'classification')
  const nTrees = options.nTrees ?? 50
  const lr = options.learningRate ?? 0.1
  const maxDepth = options.maxDepth ?? 3
  const seed = options.seed ?? 1

  if (task === 'regression') {
    const yNum = yy as number[]
    const pred = new Array(yNum.length).fill(meanY(yNum))
    const trees: CartResult[] = []
    const F0 = meanY(yNum)
    for (let t = 0; t < nTrees; t++) {
      const resid = yNum.map((yi, i) => yi - pred[i]!)
      const tree = cart(rows, resid, { task: 'regression', maxDepth, seed: seed + t })
      trees.push(tree)
      const update = tree.predict(rows) as number[]
      for (let i = 0; i < pred.length; i++) pred[i]! += lr * update[i]!
    }
    let sse = 0
    for (let i = 0; i < yNum.length; i++) sse += (yNum[i]! - pred[i]!) ** 2
    return {
      task: 'regression',
      nTrees,
      learningRate: lr,
      trainRmse: Math.sqrt(sse / yNum.length),
      predict: (Xnew) => {
        const base = new Array(Array.from(Xnew).length).fill(F0)
        for (const tree of trees) {
          const u = tree.predict(Xnew) as number[]
          for (let i = 0; i < base.length; i++) base[i]! += lr * u[i]!
        }
        return base
      },
    }
  }

  // Classification: multinomial deviance
  const classes = [...new Set(yy.map(String))].sort()
  const K = classes.length
  if (K < 2) throw new RangeError('treeNet classification: need at least 2 classes')
  const classIndex = new Map(classes.map((c, i) => [c, i]))
  const n = rows.length
  const Y = Array.from({ length: n }, () => new Array(K).fill(0))
  for (let i = 0; i < n; i++) Y[i]![classIndex.get(String(yy[i]))!] = 1
  // F[i][k] scores; init 0
  const F = Array.from({ length: n }, () => new Array(K).fill(0))
  const stageTrees: CartResult[][] = []

  for (let t = 0; t < nTrees; t++) {
    const P = softmaxRows(F)
    const treesK: CartResult[] = []
    for (let k = 0; k < K; k++) {
      const resid = Y.map((yi, i) => yi[k]! - P[i]![k]!)
      const tree = cart(rows, resid, { task: 'regression', maxDepth, seed: seed + t * 17 + k })
      treesK.push(tree)
      const update = tree.predict(rows) as number[]
      for (let i = 0; i < n; i++) F[i]![k]! += lr * update[i]!
    }
    stageTrees.push(treesK)
  }

  const Pfinal = softmaxRows(F)
  let logLoss = 0
  let correct = 0
  for (let i = 0; i < n; i++) {
    const ki = classIndex.get(String(yy[i]))!
    logLoss += -Math.log(Math.max(Pfinal[i]![ki]!, 1e-15))
    let best = 0
    for (let k = 1; k < K; k++) if (Pfinal[i]![k]! > Pfinal[i]![best]!) best = k
    if (best === ki) correct++
  }

  const predictScores = (Xnew: ArrayLike<ArrayLike<number>>) => {
    const rowsN = Array.from(Xnew).map((r) => Array.from(r))
    const scores = Array.from({ length: rowsN.length }, () => new Array(K).fill(0))
    for (const treesK of stageTrees) {
      for (let k = 0; k < K; k++) {
        const u = treesK[k]!.predict(rowsN) as number[]
        for (let i = 0; i < rowsN.length; i++) scores[i]![k]! += lr * u[i]!
      }
    }
    return scores
  }

  return {
    task: 'classification',
    nTrees,
    learningRate: lr,
    classes,
    trainLogLoss: logLoss / n,
    trainAccuracy: correct / n,
    predict: (Xnew) => {
      const scores = predictScores(Xnew)
      const P = softmaxRows(scores)
      return P.map((row) => {
        let best = 0
        for (let k = 1; k < K; k++) if (row[k]! > row[best]!) best = k
        return classes[best]!
      })
    },
    predictProba: (Xnew) => {
      const scores = predictScores(Xnew)
      const P = softmaxRows(scores)
      return P.map((row) => {
        const out: Record<string, number> = {}
        for (let k = 0; k < K; k++) out[classes[k]!] = row[k]!
        return out
      })
    },
  }
}

export interface MarsResult {
  terms: Array<{ feature: number; knot: number; sign: 1 | -1; coef: number }>
  intercept: number
  r2: number
  sse: number
  gcv: number
  nTerms: number
  predict(X: ArrayLike<ArrayLike<number>>): number[]
}

function marsGcv(sse: number, n: number, nTerms: number): number {
  const df = 1 + nTerms
  const den = 1 - df / n
  if (!(den > 1e-12)) return Infinity
  return sse / n / (den * den)
}

/**
 * Basic MARS: forward selection of hinge terms, then optional backward GCV prune.
 */
export function mars(
  X: ArrayLike<ArrayLike<number>>,
  y: ArrayLike<number>,
  options: { maxTerms?: number; prune?: boolean } = {},
): MarsResult {
  const rows = Array.from(X).map((r) => Array.from(r))
  const yy = Array.from(y)
  const n = rows.length
  const p = rows[0]!.length
  const maxTerms = options.maxTerms ?? Math.min(10, 2 * p)
  const doPrune = options.prune !== false
  type Term = { feature: number; knot: number; sign: 1 | -1 }
  const selected: Term[] = []

  const design = (terms: Term[]) => {
    const cols = [new Array(n).fill(1)]
    for (const t of terms) cols.push(rows.map((r) => Math.max(0, t.sign * (r[t.feature]! - t.knot))))
    return cols
  }
  const fitSse = (terms: Term[]) => {
    const cols = design(terms)
    const M = matrix(n, cols.length)
    for (let j = 0; j < cols.length; j++) for (let i = 0; i < n; i++) M.data[i * cols.length + j] = cols[j]![i]!
    const fit = lstsq(M, yy)
    return { sse: fit.sse, coef: Array.from(fit.coef) }
  }

  let bestSse = fitSse([]).sse
  for (let t = 0; t < maxTerms; t++) {
    let improved: Term | null = null
    let improvedSse = bestSse
    for (let j = 0; j < p; j++) {
      const knots = [...new Set(rows.map((r) => r[j]!))].sort((a, b) => a - b)
      const step = Math.max(1, Math.floor(knots.length / 10))
      for (let ki = 1; ki < knots.length - 1; ki += step) {
        for (const sign of [1, -1] as const) {
          const trial = selected.concat([{ feature: j, knot: knots[ki]!, sign }])
          const { sse } = fitSse(trial)
          if (sse < improvedSse - 1e-9) {
            improvedSse = sse
            improved = { feature: j, knot: knots[ki]!, sign }
          }
        }
      }
    }
    if (!improved) break
    selected.push(improved)
    bestSse = improvedSse
  }

  if (doPrune && selected.length > 0) {
    let gcv = marsGcv(bestSse, n, selected.length)
    let improved = true
    while (improved && selected.length > 0) {
      improved = false
      let bestIdx = -1
      let bestG = gcv
      let bestS = bestSse
      for (let i = 0; i < selected.length; i++) {
        const trial = selected.filter((_, j) => j !== i)
        const { sse } = fitSse(trial)
        const g = marsGcv(sse, n, trial.length)
        if (g < bestG - 1e-12) {
          bestG = g
          bestIdx = i
          bestS = sse
        }
      }
      if (bestIdx >= 0) {
        selected.splice(bestIdx, 1)
        gcv = bestG
        bestSse = bestS
        improved = true
      }
    }
  }

  const { coef, sse } = fitSse(selected)
  const intercept = coef[0]!
  const terms = selected.map((t, i) => ({ ...t, coef: coef[i + 1]! }))
  const m = meanY(yy)
  let sst = 0
  for (const yi of yy) sst += (yi - m) ** 2
  const gcv = marsGcv(sse, n, terms.length)
  return {
    terms,
    intercept,
    r2: sst > 0 ? 1 - sse / sst : NaN,
    sse,
    gcv,
    nTerms: terms.length,
    predict: (Xnew) =>
      Array.from(Xnew).map((r) => {
        const x = Array.from(r)
        let yhat = intercept
        for (const t of terms) yhat += t.coef * Math.max(0, t.sign * (x[t.feature]! - t.knot))
        return yhat
      }),
  }
}
