/**
 * Multivariate additions (Minitab Stat › Multivariate): Cluster Variables (hierarchical clustering of
 * variables on correlation distance), Multiple Correspondence Analysis (indicator / Burt matrix),
 * Item Analysis (Cronbach's α with item-total statistics) and the promax oblique rotation for factor
 * loadings.
 */
import { correspondence, type CorrespondenceResult, type HclustResult } from './multivariate.js'
import { inverse, matmul, matrix, transpose } from './linalg.js'
import { t as tDist } from './dist.js'

export interface ClusterVariablesResult extends HclustResult {
  names: string[]
  /** Distance measure: 1 − r (default) or 1 − |r|. */
  distance: 'correlation' | 'absolute correlation'
  correlation: number[][]
  /** Cluster membership for `nClusters` (when requested). */
  membership?: number[]
  /** Similarity levels (100·(1 − distance)) at each merge. */
  similarity: number[]
}

function corrMatrix(cols: number[][]): number[][] {
  const p = cols.length
  const n = cols[0]!.length
  const means = cols.map((c) => c.reduce((a, b) => a + b, 0) / n)
  const sds = cols.map((c, j) => Math.sqrt(c.reduce((a, b) => a + (b - means[j]!) ** 2, 0) / (n - 1)))
  const R = Array.from({ length: p }, () => new Array<number>(p).fill(0))
  for (let a = 0; a < p; a++) for (let b = a; b < p; b++) {
    let s = 0
    for (let i = 0; i < n; i++) s += (cols[a]![i]! - means[a]!) * (cols[b]![i]! - means[b]!)
    const r = s / ((n - 1) * sds[a]! * sds[b]!)
    R[a]![b] = r
    R[b]![a] = r
  }
  return R
}

/** Cut a merge tree into `k` clusters (membership 1…k in order of first appearance). */
export function cutTree(h: HclustResult, nLeaves: number, k: number): number[] {
  const parent = new Map<number, number[]>() // cluster id → leaves
  for (let i = 0; i < nLeaves; i++) parent.set(-1 - i, [i])
  const clusters = new Map<number, number[]>(parent)
  let id = 0
  const steps = h.merge.length - (k - 1)
  for (let s = 0; s < steps; s++) {
    const [a, b] = h.merge[s]!
    const leaves = [...(clusters.get(a) ?? []), ...(clusters.get(b) ?? [])]
    clusters.delete(a)
    clusters.delete(b)
    clusters.set(id++, leaves)
  }
  const membership = new Array<number>(nLeaves).fill(0)
  let c = 1
  for (const leaves of [...clusters.values()].sort((x, y) => Math.min(...x) - Math.min(...y))) {
    for (const l of leaves) membership[l] = c
    c++
  }
  return membership
}

/**
 * Cluster Variables (Minitab): agglomerative clustering of the variables using 1 − r (or 1 − |r|) as the
 * distance, with the linkage methods of `hclust`.
 */
export function clusterVariables(
  data: Record<string, ArrayLike<number>>,
  options: { method?: 'single' | 'complete' | 'average' | 'ward'; distance?: 'correlation' | 'absolute correlation'; nClusters?: number } = {},
): ClusterVariablesResult {
  const names = Object.keys(data)
  const cols = names.map((k) => Array.from(data[k]!))
  const p = names.length
  if (p < 2) throw new RangeError('clusterVariables needs at least 2 variables')
  const R = corrMatrix(cols)
  const distance = options.distance ?? 'correlation'
  // agglomerate directly on the correlation-distance matrix (Lance–Williams updates)
  const D = R.map((row) => row.map((r) => (distance === 'correlation' ? 1 - r : 1 - Math.abs(r))))
  const method = options.method ?? 'average'
  const active: number[] = names.map((_, i) => i)
  const size = new Array<number>(p).fill(1)
  const ids = names.map((_, i) => -1 - i)
  const merge: Array<[number, number, number]> = []
  const dist = D.map((r) => r.slice())
  let nextId = 0
  while (active.length > 1) {
    let bi = 0
    let bj = 1
    let best = Infinity
    for (let x = 0; x < active.length; x++) for (let y = x + 1; y < active.length; y++) {
      const d = dist[active[x]!]![active[y]!]!
      if (d < best) {
        best = d
        bi = active[x]!
        bj = active[y]!
      }
    }
    merge.push([ids[bi]!, ids[bj]!, best])
    // Lance–Williams update into bi
    for (const k of active) {
      if (k === bi || k === bj) continue
      const dik = dist[bi]![k]!
      const djk = dist[bj]![k]!
      let d: number
      if (method === 'single') d = Math.min(dik, djk)
      else if (method === 'complete') d = Math.max(dik, djk)
      else if (method === 'average') d = (size[bi]! * dik + size[bj]! * djk) / (size[bi]! + size[bj]!)
      else {
        const ni = size[bi]!
        const nj = size[bj]!
        const nk = size[k]!
        d = Math.sqrt(Math.max(0, ((ni + nk) * dik * dik + (nj + nk) * djk * djk - nk * best * best) / (ni + nj + nk)))
      }
      dist[bi]![k] = d
      dist[k]![bi] = d
    }
    size[bi] = size[bi]! + size[bj]!
    ids[bi] = nextId++
    active.splice(active.indexOf(bj), 1)
  }
  const order = (() => {
    const leavesOf = new Map<number, number[]>()
    for (let i = 0; i < p; i++) leavesOf.set(-1 - i, [i])
    merge.forEach(([a, b], s) => leavesOf.set(s, [...leavesOf.get(a)!, ...leavesOf.get(b)!]))
    return leavesOf.get(merge.length - 1) ?? names.map((_, i) => i)
  })()
  const result: ClusterVariablesResult = {
    merge,
    order,
    method,
    names,
    distance,
    correlation: R,
    similarity: merge.map(([, , d]) => 100 * (1 - d)),
  }
  if (options.nClusters) result.membership = cutTree(result, p, options.nClusters)
  return result
}

export interface McaResult extends CorrespondenceResult {
  variables: string[]
  /** Category labels as 'variable:level'. */
  categories: string[]
  /** Burt-adjusted (Benzécri) inertias when `method: 'burt'`. */
  method: 'indicator' | 'burt'
}

/**
 * Multiple Correspondence Analysis (Minitab): correspondence analysis of the indicator matrix (default)
 * or of the Burt matrix of two or more categorical variables.
 */
export function multipleCorrespondence(data: Record<string, ArrayLike<unknown>>, options: { method?: 'indicator' | 'burt'; nComponents?: number } = {}): McaResult {
  const variables = Object.keys(data)
  if (variables.length < 2) throw new RangeError('multipleCorrespondence needs at least 2 variables')
  const n = data[variables[0]!]!.length
  const categories: string[] = []
  const columns: number[][] = []
  for (const v of variables) {
    const col = data[v]!
    if (col.length !== n) throw new RangeError('multipleCorrespondence: variables must have equal length')
    const levels = [...new Set(Array.from({ length: n }, (_, i) => String(col[i])))].sort()
    for (const l of levels) {
      categories.push(`${v}:${l}`)
      columns.push(Array.from({ length: n }, (_, i) => (String(col[i]) === l ? 1 : 0)))
    }
  }
  const method = options.method ?? 'indicator'
  const Q = categories.length
  let table: number[][]
  let rowNames: string[]
  if (method === 'indicator') {
    table = Array.from({ length: n }, (_, i) => columns.map((c) => c[i]!))
    rowNames = Array.from({ length: n }, (_, i) => String(i + 1))
  } else {
    table = Array.from({ length: Q }, (_, a) => Array.from({ length: Q }, (_, b) => columns[a]!.reduce((s, v, i) => s + v * columns[b]![i]!, 0)))
    rowNames = categories
  }
  const ca = correspondence(table, { rowNames, colNames: categories, nComponents: options.nComponents })
  return { ...ca, variables, categories, method }
}

export interface ItemAnalysisResult {
  test: 'item analysis'
  alpha: number
  /** Standardized α (based on the average inter-item correlation). */
  standardizedAlpha: number
  nItems: number
  n: number
  items: Array<{ name: string; mean: number; sd: number; itemTotalCorr: number; adjustedItemTotalCorr: number; squaredMultipleCorr: number; alphaIfDeleted: number }>
  /** Correlation matrix of the items. */
  correlation: number[][]
  /** Total-score statistics. */
  total: { mean: number; sd: number }
}

/** Item Analysis (Minitab): Cronbach's α, item-total correlations, squared multiple correlations and α if deleted. */
export function itemAnalysis(items: Record<string, ArrayLike<number>>): ItemAnalysisResult {
  const names = Object.keys(items)
  const k = names.length
  if (k < 2) throw new RangeError('itemAnalysis needs at least 2 items')
  const cols = names.map((nm) => Array.from(items[nm]!))
  const n = cols[0]!.length
  for (const c of cols) if (c.length !== n) throw new RangeError('itemAnalysis: items must have equal length')
  const mean = cols.map((c) => c.reduce((a, b) => a + b, 0) / n)
  const variance = cols.map((c, j) => c.reduce((a, b) => a + (b - mean[j]!) ** 2, 0) / (n - 1))
  const total = Array.from({ length: n }, (_, i) => cols.reduce((s, c) => s + c[i]!, 0))
  const tMean = total.reduce((a, b) => a + b, 0) / n
  const tVar = total.reduce((a, b) => a + (b - tMean) ** 2, 0) / (n - 1)
  const alphaOf = (idx: number[]) => {
    const tot = Array.from({ length: n }, (_, i) => idx.reduce((s, j) => s + cols[j]![i]!, 0))
    const m = tot.reduce((a, b) => a + b, 0) / n
    const v = tot.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1)
    const sumVar = idx.reduce((s, j) => s + variance[j]!, 0)
    return (idx.length / (idx.length - 1)) * (1 - sumVar / v)
  }
  const R = corrMatrix(cols)
  let rbar = 0
  for (let a = 0; a < k; a++) for (let b = a + 1; b < k; b++) rbar += R[a]![b]!
  rbar /= (k * (k - 1)) / 2
  const corr = (x: number[], y: number[]) => {
    const mx = x.reduce((a, b) => a + b, 0) / n
    const my = y.reduce((a, b) => a + b, 0) / n
    let sxy = 0
    let sxx = 0
    let syy = 0
    for (let i = 0; i < n; i++) {
      sxy += (x[i]! - mx) * (y[i]! - my)
      sxx += (x[i]! - mx) ** 2
      syy += (y[i]! - my) ** 2
    }
    return sxy / Math.sqrt(sxx * syy)
  }
  const Rm = matrix(k, k, Float64Array.from(R.flat()))
  const Rinv = inverse(Rm)
  const rows = names.map((name, j) => {
    const others = Array.from({ length: n }, (_, i) => total[i]! - cols[j]![i]!)
    return {
      name,
      mean: mean[j]!,
      sd: Math.sqrt(variance[j]!),
      itemTotalCorr: corr(cols[j]!, total),
      adjustedItemTotalCorr: corr(cols[j]!, others),
      squaredMultipleCorr: 1 - 1 / Rinv.data[j * k + j]!,
      alphaIfDeleted: k > 2 ? alphaOf(names.map((_, i) => i).filter((i) => i !== j)) : NaN,
    }
  })
  return {
    test: 'item analysis',
    alpha: alphaOf(names.map((_, i) => i)),
    standardizedAlpha: (k * rbar) / (1 + (k - 1) * rbar),
    nItems: k,
    n,
    items: rows,
    correlation: R,
    total: { mean: tMean, sd: Math.sqrt(tVar) },
  }
}

/**
 * Promax oblique rotation (Hendrickson & White 1964) of orthogonally (varimax) rotated loadings:
 * target = sign(Λ)|Λ|^power, Procrustes fit, normalized to unit-variance factors. Returns the pattern
 * loadings and the factor correlation matrix.
 */
export function promax(loadings: ArrayLike<ArrayLike<number>>, options: { power?: number } = {}): { loadings: number[][]; factorCorrelation: number[][] } {
  const L = Array.from(loadings).map((r) => Array.from(r))
  const p = L.length
  const k = L[0]!.length
  const m = options.power ?? 4
  const A = matrix(p, k, Float64Array.from(L.flat()))
  // normalize rows (Kaiser) for the target
  const target = matrix(p, k)
  for (let i = 0; i < p; i++) {
    let h = 0
    for (let j = 0; j < k; j++) h += A.data[i * k + j]! ** 2
    h = Math.sqrt(h) || 1
    for (let j = 0; j < k; j++) {
      const v = A.data[i * k + j]! / h
      target.data[i * k + j] = Math.sign(v) * Math.abs(v) ** m * h
    }
  }
  // Procrustes: T = (AᵀA)⁻¹ Aᵀ target
  const At = transpose(A)
  let T = matmul(inverse(matmul(At, A)), matmul(At, target))
  // normalize columns of T so that diag((TᵀT)⁻¹) = 1
  const TtT = matmul(transpose(T), T)
  const TtTinv = inverse(TtT)
  const scale = new Float64Array(k)
  for (let j = 0; j < k; j++) scale[j] = Math.sqrt(TtTinv.data[j * k + j]!)
  const Ts = matrix(k, k)
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) Ts.data[i * k + j] = T.data[i * k + j]! * scale[j]!
  T = Ts
  const P = matmul(A, T)
  const Tinv = inverse(T)
  const Phi = matmul(Tinv, transpose(Tinv))
  const out: number[][] = []
  for (let i = 0; i < p; i++) out.push(Array.from({ length: k }, (_, j) => P.data[i * k + j]!))
  const fc: number[][] = []
  for (let i = 0; i < k; i++) fc.push(Array.from({ length: k }, (_, j) => Phi.data[i * k + j]!))
  return { loadings: out, factorCorrelation: fc }
}

/** Convenience: t-based test of a correlation's significance (used by item analysis reports). */
export function corrPValue(r: number, n: number): number {
  const t = r * Math.sqrt((n - 2) / Math.max(1e-300, 1 - r * r))
  return Math.min(1, 2 * tDist(n - 2).sf(Math.abs(t)))
}

