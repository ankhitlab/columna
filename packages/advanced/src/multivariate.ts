/**
 * Tier 5.4 — Multivariate:
 * PCA, Factor analysis (PCA extraction + varimax), hierarchical & k-means clustering,
 * LDA / QDA discriminant analysis, correspondence analysis.
 */
import { cleanNumbers } from './tests.js'
import { fromColumns, gram, matrix, matmul, svd, symmetricEigen, transpose, type Matrix } from './linalg.js'

function colMeans(X: Matrix): Float64Array {
  const m = new Float64Array(X.cols)
  for (let j = 0; j < X.cols; j++) {
    let s = 0
    for (let i = 0; i < X.rows; i++) s += X.data[i * X.cols + j]!
    m[j] = s / X.rows
  }
  return m
}
function standardize(X: Matrix, center = true, scale = true): { Z: Matrix; mean: Float64Array; sd: Float64Array } {
  const mean = colMeans(X)
  const sd = new Float64Array(X.cols)
  for (let j = 0; j < X.cols; j++) {
    let s2 = 0
    for (let i = 0; i < X.rows; i++) s2 += (X.data[i * X.cols + j]! - mean[j]!) ** 2
    sd[j] = Math.sqrt(s2 / Math.max(1, X.rows - 1)) || 1
  }
  const Z = matrix(X.rows, X.cols)
  for (let i = 0; i < X.rows; i++) for (let j = 0; j < X.cols; j++) {
    let v = X.data[i * X.cols + j]!
    if (center) v -= mean[j]!
    if (scale) v /= sd[j]!
    Z.data[i * X.cols + j] = v
  }
  return { Z, mean, sd }
}

function asMatrix(data: ArrayLike<ArrayLike<number>> | Record<string, ArrayLike<number | null | undefined>>, names?: string[]): { X: Matrix; names: string[] } {
  if (Array.isArray(data)) {
    const rows = data as ArrayLike<number>[]
    const n = rows.length
    const p = Array.from(rows[0]!).length
    const X = matrix(n, p)
    for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) X.data[i * p + j] = Number((rows[i] as ArrayLike<number>)[j])
    return { X, names: names ?? Array.from({ length: p }, (_, j) => `X${j + 1}`) }
  }
  const keys = names ?? Object.keys(data)
  const cols = keys.map((k) => Array.from(cleanNumbers((data as Record<string, ArrayLike<number | null | undefined>>)[k]!)))
  const n = cols[0]!.length
  if (cols.some((c) => c.length !== n)) throw new RangeError('pca: columns must have equal length')
  return { X: fromColumns(cols), names: keys }
}

// ---- PCA ------------------------------------------------------------------------------------------

export interface PcaResult {
  n: number
  p: number
  names: string[]
  /** Eigenvalues (variances of PCs), descending. */
  eigenvalues: number[]
  /** Proportion / cumulative proportion of variance. */
  proportion: number[]
  cumulative: number[]
  /** Loadings V (p × k). */
  loadings: number[][]
  /** Scores U·S (n × k). */
  scores: number[][]
  mean: number[]
  sd: number[]
}

/** Principal Component Analysis on the correlation (default) or covariance matrix via SVD. */
export function pca(
  data: ArrayLike<ArrayLike<number>> | Record<string, ArrayLike<number | null | undefined>>,
  options: { nComponents?: number; scale?: boolean; names?: string[] } = {},
): PcaResult {
  const { X, names } = asMatrix(data, options.names)
  const { Z, mean, sd } = standardize(X, true, options.scale !== false)
  // eigen-decomposition of the p × p Gram matrix (O(np²)) instead of an SVD of the n × p data
  const eig = symmetricEigen(gram(Z))
  const s = Float64Array.from(eig.values, (l) => Math.sqrt(Math.max(0, l)))
  const v = eig.vectors
  const k = Math.min(options.nComponents ?? s.length, s.length)
  const eigenvalues = Array.from({ length: k }, (_, j) => (s[j]! * s[j]!) / Math.max(1, X.rows - 1))
  const total = eigenvalues.reduce((a, b) => a + b, 0) || 1
  // If we truncated, still use full sum of all s² for proportions when possible
  let totalAll = 0
  for (let j = 0; j < s.length; j++) totalAll += (s[j]! * s[j]!) / Math.max(1, X.rows - 1)
  const denom = totalAll || total
  const proportion = eigenvalues.map((e) => e / denom)
  const cumulative: number[] = []
  let c = 0
  for (const p of proportion) {
    c += p
    cumulative.push(c)
  }
  const loadings: number[][] = []
  for (let i = 0; i < X.cols; i++) {
    const row: number[] = []
    for (let j = 0; j < k; j++) row.push(v.data[i * v.cols + j]!)
    loadings.push(row)
  }
  const scores: number[][] = []
  for (let i = 0; i < X.rows; i++) {
    const row: number[] = []
    for (let j = 0; j < k; j++) {
      let t = 0
      for (let c2 = 0; c2 < X.cols; c2++) t += Z.data[i * Z.cols + c2]! * v.data[c2 * v.cols + j]!
      row.push(t)
    }
    scores.push(row)
  }
  return {
    n: X.rows,
    p: X.cols,
    names,
    eigenvalues,
    proportion,
    cumulative,
    loadings,
    scores,
    mean: Array.from(mean),
    sd: Array.from(sd),
  }
}

// ---- Factor analysis + varimax --------------------------------------------------------------------

export interface FactorResult {
  nFactors: number
  loadings: number[][]
  uniqueVariances: number[]
  /** After varimax if requested. */
  rotatedLoadings?: number[][]
  rotation?: 'none' | 'varimax'
  eigenvalues: number[]
  names: string[]
  method: 'pca' | 'ml'
  logLik?: number
}

function varimax(loadings: Matrix, maxIter = 50): Matrix {
  // Kaiser varimax on p × k loadings
  const p = loadings.rows
  const k = loadings.cols
  const L = matrix(p, k, Float64Array.from(loadings.data))
  for (let iter = 0; iter < maxIter; iter++) {
    for (let j = 0; j < k - 1; j++) {
      for (let l = j + 1; l < k; l++) {
        let A = 0
        let B = 0
        let C = 0
        let D = 0
        for (let i = 0; i < p; i++) {
          const x = L.data[i * k + j]!
          const y = L.data[i * k + l]!
          const u = x * x - y * y
          const v = 2 * x * y
          A += u
          B += v
          C += u * u - v * v
          D += 2 * u * v
        }
        const num = D - (2 * A * B) / p
        const den = C - (A * A - B * B) / p
        const phi = 0.25 * Math.atan2(num, den)
        const c = Math.cos(phi)
        const s = Math.sin(phi)
        if (Math.abs(s) < 1e-12) continue
        for (let i = 0; i < p; i++) {
          const x = L.data[i * k + j]!
          const y = L.data[i * k + l]!
          L.data[i * k + j] = c * x + s * y
          L.data[i * k + l] = -s * x + c * y
        }
      }
    }
  }
  return L
}

function correlationMatrix(Z: Matrix): Matrix {
  const n = Z.rows
  const p = Z.cols
  const R = matrix(p, p)
  for (let i = 0; i < p; i++) {
    for (let j = i; j < p; j++) {
      let s = 0
      for (let t = 0; t < n; t++) s += Z.data[t * p + i]! * Z.data[t * p + j]!
      const r = s / Math.max(1, n - 1)
      R.data[i * p + j] = r
      R.data[j * p + i] = r
    }
  }
  return R
}

function detSymmetric(A: Matrix): number {
  // Cholesky-style product of pivots via Gaussian elimination copy
  const n = A.rows
  const M = Float64Array.from(A.data)
  let det = 1
  for (let col = 0; col < n; col++) {
    let piv = col
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r * n + col]!) > Math.abs(M[piv * n + col]!)) piv = r
    if (Math.abs(M[piv * n + col]!) < 1e-14) return 0
    if (piv !== col) {
      for (let j = 0; j < n; j++) {
        const tmp = M[col * n + j]!
        M[col * n + j] = M[piv * n + j]!
        M[piv * n + j] = tmp
      }
      det = -det
    }
    const d = M[col * n + col]!
    det *= d
    for (let r = col + 1; r < n; r++) {
      const f = M[r * n + col]! / d
      for (let j = col; j < n; j++) M[r * n + j]! -= f * M[col * n + j]!
    }
  }
  return det
}

function invertSymmetricPD(A: Matrix): Matrix | null {
  const n = A.rows
  const M = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => A.data[i * n + j]!))
  const I = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))
  for (let col = 0; col < n; col++) {
    let piv = col
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r]![col]!) > Math.abs(M[piv]![col]!)) piv = r
    if (Math.abs(M[piv]![col]!) < 1e-14) return null
    ;[M[col], M[piv]] = [M[piv]!, M[col]!]
    ;[I[col], I[piv]] = [I[piv]!, I[col]!]
    const d = M[col]![col]!
    for (let j = 0; j < n; j++) {
      M[col]![j]! /= d
      I[col]![j]! /= d
    }
    for (let r = 0; r < n; r++) {
      if (r === col) continue
      const f = M[r]![col]!
      for (let j = 0; j < n; j++) {
        M[r]![j]! -= f * M[col]![j]!
        I[r]![j]! -= f * I[col]![j]!
      }
    }
  }
  const out = matrix(n, n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out.data[i * n + j] = I[i]![j]!
  return out
}

/** Wishart log-likelihood for correlation FA: const − (n/2)[log|Σ| + tr(R Σ⁻¹)]. */
function faWishartLogLik(R: Matrix, Lambda: number[][], Psi: number[], nObs: number): number {
  const p = R.rows
  const k = Lambda[0]?.length ?? 0
  const Sigma = matrix(p, p)
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      let s = i === j ? Psi[i]! : 0
      for (let f = 0; f < k; f++) s += Lambda[i]![f]! * Lambda[j]![f]!
      Sigma.data[i * p + j] = s
    }
  }
  const det = detSymmetric(Sigma)
  if (!(det > 1e-18)) return -Infinity
  const inv = invertSymmetricPD(Sigma)
  if (!inv) return -Infinity
  let tr = 0
  for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) tr += R.data[i * p + j]! * inv.data[j * p + i]!
  return -0.5 * nObs * (Math.log(det) + tr)
}

/**
 * Factor analysis by PCA extraction (default) or maximum-likelihood (Wishart) with optional varimax.
 */
export function factorAnalysis(
  data: ArrayLike<ArrayLike<number>> | Record<string, ArrayLike<number | null | undefined>>,
  options: {
    nFactors?: number
    rotate?: 'none' | 'varimax'
    names?: string[]
    method?: 'pca' | 'ml'
  } = {},
): FactorResult {
  const method = options.method ?? 'pca'
  const p0 = pca(data, { scale: true, names: options.names })
  const nFactors = options.nFactors ?? Math.max(1, Math.min(p0.p, p0.eigenvalues.filter((e) => e >= 1).length || 1))
  let loadings: number[][] = p0.loadings.map((row) =>
    row.slice(0, nFactors).map((v, j) => v * Math.sqrt(p0.eigenvalues[j]!)),
  )
  let uniqueVariances = loadings.map((row) => Math.max(0.01, 1 - row.reduce((s, v) => s + v * v, 0)))
  let logLik: number | undefined

  if (method === 'ml') {
    const { X } = asMatrix(data, options.names)
    const { Z } = standardize(X, true, true)
    const R = correlationMatrix(Z)
    const p = p0.p
    // Coordinate ascent on Lambda and Psi
    let best = faWishartLogLik(R, loadings, uniqueVariances, X.rows)
    for (let pass = 0; pass < 80; pass++) {
      let improved = false
      for (let i = 0; i < p; i++) {
        for (let f = 0; f < nFactors; f++) {
          for (const step of [0.05, -0.05, 0.01, -0.01]) {
            const trial = loadings.map((row) => row.slice())
            trial[i]![f]! += step
            // keep communality < 1
            const h2 = trial[i]!.reduce((s, v) => s + v * v, 0)
            if (h2 >= 0.99) continue
            const psi = uniqueVariances.slice()
            psi[i] = Math.max(0.01, 1 - h2)
            const ll = faWishartLogLik(R, trial, psi, X.rows)
            if (ll > best + 1e-9) {
              loadings = trial
              uniqueVariances = psi
              best = ll
              improved = true
            }
          }
        }
        // Psi step (recompute from loadings + small free move)
        for (const step of [0.02, -0.02]) {
          const psi = uniqueVariances.slice()
          psi[i] = Math.min(0.99, Math.max(0.01, psi[i]! + step))
          const ll = faWishartLogLik(R, loadings, psi, X.rows)
          if (ll > best + 1e-9) {
            uniqueVariances = psi
            best = ll
            improved = true
          }
        }
      }
      if (!improved) break
    }
    logLik = best
  }

  const L = matrix(p0.p, nFactors)
  for (let i = 0; i < p0.p; i++) for (let j = 0; j < nFactors; j++) L.data[i * nFactors + j] = loadings[i]![j]!
  let rotatedLoadings: number[][] | undefined
  const rotation = options.rotate ?? 'varimax'
  if (rotation === 'varimax' && nFactors > 1) {
    const R = varimax(L)
    rotatedLoadings = []
    for (let i = 0; i < p0.p; i++) {
      const row: number[] = []
      for (let j = 0; j < nFactors; j++) row.push(R.data[i * nFactors + j]!)
      rotatedLoadings.push(row)
    }
  }
  return {
    nFactors,
    loadings,
    uniqueVariances,
    rotatedLoadings,
    rotation: rotation === 'varimax' && nFactors > 1 ? 'varimax' : 'none',
    eigenvalues: p0.eigenvalues.slice(0, nFactors),
    names: p0.names,
    method,
    logLik,
  }
}

// ---- Clustering -----------------------------------------------------------------------------------

export interface KMeansResult {
  k: number
  centers: number[][]
  cluster: number[]
  inertia: number
  sizes: number[]
  iterations: number
}

/** k-means with k-means++ init (seeded). */
export function kmeans(data: ArrayLike<ArrayLike<number>>, options: { k: number; maxIter?: number; seed?: number }): KMeansResult {
  const rows = Array.from(data).map((r) => Array.from(r))
  const n = rows.length
  const p = rows[0]!.length
  const k = options.k
  if (k < 1 || k > n) throw new RangeError('kmeans: invalid k')
  let seed = (options.seed ?? 1) >>> 0
  const rnd = () => {
    seed = (seed + 0x6d2b79f5) >>> 0
    let t = seed
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296
  }
  // k-means++
  const centers: number[][] = [rows[Math.floor(rnd() * n)]!.slice()]
  while (centers.length < k) {
    const dist = rows.map((r) => Math.min(...centers.map((c) => r.reduce((s, x, j) => s + (x - c[j]!) ** 2, 0))))
    const sum = dist.reduce((a, b) => a + b, 0) || 1
    let u = rnd() * sum
    let idx = n - 1
    for (let i = 0; i < n; i++) {
      u -= dist[i]!
      if (u <= 0) {
        idx = i
        break
      }
    }
    centers.push(rows[idx]!.slice())
  }
  const cluster = new Array(n).fill(0)
  let inertia = Infinity
  let iterations = 0
  const maxIter = options.maxIter ?? 100
  for (; iterations < maxIter; iterations++) {
    // assign
    for (let i = 0; i < n; i++) {
      let best = 0
      let bestD = Infinity
      for (let c = 0; c < k; c++) {
        let d = 0
        for (let j = 0; j < p; j++) d += (rows[i]![j]! - centers[c]![j]!) ** 2
        if (d < bestD) {
          bestD = d
          best = c
        }
      }
      cluster[i] = best
    }
    // update
    const sums = Array.from({ length: k }, () => new Array(p).fill(0))
    const counts = new Array(k).fill(0)
    for (let i = 0; i < n; i++) {
      const c = cluster[i]!
      counts[c]++
      for (let j = 0; j < p; j++) sums[c]![j] += rows[i]![j]!
    }
    for (let c = 0; c < k; c++) {
      if (!counts[c]) continue
      for (let j = 0; j < p; j++) centers[c]![j] = sums[c]![j]! / counts[c]!
    }
    let ine = 0
    for (let i = 0; i < n; i++) {
      const c = cluster[i]!
      for (let j = 0; j < p; j++) ine += (rows[i]![j]! - centers[c]![j]!) ** 2
    }
    if (Math.abs(ine - inertia) < 1e-12) {
      inertia = ine
      break
    }
    inertia = ine
  }
  const sizes = new Array(k).fill(0)
  for (const c of cluster) sizes[c]++
  return { k, centers, cluster, inertia, sizes, iterations: iterations + 1 }
}

export interface HclustResult {
  /** Merge steps: [i, j, height] with negative indices for leaves (−1−leaf). */
  merge: Array<[number, number, number]>
  order: number[]
  method: 'single' | 'complete' | 'average' | 'ward'
}

/**
 * Agglomerative hierarchical clustering by the nearest-neighbour-chain algorithm with Lance–Williams
 * updates on a full Euclidean distance matrix — O(n²) time and memory (n ≤ 8000). Heights follow scipy's
 * `linkage` conventions (Ward: √ of the Lance–Williams update on squared distances).
 */
export function hclust(data: ArrayLike<ArrayLike<number>>, options: { method?: 'single' | 'complete' | 'average' | 'ward' } = {}): HclustResult {
  const rows = Array.from(data).map((r) => Array.from(r))
  const n = rows.length
  const method = options.method ?? 'average'
  if (n < 2) throw new RangeError('hclust needs at least 2 observations')
  if (n > 8000) throw new RangeError('hclust: at most 8000 observations (O(n²) distance matrix)')
  const p = rows[0]!.length
  for (let i = 0; i < n; i++) {
    const r = rows[i]!
    if (r.length !== p) throw new RangeError(`hclust: row ${i} has ${r.length} values, expected ${p}`)
    for (let k = 0; k < p; k++) if (!Number.isFinite(r[k]!)) throw new RangeError(`hclust: non-finite value at row ${i}, column ${k}`)
  }
  // distance matrix (squared for Ward, Euclidean otherwise)
  const D = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    const ri = rows[i]!
    for (let j = i + 1; j < n; j++) {
      const rj = rows[j]!
      let s2 = 0
      for (let k = 0; k < p; k++) s2 += (ri[k]! - rj[k]!) ** 2
      const d = method === 'ward' ? s2 : Math.sqrt(s2)
      if (!Number.isFinite(d)) throw new RangeError(`hclust: distance between rows ${i} and ${j} overflows (values too large)`)
      D[i * n + j] = d
      D[j * n + i] = d
    }
  }
  const size = new Float64Array(n).fill(1)
  const alive = new Uint8Array(n).fill(1)
  // raw merges as (representative leaf of i, representative leaf of j, height); sorted afterwards
  const rawMerges: Array<[number, number, number]> = []
  const chain: number[] = []
  let remaining = n
  while (remaining > 1) {
    if (chain.length === 0) {
      for (let i = 0; i < n; i++) if (alive[i]) {
        chain.push(i)
        break
      }
    }
    for (;;) {
      const a2 = chain[chain.length - 1]!
      // nearest alive neighbour of a2 (prefer the previous chain element on ties)
      let best = -1
      let bestD = Infinity
      const prev = chain.length >= 2 ? chain[chain.length - 2]! : -1
      for (let j = 0; j < n; j++) {
        if (!alive[j] || j === a2) continue
        const d = D[a2 * n + j]!
        if (d < bestD || (d === bestD && j === prev)) {
          bestD = d
          best = j
        }
      }
      if (best === prev) {
        // reciprocal nearest neighbours: merge a2 and prev
        chain.pop()
        chain.pop()
        const i = Math.min(a2, prev)
        const j = Math.max(a2, prev)
        const dij = D[i * n + j]!
        rawMerges.push([i, j, method === 'ward' ? Math.sqrt(dij) : dij])
        const ni = size[i]!
        const nj = size[j]!
        for (let k = 0; k < n; k++) {
          if (!alive[k] || k === i || k === j) continue
          const dik = D[i * n + k]!
          const djk = D[j * n + k]!
          let d: number
          if (method === 'single') d = Math.min(dik, djk)
          else if (method === 'complete') d = Math.max(dik, djk)
          else if (method === 'average') d = (ni * dik + nj * djk) / (ni + nj)
          else {
            const nk = size[k]!
            d = ((ni + nk) * dik + (nj + nk) * djk - nk * dij) / (ni + nj + nk)
          }
          D[i * n + k] = d
          D[k * n + i] = d
        }
        size[i] = ni + nj
        alive[j] = 0
        remaining--
        break
      }
      chain.push(best)
    }
  }
  // scipy convention: merges ordered by height, cluster ids assigned in that order (union–find relabel)
  rawMerges.sort((x, y) => x[2] - y[2])
  const parent = Array.from({ length: n }, (_, i) => i)
  const find = (x: number): number => (parent[x] === x ? x : (parent[x] = find(parent[x]!)))
  const clusterId = Array.from({ length: n }, (_, i) => -1 - i)
  const merge: Array<[number, number, number]> = []
  rawMerges.forEach(([a2, b2, h], k) => {
    const ra = find(a2)
    const rb = find(b2)
    merge.push([clusterId[ra]!, clusterId[rb]!, h])
    parent[rb] = ra
    clusterId[ra] = k
  })
  // leaf order from the merge tree
  const leaves = new Map<number, number[]>()
  for (let i = 0; i < n; i++) leaves.set(-1 - i, [i])
  merge.forEach(([x, y], k) => leaves.set(k, [...leaves.get(x)!, ...leaves.get(y)!]))
  const order = leaves.get(merge.length - 1) ?? [0]
  return { merge, order, method }
}

// ---- Discriminant ---------------------------------------------------------------------------------

export interface DiscriminantResult {
  method: 'lda' | 'qda'
  classes: string[]
  priors: number[]
  /** LDA only: linear coefficients per class (including intercept at [0]). */
  coefficients?: number[][]
  predictions: string[]
  posterior: number[][]
  accuracy: number
}

function classStats(X: number[][], y: string[]) {
  const classes = [...new Set(y)].sort()
  const stats = classes.map((c) => {
    const rows = X.filter((_, i) => y[i] === c)
    const n = rows.length
    const p = rows[0]!.length
    const mean = new Array(p).fill(0)
    for (const r of rows) for (let j = 0; j < p; j++) mean[j] += r[j]!
    for (let j = 0; j < p; j++) mean[j] /= n
    const cov = matrix(p, p)
    for (const r of rows) for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) cov.data[a * p + b]! += (r[a]! - mean[a]!) * (r[b]! - mean[b]!)
    for (let i = 0; i < p * p; i++) cov.data[i]! /= Math.max(1, n - 1)
    return { c, n, mean, cov, rows }
  })
  return { classes, stats }
}

function invert2(cov: Matrix): Matrix {
  const { u, s, v } = svd(cov)
  const p = cov.rows
  const Sinv = matrix(p, p)
  for (let i = 0; i < p; i++) if (s[i]! > 1e-10) Sinv.data[i * p + i] = 1 / s[i]!
  return matmul(matmul(v, Sinv), transpose(u))
}

function logDet(cov: Matrix): number {
  const { s } = svd(cov)
  let ld = 0
  for (let i = 0; i < s.length; i++) ld += Math.log(Math.max(s[i]!, 1e-15))
  return ld
}

/** Linear / Quadratic Discriminant Analysis. */
export function discriminant(
  X: ArrayLike<ArrayLike<number>>,
  y: ArrayLike<string | number>,
  options: { method?: 'lda' | 'qda'; priors?: 'equal' | 'empirical' } = {},
): DiscriminantResult {
  const rows = Array.from(X).map((r) => Array.from(r))
  const labels = Array.from(y).map(String)
  const method = options.method ?? 'lda'
  const { classes, stats } = classStats(rows, labels)
  const n = rows.length
  const p = rows[0]!.length
  const priors =
    options.priors === 'equal' ? classes.map(() => 1 / classes.length) : stats.map((s) => s.n / n)

  // pooled covariance for LDA
  const pooled = matrix(p, p)
  let df = 0
  for (const s of stats) {
    for (let i = 0; i < p * p; i++) pooled.data[i]! += s.cov.data[i]! * (s.n - 1)
    df += s.n - 1
  }
  for (let i = 0; i < p * p; i++) pooled.data[i]! /= Math.max(1, df)
  const poolInv = invert2(pooled)

  const posterior: number[][] = []
  const predictions: string[] = []
  const coefficients: number[][] = []

  if (method === 'lda') {
    for (let c = 0; c < classes.length; c++) {
      const mean = stats[c]!.mean
      // δ_c(x) = x' Σ⁻¹ μ_c − ½ μ' Σ⁻¹ μ + log π
      const coef = new Array(p + 1).fill(0)
      let quad = 0
      for (let a = 0; a < p; a++) {
        let lin = 0
        for (let b = 0; b < p; b++) lin += poolInv.data[a * p + b]! * mean[b]!
        coef[a + 1] = lin
        quad += mean[a]! * lin
      }
      coef[0] = -0.5 * quad + Math.log(priors[c]!)
      coefficients.push(coef)
    }
  }

  for (let i = 0; i < n; i++) {
    const x = rows[i]!
    const scores: number[] = []
    for (let c = 0; c < classes.length; c++) {
      if (method === 'lda') {
        let s = coefficients[c]![0]!
        for (let j = 0; j < p; j++) s += coefficients[c]![j + 1]! * x[j]!
        scores.push(s)
      } else {
        const mean = stats[c]!.mean
        const inv = invert2(stats[c]!.cov)
        const ld = logDet(stats[c]!.cov)
        let quad = 0
        for (let a = 0; a < p; a++) {
          let t = 0
          for (let b = 0; b < p; b++) t += inv.data[a * p + b]! * (x[b]! - mean[b]!)
          quad += (x[a]! - mean[a]!) * t
        }
        scores.push(-0.5 * ld - 0.5 * quad + Math.log(priors[c]!))
      }
    }
    const m = Math.max(...scores)
    const exps = scores.map((s) => Math.exp(s - m))
    const sum = exps.reduce((a, b) => a + b, 0)
    const post = exps.map((e) => e / sum)
    posterior.push(post)
    predictions.push(classes[post.indexOf(Math.max(...post))]!)
  }
  let correct = 0
  for (let i = 0; i < n; i++) if (predictions[i] === labels[i]) correct++
  return {
    method,
    classes,
    priors,
    coefficients: method === 'lda' ? coefficients : undefined,
    predictions,
    posterior,
    accuracy: correct / n,
  }
}

// ---- Correspondence analysis ----------------------------------------------------------------------

export interface CorrespondenceResult {
  rowCoords: number[][]
  colCoords: number[][]
  singularValues: number[]
  inertia: number[]
  rowNames: string[]
  colNames: string[]
}

/** Simple correspondence analysis of a contingency table. */
export function correspondence(
  table: ArrayLike<ArrayLike<number>>,
  options: { rowNames?: string[]; colNames?: string[]; nComponents?: number } = {},
): CorrespondenceResult {
  const T = Array.from(table).map((r) => Array.from(r))
  const n = T.length
  const m = T[0]!.length
  let total = 0
  for (const r of T) for (const v of r) total += v
  if (!(total > 0)) throw new RangeError('correspondence: empty table')
  const rowSum = T.map((r) => r.reduce((a, b) => a + b, 0))
  const colSum = new Array(m).fill(0)
  for (const r of T) for (let j = 0; j < m; j++) colSum[j] += r[j]!
  // standardized residuals matrix
  const S = matrix(n, m)
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
    const e = (rowSum[i]! * colSum[j]!) / total
    S.data[i * m + j] = e > 0 ? (T[i]![j]! - e) / Math.sqrt(e) : 0
  }
  const { u, s: sCount, v } = svd(S)
  // S was built from counts: singular values on the proportion (correspondence) scale are sCount / √total
  const s = Float64Array.from(sCount, (val) => val / Math.sqrt(total))
  const k = Math.min(options.nComponents ?? s.length, s.length)
  const inertia = Array.from({ length: k }, (_, j) => s[j]! * s[j]!)
  const rowCoords: number[][] = []
  for (let i = 0; i < n; i++) {
    const row: number[] = []
    for (let j = 0; j < k; j++) row.push((u.data[i * u.cols + j]! * s[j]!) / Math.sqrt(rowSum[i]! / total))
    rowCoords.push(row)
  }
  const colCoords: number[][] = []
  for (let i = 0; i < m; i++) {
    const row: number[] = []
    for (let j = 0; j < k; j++) row.push((v.data[i * v.cols + j]! * s[j]!) / Math.sqrt(colSum[i]! / total))
    colCoords.push(row)
  }
  return {
    rowCoords,
    colCoords,
    singularValues: Array.from(s).slice(0, k),
    inertia,
    rowNames: options.rowNames ?? Array.from({ length: n }, (_, i) => `R${i + 1}`),
    colNames: options.colNames ?? Array.from({ length: m }, (_, i) => `C${i + 1}`),
  }
}
