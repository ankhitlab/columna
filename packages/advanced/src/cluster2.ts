/**
 * Distance utilities and DBSCAN clustering.
 */

export function pdist(X: ArrayLike<ArrayLike<number>>, metric: 'euclidean' | 'cityblock' = 'euclidean'): Float64Array {
  const rows = Array.from(X).map((r) => Array.from(r))
  const n = rows.length
  const out = new Float64Array((n * (n - 1)) / 2)
  let k = 0
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      out[k++] = dist(rows[i]!, rows[j]!, metric)
    }
  }
  return out
}

export function cdist(
  XA: ArrayLike<ArrayLike<number>>,
  XB: ArrayLike<ArrayLike<number>>,
  metric: 'euclidean' | 'cityblock' = 'euclidean',
): number[][] {
  const A = Array.from(XA).map((r) => Array.from(r))
  const B = Array.from(XB).map((r) => Array.from(r))
  return A.map((a) => B.map((b) => dist(a, b, metric)))
}

function dist(a: number[], b: number[], metric: 'euclidean' | 'cityblock') {
  if (metric === 'cityblock') {
    let s = 0
    for (let i = 0; i < a.length; i++) s += Math.abs(a[i]! - b[i]!)
    return s
  }
  let s = 0
  for (let i = 0; i < a.length; i++) {
    const d = a[i]! - b[i]!
    s += d * d
  }
  return Math.sqrt(s)
}

export interface DbscanResult {
  labels: number[]
  nClusters: number
  coreSampleIndices: number[]
  eps: number
  minSamples: number
}

/** DBSCAN (scipy-style labels: -1 = noise). */
export function dbscan(
  X: ArrayLike<ArrayLike<number>>,
  options: { eps?: number; minSamples?: number } = {},
): DbscanResult {
  const rows = Array.from(X).map((r) => Array.from(r))
  const n = rows.length
  const eps = options.eps ?? 0.5
  const minSamples = options.minSamples ?? 5
  const labels = new Array(n).fill(-1)
  const visited = new Array(n).fill(false)
  const core: number[] = []
  const neighbors = (i: number) => {
    const out: number[] = []
    for (let j = 0; j < n; j++) if (dist(rows[i]!, rows[j]!, 'euclidean') <= eps) out.push(j)
    return out
  }
  let cluster = 0
  for (let i = 0; i < n; i++) {
    if (visited[i]) continue
    visited[i] = true
    const neigh = neighbors(i)
    if (neigh.length < minSamples) continue
    core.push(i)
    labels[i] = cluster
    const seeds = neigh.filter((j) => j !== i)
    for (let s = 0; s < seeds.length; s++) {
      const q = seeds[s]!
      if (!visited[q]) {
        visited[q] = true
        const nq = neighbors(q)
        if (nq.length >= minSamples) {
          core.push(q)
          for (const r of nq) if (!seeds.includes(r)) seeds.push(r)
        }
      }
      if (labels[q] === -1) labels[q] = cluster
    }
    cluster++
  }
  return { labels, nClusters: cluster, coreSampleIndices: [...new Set(core)], eps, minSamples }
}
