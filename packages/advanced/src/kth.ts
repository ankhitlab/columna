/**
 * Order statistics of implicit pairwise sets without materialising them: the k-th smallest of
 * {aᵢ + bⱼ} (Hodges–Lehmann differences with b = −y) and of the Walsh averages {(xᵢ + xⱼ)/2, i ≤ j},
 * by value-space bisection with O(n) counting — O(n log n · precision) instead of O(n₁n₂ log).
 */

/** Number of pairs (i, j) with a[i] + b[j] ≤ v (a, b ascending). */
function countSums(a: Float64Array, b: Float64Array, v: number): number {
  let j = b.length - 1
  let c = 0
  for (let i = 0; i < a.length; i++) {
    while (j >= 0 && a[i]! + b[j]! > v) j--
    if (j < 0) break
    c += j + 1
  }
  return c
}

/** Smallest a[i] + b[j] strictly greater than v. */
function minSumAbove(a: Float64Array, b: Float64Array, v: number): number {
  let best = Infinity
  let j = b.length - 1
  for (let i = 0; i < a.length; i++) {
    // largest j with a[i] + b[j] ≤ v; candidate is j + 1
    while (j >= 0 && a[i]! + b[j]! > v) j--
    const cand = j + 1
    if (cand < b.length) best = Math.min(best, a[i]! + b[cand]!)
  }
  return best
}

/** k-th smallest (1-indexed) of {a[i] + b[j]} for ascending a and b. */
export function kthPairSum(a: Float64Array, b: Float64Array, k: number): number {
  const m = a.length * b.length
  if (!(k >= 1 && k <= m)) throw new RangeError(`kthPairSum: k must be in [1, ${m}]`)
  let lo = a[0]! + b[0]!
  let hi = a[a.length - 1]! + b[b.length - 1]!
  if (countSums(a, b, lo) >= k) return lo
  for (let it = 0; it < 200 && hi - lo > 1e-9 * (Math.abs(lo) + Math.abs(hi) + 1); it++) {
    const mid = 0.5 * (lo + hi)
    if (countSums(a, b, mid) >= k) hi = mid
    else lo = mid
  }
  // count(lo) < k ≤ count(hi): the answer is the smallest sum above lo
  return minSumAbove(a, b, lo)
}

/** Number of Walsh pairs i ≤ j with (x[i] + x[j]) ≤ s for ascending x. */
function countWalsh(x: Float64Array, s: number): number {
  let j = x.length - 1
  let c = 0
  for (let i = 0; i < x.length; i++) {
    while (j >= i && x[i]! + x[j]! > s) j--
    if (j < i) break
    c += j - i + 1
  }
  return c
}

function minWalshAbove(x: Float64Array, s: number): number {
  let best = Infinity
  let j = x.length - 1
  for (let i = 0; i < x.length; i++) {
    while (j >= i && x[i]! + x[j]! > s) j--
    const cand = Math.max(j + 1, i)
    if (cand < x.length && x[i]! + x[cand]! > s) best = Math.min(best, x[i]! + x[cand]!)
  }
  return best
}

/** k-th smallest (1-indexed) Walsh average (x[i] + x[j]) / 2 over i ≤ j, x ascending. */
export function kthWalshAverage(x: Float64Array, k: number): number {
  const n = x.length
  const m = (n * (n + 1)) / 2
  if (!(k >= 1 && k <= m)) throw new RangeError(`kthWalshAverage: k must be in [1, ${m}]`)
  let lo = 2 * x[0]!
  let hi = 2 * x[n - 1]!
  if (countWalsh(x, lo) >= k) return lo / 2
  for (let it = 0; it < 200 && hi - lo > 1e-9 * (Math.abs(lo) + Math.abs(hi) + 1); it++) {
    const mid = 0.5 * (lo + hi)
    if (countWalsh(x, mid) >= k) hi = mid
    else lo = mid
  }
  return minWalshAbove(x, lo) / 2
}
