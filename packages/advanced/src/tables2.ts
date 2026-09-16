/**
 * Contingency / paired categorical tests: McNemar, Cochran Q, Bowker.
 */
import { chi2 as chi2Dist } from './dist.js'

export interface PairedTableResult {
  test: 'McNemar' | 'Cochran Q' | 'Bowker'
  statistic: number
  df: number
  pValue: number
  n: number
}

/** McNemar test for 2×2 paired binary (scipy mcnemar). Exact binomial when b+c small. */
export function mcnemar(
  table: [[number, number], [number, number]],
  options: { exact?: boolean; correction?: boolean } = {},
): PairedTableResult {
  const b = table[0]![1]!
  const c = table[1]![0]!
  const n = b + c
  if (n < 1) throw new RangeError('mcnemar: need at least one discordant pair')
  const exact = options.exact ?? n < 25
  if (exact) {
    // two-sided binomial exact on min(b,c)
    let p = 0
    const k = Math.min(b, c)
    for (let i = 0; i <= k; i++) {
      // C(n,i) / 2^n
      let logC = 0
      for (let t = 0; t < i; t++) logC += Math.log(n - t) - Math.log(t + 1)
      p += Math.exp(logC - n * Math.LN2)
    }
    p = Math.min(1, 2 * p)
    return { test: 'McNemar', statistic: Math.abs(b - c), df: 1, pValue: p, n }
  }
  const corr = options.correction !== false
  const stat = corr ? (Math.abs(b - c) - 1) ** 2 / n : (b - c) ** 2 / n
  return { test: 'McNemar', statistic: Math.max(0, stat), df: 1, pValue: chi2Dist(1).sf(Math.max(0, stat)), n }
}

/** Cochran's Q for k related binary samples (subjects × treatments). */
export function cochranQ(data: ArrayLike<ArrayLike<number>>): PairedTableResult {
  // rows = subjects, cols = treatments (0/1)
  const rows = Array.from(data).map((r) => Array.from(r).map((v) => (Number(v) ? 1 : 0)))
  const n = rows.length
  if (n < 2) throw new RangeError('cochranQ: need ≥2 subjects')
  const k = rows[0]!.length
  if (k < 2) throw new RangeError('cochranQ: need ≥2 treatments')
  const colSum = new Array(k).fill(0)
  let T = 0
  const rowSum: number[] = []
  for (let i = 0; i < n; i++) {
    if (rows[i]!.length !== k) throw new RangeError('cochranQ: ragged rows')
    let rs = 0
    for (let j = 0; j < k; j++) {
      colSum[j]! += rows[i]![j]!
      rs += rows[i]![j]!
      T += rows[i]![j]!
    }
    rowSum.push(rs)
  }
  let num = 0
  for (let j = 0; j < k; j++) num += (k * colSum[j]! - T) ** 2
  let den = 0
  for (let i = 0; i < n; i++) den += rowSum[i]! * (k - rowSum[i]!)
  const Q = den > 0 ? ((k - 1) * num) / (k * den) : 0
  const df = k - 1
  return { test: 'Cochran Q', statistic: Q, df, pValue: chi2Dist(df).sf(Q), n }
}

/** Bowker's test of symmetry for square k×k table (generalized McNemar). */
export function bowker(table: number[][]): PairedTableResult {
  const k = table.length
  if (k < 2) throw new RangeError('bowker: need ≥2×2 table')
  for (const row of table) if (row.length !== k) throw new RangeError('bowker: table must be square')
  let X2 = 0
  let df = 0
  let n = 0
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) n += table[i]![j]!
    for (let j = i + 1; j < k; j++) {
      const a = table[i]![j]!
      const b = table[j]![i]!
      if (a + b > 0) {
        X2 += (a - b) ** 2 / (a + b)
        df++
      }
    }
  }
  return { test: 'Bowker', statistic: X2, df: Math.max(1, df), pValue: chi2Dist(Math.max(1, df)).sf(X2), n }
}
