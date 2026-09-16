/**
 * Multi-sample goodness-of-fit: KS two-sample, Anderson–Darling k-sample, energy distance.
 */
import { chi2 as chi2Dist, normal } from './dist.js'
import { cleanNumbers } from './tests.js'

const STD = normal()

export interface KsTwoSampleResult {
  test: 'KS two-sample'
  statistic: number
  pValue: number
  n1: number
  n2: number
}

export interface AndersonKSampleResult {
  test: 'Anderson-Darling k-sample'
  statistic: number
  pValue: number
  k: number
  n: number
}

export interface EnergyDistanceResult {
  test: 'energy distance'
  statistic: number
  pValue: number
  n1: number
  n2: number
}

/** Two-sample Kolmogorov–Smirnov (scipy ks_2samp asymptotic). */
export function ksTwoSample(
  a: ArrayLike<number | null | undefined>,
  b: ArrayLike<number | null | undefined>,
): KsTwoSampleResult {
  const x = Array.from(cleanNumbers(a)).sort((u, v) => u - v)
  const y = Array.from(cleanNumbers(b)).sort((u, v) => u - v)
  const n1 = x.length
  const n2 = y.length
  if (n1 < 1 || n2 < 1) throw new RangeError('ksTwoSample: empty sample')
  let i = 0
  let j = 0
  let d = 0
  while (i < n1 || j < n2) {
    const vx = i < n1 ? x[i]! : Infinity
    const vy = j < n2 ? y[j]! : Infinity
    if (vx <= vy) i++
    if (vy <= vx) j++
    d = Math.max(d, Math.abs(i / n1 - j / n2))
  }
  const en = Math.sqrt((n1 * n2) / (n1 + n2))
  // Marsaglia / asymptotic: Q_KS
  const z = (en + 0.12 + 0.11 / en) * d
  let p = 0
  for (let k = 1; k < 100; k++) {
    const term = (k % 2 ? 1 : -1) * Math.exp(-2 * k * k * z * z)
    p += term
    if (Math.abs(term) < 1e-12) break
  }
  p = Math.max(0, Math.min(1, 2 * p))
  return { test: 'KS two-sample', statistic: d, pValue: p, n1, n2 }
}

/** Anderson–Darling k-sample (Scholz–Stephens; scipy anderson_ksamp lite). */
export function andersonKSample(
  samples: Array<ArrayLike<number | null | undefined>>,
): AndersonKSampleResult {
  const gs = samples.map((s) => Array.from(cleanNumbers(s)).sort((a, b) => a - b))
  const k = gs.length
  if (k < 2) throw new RangeError('andersonKSample: need ≥2 samples')
  const n = gs.reduce((s, g) => s + g.length, 0)
  const all = gs.flat().sort((a, b) => a - b)
  // midranks
  const ranks = new Map<number, number>()
  for (let i = 0; i < n; ) {
    let j = i
    while (j + 1 < n && all[j + 1] === all[i]) j++
    const r = (i + 1 + j + 1) / 2
    for (let t = i; t <= j; t++) ranks.set(all[t]!, r)
    i = j + 1
  }
  let A2 = 0
  for (const g of gs) {
    const ni = g.length
    let sum = 0
    for (let j = 0; j < ni; j++) {
      const M = ranks.get(g[j]!)!
      sum += (2 * (j + 1) - 1) * Math.log(M / n) + (2 * (ni - j) - 1) * Math.log(1 - (M - 1) / n)
    }
    A2 += -ni - sum / ni
  }
  A2 *= (n - 1) / n // rough scale
  const df = k - 1
  const pValue = chi2Dist(df).sf(Math.max(0, A2))
  return { test: 'Anderson-Darling k-sample', statistic: A2, pValue, k, n }
}

/** Energy distance two-sample (scipy energy_distance; permutation p optional via asymptotic χ² lite). */
export function energyDistance(
  a: ArrayLike<number | null | undefined>,
  b: ArrayLike<number | null | undefined>,
): EnergyDistanceResult {
  const x = Array.from(cleanNumbers(a))
  const y = Array.from(cleanNumbers(b))
  const n1 = x.length
  const n2 = y.length
  const meanAbs = (u: number[], v: number[]) => {
    let s = 0
    for (const a0 of u) for (const b0 of v) s += Math.abs(a0 - b0)
    return s / (u.length * v.length)
  }
  const A = meanAbs(x, y)
  const B = meanAbs(x, x)
  const C = meanAbs(y, y)
  const statistic = 2 * A - B - C
  // rough p via normal on sqrt(n) scale
  const se = Math.sqrt((B + C) / Math.max(1, n1 + n2))
  const z = statistic / Math.max(1e-12, se)
  const pValue = Math.min(1, 2 * STD.sf(Math.abs(z)))
  return { test: 'energy distance', statistic, pValue, n1, n2 }
}
