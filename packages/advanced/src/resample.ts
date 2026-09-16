/**
 * Resampling: percentile bootstrap CI and two-sample permutation tests.
 */
import { cleanNumbers } from './tests.js'

export interface BootstrapResult {
  estimate: number
  ci: [number, number]
  confidence: number
  nBoot: number
  replicates: number[]
}

function mulberry(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296
  }
}

/** Percentile bootstrap for a scalar statistic of one sample. */
export function bootstrap(
  data: ArrayLike<number | null | undefined>,
  statistic: (sample: Float64Array) => number,
  options: { nBoot?: number; confidence?: number; seed?: number } = {},
): BootstrapResult {
  const v = cleanNumbers(data)
  const n = v.length
  if (n < 2) throw new RangeError('bootstrap needs ≥2 observations')
  const nBoot = options.nBoot ?? 999
  const confidence = options.confidence ?? 0.95
  const u = mulberry(options.seed ?? 1)
  const estimate = statistic(v)
  const reps: number[] = []
  const samp = new Float64Array(n)
  for (let b = 0; b < nBoot; b++) {
    for (let i = 0; i < n; i++) samp[i] = v[Math.floor(u() * n)]!
    reps.push(statistic(samp))
  }
  reps.sort((a, b) => a - b)
  const alpha = 1 - confidence
  const lo = reps[Math.floor((alpha / 2) * nBoot)]!
  const hi = reps[Math.min(nBoot - 1, Math.ceil((1 - alpha / 2) * nBoot) - 1)]!
  return { estimate, ci: [lo, hi], confidence, nBoot, replicates: reps }
}

export interface PermutationTestResult {
  test: 'permutation'
  statistic: number
  pValue: number
  nPerm: number
  alternative: 'two-sided' | 'less' | 'greater'
}

/** Two-sample permutation test for difference of means or medians. */
export function permutationTest(
  a: ArrayLike<number | null | undefined>,
  b: ArrayLike<number | null | undefined>,
  options: {
    statistic?: 'mean' | 'median'
    nPerm?: number
    alternative?: 'two-sided' | 'less' | 'greater'
    seed?: number
  } = {},
): PermutationTestResult {
  const x = Array.from(cleanNumbers(a))
  const y = Array.from(cleanNumbers(b))
  const n1 = x.length
  const n2 = y.length
  const nPerm = options.nPerm ?? 999
  const alternative = options.alternative ?? 'two-sided'
  const kind = options.statistic ?? 'mean'
  const u = mulberry(options.seed ?? 2)
  const agg = (arr: number[]) => {
    if (kind === 'median') {
      const s = arr.slice().sort((p, q) => p - q)
      const m = s.length
      return m % 2 ? s[(m - 1) / 2]! : 0.5 * (s[m / 2 - 1]! + s[m / 2]!)
    }
    return arr.reduce((s, v) => s + v, 0) / arr.length
  }
  const obs = agg(x) - agg(y)
  const pool = x.concat(y)
  let extreme = 0
  for (let p = 0; p < nPerm; p++) {
    // Fisher–Yates shuffle then split
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(u() * (i + 1))
      ;[pool[i], pool[j]] = [pool[j]!, pool[i]!]
    }
    const d = agg(pool.slice(0, n1)) - agg(pool.slice(n1))
    if (alternative === 'greater' && d >= obs) extreme++
    else if (alternative === 'less' && d <= obs) extreme++
    else if (alternative === 'two-sided' && Math.abs(d) >= Math.abs(obs)) extreme++
  }
  return {
    test: 'permutation',
    statistic: obs,
    pValue: (extreme + 1) / (nPerm + 1),
    nPerm,
    alternative,
  }
}
