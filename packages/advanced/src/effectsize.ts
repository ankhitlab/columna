/**
 * Standardized effect sizes (Cohen's d, Hedges' g, Glass's Δ).
 */
import { cleanNumbers } from './tests.js'

export interface EffectSizeResult {
  method: "Cohen's d" | "Hedges' g" | "Glass's Δ"
  estimate: number
  n1: number
  n2: number
  pooledSd: number
}

function meansSd(v: Float64Array) {
  const n = v.length
  let s = 0
  for (let i = 0; i < n; i++) s += v[i]!
  const mean = s / n
  let m2 = 0
  for (let i = 0; i < n; i++) m2 += (v[i]! - mean) ** 2
  return { n, mean, sd: Math.sqrt(m2 / Math.max(1, n - 1)) }
}

/** Cohen's d (pooled SD). */
export function cohensD(
  a: ArrayLike<number | null | undefined>,
  b: ArrayLike<number | null | undefined>,
): EffectSizeResult {
  const x = cleanNumbers(a)
  const y = cleanNumbers(b)
  const mx = meansSd(x)
  const my = meansSd(y)
  const pooled = Math.sqrt(((mx.n - 1) * mx.sd ** 2 + (my.n - 1) * my.sd ** 2) / (mx.n + my.n - 2))
  const d = pooled > 0 ? (mx.mean - my.mean) / pooled : 0
  return { method: "Cohen's d", estimate: d, n1: mx.n, n2: my.n, pooledSd: pooled }
}

/** Hedges' g = bias-corrected Cohen's d. */
export function hedgesG(
  a: ArrayLike<number | null | undefined>,
  b: ArrayLike<number | null | undefined>,
): EffectSizeResult {
  const base = cohensD(a, b)
  const df = base.n1 + base.n2 - 2
  const J = df > 1 ? 1 - 3 / (4 * df - 1) : 1
  return { ...base, method: "Hedges' g", estimate: base.estimate * J }
}

/** Glass's Δ using control-group SD (second sample). */
export function glassDelta(
  a: ArrayLike<number | null | undefined>,
  control: ArrayLike<number | null | undefined>,
): EffectSizeResult {
  const x = cleanNumbers(a)
  const y = cleanNumbers(control)
  const mx = meansSd(x)
  const my = meansSd(y)
  const d = my.sd > 0 ? (mx.mean - my.mean) / my.sd : 0
  return { method: "Glass's Δ", estimate: d, n1: mx.n, n2: my.n, pooledSd: my.sd }
}
