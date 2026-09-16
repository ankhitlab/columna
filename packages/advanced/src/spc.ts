/**
 * Tier 4.1–4.4 — Quality Tools › Control Charts (SPC):
 * chart constants (ASTM/Minitab), I-MR / X̄-R / X̄-S / Z-MR, attribute P/NP/C/U and Laney P′/U′,
 * EWMA, tabular CUSUM (and V-mask), moving average. Nelson rules 1–8 on Shewhart charts.
 */
import { lgamma } from './dist.js'
import { cleanNumbers } from './tests.js'

// ---- helpers --------------------------------------------------------------------------------------

function meanOf(v: ArrayLike<number>): number {
  let s = 0
  const n = v.length
  for (let i = 0; i < n; i++) s += v[i]!
  return s / n
}
function sdOf(v: ArrayLike<number>, ddof = 1): number {
  const n = v.length
  if (n <= ddof) return NaN
  const m = meanOf(v)
  let s2 = 0
  for (let i = 0; i < n; i++) s2 += (v[i]! - m) ** 2
  return Math.sqrt(s2 / (n - ddof))
}
function rangeOf(v: ArrayLike<number>): number {
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < v.length; i++) {
    const x = v[i]!
    if (x < lo) lo = x
    if (x > hi) hi = x
  }
  return hi - lo
}

// ---- 4.1 constants --------------------------------------------------------------------------------

/** ASTM / Minitab d₂, d₃ for subgroup size 2…25 (expected range / σ and √Var(R/σ)). */
const D2 = [
  NaN, NaN, 1.128, 1.693, 2.059, 2.326, 2.534, 2.704, 2.847, 2.97, 3.078, 3.173, 3.258, 3.336, 3.407, 3.472, 3.532, 3.588, 3.64, 3.689, 3.735, 3.778, 3.819, 3.858, 3.895, 3.931,
]
const D3 = [
  NaN, NaN, 0.8525, 0.8884, 0.8798, 0.8641, 0.848, 0.8332, 0.8198, 0.8078, 0.7971, 0.7873, 0.7785, 0.7704, 0.763, 0.7562, 0.7499, 0.7441, 0.7386, 0.7335, 0.7287, 0.7242, 0.7199, 0.7159, 0.7121, 0.7084,
]

export interface SpcConstants {
  n: number
  /** E[R]/σ */
  d2: number
  /** √Var(R/σ) */
  d3: number
  /** E[s]/σ */
  c4: number
  /** X̄ chart with R: 3/(d₂√n) */
  A2: number
  /** X̄ chart with S: 3/(c₄√n) */
  A3: number
  /** S chart lower: max(0, 1 − 3√(1−c₄²)/c₄) */
  B3: number
  /** S chart upper */
  B4: number
  /** R chart lower: max(0, 1 − 3 d₃/d₂) */
  D3: number
  /** R chart upper */
  D4: number
}

function c4Of(n: number): number {
  if (!(n >= 2) || !Number.isFinite(n)) throw new RangeError(`spcConstants: n must be ≥ 2 (got ${n})`)
  // c₄ = √(2/(n−1)) · Γ(n/2) / Γ((n−1)/2)
  return Math.sqrt(2 / (n - 1)) * Math.exp(lgamma(n / 2) - lgamma((n - 1) / 2))
}

function d2Approx(n: number): number {
  // Hartley's approximation for n > 25: accurate to ~0.001
  const a = 0.79788456 // √(2/π)
  return a * (Math.log(n) ** 0.5 * (1 + 0.25 / Math.log(n) - 0.125 / Math.log(n) ** 2) + 0.5 / Math.sqrt(Math.log(n)))
}

function d3Approx(n: number): number {
  // asymptotic: Var(R/σ) ≈ (π²/6 − (d₂*)²) with slow decay; use Montgomery approx
  return Math.sqrt(0.8264 - 0.08255 * Math.log(n) + 0.00426 * Math.log(n) ** 2)
}

/** Control-chart constants for subgroup size `n` (ASTM E2587 / Minitab). */
export function spcConstants(n: number): SpcConstants {
  const nn = Math.floor(n)
  if (!(nn >= 2)) throw new RangeError(`spcConstants: n must be ≥ 2 (got ${n})`)
  const c4 = c4Of(nn)
  const d2 = nn <= 25 ? D2[nn]! : d2Approx(nn)
  const d3 = nn <= 25 ? D3[nn]! : d3Approx(nn)
  const A2 = 3 / (d2 * Math.sqrt(nn))
  const A3 = 3 / (c4 * Math.sqrt(nn))
  const seC4 = Math.sqrt(1 - c4 * c4) / c4
  const B3 = Math.max(0, 1 - 3 * seC4)
  const B4 = 1 + 3 * seC4
  const D3lim = Math.max(0, 1 - 3 * (d3 / d2))
  const D4lim = 1 + 3 * (d3 / d2)
  return { n: nn, d2, d3, c4, A2, A3, B3, B4, D3: D3lim, D4: D4lim }
}

// ---- Nelson / Western Electric rules --------------------------------------------------------------

export type NelsonRule = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8

export interface ChartPoint {
  index: number
  value: number
  /** Subgroup size (attribute / variable charts with variable n). */
  n?: number
  ucl: number
  lcl: number
  center: number
  beyondLimits: boolean
  /** Nelson rule numbers that fire on this point. */
  rules: NelsonRule[]
}

export interface ControlChartResult {
  type: string
  center: number
  /** Process sigma estimate used for limits (individuals / Z-MR) or average subgroup sigma. */
  sigma: number
  ucl: number
  lcl: number
  points: ChartPoint[]
  /** Indices of points outside control limits (rule 1). */
  outOfControl: number[]
  /** All fired (index, rules) pairs. */
  signals: Array<{ index: number; rules: NelsonRule[] }>
  /** Companion MR / R / S chart when applicable. */
  companion?: ControlChartResult
}

function zone(value: number, center: number, sigma: number): number {
  if (!(sigma > 0)) return 0
  return (value - center) / sigma
}

/**
 * Nelson rules 1–8 on a Shewhart series. `sigma` is the one-sigma distance from the center line
 * to the control limit / 3 (so limits are center ± 3σ).
 */
export function nelsonRules(
  values: ArrayLike<number>,
  center: number,
  sigma: number,
  enabled: NelsonRule[] = [1, 2, 3, 4, 5, 6, 7, 8],
): NelsonRule[][] {
  const n = values.length
  const out: NelsonRule[][] = Array.from({ length: n }, () => [])
  const on = new Set(enabled)
  const z = Array.from({ length: n }, (_, i) => zone(values[i]!, center, sigma))
  const side = (i: number) => (z[i]! > 0 ? 1 : z[i]! < 0 ? -1 : 0)

  if (on.has(1)) {
    for (let i = 0; i < n; i++) if (Math.abs(z[i]!) > 3) out[i]!.push(1)
  }
  if (on.has(2)) {
    for (let i = 8; i < n; i++) {
      const s = side(i)
      if (s === 0) continue
      let ok = true
      for (let k = 1; k < 9; k++) if (side(i - k) !== s) {
        ok = false
        break
      }
      if (ok) out[i]!.push(2)
    }
  }
  if (on.has(3)) {
    for (let i = 5; i < n; i++) {
      let up = true
      let down = true
      for (let k = 0; k < 5; k++) {
        const d = values[i - k]! - values[i - k - 1]!
        if (!(d > 0)) up = false
        if (!(d < 0)) down = false
      }
      if (up || down) out[i]!.push(3)
    }
  }
  if (on.has(4)) {
    for (let i = 13; i < n; i++) {
      let alt = true
      for (let k = 0; k < 13; k++) {
        const d1 = values[i - k]! - values[i - k - 1]!
        const d0 = values[i - k - 1]! - values[i - k - 2]!
        if (d1 === 0 || d0 === 0 || Math.sign(d1) === Math.sign(d0)) {
          alt = false
          break
        }
      }
      if (alt) out[i]!.push(4)
    }
  }
  if (on.has(5)) {
    for (let i = 2; i < n; i++) {
      for (const s of [1, -1] as const) {
        let c = 0
        for (let k = 0; k < 3; k++) if (s * z[i - k]! > 2) c++
        if (c >= 2) {
          out[i]!.push(5)
          break
        }
      }
    }
  }
  if (on.has(6)) {
    for (let i = 4; i < n; i++) {
      for (const s of [1, -1] as const) {
        let c = 0
        for (let k = 0; k < 5; k++) if (s * z[i - k]! > 1) c++
        if (c >= 4) {
          out[i]!.push(6)
          break
        }
      }
    }
  }
  if (on.has(7)) {
    for (let i = 14; i < n; i++) {
      let ok = true
      for (let k = 0; k < 15; k++) if (Math.abs(z[i - k]!) > 1) {
        ok = false
        break
      }
      if (ok) out[i]!.push(7)
    }
  }
  if (on.has(8)) {
    for (let i = 7; i < n; i++) {
      let ok = true
      for (let k = 0; k < 8; k++) if (Math.abs(z[i - k]!) < 1) {
        ok = false
        break
      }
      if (ok) out[i]!.push(8)
    }
  }
  return out
}

function packPoints(
  values: number[],
  center: number | number[],
  ucl: number | number[],
  lcl: number | number[],
  sigma: number | number[],
  rules: NelsonRule[],
  ns?: number[],
): { points: ChartPoint[]; outOfControl: number[]; signals: Array<{ index: number; rules: NelsonRule[] }> } {
  const n = values.length
  const centers = typeof center === 'number' ? null : center
  const ucls = typeof ucl === 'number' ? null : ucl
  const lcls = typeof lcl === 'number' ? null : lcl
  const sigmas = typeof sigma === 'number' ? null : sigma
  const c0 = typeof center === 'number' ? center : 0
  const u0 = typeof ucl === 'number' ? ucl : 0
  const l0 = typeof lcl === 'number' ? lcl : 0
  const s0 = typeof sigma === 'number' ? sigma : 1
  const fired =
    centers || sigmas
      ? values.map((v, i) => {
          const ui = ucls ? ucls[i]! : u0
          const li = lcls ? lcls[i]! : l0
          const local: NelsonRule[] = []
          if (rules.includes(1) && (v > ui || v < li)) local.push(1)
          return local
        })
      : nelsonRules(values, c0, s0, rules)

  if ((centers || sigmas) && rules.some((r) => r !== 1)) {
    // Standardize to apply rules 2–8 on a common scale
    const std = values.map((v, i) => {
      const ci = centers ? centers[i]! : c0
      const si = sigmas ? sigmas[i]! : s0
      return si > 0 ? (v - ci) / si : 0
    })
    const extra = nelsonRules(std, 0, 1, rules.filter((r) => r !== 1) as NelsonRule[])
    for (let i = 0; i < n; i++) {
      const set = new Set<NelsonRule>([...fired[i]!, ...extra[i]!])
      fired[i] = [...set].sort((a, b) => a - b) as NelsonRule[]
    }
  }

  const points: ChartPoint[] = []
  const outOfControl: number[] = []
  const signals: Array<{ index: number; rules: NelsonRule[] }> = []
  for (let i = 0; i < n; i++) {
    const ci = centers ? centers[i]! : c0
    const ui = ucls ? ucls[i]! : u0
    const li = lcls ? lcls[i]! : l0
    const beyond = values[i]! > ui || values[i]! < li
    const r = fired[i]!
    points.push({ index: i, value: values[i]!, n: ns?.[i], ucl: ui, lcl: li, center: ci, beyondLimits: beyond, rules: r })
    if (beyond) outOfControl.push(i)
    if (r.length) signals.push({ index: i, rules: r })
  }
  return { points, outOfControl, signals }
}

function asSubgroups(x: ArrayLike<number | null | undefined> | number[][], subgroup?: number): number[][] {
  if (Array.isArray(x) && x.length > 0 && Array.isArray(x[0])) {
    return (x as number[][]).map((g) => Array.from(cleanNumbers(g)))
  }
  const v = Array.from(cleanNumbers(x as ArrayLike<number | null | undefined>))
  if (!(subgroup && subgroup >= 2)) return v.map((xi) => [xi])
  const groups: number[][] = []
  for (let i = 0; i + subgroup <= v.length; i += subgroup) groups.push(v.slice(i, i + subgroup))
  if (groups.length < 2) throw new RangeError('controlChart: need at least 2 complete subgroups')
  return groups
}

export type ShewhartType = 'i-mr' | 'xbar-r' | 'xbar-s' | 'z-mr'
export type AttributeType = 'p' | 'np' | 'c' | 'u' | 'laney-p' | 'laney-u'
export type ControlChartType = ShewhartType | AttributeType

export interface ControlChartOptions {
  type?: ControlChartType
  /** Subgroup size for X̄-R / X̄-S, or when `x` is a flat vector. */
  subgroup?: number
  /** Known process mean (I-MR / Z-MR); default = sample mean. */
  mu?: number
  /** Known process sigma; default estimated from MR / R / S. */
  sigma?: number
  /** Nelson rules to apply (default all 1–8). Empty → limits only. */
  rules?: NelsonRule[]
  /** Attribute: trials / sample sizes per point (P, NP, U, Laney). */
  sizes?: ArrayLike<number>
  /** Historical center for attribute charts. */
  center?: number
}

function movingRanges(x: number[]): number[] {
  const mr: number[] = []
  for (let i = 1; i < x.length; i++) mr.push(Math.abs(x[i]! - x[i - 1]!))
  return mr
}

function shewhartIMR(x: number[], options: ControlChartOptions): ControlChartResult {
  if (x.length < 2) throw new RangeError('controlChart I-MR needs at least 2 observations')
  const mr = movingRanges(x)
  const mrBar = meanOf(mr)
  const d2 = spcConstants(2).d2
  const sigma = options.sigma ?? mrBar / d2
  const center = options.mu ?? meanOf(x)
  const ucl = center + 3 * sigma
  const lcl = center - 3 * sigma
  const rules = options.rules ?? ([1, 2, 3, 4, 5, 6, 7, 8] as NelsonRule[])
  const main = packPoints(x, center, ucl, lcl, sigma, rules)
  const mrUcl = spcConstants(2).D4 * mrBar
  const mrLcl = spcConstants(2).D3 * mrBar
  const companion = packPoints(mr, mrBar, mrUcl, mrLcl, mrBar / 3 || 1, [1])
  return {
    type: 'i-mr',
    center,
    sigma,
    ucl,
    lcl,
    ...main,
    companion: { type: 'mr', center: mrBar, sigma: mrBar / 3, ucl: mrUcl, lcl: mrLcl, ...companion },
  }
}

function shewhartXbar(groups: number[][], useS: boolean, options: ControlChartOptions): ControlChartResult {
  const ns = groups.map((g) => g.length)
  if (ns.some((n) => n < 2)) throw new RangeError('controlChart X̄ needs subgroup size ≥ 2')
  const means = groups.map(meanOf)
  const stats = groups.map((g) => (useS ? sdOf(g, 1) : rangeOf(g)))
  const n0 = ns[0]!
  const equal = ns.every((n) => n === n0)
  const center = options.mu ?? meanOf(means)
  let sigma: number
  let ucl: number | number[]
  let lcl: number | number[]
  let sigmas: number | number[]
  let companionUcl: number | number[]
  let companionLcl: number | number[]
  let companionCenter: number
  if (equal) {
    const c = spcConstants(n0)
    const bar = meanOf(stats)
    sigma = options.sigma ?? (useS ? bar / c.c4 : bar / c.d2)
    // UCL = X̄ ± A₂ R̄ (or A₃ S̄)
    ucl = center + (useS ? c.A3 * bar : c.A2 * bar)
    lcl = center - (useS ? c.A3 * bar : c.A2 * bar)
    sigmas = (ucl as number - center) / 3
    companionCenter = bar
    companionUcl = (useS ? c.B4 : c.D4) * bar
    companionLcl = (useS ? c.B3 : c.D3) * bar
  } else {
    // Variable subgroup size: estimate pooled within sigma, then per-point limits
    let num = 0
    let den = 0
    for (let i = 0; i < groups.length; i++) {
      const c = spcConstants(ns[i]!)
      num += useS ? stats[i]! / c.c4 : stats[i]! / c.d2
      den++
    }
    sigma = options.sigma ?? num / den
    ucl = means.map((_, i) => center + (3 * sigma) / Math.sqrt(ns[i]!))
    lcl = means.map((_, i) => center - (3 * sigma) / Math.sqrt(ns[i]!))
    sigmas = means.map((_, i) => sigma / Math.sqrt(ns[i]!))
    companionCenter = meanOf(stats)
    companionUcl = stats.map((_, i) => {
      const c = spcConstants(ns[i]!)
      return (useS ? c.B4 : c.D4) * (useS ? sigma * c.c4 : sigma * c.d2)
    })
    companionLcl = stats.map((_, i) => {
      const c = spcConstants(ns[i]!)
      return (useS ? c.B3 : c.D3) * (useS ? sigma * c.c4 : sigma * c.d2)
    })
  }
  const rules = options.rules ?? ([1, 2, 3, 4, 5, 6, 7, 8] as NelsonRule[])
  const main = packPoints(means, center, ucl, lcl, sigmas, rules, ns)
  const companion = packPoints(
    stats,
    companionCenter,
    companionUcl,
    companionLcl,
    typeof companionUcl === 'number' ? (companionUcl - companionCenter) / 3 || 1 : companionUcl.map((u, i) => (u - (typeof companionLcl === 'number' ? companionCenter : companionCenter)) / 3 || 1),
    [1],
    ns,
  )
  return {
    type: useS ? 'xbar-s' : 'xbar-r',
    center,
    sigma,
    ucl: typeof ucl === 'number' ? ucl : meanOf(ucl),
    lcl: typeof lcl === 'number' ? lcl : meanOf(lcl),
    ...main,
    companion: {
      type: useS ? 's' : 'r',
      center: companionCenter,
      sigma: companionCenter / 3,
      ucl: typeof companionUcl === 'number' ? companionUcl : meanOf(companionUcl),
      lcl: typeof companionLcl === 'number' ? companionLcl : meanOf(companionLcl),
      ...companion,
    },
  }
}

function shewhartZMR(x: number[], options: ControlChartOptions): ControlChartResult {
  if (x.length < 2) throw new RangeError('controlChart Z-MR needs at least 2 observations')
  // Short-run: standardize each observation by historical μ, σ (or estimate from data)
  const mu = options.mu ?? meanOf(x)
  const mr = movingRanges(x)
  const sigma = options.sigma ?? meanOf(mr) / spcConstants(2).d2
  if (!(sigma > 0)) throw new RangeError('controlChart Z-MR: sigma must be > 0')
  const z = x.map((xi) => (xi - mu) / sigma)
  const zmr = movingRanges(z)
  const mrBar = meanOf(zmr)
  const rules = options.rules ?? ([1, 2, 3, 4, 5, 6, 7, 8] as NelsonRule[])
  const main = packPoints(z, 0, 3, -3, 1, rules)
  const companion = packPoints(zmr, mrBar, spcConstants(2).D4 * mrBar, spcConstants(2).D3 * mrBar, mrBar / 3 || 1, [1])
  return {
    type: 'z-mr',
    center: 0,
    sigma: 1,
    ucl: 3,
    lcl: -3,
    ...main,
    companion: { type: 'z-mr-mr', center: mrBar, sigma: mrBar / 3, ucl: spcConstants(2).D4 * mrBar, lcl: spcConstants(2).D3 * mrBar, ...companion },
  }
}

function attributeChart(
  counts: number[],
  sizes: number[] | undefined,
  type: AttributeType,
  options: ControlChartOptions,
): ControlChartResult {
  const m = counts.length
  if (m < 2) throw new RangeError('controlChart attribute needs at least 2 points')
  const rules = options.rules ?? ([1] as NelsonRule[])

  if (type === 'c') {
    const center = options.center ?? meanOf(counts)
    if (!(center >= 0)) throw new RangeError('controlChart C: center must be ≥ 0')
    const sigma = Math.sqrt(center)
    const ucl = center + 3 * sigma
    const lcl = Math.max(0, center - 3 * sigma)
    return { type: 'c', center, sigma, ucl, lcl, ...packPoints(counts, center, ucl, lcl, sigma, rules) }
  }

  if (!sizes || sizes.length !== m) throw new RangeError(`controlChart ${type}: sizes must match the number of points`)
  if (sizes.some((n) => !(n > 0))) throw new RangeError('controlChart: sizes must be > 0')

  if (type === 'p' || type === 'laney-p') {
    const p = counts.map((c, i) => c / sizes[i]!)
    const pBar = options.center ?? counts.reduce((s, c, i) => s + c, 0) / sizes.reduce((s, n) => s + n, 0)
    const sigmaI = sizes.map((n) => Math.sqrt((pBar * (1 - pBar)) / n))
    if (type === 'p') {
      const ucl = sigmaI.map((s) => Math.min(1, pBar + 3 * s))
      const lcl = sigmaI.map((s) => Math.max(0, pBar - 3 * s))
      return { type: 'p', center: pBar, sigma: meanOf(sigmaI), ucl: meanOf(ucl), lcl: meanOf(lcl), ...packPoints(p, pBar, ucl, lcl, sigmaI, rules, sizes) }
    }
    // Laney P′: inflate by σ_z from standardized residuals
    const z = p.map((pi, i) => (pi - pBar) / sigmaI[i]!)
    const mr = movingRanges(z)
    const sigmaZ = meanOf(mr) / spcConstants(2).d2
    const sig = sigmaI.map((s) => s * sigmaZ)
    const ucl = sig.map((s) => Math.min(1, pBar + 3 * s))
    const lcl = sig.map((s) => Math.max(0, pBar - 3 * s))
    return { type: 'laney-p', center: pBar, sigma: meanOf(sig), ucl: meanOf(ucl), lcl: meanOf(lcl), ...packPoints(p, pBar, ucl, lcl, sig, rules, sizes) }
  }

  if (type === 'np') {
    const n0 = sizes[0]!
    if (sizes.some((n) => n !== n0)) throw new RangeError('controlChart NP requires constant sample size (use P for variable n)')
    const center = options.center ?? meanOf(counts)
    const pBar = center / n0
    const sigma = Math.sqrt(n0 * pBar * (1 - pBar))
    const ucl = Math.min(n0, center + 3 * sigma)
    const lcl = Math.max(0, center - 3 * sigma)
    return { type: 'np', center, sigma, ucl, lcl, ...packPoints(counts, center, ucl, lcl, sigma, rules, sizes) }
  }

  // U / Laney U′
  const u = counts.map((c, i) => c / sizes[i]!)
  const uBar = options.center ?? counts.reduce((s, c) => s + c, 0) / sizes.reduce((s, n) => s + n, 0)
  const sigmaI = sizes.map((n) => Math.sqrt(uBar / n))
  if (type === 'u') {
    const ucl = sigmaI.map((s) => uBar + 3 * s)
    const lcl = sigmaI.map((s) => Math.max(0, uBar - 3 * s))
    return { type: 'u', center: uBar, sigma: meanOf(sigmaI), ucl: meanOf(ucl), lcl: meanOf(lcl), ...packPoints(u, uBar, ucl, lcl, sigmaI, rules, sizes) }
  }
  const z = u.map((ui, i) => (ui - uBar) / sigmaI[i]!)
  const mr = movingRanges(z)
  const sigmaZ = meanOf(mr) / spcConstants(2).d2
  const sig = sigmaI.map((s) => s * sigmaZ)
  const ucl = sig.map((s) => uBar + 3 * s)
  const lcl = sig.map((s) => Math.max(0, uBar - 3 * s))
  return { type: 'laney-u', center: uBar, sigma: meanOf(sig), ucl: meanOf(ucl), lcl: meanOf(lcl), ...packPoints(u, uBar, ucl, lcl, sig, rules, sizes) }
}

/**
 * Shewhart / attribute control chart.
 * - Variables: pass a flat series (`type: 'i-mr'`) or subgroups (`type: 'xbar-r' | 'xbar-s'`, `subgroup` or `number[][]`).
 * - Attributes: pass counts with `sizes` for P/NP/U/Laney; C needs only counts.
 */
export function controlChart(
  x: ArrayLike<number | null | undefined> | number[][],
  options: ControlChartOptions = {},
): ControlChartResult {
  const type = options.type ?? (Array.isArray(x) && Array.isArray((x as number[][])[0]) ? 'xbar-r' : options.subgroup && options.subgroup >= 2 ? 'xbar-r' : 'i-mr')

  if (type === 'p' || type === 'np' || type === 'c' || type === 'u' || type === 'laney-p' || type === 'laney-u') {
    const counts = Array.from(cleanNumbers(x as ArrayLike<number | null | undefined>))
    const sizes = options.sizes ? Array.from(options.sizes) : undefined
    return attributeChart(counts, sizes, type, options)
  }

  if (type === 'i-mr') {
    return shewhartIMR(Array.from(cleanNumbers(x as ArrayLike<number | null | undefined>)), options)
  }
  if (type === 'z-mr') {
    return shewhartZMR(Array.from(cleanNumbers(x as ArrayLike<number | null | undefined>)), options)
  }
  const groups = asSubgroups(x, options.subgroup)
  return shewhartXbar(groups, type === 'xbar-s', options)
}

// ---- 4.4 EWMA / CUSUM / MA ------------------------------------------------------------------------

export interface EwmaResult {
  type: 'ewma'
  lambda: number
  L: number
  center: number
  sigma: number
  z: number[]
  ucl: number[]
  lcl: number[]
  outOfControl: number[]
}

/** EWMA chart: zₜ = λ xₜ + (1−λ) zₜ₋₁ with time-varying limits (Lucas & Saccucci). */
export function ewma(
  x: ArrayLike<number | null | undefined>,
  options: { lambda?: number; L?: number; mu?: number; sigma?: number } = {},
): EwmaResult {
  const v = Array.from(cleanNumbers(x))
  if (v.length < 2) throw new RangeError('ewma needs at least 2 observations')
  const lambda = options.lambda ?? 0.2
  const L = options.L ?? 3
  if (!(lambda > 0 && lambda <= 1)) throw new RangeError(`ewma: lambda must be in (0, 1] (got ${lambda})`)
  if (!(L > 0)) throw new RangeError(`ewma: L must be > 0 (got ${L})`)
  const center = options.mu ?? meanOf(v)
  const mr = movingRanges(v)
  const sigma = options.sigma ?? meanOf(mr) / spcConstants(2).d2
  const z: number[] = []
  const ucl: number[] = []
  const lcl: number[] = []
  const outOfControl: number[] = []
  let prev = center
  for (let t = 0; t < v.length; t++) {
    const zt = lambda * v[t]! + (1 - lambda) * prev
    const varFactor = (lambda / (2 - lambda)) * (1 - (1 - lambda) ** (2 * (t + 1)))
    const half = L * sigma * Math.sqrt(varFactor)
    z.push(zt)
    ucl.push(center + half)
    lcl.push(center - half)
    if (zt > center + half || zt < center - half) outOfControl.push(t)
    prev = zt
  }
  return { type: 'ewma', lambda, L, center, sigma, z, ucl, lcl, outOfControl }
}

export interface CusumResult {
  type: 'cusum'
  k: number
  h: number
  center: number
  sigma: number
  /** High-side CUSUM C⁺ */
  cPlus: number[]
  /** Low-side CUSUM C⁻ */
  cMinus: number[]
  outOfControl: number[]
  /** V-mask decision interval equivalent: signal when |S| exceeds h with slope k. */
  vMask?: { slope: number; decisionInterval: number }
}

/** Tabular two-sided CUSUM (and optional V-mask parameters). Defaults k = 0.5, h = 5 (ARL₀ ≈ 465). */
export function cusum(
  x: ArrayLike<number | null | undefined>,
  options: { k?: number; h?: number; mu?: number; sigma?: number; vMask?: boolean } = {},
): CusumResult {
  const v = Array.from(cleanNumbers(x))
  if (v.length < 2) throw new RangeError('cusum needs at least 2 observations')
  const k = options.k ?? 0.5
  const h = options.h ?? 5
  if (!(k >= 0)) throw new RangeError(`cusum: k must be ≥ 0 (got ${k})`)
  if (!(h > 0)) throw new RangeError(`cusum: h must be > 0 (got ${h})`)
  const center = options.mu ?? meanOf(v)
  const mr = movingRanges(v)
  const sigma = options.sigma ?? meanOf(mr) / spcConstants(2).d2
  if (!(sigma > 0)) throw new RangeError('cusum: sigma must be > 0')
  const cPlus: number[] = []
  const cMinus: number[] = []
  const outOfControl: number[] = []
  let cp = 0
  let cm = 0
  for (let i = 0; i < v.length; i++) {
    const z = (v[i]! - center) / sigma
    cp = Math.max(0, cp + z - k)
    cm = Math.max(0, cm - z - k)
    cPlus.push(cp)
    cMinus.push(cm)
    if (cp > h || cm > h) outOfControl.push(i)
  }
  const result: CusumResult = { type: 'cusum', k, h, center, sigma, cPlus, cMinus, outOfControl }
  if (options.vMask) result.vMask = { slope: k, decisionInterval: h }
  return result
}

export interface MovingAverageResult {
  type: 'ma'
  span: number
  center: number
  sigma: number
  ma: number[]
  ucl: number[]
  lcl: number[]
  outOfControl: number[]
}

/** Moving-average chart of span `w` (default 5) with limits ±3 σ/√min(t,w). */
export function movingAverage(
  x: ArrayLike<number | null | undefined>,
  options: { span?: number; mu?: number; sigma?: number } = {},
): MovingAverageResult {
  const v = Array.from(cleanNumbers(x))
  if (v.length < 2) throw new RangeError('movingAverage needs at least 2 observations')
  const span = options.span ?? 5
  if (!(span >= 2 && Number.isInteger(span))) throw new RangeError(`movingAverage: span must be an integer ≥ 2 (got ${span})`)
  const center = options.mu ?? meanOf(v)
  const mr = movingRanges(v)
  const sigma = options.sigma ?? meanOf(mr) / spcConstants(2).d2
  const ma: number[] = []
  const ucl: number[] = []
  const lcl: number[] = []
  const outOfControl: number[] = []
  for (let i = 0; i < v.length; i++) {
    const w = Math.min(span, i + 1)
    let s = 0
    for (let j = i - w + 1; j <= i; j++) s += v[j]!
    const m = s / w
    const half = (3 * sigma) / Math.sqrt(w)
    ma.push(m)
    ucl.push(center + half)
    lcl.push(center - half)
    if (m > center + half || m < center - half) outOfControl.push(i)
  }
  return { type: 'ma', span, center, sigma, ma, ucl, lcl, outOfControl }
}
