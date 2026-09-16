/**
 * Multiple comparisons after one-way ANOVA, Minitab's "Comparisons" dialog:
 *   Fisher LSD — pairwise t-tests on the pooled MS_within (individual error rate),
 *   Dunnett    — each treatment vs a control, family error rate via Dunnett's distribution.
 *
 * Dunnett's distribution is the maximum of correlated t statistics with product correlation
 * ρᵢⱼ = λᵢλⱼ, λᵢ = √(nᵢ/(nᵢ+n_c)), so P(max|Tᵢ| ≤ c) collapses to a double integral
 * (standard normal z × the chi distribution of the pooled sd) evaluated by Gauss–Legendre —
 * the same representation Dunnett (1955) used for the tables; scipy `dunnett` integrates it by QMC.
 */
import { lgamma, normal, t as tDist } from './dist.js'
import { groupingLetters, pooledGroups, ptukey, qtukey, type TukeyComparison } from './tukey.js'
import type { Alternative } from './tests.js'

const STD = normal()

export interface FisherResult {
  test: 'Fisher LSD'
  alpha: number
  /** t(1 − α/2; df) — the individual (per-comparison) critical value. */
  tCritical: number
  df: number
  msWithin: number
  groups: Array<{ name: string; n: number; mean: number; sd: number; letters: string }>
  comparisons: Array<Omit<TukeyComparison, 'q'> & { t: number }>
}

/**
 * Fisher's least significant difference: every pair compared with a t-test on the pooled error term
 * at the individual level `alpha` (no family-wise adjustment — Minitab reports the resulting
 * family error rate; here it is `familyAlpha` = 1 − (1 − α)^m).
 */
export function fisherLSD(
  groups: Record<string, ArrayLike<number | null | undefined>> | Array<ArrayLike<number | null | undefined>>,
  options: { alpha?: number } = {},
): FisherResult & { familyAlpha: number } {
  const alpha = options.alpha ?? 0.05
  if (!(alpha > 0 && alpha < 1)) throw new RangeError(`alpha must be in (0, 1), got ${alpha}`)
  const { stats, k, df, msWithin } = pooledGroups(groups, 'fisherLSD')
  const d = tDist(df)
  const tCritical = d.ppf(1 - alpha / 2)
  const comparisons: FisherResult['comparisons'] = []
  const differ = new Set<string>()
  for (let i = 0; i < k; i++) {
    for (let j = i + 1; j < k; j++) {
      const a = stats[i]!
      const b = stats[j]!
      const diff = a.mean - b.mean
      const se = Math.sqrt(msWithin * (1 / a.n + 1 / b.n))
      const t = diff / se
      const pValue = 2 * d.sf(Math.abs(t))
      const h = tCritical * se
      const significant = Math.abs(t) > tCritical
      if (significant) differ.add(`${a.name}\0${b.name}`).add(`${b.name}\0${a.name}`)
      comparisons.push({ a: a.name, b: b.name, diff, se, t, pValue: Math.min(1, pValue), ci: [diff - h, diff + h], significant })
    }
  }
  const letterOf = groupingLetters(stats, differ)
  const m = (k * (k - 1)) / 2
  return {
    test: 'Fisher LSD',
    alpha,
    familyAlpha: 1 - Math.pow(1 - alpha, m),
    tCritical,
    df,
    msWithin,
    groups: stats.map((s) => ({ name: s.name, n: s.n, mean: s.mean, sd: s.sd, letters: letterOf.get(s.name) ?? '' })),
    comparisons,
  }
}

// ---- Dunnett's distribution ------------------------------------------------------------------------

/** Gauss–Legendre nodes / weights on [−1, 1] (full set) for order n. */
function legendreFull(n: number): { x: Float64Array; w: Float64Array } {
  const x = new Float64Array(n)
  const w = new Float64Array(n)
  const m = (n + 1) >> 1
  for (let i = 1; i <= m; i++) {
    let z = Math.cos((Math.PI * (i - 0.25)) / (n + 0.5))
    let pp = 0
    for (let it = 0; it < 100; it++) {
      let p1 = 1
      let p2 = 0
      for (let j = 1; j <= n; j++) {
        const p3 = p2
        p2 = p1
        p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j
      }
      pp = (n * (z * p1 - p2)) / (z * z - 1)
      const z1 = z
      z = z1 - p1 / pp
      if (Math.abs(z - z1) < 1e-15) break
    }
    x[i - 1] = -z
    x[n - i] = z
    w[i - 1] = w[n - i] = 2 / ((1 - z * z) * pp * pp)
  }
  return { x, w }
}
const GL_Z = legendreFull(96) // for z ∈ [−8, 8]
const GL_U = legendreFull(96) // for the chi variable

/**
 * P(max Tᵢ ≤ c) (one-sided) or P(max |Tᵢ| ≤ c) (two-sided) for treatment-vs-control t statistics
 * sharing the pooled sd with `df` degrees of freedom; `lambdas[i] = √(nᵢ/(nᵢ+n_c))`.
 */
export function pdunnett(c: number, lambdas: number[], df: number, twoSided = true): number {
  if (Number.isNaN(c)) return NaN
  // two-sided: |T| ≤ c is empty for c ≤ 0; one-sided: P(all Tᵢ ≤ c) is small but positive for c < 0
  if (c === -Infinity || (twoSided && c <= 0)) return 0
  if (c === Infinity) return 1
  const gammas = lambdas.map((l) => Math.sqrt(1 - l * l))

  // P(all Zᵢ ≤ b | Z₀ = z) integrated over z ~ N(0,1) — Zᵢ = λᵢ Z₀ + γᵢ Wᵢ, Wᵢ independent
  const inner = (b: number): number => {
    let total = 0
    for (let i = 0; i < GL_Z.x.length; i++) {
      const z = 8 * GL_Z.x[i]!
      let prod = 1
      for (let j = 0; j < lambdas.length; j++) {
        const shift = lambdas[j]! * z
        const g = gammas[j]!
        const hi = STD.cdf((b + shift) / g)
        prod *= twoSided ? hi - STD.cdf((-b + shift) / g) : hi
        if (prod < 1e-300) break
      }
      total += GL_Z.w[i]! * 8 * STD.pdf(z) * prod
    }
    return total
  }
  if (df > 25000) return Math.min(1, inner(c))

  // Outer integral over u = s/σ: df·u² ~ χ²(df), so
  //   density(u) = df^(df/2) · u^(df−1) · exp(−df u²/2) / (2^(df/2−1) Γ(df/2)),
  // concentrated near 1 with sd ≈ 1/√(2df) for large df and right-skewed for small df.
  const sd = 1 / Math.sqrt(2 * df)
  const lo = Math.max(1e-9, 1 - 8 * sd)
  const hi = 1 + Math.max(8 * sd, 10 / Math.sqrt(df))
  const logNorm = (df / 2) * Math.log(df) - (df / 2 - 1) * Math.LN2 - lgamma(df / 2)
  let total = 0
  const mid = 0.5 * (lo + hi)
  const half = 0.5 * (hi - lo)
  for (let i = 0; i < GL_U.x.length; i++) {
    const u = mid + half * GL_U.x[i]!
    const logDensity = logNorm + (df - 1) * Math.log(u) - (df * u * u) / 2
    if (logDensity < -700) continue
    total += GL_U.w[i]! * half * Math.exp(logDensity) * inner(c * u)
  }
  return Math.min(1, total)
}

/** Inverse: c with P(max Tᵢ ≤ c) = p (bracketed bisection + secant polish). */
export function qdunnett(p: number, lambdas: number[], df: number, twoSided = true): number {
  if (Number.isNaN(p) || p <= 0 || p >= 1) return NaN
  let lo = twoSided ? 0 : -1
  let hi = 1
  while (pdunnett(hi, lambdas, df, twoSided) < p && hi < 1e6) hi *= 2
  while (!twoSided && pdunnett(lo, lambdas, df, twoSided) > p && lo > -1e6) lo *= 2
  for (let i = 0; i < 60; i++) {
    const mid = 0.5 * (lo + hi)
    if (pdunnett(mid, lambdas, df, twoSided) < p) lo = mid
    else hi = mid
    if (hi - lo < 1e-9 * hi) break
  }
  return 0.5 * (lo + hi)
}

export interface DunnettResult {
  test: 'Dunnett'
  control: string
  alpha: number
  alternative: Alternative
  /** Critical value of Dunnett's t for the simultaneous intervals. */
  tCritical: number
  df: number
  msWithin: number
  groups: Array<{ name: string; n: number; mean: number; sd: number }>
  comparisons: Array<{
    group: string
    /** mean(group) − mean(control) */
    diff: number
    se: number
    t: number
    /** Family-wise adjusted p-value: P(max|T| ≥ |t|) under H0. */
    pValue: number
    ci: [number, number]
    significant: boolean
  }>
}

/**
 * Dunnett's comparisons of each treatment with a control group (Minitab "Dunnett" / scipy `dunnett`).
 * `alternative: 'greater'` tests mean(group) > mean(control), `'less'` the opposite; two-sided default.
 */
export function dunnett(
  groups: Record<string, ArrayLike<number | null | undefined>> | Array<ArrayLike<number | null | undefined>>,
  options: { control: string; alpha?: number; alternative?: Alternative },
): DunnettResult {
  const alpha = options.alpha ?? 0.05
  if (!(alpha > 0 && alpha < 1)) throw new RangeError(`alpha must be in (0, 1), got ${alpha}`)
  const alternative = options.alternative ?? 'two-sided'
  const { stats, k, df, msWithin } = pooledGroups(groups, 'dunnett')
  const control = stats.find((s) => s.name === String(options.control))
  if (!control) throw new RangeError(`dunnett: control group "${options.control}" not found among ${stats.map((s) => s.name).join(', ')}`)
  const treatments = stats.filter((s) => s !== control)
  if (treatments.length !== k - 1) throw new RangeError('dunnett: control must appear once')
  const lambdas = treatments.map((s) => Math.sqrt(s.n / (s.n + control.n)))
  const twoSided = alternative === 'two-sided'
  const tCritical = qdunnett(1 - alpha, lambdas, df, twoSided)

  const comparisons = treatments.map((s, i) => {
    const diff = s.mean - control.mean
    const se = Math.sqrt(msWithin * (1 / s.n + 1 / control.n))
    const t = diff / se
    // adjusted p: probability that the maximum (|T| or signed T) exceeds the observed one
    const observed = twoSided ? Math.abs(t) : alternative === 'greater' ? t : -t
    const pValue = 1 - pdunnett(observed, lambdas, df, twoSided)
    const h = tCritical * se
    const ci: [number, number] = twoSided ? [diff - h, diff + h] : alternative === 'greater' ? [diff - h, Infinity] : [-Infinity, diff + h]
    const significant = observed > tCritical
    void i
    return { group: s.name, diff, se, t, pValue: Math.min(1, Math.max(0, pValue)), ci, significant }
  })
  return {
    test: 'Dunnett',
    control: control.name,
    alpha,
    alternative,
    tCritical,
    df,
    msWithin,
    groups: stats.map((s) => ({ name: s.name, n: s.n, mean: s.mean, sd: s.sd })),
    comparisons,
  }
}

// ---- Hsu's MCB --------------------------------------------------------------------------------------

export interface HsuResult {
  test: 'Hsu MCB'
  best: 'largest' | 'smallest'
  alpha: number
  df: number
  msWithin: number
  groups: Array<{
    name: string
    n: number
    mean: number
    sd: number
    /** ȳᵢ − best of the others (max for 'largest', min for 'smallest'). */
    diffToBest: number
    /** Constrained simultaneous CI for μᵢ − best of the others; always contains 0 at one end. */
    ci: [number, number]
    /** One-sided Dunnett critical value used for this group's intervals. */
    dCritical: number
    /** The interval leaves room for μᵢ to be the best (upper > 0 for 'largest', lower < 0 for 'smallest'). */
    canBeBest: boolean
    /** The interval shows μᵢ beats every other group (the zero end is the *other* side). */
    isBest: boolean
  }>
}

/**
 * Hsu's multiple comparisons with the best (Minitab "Hsu MCB"). For each group the constrained
 * simultaneous CI for μᵢ − (best of the other means) is
 *   largest is best:  [ min(0, minⱼ(ȳᵢ − ȳⱼ − Dᵢⱼ)),  max(0, minⱼ(ȳᵢ − ȳⱼ + Dᵢⱼ)) ]
 *   smallest is best: [ min(0, maxⱼ(ȳᵢ − ȳⱼ − Dᵢⱼ)),  max(0, maxⱼ(ȳᵢ − ȳⱼ + Dᵢⱼ)) ]
 * with Dᵢⱼ = dᵢ·√(MS_within (1/nᵢ + 1/nⱼ)) and dᵢ the one-sided Dunnett quantile treating group i as
 * the control (k − 1 comparisons, λⱼ = √(nⱼ/(nⱼ + nᵢ))) — Hsu (1996) for unbalanced designs.
 */
export function hsuMCB(
  groups: Record<string, ArrayLike<number | null | undefined>> | Array<ArrayLike<number | null | undefined>>,
  options: { best?: 'largest' | 'smallest'; alpha?: number } = {},
): HsuResult {
  const alpha = options.alpha ?? 0.05
  if (!(alpha > 0 && alpha < 1)) throw new RangeError(`alpha must be in (0, 1), got ${alpha}`)
  const best = options.best ?? 'largest'
  const { stats, k, df, msWithin } = pooledGroups(groups, 'hsuMCB')
  const dCache = new Map<string, number>()
  const out: HsuResult['groups'] = []
  for (let i = 0; i < k; i++) {
    const g = stats[i]!
    const others = stats.filter((_, j) => j !== i)
    const lambdas = others.map((o) => Math.sqrt(o.n / (o.n + g.n)))
    const key = [...lambdas].sort().map((l) => l.toPrecision(12)).join(',')
    let d = dCache.get(key)
    if (d === undefined) dCache.set(key, (d = qdunnett(1 - alpha, lambdas, df, false)))
    let lower = Infinity
    let upper = Infinity
    let bestOther = -Infinity
    if (best === 'smallest') {
      lower = -Infinity
      upper = -Infinity
      bestOther = Infinity
    }
    for (const o of others) {
      const D = d * Math.sqrt(msWithin * (1 / g.n + 1 / o.n))
      const diff = g.mean - o.mean
      if (best === 'largest') {
        lower = Math.min(lower, diff - D)
        upper = Math.min(upper, diff + D)
        bestOther = Math.max(bestOther, o.mean)
      } else {
        lower = Math.max(lower, diff - D)
        upper = Math.max(upper, diff + D)
        bestOther = Math.min(bestOther, o.mean)
      }
    }
    const ci: [number, number] = [Math.min(0, lower), Math.max(0, upper)]
    const canBeBest = best === 'largest' ? ci[1] > 0 : ci[0] < 0
    const isBest = best === 'largest' ? lower >= 0 : upper <= 0
    out.push({ name: g.name, n: g.n, mean: g.mean, sd: g.sd, diffToBest: g.mean - bestOther, ci, dCritical: d, canBeBest, isBest })
  }
  return { test: 'Hsu MCB', best, alpha, df, msWithin, groups: out }
}

export interface GamesHowellResult {
  test: 'Games-Howell'
  alpha: number
  groups: Array<{ name: string; n: number; mean: number; sd: number; letters: string }>
  comparisons: Array<Omit<TukeyComparison, 'q'> & { t: number; df: number }>
}

/**
 * Games–Howell pairwise comparisons (Welch SE + studentized range; no equal-variance assumption).
 */
export function gamesHowell(
  groups: Record<string, ArrayLike<number | null | undefined>> | Array<ArrayLike<number | null | undefined>>,
  options: { alpha?: number } = {},
): GamesHowellResult {
  const alpha = options.alpha ?? 0.05
  if (!(alpha > 0 && alpha < 1)) throw new RangeError(`alpha must be in (0, 1), got ${alpha}`)
  // Per-group stats without pooling MS
  const entries: Array<{ name: string; values: number[] }> = Array.isArray(groups)
    ? (groups as Array<ArrayLike<number | null | undefined>>).map((g, i) => ({
        name: `G${i + 1}`,
        values: Array.from(g).filter((v): v is number => typeof v === 'number' && Number.isFinite(v)),
      }))
    : Object.entries(groups as Record<string, ArrayLike<number | null | undefined>>).map(([name, g]) => ({
        name,
        values: Array.from(g).filter((v): v is number => typeof v === 'number' && Number.isFinite(v)),
      }))
  const stats = entries.map(({ name, values }) => {
    const n = values.length
    if (n < 2) throw new RangeError(`gamesHowell: group ${name} needs ≥2 observations`)
    const mean = values.reduce((a, b) => a + b, 0) / n
    let ss = 0
    for (const v of values) ss += (v - mean) ** 2
    const sd = Math.sqrt(ss / (n - 1))
    return { name, n, mean, sd, var: sd * sd }
  })
  const k = stats.length
  if (k < 2) throw new RangeError('gamesHowell: need ≥2 groups')
  const comparisons: GamesHowellResult['comparisons'] = []
  const differ = new Set<string>()
  for (let i = 0; i < k; i++) {
    for (let j = i + 1; j < k; j++) {
      const a = stats[i]!
      const b = stats[j]!
      const diff = a.mean - b.mean
      const se = Math.sqrt(a.var / a.n + b.var / b.n)
      // Welch–Satterthwaite df
      const num = (a.var / a.n + b.var / b.n) ** 2
      const den = (a.var / a.n) ** 2 / (a.n - 1) + (b.var / b.n) ** 2 / (b.n - 1)
      const df = Math.max(1, num / Math.max(1e-300, den))
      const t = se > 0 ? diff / se : 0
      const qCrit = qtukey(1 - alpha, k, df)
      const tCrit = qCrit / Math.SQRT2
      const pValue = Math.min(1, 2 * tDist(df).sf(Math.abs(t)))
      // Approximate family-wise p via studentized range of |t|√2
      const qObs = Math.abs(t) * Math.SQRT2
      const pTukey = Math.min(1, 1 - ptukey(qObs, k, df))
      const h = tCrit * se
      const significant = Math.abs(t) > tCrit
      if (significant) differ.add(`${a.name}\0${b.name}`).add(`${b.name}\0${a.name}`)
      comparisons.push({
        a: a.name,
        b: b.name,
        diff,
        se,
        t,
        df,
        pValue: Number.isFinite(pTukey) ? pTukey : pValue,
        ci: [diff - h, diff + h],
        significant,
      })
    }
  }
  const letterOf = groupingLetters(
    stats.map((s) => ({ name: s.name, n: s.n, mean: s.mean, m2: s.var * (s.n - 1), sd: s.sd })),
    differ,
  )
  return {
    test: 'Games-Howell',
    alpha,
    groups: stats.map((s) => ({ name: s.name, n: s.n, mean: s.mean, sd: s.sd, letters: letterOf.get(s.name) ?? '' })),
    comparisons,
  }
}
