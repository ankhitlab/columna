/**
 * Studentized range distribution and Tukey's honest significant difference (Tukey–Kramer for
 * unequal group sizes) — Minitab's "Tukey pairwise comparisons" after One-Way ANOVA.
 *
 * `ptukey` follows the algorithm of Copenhaver & Holland (1988) as implemented in R (`ptukey.c`):
 * Gauss–Legendre quadrature of the range probability over the chi distribution of the pooled
 * standard deviation. Gauss–Legendre nodes / weights are generated numerically at load time
 * instead of being transcribed.
 */
import { lgamma, normal, t as tDist, chi2 as chi2Dist } from './dist.js'
import { cleanNumbers } from './tests.js'

const STD = normal()

/** Gauss–Legendre nodes (positive half, descending) and weights for an even order n. */
function legendre(n: number): { x: number[]; w: number[] } {
  const x: number[] = []
  const w: number[] = []
  const m = n >> 1
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
    x.push(z)
    w.push(2 / ((1 - z * z) * pp * pp))
  }
  return { x, w }
}
const LEG12 = legendre(12)
const LEG16 = legendre(16)

/** P(range of cc standard normals ≤ w)^rr — Hartley's form, integrated numerically. */
function wprob(w: number, rr: number, cc: number): number {
  const C1 = -30
  const C2 = -50
  const C3 = 60
  const bb = 8
  const wlar = 3
  const qsqz = w * 0.5
  if (qsqz >= bb) return 1
  let prW = 2 * STD.cdf(qsqz) - 1
  prW = prW >= Math.exp(C2 / cc) ? Math.pow(prW, cc) : 0
  const wincr = w > wlar ? 2 : 3
  let blb = qsqz
  const binc = (bb - qsqz) / wincr
  let bub = blb + binc
  let einsum = 0
  const cc1 = cc - 1
  for (let wi = 1; wi <= wincr; wi++) {
    let elsum = 0
    const a = 0.5 * (bub + blb)
    const b = 0.5 * (bub - blb)
    for (let jj = 1; jj <= 12; jj++) {
      let j: number
      let xx: number
      if (jj > 6) {
        j = 12 - jj + 1
        xx = LEG12.x[j - 1]!
      } else {
        j = jj
        xx = -LEG12.x[j - 1]!
      }
      const ac = a + b * xx
      const qexpo = ac * ac
      if (qexpo > C3) break
      const pplus = 2 * STD.cdf(ac)
      const pminus = 2 * STD.cdf(ac - w)
      let rinsum = pplus * 0.5 - pminus * 0.5
      if (rinsum >= Math.exp(C1 / cc1)) {
        rinsum = LEG12.w[j - 1]! * Math.exp(-(0.5 * qexpo)) * Math.pow(rinsum, cc1)
        elsum += rinsum
      }
    }
    elsum *= 2 * b * cc * 0.3989422804014327 // 1/√(2π)
    einsum += elsum
    blb = bub
    bub += binc
  }
  prW += einsum
  if (prW <= Math.exp(C1 / rr)) return 0
  prW = Math.pow(prW, rr)
  return prW >= 1 ? 1 : prW
}

/**
 * Cdf of the studentized range: P(Q ≤ q) for `k` groups and `df` error degrees of freedom
 * (`nranges` independent ranges, normally 1). Matches R `ptukey` / scipy `studentized_range.cdf`.
 */
export function ptukey(q: number, k: number, df: number, nranges = 1): number {
  if (Number.isNaN(q) || Number.isNaN(k) || Number.isNaN(df)) return NaN
  if (q <= 0) return 0
  if (df < 2 || nranges < 1 || k < 2) return NaN
  if (!Number.isFinite(q)) return 1
  if (k === 2 && nranges === 1) return Math.max(0, 1 - 2 * tDist(df).sf(q / Math.SQRT2))
  // Integrating over the standard deviation, rather than variance, avoids the
  // endpoint quadrature bias of the original fixed panels for small error df.
  if (df <= 10) {
    const max = Math.sqrt(chi2Dist(df).isf(1e-13) / df)
    const boundaries = [0, max]
    for (let x = .25; x < max; x += .25) boundaries.push(x)
    for (const c of [.5, 1, 2, 4, 8, 16]) if (c / q < max) boundaries.push(c / q)
    boundaries.sort((a, b) => a - b)
    const logConst = Math.LN2 + (df / 2) * Math.log(df / 2) - lgamma(df / 2)
    let sum = 0
    for (let i = 1; i < boundaries.length; i++) {
      const mid = (boundaries[i]! + boundaries[i - 1]!) / 2, half = (boundaries[i]! - boundaries[i - 1]!) / 2
      for (let j = 0; j < LEG16.x.length; j++) for (const sign of [-1, 1]) {
        const s = mid + sign * half * LEG16.x[j]!
        sum += half * LEG16.w[j]! * wprob(q * s, nranges, k) * Math.exp(logConst + (df - 1) * Math.log(s) - df * s * s / 2)
      }
    }
    return Math.min(1, Math.max(0, sum))
  }
  if (df > 25000) return wprob(q, nranges, k)

  const f2 = df * 0.5
  let f2lf = f2 * Math.log(df) - df * Math.LN2 - lgamma(f2)
  const f21 = f2 - 1
  const ff4 = df * 0.25
  const ulen = df <= 100 ? 1 : df <= 800 ? 0.5 : df <= 5000 ? 0.25 : 0.125
  f2lf += Math.log(ulen)

  let ans = 0
  let otsum = 0
  for (let i = 1; i <= 50; i++) {
    otsum = 0
    const twa1 = (2 * i - 1) * ulen
    for (let jj = 1; jj <= 16; jj++) {
      let j: number
      let t1: number
      let qsqz: number
      if (jj > 8) {
        j = jj - 8 - 1
        const xu = LEG16.x[j]! * ulen
        t1 = f2lf + f21 * Math.log(twa1 + xu) - (xu + twa1) * ff4
        qsqz = q * Math.sqrt((xu + twa1) * 0.5)
      } else {
        j = jj - 1
        const xu = LEG16.x[j]! * ulen
        t1 = f2lf + f21 * Math.log(twa1 - xu) + (xu - twa1) * ff4
        qsqz = q * Math.sqrt((-xu + twa1) * 0.5)
      }
      if (t1 >= -30) otsum += wprob(qsqz, nranges, k) * LEG16.w[j]! * Math.exp(t1)
    }
    if (i * ulen >= 1 && otsum <= 1e-14) break
    ans += otsum
  }
  return ans > 1 ? 1 : ans
}

/** Inverse of `ptukey`: the q with P(Q ≤ q) = p (R `qtukey`, scipy `studentized_range.ppf`). */
export function qtukey(p: number, k: number, df: number, nranges = 1): number {
  if (Number.isNaN(p) || p < 0 || p > 1) return NaN
  if (p === 0) return 0
  if (p === 1) return Infinity
  if (!(k >= 2 && df >= 2 && nranges >= 1)) throw new RangeError('qtukey: need k >= 2 and df >= 2')
  let lo = 0, hi = 4
  while (ptukey(hi, k, df, nranges) < p && hi < 1e8) hi *= 2
  if (ptukey(hi, k, df, nranges) < p) throw new RangeError('qtukey: failed to bracket quantile')
  for (let i = 0; i < 80; i++) {
    const mid = (lo + hi) / 2
    if (ptukey(mid, k, df, nranges) < p) lo = mid
    else hi = mid
    if (hi - lo <= 1e-9 * Math.max(1, hi)) break
  }
  return (lo + hi) / 2
}

export interface TukeyComparison {
  a: string
  b: string
  /** mean(a) − mean(b) */
  diff: number
  se: number
  /** Studentized range statistic |diff| / se. */
  q: number
  pValue: number
  ci: [number, number]
  significant: boolean
}

export interface TukeyResult {
  test: 'Tukey HSD'
  alpha: number
  /** Critical studentized range q(1 − α; k, df). */
  qCritical: number
  df: number
  msWithin: number
  groups: Array<{ name: string; n: number; mean: number; sd: number; letters: string }>
  comparisons: TukeyComparison[]
}

// ---- shared pieces for the pairwise-comparison family ------------------------------------------

export type GroupStats = { name: string; n: number; mean: number; m2: number; sd: number }

/** Per-group descriptives plus the one-way ANOVA error term (df = N − k, MS_within). */
export function pooledGroups(
  groups: Record<string, ArrayLike<number | null | undefined>> | Array<ArrayLike<number | null | undefined>>,
  fn: string,
): { stats: GroupStats[]; k: number; df: number; msWithin: number } {
  const entries = Array.isArray(groups) ? groups.map((g, i) => [String(i), g] as const) : Object.entries(groups)
  const stats = entries
    .map(([name, g]) => {
      const v = cleanNumbers(g)
      const n = v.length
      let s = 0
      for (let i = 0; i < n; i++) s += v[i]!
      const mean = n ? s / n : NaN
      let m2 = 0
      for (let i = 0; i < n; i++) m2 += (v[i]! - mean) ** 2
      return { name, n, mean, m2, sd: n > 1 ? Math.sqrt(m2 / (n - 1)) : NaN }
    })
    .filter((s) => s.n > 0)
  const k = stats.length
  if (k < 2) throw new RangeError(`${fn} needs at least 2 non-empty groups, got ${k}`)
  let N = 0
  let ssw = 0
  for (const s of stats) {
    N += s.n
    ssw += s.m2
  }
  const df = N - k
  if (df < 2) throw new RangeError(`${fn} needs error degrees of freedom ≥ 2, got ${df}`)
  return { stats, k, df, msWithin: ssw / df }
}

/**
 * Minitab-style grouping letters: walk means in descending order; each letter covers a maximal run
 * of groups that are pairwise not significantly different; letter sets contained in another are dropped.
 */
export function groupingLetters(stats: GroupStats[], differ: Set<string>): Map<string, string> {
  const order = [...stats].sort((p, q) => q.mean - p.mean).map((s) => s.name)
  let sets = [order]
  // Insert-and-absorb: every nonsignificant pair shares a letter, even when means
  // are noncontiguous in their order (unequal SEs invalidate the old run heuristic).
  for (let i = 0; i < order.length; i++) for (let j = i + 1; j < order.length; j++) {
    const a = order[i]!, b = order[j]!
    if (!differ.has(`${a}\0${b}`) && !differ.has(`${b}\0${a}`)) continue
    sets = sets.flatMap((set) => set.includes(a) && set.includes(b) ? [set.filter((v) => v !== a), set.filter((v) => v !== b)] : [set])
    sets = sets.filter((set, idx) => !sets.some((other, jdx) => jdx !== idx && set.every((v) => other.includes(v)) && (other.length > set.length || jdx < idx)))
  }
  sets.sort((a, b) => order.indexOf(a[0]!) - order.indexOf(b[0]!))
  const letterOf = new Map<string, string>()
  const label = (i: number) => { let s = ''; for (i++; i > 0; i = Math.floor((i - 1) / 26)) s = String.fromCharCode(65 + (i - 1) % 26) + s; return s }
  sets.forEach((set, i) => { for (const name of set) letterOf.set(name, [letterOf.get(name), label(i)].filter(Boolean).join(' ')) })
  return letterOf
}

/**
 * Tukey HSD pairwise comparisons of group means (Tukey–Kramer standard error for unequal sizes).
 * p-values and simultaneous confidence intervals use the studentized range with the ANOVA error
 * degrees of freedom; `letters` is Minitab-style grouping information (groups sharing a letter are
 * not significantly different at `alpha`).
 */
export function tukeyHSD(
  groups: Record<string, ArrayLike<number | null | undefined>> | Array<ArrayLike<number | null | undefined>>,
  options: { alpha?: number } = {},
): TukeyResult {
  const alpha = options.alpha ?? 0.05
  if (!(alpha > 0 && alpha < 1)) throw new RangeError(`alpha must be in (0, 1), got ${alpha}`)
  const { stats, k, df, msWithin } = pooledGroups(groups, 'tukeyHSD')
  const qCritical = qtukey(1 - alpha, k, df)

  const comparisons: TukeyComparison[] = []
  const differ = new Set<string>()
  for (let i = 0; i < k; i++) {
    for (let j = i + 1; j < k; j++) {
      const a = stats[i]!
      const b = stats[j]!
      const diff = a.mean - b.mean
      const se = Math.sqrt((msWithin / 2) * (1 / a.n + 1 / b.n))
      const q = Math.abs(diff) / se
      const pValue = 1 - ptukey(q, k, df)
      const h = qCritical * se
      const significant = q > qCritical
      if (significant) differ.add(`${a.name}\0${b.name}`).add(`${b.name}\0${a.name}`)
      comparisons.push({ a: a.name, b: b.name, diff, se, q, pValue: Math.min(1, Math.max(0, pValue)), ci: [diff - h, diff + h], significant })
    }
  }
  const letterOf = groupingLetters(stats, differ)
  return {
    test: 'Tukey HSD',
    alpha,
    qCritical,
    df,
    msWithin,
    groups: stats.map((s) => ({ name: s.name, n: s.n, mean: s.mean, sd: s.sd, letters: letterOf.get(s.name) ?? '' })),
    comparisons,
  }
}
