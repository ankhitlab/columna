/**
 * Studentized range distribution and Tukey's honest significant difference (Tukey–Kramer for
 * unequal group sizes) — Minitab's "Tukey pairwise comparisons" after One-Way ANOVA.
 *
 * `ptukey` follows the algorithm of Copenhaver & Holland (1988) as implemented in R (`ptukey.c`):
 * Gauss–Legendre quadrature of the range probability over the chi distribution of the pooled
 * standard deviation. Gauss–Legendre nodes / weights are generated numerically at load time
 * instead of being transcribed.
 */
import { lgamma, normal } from './dist.js'
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
  // initial guess from R's qinv (Copenhaver & Holland) then secant iterations
  const p0 = 0.322232421088
  const q0 = 0.99348462606
  const p1 = -1
  const q1 = 0.588581570495
  const p2 = -0.342242088547
  const q2 = 0.531103462366
  const p3 = -0.204231210125
  const q3 = 0.10353775285
  const p4 = -0.453642210148e-4
  const q4 = 0.38560700634e-2
  const c1 = 0.8832
  const c2 = 0.2368
  const c3 = 1.214
  const c4 = 1.208
  const c5 = 1.4142
  const ps = 0.5 - 0.5 * p
  const yi = Math.sqrt(Math.log(1 / (ps * ps)))
  let t = yi + ((((yi * p4 + p3) * yi + p2) * yi + p1) * yi + p0) / ((((yi * q4 + q3) * yi + q2) * yi + q1) * yi + q0)
  if (df < 120) t += (t * t * t + t) / df / 4
  let qq = c1 - c2 * t
  if (df < 120) qq += -c3 / df + (c4 * t) / df
  let x0 = t * (qq * Math.log(k - 1) + c5)

  // secant with bisection safeguard on f(x) = ptukey(x) − p
  let valx0 = ptukey(x0, k, df, nranges) - p
  let x1 = valx0 > 0 ? Math.max(0, x0 - 1) : x0 + 1
  let valx1 = ptukey(x1, k, df, nranges) - p
  let ans = 0
  for (let iter = 1; iter < 100; iter++) {
    if (valx1 === valx0) break
    ans = x1 - (valx1 * (x1 - x0)) / (valx1 - valx0)
    if (ans < 0) ans = 0
    valx0 = valx1
    x0 = x1
    valx1 = ptukey(ans, k, df, nranges) - p
    x1 = ans
    if (Math.abs(x1 - x0) < 1e-10 * Math.max(1, x1)) break
  }
  return ans
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
  const order = [...stats].sort((p, q) => q.mean - p.mean)
  const sets: string[][] = []
  for (let i = 0; i < order.length; i++) {
    const set = [order[i]!.name]
    for (let j = i + 1; j < order.length; j++) {
      const cand = order[j]!.name
      if (set.every((m) => !differ.has(`${m}\0${cand}`))) set.push(cand)
      else break
    }
    if (!sets.some((s) => set.every((m) => s.includes(m)))) sets.push(set)
  }
  const letterOf = new Map<string, string>()
  sets.forEach((set, idx) => {
    const letter = String.fromCharCode(65 + (idx % 26))
    for (const m of set) letterOf.set(m, (letterOf.get(m) ?? '') + letter)
  })
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
