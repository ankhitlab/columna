/**
 * MANOVA — Pillai, Wilks, Hotelling–Lawley, Roy: one-way (`manova`) and the general linear model form
 * (`manovaModel`, Minitab Stat › ANOVA › General MANOVA) with Type III SSCP matrices per term.
 */
import { f as fDist } from './dist.js'
import { cholesky, lstsq, matrix, type Matrix } from './linalg.js'
import { modelMatrix, parseFormula, type Data } from './lm.js'

export interface ManovaResult {
  test: 'one-way MANOVA'
  n: number
  p: number
  k: number
  dfHypothesis: number
  dfError: number
  pillai: { statistic: number; f: number; df1: number; df2: number; pValue: number }
  wilks: { statistic: number; f: number; df1: number; df2: number; pValue: number }
  hotelling: { statistic: number; f: number; df1: number; df2: number; pValue: number }
  roy: { statistic: number; f: number; df1: number; df2: number; pValue: number }
  groupNames: string[]
}

function det(A: Matrix): number {
  const n = A.rows
  if (n !== A.cols) throw new RangeError('det: square matrix required')
  const M = matrix(n, n, A.data.slice())
  let d = 1
  for (let col = 0; col < n; col++) {
    let piv = col
    for (let r = col + 1; r < n; r++) if (Math.abs(M.data[r * n + col]!) > Math.abs(M.data[piv * n + col]!)) piv = r
    if (Math.abs(M.data[piv * n + col]!) < 1e-14) return 0
    if (piv !== col) {
      for (let j = 0; j < n; j++) {
        const tmp = M.data[col * n + j]!
        M.data[col * n + j] = M.data[piv * n + j]!
        M.data[piv * n + j] = tmp
      }
      d = -d
    }
    const diag = M.data[col * n + col]!
    d *= diag
    for (let r = col + 1; r < n; r++) {
      const f = M.data[r * n + col]! / diag
      for (let j = col; j < n; j++) M.data[r * n + j]! -= f * M.data[col * n + j]!
    }
  }
  return d
}

/** Eigenvalues of E⁻¹H computed exactly from the symmetric L⁻¹ H L⁻ᵀ with E = LLᵀ (Jacobi). */
function eigenvaluesOfHE(H: Matrix, E: Matrix): number[] {
  const p = H.rows
  let L = cholesky(E)
  if (!L) {
    const Er = matrix(p, p, E.data.slice())
    for (let i = 0; i < p; i++) Er.data[i * p + i]! += 1e-8 * (1 + Math.abs(Er.data[i * p + i]!))
    L = cholesky(Er)
    if (!L) throw new RangeError('manova: error SSCP matrix is singular')
  }
  // W = L⁻¹ H: solve L W = H column by column, then S = W L⁻ᵀ = (L⁻¹ Wᵀ)ᵀ
  const solveL = (M: Matrix): Matrix => {
    const out = matrix(p, p)
    for (let c = 0; c < p; c++) {
      for (let i = 0; i < p; i++) {
        let v = M.data[i * p + c]!
        for (let k = 0; k < i; k++) v -= L!.data[i * p + k]! * out.data[k * p + c]!
        out.data[i * p + c] = v / L!.data[i * p + i]!
      }
    }
    return out
  }
  const W = solveL(H)
  const Wt = matrix(p, p)
  for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) Wt.data[i * p + j] = W.data[j * p + i]!
  const S = solveL(Wt) // = L⁻¹ Wᵀ = (W L⁻ᵀ)ᵀ, symmetric
  return symmetricEigenvalues(S)
}

/** Eigenvalues of a symmetric matrix by cyclic Jacobi rotations (descending). */
function symmetricEigenvalues(S: Matrix): number[] {
  const p = S.rows
  const B = matrix(p, p, S.data.slice())
  for (let i = 0; i < p; i++) for (let j = i + 1; j < p; j++) {
    const m = 0.5 * (B.data[i * p + j]! + B.data[j * p + i]!)
    B.data[i * p + j] = m
    B.data[j * p + i] = m
  }
  for (let iter = 0; iter < 60 * p * p; iter++) {
    let max = 0
    let pI = 0
    let pJ = 1
    for (let i = 0; i < p; i++) for (let j = i + 1; j < p; j++) {
      const v = Math.abs(B.data[i * p + j]!)
      if (v > max) {
        max = v
        pI = i
        pJ = j
      }
    }
    if (max < 1e-13) break
    const app = B.data[pI * p + pI]!
    const aqq = B.data[pJ * p + pJ]!
    const apq = B.data[pI * p + pJ]!
    const tau = (aqq - app) / (2 * apq)
    const t = Math.sign(tau || 1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau))
    const c = 1 / Math.sqrt(1 + t * t)
    const sn = t * c
    for (let k = 0; k < p; k++) {
      const bik = B.data[pI * p + k]!
      const bjk = B.data[pJ * p + k]!
      B.data[pI * p + k] = c * bik - sn * bjk
      B.data[pJ * p + k] = sn * bik + c * bjk
    }
    for (let k = 0; k < p; k++) {
      const bki = B.data[k * p + pI]!
      const bkj = B.data[k * p + pJ]!
      B.data[k * p + pI] = c * bki - sn * bkj
      B.data[k * p + pJ] = sn * bki + c * bkj
    }
  }
  const eigs: number[] = []
  for (let i = 0; i < p; i++) eigs.push(Math.max(0, B.data[i * p + i]!))
  return eigs.sort((x, y) => y - x)
}

function fApprox(
  stat: number,
  kind: 'pillai' | 'wilks' | 'hotelling' | 'roy',
  p: number,
  dfH: number,
  dfE: number,
): { f: number; df1: number; df2: number; pValue: number } {
  const s = Math.min(p, dfH)
  const m = (Math.abs(p - dfH) - 1) / 2
  const n = (dfE - p - 1) / 2
  let f = NaN
  let df1 = s * p
  let df2 = s * (2 * n + s + 1)
  if (kind === 'pillai') {
    // V / (s - V) * (df2/df1)
    const V = stat
    df1 = s * (2 * m + s + 1)
    df2 = s * (2 * n + s + 1)
    f = (V / Math.max(1e-12, s - V)) * (df2 / df1)
  } else if (kind === 'wilks') {
    const Λ = Math.max(1e-300, Math.min(1, stat))
    const r = dfE - (p - dfH + 1) / 2
    const u = (p * dfH - 2) / 4
    const t = p * p * dfH * dfH - 4 < 0 ? 1 : Math.sqrt((p * p * dfH * dfH - 4) / (p * p + dfH * dfH - 5))
    df1 = p * dfH
    df2 = r * t - 2 * u
    f = ((1 - Math.pow(Λ, 1 / t)) / Math.max(1e-12, Math.pow(Λ, 1 / t))) * (df2 / df1)
  } else if (kind === 'hotelling') {
    const U = stat
    df1 = s * (2 * m + s + 1)
    df2 = 2 * (s * n + 1)
    f = (U / s) * (df2 / df1)
  } else {
    // Roy largest root → conservative F
    const θ = Math.max(0, Math.min(1 - 1e-12, stat))
    df1 = Math.max(p, dfH)
    df2 = dfE - df1 + dfH
    f = (θ / Math.max(1e-12, 1 - θ)) * (df2 / df1)
  }
  const pValue = Number.isFinite(f) && f > 0 ? fDist(df1, Math.max(1, df2)).sf(f) : NaN
  return { f, df1, df2: Math.max(1, df2), pValue }
}

/**
 * One-way MANOVA: `Y` is n×p responses, `group` length n.
 */
export function manova(
  Y: ArrayLike<ArrayLike<number>>,
  group: ArrayLike<string | number>,
): ManovaResult {
  const rows = Array.from(Y).map((r) => Array.from(r))
  const gArr = Array.from(group).map(String)
  if (rows.length !== gArr.length) throw new RangeError('manova: Y/group length mismatch')
  const n = rows.length
  if (n < 4) throw new RangeError('manova: need ≥4 observations')
  const p = rows[0]?.length ?? 0
  if (p < 1) throw new RangeError('manova: need ≥1 response')
  const keys = [...new Set(gArr)]
  const k = keys.length
  if (k < 2) throw new RangeError('manova: need ≥2 groups')
  const map = new Map(keys.map((key, i) => [key, i]))
  const idx = gArr.map((g) => map.get(g)!)
  const sizes = new Array(k).fill(0)
  for (const i of idx) sizes[i]++

  const grand = new Array(p).fill(0)
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) grand[j]! += rows[i]![j]!
  for (let j = 0; j < p; j++) grand[j]! /= n

  const gMean = Array.from({ length: k }, () => new Array(p).fill(0))
  for (let i = 0; i < n; i++) {
    const gi = idx[i]!
    for (let j = 0; j < p; j++) gMean[gi]![j]! += rows[i]![j]!
  }
  for (let g = 0; g < k; g++) for (let j = 0; j < p; j++) gMean[g]![j]! /= sizes[g]!

  const H = matrix(p, p)
  const E = matrix(p, p)
  for (let g = 0; g < k; g++) {
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) {
      const da = gMean[g]![a]! - grand[a]!
      const db = gMean[g]![b]! - grand[b]!
      H.data[a * p + b]! += sizes[g]! * da * db
    }
  }
  for (let i = 0; i < n; i++) {
    const gi = idx[i]!
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) {
      const da = rows[i]![a]! - gMean[gi]![a]!
      const db = rows[i]![b]! - gMean[gi]![b]!
      E.data[a * p + b]! += da * db
    }
  }

  const dfH = k - 1
  const dfE = n - k
  const HE = matrix(p, p)
  for (let i = 0; i < p * p; i++) HE.data[i] = H.data[i]! + E.data[i]!

  const eigs = eigenvaluesOfHE(H, E)
  const pillaiStat = eigs.reduce((s, λ) => s + λ / (1 + λ), 0)
  const wilksStat = eigs.reduce((s, λ) => s / (1 + λ), 1)
  const hotellingStat = eigs.reduce((s, λ) => s + λ, 0)
  const royStat = eigs[0]! / (1 + eigs[0]!)

  // Also verify Wilks via det ratio when possible
  const detE = det(E)
  const detHE = det(HE)
  const wilksDet = detHE > 0 && detE >= 0 ? detE / detHE : wilksStat

  const pillai = { statistic: pillaiStat, ...fApprox(pillaiStat, 'pillai', p, dfH, dfE) }
  const wilks = { statistic: wilksDet, ...fApprox(wilksDet, 'wilks', p, dfH, dfE) }
  const hotelling = { statistic: hotellingStat, ...fApprox(hotellingStat, 'hotelling', p, dfH, dfE) }
  const roy = { statistic: royStat, ...fApprox(royStat, 'roy', p, dfH, dfE) }

  return {
    test: 'one-way MANOVA',
    n,
    p,
    k,
    dfHypothesis: dfH,
    dfError: dfE,
    pillai,
    wilks,
    hotelling,
    roy,
    groupNames: keys,
  }
}

export interface ManovaTerm {
  term: string
  dfHypothesis: number
  pillai: { statistic: number; f: number; df1: number; df2: number; pValue: number }
  wilks: { statistic: number; f: number; df1: number; df2: number; pValue: number }
  hotelling: { statistic: number; f: number; df1: number; df2: number; pValue: number }
  roy: { statistic: number; f: number; df1: number; df2: number; pValue: number }
  /** Hypothesis SSCP matrix (p × p). */
  H: Matrix
}

export interface ManovaModelResult {
  test: 'general MANOVA'
  responses: string[]
  n: number
  p: number
  dfError: number
  /** Error SSCP matrix. */
  E: Matrix
  terms: ManovaTerm[]
  /** Univariate ANOVA F-tests per response and term (adjusted SS). */
  univariate: Array<{ term: string; response: string; f: number; pValue: number; adjSS: number }>
}

/**
 * General MANOVA (Minitab): several responses on a model formula over factors / covariates with effects
 * coding; each term is tested with the Type III hypothesis SSCP H = E(reduced) − E(full).
 *   manovaModel({ y1, y2, a, b }, ['y1', 'y2'], 'a*b')
 */
export function manovaModel(data: Data, responses: string[], rhs: string, options: { factors?: string[] } = {}): ManovaModelResult {
  if (responses.length < 1) throw new RangeError('manovaModel needs at least one response')
  const formula = parseFormula(`${responses[0]} ~ ${rhs}`)
  // one design for all responses: rows complete on every response
  const complete: number[] = []
  const n0 = data[responses[0]!]!.length
  for (let i = 0; i < n0; i++) if (responses.every((r) => typeof data[r]![i] === 'number' && Number.isFinite(data[r]![i] as number))) complete.push(i)
  const sub: Data = {}
  for (const k of Object.keys(data)) sub[k] = complete.map((i) => data[k]![i])
  const mm = modelMatrix(sub, formula, options)
  const X = mm.X
  const n = X.rows
  const q = X.cols
  const p = responses.length
  const Y = responses.map((r) => Float64Array.from(mm.keep, (i) => sub[r]![i] as number))
  const residualSscp = (cols: number[]): { E: Matrix; rank: number } => {
    const Xs = matrix(n, cols.length)
    for (let i = 0; i < n; i++) for (let k = 0; k < cols.length; k++) Xs.data[i * cols.length + k] = X.data[i * q + cols[k]!]!
    const resid = Y.map((y) => (cols.length ? lstsq(Xs, y).residuals : Float64Array.from(y)))
    const E = matrix(p, p)
    for (let a = 0; a < p; a++) for (let b = a; b < p; b++) {
      let sum = 0
      for (let i = 0; i < n; i++) sum += resid[a]![i]! * resid[b]![i]!
      E.data[a * p + b] = sum
      E.data[b * p + a] = sum
    }
    return { E, rank: cols.length ? lstsq(Xs, Y[0]!).rank : 0 }
  }
  const all = Array.from({ length: q }, (_, j) => j)
  const full = residualSscp(all)
  const dfError = n - full.rank
  const terms: ManovaTerm[] = []
  const univariate: ManovaModelResult['univariate'] = []
  for (const tc of mm.termColumns) {
    const drop = new Set(tc.cols)
    const red = residualSscp(all.filter((j) => !drop.has(j)))
    const dfH = full.rank - red.rank
    const H = matrix(p, p)
    for (let i = 0; i < p * p; i++) H.data[i] = red.E.data[i]! - full.E.data[i]!
    const eigs = eigenvaluesOfHE(H, full.E)
    const pillaiStat = eigs.reduce((acc, l) => acc + l / (1 + l), 0)
    const wilksStat = eigs.reduce((acc, l) => acc / (1 + l), 1)
    const hotellingStat = eigs.reduce((acc, l) => acc + l, 0)
    const royStat = eigs[0]! / (1 + eigs[0]!)
    terms.push({
      term: tc.term.name,
      dfHypothesis: dfH,
      pillai: { statistic: pillaiStat, ...fApprox(pillaiStat, 'pillai', p, dfH, dfError) },
      wilks: { statistic: wilksStat, ...fApprox(wilksStat, 'wilks', p, dfH, dfError) },
      hotelling: { statistic: hotellingStat, ...fApprox(hotellingStat, 'hotelling', p, dfH, dfError) },
      roy: { statistic: royStat, ...fApprox(royStat, 'roy', p, dfH, dfError) },
      H,
    })
    for (let r = 0; r < p; r++) {
      const adjSS = H.data[r * p + r]!
      const f = adjSS / dfH / (full.E.data[r * p + r]! / dfError)
      univariate.push({ term: tc.term.name, response: responses[r]!, f, pValue: fDist(dfH, dfError).sf(f), adjSS })
    }
  }
  return { test: 'general MANOVA', responses, n, p, dfError, E: full.E, terms, univariate }
}
