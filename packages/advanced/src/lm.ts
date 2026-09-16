/**
 * General Linear Model (Minitab Stat › ANOVA › General Linear Model): a formula `y ~ a*b + x` with
 * factors (effects / sum-to-zero coding, as Minitab's −1, 0, +1 coding), covariates, interactions and
 * polynomial terms; adjusted (Type III) and sequential (Type I) sums of squares, coefficient table,
 * S / R² / R²(adj), fitted means per factor level and residual diagnostics.
 */
import { f as fDist, t as tDist } from './dist.js'
import { inverse, lstsq, matrix, type Matrix } from './linalg.js'
import { vifFromFit, type AnovaRow } from './regression.js'

export type Data = Record<string, ArrayLike<unknown>>

export interface Term {
  /** Variables in the term (a:b → ['a', 'b']; x:x → ['x', 'x']). */
  vars: string[]
  name: string
}

export interface Formula {
  response: string
  terms: Term[]
  intercept: boolean
}

/** Parse `y ~ a + b*c + x:x − 1` into terms; `*` expands to main effects plus all interactions. */
/** Cap on factors joined by "*" in one term (the expansion has 2^k − 1 terms). */
const MAX_STAR_FACTORS = 12

export function parseFormula(formula: string): Formula {
  const m = formula.split('~')
  if (m.length !== 2) throw new RangeError(`formula must look like "y ~ a + b", got "${formula}"`)
  const response = m[0]!.trim()
  let rhs = m[1]!.replace(/\s+/g, '')
  let intercept = true
  rhs = rhs.replace(/[-−]1(?=\+|$)/g, () => {
    intercept = false
    return ''
  })
  rhs = rhs.replace(/^\+|\+$/g, '').replace(/\+\+/g, '+')
  const terms: Term[] = []
  const seen = new Set<string>()
  const push = (vars: string[]) => {
    const key = [...vars].sort().join(':')
    if (seen.has(key)) return
    seen.add(key)
    terms.push({ vars, name: vars.join('*') })
  }
  for (const piece of rhs.split('+')) {
    if (!piece || piece === '1') continue
    if (piece === '0') {
      intercept = false
      continue
    }
    const parts = piece.split('*').map((part) => part.split(':').flatMap(expandPower))
    const k = parts.length
    if (k > MAX_STAR_FACTORS) {
      throw new RangeError(`formula: "${piece}" crosses ${k} factors (2^${k} terms); at most ${MAX_STAR_FACTORS} are supported`)
    }
    // all non-empty subsets, main effects first, then 2-way, … (one pass; sorting by popcount keeps the Minitab order)
    const masks: number[] = []
    for (let mask = 1; mask < 1 << k; mask++) masks.push(mask)
    masks.sort((a, b) => countBits(a) - countBits(b) || a - b)
    for (const mask of masks) {
      const vars: string[] = []
      for (let i = 0; i < k; i++) if (mask & (1 << i)) vars.push(...parts[i]!)
      push(vars)
    }
  }
  if (!response) throw new RangeError('formula needs a response before "~"')
  return { response, terms, intercept }
}

function countBits(mask: number): number {
  let c = 0
  while (mask) {
    c += mask & 1
    mask >>= 1
  }
  return c
}

/** x^2 → x:x */
function expandPower(v: string): string[] {
  const m = /^(.+)\^(\d+)$/.exec(v)
  if (!m) return [v]
  return new Array(Number(m[2])).fill(m[1]!)
}

interface FactorInfo {
  name: string
  levels: string[]
}

export interface ModelMatrix {
  X: Matrix
  y: Float64Array
  keep: number[]
  omitted: number[]
  columnNames: string[]
  /** Column index ranges per term (after the constant). */
  termColumns: Array<{ term: Term; cols: number[] }>
  factors: Map<string, FactorInfo>
  intercept: boolean
}

function isFactorColumn(col: ArrayLike<unknown>): boolean {
  for (let i = 0; i < col.length; i++) {
    const v = col[i]
    if (v === null || v === undefined) continue
    return typeof v !== 'number'
  }
  return false
}

/** Effects-coded design matrix for a formula over the data columns (listwise-complete rows). */
export function modelMatrix(data: Data, formula: Formula | string, options: { factors?: string[] } = {}): ModelMatrix {
  const f = typeof formula === 'string' ? parseFormula(formula) : formula
  const vars = new Set<string>([f.response, ...f.terms.flatMap((t) => t.vars)])
  for (const v of vars) if (!(v in data)) throw new RangeError(`unknown column "${v}" in formula`)
  const n0 = data[f.response]!.length
  const explicit = new Set(options.factors ?? [])
  const factors = new Map<string, FactorInfo>()
  for (const v of vars) {
    if (v === f.response) continue
    const col = data[v]!
    if (col.length !== n0) throw new RangeError(`column "${v}" has ${col.length} rows, expected ${n0}`)
    if (explicit.has(v) || isFactorColumn(col)) factors.set(v, { name: v, levels: [] })
  }
  const keep: number[] = []
  const omitted: number[] = []
  const yCol = data[f.response]!
  for (let i = 0; i < n0; i++) {
    let ok = typeof yCol[i] === 'number' && Number.isFinite(yCol[i] as number)
    for (const v of vars) {
      if (!ok) break
      if (v === f.response) continue
      const val = data[v]![i]
      if (val === null || val === undefined || (!factors.has(v) && !(typeof val === 'number' && Number.isFinite(val)))) ok = false
    }
    ;(ok ? keep : omitted).push(i)
  }
  for (const info of factors.values()) {
    const col = data[info.name]!
    const numeric = keep.every((i) => typeof col[i] === 'number')
    const set = new Set<string>()
    for (const i of keep) set.add(String(col[i]))
    info.levels = [...set].sort(numeric ? (a, b) => Number(a) - Number(b) : undefined)
    if (info.levels.length < 2) throw new RangeError(`factor "${info.name}" has fewer than 2 levels`)
  }
  const n = keep.length
  // per-variable coded columns
  const coded = new Map<string, { names: string[]; cols: Float64Array[] }>()
  for (const v of vars) {
    if (v === f.response) continue
    const col = data[v]!
    const info = factors.get(v)
    if (!info) {
      coded.set(v, { names: [v], cols: [Float64Array.from(keep, (i) => col[i] as number)] })
      continue
    }
    const m = info.levels.length
    const idx = new Map(info.levels.map((l, i) => [l, i]))
    const cols: Float64Array[] = []
    const names: string[] = []
    for (let j = 0; j < m - 1; j++) {
      const c = new Float64Array(n)
      for (let r = 0; r < n; r++) {
        const li = idx.get(String(col[keep[r]!]))!
        c[r] = li === j ? 1 : li === m - 1 ? -1 : 0
      }
      cols.push(c)
      names.push(`${v}[${info.levels[j]}]`)
    }
    coded.set(v, { names, cols })
  }
  const columnNames: string[] = []
  const columns: Float64Array[] = []
  if (f.intercept) {
    columnNames.push('Constant')
    columns.push(new Float64Array(n).fill(1))
  }
  const termColumns: ModelMatrix['termColumns'] = []
  for (const term of f.terms) {
    const start = columns.length
    // cartesian product of the component codings
    let acc: Array<{ name: string; col: Float64Array }> = [{ name: '', col: new Float64Array(n).fill(1) }]
    for (const v of term.vars) {
      const c = coded.get(v)!
      const next: typeof acc = []
      for (const a of acc) {
        for (let j = 0; j < c.cols.length; j++) {
          const col = new Float64Array(n)
          for (let r = 0; r < n; r++) col[r] = a.col[r]! * c.cols[j]![r]!
          next.push({ name: a.name ? `${a.name}*${c.names[j]}` : c.names[j]!, col })
        }
      }
      acc = next
    }
    for (const a of acc) {
      columnNames.push(a.name)
      columns.push(a.col)
    }
    termColumns.push({ term, cols: Array.from({ length: columns.length - start }, (_, i) => start + i) })
  }
  const p = columns.length
  const X = matrix(n, p)
  for (let r = 0; r < n; r++) for (let j = 0; j < p; j++) X.data[r * p + j] = columns[j]![r]!
  return { X, y: Float64Array.from(keep, (i) => yCol[i] as number), keep, omitted, columnNames, termColumns, factors, intercept: f.intercept }
}

export interface LmTermRow {
  term: string
  df: number
  adjSS: number
  adjMS: number
  seqSS: number
  f: number
  pValue: number
}

export interface LmCoefficient {
  name: string
  coef: number
  se: number
  t: number
  pValue: number
  ci: [number, number]
  vif?: number
  aliased?: boolean
}

export interface LinearModelResult {
  test: 'general linear model'
  formula: Formula
  n: number
  /** ANOVA table with adjusted (Type III) SS per term, then Error and Total. */
  anova: { terms: LmTermRow[]; error: AnovaRow; total: AnovaRow; model: AnovaRow }
  coefficients: LmCoefficient[]
  s: number
  r2: number
  r2adj: number
  r2pred: number
  /** Fitted (LS) means per factor level for each main-effect factor: mean of fitted values at that level. */
  means: Record<string, Array<{ level: string; n: number; mean: number; fittedMean: number; se: number }>>
  fitted: Float64Array
  residuals: Float64Array
  standardizedResiduals: Float64Array
  leverage: Float64Array
  cooksD: Float64Array
  unusual: Array<{ index: number; y: number; fitted: number; residual: number; stdResidual: number; flags: string }>
  factors: Record<string, string[]>
  columnNames: string[]
  omitted: number[]
  design: Matrix
}

/**
 * Fit a general linear model.
 *   linearModel({ y, a, b, x }, 'y ~ a*b + x')          // a, b factors (non-numeric or listed in `factors`)
 */
export function linearModel(data: Data, formula: string, options: { factors?: string[]; confidence?: number } = {}): LinearModelResult {
  const confidence = options.confidence ?? 0.95
  const f = parseFormula(formula)
  const mm = modelMatrix(data, f, options)
  const { X, y, keep, columnNames } = mm
  const n = X.rows
  const pAll = X.cols
  if (n <= pAll - 0 && n <= 1) throw new RangeError('linearModel needs more observations than coefficients')
  const full = lstsq(X, y)
  const p = full.rank
  const dfe = n - p
  if (dfe < 1) throw new RangeError(`linearModel: no degrees of freedom for error (n = ${n}, rank = ${p})`)
  const sse = full.sse
  const mse = sse / dfe
  const s = Math.sqrt(mse)
  let ybar = 0
  for (let i = 0; i < n; i++) ybar += y[i]!
  ybar /= n
  let sst = 0
  if (mm.intercept) for (let i = 0; i < n; i++) sst += (y[i]! - ybar) ** 2
  else for (let i = 0; i < n; i++) sst += y[i]! ** 2
  const dft = n - (mm.intercept ? 1 : 0)
  const fitWithout = (drop: Set<number>): { sse: number; rank: number } => {
    const cols = Array.from({ length: pAll }, (_, j) => j).filter((j) => !drop.has(j))
    if (cols.length === 0) {
      return { sse: mm.intercept ? sst : (() => { let s2 = 0; for (let i = 0; i < n; i++) s2 += y[i]! ** 2; return s2 })(), rank: 0 }
    }
    const sub = matrix(n, cols.length)
    for (let i = 0; i < n; i++) for (let k = 0; k < cols.length; k++) sub.data[i * cols.length + k] = X.data[i * pAll + cols[k]!]!
    const r = lstsq(sub, y)
    return { sse: r.sse, rank: r.rank }
  }
  // adjusted (Type III) SS: for a full-rank fit, SS_T = b_Tᵀ [C_TT]⁻¹ b_T with C = (XᵀX)⁻¹ (identical to the
  // reduced-model refit, without refitting); sequential SS from Qᵀy of the ordered design. Rank-deficient
  // designs fall back to explicit refits.
  const fullRank = full.dependent.length === 0
  const terms: LmTermRow[] = []
  let prev = fullRank ? { sse: NaN, rank: 0 } : fitWithout(new Set(mm.termColumns.flatMap((t) => t.cols)))
  const cumulative = new Set(mm.termColumns.flatMap((t) => t.cols))
  for (const tc of mm.termColumns) {
    let adjSS: number
    let df: number
    let seqSS: number
    if (fullRank) {
      const q = tc.cols.length
      const C = matrix(q, q)
      for (let a = 0; a < q; a++) for (let b = 0; b < q; b++) C.data[a * q + b] = full.xtxInv.data[tc.cols[a]! * pAll + tc.cols[b]!]!
      const Ci = inverse(C)
      adjSS = 0
      for (let a = 0; a < q; a++) for (let b = 0; b < q; b++) adjSS += full.coef[tc.cols[a]!]! * Ci.data[a * q + b]! * full.coef[tc.cols[b]!]!
      adjSS = Math.max(0, adjSS)
      df = q
      seqSS = 0
      for (const c of tc.cols) seqSS += full.qty[c]! ** 2
    } else {
      const without = fitWithout(new Set(tc.cols))
      adjSS = Math.max(0, without.sse - sse)
      df = p - without.rank
      for (const c of tc.cols) cumulative.delete(c)
      const seq = fitWithout(cumulative)
      seqSS = Math.max(0, prev.sse - seq.sse)
      prev = seq
    }
    const adjMS = df > 0 ? adjSS / df : NaN
    const F = df > 0 ? adjMS / mse : NaN
    terms.push({ term: tc.term.name, df, adjSS, adjMS, seqSS, f: F, pValue: df > 0 ? fDist(df, dfe).sf(F) : NaN })
  }
  const dfModel = p - (mm.intercept ? 1 : 0)
  const ssModel = sst - sse
  const fModel = dfModel > 0 ? ssModel / dfModel / mse : NaN
  const anova = {
    terms,
    model: { source: 'Model', df: dfModel, ss: ssModel, ms: dfModel > 0 ? ssModel / dfModel : NaN, f: fModel, pValue: dfModel > 0 ? fDist(dfModel, dfe).sf(fModel) : NaN },
    error: { source: 'Error', df: dfe, ss: sse, ms: mse },
    total: { source: 'Total', df: dft, ss: sst, ms: NaN },
  }
  const td = tDist(dfe)
  const tc = td.ppf(0.5 + confidence / 2)
  const aliased = new Set(full.dependent)
  const coefficients: LmCoefficient[] = columnNames.map((name, j) => {
    if (aliased.has(j)) return { name, coef: 0, se: NaN, t: NaN, pValue: NaN, ci: [NaN, NaN], aliased: true }
    const se = Math.sqrt(full.xtxInv.data[j * pAll + j]! * mse)
    const tt = full.coef[j]! / se
    return { name, coef: full.coef[j]!, se, t: tt, pValue: Math.min(1, 2 * td.sf(Math.abs(tt))), ci: [full.coef[j]! - tc * se, full.coef[j]! + tc * se] }
  })
  // VIF for non-constant columns (from the full fit)
  if (mm.intercept && pAll > 2) {
    const vif = vifFromFit(X, full.xtxInv, 0, aliased)
    for (let j = 1; j < pAll; j++) if (!aliased.has(j)) coefficients[j]!.vif = vif[j]!
  }
  const std = new Float64Array(n)
  const cook = new Float64Array(n)
  let press = 0
  for (let i = 0; i < n; i++) {
    const h = full.leverage[i]!
    const e = full.residuals[i]!
    const d = 1 - h
    press += d > 1e-12 ? (e / d) ** 2 : 0
    std[i] = d > 1e-12 ? e / (s * Math.sqrt(d)) : NaN
    cook[i] = (std[i]! ** 2 * h) / (p * d)
  }
  const unusual: LinearModelResult['unusual'] = []
  for (let i = 0; i < n; i++) {
    let flags = ''
    if (Math.abs(std[i]!) > 2) flags += 'R'
    if (full.leverage[i]! > (3 * p) / n || full.leverage[i]! > 0.99) flags += 'X'
    if (flags) unusual.push({ index: keep[i]!, y: y[i]!, fitted: full.fitted[i]!, residual: full.residuals[i]!, stdResidual: std[i]!, flags })
  }
  // means per main-effect factor level
  const means: LinearModelResult['means'] = {}
  for (const [name, info] of mm.factors) {
    const col = data[name]!
    means[name] = info.levels.map((level) => {
      let cnt = 0
      let sy = 0
      let sf = 0
      for (let i = 0; i < n; i++) {
        if (String(col[keep[i]!]) !== level) continue
        cnt++
        sy += y[i]!
        sf += full.fitted[i]!
      }
      return { level, n: cnt, mean: cnt ? sy / cnt : NaN, fittedMean: cnt ? sf / cnt : NaN, se: cnt ? s / Math.sqrt(cnt) : NaN }
    })
  }
  return {
    test: 'general linear model',
    formula: f,
    n,
    anova,
    coefficients,
    s,
    r2: sst > 0 ? 1 - sse / sst : NaN,
    r2adj: sst > 0 ? 1 - mse / (sst / dft) : NaN,
    r2pred: sst > 0 ? 1 - press / sst : NaN,
    means,
    fitted: full.fitted,
    residuals: full.residuals,
    standardizedResiduals: std,
    leverage: full.leverage,
    cooksD: cook,
    unusual,
    factors: Object.fromEntries([...mm.factors].map(([k, v]) => [k, v.levels])),
    columnNames,
    omitted: mm.omitted,
    design: X,
  }
}
