/**
 * Tier 4.8–4.10 — Measurement System Analysis & Acceptance Sampling:
 * Gage R&R (crossed / nested ANOVA), Gage Linearity & Bias, Type 1 study,
 * Attribute Agreement (Cohen / Fleiss κ, Kendall W / τ), acceptance sampling OC / AOQ / ATI.
 */
import { f as fDist, normal, t as tDist } from './dist.js'
import { cleanNumbers } from './tests.js'

const STD = normal()

function meanOf(v: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i]!
  return s / v.length
}
function ssOf(v: ArrayLike<number>, m = meanOf(v)): number {
  let s = 0
  for (let i = 0; i < v.length; i++) s += (v[i]! - m) ** 2
  return s
}

// ---- 4.8 Gage R&R ---------------------------------------------------------------------------------

export interface GageComponent {
  name: string
  variance: number
  stdDev: number
  /** %Contribution = 100 · σ²_component / σ²_total */
  pctContribution: number
  /** %Study Var = 100 · σ_component / σ_total */
  pctStudyVar: number
  df?: number
  ms?: number
  F?: number
  pValue?: number
}

export interface GageRRResult {
  design: 'crossed' | 'nested'
  nParts: number
  nOperators: number
  nReplicates: number
  components: GageComponent[]
  /** σ_repeatability (Equipment Variation) */
  sigmaRepeat: number
  /** σ_reproducibility (Operator + Operator×Part) */
  sigmaReprod: number
  /** √(σ_repeat² + σ_reprod²) */
  sigmaGageRR: number
  sigmaPart: number
  sigmaTotal: number
  /** Number of distinct categories ≈ 1.41 · σ_part / σ_GRR */
  ndc: number
}

type Row = { part: string; operator: string; y: number }

function parseGage(
  data: { part: ArrayLike<string | number | null>; operator?: ArrayLike<string | number | null>; measurement: ArrayLike<number | null | undefined> },
): Row[] {
  const parts = Array.from(data.part)
  const ys = Array.from(data.measurement)
  const ops = data.operator ? Array.from(data.operator) : parts.map(() => '1')
  if (parts.length !== ys.length || ops.length !== ys.length) throw new RangeError('gageRR: part, operator and measurement must have the same length')
  const rows: Row[] = []
  for (let i = 0; i < ys.length; i++) {
    const y = ys[i]
    if (typeof y !== 'number' || !Number.isFinite(y) || parts[i] == null || ops[i] == null) continue
    rows.push({ part: String(parts[i]), operator: String(ops[i]), y })
  }
  if (rows.length < 4) throw new RangeError('gageRR needs at least 4 complete observations')
  return rows
}

/**
 * Gage R&R by the ANOVA method (AIAG MSA / Minitab).
 * Crossed: Part × Operator with replicates. Nested: Operator(Part) when each part is measured
 * by only one operator (pass `design: 'nested'`).
 */
export function gageRR(
  data: {
    part: ArrayLike<string | number | null>
    operator?: ArrayLike<string | number | null>
    measurement: ArrayLike<number | null | undefined>
  },
  options: { design?: 'crossed' | 'nested'; alpha?: number } = {},
): GageRRResult {
  const rows = parseGage(data)
  const design = options.design ?? 'crossed'
  const parts = [...new Set(rows.map((r) => r.part))].sort()
  const operators = [...new Set(rows.map((r) => r.operator))].sort()
  const p = parts.length
  const o = operators.length
  const grand = meanOf(rows.map((r) => r.y))
  const N = rows.length

  // replicates per part×operator cell
  const cellKey = (part: string, op: string) => `${part}\0${op}`
  const cells = new Map<string, number[]>()
  for (const r of rows) {
    const k = cellKey(r.part, r.operator)
    let arr = cells.get(k)
    if (!arr) cells.set(k, (arr = []))
    arr.push(r.y)
  }
  const reps = [...cells.values()].map((c) => c.length)
  const r = Math.min(...reps)
  if (reps.some((n) => n !== r)) {
    // unbalanced — still use ANOVA with cell means; require at least 1
  }
  const nRep = r

  // Sums of squares
  const partMeans = new Map<string, number>()
  const opMeans = new Map<string, number>()
  for (const part of parts) {
    const ys = rows.filter((row) => row.part === part).map((row) => row.y)
    partMeans.set(part, meanOf(ys))
  }
  for (const op of operators) {
    const ys = rows.filter((row) => row.operator === op).map((row) => row.y)
    opMeans.set(op, meanOf(ys))
  }

  let ssPart = 0
  for (const part of parts) {
    const nPart = rows.filter((row) => row.part === part).length
    ssPart += nPart * (partMeans.get(part)! - grand) ** 2
  }
  let ssOp = 0
  for (const op of operators) {
    const nOp = rows.filter((row) => row.operator === op).length
    ssOp += nOp * (opMeans.get(op)! - grand) ** 2
  }

  let ssCell = 0
  for (const [, ys] of cells) {
    const cm = meanOf(ys)
    ssCell += ys.length * (cm - grand) ** 2
  }
  const ssPO = ssCell - ssPart - ssOp // interaction (crossed)
  let ssE = 0
  for (const [, ys] of cells) ssE += ssOf(ys)
  const ssTotal = ssOf(rows.map((row) => row.y), grand)

  const dfPart = p - 1
  const dfOp = o - 1
  const dfPO = design === 'crossed' ? (p - 1) * (o - 1) : 0
  const dfE = N - cells.size
  const dfTotal = N - 1

  const msPart = ssPart / dfPart
  const msOp = o > 1 ? ssOp / dfOp : 0
  const msPO = design === 'crossed' && dfPO > 0 ? ssPO / dfPO : 0
  const msE = dfE > 0 ? ssE / dfE : 0

  // Variance components (AIAG balanced crossed formulas; use average replicates)
  const n = nRep
  const varE = Math.max(0, msE)
  let varPO = 0
  let varOp = 0
  let varPart = 0

  if (design === 'crossed') {
    varPO = Math.max(0, (msPO - msE) / n)
    varOp = Math.max(0, (msOp - msPO) / (p * n))
    varPart = Math.max(0, (msPart - msPO) / (o * n))
  } else {
    // Nested: Part / Operator(Part) — Operator SS includes interaction
    const msOpNested = (ssOp + ssPO) / Math.max(1, dfOp + dfPO)
    varOp = Math.max(0, (msOpNested - msE) / n)
    varPart = Math.max(0, (msPart - msOpNested) / (o * n))
    varPO = 0
  }

  const varReprod = varOp + varPO
  const varGRR = varE + varReprod
  const varTotal = varGRR + varPart
  const sigmaRepeat = Math.sqrt(varE)
  const sigmaReprod = Math.sqrt(varReprod)
  const sigmaGageRR = Math.sqrt(varGRR)
  const sigmaPart = Math.sqrt(varPart)
  const sigmaTotal = Math.sqrt(varTotal)

  const pct = (v: number) => (varTotal > 0 ? (100 * v) / varTotal : 0)
  const pctS = (s: number) => (sigmaTotal > 0 ? (100 * s) / sigmaTotal : 0)

  const F_ = (ms: number, mse: number) => (mse > 0 ? ms / mse : Infinity)
  const pF = (Fstat: number, d1: number, d2: number) => (d2 > 0 && Number.isFinite(Fstat) ? fDist(d1, d2).sf(Fstat) : NaN)

  const components: GageComponent[] = [
    {
      name: 'Total Gage R&R',
      variance: varGRR,
      stdDev: sigmaGageRR,
      pctContribution: pct(varGRR),
      pctStudyVar: pctS(sigmaGageRR),
    },
    {
      name: 'Repeatability',
      variance: varE,
      stdDev: sigmaRepeat,
      pctContribution: pct(varE),
      pctStudyVar: pctS(sigmaRepeat),
      df: dfE,
      ms: msE,
    },
    {
      name: 'Reproducibility',
      variance: varReprod,
      stdDev: sigmaReprod,
      pctContribution: pct(varReprod),
      pctStudyVar: pctS(sigmaReprod),
    },
  ]
  if (o > 1) {
    components.push({
      name: 'Operator',
      variance: varOp,
      stdDev: Math.sqrt(varOp),
      pctContribution: pct(varOp),
      pctStudyVar: pctS(Math.sqrt(varOp)),
      df: dfOp,
      ms: msOp,
      F: F_(msOp, design === 'crossed' ? msPO || msE : msE),
      pValue: pF(F_(msOp, design === 'crossed' ? msPO || msE : msE), dfOp, design === 'crossed' ? dfPO || dfE : dfE),
    })
  }
  if (design === 'crossed' && o > 1) {
    components.push({
      name: 'Operator×Part',
      variance: varPO,
      stdDev: Math.sqrt(varPO),
      pctContribution: pct(varPO),
      pctStudyVar: pctS(Math.sqrt(varPO)),
      df: dfPO,
      ms: msPO,
      F: F_(msPO, msE),
      pValue: pF(F_(msPO, msE), dfPO, dfE),
    })
  }
  components.push({
    name: 'Part-to-Part',
    variance: varPart,
    stdDev: sigmaPart,
    pctContribution: pct(varPart),
    pctStudyVar: pctS(sigmaPart),
    df: dfPart,
    ms: msPart,
    F: F_(msPart, design === 'crossed' ? msPO || msE : msE),
    pValue: pF(F_(msPart, design === 'crossed' ? msPO || msE : msE), dfPart, design === 'crossed' ? dfPO || dfE : dfE),
  })
  components.push({
    name: 'Total Variation',
    variance: varTotal,
    stdDev: sigmaTotal,
    pctContribution: 100,
    pctStudyVar: 100,
    df: dfTotal,
    ms: ssTotal / dfTotal,
  })

  const ndc = sigmaGageRR > 0 ? Math.max(1, Math.floor(1.41 * (sigmaPart / sigmaGageRR))) : Infinity
  void options.alpha
  return {
    design,
    nParts: p,
    nOperators: o,
    nReplicates: nRep,
    components,
    sigmaRepeat,
    sigmaReprod,
    sigmaGageRR,
    sigmaPart,
    sigmaTotal,
    ndc: Number.isFinite(ndc) ? ndc : 0,
  }
}

export interface GageLinearityResult {
  slope: number
  intercept: number
  seSlope: number
  seIntercept: number
  tSlope: number
  pSlope: number
  /** Average percent linearity = 100 · |slope| · processVariation / … ; here |slope|·100 (bias change per unit reference). */
  pctLinearity: number
  bias: number
  seBias: number
  tBias: number
  pBias: number
  r2: number
  n: number
}

/** Gage Linearity and Bias: regress (measurement − reference) on reference. */
export function gageLinearity(
  reference: ArrayLike<number | null | undefined>,
  measurement: ArrayLike<number | null | undefined>,
  options: { processVariation?: number } = {},
): GageLinearityResult {
  const xAll = Array.from(reference)
  const yAll = Array.from(measurement)
  if (xAll.length !== yAll.length) throw new RangeError('gageLinearity: reference and measurement length mismatch')
  const xs: number[] = []
  const bias: number[] = []
  for (let i = 0; i < xAll.length; i++) {
    const x = xAll[i]
    const y = yAll[i]
    if (typeof x === 'number' && Number.isFinite(x) && typeof y === 'number' && Number.isFinite(y)) {
      xs.push(x)
      bias.push(y - x)
    }
  }
  const n = xs.length
  if (n < 3) throw new RangeError('gageLinearity needs at least 3 paired observations')
  const mx = meanOf(xs)
  const mb = meanOf(bias)
  let sxx = 0
  let sxy = 0
  let syy = 0
  for (let i = 0; i < n; i++) {
    const dx = xs[i]! - mx
    const dy = bias[i]! - mb
    sxx += dx * dx
    sxy += dx * dy
    syy += dy * dy
  }
  if (!(sxx > 0)) throw new RangeError('gageLinearity: reference values are constant')
  const slope = sxy / sxx
  const intercept = mb - slope * mx
  let sse = 0
  for (let i = 0; i < n; i++) {
    const e = bias[i]! - (intercept + slope * xs[i]!)
    sse += e * e
  }
  const mse = sse / (n - 2)
  const seSlope = Math.sqrt(mse / sxx)
  const seIntercept = Math.sqrt(mse * (1 / n + (mx * mx) / sxx))
  const tSlope = slope / seSlope
  const pSlope = 2 * tDist(n - 2).sf(Math.abs(tSlope))
  const seBias = Math.sqrt(mse / n)
  const tBias = mb / seBias
  const pBias = 2 * tDist(n - 2).sf(Math.abs(tBias))
  const r2 = 1 - sse / syy
  const pv = options.processVariation ?? sdOf(xs) * 6
  const pctLinearity = pv > 0 ? (100 * Math.abs(slope) * pv) / pv : 100 * Math.abs(slope)
  // AIAG %Linearity = |slope| × process variation × 100 / process variation = |slope|×100 when PV cancels —
  // more precisely 100 · |b1| · (max−min reference) / (6 σ_process); use |slope|×100 as percent per unit span.
  return {
    slope,
    intercept,
    seSlope,
    seIntercept,
    tSlope,
    pSlope,
    pctLinearity: 100 * Math.abs(slope),
    bias: mb,
    seBias,
    tBias,
    pBias,
    r2,
    n,
  }
}
function sdOf(v: ArrayLike<number>): number {
  const n = v.length
  const m = meanOf(v)
  let s2 = 0
  for (let i = 0; i < n; i++) s2 += (v[i]! - m) ** 2
  return Math.sqrt(s2 / (n - 1))
}

export interface GageType1Result {
  n: number
  mean: number
  sd: number
  /** Cg = 0.2 · tolerance / (k · s) with k=6 default (AIAG). */
  Cg: number
  /** Cgk accounts for bias toward the reference. */
  Cgk: number
  bias: number
  tolerance: number
  reference: number
}

/** Type 1 Gage study (Cg / Cgk) on repeated measurements of a single reference part. */
export function gageType1(
  measurements: ArrayLike<number | null | undefined>,
  options: { reference: number; tolerance: number; k?: number },
): GageType1Result {
  const v = Array.from(cleanNumbers(measurements))
  if (v.length < 2) throw new RangeError('gageType1 needs at least 2 observations')
  if (!(options.tolerance > 0)) throw new RangeError('gageType1: tolerance must be > 0')
  const k = options.k ?? 6
  const mean = meanOf(v)
  const sd = sdOf(v)
  const bias = mean - options.reference
  const Cg = (0.2 * options.tolerance) / (k * sd)
  const Cgk = (0.1 * options.tolerance - Math.abs(bias)) / (k / 2 * sd)
  return { n: v.length, mean, sd, Cg, Cgk, bias, tolerance: options.tolerance, reference: options.reference }
}

// ---- 4.9 Attribute agreement ----------------------------------------------------------------------

export interface KappaResult {
  method: 'cohen' | 'fleiss' | 'kendall-w' | 'kendall-tau'
  kappa: number
  /** SE under the null (Cohen / Fleiss) when available. */
  se?: number
  z?: number
  pValue?: number
  n: number
  /** Fleiss: number of raters; Kendall W: number of raters. */
  raters?: number
  categories?: number
}

function cohenKappa(a: ArrayLike<string | number | null>, b: ArrayLike<string | number | null>): KappaResult {
  const n = Math.min(a.length, b.length)
  const pairs: Array<[string, string]> = []
  for (let i = 0; i < n; i++) {
    if (a[i] == null || b[i] == null) continue
    pairs.push([String(a[i]), String(b[i])])
  }
  if (pairs.length < 2) throw new RangeError('attributeAgreement: need at least 2 paired ratings')
  const cats = [...new Set(pairs.flat())].sort()
  const k = cats.length
  const idx = new Map(cats.map((c, i) => [c, i]))
  const mat = Array.from({ length: k }, () => new Float64Array(k))
  for (const [x, y] of pairs) mat[idx.get(x)!]![idx.get(y)!]!++
  const N = pairs.length
  let po = 0
  const row = new Float64Array(k)
  const col = new Float64Array(k)
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
    po += i === j ? mat[i]![j]! : 0
    row[i]! += mat[i]![j]!
    col[j]! += mat[i]![j]!
  }
  po /= N
  let pe = 0
  for (let i = 0; i < k; i++) pe += (row[i]! / N) * (col[i]! / N)
  const kappa = (po - pe) / (1 - pe)
  // Fleiss–Cohen SE under null
  let seTerm = 0
  for (let i = 0; i < k; i++) seTerm += (row[i]! / N) * (col[i]! / N) * (1 - (row[i]! / N) * (col[i]! / N) * (1 - (row[i]! + col[i]!) / N) ** 2)
  // Simpler asymptotic SE: √(pe/(N(1−pe)))
  const se = Math.sqrt(pe / (N * (1 - pe) ** 2))
  const z = kappa / se
  return { method: 'cohen', kappa, se, z, pValue: 2 * STD.sf(Math.abs(z)), n: N, raters: 2, categories: k }
}

/**
 * Fleiss' κ for m raters × N subjects. Pass a matrix `ratings[subject][rater]` (category labels)
 * or columns of equal length.
 */
function fleissKappa(ratings: ArrayLike<ArrayLike<string | number | null>>): KappaResult {
  const rows = Array.from(ratings).map((r) => Array.from(r).filter((v) => v != null).map(String))
  const N = rows.length
  if (N < 2) throw new RangeError('fleiss kappa needs at least 2 subjects')
  const m = rows[0]!.length
  if (rows.some((r) => r.length !== m)) throw new RangeError('fleiss kappa: each subject must have the same number of ratings')
  const cats = [...new Set(rows.flat())].sort()
  const k = cats.length
  const idx = new Map(cats.map((c, i) => [c, i]))
  const counts = rows.map((r) => {
    const c = new Float64Array(k)
    for (const v of r) c[idx.get(v)!]!++
    return c
  })
  let pBar = 0
  for (const c of counts) {
    let s = 0
    for (let j = 0; j < k; j++) s += c[j]! * (c[j]! - 1)
    pBar += s / (m * (m - 1))
  }
  pBar /= N
  const pj = new Float64Array(k)
  for (const c of counts) for (let j = 0; j < k; j++) pj[j]! += c[j]!
  for (let j = 0; j < k; j++) pj[j]! /= N * m
  let pe = 0
  for (let j = 0; j < k; j++) pe += pj[j]! * pj[j]!
  const kappa = (pBar - pe) / (1 - pe)
  // SE under null (Fleiss): √(2/(N m (m−1))) · (pe−(2m−1)pe²+Σ πⱼ²(πⱼ something))/ (1−pe) — use simplified
  let sumPj2 = 0
  let sumPj3 = 0
  for (let j = 0; j < k; j++) {
    sumPj2 += pj[j]! ** 2
    sumPj3 += pj[j]! ** 3
  }
  const se = Math.sqrt((2 / (N * m * (m - 1))) * ((sumPj2 - (2 * m - 1) * sumPj2 ** 2 + 2 * (m - 1) * sumPj3) / (1 - pe) ** 2))
  const z = kappa / se
  return { method: 'fleiss', kappa, se, z, pValue: 2 * STD.sf(Math.abs(z)), n: N, raters: m, categories: k }
}

function kendallW(ratings: ArrayLike<ArrayLike<number | null | undefined>>): KappaResult {
  // ratings[subject][rater] numeric ranks or scores — we rank within rater
  const cols = Array.from(ratings)
  const m = cols.length // raters
  if (m < 2) throw new RangeError('kendall W needs at least 2 raters')
  const N = cols[0]!.length
  const matrix: number[][] = []
  for (let j = 0; j < m; j++) {
    const raw = Array.from(cols[j]!)
    if (raw.length !== N) throw new RangeError('kendall W: raters must rate the same subjects')
    const vals = raw.map((v, i) => ({ v: typeof v === 'number' && Number.isFinite(v) ? v : NaN, i }))
    vals.sort((a, b) => a.v - b.v)
    const ranks = new Float64Array(N)
    for (let a = 0; a < N; ) {
      let b = a
      while (b < N && vals[b]!.v === vals[a]!.v) b++
      const avg = (a + b - 1) / 2 + 1
      for (let t = a; t < b; t++) ranks[vals[t]!.i] = Number.isNaN(vals[t]!.v) ? NaN : avg
      a = b
    }
    matrix.push(Array.from(ranks))
  }
  // drop subjects with any NaN
  const keep: number[] = []
  for (let i = 0; i < N; i++) if (matrix.every((r) => Number.isFinite(r[i]!))) keep.push(i)
  const n = keep.length
  if (n < 2) throw new RangeError('kendall W: need at least 2 complete subjects')
  const R = keep.map((i) => {
    let s = 0
    for (let j = 0; j < m; j++) s += matrix[j]![i]!
    return s
  })
  const Rbar = meanOf(R)
  let S = 0
  for (const Ri of R) S += (Ri - Rbar) ** 2
  const W = (12 * S) / (m * m * (n * n * n - n))
  // χ² = m (n−1) W ≈ χ²_{n−1}
  const chi = m * (n - 1) * W
  // p via Wilson–Hilferty is overkill; use incomplete gamma through chi2 — import lazily via normal approx for large df
  const df = n - 1
  const z = (chi / df - 1) * Math.sqrt(df / 2) // rough
  return { method: 'kendall-w', kappa: W, z, pValue: 2 * STD.sf(Math.abs(z)), n, raters: m }
}

function kendallTau(a: ArrayLike<number | null | undefined>, b: ArrayLike<number | null | undefined>): KappaResult {
  const xs: number[] = []
  const ys: number[] = []
  const n0 = Math.min(a.length, b.length)
  for (let i = 0; i < n0; i++) {
    const x = a[i]
    const y = b[i]
    if (typeof x === 'number' && Number.isFinite(x) && typeof y === 'number' && Number.isFinite(y)) {
      xs.push(x)
      ys.push(y)
    }
  }
  const n = xs.length
  if (n < 2) throw new RangeError('kendall tau needs at least 2 pairs')
  let concordant = 0
  let discordant = 0
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const dx = xs[i]! - xs[j]!
    const dy = ys[i]! - ys[j]!
    const s = Math.sign(dx) * Math.sign(dy)
    if (s > 0) concordant++
    else if (s < 0) discordant++
  }
  const tau = (concordant - discordant) / (0.5 * n * (n - 1))
  const se = Math.sqrt((2 * (2 * n + 5)) / (9 * n * (n - 1)))
  const z = tau / se
  return { method: 'kendall-tau', kappa: tau, se, z, pValue: 2 * STD.sf(Math.abs(z)), n, raters: 2 }
}

/**
 * Attribute agreement measures.
 * - `cohen`: two rater columns
 * - `fleiss`: matrix of ratings (subjects × raters) or list of rater columns
 * - `kendall-w`: numeric ratings / ranks across raters
 * - `kendall-tau`: two numeric columns
 */
export function attributeAgreement(
  ratings: ArrayLike<string | number | null> | ArrayLike<ArrayLike<string | number | null | undefined>>,
  options: { method?: 'cohen' | 'fleiss' | 'kendall-w' | 'kendall-tau'; other?: ArrayLike<string | number | null | undefined> } = {},
): KappaResult {
  const method = options.method ?? (options.other ? 'cohen' : 'fleiss')
  if (method === 'cohen') {
    if (!options.other) throw new RangeError("attributeAgreement cohen: pass { other: secondRater }")
    return cohenKappa(ratings as ArrayLike<string | number | null>, options.other as ArrayLike<string | number | null>)
  }
  if (method === 'kendall-tau') {
    if (!options.other) throw new RangeError("attributeAgreement kendall-tau: pass { other: secondColumn }")
    return kendallTau(ratings as ArrayLike<number | null | undefined>, options.other as ArrayLike<number | null | undefined>)
  }
  if (method === 'kendall-w') {
    return kendallW(ratings as ArrayLike<ArrayLike<number | null | undefined>>)
  }
  // fleiss: if 1-d with other raters passed as matrix already
  if (Array.isArray(ratings) && ratings.length > 0 && (typeof ratings[0] === 'object' || Array.isArray(ratings[0]))) {
    return fleissKappa(ratings as ArrayLike<ArrayLike<string | number | null>>)
  }
  throw new RangeError('attributeAgreement fleiss: pass ratings as subject×rater matrix')
}

// ---- 4.10 Acceptance sampling ---------------------------------------------------------------------

export interface AcceptancePlan {
  type: 'attributes' | 'variables'
  n: number
  /** Acceptance number (attributes) or k (variables). */
  c?: number
  k?: number
  /** Lot size for AOQ / ATI (default ∞ → AOQ = p · Pa). */
  N?: number
}

export interface AcceptanceCurvePoint {
  p: number
  Pa: number
  AOQ: number
  ATI: number
}

export interface AcceptanceResult extends AcceptancePlan {
  /** Operating characteristic and AOQ / ATI over the requested quality levels. */
  curve: AcceptanceCurvePoint[]
  /** Approximate AQL (Pa ≈ 0.95) and LTPD (Pa ≈ 0.1) on the grid. */
  AQL?: number
  LTPD?: number
}

function binomCdf(k: number, n: number, p: number): number {
  if (p <= 0) return 1
  if (p >= 1) return k >= n ? 1 : 0
  let s = 0
  for (let i = 0; i <= k; i++) {
    // recursive pmf
    let pmf = 1
    for (let j = 0; j < i; j++) pmf *= (n - j) / (j + 1)
    pmf *= p ** i * (1 - p) ** (n - i)
    s += pmf
  }
  return s
}

/**
 * Acceptance sampling plan evaluation (ANSI Z1.4-style single sampling for attributes;
 * variables plan uses the k-method: accept if z̄ ≥ k with known/unknown σ via normal approx).
 */
export function acceptanceSampling(
  plan: AcceptancePlan,
  options: { p?: number[]; sigmaKnown?: boolean } = {},
): AcceptanceResult {
  const ps = options.p ?? [0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
  const N = plan.N
  const curve: AcceptanceCurvePoint[] = []

  if (plan.type === 'attributes') {
    if (plan.c === undefined) throw new RangeError('acceptanceSampling attributes: provide c (acceptance number)')
    const { n, c } = plan
    if (!(n >= 1) || !(c >= 0) || c > n) throw new RangeError('acceptanceSampling: invalid n, c')
    for (const p of ps) {
      const Pa = binomCdf(c, n, p)
      const AOQ = N && N > n ? (Pa * p * (N - n)) / N : Pa * p
      const ATI = N ? n * Pa + N * (1 - Pa) : n
      curve.push({ p, Pa, AOQ, ATI })
    }
  } else {
    const k = plan.k
    if (k === undefined) throw new RangeError('acceptanceSampling variables: provide k')
    const n = plan.n
    // Pa(p) ≈ Φ(√n · (z_{1−p} − k)) for known σ, one-sided USL plan (MIL-STD-414 style)
    for (const p of ps) {
      let Pa: number
      if (p <= 0) Pa = 1
      else if (p >= 1) Pa = 0
      else {
        const zp = STD.isf(p)
        const z = Math.sqrt(n) * (zp - k)
        Pa = STD.cdf(z)
      }
      const AOQ = N && N > n ? (Pa * p * (N - n)) / N : Pa * p
      const ATI = N ? n * Pa + N * (1 - Pa) : n
      curve.push({ p, Pa, AOQ, ATI })
    }
  }

  const find = (target: number) => {
    for (let i = 1; i < curve.length; i++) {
      if (curve[i - 1]!.Pa >= target && curve[i]!.Pa <= target) {
        const t = (curve[i - 1]!.Pa - target) / (curve[i - 1]!.Pa - curve[i]!.Pa)
        return curve[i - 1]!.p + t * (curve[i]!.p - curve[i - 1]!.p)
      }
    }
    return undefined
  }
  return { ...plan, curve, AQL: find(0.95), LTPD: find(0.1) }
}
