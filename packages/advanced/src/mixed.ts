/**
 * Linear / generalized linear mixed models — random intercept (+ optional slope for LMM).
 */
import { lgamma, normal, t as tDist } from './dist.js'
import { lstsq, matrix } from './linalg.js'

const STD = normal()

export interface MixedFixedEffect {
  name: string
  coefficient: number
  se: number
  t: number
  pValue: number
  ci: [number, number]
}

export interface MixedRanef {
  group: string
  blup: number
  slope?: number
}

export interface MixedModelResult {
  method: 'REML' | 'ML'
  fixed: MixedFixedEffect[]
  sigma: number
  sigmaRandom: number
  /** Second crossed random-intercept SD when `group2` is set. */
  sigmaRandom2?: number
  /** Inner nested random-intercept SD when `groupNested` is set. */
  sigmaNested?: number
  sigmaSlope?: number
  /** Corr(RI, RS) when slope is present (estimated). */
  rho?: number
  icc: number
  logLik: number
  n: number
  nGroups: number
  nGroups2?: number
  nNested?: number
  ranef: MixedRanef[]
  ranef2?: MixedRanef[]
  nestedRanef?: MixedRanef[]
  iterations: number
}

export interface GlmmResult {
  family: 'binomial' | 'poisson' | 'negbin'
  method: 'pql' | 'laplace' | 'agq'
  nAGQ?: number
  /** NB dispersion θ (Var = μ + μ²/θ). */
  theta?: number
  fixed: MixedFixedEffect[]
  sigmaRandom: number
  /** Random-slope SD when `slope` is set. */
  sigmaSlope?: number
  /** Corr(RI, RS); diagonal-G MVP keeps ≈0. */
  rho?: number
  ranef: MixedRanef[]
  logLik: number
  n: number
  nGroups: number
  iterations: number
}

function groupIndex(groups: ArrayLike<string | number>): { keys: string[]; idx: number[]; sizes: number[] } {
  const gArr = Array.from(groups).map(String)
  const keys = [...new Set(gArr)]
  const map = new Map(keys.map((k, i) => [k, i]))
  const idx = gArr.map((g) => map.get(g)!)
  const sizes = new Array(keys.length).fill(0)
  for (const i of idx) sizes[i]++
  return { keys, idx, sizes }
}

/**
 * Random-intercept (+ optional independent random slope) linear mixed model via iterative GLS.
 * Backward compatible: pass `group` for RI-only; or `slope` covariate for RI+RS.
 * Nested two-level RI: `group` = outer, `groupNested` = inner (IDs unique within outer).
 * Crossed RI: `group` + `group2` (independent intercepts).
 */
export function mixedModel(
  y: ArrayLike<number>,
  options: {
    fixed: ArrayLike<ArrayLike<number>>
    group: ArrayLike<string | number>
    /** Inner nesting factor (nested within `group`). */
    groupNested?: ArrayLike<string | number>
    /** Second crossed random-intercept factor (independent of `group`). */
    group2?: ArrayLike<string | number>
    /** Optional continuous covariate for random slope (correlated with intercept). */
    slope?: ArrayLike<number>
    names?: string[]
    reml?: boolean
    intercept?: boolean
    confidence?: number
    maxIter?: number
  },
): MixedModelResult {
  const yy = Array.from(y)
  const n = yy.length
  const Xraw = Array.from(options.fixed).map((r) => Array.from(r))
  if (Xraw.length !== n) throw new RangeError('mixedModel: fixed/y length mismatch')
  const intercept = options.intercept !== false
  const p0 = Xraw[0]?.length ?? 0
  const p = p0 + (intercept ? 1 : 0)
  const names =
    options.names ??
    (intercept
      ? ['(Intercept)', ...Array.from({ length: p0 }, (_, j) => `X${j + 1}`)]
      : Array.from({ length: p0 }, (_, j) => `X${j + 1}`))
  const reml = options.reml !== false
  const confidence = options.confidence ?? 0.95
  const maxIter = options.maxIter ?? 40
  const { keys, idx, sizes } = groupIndex(options.group)
  const g = keys.length
  if (g < 2) throw new RangeError('mixedModel: need ≥2 groups')
  const hasNested = options.groupNested != null
  const hasCrossed = options.group2 != null
  if (hasNested && hasCrossed) throw new RangeError('mixedModel: groupNested and group2 are mutually exclusive')
  const nestedRaw = hasNested ? Array.from(options.groupNested!).map(String) : null
  if (nestedRaw && nestedRaw.length !== n) throw new RangeError('mixedModel: groupNested length mismatch')
  // Unique nested keys = outer::inner
  const nestedLabels = nestedRaw
    ? nestedRaw.map((inn, i) => `${String(options.group[i])}::${inn}`)
    : null
  const nestedIdx = nestedLabels ? groupIndex(nestedLabels) : null
  const gN = nestedIdx?.keys.length ?? 0
  if (hasNested && gN < 2) throw new RangeError('mixedModel: need ≥2 nested groups')
  const crossedIdx = hasCrossed ? groupIndex(options.group2!) : null
  const g2 = crossedIdx?.keys.length ?? 0
  if (hasCrossed && g2 < 2) throw new RangeError('mixedModel: need ≥2 group2 levels')
  if (hasCrossed && options.group2!.length !== n) throw new RangeError('mixedModel: group2 length mismatch')
  const hasSlope = options.slope != null && !hasNested && !hasCrossed
  const slopeX = hasSlope ? Array.from(options.slope!) : null
  if (slopeX && slopeX.length !== n) throw new RangeError('mixedModel: slope length mismatch')
  if (options.slope != null && (hasNested || hasCrossed)) {
    throw new RangeError('mixedModel: slope not supported with groupNested/group2')
  }

  const X = matrix(n, p)
  for (let i = 0; i < n; i++) {
    let c = 0
    if (intercept) {
      X.data[i * p] = 1
      c = 1
    }
    for (let j = 0; j < p0; j++) X.data[i * p + c + j] = Xraw[i]![j]!
  }

  // Random intercept weight: V^{-1} application with λ = σu²/σ²
  const applyW = (
    lam: number,
    vec: Float64Array | number[],
    out: Float64Array,
    gIdx: number[],
    gSizes: number[],
    gCount: number,
  ) => {
    const sum = new Array(gCount).fill(0)
    for (let i = 0; i < n; i++) sum[gIdx[i]!] += vec[i]!
    for (let i = 0; i < n; i++) {
      const gi = gIdx[i]!
      const ni = gSizes[gi]!
      const theta = lam / (1 + ni * lam)
      out[i] = vec[i]! - theta * sum[gi]!
    }
  }

  const applyNestedThenOuter = (lamO: number, lamN: number, vec: Float64Array | number[], out: Float64Array) => {
    const mid = new Float64Array(n)
    if (nestedIdx) applyW(lamN, vec, mid, nestedIdx.idx, nestedIdx.sizes, gN)
    else {
      for (let i = 0; i < n; i++) mid[i] = vec[i]!
    }
    applyW(lamO, mid, out, idx, sizes, g)
  }

  const applyCrossed = (lam1: number, lam2: number, vec: Float64Array | number[], out: Float64Array) => {
    const mid = new Float64Array(n)
    applyW(lam1, vec, mid, idx, sizes, g)
    applyW(lam2, mid, out, crossedIdx!.idx, crossedIdx!.sizes, g2)
  }

  const applyV = (lam1: number, lamExtra: number, vec: Float64Array | number[], out: Float64Array) => {
    if (hasNested && nestedIdx) applyNestedThenOuter(lam1, lamExtra, vec, out)
    else if (hasCrossed && crossedIdx) applyCrossed(lam1, lamExtra, vec, out)
    else applyW(lam1, vec, out, idx, sizes, g)
  }

  const wy = new Float64Array(n)
  const WX = matrix(n, p)
  const col = new Float64Array(n)
  const wcol = new Float64Array(n)

  let lam = 1
  let lamN = hasNested ? 0.5 : 0
  let lam2 = hasCrossed ? 0.5 : 0
  let lamS = hasSlope ? 0.5 : 0
  let rhoEst = 0
  let beta = new Array(p).fill(0)
  let sigma2 = 1
  let logLik = -Infinity
  let iter = 0

  for (iter = 0; iter < maxIter; iter++) {
    // For RI+RS MVP: first peel slope RE approximately by centering within groups on slope residual,
    // then apply RI weight. Simple approach: work on y - Z_s û_s with MoM updates.
    const yWork = yy.slice()
    if (hasSlope && slopeX) {
      // OLS within groups of resid ~ slope for crude û_s
      const resid0 = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        let xb = 0
        for (let j = 0; j < p; j++) xb += beta[j]! * X.data[i * p + j]!
        resid0[i] = yy[i]! - xb
      }
      for (let gi = 0; gi < g; gi++) {
        let sxx = 0
        let sxy = 0
        for (let i = 0; i < n; i++) {
          if (idx[i] !== gi) continue
          const z = slopeX[i]!
          sxx += z * z
          sxy += z * resid0[i]!
        }
        const us = sxx > 1e-12 ? (lamS / (1 + lamS)) * (sxy / sxx) : 0
        for (let i = 0; i < n; i++) if (idx[i] === gi) yWork[i]! -= us * slopeX[i]!
      }
    }

    if (hasNested && nestedIdx) applyV(lam, lamN, yWork, wy)
    else if (hasCrossed) applyV(lam, lam2, yWork, wy)
    else applyW(lam, yWork, wy, idx, sizes, g)
    for (let j = 0; j < p; j++) {
      for (let i = 0; i < n; i++) col[i] = X.data[i * p + j]!
      if (hasNested && nestedIdx) applyV(lam, lamN, col, wcol)
      else if (hasCrossed) applyV(lam, lam2, col, wcol)
      else applyW(lam, col, wcol, idx, sizes, g)
      for (let i = 0; i < n; i++) WX.data[i * p + j] = wcol[i]!
    }
    const fit = lstsq(WX, wy)
    beta = Array.from(fit.coef)

    const resid = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let xb = 0
      for (let j = 0; j < p; j++) xb += beta[j]! * X.data[i * p + j]!
      resid[i] = yy[i]! - xb
    }

    let ssWithin = 0
    let ssBetween = 0
    const gMean = new Array(g).fill(0)
    for (let i = 0; i < n; i++) gMean[idx[i]!] += resid[i]!
    for (let j = 0; j < g; j++) gMean[j]! /= sizes[j]!
    let grand = 0
    for (let i = 0; i < n; i++) grand += resid[i]!
    grand /= n
    for (let i = 0; i < n; i++) {
      const d = resid[i]! - gMean[idx[i]!]!
      ssWithin += d * d
    }
    for (let j = 0; j < g; j++) {
      const d = gMean[j]! - grand
      ssBetween += sizes[j]! * d * d
    }

    let lamNew = lam
    let lamNNew = lamN
    let lam2New = lam2
    let sigma2New = sigma2

    if (hasNested && nestedIdx) {
      // Nested MoM: within-nested → σ²; nested-within-outer → σ_N²; between-outer → σ_O²
      const nMean = new Array(gN).fill(0)
      for (let i = 0; i < n; i++) nMean[nestedIdx.idx[i]!] += resid[i]!
      for (let j = 0; j < gN; j++) nMean[j]! /= nestedIdx.sizes[j]!
      let ssW = 0
      for (let i = 0; i < n; i++) {
        const d = resid[i]! - nMean[nestedIdx.idx[i]!]!
        ssW += d * d
      }
      sigma2New = ssW / Math.max(1, n - gN)
      let ssNested = 0
      for (let j = 0; j < gN; j++) {
        // map nested group to outer via first observation
        let oi = 0
        for (let i = 0; i < n; i++) if (nestedIdx.idx[i] === j) {
          oi = idx[i]!
          break
        }
        const d = nMean[j]! - gMean[oi]!
        ssNested += nestedIdx.sizes[j]! * d * d
      }
      const dfN = Math.max(1, gN - g)
      const msn = ssNested / dfN
      // average nested size within outer
      const nBar = n / gN
      const sigmaN2 = Math.max(0, (msn - sigma2New) / Math.max(1e-8, nBar))
      const n0 = (1 / (g - 1)) * (n - sizes.reduce((s, ni) => s + ni * ni, 0) / n)
      const msb = ssBetween / Math.max(1, g - 1)
      const sigmaU2 = Math.max(0, (msb - msn) / Math.max(1e-8, n0))
      lamNew = sigma2New > 0 ? sigmaU2 / sigma2New : lam
      lamNNew = sigma2New > 0 ? sigmaN2 / sigma2New : lamN
    } else if (hasCrossed && crossedIdx) {
      // Crossed MoM: within both factors ≈ σ²; between each factor → σ1², σ2²
      const g2Mean = new Array(g2).fill(0)
      for (let i = 0; i < n; i++) g2Mean[crossedIdx.idx[i]!] += resid[i]!
      for (let j = 0; j < g2; j++) g2Mean[j]! /= crossedIdx.sizes[j]!
      let ssW = 0
      for (let i = 0; i < n; i++) {
        const pred = gMean[idx[i]!]! + g2Mean[crossedIdx.idx[i]!]! - grand
        const d = resid[i]! - pred
        ssW += d * d
      }
      sigma2New = ssW / Math.max(1, n - g - g2 + 1)
      const n0 = (1 / (g - 1)) * (n - sizes.reduce((s, ni) => s + ni * ni, 0) / n)
      const msb1 = ssBetween / Math.max(1, g - 1)
      let ssB2 = 0
      for (let j = 0; j < g2; j++) {
        const d = g2Mean[j]! - grand
        ssB2 += crossedIdx.sizes[j]! * d * d
      }
      const n02 = (1 / (g2 - 1)) * (n - crossedIdx.sizes.reduce((s, ni) => s + ni * ni, 0) / n)
      const msb2 = ssB2 / Math.max(1, g2 - 1)
      const sigmaU2 = Math.max(0, (msb1 - sigma2New) / Math.max(1e-8, n0))
      const sigmaU22 = Math.max(0, (msb2 - sigma2New) / Math.max(1e-8, n02))
      lamNew = sigma2New > 0 ? sigmaU2 / sigma2New : lam
      lam2New = sigma2New > 0 ? sigmaU22 / sigma2New : lam2
    } else {
      const dfW = n - g
      sigma2New = ssWithin / Math.max(1, dfW)
      const n0 = (1 / (g - 1)) * (n - sizes.reduce((s, ni) => s + ni * ni, 0) / n)
      const msb = ssBetween / Math.max(1, g - 1)
      const sigmaU2 = Math.max(0, (msb - sigma2New) / Math.max(1e-8, n0))
      lamNew = sigma2New > 0 ? sigmaU2 / sigma2New : lam
    }

    let lamSNew = lamS
    let rhoMom = 0
    if (hasSlope && slopeX) {
      // MoM slope variance + cov(RI, RS) from within-group OLS
      let ssSlope = 0
      let covUS = 0
      let ssU = 0
      let dfS = 0
      for (let gi = 0; gi < g; gi++) {
        let sxx = 0
        let sxy = 0
        let ngi = 0
        for (let i = 0; i < n; i++) {
          if (idx[i] !== gi) continue
          const z = slopeX[i]!
          const r = resid[i]! - gMean[gi]!
          sxx += z * z
          sxy += z * r
          ngi++
        }
        if (sxx > 1e-8 && ngi > 2) {
          const b = sxy / sxx
          const u0 = gMean[gi]!
          ssSlope += b * b
          covUS += u0 * b
          ssU += u0 * u0
          dfS++
        }
      }
      const sigmaS2 = dfS > 0 ? ssSlope / dfS : 0
      lamSNew = sigma2New > 0 ? Math.max(0, sigmaS2 / sigma2New) : lamS
      if (dfS > 1 && ssU > 1e-12 && ssSlope > 1e-12) {
        rhoMom = Math.max(-0.99, Math.min(0.99, covUS / Math.sqrt(ssU * ssSlope)))
      }
    }
    if (hasSlope) rhoEst = 0.7 * rhoEst + 0.3 * rhoMom

    let logDet = 0
    for (let j = 0; j < g; j++) logDet += Math.log(1 + sizes[j]! * lamNew)
    if (nestedIdx) {
      for (let j = 0; j < gN; j++) logDet += Math.log(1 + nestedIdx.sizes[j]! * lamNNew)
    }
    if (crossedIdx) {
      for (let j = 0; j < g2; j++) logDet += Math.log(1 + crossedIdx.sizes[j]! * lam2New)
    }
    const rss = (() => {
      const wr = new Float64Array(n)
      if (hasNested && nestedIdx) applyV(lamNew, lamNNew, resid, wr)
      else if (hasCrossed) applyV(lamNew, lam2New, resid, wr)
      else applyW(lamNew, resid, wr, idx, sizes, g)
      let s = 0
      for (let i = 0; i < n; i++) s += resid[i]! * wr[i]!
      return s
    })()
    const sigma2Prof = reml ? rss / (n - p) : rss / n
    const ll =
      -0.5 *
      (n * Math.log(2 * Math.PI) +
        n * Math.log(Math.max(sigma2Prof, 1e-300)) +
        logDet +
        rss / Math.max(sigma2Prof, 1e-300))

    const delta =
      Math.abs(lamNew - lam) +
      Math.abs(sigma2New - sigma2) +
      Math.abs(lamSNew - lamS) +
      Math.abs(lamNNew - lamN) +
      Math.abs(lam2New - lam2)
    lam = 0.5 * lam + 0.5 * lamNew
    lamN = 0.5 * lamN + 0.5 * lamNNew
    lam2 = 0.5 * lam2 + 0.5 * lam2New
    lamS = 0.5 * lamS + 0.5 * lamSNew
    sigma2 = sigma2Prof
    logLik = ll
    if (delta < 1e-8) break
  }

  const sigmaU = Math.sqrt(Math.max(0, lam * sigma2))
  const sigmaU2 = hasCrossed ? Math.sqrt(Math.max(0, lam2 * sigma2)) : undefined
  const sigmaNested = hasNested ? Math.sqrt(Math.max(0, lamN * sigma2)) : undefined
  const sigmaS = hasSlope ? Math.sqrt(Math.max(0, lamS * sigma2)) : undefined
  const sigma = Math.sqrt(Math.max(0, sigma2))
  const icc =
    sigmaU * sigmaU /
    (sigmaU * sigmaU +
      (sigmaU2 ? sigmaU2 * sigmaU2 : 0) +
      (sigmaNested ? sigmaNested * sigmaNested : 0) +
      sigma * sigma ||
      1)

  if (hasNested && nestedIdx) applyV(lam, lamN, yy, wy)
  else if (hasCrossed) applyV(lam, lam2, yy, wy)
  else applyW(lam, yy, wy, idx, sizes, g)
  for (let j = 0; j < p; j++) {
    for (let i = 0; i < n; i++) col[i] = X.data[i * p + j]!
    if (hasNested && nestedIdx) applyV(lam, lamN, col, wcol)
    else if (hasCrossed) applyV(lam, lam2, col, wcol)
    else applyW(lam, col, wcol, idx, sizes, g)
    for (let i = 0; i < n; i++) WX.data[i * p + j] = wcol[i]!
  }
  const fit = lstsq(WX, wy)
  beta = Array.from(fit.coef)
  const xtxInv = fit.xtxInv
  const df = Math.max(1, n - p)
  const tCrit = tDist(df).ppf(0.5 + confidence / 2)
  const fixed: MixedFixedEffect[] = []
  for (let j = 0; j < p; j++) {
    const se = Math.sqrt(Math.max(0, sigma2 * xtxInv.data[j * p + j]!))
    const t = se > 0 ? beta[j]! / se : NaN
    const pValue = Number.isFinite(t) ? 2 * tDist(df).sf(Math.abs(t)) : NaN
    fixed.push({
      name: names[j]!,
      coefficient: beta[j]!,
      se,
      t,
      pValue,
      ci: [beta[j]! - tCrit * se, beta[j]! + tCrit * se],
    })
  }

  const resid = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let xb = 0
    for (let j = 0; j < p; j++) xb += beta[j]! * X.data[i * p + j]!
    resid[i] = yy[i]! - xb
  }
  const gMean = new Array(g).fill(0)
  for (let i = 0; i < n; i++) gMean[idx[i]!] += resid[i]!
  for (let j = 0; j < g; j++) gMean[j]! /= sizes[j]!

  const ranef: MixedRanef[] = keys.map((group, j) => {
    const ni = sizes[j]!
    const blup = ((ni * lam) / (1 + ni * lam)) * gMean[j]!
    let slopeBlup: number | undefined
    if (hasSlope && slopeX) {
      let sxx = 0
      let sxy = 0
      for (let i = 0; i < n; i++) {
        if (idx[i] !== j) continue
        const z = slopeX[i]!
        sxx += z * z
        sxy += z * (resid[i]! - gMean[j]!)
      }
      const bhat = sxx > 1e-12 ? sxy / sxx : 0
      slopeBlup = (lamS / (1 + lamS)) * bhat
    }
    return { group, blup, slope: slopeBlup }
  })

  let nestedRanef: MixedRanef[] | undefined
  if (nestedIdx) {
    const nMean = new Array(gN).fill(0)
    for (let i = 0; i < n; i++) nMean[nestedIdx.idx[i]!] += resid[i]!
    for (let j = 0; j < gN; j++) nMean[j]! /= nestedIdx.sizes[j]!
    nestedRanef = nestedIdx.keys.map((group, j) => {
      let oi = 0
      for (let i = 0; i < n; i++) if (nestedIdx.idx[i] === j) {
        oi = idx[i]!
        break
      }
      const ni = nestedIdx.sizes[j]!
      const blup = ((ni * lamN) / (1 + ni * lamN)) * (nMean[j]! - gMean[oi]!)
      return { group, blup }
    })
  }

  let ranef2: MixedRanef[] | undefined
  if (crossedIdx) {
    const g2Mean = new Array(g2).fill(0)
    for (let i = 0; i < n; i++) g2Mean[crossedIdx.idx[i]!] += resid[i]!
    for (let j = 0; j < g2; j++) g2Mean[j]! /= crossedIdx.sizes[j]!
    ranef2 = crossedIdx.keys.map((group, j) => {
      const ni = crossedIdx.sizes[j]!
      const blup = ((ni * lam2) / (1 + ni * lam2)) * g2Mean[j]!
      return { group, blup }
    })
  }

  return {
    method: reml ? 'REML' : 'ML',
    fixed,
    sigma,
    sigmaRandom: sigmaU,
    sigmaRandom2: sigmaU2,
    sigmaNested,
    sigmaSlope: sigmaS,
    rho: hasSlope ? rhoEst : undefined,
    icc,
    logLik,
    n,
    nGroups: g,
    nGroups2: hasCrossed ? g2 : undefined,
    nNested: hasNested ? gN : undefined,
    ranef,
    ranef2,
    nestedRanef,
    iterations: iter + 1,
  }
}

/**
 * GLMM with random intercept (optional random slope) via PQL / Laplace / AGQ.
 * Families: binomial (logit), Poisson (log), or negbin (log).
 * AGQ+slope uses 2D Gauss–Hermite with diagonal G (ρ≈0).
 */
export function glmm(
  y: ArrayLike<number>,
  options: {
    family: 'binomial' | 'poisson' | 'negbin'
    fixed: ArrayLike<ArrayLike<number>>
    group: ArrayLike<string | number>
    /** Optional continuous covariate for random slope (AGQ/Laplace/PQL). */
    slope?: ArrayLike<number>
    trials?: ArrayLike<number>
    names?: string[]
    intercept?: boolean
    confidence?: number
    maxIter?: number
    /** Estimation method; default PQL. Laplace = mode + Hessian; AGQ = Gauss–Hermite (≥5). */
    method?: 'pql' | 'laplace' | 'agq'
    /** Quadrature points for AGQ (default 5). */
    nAGQ?: number
  },
): GlmmResult {
  const family = options.family
  const estMethod = options.method ?? 'pql'
  const nAGQ = Math.max(5, Math.min(15, options.nAGQ ?? 5))
  // AGQ uses Laplace mode as start
  const useLaplace = estMethod === 'laplace' || estMethod === 'agq'
  const yy = Array.from(y)
  const n = yy.length
  const Xraw = Array.from(options.fixed).map((r) => Array.from(r))
  if (Xraw.length !== n) throw new RangeError('glmm: fixed/y length mismatch')
  const hasSlope = options.slope != null
  const slopeX = hasSlope ? Array.from(options.slope!) : null
  if (slopeX && slopeX.length !== n) throw new RangeError('glmm: slope length mismatch')
  const trials = options.trials ? Array.from(options.trials) : yy.map(() => 1)
  if (trials.length !== n) throw new RangeError('glmm: trials length mismatch')
  const intercept = options.intercept !== false
  const p0 = Xraw[0]?.length ?? 0
  const p = p0 + (intercept ? 1 : 0)
  const names =
    options.names ??
    (intercept
      ? ['(Intercept)', ...Array.from({ length: p0 }, (_, j) => `X${j + 1}`)]
      : Array.from({ length: p0 }, (_, j) => `X${j + 1}`))
  const confidence = options.confidence ?? 0.95
  const maxIter = options.maxIter ?? 30
  const { keys, idx } = groupIndex(options.group)
  const g = keys.length
  if (g < 2) throw new RangeError('glmm: need ≥2 groups')

  const X = matrix(n, p)
  for (let i = 0; i < n; i++) {
    let c = 0
    if (intercept) {
      X.data[i * p] = 1
      c = 1
    }
    for (let j = 0; j < p0; j++) X.data[i * p + c + j] = Xraw[i]![j]!
  }

  let beta = new Array(p).fill(0)
  const u = new Array(g).fill(0)
  const sBlup = new Array(g).fill(0)
  let sigmaU2 = 0.5
  let sigmaS2 = hasSlope ? 0.25 : 0
  let rhoEst = 0
  let theta = 1 // NB dispersion
  let iter = 0

  const linkInv = (eta: number) => {
    if (family === 'binomial') {
      const e = Math.exp(Math.max(-20, Math.min(20, eta)))
      return e / (1 + e)
    }
    return Math.exp(Math.max(-20, Math.min(20, eta)))
  }
  const variance = (mu: number, m: number) => {
    if (family === 'binomial') return Math.max(1e-8, m * mu * (1 - mu))
    if (family === 'negbin') return Math.max(1e-8, mu + (mu * mu) / Math.max(1e-6, theta))
    return Math.max(1e-8, mu)
  }
  const reAt = (i: number) => u[idx[i]!]! + (hasSlope && slopeX ? sBlup[idx[i]!]! * slopeX[i]! : 0)

  for (iter = 0; iter < maxIter; iter++) {
    // working response and weights
    const z = new Float64Array(n)
    const w = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let eta = reAt(i)
      for (let j = 0; j < p; j++) eta += beta[j]! * X.data[i * p + j]!
      const mu = linkInv(eta)
      const m = trials[i]!
      if (family === 'binomial') {
        const yi = yy[i]! / m
        z[i] = eta + (yi - mu) / Math.max(1e-8, mu * (1 - mu))
        w[i] = Math.max(1e-8, m * mu * (1 - mu))
      } else if (family === 'negbin') {
        const v = variance(mu, m)
        z[i] = eta + (yy[i]! - mu) / Math.max(1e-8, mu)
        w[i] = Math.max(1e-8, (mu * mu) / v)
      } else {
        z[i] = eta + (yy[i]! - mu) / Math.max(1e-8, mu)
        w[i] = Math.max(1e-8, mu)
      }
    }

    // Weighted OLS on z after peeling RE
    const zw = new Float64Array(n)
    const Xw = matrix(n, p)
    for (let i = 0; i < n; i++) {
      const sw = Math.sqrt(w[i]!)
      zw[i] = (z[i]! - reAt(i)) * sw
      for (let j = 0; j < p; j++) Xw.data[i * p + j] = X.data[i * p + j]! * sw
    }
    const fit = lstsq(Xw, zw)
    beta = Array.from(fit.coef)

    // Update u_i (and slope) from working residuals
    const resid = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let eta = 0
      for (let j = 0; j < p; j++) eta += beta[j]! * X.data[i * p + j]!
      resid[i] = z[i]! - eta
    }
    const lam = sigmaU2
    for (let gi = 0; gi < g; gi++) {
      let s = 0
      let sw = 0
      for (let i = 0; i < n; i++) {
        if (idx[i] !== gi) continue
        const pe = resid[i]! - (hasSlope && slopeX ? sBlup[gi]! * slopeX[i]! : 0)
        s += w[i]! * pe
        sw += w[i]!
      }
      u[gi] = (lam * s) / (1 + lam * sw)
    }
    if (hasSlope && slopeX) {
      const lamS = sigmaS2
      for (let gi = 0; gi < g; gi++) {
        let sz = 0
        let szz = 0
        for (let i = 0; i < n; i++) {
          if (idx[i] !== gi) continue
          const pe = resid[i]! - u[gi]!
          const zi = slopeX[i]!
          sz += w[i]! * zi * pe
          szz += w[i]! * zi * zi
        }
        sBlup[gi] = (lamS * sz) / (1 + lamS * Math.max(1e-8, szz))
      }
      let ssS = 0
      let covUS = 0
      for (let gi = 0; gi < g; gi++) {
        ssS += sBlup[gi]! * sBlup[gi]!
        covUS += u[gi]! * sBlup[gi]!
      }
      sigmaS2 = Math.max(1e-6, 0.7 * sigmaS2 + 0.3 * (ssS / g))
      const ssUtmp = u.reduce((a, b) => a + b * b, 0)
      if (ssUtmp > 0 && ssS > 0) {
        const rMom = Math.max(-0.95, Math.min(0.95, covUS / Math.sqrt(ssUtmp * ssS)))
        rhoEst = 0.7 * rhoEst + 0.3 * rMom
      }
    }
    let ssU = 0
    for (let gi = 0; gi < g; gi++) ssU += u[gi]! * u[gi]!
    sigmaU2 = Math.max(1e-6, 0.7 * sigmaU2 + 0.3 * (ssU / g))

    if (family === 'negbin') {
      let num = 0
      let den = 0
      for (let i = 0; i < n; i++) {
        let eta = reAt(i)
        for (let j = 0; j < p; j++) eta += beta[j]! * X.data[i * p + j]!
        const mu = Math.max(1e-8, linkInv(eta))
        const pearson = (yy[i]! - mu) ** 2
        num += mu * mu
        den += Math.max(1e-8, pearson - mu)
      }
      const th = den > 0 ? num / den : theta
      theta = Math.max(0.05, 0.7 * theta + 0.3 * th)
    }
  }

  // Laplace: refine RE modes
  if (useLaplace) {
    for (let gi = 0; gi < g; gi++) {
      let ui = u[gi]!
      let si = hasSlope ? sBlup[gi]! : 0
      for (let newt = 0; newt < 12; newt++) {
        let scoreU = -ui / Math.max(1e-8, sigmaU2)
        let hessU = -1 / Math.max(1e-8, sigmaU2)
        let scoreS = hasSlope ? -si / Math.max(1e-8, sigmaS2) : 0
        let hessS = hasSlope ? -1 / Math.max(1e-8, sigmaS2) : -1
        for (let i = 0; i < n; i++) {
          if (idx[i] !== gi) continue
          const zi = hasSlope && slopeX ? slopeX[i]! : 0
          let eta = ui + si * zi
          for (let j = 0; j < p; j++) eta += beta[j]! * X.data[i * p + j]!
          const mu = linkInv(eta)
          const m = trials[i]!
          let dldeta: number
          let d2: number
          if (family === 'binomial') {
            const pmu = Math.min(1 - 1e-12, Math.max(1e-12, mu))
            dldeta = yy[i]! - m * pmu
            d2 = -m * pmu * (1 - pmu)
          } else if (family === 'negbin') {
            const th = Math.max(1e-6, theta)
            dldeta = (yy[i]! - mu) / (1 + mu / th)
            d2 = -(mu * th) / (th + mu)
          } else {
            dldeta = yy[i]! - mu
            d2 = -mu
          }
          scoreU += dldeta
          hessU += d2
          if (hasSlope) {
            scoreS += dldeta * zi
            hessS += d2 * zi * zi
          }
        }
        const stepU = scoreU / Math.min(-1e-8, hessU)
        ui += Math.max(-2, Math.min(2, stepU))
        if (hasSlope) {
          const stepS = scoreS / Math.min(-1e-8, hessS)
          si += Math.max(-2, Math.min(2, stepS))
        }
        if (Math.abs(stepU) < 1e-8 && (!hasSlope || Math.abs(scoreS / Math.min(-1e-8, hessS)) < 1e-8)) break
      }
      u[gi] = ui
      if (hasSlope) sBlup[gi] = si
    }
    let ssU = 0
    let ssS = 0
    for (let gi = 0; gi < g; gi++) {
      ssU += u[gi]! * u[gi]!
      if (hasSlope) ssS += sBlup[gi]! * sBlup[gi]!
    }
    sigmaU2 = Math.max(1e-6, ssU / g)
    if (hasSlope) sigmaS2 = Math.max(1e-6, ssS / g)
  }

  // Final SE from weighted X'WX
  const WX = matrix(n, p)
  const wy = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let eta = reAt(i)
    for (let j = 0; j < p; j++) eta += beta[j]! * X.data[i * p + j]!
    const mu = linkInv(eta)
    const m = trials[i]!
    let wt: number
    if (family === 'binomial') wt = Math.max(1e-8, m * mu * (1 - mu))
    else if (family === 'negbin') {
      const v = variance(mu, m)
      wt = Math.max(1e-8, (mu * mu) / v)
    } else wt = Math.max(1e-8, mu)
    const sw = Math.sqrt(wt)
    for (let j = 0; j < p; j++) WX.data[i * p + j] = X.data[i * p + j]! * sw
    const yi = family === 'binomial' ? yy[i]! / m : yy[i]!
    const mustar = mu
    const z_i =
      family === 'binomial'
        ? eta + (yi - mustar) / Math.max(1e-8, mu * (1 - mu))
        : eta + (yi - mustar) / Math.max(1e-8, mu)
    wy[i] = (z_i - reAt(i)) * sw
  }
  const fit = lstsq(WX, wy)
  beta = Array.from(fit.coef)
  const zCrit = STD.ppf(0.5 + confidence / 2)
  const fixed: MixedFixedEffect[] = []
  for (let j = 0; j < p; j++) {
    const se = Math.sqrt(Math.max(0, fit.xtxInv.data[j * p + j]!))
    const t = se > 0 ? beta[j]! / se : NaN
    const pValue = Number.isFinite(t) ? 2 * STD.sf(Math.abs(t)) : NaN
    fixed.push({
      name: names[j]!,
      coefficient: beta[j]!,
      se,
      t,
      pValue,
      ci: [beta[j]! - zCrit * se, beta[j]! + zCrit * se],
    })
  }

  const obsLik = (eta: number, i: number) => {
    const mu = linkInv(eta)
    if (family === 'binomial') {
      const m = trials[i]!
      const k = yy[i]!
      const pmu = Math.min(1 - 1e-12, Math.max(1e-12, mu))
      return k * Math.log(pmu) + (m - k) * Math.log(1 - pmu)
    }
    if (family === 'negbin') {
      const mui = Math.max(1e-12, mu)
      const th = Math.max(1e-6, theta)
      return (
        lgamma(yy[i]! + th) -
        lgamma(th) -
        lgamma(yy[i]! + 1) +
        th * Math.log(th / (th + mui)) +
        yy[i]! * Math.log(mui / (th + mui))
      )
    }
    const mui = Math.max(1e-12, mu)
    return yy[i]! * Math.log(mui) - mui
  }

  // Laplace-ish logLik
  let ll = 0
  for (let i = 0; i < n; i++) {
    let eta = reAt(i)
    for (let j = 0; j < p; j++) eta += beta[j]! * X.data[i * p + j]!
    ll += obsLik(eta, i)
  }
  ll -= 0.5 * g * Math.log(2 * Math.PI * sigmaU2)
  for (let gi = 0; gi < g; gi++) ll -= 0.5 * (u[gi]! * u[gi]!) / sigmaU2
  if (hasSlope) {
    ll -= 0.5 * g * Math.log(2 * Math.PI * sigmaS2)
    for (let gi = 0; gi < g; gi++) ll -= 0.5 * (sBlup[gi]! * sBlup[gi]!) / sigmaS2
  }

  if (estMethod === 'laplace') {
    for (let gi = 0; gi < g; gi++) {
      let hessU = -1 / Math.max(1e-8, sigmaU2)
      let hessS = hasSlope ? -1 / Math.max(1e-8, sigmaS2) : 0
      for (let i = 0; i < n; i++) {
        if (idx[i] !== gi) continue
        const zi = hasSlope && slopeX ? slopeX[i]! : 0
        let eta = u[gi]! + (hasSlope ? sBlup[gi]! * zi : 0)
        for (let j = 0; j < p; j++) eta += beta[j]! * X.data[i * p + j]!
        const mu = linkInv(eta)
        const m = trials[i]!
        let d2: number
        if (family === 'binomial') d2 = -m * mu * (1 - mu)
        else if (family === 'negbin') {
          const th = Math.max(1e-6, theta)
          d2 = -(mu * th) / (th + mu)
        } else d2 = -mu
        hessU += d2
        if (hasSlope) hessS += d2 * zi * zi
      }
      ll -= 0.5 * Math.log(Math.max(1e-12, -hessU))
      if (hasSlope) ll -= 0.5 * Math.log(Math.max(1e-12, -hessS))
    }
  }

  if (estMethod === 'agq') {
    const { x: ghX, w: ghW } = gaussHermiteNodes(nAGQ)
    ll = 0
    if (!hasSlope || !slopeX) {
      for (let gi = 0; gi < g; gi++) {
        let hess = -1 / sigmaU2
        for (let i = 0; i < n; i++) {
          if (idx[i] !== gi) continue
          let eta = u[gi]!
          for (let j = 0; j < p; j++) eta += beta[j]! * X.data[i * p + j]!
          const mu = linkInv(eta)
          const m = trials[i]!
          if (family === 'binomial') hess -= m * mu * (1 - mu)
          else if (family === 'negbin') {
            const th = Math.max(1e-6, theta)
            hess -= (mu * th) / (th + mu)
          } else hess -= mu
        }
        const scale = 1 / Math.sqrt(Math.max(1e-8, -hess))
        let integral = 0
        for (let k = 0; k < ghX.length; k++) {
          const uk = u[gi]! + scale * Math.SQRT2 * ghX[k]!
          let lik = 0
          for (let i = 0; i < n; i++) {
            if (idx[i] !== gi) continue
            let eta = uk
            for (let j = 0; j < p; j++) eta += beta[j]! * X.data[i * p + j]!
            lik += obsLik(eta, i)
          }
          const prior = -0.5 * Math.log(2 * Math.PI * sigmaU2) - 0.5 * (uk * uk) / sigmaU2
          const prop = -0.5 * Math.log(2 * Math.PI * scale * scale) - ((uk - u[gi]!) ** 2) / (2 * scale * scale)
          integral += (ghW[k]! / Math.sqrt(Math.PI)) * Math.exp(lik + prior - prop + Math.log(scale * Math.SQRT2))
        }
        ll += Math.log(Math.max(1e-300, integral))
      }
    } else {
      // 2D AGQ with correlated G (ρ between intercept and slope)
      const rho = Math.max(-0.95, Math.min(0.95, rhoEst))
      const detG = Math.max(1e-12, sigmaU2 * sigmaS2 * (1 - rho * rho))
      for (let gi = 0; gi < g; gi++) {
        let hessU = -1 / sigmaU2
        let hessS = -1 / sigmaS2
        let hessUS = rho / Math.sqrt(Math.max(1e-12, sigmaU2 * sigmaS2)) // approx off-diag precision
        for (let i = 0; i < n; i++) {
          if (idx[i] !== gi) continue
          const zi = slopeX[i]!
          let eta = u[gi]! + sBlup[gi]! * zi
          for (let j = 0; j < p; j++) eta += beta[j]! * X.data[i * p + j]!
          const mu = linkInv(eta)
          const m = trials[i]!
          let d2: number
          if (family === 'binomial') d2 = -m * mu * (1 - mu)
          else if (family === 'negbin') {
            const th = Math.max(1e-6, theta)
            d2 = -(mu * th) / (th + mu)
          } else d2 = -mu
          hessU += d2
          hessS += d2 * zi * zi
          hessUS += d2 * zi
        }
        // Cholesky-scale proposal from Hessian (diagonal fallback if indefinite)
        const scaleU = 1 / Math.sqrt(Math.max(1e-8, -hessU))
        const scaleS = 1 / Math.sqrt(Math.max(1e-8, -hessS))
        let integral = 0
        for (let ku = 0; ku < ghX.length; ku++) {
          for (let ks = 0; ks < ghX.length; ks++) {
            const uk = u[gi]! + scaleU * Math.SQRT2 * ghX[ku]!
            const sk = sBlup[gi]! + scaleS * Math.SQRT2 * ghX[ks]!
            let lik = 0
            for (let i = 0; i < n; i++) {
              if (idx[i] !== gi) continue
              let eta = uk + sk * slopeX[i]!
              for (let j = 0; j < p; j++) eta += beta[j]! * X.data[i * p + j]!
              lik += obsLik(eta, i)
            }
            // bivariate normal prior with corr ρ
            const zu = uk / Math.sqrt(sigmaU2)
            const zs = sk / Math.sqrt(sigmaS2)
            const quad = (zu * zu - 2 * rho * zu * zs + zs * zs) / (1 - rho * rho)
            const prior =
              -0.5 * Math.log(4 * Math.PI * Math.PI * detG) - 0.5 * quad
            const prop =
              -0.5 * Math.log(2 * Math.PI * scaleU * scaleU) -
              ((uk - u[gi]!) ** 2) / (2 * scaleU * scaleU) -
              0.5 * Math.log(2 * Math.PI * scaleS * scaleS) -
              ((sk - sBlup[gi]!) ** 2) / (2 * scaleS * scaleS)
            const wgh = (ghW[ku]! / Math.sqrt(Math.PI)) * (ghW[ks]! / Math.sqrt(Math.PI))
            integral +=
              wgh *
              Math.exp(lik + prior - prop + Math.log(scaleU * Math.SQRT2) + Math.log(scaleS * Math.SQRT2))
          }
        }
        ll += Math.log(Math.max(1e-300, integral))
      }
    }
  }

  return {
    family,
    method: estMethod,
    nAGQ: estMethod === 'agq' ? nAGQ : undefined,
    theta: family === 'negbin' ? theta : undefined,
    fixed,
    sigmaRandom: Math.sqrt(sigmaU2),
    sigmaSlope: hasSlope ? Math.sqrt(sigmaS2) : undefined,
    rho: hasSlope ? rhoEst : undefined,
    ranef: keys.map((group, j) => ({
      group,
      blup: u[j]!,
      slope: hasSlope ? sBlup[j]! : undefined,
    })),
    logLik: ll,
    n,
    nGroups: g,
    iterations: iter + 1,
  }
}

/** Gauss–Hermite nodes/weights for ∫ e^{-x²} f(x) dx (n=5…15 odd). */
function gaussHermiteNodes(n: number): { x: number[]; w: number[] } {
  // Precomputed for common AGQ sizes
  const tables: Record<number, { x: number[]; w: number[] }> = {
    5: {
      x: [-2.0201828705, -0.9585724646, 0, 0.9585724646, 2.0201828705],
      w: [0.01995324206, 0.39361932315, 0.94530872048, 0.39361932315, 0.01995324206],
    },
    7: {
      x: [-2.6519613568, -1.6735516287, -0.8162878828, 0, 0.8162878828, 1.6735516287, 2.6519613568],
      w: [0.000971781245, 0.05451558282, 0.4256072526, 0.8102646175, 0.4256072526, 0.05451558282, 0.000971781245],
    },
    9: {
      x: [-3.1909932018, -2.2665805845, -1.5174274951, -0.7235510185, 0, 0.7235510185, 1.5174274951, 2.2665805845, 3.1909932018],
      w: [0.000039606977, 0.00494373515, 0.08847452739, 0.4326515451, 0.7202352156, 0.4326515451, 0.08847452739, 0.00494373515, 0.000039606977],
    },
  }
  const key = n <= 5 ? 5 : n <= 7 ? 7 : 9
  return tables[key]!
}
