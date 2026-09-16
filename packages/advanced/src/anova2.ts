/**
 * Two-way ANOVA and ANCOVA via Type III GLM (`linearModel`).
 */
import { linearModel, type LinearModelResult, type LmTermRow } from './lm.js'
import { f as fDist } from './dist.js'
import { cleanNumbers } from './tests.js'

export interface AnovaTwoWayResult {
  test: 'Two-way ANOVA'
  terms: LmTermRow[]
  residualDf: number
  residualSS: number
  residualMS: number
  n: number
  interaction: boolean
  /** Convenience F / p for row, column, interaction (if present). */
  row?: LmTermRow
  col?: LmTermRow
  interactionTerm?: LmTermRow
}

export interface AncovaResult {
  test: 'ANCOVA'
  terms: LmTermRow[]
  residualDf: number
  residualSS: number
  residualMS: number
  n: number
  group?: LmTermRow
  covariate?: LmTermRow
  fit: LinearModelResult
}

/**
 * Two-way ANOVA: `y ~ row * col` (with interaction) or `y ~ row + col`.
 * Factors are treated as categorical via `linearModel` Type III SS.
 */
export function anovaTwoWay(
  y: ArrayLike<number | null | undefined>,
  row: ArrayLike<string | number | null | undefined>,
  col: ArrayLike<string | number | null | undefined>,
  options: { interaction?: boolean } = {},
): AnovaTwoWayResult {
  const interaction = options.interaction !== false
  const n = y.length
  if (row.length !== n || col.length !== n) throw new RangeError('anovaTwoWay: y, row, col must have equal length')
  const data = { y, row, col }
  const formula = interaction ? 'y ~ row * col' : 'y ~ row + col'
  const fit = linearModel(data, formula, { factors: ['row', 'col'] })
  const terms = fit.anova.terms
  const find = (name: string) => terms.find((t) => t.term === name || t.term.replace(/\*/g, ':') === name)
  return {
    test: 'Two-way ANOVA',
    terms,
    residualDf: fit.anova.error.df,
    residualSS: fit.anova.error.ss,
    residualMS: fit.anova.error.ms,
    n: fit.n,
    interaction,
    row: find('row'),
    col: find('col'),
    interactionTerm: interaction ? find('row*col') ?? find('row:col') : undefined,
  }
}

/**
 * One-way ANCOVA: `y ~ group + covariate` with group as factor.
 */
export function ancova(
  y: ArrayLike<number | null | undefined>,
  group: ArrayLike<string | number | null | undefined>,
  covariate: ArrayLike<number | null | undefined>,
): AncovaResult {
  const n = y.length
  if (group.length !== n || covariate.length !== n) throw new RangeError('ancova: inputs must have equal length')
  const fit = linearModel({ y, group, covariate }, 'y ~ group + covariate', { factors: ['group'] })
  const terms = fit.anova.terms
  return {
    test: 'ANCOVA',
    terms,
    residualDf: fit.anova.error.df,
    residualSS: fit.anova.error.ss,
    residualMS: fit.anova.error.ms,
    n: fit.n,
    group: terms.find((t) => t.term === 'group'),
    covariate: terms.find((t) => t.term === 'covariate'),
    fit,
  }
}

/** Classic balanced two-way ANOVA without interaction (cell means) — used when design is complete and balanced. */
export function anovaTwoWayBalanced(
  cells: Record<string, Record<string, ArrayLike<number | null | undefined>>>,
): { SSA: number; SSB: number; SSE: number; dfA: number; dfB: number; dfE: number; FA: number; FB: number; pA: number; pB: number } {
  const rows = Object.keys(cells)
  const cols = Object.keys(cells[rows[0]!]!)
  const a = rows.length
  const b = cols.length
  let nCell = -1
  const means: number[][] = []
  let grand = 0
  let N = 0
  for (let i = 0; i < a; i++) {
    means[i] = []
    for (let j = 0; j < b; j++) {
      const v = cleanNumbers(cells[rows[i]!]![cols[j]!]!)
      if (nCell < 0) nCell = v.length
      if (v.length !== nCell) throw new RangeError('anovaTwoWayBalanced: unbalanced cells')
      let s = 0
      for (const x of v) s += x
      means[i]![j] = s / nCell
      grand += s
      N += nCell
    }
  }
  grand /= N
  const rowMean = means.map((r) => r.reduce((s, m) => s + m, 0) / b)
  const colMean = cols.map((_, j) => means.reduce((s, r) => s + r[j]!, 0) / a)
  let SSA = 0
  let SSB = 0
  let SSE = 0
  for (let i = 0; i < a; i++) SSA += b * nCell * (rowMean[i]! - grand) ** 2
  for (let j = 0; j < b; j++) SSB += a * nCell * (colMean[j]! - grand) ** 2
  for (let i = 0; i < a; i++) {
    for (let j = 0; j < b; j++) {
      const v = cleanNumbers(cells[rows[i]!]![cols[j]!]!)
      for (const x of v) SSE += (x - means[i]![j]!) ** 2
    }
  }
  const dfA = a - 1
  const dfB = b - 1
  const dfE = a * b * (nCell - 1)
  const FA = (SSA / dfA) / (SSE / dfE)
  const FB = (SSB / dfB) / (SSE / dfE)
  return {
    SSA,
    SSB,
    SSE,
    dfA,
    dfB,
    dfE,
    FA,
    FB,
    pA: fDist(dfA, dfE).sf(FA),
    pB: fDist(dfB, dfE).sf(FB),
  }
}
