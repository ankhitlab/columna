/** Proposed Columna engineering workbench APIs, MIT.
 * Composes native DOE, ANOVA/MSA, forecasting, survival and multivariate algorithms.
 * All estimators and statistical transformations remain in this library module.
 * Results contain data only and can cross a Web Worker structured-clone boundary.
 */
import { fullFactorial, fractionalFactorial, plackettBurman, ccd, boxBehnken, type DoeDesign } from './doe.js'
import { gageRR, gageType1, attributeAgreement } from './msa.js'
import { trendAnalysis, ets, decompose, acf, pacf, ljungBox, arima } from './timeseries.js'
import { kaplanMeier, reliabilityFit, warrantyPrediction, logRank } from './reliability.js'
import { pca, factorAnalysis, kmeans, hclust } from './multivariate.js'
import { cutTree } from './multivariate2.js'
import { linearWorkbench } from './linear-workbench.js'
import { predictLinearModel, type Data } from './lm.js'
import { ols } from './regression.js'
import { descriptiveStats, type DescriptiveStats } from './descriptive.js'
import { ttest1 } from './tests.js'
import { f, normal } from './dist.js'
import { controlChart } from './spc.js'

export type EngineeringCell = number | string | null
export type EngineeringDomain = 'doe' | 'msa' | 'timeseries' | 'reliability' | 'multivariate'
export interface EngineeringInput {
  task: string
  data?: Record<string, EngineeringCell[]>
  labels?: Record<string, string>
  y?: string; columns?: string[]; group?: string; part?: string; operator?: string; referenceColumn?: string; censor?: string; time?: string; block?: string
  factors?: Array<{ name: string; low: number; high: number }>
  generators?: string[]
  design?: string; structure?: string; interaction?: 'keep' | 'auto' | 'pool'; method?: string
  alpha?: number; confidence?: number; seed?: number; replicates?: number; centerPoints?: number; axial?: 'rotatable' | 'face-centered' | 'orthogonal'; randomize?: boolean
  tolerance?: number; reference?: number; studyMultiplier?: number
  horizon?: number; seasonLength?: number; lag?: number; start?: number; end?: number; p?: number; d?: number; q?: number; includeMean?: boolean; holdout?: number
  mission?: number; components?: number; scale?: boolean; k?: number; starts?: number; maxIter?: number; memoryMB?: number
}
export interface EngineeringTable { title: string; headers: string[]; rows: EngineeringCell[][]; formats?: string[] }
export interface EngineeringPlot {
  type: 'eng-line' | 'eng-scatter' | 'eng-bars' | 'eng-dendrogram'
  title: string; xLabel?: string; yLabel?: string; note?: string
  series?: Array<{ name: string; points: Array<[number, number | null, string?]>; step?: boolean; markers?: boolean; dashed?: boolean }>
  items?: Array<{ label: string; value: number }>
  merge?: Array<[number, number, number]>; order?: number[]; labels?: string[]
  pageSize?: number; fixedY?: [number, number]
}
export interface EngineeringResult {
  domain: EngineeringDomain; task: string; n: number; omitted: number; sourceRows: number[]; independent?: boolean
  metrics: Array<[string, number | string | null, string?]>; tables: EngineeringTable[]; plots: EngineeringPlot[]; warnings: string[]
  output?: { name: string; names: string[]; types: ('number' | 'text')[]; rows: EngineeringCell[][] }
  details?: unknown
}
const finite = Number.isFinite
const val = (x: unknown): EngineeringCell => typeof x === 'number' ? finite(x) ? x : null : typeof x === 'string' ? x : null
const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / v.length
const desc = (v: number[]) => descriptiveStats(v) as DescriptiveStats
function num(x: number | undefined, fallback: number, label: string, min = -Infinity, max = Infinity, integer = false) {
  const n = x ?? fallback
  if (!finite(n) || n < min || n > max || integer && !Number.isInteger(n)) throw new RangeError(label + ': invalid numeric value')
  return n
}
function confidence(i: EngineeringInput) { const c = i.confidence ?? (1 - (i.alpha ?? .05)); if (!(c > .5 && c < 1)) throw new RangeError('confidence must be between 0.5 and 1'); return c }
function choose<T extends string>(x: string | undefined, fallback: T, choices: readonly T[]): T { const v = x ?? fallback; if (!choices.includes(v as T)) throw new RangeError('Unknown method: ' + v); return v as T }
function base(domain: EngineeringDomain, i: EngineeringInput): EngineeringResult { return { domain, task: i.task, n: 0, omitted: 0, sourceRows: [], metrics: [], tables: [], plots: [], warnings: [] } }
function table(r: EngineeringResult, title: string, headers: string[], rows: unknown[][], formats: string[] = []) { r.tables.push({ title, headers, rows: rows.map((row) => row.map(val)), formats }) }
function line(r: EngineeringResult, title: string, xLabel: string, yLabel: string, series: NonNullable<EngineeringPlot['series']>, note?: string) { r.plots.push({ type: 'eng-line', title, xLabel, yLabel, series, note }) }
function scatter(r: EngineeringResult, title: string, xLabel: string, yLabel: string, series: NonNullable<EngineeringPlot['series']>) { r.plots.push({ type: 'eng-scatter', title, xLabel, yLabel, series }) }
function bars(r: EngineeringResult, title: string, items: Array<{ label: string; value: number }>, xLabel: string) { r.plots.push({ type: 'eng-bars', title, items, xLabel, pageSize: 20 }) }
const xy = (v: ArrayLike<number>, x?: number[]): Array<[number, number | null]> => Array.from(v, (v, j) => [x?.[j] ?? j + 1, finite(v) ? v : null])
const label = (i: EngineeringInput, id: string) => i.labels?.[id] ?? id
function get(i: EngineeringInput, id: string | undefined): EngineeringCell[] { if (!id || !i.data || !Object.hasOwn(i.data, id)) throw new RangeError('Select a valid worksheet column'); return i.data[id]! }
function complete(i: EngineeringInput, ids: string[], numeric: string[], minimum: number, r: EngineeringResult) {
  if (new Set(ids).size !== ids.length) throw new RangeError('Select different columns for different roles')
  const columns = ids.map((id) => get(i, id)), n = columns[0]?.length ?? 0
  if (columns.some((c) => c.length !== n)) throw new RangeError('Column lengths differ')
  const rows: EngineeringCell[][] = [], keep: number[] = []
  for (let j = 0; j < n; j++) {
    const row = columns.map((c) => c[j]!)
    if (row.some((x) => x === null || x === undefined || x === '')) continue
    row.forEach((v, k) => { if (numeric.includes(ids[k]!) && (typeof v !== 'number' || !finite(v))) throw new RangeError('Non-numeric value in a numeric analysis column') })
    rows.push(row); keep.push(j + 1)
  }
  if (rows.length < minimum) throw new RangeError('Insufficient complete observations: need at least ' + minimum)
  r.n = rows.length; r.omitted = n - rows.length; r.sourceRows = keep
  if (r.omitted) r.warnings.push('Rows with missing values are omitted jointly across all selected analysis columns; source row numbers are preserved.')
  return { rows, keep }
}
function seeded(seed: number) { let state = seed >>> 0; return () => { state += 0x6d2b79f5; let t = state; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296 } }
function renameTerm(term: string, ids: string[], i: EngineeringInput) { return term.replace(/X(\d+)/g, (_, n) => label(i, ids[Number(n) - 1]!) ) }

/** Designs retain both coded and natural settings; every generated row is exported. */
export function doeStudy(i: EngineeringInput): EngineeringResult {
  const r = base('doe', i), conf = confidence(i)
  if (i.task === 'create') {
    r.independent = true
    const fs = i.factors ?? [], k = fs.length
    if (k < 2 || k > 12 || new Set(fs.map((f) => f.name)).size !== k || fs.some((f) => !f.name.trim() || !finite(f.low) || !finite(f.high) || !(f.high > f.low))) throw new RangeError('DOE: 2…12 unique factor names with finite low < high are required')
    const names = fs.map((f) => f.name), design = choose(i.design, 'full', ['full', 'fractional', 'pb', 'ccd', 'box-behnken'] as const)
    const centers = num(i.centerPoints, design === 'ccd' || design === 'box-behnken' ? 3 : 0, 'centerPoints', 0, 10000, true)
    const reps = num(i.replicates, 1, 'replicates', 1, 10000, true), seed = num(i.seed, 42, 'seed', 0, 4294967295, true)
    let d: DoeDesign
    if (design === 'full') d = fullFactorial(names)
    else if (design === 'fractional') d = fractionalFactorial(names, i.generators ?? [])
    else if (design === 'pb') d = plackettBurman(names)
    else if (design === 'ccd') d = ccd(names, { centerPoints: centers, alpha: i.axial === 'face-centered' ? 1 : i.axial ?? 'rotatable' })
    else {
      // Native all-pairs construction is a standard BBD for 3/4 factors, not the tabulated 5…7-factor designs.
      if (k !== 3 && k !== 4) throw new RangeError('DOE: the validated Box–Behnken generator supports 3 or 4 factors')
      d = boxBehnken(names, { centerPoints: centers })
    }
    const designRows = d.matrix.map((row) => row.slice())
    if (!['ccd', 'box-behnken'].includes(design)) for (let c = 0; c < centers; c++) designRows.push(new Array(k).fill(0))
    const memory = num(i.memoryMB, 256, 'design memory budget MB', 1, 8192)
    if (designRows.length * reps * (2 * k + 4) * 32 > memory * 1024 * 1024) throw new RangeError('DOE plan exceeds the chosen memory budget; reduce replicates or increase the budget')
    const indexed = Array.from({ length: reps }, (_, rep) => designRows.map((code, j) => ({ code, standard: rep * designRows.length + j + 1, replicate: rep + 1 }))).flat()
    if (i.randomize !== false) { const rnd = seeded(seed); for (let j = indexed.length - 1; j > 0; j--) { const h = Math.floor(rnd() * (j + 1)); [indexed[j], indexed[h]] = [indexed[h]!, indexed[j]!] } }
    const outNames = ['RunOrder', 'StdOrder', 'Replicate', ...names, ...names.map((n) => n + ' [coded]'), 'Response']
    if (new Set(outNames).size !== outNames.length) throw new RangeError('DOE: factor names conflict with generated worksheet columns')
    const rows = indexed.map((rr, j) => [j + 1, rr.standard, rr.replicate, ...rr.code.map((c, h) => (fs[h]!.low + fs[h]!.high) / 2 + c * (fs[h]!.high - fs[h]!.low) / 2), ...rr.code, null])
    r.n = rows.length; r.metrics = [['Runs', r.n], ['Factors', k], ['Replicates', reps], ['Resolution', typeof d.info?.resolution === 'number' ? d.info.resolution : '—']]
    table(r, 'Design worksheet', outNames, rows)
    table(r, 'Factor settings', ['Factor', 'Low (−1)', 'High (+1)'], fs.map((f) => [f.name, f.low, f.high]))
    if (d.info?.definingRelation) table(r, 'Defining relation', ['I = word'], (d.info.definingRelation as string[]).filter((w) => w !== 'I').map((w) => [w]))
    const terms = names.map((n, j) => ({ name: String.fromCharCode(65 + j), col: d.matrix.map((rr) => rr[j]!) }))
    for (let a = 0; a < k; a++) for (let b = a + 1; b < k; b++) terms.push({ name: String.fromCharCode(65 + a) + String.fromCharCode(65 + b), col: d.matrix.map((rr) => rr[a]! * rr[b]!) })
    if (design === 'fractional' || design === 'pb') {
      const alias: unknown[][] = []
      for (let a = 0; a < terms.length; a++) for (let b = a + 1; b < terms.length; b++) {
        const va = terms[a]!.col, vb = terms[b]!.col, ma = mean(va), mb = mean(vb)
        let cov = 0, s1 = 0, s2 = 0
        for (let h = 0; h < va.length; h++) { cov += (va[h]! - ma) * (vb[h]! - mb); s1 += (va[h]! - ma) ** 2; s2 += (vb[h]! - mb) ** 2 }
        const cor = cov / Math.sqrt(s1 * s2)
        if (finite(cor) && Math.abs(cor) > 1e-9) alias.push([terms[a]!.name, terms[b]!.name, cor, Math.abs(cor) > 1 - 1e-10 ? 'Exact alias' : 'Partial confounding'])
      }
      table(r, 'Aliasing: main effects and two-factor interactions', ['Term A', 'Term B', 'Correlation', 'Type'], alias)
      r.warnings.push('Alias diagnostics cover main effects and two-factor interactions; higher-order aliasing is described by the defining relation. Main effects may be confounded with interactions.')
    }
    scatter(r, 'Design: first two coded factors', names[0]!, names[1]!, [{ name: 'Design runs', points: indexed.map((x, j) => [x.code[0]!, x.code[1]!, 'Run ' + (j + 1)]) }])
    r.output = { name: 'DOE · ' + design, names: outNames, types: outNames.map(() => 'number'), rows }
    r.details = { design: d, seed, randomized: i.randomize !== false, centerPointsPerReplicate: centers }
    r.warnings.push('Generated response cells are empty: enter real observations before fitting. Standard order and randomized run order are both retained. Replicate is not a statistical block.')
    if (design === 'ccd' && i.axial !== 'face-centered') r.warnings.push('Axial CCD points extend beyond the low/high factorial settings. Check physical and safety feasibility before conducting experiments.')
    r.warnings.push('Randomization is reproducible using the saved seed. Split plots, hard-to-change factors, constraints, mixtures and Taguchi arrays are not implied by this generator.')
    return r
  }
  if (i.task !== 'analyze') throw new RangeError('Unknown DOE task')
  const ids = i.columns ?? [], yid = i.y!, block = i.block, structure = choose(i.structure, '2fi', ['main', '2fi', 'quadratic'] as const)
  if (ids.length < 1) throw new RangeError('DOE: select at least one numeric factor')
  const { rows, keep } = complete(i, [yid, ...ids, ...(block ? [block] : [])], [yid, ...ids], 4, r)
  const data: Data = { Y: rows.map((rr) => rr[0]) }, terms: string[] = []
  ids.forEach((id, j) => { const name = 'X' + (j + 1); data[name] = rows.map((rr) => rr[j + 1]); terms.push(name); if (!(desc(data[name] as number[]).sd > 0)) throw new RangeError('DOE: each factor needs more than one setting') })
  if (structure !== 'main') for (let a = 0; a < ids.length; a++) for (let b = a + 1; b < ids.length; b++) terms.push(`X${a + 1}:X${b + 1}`)
  if (structure === 'quadratic') ids.forEach((_, j) => { if (new Set(Array.from(data['X' + (j + 1)]!)).size < 3) throw new RangeError('Quadratic DOE analysis requires at least three settings of each factor'); terms.push(`X${j + 1}^2`) })
  if (block) { data.Block = rows.map((rr) => String(rr.at(-1))); terms.push('Block') }
  const work = linearWorkbench(data, 'Y ~ ' + terms.join(' + '), { factors: block ? ['Block'] : [], confidence: conf, ssType: 'III' }), fit = work.fit
  r.metrics = [['Runs used', fit.n], ['R²', fit.r2, 'percent'], ['Adjusted R²', fit.r2adj, 'percent'], ['Residual S', fit.s]]
  table(r, 'Model coefficients (natural units)', ['Term', 'Coefficient', 'SE', 't', 'p', 'CI lower', 'CI upper'], fit.coefficients.map((c) => [renameTerm(c.name, ids, i), c.coef, c.se, c.t, c.pValue, ...c.ci]), ['', '', '', '', 'p'])
  table(r, 'ANOVA · adjusted sums of squares', ['Term', 'DF', 'SS', 'MS', 'F', 'p'], [...work.terms.map((t) => [renameTerm(t.term, ids, i), t.df, t.ss, t.ms, t.f, t.pValue]), ['Error', fit.anova.error.df, fit.anova.error.ss, fit.anova.error.ms, null, null]], ['', '', '', '', '', 'p'])
  const cells = new Map<string, number[]>()
  rows.forEach((rr) => { const key = JSON.stringify(rr.slice(1)), a = cells.get(key) ?? []; a.push(rr[0] as number); cells.set(key, a) })
  const pe = [...cells.values()].reduce((ss, vs) => { const m = mean(vs); return ss + vs.reduce((s, y) => s + (y - m) ** 2, 0) }, 0), peDF = rows.length - cells.size, lofDF = fit.anova.error.df - peDF
  const lof = Math.max(0, fit.anova.error.ss - pe), stat = peDF > 0 && lofDF > 0 && pe > 0 ? (lof / lofDF) / (pe / peDF) : NaN
  table(r, 'Lack of fit and pure error', ['Source', 'SS', 'DF', 'F', 'p'], [['Lack of fit', lof, lofDF, stat, finite(stat) ? f(lofDF, peDF).sf(stat) : null], ['Pure error', pe, peDF, null, null]], ['', '', '', '', 'p'])
  if (!finite(stat)) r.warnings.push('Lack-of-fit test is not estimable without replicated identical settings, positive pure error and positive lack-of-fit degrees of freedom.')
  const diagnostics = work.diagnostics.map((d, j) => [keep[j], d.observed, d.fitted, d.residual, d.leverage, d.cooksD])
  table(r, 'DOE diagnostics', ['Source row', 'Observed', 'Fitted', 'Residual', 'Leverage', 'Cook D'], diagnostics)
  bars(r, 'Coefficient diagnostics · |t|', fit.coefficients.filter((c) => c.name !== 'Intercept').map((c) => ({ label: renameTerm(c.name, ids, i), value: Math.abs(c.t) })).sort((a, b) => b.value - a.value), '|t|')
  scatter(r, 'Residuals vs fitted', 'Fitted', 'Residual', [{ name: 'Residual', points: work.diagnostics.map((d, j) => [d.fitted, d.residual, 'Row ' + keep[j]]) }])
  const levels = ids.map((_, j) => desc(data['X' + (j + 1)] as number[])), profile: unknown[][] = []
  const baseRow: Record<string, unknown> = Object.fromEntries(ids.map((_, j) => ['X' + (j + 1), levels[j]!.mean]))
  if (block) baseRow.Block = data.Block![0]
  ids.forEach((id, j) => { for (const setting of [levels[j]!.min, levels[j]!.max]) { const pred = predictLinearModel(fit, [{ ...baseRow, ['X' + (j + 1)]: setting }])[0]!; profile.push([label(i, id), setting, pred.fit, ...pred.ci]) } })
  table(r, 'Model profiles: other factors at sample means', ['Factor', 'Setting', 'Predicted mean', 'CI lower', 'CI upper'], profile)
  r.output = { name: 'DOE diagnostics', names: ['Source row', 'Observed', 'Fitted', 'Residual', 'Leverage', 'Cook D'], types: Array(6).fill('number'), rows: diagnostics.map((x) => x.map(val)) }
  r.details = { formula: fit.formula, lackOfFit: { ss: lof, df: lofDF, pureError: pe, pureErrorDF: peDF, pValue: finite(stat) ? f(lofDF, peDF).sf(stat) : null } }
  r.warnings.push(...work.warnings, 'The model is fitted in the selected columns’ units. A coefficient is not automatically a two-level effect. |t| bars are not Lenth PSE tests.', 'Use randomized independent runs and a model consistent with the experimental design. A saturated/aliased model is rejected rather than assigned artificial residual degrees of freedom.')
  if (block) r.warnings.push('Block is a fixed categorical effect. Model profiles hold the block at its first observed level; this is not a split-plot or random-effects analysis.')
  return r
}

export function msaStudy(i: EngineeringInput): EngineeringResult {
  const r = base('msa', i), conf = confidence(i), alpha = 1 - conf
  if (i.task === 'crossed' || i.task === 'nested') {
    const { rows, keep } = complete(i, [i.y!, i.part!, i.operator!], [i.y!], 8, r)
    const y = rows.map((rr) => rr[0] as number), parts = rows.map((rr) => String(rr[1])), ops = rows.map((rr) => String(rr[2]))
    const result = gageRR({ measurement: y, part: parts, operator: ops }, { design: i.task, interaction: i.interaction ?? 'keep', alpha, tolerance: i.tolerance, studyMultiplier: num(i.studyMultiplier, 6, 'study multiplier', .01, 100) })
    const total = result.components[0]!
    r.metrics = [['% Study Var · GRR', total.pctStudyVar], ['% Contribution · GRR', total.pctContribution], ['Distinct categories (ndc)', finite(result.ndc) ? result.ndc : '∞'], ['σ Gage R&R', result.sigmaGageRR]]
    table(r, 'Variance components', ['Source', 'Variance', 'Std dev', '% Contribution', 'Study variation', '% Study Var', '% Tolerance'], result.components.map((c) => [c.name, c.variance, c.stdDev, c.pctContribution, c.studyVariation, c.pctStudyVar, c.pctTolerance]))
    table(r, 'Gage ANOVA', ['Source', 'SS', 'DF', 'MS', 'F', 'p'], result.anova!.map((a) => [a.source, a.ss, a.df, a.ms, a.F, a.pValue]), ['', '', '', '', '', 'p'])
    bars(r, 'Measurement variation components', result.components.filter((c) => !['Total Variation', 'Total Gage R&R'].includes(c.name)).map((c) => ({ label: c.name, value: c.pctStudyVar })), '% Study Var (not additive)')
    const levels = [...new Set(parts)].sort(), operators = [...new Set(ops)].sort(), summaries: unknown[][] = [], groups: number[][] = []
    const series = operators.map((op) => ({ name: op, points: levels.flatMap((part, j): Array<[number, number, string]> => { const a = y.filter((_, h) => ops[h] === op && parts[h] === part); if (!a.length) return []; const d = desc(a); groups.push(a); summaries.push([op, part, a.length, d.mean, d.range, d.sd]); return [[j + 1, d.mean, op + ' / ' + part]] }) }))
    line(r, 'Part–operator mean profile', 'Part position', 'Mean measurement', series)
    const chart = controlChart(groups, { type: 'xbar-r', rules: [1] }).companion!
    line(r, 'Repeatability: ranges within part × operator', 'Cell (operator then part)', 'Range', [{ name: 'R', points: chart.points.map((p) => [p.index + 1, p.value]) }, { name: 'UCL', points: chart.points.map((p) => [p.index + 1, p.ucl]), dashed: true }, { name: 'CL', points: chart.points.map((p) => [p.index + 1, p.center]), dashed: true }])
    table(r, 'Cell summaries', ['Operator', 'Part', 'N', 'Mean', 'Range', 'Std dev'], summaries)
    r.details = result; r.warnings.push(...result.warnings!, 'ANOVA R&R requires a balanced design with equal replication and complete cells. Missing observations that destroy balance cause an error.', 'Percent contribution uses variance; percent study variation uses standard deviations and is not additive. Tolerance is USL−LSL, not a control-limit width.', 'R&R acceptance thresholds depend on the application and measurement risk; no universal pass/fail conclusion is assigned.')
    r.output = { name: 'MSA cell summaries', names: ['Operator', 'Part', 'N', 'Mean', 'Range', 'Std dev'], types: ['text', 'text', 'number', 'number', 'number', 'number'], rows: summaries.map((x) => x.map(val)) }
    void keep
  } else if (i.task === 'type1') {
    const { rows, keep } = complete(i, [i.y!], [i.y!], 3, r), y = rows.map((r) => r[0] as number)
    if (!(desc(y).sd > 0)) throw new RangeError('Type 1: nonzero measurement variation is required')
    const result = gageType1(y, { reference: num(i.reference, 0, 'reference'), tolerance: num(i.tolerance, 1, 'tolerance', Number.MIN_VALUE), k: num(i.studyMultiplier, 6, 'study multiplier', .01, 100) })
    const bias = ttest1(y.map((v) => v - result.reference), { confidence: conf })
    r.metrics = [['Cg', result.Cg], ['Cgk', result.Cgk], ['Bias', result.bias], ['σ repeatability', result.sd]]
    table(r, 'Type 1 gage study', ['N', 'Reference', 'Mean', 'Std dev', 'Tolerance width', 'Cg', 'Cgk'], [[result.n, result.reference, result.mean, result.sd, result.tolerance, result.Cg, result.Cgk]])
    table(r, 'Bias against reference', ['Bias', 't', 'DF', 'p', 'CI lower', 'CI upper'], [[result.bias, bias.statistic, bias.df, bias.pValue, ...bias.ci]], ['', '', '', 'p'])
    line(r, 'Repeated measurements of reference', 'Source row', label(i, i.y!), [{ name: 'Measurement', points: xy(y, keep) }, { name: 'Reference', points: keep.map((k) => [k, result.reference]), dashed: true }])
    r.details = { result, bias }; r.warnings.push('Type 1 evaluates one reference part, not reproducibility between operators. Cg uses 20% of the tolerance and the selected study multiplier. Bias inference assumes independent approximately normal errors.')
  } else if (i.task === 'linearity') {
    const { rows } = complete(i, [i.y!, i.referenceColumn!], [i.y!, i.referenceColumn!], 4, r)
    const x = rows.map((rr) => rr[1] as number), bias = rows.map((rr) => (rr[0] as number) - (rr[1] as number))
    if (new Set(x).size < 3 || !(desc(bias).sd > 0)) throw new RangeError('Linearity: at least three distinct reference levels and nonzero bias variation required')
    const fit = ols(bias, { Reference: x }, { confidence: conf }), meanX = mean(x), b0 = fit.coefficients[0]!, b1 = fit.coefficients[1]!
    const sxx = x.reduce((s, a) => s + (a - meanX) ** 2, 0), seBias = fit.s / Math.sqrt(x.length), averageBias = mean(bias)
    r.metrics = [['Average bias', averageBias], ['Linearity slope', b1.coef], ['Slope p-value', b1.pValue, 'p'], ['R² · bias model', fit.r2, 'percent']]
    table(r, 'Bias regression', ['Term', 'Coefficient', 'SE', 't', 'p', 'CI lower', 'CI upper'], fit.coefficients.map((c) => [c.name, c.coef, c.se, c.t, c.pValue, ...c.ci]), ['', '', '', '', 'p'])
    const levels = [...new Set(x)].sort((a, b) => a - b), critical = normal().ppf(.5 + conf / 2)
    table(r, 'Reference levels and bias', ['Reference', 'Replicates', 'Mean bias', 'Fitted bias'], levels.map((level) => [level, x.filter((a) => a === level).length, mean(bias.filter((_, j) => x[j] === level)), b0.coef + b1.coef * level]))
    scatter(r, 'Bias vs reference', 'Reference', 'Measurement − reference', [{ name: 'Bias', points: x.map((v, j) => [v, bias[j]!]) }, { name: 'OLS fit', dashed: true, points: levels.map((v) => [v, b0.coef + b1.coef * v]) }])
    r.details = { slope: b1.coef, intercept: b0.coef, averageBias, seBias, n: fit.n, s: fit.s, r2: fit.r2, coefficients: fit.coefficients }
    r.warnings.push('Linearity uses OLS of measurement−reference on the reference. Reference values are treated as known without error; independent equal-variance errors are assumed. A non-significant slope does not establish negligible bias.')
    void sxx; void critical
  } else if (i.task === 'agreement') {
    const ids = i.columns ?? []; if (ids.length < 2) throw new RangeError('Agreement: select at least two rating columns')
    const { rows } = complete(i, ids, [], 3, r), ratings = rows.map((rr) => rr.map(String)), method = ids.length === 2 ? 'cohen' : 'fleiss'
    const result = ids.length === 2 ? attributeAgreement(ratings.map((rr) => rr[0]!), { method: 'cohen', other: ratings.map((rr) => rr[1]!) }) : attributeAgreement(ratings, { method: 'fleiss' })
    if (!finite(result.kappa)) throw new RangeError('Kappa is undefined when every rating has the same category')
    const exact = ratings.filter((rr) => rr.every((x) => x === rr[0])).length / ratings.length
    r.metrics = [['Kappa', result.kappa], ['All-rater agreement', exact, 'percent'], ['Subjects', ratings.length], ['Raters', ids.length]]
    table(r, 'Attribute agreement', ['Method', 'Kappa', 'Subjects', 'Raters', 'All-rater agreement'], [[method === 'cohen' ? 'Cohen κ' : 'Fleiss κ', result.kappa, r.n, ids.length, exact]])
    table(r, 'Ratings used', ['Source row', ...ids.map((id) => label(i, id))], ratings.map((rr, j) => [r.sourceRows[j], ...rr]))
    const cats = [...new Set(ratings.flat())].sort(); table(r, 'Category counts by rater', ['Category', ...ids.map((id) => label(i, id))], cats.map((c) => [c, ...ids.map((_, j) => ratings.filter((rr) => rr[j] === c).length)]))
    bars(r, 'Category totals', cats.map((c) => ({ label: c, value: ratings.reduce((n, rr) => n + rr.filter((x) => x === c).length, 0) })), 'Ratings')
    r.details = { kappa: result.kappa, method, exactAgreement: exact }
    r.warnings.push('Rows must represent the same subjects across raters. These are unweighted nominal kappas; ordinal weighted kappa, repeated appraisal and reference-standard accuracy require separate procedures. No unvalidated kappa SE or p-value is reported.')
  } else throw new RangeError('Unknown MSA task')
  return r
}

/** Ordered, equally spaced series only. Missing values are never silently compressed. */
export function timeSeriesStudy(i: EngineeringInput): EngineeringResult {
  const r = base('timeseries', i), raw = get(i, i.y), conf = confidence(i), start = num(i.start, 1, 'start row', 1, raw.length, true), end = num(i.end, raw.length, 'end row', start, raw.length, true)
  const values = raw.slice(start - 1, end)
  if (values.some((v) => typeof v !== 'number' || !finite(v))) throw new RangeError('Time series: fill missing observations or choose a complete contiguous row interval; gaps cannot be dropped')
  const y = values as number[], n = y.length
  if (n < 5 || !(desc(y).sd > 0)) throw new RangeError('Time series: at least five nonconstant observations required')
  let times = y.map((_, j) => start + j), step = 1
  if (i.time) {
    if (i.time === i.y) throw new RangeError('Time and response columns must differ')
    const t = get(i, i.time); if (t.length !== raw.length) throw new RangeError('Time column length mismatch')
    const selected = t.slice(start - 1, end)
    if (selected.some((v) => typeof v !== 'number' || !finite(v))) throw new RangeError('Time index must be numeric without gaps')
    times = selected as number[]; step = times[1]! - times[0]!
    if (!(step > 0) || times.some((t, j) => j && Math.abs((t - times[j - 1]!) - step) > 1e-7 * Math.max(Math.abs(step), 1e-10))) throw new RangeError('Time index must be strictly increasing and equally spaced; no automatic sorting')
  }
  r.n = n; r.omitted = raw.length - n; r.sourceRows = y.map((_, j) => start + j)
  const hold = num(i.holdout, 0, 'holdout', 0, Math.max(0, n - 5), true), train = y.slice(0, n - hold), nt = train.length, horizon = num(i.horizon, 10, 'horizon', 1, 10000, true), lag = num(i.lag, Math.min(20, Math.floor(nt / 3)), 'lag', 1, nt - 1, true)
  const period = num(i.seasonLength, 12, 'season length', 2, 10000, true)
  let fitted: number[] = [], residuals: number[] = [], forecasts: number[] = [], lower: number[] | undefined, upper: number[] | undefined, modelDF = 0
  if (i.task === 'acf') {
    if (hold) throw new RangeError('ACF: holdout is not used; set it to zero')
    const a = acf(y, { maxLag: lag, confidence: conf }), p = pacf(y, { maxLag: lag, confidence: conf }), lb = ljungBox(y, { lags: lag })
    r.metrics = [['Observations', n], ['Ljung–Box Q', lb.statistic], ['p-value', lb.pValue, 'p'], ['Lags', lag]]
    table(r, 'Autocorrelation diagnostics', ['Lag', 'ACF', 'PACF', 'Lower band', 'Upper band'], a.lag.slice(1).map((k) => [k, a.acf[k], p.pacf[k], a.lower[k], a.upper[k]]))
    table(r, 'Ljung–Box test · raw series', ['Q', 'DF', 'p'], [[lb.statistic, lb.df, lb.pValue]], ['', '', 'p'])
    for (const [title, vv] of [['ACF', a.acf], ['PACF', p.pacf]] as const) line(r, title, 'Lag', 'Correlation', [{ name: title, points: a.lag.slice(1).map((k) => [k, vv[k]!]), markers: true }, { name: 'Upper band', points: a.lag.slice(1).map((k) => [k, a.upper[k]!]), dashed: true }, { name: 'Lower band', points: a.lag.slice(1).map((k) => [k, a.lower[k]!]), dashed: true }])
    r.warnings.push('ACF/PACF bands are individual ±z/√N white-noise reference bands, not simultaneous confidence intervals. Trends or seasonality can create autocorrelation; inspect stationarity before modelling.')
    r.details = { acf: a, pacf: p, ljungBox: lb }; return r
  }
  if (i.task === 'decompose') {
    if (hold) throw new RangeError('Decomposition: holdout is not used; set it to zero')
    const method = choose(i.method, 'additive', ['additive', 'multiplicative'] as const), d = decompose(y, { seasonLength: period, method })
    r.metrics = [['Observations', n], ['Season length', period], ['Method', method], ['Full seasons', Math.floor(n / period)]]
    table(r, 'Decomposition', ['Source row', 'Time', 'Observed', 'Trend', 'Seasonal', 'Remainder'], y.map((v, j) => [r.sourceRows[j], times[j], v, d.trend[j], d.seasonal[j], d.residual[j]]))
    table(r, 'Seasonal indices', ['Season position', 'Index'], d.seasonalIndices.map((v, j) => [j + 1, v]))
    line(r, 'Observed and centered moving-average trend', 'Time', label(i, i.y!), [{ name: 'Observed', points: xy(y, times) }, { name: 'Trend', points: xy(d.trend, times) }])
    line(r, 'Seasonal component', 'Time', 'Seasonality', [{ name: 'Seasonal', points: xy(d.seasonal, times) }]); line(r, 'Remainder', 'Time', 'Remainder', [{ name: 'Remainder', points: xy(d.residual, times) }])
    r.output = { name: 'Time-series decomposition', names: ['Source row', 'Time', 'Observed', 'Trend', 'Seasonal', 'Remainder'], types: Array(6).fill('number'), rows: r.tables[0]!.rows }
    r.details = { seasonalIndices: d.seasonalIndices }; r.warnings.push('Classical centered-moving-average decomposition; boundary trend and remainder are undefined and exported as missing, not zero. This decomposition does not itself forecast.'); return r
  }
  if (i.task === 'trend') {
    const model = choose(i.method, 'linear', ['linear', 'quadratic', 'exponential'] as const), d = trendAnalysis(train, { model, horizon: hold + horizon })
    fitted = d.fitted; forecasts = d.forecast; residuals = train.map((v, j) => v - fitted[j]!)
    table(r, 'Trend model parameters', ['Parameter', 'Estimate'], Object.entries(d.parameters)); r.details = { parameters: d.parameters, r2: d.r2 }
    r.warnings.push('Trend forecasts are point forecasts only. No unsupported confidence interval is manufactured; exponential trend uses the native log-scale fit.')
  } else if (i.task === 'ets') {
    const method = choose(i.method, 'ses', ['ses', 'des', 'winters-add', 'winters-mul'] as const)
    if (method.startsWith('winters') && nt < 2 * period) throw new RangeError('Holt–Winters: at least two complete training seasons required')
    const d = ets(train, { method, seasonLength: period, horizon: hold + horizon, confidence: conf })
    fitted = d.fitted; residuals = train.map((v, j) => v - fitted[j]!); forecasts = d.forecast; lower = d.forecastLower; upper = d.forecastUpper
    table(r, 'Smoothing parameters', ['Parameter', 'Estimate'], [['α', d.alpha], ['β', d.beta], ['γ', d.gamma], ['Residual sigma', d.sigma]])
    r.details = { method: d.method, alpha: d.alpha, beta: d.beta, gamma: d.gamma }
    r.warnings.push('ETS smoothing parameters are selected by Columna SSE grid search. Forecast intervals are approximate conditional intervals and exclude parameter-estimation uncertainty. Initialization affects in-sample errors.')
  } else if (i.task === 'arima') {
    const p = num(i.p, 1, 'p', 0, 5, true), d = num(i.d, 0, 'd', 0, 2, true), q = num(i.q, 0, 'q', 0, 5, true), method = choose(i.method, 'CSS', ['CSS', 'CSS-ML'] as const)
    if (nt - d < Math.max(12, 5 * (p + q + 1))) throw new RangeError('ARIMA: insufficient training observations for the selected order')
    const a = arima(train, { p, d, q, method, includeMean: i.includeMean ?? d === 0, horizon: hold + horizon, confidence: conf })
    fitted = a.fitted; residuals = a.residuals; forecasts = a.forecast; lower = a.forecastLower; upper = a.forecastUpper; modelDF = p + q
    if (fitted.length !== train.length) throw new RangeError('ARIMA source indexing mismatch')
    const terms = [...a.ar.map((v, j) => ['AR(' + (j + 1) + ')', v]), ...a.ma.map((v, j) => ['MA(' + (j + 1) + ')', v]), ...((i.includeMean ?? d === 0) ? [['Mean / drift', a.intercept]] : [])]
    table(r, 'ARIMA estimates', ['Parameter', 'Estimate', 'SE', 'z', 'p'], terms.map((v, j) => [...v, a.se[j], a.z[j], a.pValue[j]]), ['', '', '', '', 'p'])
    table(r, 'ARIMA fit criteria', ['Method', 'AIC', 'BIC', 'Log likelihood', 'Innovation variance'], [[a.method, a.aic, a.bic, a.logLik, a.sigma2]])
    r.details = { order: a.order, method: a.method, ar: a.ar, ma: a.ma, intercept: a.intercept, sigma2: a.sigma2, aic: a.aic, bic: a.bic }
    r.warnings.push('ARIMA is nonseasonal (p,d,q), conditional CSS or CSS-started likelihood. Seasonal AR/MA, external regressors and automatic order selection are not enabled. Inspect residuals and numerical stability; successful execution is not a convergence guarantee.')
  } else throw new RangeError('Unknown time-series task')
  if (forecasts.some((v) => !finite(v)) || !forecasts.length || fitted.filter(finite).length < 3) throw new RangeError('Forecasting failed: non-finite estimates; revise the model')
  const errors = residuals.filter(finite), trainRMSE = Math.sqrt(mean(errors.map((v) => v * v))), hErrors = hold ? y.slice(nt).map((v, j) => v - forecasts[j]!) : []
  r.metrics = [['Training N', nt], ['Training RMSE', trainRMSE], ['Holdout N', hold], [hold ? 'Holdout RMSE' : 'Forecast horizon', hold ? Math.sqrt(mean(hErrors.map((v) => v * v))) : horizon]]
  const futureTime = forecasts.map((_, j) => times[nt - 1]! + step * (j + 1)), forecastRows = forecasts.map((v, j) => [j < hold ? 'Holdout' : 'Future', futureTime[j], j < hold ? y[nt + j] : null, v, lower?.[j], upper?.[j], j < hold ? hErrors[j] : null])
  table(r, 'Forecast and validation', ['Segment', 'Time', 'Observed', 'Forecast', 'Lower PI', 'Upper PI', 'Error'], forecastRows)
  table(r, 'Fitted values and residuals', ['Source row', 'Time', 'Observed', 'Fitted', 'Residual'], train.map((v, j) => [start + j, times[j], v, fitted[j], residuals[j]]))
  if (hold) table(r, 'Out-of-sample validation', ['Holdout N', 'RMSE', 'MAE'], [[hold, Math.sqrt(mean(hErrors.map((v) => v * v))), mean(hErrors.map(Math.abs))]])
  line(r, 'Series, fitted values and forecast', 'Time', label(i, i.y!), [{ name: 'Observed', points: xy(y, times) }, { name: 'Fitted', points: xy(fitted, times.slice(0, nt)) }, { name: 'Forecast', points: xy(forecasts, futureTime) }, ...(lower && upper ? [{ name: 'Lower PI', points: xy(lower, futureTime), dashed: true }, { name: 'Upper PI', points: xy(upper, futureTime), dashed: true }] : [])])
  line(r, 'Residuals in original order', 'Time', 'Residual', [{ name: 'Residual', points: xy(residuals, times.slice(0, nt)) }])
  if (errors.length > lag && desc(errors).sd > 0) {
    const a = acf(errors, { maxLag: lag, confidence: conf }); line(r, 'Residual ACF', 'Lag', 'ACF', [{ name: 'Residual ACF', points: a.lag.slice(1).map((k) => [k, a.acf[k]!]) }, { name: 'Upper band', points: a.lag.slice(1).map((k) => [k, a.upper[k]!]), dashed: true }, { name: 'Lower band', points: a.lag.slice(1).map((k) => [k, a.lower[k]!]), dashed: true }])
    if (lag > modelDF) { const lb = ljungBox(errors, { lags: lag, df: modelDF }); table(r, 'Residual Ljung–Box (approximate)', ['Lags', 'Model DF deducted', 'Q', 'DF', 'p'], [[lag, modelDF, lb.statistic, lb.df, lb.pValue]], ['', '', '', '', 'p']) }
  }
  r.output = { name: 'Forecast · ' + i.task, names: r.tables.find((t) => t.title === 'Forecast and validation')!.headers, types: ['text', 'number', 'number', 'number', 'number', 'number', 'number'], rows: forecastRows.map((x) => x.map(val)) }
  r.warnings.push('Row order is chronological. The selected interval must be complete and equally spaced. No missing periods are silently removed; model warm-up residuals may be undefined.', 'With holdout > 0, the last observations are excluded from fitting. Forecast intervals are not simultaneous bands. Future predictions are conditional on model assumptions.')
  return r
}

export function reliabilityStudy(i: EngineeringInput): EngineeringResult {
  const r = base('reliability', i), conf = confidence(i), ids = [i.y!, ...(i.censor ? [i.censor] : []), ...(i.group ? [i.group] : [])]
  const { rows } = complete(i, ids, [i.y!, ...(i.censor ? [i.censor] : [])], 3, r)
  const times = rows.map((rr) => rr[0] as number), censor = rows.map((rr) => i.censor ? rr[1] as number : 0), groups = rows.map((rr) => i.group ? String(rr.at(-1)) : 'All observations')
  if (times.some((t) => t < 0) || censor.some((c) => c !== 0 && c !== 1)) throw new RangeError('Reliability: nonnegative times; censor codes exactly 0 = failure, 1 = right-censored')
  const levels = [...new Set(groups)].sort(), curves = levels.map((g) => { const idx = groups.map((v, j) => v === g ? j : -1).filter((j) => j >= 0), t = idx.map((j) => times[j]!), c = idx.map((j) => censor[j]!); return { group: g, times: t, censor: c, km: kaplanMeier(t, { censor: c, confidence: conf }) } })
  const events = censor.filter((c) => c === 0).length
  r.metrics = [['Units', times.length], ['Failures', events], ['Right-censored', times.length - events], ['Groups', levels.length]]
  table(r, 'Risk set and survival estimates', ['Group', 'Time', 'At risk', 'Failures', 'Censored', 'Survival', 'SE', 'Lower CI', 'Upper CI'], curves.flatMap((g) => g.km.curve.map((p) => [g.group, p.time, p.atRisk, p.events, p.censored, p.survival, p.se, p.lower, p.upper])))
  const empiricalSeries = curves.flatMap((g) => [{ name: g.group, step: true, points: [[0, 1] as [number, number], ...g.km.curve.map((p): [number, number, string] => [p.time, p.survival, 'Risk=' + p.atRisk + '; failures=' + p.events + '; censored=' + p.censored])] }, { name: g.group + ' · censored', markers: true, points: g.km.curve.filter((p) => (p.censored ?? 0) > 0).map((p): [number, number, string] => [p.time, p.survival, 'Censored=' + p.censored]) }])
  line(r, 'Kaplan–Meier survival', 'Time', 'S(t)', empiricalSeries); r.plots.at(-1)!.fixedY = [0, 1]
  table(r, 'Survival summary', ['Group', 'N', 'Failures', 'Censored', 'Median survival'], curves.map((g) => [g.group, g.km.n, g.km.nEvents, g.km.n - g.km.nEvents, g.km.curve.find((p) => p.survival <= .5)?.time ?? null]))
  if (i.task === 'km' || i.task === 'logrank') {
    if (i.task === 'logrank') {
      if (levels.length < 2 || events === 0) throw new RangeError('Log-rank: at least two groups and observed failures required')
      const test = logRank(times, groups, { censor, weight: choose(i.method, 'logrank', ['logrank', 'wilcoxon', 'tarone-ware'] as const) })
      if (!finite(test.statistic) || !finite(test.pValue)) throw new RangeError('Log-rank: singular event/risk information; comparison is not estimable')
      table(r, 'Survival curve comparison', ['Method', 'Statistic', 'DF', 'p'], [[test.test, test.statistic, test.df, test.pValue]], ['', '', '', 'p']); r.metrics[3] = ['Comparison p-value', test.pValue, 'p']; r.details = test
      table(r, test.weight === 'logrank' ? 'Observed and expected failures' : 'Weighted observed and expected failures', ['Group', 'Observed', 'Expected'], test.groups.map((g, j) => [g, test.observed[j], test.expected[j]]))
    }
    if (levels.length === 1) { const c = curves[0]!.km.curve; line(r, 'Survival confidence limits · log-log Greenwood', 'Time', 'S(t)', [{ name: 'S(t)', step: true, points: [[0, 1], ...c.map((p): [number, number] => [p.time, p.survival])] }, { name: 'Lower CI', step: true, dashed: true, points: c.map((p) => [p.time, p.lower]) }, { name: 'Upper CI', step: true, dashed: true, points: c.map((p) => [p.time, p.upper]) }]); r.plots.at(-1)!.fixedY = [0, 1] }
  } else if (i.task === 'parametric') {
    const distribution = choose(i.method, 'weibull', ['weibull', 'lognormal', 'exponential'] as const), mission = num(i.mission, times.reduce((a, b) => Math.max(a, b), 0), 'mission time', Number.MIN_VALUE)
    const fitted = curves.map((g) => {
      const fails = g.censor.filter((c) => c === 0).length
      if (g.times.some((t) => !(t > 0)) || fails < (distribution === 'exponential' ? 1 : 2) || distribution !== 'exponential' && !(desc(g.times).sd > 0)) throw new RangeError('Parametric reliability: positive times, variation and sufficient failures are required in every group')
      const fit = reliabilityFit(g.times, { distribution, method: 'MLE', censor: g.censor, confidence: conf, percentiles: [.01, .05, .1, .5, .9] })
      if (!finite(fit.scale) || !(fit.scale > 0) || !finite(fit.logLik) || fit.shape !== undefined && (!(fit.shape > 0) || !finite(fit.shape))) throw new RangeError('Parametric reliability: fit failed')
      return { group: g.group, fit, mission: warrantyPrediction(fit, { warranty: mission }) }
    })
    table(r, 'Parametric lifetime model · MLE', ['Group', 'Distribution', 'Shape / σlog', 'Scale', 'Log likelihood', 'AIC', 'Mission time', 'R(mission)'], fitted.map((g) => [g.group, g.fit.distribution, g.fit.shape, g.fit.scale, g.fit.logLik, 2 * (distribution === 'exponential' ? 1 : 2) - 2 * g.fit.logLik, mission, g.mission.reliability]))
    table(r, 'Estimated failure percentiles', ['Group', 'Failure probability', 'Time'], fitted.flatMap((g) => g.fit.percentiles!.map((p) => [g.group, p.p, p.time])))
    table(r, 'Parameter Wald intervals', ['Group', 'Parameter', 'Lower CI', 'Upper CI', 'SE'], fitted.flatMap((g) => Object.entries(g.fit.ci ?? {}).map(([key, ci]) => [g.group, key, ...ci, g.fit.se?.[key]])))
    const maxTime = times.reduce((a, b) => Math.max(a, b), mission), grid = Array.from({ length: 151 }, (_, j) => maxTime * j / 150)
    line(r, 'Fitted survival vs Kaplan–Meier', 'Time', 'S(t)', [...fitted.map((g) => ({ name: g.group + ' · ' + distribution, points: grid.map((t): [number, number] => [t, t === 0 ? 1 : warrantyPrediction(g.fit, { warranty: t }).reliability]) })), ...curves.map((g) => ({ name: g.group + ' · KM', step: true, dashed: true, points: [[0, 1] as [number, number], ...g.km.curve.map((p): [number, number] => [p.time, p.survival])] }))]); r.plots.at(-1)!.fixedY = [0, 1]
    r.details = fitted; r.warnings.push('Parametric MLE assumes the chosen distribution and independent non-informative censoring. Weibull/lognormal/exponential fits are not automatic distribution selection. Percentiles are point estimates; parameter Wald intervals may be unreliable with few failures.')
  } else throw new RangeError('Unknown reliability task')
  r.output = { name: 'Reliability risk table', names: r.tables[0]!.headers, types: ['text', ...Array(8).fill('number')], rows: r.tables[0]!.rows }
  r.warnings.push('Censor coding: 0 = observed failure; 1 = right-censored. Missing status is omitted with its time, never converted to a failure. Left/interval censoring and competing risks are not enabled.', 'Tied failures are evaluated before censoring at the same time. Censor marks do not lower survival. A missing median means the curve never reached 0.5; it is not zero.', 'At survival 0 or 1, displayed Greenwood limits are boundary conventions; they are not distribution-free guarantees beyond follow-up.')
  return r
}

export function multivariateStudy(i: EngineeringInput): EngineeringResult {
  const r = base('multivariate', i), ids = i.columns ?? [], conf = confidence(i)
  if (ids.length < 2) throw new RangeError('Multivariate: select at least two numeric variables')
  const { rows, keep } = complete(i, ids, ids, 3, r), raw = rows as number[][], p = ids.length
  const ds = ids.map((_, j) => desc(raw.map((rr) => rr[j]!)))
  if (ds.some((s) => !(s.sd > 0))) throw new RangeError('Multivariate: remove constant variables explicitly')
  const scale = i.task === 'factor' || i.scale !== false, X = raw.map((rr) => rr.map((v, j) => scale ? (v - ds[j]!.mean) / ds[j]!.sd : v)), names = ids.map((id) => label(i, id))
  table(r, 'Preprocessing', ['Variable', 'N', 'Mean', 'Sample SD', 'Standardized'], names.map((name, j) => [name, raw.length, ds[j]!.mean, ds[j]!.sd, scale ? 'Yes' : 'No']))
  if (i.task === 'pca') {
    const count = num(i.components, Math.min(2, p), 'components', 1, p, true), result = pca(raw, { scale, nComponents: p, names })
    r.metrics = [['Observations', r.n], ['Variables', p], ['Retained PCs', count], ['Cumulative variance', result.cumulative[count - 1]!, 'percent']]
    table(r, 'Principal component variance', ['PC', 'Eigenvalue', 'Proportion', 'Cumulative'], result.eigenvalues.map((v, j) => [j + 1, v, result.proportion[j], result.cumulative[j]]))
    table(r, 'Eigenvectors / loadings', ['Variable', ...Array.from({ length: count }, (_, j) => 'PC' + (j + 1))], result.loadings.map((v, j) => [names[j], ...v.slice(0, count)]))
    table(r, 'PCA scores', ['Source row', ...Array.from({ length: count }, (_, j) => 'PC' + (j + 1))], result.scores.map((v, j) => [keep[j], ...v.slice(0, count)]))
    line(r, 'Scree plot', 'Component', 'Eigenvalue', [{ name: 'Eigenvalue', points: xy(result.eigenvalues) }])
    if (count >= 2) { scatter(r, 'PCA score plot', 'PC1', 'PC2', [{ name: 'Scores', points: result.scores.map((v, j) => [v[0]!, v[1]!, 'Row ' + keep[j]]) }]); scatter(r, 'Loading plot', 'PC1 eigenvector', 'PC2 eigenvector', [{ name: 'Variables', points: result.loadings.map((v, j) => [v[0]!, v[1]!, names[j]!]) }]) }
    r.output = { name: 'PCA scores', names: ['Source row', ...Array.from({ length: count }, (_, j) => 'PC' + (j + 1))], types: Array(count + 1).fill('number'), rows: result.scores.map((rr, j) => [keep[j]!, ...rr.slice(0, count)]) }
    r.details = { eigenvalues: result.eigenvalues, loadings: result.loadings, proportion: result.proportion }
    r.warnings.push('PCA loadings here are unit-length eigenvectors, not variable–component correlations. Component signs are arbitrary; rotations of tied eigenvalues are not unique. PCA is descriptive, not proof of a causal latent mechanism.')
  } else if (i.task === 'factor') {
    const count = num(i.components, Math.min(2, p - 1), 'factors', 1, p - 1, true), rotate = choose(i.method, 'varimax', ['none', 'varimax'] as const)
    const result = factorAnalysis(raw, { nFactors: count, method: 'pca', rotate, names }), loadings = result.rotatedLoadings ?? result.loadings
    r.metrics = [['Observations', r.n], ['Variables', p], ['Factors', count], ['Extraction', 'PCA · correlation']]
    table(r, 'Factor loadings · PCA extraction', ['Variable', ...Array.from({ length: count }, (_, j) => 'Factor ' + (j + 1)), 'Communality', 'Uniqueness'], loadings.map((v, j) => [names[j], ...v, v.reduce((s, x) => s + x * x, 0), result.uniqueVariances[j]]))
    if (count >= 2) scatter(r, 'Rotated factor loadings', 'Factor 1', 'Factor 2', [{ name: 'Variables', points: loadings.map((v, j) => [v[0]!, v[1]!, names[j]!]) }])
    else bars(r, 'Factor loadings', loadings.map((v, j) => ({ label: names[j]!, value: v[0]! })), 'Loading')
    r.details = result; r.warnings.push('Exploratory factor analysis uses PCA extraction from the correlation matrix and optional orthogonal varimax rotation. This is not maximum-likelihood or confirmatory factor analysis; no likelihood-fit p-value is reported.')
  } else if (i.task === 'kmeans' || i.task === 'hierarchical') {
    const k = num(i.k, 3, 'clusters', 2, raw.length - 1, true)
    if (new Set(X.map((rr) => JSON.stringify(rr))).size < k) throw new RangeError('Clustering: fewer distinct observations than requested clusters')
    let membership: number[], centers: number[][] = [], info: unknown
    if (i.task === 'kmeans') {
      const starts = num(i.starts, 10, 'starts', 1, 1000, true), seed = num(i.seed, 42, 'seed', 0, 4294967295, true), maxIter = num(i.maxIter, 100, 'maxIter', 1, 10000, true)
      let best: ReturnType<typeof kmeans> | undefined
      for (let s = 0; s < starts; s++) { const result = kmeans(X, { k, seed: (seed + s * 2654435761) >>> 0, maxIter }); if (result.sizes.every((n) => n > 0) && (!best || result.inertia < best.inertia)) best = result }
      if (!best) throw new RangeError('k-means: all starts produced an empty cluster; revise k or increase starts')
      membership = best.cluster.map((c) => c + 1); centers = best.centers; info = best
      r.metrics = [['Observations', r.n], ['Clusters', k], ['Within-cluster SSE', best.inertia], ['Starts', starts]]
      if (best.iterations > maxIter) r.warnings.push('The best k-means run reached the iteration budget; convergence is not established.')
      r.warnings.push('k-means uses seeded k-means++ starts and chooses the lowest within-cluster SSE among nonempty solutions. Multiple starts reduce, but do not eliminate, local minima.')
    } else {
      const budget = num(i.memoryMB, 256, 'distance-matrix memory budget', 1, 8192), method = choose(i.method, 'ward', ['single', 'complete', 'average', 'ward'] as const)
      if (8 * raw.length * raw.length > budget * 1024 * 1024) throw new RangeError('Hierarchical clustering exceeds the chosen distance-matrix memory budget; increase it or use k-means')
      const tree = hclust(X, { method }); membership = cutTree(tree, X.length, k); info = tree
      centers = Array.from({ length: k }, (_, c) => ids.map((_, j) => mean(X.filter((_, h) => membership[h] === c + 1).map((rr) => rr[j]!))))
      r.metrics = [['Observations', r.n], ['Clusters', k], ['Linkage', method], ['Matrix MB', 8 * raw.length ** 2 / 1024 ** 2]]
      table(r, 'Hierarchical merges', ['Merge', 'Left node', 'Right node', 'Height'], tree.merge.map((m, j) => [j, ...m]))
      r.plots.push({ type: 'eng-dendrogram', title: 'Hierarchical dendrogram', merge: tree.merge, order: tree.order, labels: keep.map(String), yLabel: 'Linkage height', note: 'All merge branches are drawn; dense leaf labels are thinned for readability.' })
      r.warnings.push('Hierarchical clustering uses an O(N²) distance matrix; its explicit memory budget is not a worksheet-size limit. Native Columna additionally supports at most 8000 observations in this algorithm. Negative merge IDs are −1−leafIndex; nonnegative IDs refer to zero-based earlier merges.')
    }
    const sizes = Array.from({ length: k }, (_, c) => membership.filter((v) => v === c + 1).length)
    table(r, 'Cluster centers (analysis scale)', ['Cluster', 'N', ...names], centers.map((c, j) => [j + 1, sizes[j], ...c]))
    table(r, 'Cluster centers (original units)', ['Cluster', 'N', ...names], centers.map((c, j) => [j + 1, sizes[j], ...c.map((v, h) => scale ? v * ds[h]!.sd + ds[h]!.mean : v)]))
    table(r, 'Cluster assignments', ['Source row', 'Cluster', ...names], raw.map((rr, j) => [keep[j], membership[j], ...rr]))
    const pc = pca(raw, { scale, nComponents: 2 })
    scatter(r, 'Clusters projected onto two PCs', 'PC1', 'PC2', Array.from({ length: k }, (_, c) => ({ name: 'Cluster ' + (c + 1), points: pc.scores.flatMap((rr, j): Array<[number, number, string]> => membership[j] === c + 1 ? [[rr[0]!, rr[1]!, 'Row ' + keep[j]]] : []) })))
    r.output = { name: 'Cluster assignments', names: ['Source row', 'Cluster', ...names], types: Array(p + 2).fill('number'), rows: raw.map((rr, j) => [keep[j]!, membership[j]!, ...rr]) }
    r.details = { membership, sizes, fit: info }; r.warnings.push('Euclidean distances are computed on all selected variables. The two-PC plot is only a projection, not the space in which clustering was performed. Cluster labels are arbitrary; no classification accuracy is claimed.')
  } else throw new RangeError('Unknown multivariate task')
  r.warnings.push('Complete-case rows are selected jointly across variables; columns are never cleaned independently. Scaling is explicit. The output worksheet includes original source-row indices.')
  void conf
  return r
}

export function engineeringStudy(domain: EngineeringDomain, input: EngineeringInput): EngineeringResult {
  if (domain === 'doe') return doeStudy(input)
  if (domain === 'msa') return msaStudy(input)
  if (domain === 'timeseries') return timeSeriesStudy(input)
  if (domain === 'reliability') return reliabilityStudy(input)
  if (domain === 'multivariate') return multivariateStudy(input)
  throw new RangeError('Unknown engineering domain')
}
