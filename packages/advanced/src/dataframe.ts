/**
 * Column-level entry points for the advanced statistics, attached to `DataFrame` / `LazyFrame` from
 * @columna/core. Importing @columna/advanced (or `columna/advanced`) installs them:
 *
 *   import '@columna/advanced'
 *   df.ttest('x', { mu: 50 }); df.anova('y', 'factor'); await lazy.tukey('y', 'factor')
 */
import { DataFrame, LazyFrame } from '@columna/core'
import {
  anova as anovaTest,
  bonett as bonettTest,
  chi2test as chi2Independence,
  crosstab,
  equalVariances as equalVariancesTest,
  ttest1,
  ttest2,
  ttestPaired,
  type Alternative,
  type AnovaResult,
  type Chi2Result,
  type TTestOptions,
  type TTestResult,
  type VarianceTestResult,
} from './tests.js'
import { normalityTest as normalityTestFn, type NormalityMethod, type NormalityResult } from './normality.js'
import { tukeyHSD, type TukeyResult } from './tukey.js'
import { dunnett as dunnettTest, fisherLSD, gamesHowell as gamesHowellFn, hsuMCB, type DunnettResult, type FisherResult, type HsuResult } from './multcomp.js'
import { kruskal as kruskalTest, mannWhitney as mannWhitneyTest, type KruskalResult, type MannWhitneyResult } from './nonparametric.js'
import { corrTest as corrTestFn, dixon as dixonFn, grubbs as grubbsFn, partialCorr as partialCorrFn, propTest1, propTest2, varTest1 as varTest1Fn, ztest1 as ztest1Fn, type CorrTestResult, type OutlierResult, type PartialCorrResult, type PropTestResult, type VarTest1Result, type ZTestResult } from './basic.js'
import { friedman as friedmanFn, moodMedian as moodFn, runsTest as runsFn, signTest as signFn, wilcoxonSigned as wilcoxonFn, type FriedmanResult, type MoodResult, type RunsTestResult, type SignTestResult, type WilcoxonResult } from './nonparametric2.js'
import { tost1, tost2, tostPaired, type TostResult } from './equivalence.js'
import { fittedLine as fittedLineFn, ols, type OlsOptions, type OlsResult } from './regression.js'
import { bestSubsets as bestSubsetsFn, stepwise as stepwiseFn, type BestSubsetsResult, type StepwiseOptions, type StepwiseResult } from './stepwise.js'
import { glm as glmFn, mlogit as mlogitFn, ologit as ologitFn, type GlmOptions, type GlmResult, type NominalResult, type OrdinalResult } from './glm.js'
import { linearModel as linearModelFn, type LinearModelResult } from './lm.js'
import { nls as nlsFn, type NlsModel, type NlsOptions, type NlsResult } from './nls.js'
import { orthogonalRegression as orthogonalFn, pls as plsFn, type OrthogonalResult, type PlsOptions, type PlsResult } from './pls.js'
import { controlChart as controlChartFn, ewma as ewmaFn, cusum as cusumFn, type ControlChartOptions, type ControlChartResult, type CusumResult, type EwmaResult } from './spc.js'
import { capability as capabilityFn, toleranceInterval as toleranceFn, type CapabilityResult, type ToleranceIntervalResult } from './capability.js'
import { gageRR as gageRRFn, type GageRRResult } from './msa.js'
import { pareto as paretoFn, runChart as runChartFn, individualDistributionID as idiFn, type ParetoResult, type RunChartResult, type IdResult } from './quality.js'
import { trendAnalysis as trendFn, decompose as decompFn, stl as stlFn, ets as etsFn, acf as acfFn, arima as arimaFn, autoArima as autoArimaFn, type TrendResult, type DecompositionResult, type StlResult, type EtsResult, type AcfResult, type ArimaResult } from './timeseries.js'
import { reliabilityFit as relFitFn, kaplanMeier as kmFn, logRank as logRankFn, coxPH as coxPHFn, fineGray as fineGrayFn, type ParametricSurvival, type KaplanMeierResult, type LogRankResult } from './reliability.js'
import { mixedModel as mixedFn, glmm as glmmFn } from './mixed.js'
import { manova as manovaFn } from './manova.js'
import { anovaTwoWay as anovaTwoWayFn } from './anova2.js'
import { nestedAnova as nestedAnovaFn } from './doe.js'
import { capabilitySixpack as sixpackFn } from './capability.js'
import { boxplotStats as boxplotFn, descriptiveStats as descFn, graphicalSummary as gsFn, interactionPlot as interFn, intervalPlot as intervalFn, mainEffectsPlot as mainFn, poissonGof as poissonGofFn } from './descriptive.js'
import { stabilityStudy as stabilityFn, type StabilityOptions, type StabilityResult } from './stability.js'
import { gChart as gChartFn, mewma as mewmaFn, t2Chart as t2Fn, tChart as tChartFn } from './spc2.js'
import { periodogram as periodogramFn } from './spectral.js'
import { altRegression as altFn, lifeRegression as lifeRegFn, powerLawNHPP as nhppFn, probitAnalysis as probitFn } from './reliability2.js'
import { clusterVariables as clusterVarsFn, itemAnalysis as itemFn, multipleCorrespondence as mcaFn } from './multivariate2.js'
import { manovaModel as manovaModelFn } from './manova.js'
import { autoModel as autoModelFn, crossValidate as cvFn, type ModelKind } from './automl.js'
import { pca as pcaFn, kmeans as kmeansFn, type PcaResult, type KMeansResult } from './multivariate.js'
import { cart as cartFn, randomForest as rfFn, type CartResult, type RandomForestResult, type CartOptions } from './predictive.js'
import { ancova as ancovaFn, type AncovaResult } from './anova2.js'
import { adfTest as adfTestFn, kpssTest as kpssTestFn, type UnitRootResult } from './unitroot.js'
import { ksTwoSample as ksTwoSampleFn, type KsTwoSampleResult } from './gof2.js'
import { ridge as ridgeFn, lasso as lassoFn, type PenalizedResult } from './penalized.js'

void bonettTest

export type MannWhitneyOptions = { alternative?: Alternative; confidence?: number; method?: 'auto' | 'exact' | 'asymptotic' }

export type DataFramePropTestOptions = {
  /** Denominator column (summed with `eventsCol` for one sample, or per group for two samples). Required without `by`. */
  trials?: string
  /** Exactly two levels → two-sample test on aggregated events / trials per level. */
  by?: string
  p0?: number
  alternative?: Alternative
  confidence?: number
  method?: 'exact' | 'normal' | 'fisher'
  pooled?: boolean
}

/** Column-based t-test: one-sample by default, two-sample with `by` (a 2-level column), paired with `paired`. */
export type DataFrameTTestOptions = TTestOptions & {
  /** Grouping column with exactly two levels → two-sample test, difference = first level − second (sorted). */
  by?: string
  /** Second column of the same length → paired test on column − paired. */
  paired?: string
  /** Two-sample only: pool variances instead of Welch. */
  equalVar?: boolean
}

type Num = Array<number | null>

/** Values of `column` split by the levels of `by` (sorted level names; null levels dropped). */
function groupsOf(df: DataFrame, column: string, by: string): Record<string, Num> {
  const x = df.getColumn(column).toArray() as Num
  const g = df.getColumn(by).toArray()
  const groups = new Map<string, Num>()
  for (let i = 0; i < x.length; i++) {
    if (g[i] === null) continue
    const k = String(g[i])
    let arr = groups.get(k)
    if (!arr) groups.set(k, (arr = []))
    arr.push(x[i]!)
  }
  return Object.fromEntries([...groups.entries()].sort(([p], [q]) => (p < q ? -1 : p > q ? 1 : 0)))
}

/** Sum numeric `valueCol` and optional `trialsCol` per level of `by` (sorted). Without `trialsCol`, each row counts as one trial and `valueCol` is summed as events. */
function propAggByGroup(
  df: DataFrame,
  eventsCol: string,
  by: string,
  trialsCol?: string,
): Record<string, { events: number; trials: number }> {
  const e = df.getColumn(eventsCol).toArray()
  const g = df.getColumn(by).toArray()
  const t = trialsCol ? df.getColumn(trialsCol).toArray() : null
  const map = new Map<string, { events: number; trials: number }>()
  for (let i = 0; i < e.length; i++) {
    if (g[i] === null) continue
    const k = String(g[i])
    let slot = map.get(k)
    if (!slot) map.set(k, (slot = { events: 0, trials: 0 }))
    if (t) {
      const ev = e[i]
      const tr = t[i]
      if (typeof ev !== 'number' || typeof tr !== 'number' || !Number.isFinite(ev) || !Number.isFinite(tr)) continue
      slot.events += ev
      slot.trials += tr
    } else {
      const ev = e[i]
      if (typeof ev !== 'number' || !Number.isFinite(ev)) continue
      slot.events += ev
      slot.trials += 1
    }
  }
  return Object.fromEntries([...map.entries()].sort(([p], [q]) => (p < q ? -1 : p > q ? 1 : 0)))
}

function sumNumericCol(df: DataFrame, column: string): number {
  let s = 0
  for (const v of df.getColumn(column).toArray()) {
    if (typeof v === 'number' && Number.isFinite(v)) s += v
  }
  return s
}

declare module '@columna/core' {
  interface DataFrame {
    /**
     * t-test on a column (Minitab 1-Sample t / 2-Sample t / Paired t):
     *   df.ttest('x', { mu: 50 })                  // one-sample, H0: mean = 50
     *   df.ttest('x', { by: 'group' })             // two-sample (Welch) between the two levels of group
     *   df.ttest('after', { paired: 'before' })    // paired, after − before
     */
    ttest(column: string, options?: DataFrameTTestOptions): TTestResult
    /** One-way ANOVA of `column` across the levels of `by` (Minitab One-Way ANOVA). */
    anova(column: string, by: string): AnovaResult
    /** Two-way ANOVA Type III via `anovaTwoWay`. */
    anovaTwoWay(y: string, row: string, col: string, options?: { interaction?: boolean }): import('./anova2.js').AnovaTwoWayResult
    /** Tukey HSD pairwise comparisons across the levels of `by`. */
    tukey(column: string, by: string, options?: { alpha?: number }): TukeyResult
    /** Fisher LSD pairwise comparisons (individual error rate). */
    fisher(column: string, by: string, options?: { alpha?: number }): FisherResult & { familyAlpha: number }
    /** Dunnett comparisons of every level of `by` against `control`. */
    dunnett(column: string, by: string, options: { control: string; alpha?: number; alternative?: Alternative }): DunnettResult
    /** Hsu's multiple comparisons with the best. */
    hsu(column: string, by: string, options?: { best?: 'largest' | 'smallest'; alpha?: number }): HsuResult
    /** Mann–Whitney test between the two levels of `by` (first level − second, sorted). */
    mannWhitney(column: string, by: string, options?: MannWhitneyOptions): MannWhitneyResult
    /** Kruskal–Wallis test across the levels of `by`. */
    kruskal(column: string, by: string): KruskalResult
    /** Test for equal variances: Levene / Brown–Forsythe (default, as Minitab), Bartlett, or Bonett. */
    equalVariances(column: string, by: string, method?: 'levene' | 'bartlett' | 'bonett'): VarianceTestResult
    /** Normality test: Anderson–Darling (default, as Minitab), Ryan–Joiner, Kolmogorov–Smirnov or Shapiro–Wilk. */
    normalityTest(column: string, method?: NormalityMethod): NormalityResult
    /** Chi-square test of independence between two categorical columns. */
    chi2test(a: string, b: string, options?: { correction?: boolean }): Chi2Result
    /** 1-Sample Z with known sigma. */
    ztest(column: string, options: { sigma: number; mu?: number; alternative?: Alternative; confidence?: number }): ZTestResult
    /** 1 Variance: χ² (default) or Bonett. */
    varTest(column: string, options: { sigma0: number; alternative?: Alternative; confidence?: number; method?: 'chi-square' | 'bonett' }): VarTest1Result
    /** Correlation between two columns with p-value and CI. */
    corrTest(a: string, b: string, options?: { method?: 'pearson' | 'spearman' | 'kendall'; alternative?: Alternative; confidence?: number }): CorrTestResult
    /** Partial correlation controlling for one or more columns. */
    partialCorr(a: string, b: string, controls: string | string[], options?: { method?: 'pearson' | 'spearman'; alternative?: Alternative }): PartialCorrResult
    /** Outlier test: Grubbs (default) or Dixon. */
    outlierTest(column: string, options?: { method?: 'grubbs' | 'dixon'; alpha?: number; alternative?: 'two-sided' | 'min' | 'max' }): OutlierResult
    /** 1-Sample Sign test. */
    signTest(column: string, options?: { median?: number; alternative?: Alternative; confidence?: number }): SignTestResult
    /** 1-Sample Wilcoxon signed-rank test. */
    wilcoxon(column: string, options?: { median?: number; alternative?: Alternative; confidence?: number; method?: 'auto' | 'exact' | 'asymptotic' }): WilcoxonResult
    /** Mood's median test across the levels of `by`. */
    mood(column: string, by: string): MoodResult
    /** Friedman test: `column` by `treatment` within `block` (one observation per cell). */
    friedman(column: string, treatment: string, block: string): FriedmanResult
    /** Runs test for randomness about the mean (or `k`). */
    runsTest(column: string, options?: { k?: number; correction?: boolean }): RunsTestResult
    /**
     * Equivalence test (TOST): one-sample by default, two-sample with `by` (test level − reference level,
     * pass `reference` to name it), paired with `paired`.
     */
    equivalence(column: string, options: { limits: [number, number]; alpha?: number; by?: string; reference?: string; paired?: string; equalVar?: boolean }): TostResult
    /** Regression (OLS) of `y` on the predictor columns: coefficient table, ANOVA, R², diagnostics, predict(). */
    regress(y: string, predictors: string[], options?: Omit<OlsOptions, 'names' | 'weights'> & { weights?: string }): OlsResult
    /** Fitted Line Plot: polynomial (degree 1–3) regression of y on x with the fitted curve. */
    fittedLine(x: string, y: string, options?: { degree?: 1 | 2 | 3; logX?: boolean; logY?: boolean; confidence?: number }): OlsResult & { degree: number; curve(x: number): number; equation: string }
    /** Stepwise / forward / backward selection among the predictor columns. */
    stepwise(y: string, predictors: string[], options?: Omit<StepwiseOptions, 'names'>): StepwiseResult
    /** Best Subsets over the predictor columns. */
    bestSubsets(y: string, predictors: string[], options?: { maxK?: number; nBest?: number; include?: string[] }): BestSubsetsResult
    /** Generalized linear model; `family: 'binomial'` is Binary Logistic, `'poisson'` Poisson Regression. `trials` / `offset` / `weights` name columns. */
    glm(y: string, predictors: string[], options: Omit<GlmOptions, 'names' | 'trials' | 'offset' | 'weights'> & { trials?: string; offset?: string; weights?: string }): GlmResult
    /** Binary Logistic Regression (glm with the binomial family). */
    logistic(y: string, predictors: string[], options?: Omit<GlmOptions, 'family' | 'names' | 'trials' | 'offset' | 'weights'> & { trials?: string; offset?: string; weights?: string }): GlmResult
    /** Ordinal Logistic Regression (proportional odds) of a categorical / integer response. */
    ologit(y: string, predictors: string[], options?: { confidence?: number; maxIter?: number }): OrdinalResult
    /** Nominal Logistic Regression (multinomial). */
    mlogit(y: string, predictors: string[], options?: { reference?: string | number; confidence?: number; maxIter?: number }): NominalResult
    /** General Linear Model from a formula over the columns: `df.linearModel('y ~ a*b + x')`. */
    linearModel(formula: string, options?: { factors?: string[]; confidence?: number }): LinearModelResult
    /** Nonlinear regression of column `y` on column(s) `x` with a model function and start values. */
    nls(model: NlsModel, x: string | string[], y: string, options: NlsOptions): NlsResult
    /** Orthogonal (Deming) regression of y on x. */
    orthogonalRegression(x: string, y: string, options?: { errorVarianceRatio?: number; confidence?: number }): OrthogonalResult
    /** Partial least squares of one or more response columns on the predictor columns. */
    pls(y: string | string[], predictors: string[], options?: Omit<PlsOptions, 'names' | 'responseNames'>): PlsResult
    /** Control chart on a column (I-MR by default; pass `subgroup` / `type` for X̄-R / X̄-S / attributes). */
    controlChart(column: string, options?: ControlChartOptions): ControlChartResult
    /** EWMA chart. */
    ewma(column: string, options?: { lambda?: number; L?: number; mu?: number; sigma?: number }): EwmaResult
    /** Tabular CUSUM chart. */
    cusum(column: string, options?: { k?: number; h?: number; mu?: number; sigma?: number; vMask?: boolean }): CusumResult
    /** Process capability (normal) of a column. */
    capability(column: string, options: { lsl?: number; usl?: number; target?: number; subgroup?: number; method?: 'r' | 's'; confidence?: number }): CapabilityResult
    /** Tolerance interval for a column. */
    toleranceInterval(column: string, options?: { coverage?: number; confidence?: number; method?: 'normal' | 'nonparametric' }): ToleranceIntervalResult
    /** Gage R&R (ANOVA) with part / operator / measurement columns. */
    gageRR(options: { part: string; operator?: string; measurement: string; design?: 'crossed' | 'nested' }): GageRRResult
    /** Pareto of a categorical column. */
    pareto(column: string, options?: { weights?: string }): ParetoResult
    /** Run chart tests on a column. */
    runChart(column: string, options?: { alpha?: number }): RunChartResult
    /** Individual Distribution Identification on a column. */
    individualDistributionID(column: string): IdResult
    /** Trend Analysis on a time-series column. */
    trendAnalysis(column: string, options?: { model?: 'linear' | 'quadratic' | 'exponential' | 's-curve'; horizon?: number }): TrendResult
    /** Classical seasonal decomposition. */
    decompose(column: string, options: { seasonLength: number; method?: 'additive' | 'multiplicative' }): DecompositionResult
    /** STL (LOESS) seasonal-trend decomposition. */
    stl(column: string, options: { seasonLength: number; seasonalSpan?: number; trendSpan?: number; robust?: boolean }): StlResult
    /** Exponential smoothing (SES / DES / Winters). */
    ets(column: string, options?: { method?: 'ses' | 'des' | 'winters-add' | 'winters-mul'; seasonLength?: number; horizon?: number }): EtsResult
    /** Sample ACF. */
    acf(column: string, options?: { maxLag?: number; confidence?: number }): AcfResult
    /** ARIMA / SARIMA / ARIMAX via CSS-ML. */
    arima(column: string, options?: {
      p?: number; d?: number; q?: number; includeMean?: boolean; horizon?: number; method?: 'CSS' | 'CSS-ML' | 'ML'
      seasonal?: { P?: number; D?: number; Q?: number; period: number }
      xreg?: string[]
    }): ArimaResult
    /** Parametric reliability fit (Weibull / lognormal / exponential). */
    reliabilityFit(time: string, options?: { distribution?: 'weibull' | 'lognormal' | 'exponential'; method?: 'MLE' | 'LSXY'; censor?: string }): ParametricSurvival
    /** Kaplan–Meier estimate. */
    kaplanMeier(time: string, options?: { censor?: string; confidence?: number }): KaplanMeierResult
    /** Mantel–Haenszel / Wilcoxon / Tarone–Ware log-rank across levels of `group`. */
    logRank(time: string, group: string, options?: { censor?: string; weight?: 'logrank' | 'wilcoxon' | 'tarone-ware'; strata?: string }): LogRankResult
    /** Cox PH regression; predictors are numeric columns. */
    coxPH(time: string, predictors: string[], options?: {
      censor?: string; confidence?: number; start?: string; strata?: string; frailtyGroup?: string
      frailtyBands?: string; frailtyProcess?: 'piecewise' | 'ar1'; cluster?: string; eventType?: string; cause?: string | number
    }): import('./reliability.js').CoxPHResult
    /** Auto-ARIMA (stepwise AIC/BIC). */
    autoArima(column: string, options?: { seasonalPeriod?: number; maxP?: number; maxQ?: number; maxD?: number; information?: 'aic' | 'bic'; horizon?: number }): ArimaResult
    /** Random-intercept (+ optional slope / nested) linear mixed model. */
    mixedModel(y: string, options: { fixed: string[]; group: string; groupNested?: string; group2?: string; slope?: string; reml?: boolean; confidence?: number }): import('./mixed.js').MixedModelResult
    /** GLMM (binomial/Poisson/negbin) with random intercept (+ optional slope). */
    glmm(y: string, options: { family: 'binomial' | 'poisson' | 'negbin'; fixed: string[]; group: string; slope?: string; trials?: string; confidence?: number; method?: 'pql' | 'laplace' | 'agq'; nAGQ?: number }): import('./mixed.js').GlmmResult
    /** Fine–Gray competing risks. */
    fineGray(time: string, predictors: string[], options: { eventType: string; cause: string | number; censor?: string; confidence?: number }): import('./reliability.js').FineGrayResult
    /** Fully nested ANOVA. */
    nestedAnova(y: string, factors: string[]): import('./doe.js').NestedAnovaResult
    /** Capability Sixpack plot-ready panels. */
    capabilitySixpack(column: string, options: { lsl?: number; usl?: number; target?: number; subgroup?: number; method?: 'r' | 's'; confidence?: number }): import('./capability.js').CapabilitySixpackResult
    /** Games–Howell pairwise comparisons. */
    gamesHowell(column: string, by: string, options?: { alpha?: number }): import('./multcomp.js').GamesHowellResult
    /** One-way MANOVA on response columns by group. */
    manova(responses: string[], by: string): import('./manova.js').ManovaResult
    /** PCA on the named numeric columns. */
    pca(columns: string[], options?: { nComponents?: number; scale?: boolean }): PcaResult
    /** k-means on the named numeric columns. */
    kmeans(columns: string[], options: { k: number; maxIter?: number; seed?: number }): KMeansResult
    /** CART on predictors → response. */
    cart(y: string, predictors: string[], options?: CartOptions): CartResult
    /** Random Forest on predictors → response. */
    randomForest(y: string, predictors: string[], options?: CartOptions & { nTrees?: number }): RandomForestResult
    /** Display Descriptive Statistics for one or more columns, optionally by a grouping column. */
    descriptiveStats(columns: string | string[], options?: { by?: string }): import('./descriptive.js').DescriptiveStats[]
    /** Graphical Summary: descriptives, Anderson–Darling, CIs, histogram and boxplot data. */
    graphicalSummary(column: string, options?: { confidence?: number; bins?: number }): import('./descriptive.js').GraphicalSummary
    /** Goodness-of-fit test for Poisson on a column of counts. */
    poissonGof(column: string, options?: { minExpected?: number }): import('./descriptive.js').PoissonGofResult
    /** Boxplot statistics, optionally per level of `by`. */
    boxplotStats(column: string, by?: string): import('./descriptive.js').BoxplotStats | import('./descriptive.js').BoxplotStats[]
    /** Interval plot data: group means with confidence intervals. */
    intervalPlot(column: string, by: string, options?: { confidence?: number; pooled?: boolean }): import('./descriptive.js').IntervalPlotRow[]
    /** Main effects plot data for several factor columns. */
    mainEffectsPlot(y: string, factors: string[]): import('./descriptive.js').MainEffectsPlot
    /** Interaction plot data (cell means) for two factor columns. */
    interactionPlot(y: string, a: string, b: string): import('./descriptive.js').InteractionPlot
    /** Stability Study: response vs time with a batch factor; shelf life against spec limits. */
    stabilityStudy(response: string, time: string, batch: string, options: StabilityOptions): StabilityResult
    /** G chart (opportunities between events) on a count column. */
    gChart(column: string, options?: { estimator?: 'mle' | 'mvue'; limits?: 'probability' | 'sigma'; alpha?: number }): import('./spc2.js').RareEventChartResult
    /** T chart (time between events). */
    tChart(column: string, options?: { distribution?: 'weibull' | 'exponential'; alpha?: number }): import('./spc2.js').RareEventChartResult
    /** Hotelling T² chart on several columns (optionally subgrouped). */
    t2Chart(columns: string[], options?: { subgroup?: string; phase?: 1 | 2; alpha?: number }): import('./spc2.js').T2ChartResult
    /** MEWMA chart on several columns. */
    mewma(columns: string[], options?: { lambda?: number; h?: number; arl0?: number }): import('./spc2.js').MewmaResult
    /** Periodogram / spectral analysis of a series column. */
    periodogram(column: string, options?: { detrend?: 'none' | 'mean' | 'linear'; taper?: number; spans?: number[] }): import('./spectral.js').PeriodogramResult
    /** Regression with life data (AFT) with optional censor column (0 exact, 1 right, 2 left, 3 interval + time2). */
    lifeRegression(time: string, predictors: string[], options?: { distribution?: import('./reliability2.js').LifeDistribution; censor?: string; time2?: string; confidence?: number }): import('./reliability2.js').LifeRegressionResult
    /** Accelerated life testing on a stress column. */
    altRegression(time: string, stress: string, options?: { relation?: import('./reliability2.js').AccelerationRelation; distribution?: import('./reliability2.js').LifeDistribution; censor?: string; useStress?: number; percentiles?: number[]; confidence?: number }): import('./reliability2.js').AltResult
    /** Power-law NHPP (Crow–AMSAA) on a column of failure times, optionally per system. */
    powerLawNHPP(time: string, options: { endTime: number | number[]; system?: string; confidence?: number }): import('./reliability2.js').NhppResult
    /** Probit analysis from events / trials / stress columns. */
    probitAnalysis(events: string, trials: string, stress: string, options?: { distribution?: 'normal' | 'logistic'; naturalResponse?: number; percentiles?: number[]; confidence?: number; logStress?: boolean }): import('./reliability2.js').ProbitResult
    /** Cluster Variables (correlation-distance hierarchical clustering). */
    clusterVariables(columns: string[], options?: { method?: 'single' | 'complete' | 'average' | 'ward'; distance?: 'correlation' | 'absolute correlation'; nClusters?: number }): import('./multivariate2.js').ClusterVariablesResult
    /** Multiple Correspondence Analysis of categorical columns. */
    multipleCorrespondence(columns: string[], options?: { method?: 'indicator' | 'burt'; nComponents?: number }): import('./multivariate2.js').McaResult
    /** Item Analysis (Cronbach's α) over item columns. */
    itemAnalysis(columns: string[]): import('./multivariate2.js').ItemAnalysisResult
    /** General MANOVA: responses on a model formula right-hand side, e.g. 'a*b + x'. */
    manovaModel(responses: string[], rhs: string, options?: { factors?: string[] }): import('./manova.js').ManovaModelResult
    /** k-fold cross-validation of a learner on predictor columns → response. */
    crossValidate(y: string, predictors: string[], options: { model: ModelKind; task?: 'regression' | 'classification'; folds?: number; seed?: number; nTrees?: number }): import('./automl.js').CvResult
    /** Automated model selection over the tree learners (+ OLS / logistic). */
    autoModel(y: string, predictors: string[], options?: { task?: 'regression' | 'classification'; folds?: number; seed?: number; models?: ModelKind[]; nTrees?: number }): import('./automl.js').AutoModelResult
    /**
     * 1 or 2 Proportions: without `by`, sums `eventsCol` and `trials` for propTest1; with `by` (2 levels),
     * aggregates per group (binary column or events + trials columns) for propTest2.
     */
    propTest(eventsCol: string, options?: DataFramePropTestOptions): PropTestResult
    /** Augmented Dickey–Fuller unit-root test on a series column. */
    adfTest(column: string, options?: { lags?: number; regression?: 'c' | 'ct' | 'n' }): UnitRootResult
    /** KPSS stationarity test on a series column. */
    kpssTest(column: string, options?: { lags?: number; regression?: 'c' | 'ct' }): UnitRootResult
    /** Ridge regression of `y` on predictor columns. */
    ridge(y: string, predictors: string[], options?: { alpha?: number }): PenalizedResult
    /** Lasso regression of `y` on predictor columns. */
    lasso(y: string, predictors: string[], options?: { alpha?: number; maxIter?: number; tol?: number }): PenalizedResult
    /** One-way ANCOVA: `y` by `group` adjusting for `covariate`. */
    ancova(y: string, group: string, covariate: string): AncovaResult
    /** Two-sample Kolmogorov–Smirnov between the two levels of `by`. */
    ksTwoSample(column: string, by: string): KsTwoSampleResult
  }
  interface LazyFrame {
    ttest(column: string, options?: DataFrameTTestOptions): Promise<TTestResult>
    anova(column: string, by: string): Promise<AnovaResult>
    anovaTwoWay(y: string, row: string, col: string, options?: { interaction?: boolean }): Promise<import('./anova2.js').AnovaTwoWayResult>
    tukey(column: string, by: string, options?: { alpha?: number }): Promise<TukeyResult>
    fisher(column: string, by: string, options?: { alpha?: number }): Promise<FisherResult & { familyAlpha: number }>
    dunnett(column: string, by: string, options: { control: string; alpha?: number; alternative?: Alternative }): Promise<DunnettResult>
    hsu(column: string, by: string, options?: { best?: 'largest' | 'smallest'; alpha?: number }): Promise<HsuResult>
    mannWhitney(column: string, by: string, options?: MannWhitneyOptions): Promise<MannWhitneyResult>
    kruskal(column: string, by: string): Promise<KruskalResult>
    equalVariances(column: string, by: string, method?: 'levene' | 'bartlett' | 'bonett'): Promise<VarianceTestResult>
    normalityTest(column: string, method?: NormalityMethod): Promise<NormalityResult>
    chi2test(a: string, b: string, options?: { correction?: boolean }): Promise<Chi2Result>
    ztest(column: string, options: { sigma: number; mu?: number; alternative?: Alternative; confidence?: number }): Promise<ZTestResult>
    varTest(column: string, options: { sigma0: number; alternative?: Alternative; confidence?: number; method?: 'chi-square' | 'bonett' }): Promise<VarTest1Result>
    corrTest(a: string, b: string, options?: { method?: 'pearson' | 'spearman' | 'kendall'; alternative?: Alternative; confidence?: number }): Promise<CorrTestResult>
    partialCorr(a: string, b: string, controls: string | string[], options?: { method?: 'pearson' | 'spearman'; alternative?: Alternative }): Promise<PartialCorrResult>
    outlierTest(column: string, options?: { method?: 'grubbs' | 'dixon'; alpha?: number; alternative?: 'two-sided' | 'min' | 'max' }): Promise<OutlierResult>
    signTest(column: string, options?: { median?: number; alternative?: Alternative; confidence?: number }): Promise<SignTestResult>
    wilcoxon(column: string, options?: { median?: number; alternative?: Alternative; confidence?: number; method?: 'auto' | 'exact' | 'asymptotic' }): Promise<WilcoxonResult>
    mood(column: string, by: string): Promise<MoodResult>
    friedman(column: string, treatment: string, block: string): Promise<FriedmanResult>
    runsTest(column: string, options?: { k?: number; correction?: boolean }): Promise<RunsTestResult>
    equivalence(column: string, options: { limits: [number, number]; alpha?: number; by?: string; reference?: string; paired?: string; equalVar?: boolean }): Promise<TostResult>
    regress(y: string, predictors: string[], options?: Omit<OlsOptions, 'names' | 'weights'> & { weights?: string }): Promise<OlsResult>
    fittedLine(x: string, y: string, options?: { degree?: 1 | 2 | 3; logX?: boolean; logY?: boolean; confidence?: number }): Promise<OlsResult & { degree: number; curve(x: number): number; equation: string }>
    stepwise(y: string, predictors: string[], options?: Omit<StepwiseOptions, 'names'>): Promise<StepwiseResult>
    bestSubsets(y: string, predictors: string[], options?: { maxK?: number; nBest?: number; include?: string[] }): Promise<BestSubsetsResult>
    glm(y: string, predictors: string[], options: Omit<GlmOptions, 'names' | 'trials' | 'offset' | 'weights'> & { trials?: string; offset?: string; weights?: string }): Promise<GlmResult>
    logistic(y: string, predictors: string[], options?: Omit<GlmOptions, 'family' | 'names' | 'trials' | 'offset' | 'weights'> & { trials?: string; offset?: string; weights?: string }): Promise<GlmResult>
    ologit(y: string, predictors: string[], options?: { confidence?: number; maxIter?: number }): Promise<OrdinalResult>
    mlogit(y: string, predictors: string[], options?: { reference?: string | number; confidence?: number; maxIter?: number }): Promise<NominalResult>
    linearModel(formula: string, options?: { factors?: string[]; confidence?: number }): Promise<LinearModelResult>
    nls(model: NlsModel, x: string | string[], y: string, options: NlsOptions): Promise<NlsResult>
    orthogonalRegression(x: string, y: string, options?: { errorVarianceRatio?: number; confidence?: number }): Promise<OrthogonalResult>
    pls(y: string | string[], predictors: string[], options?: Omit<PlsOptions, 'names' | 'responseNames'>): Promise<PlsResult>
    controlChart(column: string, options?: ControlChartOptions): Promise<ControlChartResult>
    ewma(column: string, options?: { lambda?: number; L?: number; mu?: number; sigma?: number }): Promise<EwmaResult>
    cusum(column: string, options?: { k?: number; h?: number; mu?: number; sigma?: number; vMask?: boolean }): Promise<CusumResult>
    capability(column: string, options: { lsl?: number; usl?: number; target?: number; subgroup?: number; method?: 'r' | 's'; confidence?: number }): Promise<CapabilityResult>
    toleranceInterval(column: string, options?: { coverage?: number; confidence?: number; method?: 'normal' | 'nonparametric' }): Promise<ToleranceIntervalResult>
    gageRR(options: { part: string; operator?: string; measurement: string; design?: 'crossed' | 'nested' }): Promise<GageRRResult>
    pareto(column: string, options?: { weights?: string }): Promise<ParetoResult>
    runChart(column: string, options?: { alpha?: number }): Promise<RunChartResult>
    individualDistributionID(column: string): Promise<IdResult>
    trendAnalysis(column: string, options?: { model?: 'linear' | 'quadratic' | 'exponential' | 's-curve'; horizon?: number }): Promise<TrendResult>
    decompose(column: string, options: { seasonLength: number; method?: 'additive' | 'multiplicative' }): Promise<DecompositionResult>
    stl(column: string, options: { seasonLength: number; seasonalSpan?: number; trendSpan?: number; robust?: boolean }): Promise<StlResult>
    ets(column: string, options?: { method?: 'ses' | 'des' | 'winters-add' | 'winters-mul'; seasonLength?: number; horizon?: number }): Promise<EtsResult>
    acf(column: string, options?: { maxLag?: number; confidence?: number }): Promise<AcfResult>
    arima(column: string, options?: {
      p?: number; d?: number; q?: number; includeMean?: boolean; horizon?: number; method?: 'CSS' | 'CSS-ML' | 'ML' | 'ML'
      seasonal?: { P?: number; D?: number; Q?: number; period: number }
      xreg?: string[]
    }): Promise<ArimaResult>
    reliabilityFit(time: string, options?: { distribution?: 'weibull' | 'lognormal' | 'exponential'; method?: 'MLE' | 'LSXY'; censor?: string }): Promise<ParametricSurvival>
    kaplanMeier(time: string, options?: { censor?: string; confidence?: number }): Promise<KaplanMeierResult>
    logRank(time: string, group: string, options?: { censor?: string; weight?: 'logrank' | 'wilcoxon' | 'tarone-ware'; strata?: string }): Promise<LogRankResult>
    coxPH(time: string, predictors: string[], options?: {
      censor?: string; confidence?: number; start?: string; strata?: string; frailtyGroup?: string
      frailtyBands?: string; frailtyProcess?: 'piecewise' | 'ar1'; cluster?: string; eventType?: string; cause?: string | number
    }): Promise<import('./reliability.js').CoxPHResult>
    autoArima(column: string, options?: { seasonalPeriod?: number; maxP?: number; maxQ?: number; maxD?: number; information?: 'aic' | 'bic'; horizon?: number }): Promise<ArimaResult>
    mixedModel(y: string, options: { fixed: string[]; group: string; groupNested?: string; slope?: string; reml?: boolean; confidence?: number }): Promise<import('./mixed.js').MixedModelResult>
    glmm(y: string, options: { family: 'binomial' | 'poisson' | 'negbin'; fixed: string[]; group: string; slope?: string; trials?: string; confidence?: number; method?: 'pql' | 'laplace' | 'agq'; nAGQ?: number }): Promise<import('./mixed.js').GlmmResult>
    gamesHowell(column: string, by: string, options?: { alpha?: number }): Promise<import('./multcomp.js').GamesHowellResult>
    manova(responses: string[], by: string): Promise<import('./manova.js').ManovaResult>
    pca(columns: string[], options?: { nComponents?: number; scale?: boolean }): Promise<PcaResult>
    kmeans(columns: string[], options: { k: number; maxIter?: number; seed?: number }): Promise<KMeansResult>
    cart(y: string, predictors: string[], options?: CartOptions): Promise<CartResult>
    randomForest(y: string, predictors: string[], options?: CartOptions & { nTrees?: number }): Promise<RandomForestResult>
    descriptiveStats(columns: string | string[], options?: { by?: string }): Promise<import('./descriptive.js').DescriptiveStats[]>
    graphicalSummary(column: string, options?: { confidence?: number; bins?: number }): Promise<import('./descriptive.js').GraphicalSummary>
    poissonGof(column: string, options?: { minExpected?: number }): Promise<import('./descriptive.js').PoissonGofResult>
    boxplotStats(column: string, by?: string): Promise<import('./descriptive.js').BoxplotStats | import('./descriptive.js').BoxplotStats[]>
    intervalPlot(column: string, by: string, options?: { confidence?: number; pooled?: boolean }): Promise<import('./descriptive.js').IntervalPlotRow[]>
    mainEffectsPlot(y: string, factors: string[]): Promise<import('./descriptive.js').MainEffectsPlot>
    interactionPlot(y: string, a: string, b: string): Promise<import('./descriptive.js').InteractionPlot>
    stabilityStudy(response: string, time: string, batch: string, options: StabilityOptions): Promise<StabilityResult>
    gChart(column: string, options?: { estimator?: 'mle' | 'mvue'; limits?: 'probability' | 'sigma'; alpha?: number }): Promise<import('./spc2.js').RareEventChartResult>
    tChart(column: string, options?: { distribution?: 'weibull' | 'exponential'; alpha?: number }): Promise<import('./spc2.js').RareEventChartResult>
    t2Chart(columns: string[], options?: { subgroup?: string; phase?: 1 | 2; alpha?: number }): Promise<import('./spc2.js').T2ChartResult>
    mewma(columns: string[], options?: { lambda?: number; h?: number; arl0?: number }): Promise<import('./spc2.js').MewmaResult>
    periodogram(column: string, options?: { detrend?: 'none' | 'mean' | 'linear'; taper?: number; spans?: number[] }): Promise<import('./spectral.js').PeriodogramResult>
    lifeRegression(time: string, predictors: string[], options?: { distribution?: import('./reliability2.js').LifeDistribution; censor?: string; time2?: string; confidence?: number }): Promise<import('./reliability2.js').LifeRegressionResult>
    altRegression(time: string, stress: string, options?: { relation?: import('./reliability2.js').AccelerationRelation; distribution?: import('./reliability2.js').LifeDistribution; censor?: string; useStress?: number; percentiles?: number[]; confidence?: number }): Promise<import('./reliability2.js').AltResult>
    powerLawNHPP(time: string, options: { endTime: number | number[]; system?: string; confidence?: number }): Promise<import('./reliability2.js').NhppResult>
    probitAnalysis(events: string, trials: string, stress: string, options?: { distribution?: 'normal' | 'logistic'; naturalResponse?: number; percentiles?: number[]; confidence?: number; logStress?: boolean }): Promise<import('./reliability2.js').ProbitResult>
    clusterVariables(columns: string[], options?: { method?: 'single' | 'complete' | 'average' | 'ward'; distance?: 'correlation' | 'absolute correlation'; nClusters?: number }): Promise<import('./multivariate2.js').ClusterVariablesResult>
    multipleCorrespondence(columns: string[], options?: { method?: 'indicator' | 'burt'; nComponents?: number }): Promise<import('./multivariate2.js').McaResult>
    itemAnalysis(columns: string[]): Promise<import('./multivariate2.js').ItemAnalysisResult>
    manovaModel(responses: string[], rhs: string, options?: { factors?: string[] }): Promise<import('./manova.js').ManovaModelResult>
    crossValidate(y: string, predictors: string[], options: { model: ModelKind; task?: 'regression' | 'classification'; folds?: number; seed?: number; nTrees?: number }): Promise<import('./automl.js').CvResult>
    autoModel(y: string, predictors: string[], options?: { task?: 'regression' | 'classification'; folds?: number; seed?: number; models?: ModelKind[]; nTrees?: number }): Promise<import('./automl.js').AutoModelResult>
    propTest(eventsCol: string, options?: DataFramePropTestOptions): Promise<PropTestResult>
    adfTest(column: string, options?: { lags?: number; regression?: 'c' | 'ct' | 'n' }): Promise<UnitRootResult>
    kpssTest(column: string, options?: { lags?: number; regression?: 'c' | 'ct' }): Promise<UnitRootResult>
    ridge(y: string, predictors: string[], options?: { alpha?: number }): Promise<PenalizedResult>
    lasso(y: string, predictors: string[], options?: { alpha?: number; maxIter?: number; tol?: number }): Promise<PenalizedResult>
    ancova(y: string, group: string, covariate: string): Promise<AncovaResult>
    ksTwoSample(column: string, by: string): Promise<KsTwoSampleResult>
  }
}

const D = DataFrame.prototype
D.ttest = function (this: DataFrame, column: string, options: DataFrameTTestOptions = {}): TTestResult {
  const { by, paired, equalVar, ...rest } = options
  if (by && paired) throw new RangeError('ttest: use either by (two-sample) or paired, not both')
  const x = this.getColumn(column).toArray() as Num
  if (paired) return ttestPaired(x, this.getColumn(paired).toArray() as Num, rest)
  if (!by) return ttest1(x, rest)
  const g = groupsOf(this, column, by)
  const levels = Object.keys(g)
  if (levels.length !== 2) throw new RangeError(`ttest: column "${by}" must have exactly 2 levels for a two-sample test, got ${levels.length}`)
  return ttest2(g[levels[0]!]!, g[levels[1]!]!, { ...rest, equalVar })
}
D.anova = function (this: DataFrame, column, by) {
  return anovaTest(groupsOf(this, column, by))
}
D.anovaTwoWay = function (this: DataFrame, y: string, row: string, col: string, options = {}) {
  return anovaTwoWayFn(
    this.getColumn(y).toArray() as Num,
    this.getColumn(row).toArray() as Array<string | number>,
    this.getColumn(col).toArray() as Array<string | number>,
    options,
  )
}
D.tukey = function (this: DataFrame, column, by, options = {}) {
  return tukeyHSD(groupsOf(this, column, by), options)
}
D.fisher = function (this: DataFrame, column, by, options = {}) {
  return fisherLSD(groupsOf(this, column, by), options)
}
D.dunnett = function (this: DataFrame, column, by, options) {
  return dunnettTest(groupsOf(this, column, by), options)
}
D.hsu = function (this: DataFrame, column, by, options = {}) {
  return hsuMCB(groupsOf(this, column, by), options)
}
D.mannWhitney = function (this: DataFrame, column, by, options = {}) {
  const g = groupsOf(this, column, by)
  const levels = Object.keys(g)
  if (levels.length !== 2) throw new RangeError(`mannWhitney: column "${by}" must have exactly 2 levels, got ${levels.length}`)
  return mannWhitneyTest(g[levels[0]!]!, g[levels[1]!]!, options)
}
D.kruskal = function (this: DataFrame, column, by) {
  return kruskalTest(groupsOf(this, column, by))
}
D.equalVariances = function (this: DataFrame, column, by, method = 'levene') {
  return equalVariancesTest(groupsOf(this, column, by), method)
}
D.normalityTest = function (this: DataFrame, column, method = 'anderson-darling') {
  return normalityTestFn(this.getColumn(column).toArray() as Num, method)
}
D.chi2test = function (this: DataFrame, a, b, options = {}) {
  const ct = crosstab(this.getColumn(a).toArray(), this.getColumn(b).toArray())
  return { ...chi2Independence(ct.table, options), rows: ct.rows, cols: ct.cols }
}

D.ztest = function (this: DataFrame, column, options) {
  return ztest1Fn(this.getColumn(column).toArray() as Num, options)
}
D.varTest = function (this: DataFrame, column, options) {
  return varTest1Fn(this.getColumn(column).toArray() as Num, options)
}
D.corrTest = function (this: DataFrame, a, b, options = {}) {
  return corrTestFn(this.getColumn(a).toArray() as Num, this.getColumn(b).toArray() as Num, options)
}
D.partialCorr = function (this: DataFrame, a: string, b: string, controls: string | string[], options = {}) {
  const cols = (Array.isArray(controls) ? controls : [controls]).map((c) => this.getColumn(c).toArray() as Num)
  return partialCorrFn(this.getColumn(a).toArray() as Num, this.getColumn(b).toArray() as Num, cols, options)
}
D.outlierTest = function (this: DataFrame, column, options = {}) {
  const { method, ...rest } = options
  const x = this.getColumn(column).toArray() as Num
  return method === 'dixon' ? dixonFn(x, rest) : grubbsFn(x, rest)
}
D.signTest = function (this: DataFrame, column, options = {}) {
  return signFn(this.getColumn(column).toArray() as Num, options)
}
D.wilcoxon = function (this: DataFrame, column, options = {}) {
  return wilcoxonFn(this.getColumn(column).toArray() as Num, options)
}
D.mood = function (this: DataFrame, column, by) {
  return moodFn(groupsOf(this, column, by))
}
D.friedman = function (this: DataFrame, column, treatment, block) {
  const y = this.getColumn(column).toArray() as Num
  const tr = this.getColumn(treatment).toArray()
  const bl = this.getColumn(block).toArray()
  const treatments = [...new Set(tr.filter((v) => v !== null).map(String))].sort()
  const blocks = [...new Set(bl.filter((v) => v !== null).map(String))].sort()
  const tIdx = new Map(treatments.map((t, i) => [t, i]))
  const bIdx = new Map(blocks.map((b, i) => [b, i]))
  const table = blocks.map(() => new Array<number>(treatments.length).fill(NaN))
  for (let i = 0; i < y.length; i++) {
    if (tr[i] === null || bl[i] === null || y[i] === null) continue
    const r = bIdx.get(String(bl[i]))!
    const c = tIdx.get(String(tr[i]))!
    if (!Number.isNaN(table[r]![c]!)) throw new RangeError(`friedman: more than one observation for block "${bl[i]}" × treatment "${tr[i]}"`)
    table[r]![c] = y[i]!
  }
  for (let r = 0; r < blocks.length; r++) for (let c = 0; c < treatments.length; c++) if (Number.isNaN(table[r]![c]!)) throw new RangeError(`friedman: missing observation for block "${blocks[r]}" × treatment "${treatments[c]}"`)
  return friedmanFn(table, treatments)
}
D.runsTest = function (this: DataFrame, column, options = {}) {
  return runsFn(this.getColumn(column).toArray() as Num, options)
}
D.equivalence = function (this: DataFrame, column, options) {
  const { by, reference, paired, equalVar, ...rest } = options
  if (by && paired) throw new RangeError('equivalence: use either by (two-sample) or paired, not both')
  const x = this.getColumn(column).toArray() as Num
  if (paired) return tostPaired(x, this.getColumn(paired).toArray() as Num, rest)
  if (!by) return tost1(x, rest)
  const g = groupsOf(this, column, by)
  const levels = Object.keys(g)
  if (levels.length !== 2) throw new RangeError(`equivalence: column "${by}" must have exactly 2 levels, got ${levels.length}`)
  const ref = reference ?? levels[1]!
  if (!(ref in g)) throw new RangeError(`equivalence: reference level "${ref}" not found in "${by}"`)
  const testLevel = levels.find((l) => l !== ref)!
  return tost2(g[testLevel]!, g[ref]!, { ...rest, equalVar })
}

const numCols = (df: DataFrame, names: string[]) => names.map((c) => df.getColumn(c).toArray() as Num)
D.regress = function (this: DataFrame, y, predictors, options = {}) {
  const { weights, ...rest } = options
  return ols(this.getColumn(y).toArray() as Num, numCols(this, predictors), { ...rest, names: predictors, weights: weights ? (this.getColumn(weights).toArray() as Num) : undefined })
}
D.fittedLine = function (this: DataFrame, x, y, options = {}) {
  return fittedLineFn(this.getColumn(x).toArray() as Num, this.getColumn(y).toArray() as Num, options)
}
D.stepwise = function (this: DataFrame, y, predictors, options = {}) {
  return stepwiseFn(this.getColumn(y).toArray() as Num, numCols(this, predictors), { ...options, names: predictors })
}
D.bestSubsets = function (this: DataFrame, y, predictors, options = {}) {
  return bestSubsetsFn(this.getColumn(y).toArray() as Num, numCols(this, predictors), { ...options, names: predictors })
}
D.glm = function (this: DataFrame, y, predictors, options) {
  const { trials, offset, weights, ...rest } = options
  return glmFn(this.getColumn(y).toArray() as Num, numCols(this, predictors), {
    ...rest,
    names: predictors,
    trials: trials ? (this.getColumn(trials).toArray() as Num) : undefined,
    offset: offset ? (this.getColumn(offset).toArray() as Num) : undefined,
    weights: weights ? (this.getColumn(weights).toArray() as Num) : undefined,
  })
}
D.logistic = function (this: DataFrame, y, predictors, options = {}) {
  return this.glm(y, predictors, { ...options, family: 'binomial' })
}
D.ologit = function (this: DataFrame, y, predictors, options = {}) {
  return ologitFn(this.getColumn(y).toArray() as Array<number | string | null>, numCols(this, predictors), { ...options, names: predictors })
}
D.mlogit = function (this: DataFrame, y, predictors, options = {}) {
  return mlogitFn(this.getColumn(y).toArray() as Array<number | string | null>, numCols(this, predictors), { ...options, names: predictors })
}
D.linearModel = function (this: DataFrame, formula, options = {}) {
  const data: Record<string, Array<number | string | boolean | null>> = {}
  for (const c of this.columns) data[c] = this.getColumn(c).toArray()
  return linearModelFn(data, formula, options)
}
D.nls = function (this: DataFrame, model, x, y, options) {
  const xs = Array.isArray(x) ? numCols(this, x) : null
  const xv: Array<number | number[]> = xs ? xs[0]!.map((_, i) => xs.map((c) => c[i] as number)) : (this.getColumn(x as string).toArray() as number[])
  return nlsFn(model, xv, this.getColumn(y).toArray() as Num, options)
}
D.orthogonalRegression = function (this: DataFrame, x, y, options = {}) {
  return orthogonalFn(this.getColumn(x).toArray() as Num, this.getColumn(y).toArray() as Num, options)
}
D.pls = function (this: DataFrame, y, predictors, options = {}) {
  const ys = Array.isArray(y) ? y : [y]
  return plsFn(numCols(this, ys), numCols(this, predictors), { ...options, names: predictors, responseNames: ys })
}
D.controlChart = function (this: DataFrame, column, options = {}) {
  return controlChartFn(this.getColumn(column).toArray() as Num, options)
}
D.ewma = function (this: DataFrame, column, options = {}) {
  return ewmaFn(this.getColumn(column).toArray() as Num, options)
}
D.cusum = function (this: DataFrame, column, options = {}) {
  return cusumFn(this.getColumn(column).toArray() as Num, options)
}
D.capability = function (this: DataFrame, column, options) {
  return capabilityFn(this.getColumn(column).toArray() as Num, options)
}
D.toleranceInterval = function (this: DataFrame, column, options = {}) {
  return toleranceFn(this.getColumn(column).toArray() as Num, options)
}
D.gageRR = function (this: DataFrame, options) {
  return gageRRFn(
    {
      part: this.getColumn(options.part).toArray() as Array<string | number | null>,
      operator: options.operator ? (this.getColumn(options.operator).toArray() as Array<string | number | null>) : undefined,
      measurement: this.getColumn(options.measurement).toArray() as Num,
    },
    { design: options.design },
  )
}
D.pareto = function (this: DataFrame, column, options = {}) {
  return paretoFn(this.getColumn(column).toArray() as Array<string | number | null>, {
    weights: options.weights ? (this.getColumn(options.weights).toArray() as Num) : undefined,
  })
}
D.runChart = function (this: DataFrame, column, options = {}) {
  return runChartFn(this.getColumn(column).toArray() as Num, options)
}
D.individualDistributionID = function (this: DataFrame, column) {
  return idiFn(this.getColumn(column).toArray() as Num)
}
D.trendAnalysis = function (this: DataFrame, column, options = {}) {
  return trendFn(this.getColumn(column).toArray() as Num, options)
}
D.decompose = function (this: DataFrame, column, options) {
  return decompFn(this.getColumn(column).toArray() as Num, options)
}
D.stl = function (this: DataFrame, column, options) {
  return stlFn(this.getColumn(column).toArray() as Num, options)
}
D.ets = function (this: DataFrame, column, options = {}) {
  return etsFn(this.getColumn(column).toArray() as Num, options)
}
D.acf = function (this: DataFrame, column, options = {}) {
  return acfFn(this.getColumn(column).toArray() as Num, options)
}
D.arima = function (this: DataFrame, column, options = {}) {
  const { xreg, ...rest } = options
  const x =
    xreg && xreg.length
      ? this.getColumn(column)
          .toArray()
          .map((_, i) => xreg.map((c) => Number(this.getColumn(c).toArray()[i])))
      : undefined
  return arimaFn(this.getColumn(column).toArray() as Num, { ...rest, xreg: x })
}
D.reliabilityFit = function (this: DataFrame, time, options = {}) {
  return relFitFn(this.getColumn(time).toArray() as Num, {
    ...options,
    censor: options.censor ? (this.getColumn(options.censor).toArray() as Num) : undefined,
  })
}
D.kaplanMeier = function (this: DataFrame, time, options = {}) {
  return kmFn(this.getColumn(time).toArray() as Num, {
    ...options,
    censor: options.censor ? (this.getColumn(options.censor).toArray() as Num) : undefined,
  })
}
D.logRank = function (this: DataFrame, time, group, options = {}) {
  return logRankFn(this.getColumn(time).toArray() as Num, this.getColumn(group).toArray() as Array<string | number | null>, {
    censor: options.censor ? (this.getColumn(options.censor).toArray() as Num) : undefined,
    weight: options.weight,
    strata: options.strata ? (this.getColumn(options.strata).toArray() as Array<string | number | null>) : undefined,
  })
}
D.coxPH = function (this: DataFrame, time, predictors, options = {}) {
  const n = this.getColumn(time).toArray().length
  const X = Array.from({ length: n }, (_, i) => predictors.map((c) => Number(this.getColumn(c).toArray()[i])))
  return coxPHFn(this.getColumn(time).toArray() as Num, X, {
    names: predictors,
    confidence: options.confidence,
    censor: options.censor ? (this.getColumn(options.censor).toArray() as Num) : undefined,
    start: options.start ? (this.getColumn(options.start).toArray() as Num) : undefined,
    strata: options.strata ? (this.getColumn(options.strata).toArray() as Array<string | number | null>) : undefined,
    frailty: options.frailtyGroup
      ? {
          group: this.getColumn(options.frailtyGroup).toArray() as Array<string | number>,
          bands: options.frailtyBands
            ? (this.getColumn(options.frailtyBands).toArray() as Array<string | number>)
            : undefined,
          process: options.frailtyProcess,
        }
      : undefined,
    cluster: options.cluster ? (this.getColumn(options.cluster).toArray() as Array<string | number | null>) : undefined,
    eventType: options.eventType ? (this.getColumn(options.eventType).toArray() as Array<string | number | null>) : undefined,
    cause: options.cause,
  })
}
D.autoArima = function (this: DataFrame, column, options = {}) {
  return autoArimaFn(this.getColumn(column).toArray() as Num, options)
}
D.mixedModel = function (this: DataFrame, y, options) {
  const n = this.getColumn(y).toArray().length
  const fixed = Array.from({ length: n }, (_, i) => options.fixed.map((c) => Number(this.getColumn(c).toArray()[i])))
  return mixedFn(this.getColumn(y).toArray() as number[], {
    fixed,
    group: this.getColumn(options.group).toArray() as Array<string | number>,
    groupNested: options.groupNested
      ? (this.getColumn(options.groupNested).toArray() as Array<string | number>)
      : undefined,
    group2: options.group2 ? (this.getColumn(options.group2).toArray() as Array<string | number>) : undefined,
    slope: options.slope ? (this.getColumn(options.slope).toArray() as number[]) : undefined,
    names: ['(Intercept)', ...options.fixed],
    reml: options.reml,
    confidence: options.confidence,
  })
}
D.glmm = function (this: DataFrame, y, options) {
  const n = this.getColumn(y).toArray().length
  const fixed = Array.from({ length: n }, (_, i) => options.fixed.map((c) => Number(this.getColumn(c).toArray()[i])))
  return glmmFn(this.getColumn(y).toArray() as number[], {
    family: options.family,
    fixed,
    group: this.getColumn(options.group).toArray() as Array<string | number>,
    slope: options.slope ? (this.getColumn(options.slope).toArray() as number[]) : undefined,
    trials: options.trials ? (this.getColumn(options.trials).toArray() as number[]) : undefined,
    names: ['(Intercept)', ...options.fixed],
    confidence: options.confidence,
    method: options.method,
    nAGQ: options.nAGQ,
  })
}
D.fineGray = function (this: DataFrame, time, predictors, options) {
  const n = this.getColumn(time).toArray().length
  const X = Array.from({ length: n }, (_, i) => predictors.map((c) => Number(this.getColumn(c).toArray()[i])))
  return fineGrayFn(this.getColumn(time).toArray() as Num, X, {
    names: predictors,
    confidence: options.confidence,
    censor: options.censor ? (this.getColumn(options.censor).toArray() as Num) : undefined,
    eventType: this.getColumn(options.eventType).toArray() as Array<string | number | null>,
    cause: options.cause,
  })
}
D.nestedAnova = function (this: DataFrame, y, factors) {
  return nestedAnovaFn(
    this.getColumn(y).toArray() as number[],
    factors.map((f) => this.getColumn(f).toArray() as Array<string | number>),
  )
}
D.capabilitySixpack = function (this: DataFrame, column, options) {
  return sixpackFn(this.getColumn(column).toArray() as Num, options)
}
D.gamesHowell = function (this: DataFrame, column: string, by: string, options: { alpha?: number } = {}) {
  return gamesHowellFn(groupsOf(this, column, by), options)
}
D.manova = function (this: DataFrame, responses: string[], by: string) {
  const n = this.getColumn(by).toArray().length
  const Y = Array.from({ length: n }, (_, i) => responses.map((c) => Number(this.getColumn(c).toArray()[i])))
  return manovaFn(Y, this.getColumn(by).toArray() as Array<string | number>)
}
D.pca = function (this: DataFrame, columns, options = {}) {
  const data: Record<string, Num> = {}
  for (const c of columns) data[c] = this.getColumn(c).toArray() as Num
  return pcaFn(data, { ...options, names: columns })
}
D.kmeans = function (this: DataFrame, columns, options) {
  const rows = this.getColumn(columns[0]!).toArray().map((_, i) => columns.map((c) => Number(this.getColumn(c).toArray()[i])))
  return kmeansFn(rows as number[][], options)
}
D.cart = function (this: DataFrame, y, predictors, options = {}) {
  const X = numCols(this, predictors)
  const rows = X[0]!.map((_, i) => X.map((c) => c[i] as number))
  return cartFn(rows, this.getColumn(y).toArray() as Array<number | string | null> as Array<number | string>, options)
}
D.randomForest = function (this: DataFrame, y, predictors, options = {}) {
  const X = numCols(this, predictors)
  const rows = X[0]!.map((_, i) => X.map((c) => c[i] as number))
  return rfFn(rows, this.getColumn(y).toArray() as Array<number | string | null> as Array<number | string>, options)
}

const rowsOf = (df: DataFrame, names: string[]): number[][] => {
  const cols = numCols(df, names)
  return cols[0]!.map((_, i) => cols.map((c) => Number(c[i])))
}
const allData = (df: DataFrame): Record<string, Array<number | string | boolean | null>> => {
  const data: Record<string, Array<number | string | boolean | null>> = {}
  for (const c of df.columns) data[c] = df.getColumn(c).toArray()
  return data
}
D.descriptiveStats = function (this: DataFrame, columns, options = {}) {
  const names = Array.isArray(columns) ? columns : [columns]
  const data: Record<string, Num> = {}
  for (const c of names) data[c] = this.getColumn(c).toArray() as Num
  return descFn(data, { by: options.by ? this.getColumn(options.by).toArray() : undefined })
}
D.graphicalSummary = function (this: DataFrame, column, options = {}) {
  return gsFn(this.getColumn(column).toArray() as Num, options)
}
D.poissonGof = function (this: DataFrame, column, options = {}) {
  return poissonGofFn(this.getColumn(column).toArray() as Num, options)
}
D.boxplotStats = function (this: DataFrame, column, by) {
  return boxplotFn(this.getColumn(column).toArray() as Num, { by: by ? this.getColumn(by).toArray() : undefined })
}
D.intervalPlot = function (this: DataFrame, column, by, options = {}) {
  return intervalFn(this.getColumn(column).toArray() as Num, this.getColumn(by).toArray(), options)
}
D.mainEffectsPlot = function (this: DataFrame, y, factors) {
  const f: Record<string, Array<number | string | boolean | null>> = {}
  for (const c of factors) f[c] = this.getColumn(c).toArray()
  return mainFn(this.getColumn(y).toArray() as Num, f)
}
D.interactionPlot = function (this: DataFrame, y, a, b) {
  return interFn(this.getColumn(y).toArray() as Num, this.getColumn(a).toArray(), this.getColumn(b).toArray(), [a, b])
}
D.stabilityStudy = function (this: DataFrame, response, time, batch, options) {
  return stabilityFn(this.getColumn(response).toArray() as Num, this.getColumn(time).toArray() as Num, this.getColumn(batch).toArray(), options)
}
D.gChart = function (this: DataFrame, column, options = {}) {
  return gChartFn(this.getColumn(column).toArray() as Num, options)
}
D.tChart = function (this: DataFrame, column, options = {}) {
  return tChartFn(this.getColumn(column).toArray() as Num, options)
}
D.t2Chart = function (this: DataFrame, columns, options = {}) {
  const { subgroup, ...rest } = options
  return t2Fn(rowsOf(this, columns), { ...rest, subgroup: subgroup ? this.getColumn(subgroup).toArray() : undefined })
}
D.mewma = function (this: DataFrame, columns, options = {}) {
  return mewmaFn(rowsOf(this, columns), options)
}
D.periodogram = function (this: DataFrame, column, options = {}) {
  return periodogramFn(this.getColumn(column).toArray() as Num, options)
}
D.lifeRegression = function (this: DataFrame, time, predictors, options = {}) {
  const { censor, time2, ...rest } = options
  return lifeRegFn(this.getColumn(time).toArray() as Num, numCols(this, predictors), {
    ...rest,
    names: predictors,
    censor: censor ? (this.getColumn(censor).toArray() as Num) : undefined,
    time2: time2 ? (this.getColumn(time2).toArray() as Num) : undefined,
  })
}
D.altRegression = function (this: DataFrame, time, stress, options = {}) {
  const { censor, ...rest } = options
  return altFn(this.getColumn(time).toArray() as Num, this.getColumn(stress).toArray() as Num, { ...rest, censor: censor ? (this.getColumn(censor).toArray() as Num) : undefined })
}
D.powerLawNHPP = function (this: DataFrame, time, options) {
  const t = this.getColumn(time).toArray() as Num
  if (!options.system) return nhppFn(t.filter((v): v is number => typeof v === 'number'), { endTime: options.endTime as number, confidence: options.confidence })
  const sys = this.getColumn(options.system).toArray()
  const groups = new Map<string, number[]>()
  for (let i = 0; i < t.length; i++) {
    if (typeof t[i] !== 'number' || sys[i] == null) continue
    const k = String(sys[i])
    let g = groups.get(k)
    if (!g) groups.set(k, (g = []))
    g.push(t[i] as number)
  }
  return nhppFn([...groups.values()], { endTime: options.endTime, confidence: options.confidence })
}
D.probitAnalysis = function (this: DataFrame, events, trials, stress, options = {}) {
  return probitFn(this.getColumn(events).toArray() as Num, this.getColumn(trials).toArray() as Num, this.getColumn(stress).toArray() as Num, options)
}
D.clusterVariables = function (this: DataFrame, columns, options = {}) {
  const data: Record<string, number[]> = {}
  for (const c of columns) data[c] = (this.getColumn(c).toArray() as Num).map((v) => Number(v))
  return clusterVarsFn(data, options)
}
D.multipleCorrespondence = function (this: DataFrame, columns, options = {}) {
  const data: Record<string, Array<number | string | boolean | null>> = {}
  for (const c of columns) data[c] = this.getColumn(c).toArray()
  return mcaFn(data, options)
}
D.itemAnalysis = function (this: DataFrame, columns) {
  const data: Record<string, number[]> = {}
  for (const c of columns) data[c] = (this.getColumn(c).toArray() as Num).map((v) => Number(v))
  return itemFn(data)
}
D.manovaModel = function (this: DataFrame, responses, rhs, options = {}) {
  return manovaModelFn(allData(this), responses, rhs, options)
}
D.crossValidate = function (this: DataFrame, y, predictors, options) {
  return cvFn(rowsOf(this, predictors), this.getColumn(y).toArray() as Array<number | string>, options)
}
D.autoModel = function (this: DataFrame, y, predictors, options = {}) {
  return autoModelFn(rowsOf(this, predictors), this.getColumn(y).toArray() as Array<number | string>, options)
}
D.propTest = function (this: DataFrame, eventsCol, options = {}) {
  const { trials: trialsCol, by, ...rest } = options
  if (by) {
    const agg = propAggByGroup(this, eventsCol, by, trialsCol)
    const levels = Object.keys(agg)
    if (levels.length !== 2) throw new RangeError(`propTest: column "${by}" must have exactly 2 levels, got ${levels.length}`)
    const a = agg[levels[0]!]!
    const b = agg[levels[1]!]!
    return propTest2(a.events, a.trials, b.events, b.trials, rest)
  }
  if (!trialsCol) throw new RangeError('propTest: trials column is required for a one-sample test (omit by)')
  const events = sumNumericCol(this, eventsCol)
  const trials = sumNumericCol(this, trialsCol)
  return propTest1(events, trials, rest)
}
D.adfTest = function (this: DataFrame, column, options = {}) {
  return adfTestFn(this.getColumn(column).toArray() as Num, options)
}
D.kpssTest = function (this: DataFrame, column, options = {}) {
  return kpssTestFn(this.getColumn(column).toArray() as Num, options)
}
D.ridge = function (this: DataFrame, y, predictors, options = {}) {
  return ridgeFn(this.getColumn(y).toArray() as Num, rowsOf(this, predictors), options)
}
D.lasso = function (this: DataFrame, y, predictors, options = {}) {
  return lassoFn(this.getColumn(y).toArray() as Num, rowsOf(this, predictors), options)
}
D.ancova = function (this: DataFrame, y, group, covariate) {
  return ancovaFn(this.getColumn(y).toArray() as Num, this.getColumn(group).toArray(), this.getColumn(covariate).toArray() as Num)
}
D.ksTwoSample = function (this: DataFrame, column, by) {
  const g = groupsOf(this, column, by)
  const levels = Object.keys(g)
  if (levels.length !== 2) throw new RangeError(`ksTwoSample: column "${by}" must have exactly 2 levels, got ${levels.length}`)
  return ksTwoSampleFn(g[levels[0]!]!, g[levels[1]!]!)
}

// LazyFrame: materialize, then delegate
const L = LazyFrame.prototype
const lazy = <K extends keyof DataFrame>(name: K) =>
  async function (this: LazyFrame, ...args: unknown[]) {
    const frame = await this.collect()
    return (frame[name] as unknown as (...a: unknown[]) => unknown).apply(frame, args)
  }
for (const name of [
  'ttest', 'anova', 'anovaTwoWay', 'tukey', 'fisher', 'dunnett', 'hsu', 'gamesHowell', 'mannWhitney', 'kruskal', 'equalVariances', 'normalityTest', 'chi2test',
  'ztest', 'varTest', 'corrTest', 'partialCorr', 'outlierTest', 'signTest', 'wilcoxon', 'mood', 'friedman', 'runsTest', 'equivalence',
  'regress', 'fittedLine', 'stepwise', 'bestSubsets', 'glm', 'logistic', 'ologit', 'mlogit', 'linearModel', 'nls', 'orthogonalRegression', 'pls',
  'controlChart', 'ewma', 'cusum', 'capability', 'toleranceInterval', 'gageRR', 'pareto', 'runChart', 'individualDistributionID',
  'trendAnalysis', 'decompose', 'stl', 'ets', 'acf', 'arima', 'autoArima', 'reliabilityFit', 'kaplanMeier', 'logRank', 'coxPH', 'fineGray', 'mixedModel', 'glmm', 'nestedAnova', 'capabilitySixpack', 'manova', 'pca', 'kmeans', 'cart', 'randomForest',
  'descriptiveStats', 'graphicalSummary', 'poissonGof', 'boxplotStats', 'intervalPlot', 'mainEffectsPlot', 'interactionPlot', 'stabilityStudy', 'gChart', 'tChart', 't2Chart', 'mewma', 'periodogram',
  'lifeRegression', 'altRegression', 'powerLawNHPP', 'probitAnalysis', 'clusterVariables', 'multipleCorrespondence', 'itemAnalysis', 'manovaModel', 'crossValidate', 'autoModel',
  'propTest', 'adfTest', 'kpssTest', 'ridge', 'lasso', 'ancova', 'ksTwoSample',
] as const) {
  ;(L as unknown as Record<string, unknown>)[name] = lazy(name)
}

export const ADVANCED_INSTALLED = true
