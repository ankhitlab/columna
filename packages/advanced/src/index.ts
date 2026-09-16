/**
 * @columna/advanced — the statistics layer that brings columna up to Minitab: distributions,
 * hypothesis tests, ANOVA comparisons, variance and normality tests, nonparametrics. Importing this
 * module also installs the column-level methods on DataFrame / LazyFrame (`df.ttest(...)`, …).
 */
export {
  dist,
  normal,
  t,
  chi2,
  f,
  binomial,
  poisson,
  negativeBinomial,
  hypergeometric,
  hypergeomPmf,
  gumbel,
  invgauss,
  nctCdf,
  ncfCdf,
  ncChi2Cdf,
  lgamma,
  gammainc,
  gammaincc,
  betainc,
  erf,
  erfc,
  type Distribution,
  type DiscreteDistribution,
  type Dist,
} from './dist.js'
export {
  ztest1,
  propTest1,
  propTest2,
  poissonRateTest1,
  poissonRateTest2,
  varTest1,
  corrTest,
  partialCorr,
  grubbs,
  dixon,
  type ZTestResult,
  type PropTestResult,
  type RateTestResult,
  type VarTest1Result,
  type CorrTestResult,
  type PartialCorrResult,
  type OutlierResult,
} from './basic.js'
export {
  signTest,
  wilcoxonSigned,
  moodMedian,
  friedman,
  runsTest,
  type SignTestResult,
  type WilcoxonResult,
  type MoodResult,
  type FriedmanResult,
  type RunsTestResult,
} from './nonparametric2.js'
export { tost1, tost2, tostPaired, type TostResult } from './equivalence.js'
export { power, type PowerOptions, type PowerResult, type PowerTest } from './power.js'
export {
  stats,
  ttest1,
  ttest2,
  ttestPaired,
  anova,
  chi2test,
  chi2gof,
  crosstab,
  cleanNumbers,
  levene,
  bartlett,
  bonett,
  bonett2,
  varTest2,
  equalVariances,
  type Alternative,
  type TTestOptions,
  type TTestResult,
  type AnovaResult,
  type Chi2Result,
  type VarianceTestResult,
} from './tests.js'
export {
  andersonDarling,
  shapiroWilk,
  ryanJoiner,
  kolmogorovSmirnov,
  jarqueBera,
  dagostinoK2,
  cramerVonMises,
  normalityTest,
  type NormalityResult,
  type NormalityMethod,
} from './normality.js'
export { tukeyHSD, ptukey, qtukey, pooledGroups, groupingLetters, type TukeyResult, type TukeyComparison, type GroupStats } from './tukey.js'
export { fisherLSD, dunnett, pdunnett, qdunnett, hsuMCB, gamesHowell, type FisherResult, type DunnettResult, type HsuResult, type GamesHowellResult } from './multcomp.js'
export {
  mannWhitney,
  kruskal,
  fligner,
  ansariBradley,
  brunnerMunzel,
  moodTwoSample,
  eppsSingleton,
  type MannWhitneyResult,
  type KruskalResult,
  type TwoSampleScaleResult,
} from './nonparametric.js'
export { mcnemar, cochranQ, bowker, type PairedTableResult } from './tables2.js'
export { cohensD, hedgesG, glassDelta, type EffectSizeResult } from './effectsize.js'
export { gaussianKde, type GaussianKdeResult } from './density.js'
export { bootstrap, permutationTest, type BootstrapResult, type PermutationTestResult } from './resample.js'
export {
  ols,
  fittedLine,
  type OlsOptions,
  type OlsResult,
  type Coefficient,
  type AnovaRow,
  type TermRow,
  type Prediction,
  type Predictors,
} from './regression.js'
export { stepwise, bestSubsets, type StepwiseOptions, type StepwiseResult, type StepwiseStep, type BestSubsetsResult, type Subset } from './stepwise.js'
export {
  glm,
  logit,
  poissonRegression,
  ologit,
  mlogit,
  type GlmOptions,
  type GlmResult,
  type GlmCoefficient,
  type Family,
  type Link,
  type OrdinalResult,
  type OrdinalCoefficient,
  type NominalResult,
} from './glm.js'
export { linearModel, modelMatrix, parseFormula, type LinearModelResult, type LmTermRow, type LmCoefficient, type Formula, type Term, type ModelMatrix } from './lm.js'
export { nls, type NlsOptions, type NlsResult, type NlsParameter, type NlsModel } from './nls.js'
export { orthogonalRegression, pls, type OrthogonalResult, type PlsOptions, type PlsResult, type PlsComponentRow } from './pls.js'
export { matrix, fromRows, fromColumns as matrixFromColumns, transpose, matmul, inverse, qr, lstsq, svd, cholesky, symmetricEigen, eigh, nnls, type Matrix } from './linalg.js'
export {
  spcConstants,
  controlChart,
  nelsonRules,
  ewma,
  cusum,
  movingAverage,
  type SpcConstants,
  type ControlChartResult,
  type ControlChartType,
  type ControlChartOptions,
  type ChartPoint,
  type NelsonRule,
  type EwmaResult,
  type CusumResult,
  type MovingAverageResult,
} from './spc.js'
export {
  capability,
  boxCoxLambda,
  johnsonFit,
  weibullFit,
  toleranceInterval,
  capabilitySixpack,
  type CapabilityResult,
  type BoxCoxResult,
  type JohnsonFit,
  type JohnsonFamily,
  type WeibullFit,
  type ToleranceIntervalResult,
  type CapabilitySixpackResult,
} from './capability.js'
export {
  gageRR,
  gageLinearity,
  gageType1,
  attributeAgreement,
  acceptanceSampling,
  type GageRRResult,
  type GageComponent,
  type GageLinearityResult,
  type GageType1Result,
  type KappaResult,
  type AcceptancePlan,
  type AcceptanceResult,
  type AcceptanceCurvePoint,
} from './msa.js'
export {
  pareto,
  runChart,
  multiVari,
  symmetryTest,
  individualDistributionID,
  type ParetoResult,
  type ParetoItem,
  type RunChartResult,
  type MultiVariResult,
  type SymmetryResult,
  type IdResult,
  type IdFit,
  type IdDistribution,
} from './quality.js'
export {
  trendAnalysis,
  decompose,
  stl,
  ets,
  acf,
  pacf,
  ccf,
  transferIdentify,
  ljungBox,
  arima,
  autoArima,
  type TrendModel,
  type TrendResult,
  type DecompositionResult,
  type StlResult,
  type EtsResult,
  type AcfResult,
  type PacfResult,
  type CcfResult,
  type TransferIdentifyResult,
  type LjungBoxResult,
  type ArimaResult,
} from './timeseries.js'
export {
  fullFactorial,
  fractionalFactorial,
  plackettBurman,
  ccd,
  boxBehnken,
  taguchi,
  analyzeEffects,
  analyzeDoe,
  designMatrix,
  fitDesign,
  analyzeTaguchi,
  definitiveScreening,
  mixtureDesign,
  analyzeMixture,
  responseOptimizer,
  nestedAnova,
  type DoeDesign,
  type EffectEstimate,
  type EffectsAnalysis,
  type SnRatio,
  type TaguchiFactorSummary,
  type TaguchiAnalysis,
  type MixtureAnalysis,
  type OptimizerGoal,
  type ResponseOptimizerResult,
  type NestedAnovaResult,
  type NestedAnovaComponent,
} from './doe.js'
export {
  mixedModel,
  glmm,
  type MixedModelResult,
  type MixedFixedEffect,
  type MixedRanef,
  type GlmmResult,
} from './mixed.js'
export {
  reliabilityFit,
  kaplanMeier,
  probabilityPlot,
  warrantyPrediction,
  logRank,
  coxPH,
  fineGray,
  type SurvivalObs,
  type ParametricSurvival,
  type KaplanMeierResult,
  type KaplanMeierPoint,
  type ProbabilityPlotData,
  type WarrantyResult,
  type LogRankResult,
  type CoxPHResult,
  type FineGrayResult,
} from './reliability.js'
export {
  pca,
  factorAnalysis,
  kmeans,
  hclust,
  discriminant,
  correspondence,
  type PcaResult,
  type FactorResult,
  type KMeansResult,
  type HclustResult,
  type DiscriminantResult,
  type CorrespondenceResult,
} from './multivariate.js'
export { manova, type ManovaResult } from './manova.js'
export { plotSeries, renderPlotSeries, type PlotSeries, type PlotSeriesResult, type PlotRole, type PlottableResult } from './plot.js'
export {
  cart,
  randomForest,
  treeNet,
  mars,
  type CartOptions,
  type CartResult,
  type RandomForestResult,
  type TreeNetResult,
  type MarsResult,
  type TreeTask,
} from './predictive.js'
export { beta, gamma, weibull, lognormal, exponential, logistic, smallestExtremeValue, pareto as paretoDist } from './dist.js'
export {
  quantile,
  descriptiveStats,
  graphicalSummary,
  poissonGof,
  boxplotStats,
  intervalPlot,
  mainEffectsPlot,
  interactionPlot,
  ecdf,
  dotplot,
  causeAndEffect,
  type DescriptiveStats,
  type GraphicalSummary,
  type PoissonGofResult,
  type BoxplotStats,
  type IntervalPlotRow,
  type MainEffectsPlot,
  type InteractionPlot,
  type EcdfResult,
  type DotplotResult,
  type CauseAndEffect,
  type FishboneNode,
} from './descriptive.js'
export { stabilityStudy, type StabilityOptions, type StabilityResult } from './stability.js'
export { gChart, tChart, t2Chart, mewma, generalizedVarianceChart, type RareEventChartResult, type T2ChartResult, type MewmaResult, type GeneralizedVarianceResult } from './spc2.js'
export { periodogram, cumulativePeriodogram, type PeriodogramResult, type CumulativePeriodogramResult } from './spectral.js'
export { aliasStructure, type AliasStructure } from './alias.js'
export {
  lifeRegression,
  altRegression,
  demonstrationTestPlan,
  estimationTestPlan,
  powerLawNHPP,
  probitAnalysis,
  type LifeDistribution,
  type LifeRegressionResult,
  type LifeCoefficient,
  type AltResult,
  type AccelerationRelation,
  type DemonstrationPlan,
  type EstimationPlan,
  type NhppResult,
  type ProbitResult,
} from './reliability2.js'
export { clusterVariables, cutTree, multipleCorrespondence, itemAnalysis, promax, type ClusterVariablesResult, type McaResult, type ItemAnalysisResult } from './multivariate2.js'
export { manovaModel, type ManovaModelResult, type ManovaTerm } from './manova.js'
export { crossValidate, autoModel, type CvResult, type AutoModelResult, type ModelKind, type Metrics } from './automl.js'
export { random, patterned, type Random } from './random.js'
export { nelderMead, newtonMax, numGradient, numHessian, type NelderMeadResult, type NewtonResult } from './optim.js'
export { ADVANCED_INSTALLED, type DataFrameTTestOptions, type MannWhitneyOptions } from './dataframe.js'
export { anovaTwoWay, ancova, anovaTwoWayBalanced, type AnovaTwoWayResult, type AncovaResult } from './anova2.js'
export { adfTest, kpssTest, type UnitRootResult } from './unitroot.js'
export { ksTwoSample, andersonKSample, energyDistance, type KsTwoSampleResult, type AndersonKSampleResult, type EnergyDistanceResult } from './gof2.js'
export { lowess, isotonicRegression } from './smooth.js'
export { ridge, lasso, elasticNet, quantileRegression, type PenalizedResult } from './penalized.js'
export { dbscan, pdist, cdist, type DbscanResult } from './cluster2.js'
export { welchPsd, savitzkyGolay, type WelchPsdResult } from './signal2.js'
export { brentq, trapz, simpson } from './numerics.js'
