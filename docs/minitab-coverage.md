# Minitab → columna coverage

Status legend:

| Status | Meaning |
|---|---|
| **есть** | API covers the usual Minitab workflow for this tool (numbers / options close enough for production use) |
| **MVP** | Usable core exists; missing depth, variants, auto-options, or report-grade parity |
| **нет** | Not implemented (or only a distant substitute) |

Scope: **Stat / Quality / DOE / Predictive** analytic tools. Graph Builder, Assistant guided UX, Session reports, and project/worksheet UI are out of scope for `@columna/advanced` (marked **нет** as product features).

Overall estimate (2026-09): **~75–85%** of typical QC/engineering Minitab use; **~50–60%** of full Minitab Statistical Software surface.

Related: [advanced-roadmap.md](advanced-roadmap.md), [pandas-to-columna.md](pandas-to-columna.md).

---

## Stat › Basic Statistics

| Minitab | Status | columna |
|---|---|---|
| Display Descriptive Statistics | **есть** | `descriptiveStats` / `df.descriptiveStats` — full Minitab table (N, N*, SE, CoefVar, quartiles, mode, skew, kurt, MSSD, trimmed; by group); Minitab quartile definition (p(n + 1)) also in the engine: `col('x').quantile(p, 'minitab')`, `df.describe({ quantileMethod: 'minitab' })`, `quantile(x, p)` |
| Store Descriptive Statistics | **есть** | `descriptiveStats` rows (one per variable × level) |
| 1-Sample Z | **есть** | `ztest1` / `df.ztest` |
| 1-Sample t | **есть** | `ttest1` / `df.ttest` |
| 2-Sample t | **есть** | `ttest2` / `df.ttest({ by })` |
| Paired t | **есть** | `ttestPaired` / `df.ttest({ paired })` |
| 1 Proportion | **есть** | `propTest1` |
| 2 Proportions | **есть** | `propTest2` |
| 1-Sample Poisson Rate | **есть** | `poissonRateTest1` |
| 2-Sample Poisson Rate | **есть** | `poissonRateTest2` |
| 1 Variance | **есть** | `varTest1` (χ² / Bonett) |
| 2 Variances | **есть** | `varTest2`, `levene`, `bartlett`, `bonett2` |
| Correlation | **есть** | `corrTest` (Pearson / Spearman / Kendall + CI); `partialCorr` / `df.partialCorr` |
| Graphical Summary | **есть** | `graphicalSummary` — descriptives + A-D + CIs + histogram / boxplot data |
| Goodness-of-Fit Test for Poisson | **есть** | `poissonGof` |
| Covariance | **есть** | `df.cov()` (standard) |
| Normality Test | **есть** | AD, RJ, KS, Shapiro, Jarque–Bera, D’Agostino K², Cramér–von Mises / `normalityTest` |
| Outlier Test | **есть** | `grubbs`, `dixon` / `df.outlierTest` |

## Stat › Nonparametrics

| Minitab | Status | columna |
|---|---|---|
| 1-Sample Sign | **есть** | `signTest` |
| 1-Sample Wilcoxon | **есть** | `wilcoxonSigned` |
| Mann-Whitney | **есть** | `mannWhitney` |
| Kruskal-Wallis | **есть** | `kruskal` |
| Mood’s Median Test | **есть** | `moodMedian` |
| Friedman | **есть** | `friedman` |
| Runs Test | **есть** | `runsTest` |

## Stat › ANOVA / GLM

| Minitab | Status | columna |
|---|---|---|
| One-Way ANOVA | **есть** | `anova` |
| Two-Way ANOVA / ANCOVA | **MVP** | `anovaTwoWay`, `ancova` (Type III via `linearModel`) |
| Balanced ANOVA | **есть** | `linearModel` (balanced designs: Type III = Type I) + `nestedAnova` EMS |
| General Linear Model | **есть** | `linearModel` (Type I/III, effect coding) |
| Mixed Effects / Random | **MVP** | `mixedModel` RI (+ RS / nested / crossed `group2`, correlated `rho`); **`glmm`** PQL/Laplace/AGQ (+ optional RS, **correlated AGQ slope ρ**) |
| Fully Nested ANOVA | **MVP** | `nestedAnova` EMS/F + `mixedModel({ groupNested })` |
| MANOVA | **есть** | one-way `manova` + general `manovaModel(data, responses, 'a*b + x')` (Type III SSCP, 4 statistics, univariate F) |
| Comparisons: Tukey | **есть** | `tukeyHSD` |
| Comparisons: Fisher LSD | **есть** | `fisherLSD` |
| Comparisons: Dunnett | **есть** | `dunnett` |
| Comparisons: Hsu MCB | **есть** | `hsuMCB` |
| Comparisons: other (Games-Howell, …) | **есть** | `gamesHowell` |
| Main Effects / Interaction Plots | **есть** | `mainEffectsPlot`, `interactionPlot`, `intervalPlot` (plot data) |
| Test for Equal Variances | **есть** | Levene / Bartlett / Bonett |

## Stat › Regression

| Minitab | Status | columna |
|---|---|---|
| Fitted Line Plot | **есть** | `fittedLine` |
| Regression | **есть** | `ols` / `df.regress` (+ diagnostics) |
| Stepwise | **есть** | `stepwise` |
| Best Subsets | **есть** | `bestSubsets` |
| Binary Logistic | **есть** | `glm` / `logit` (logit/probit/cloglog) |
| Ordinal Logistic | **есть** | `ologit` |
| Nominal Logistic | **есть** | `mlogit` |
| Poisson Regression | **есть** | `poissonRegression` |
| Nonlinear Regression | **есть** | `nls` |
| Orthogonal Regression | **есть** | `orthogonalRegression` |
| Partial Least Squares | **есть** | `pls` |
| Stability Study | **есть** | `stabilityStudy` — model pooling at α = 0.25, shelf life per batch / overall |
| Response Optimizer | **MVP** | `responseOptimizer` (Derringer–Suich; also under DOE) |
| Predict | **MVP** | `predict()` on fits; no Session report UI |

## Stat › Tables / Equivalence / Power

| Minitab | Status | columna |
|---|---|---|
| Chi-Square Test (tables) | **есть** | `chi2test`, `crosstab` |
| Chi-Square Goodness-of-Fit | **есть** | `chi2gof` |
| Cross Tabulation | **есть** | `crosstab` |
| Equivalence Tests (TOST) | **есть** | `tost1` / `tost2` / `tostPaired` |
| Power and Sample Size | **есть** | `power` (9 Minitab tests) |

## Stat › Quality Tools

| Minitab | Status | columna |
|---|---|---|
| Capability Analysis (Normal) | **есть** | `capability` |
| Capability (Nonnormal) | **MVP** | Box–Cox / Johnson / Weibull fits; not full Minitab nonnormal report |
| Capability Sixpack | **MVP** | `capabilitySixpack` plot-ready panels (no renderer) |
| Tolerance Intervals | **есть** | `toleranceInterval` |
| Individual Distribution Identification | **есть** | `individualDistributionID` |
| Pareto Chart | **MVP** | `pareto` (counts/weights; no chart object) |
| Cause-and-Effect | **есть** | `causeAndEffect` — structure, layout, SVG |
| Multi-Vari Chart | **MVP** | `multiVari` |
| Symmetry Plot / Test | **MVP** | `symmetryTest` |
| Run Chart | **есть** | `runChart` |
| Acceptance Sampling | **есть** | `acceptanceSampling` |
| Attribute Agreement Analysis | **есть** | `attributeAgreement` |
| Gage R&R (Crossed/Nested) | **есть** | `gageRR` |
| Gage Linearity and Bias | **есть** | `gageLinearity` |
| Gage Run Chart / Type 1 | **MVP** | `gageType1` |

## Stat › Control Charts

| Minitab | Status | columna |
|---|---|---|
| I-MR | **есть** | `controlChart` |
| X̄-R / X̄-S | **есть** | `controlChart` |
| Z-MR | **есть** | `controlChart` |
| P / NP / C / U | **есть** | attribute charts |
| Laney P′ / U′ | **есть** | Laney |
| EWMA | **есть** | `ewma` |
| CUSUM | **есть** | `cusum` (+ V-mask params) |
| Moving Average | **есть** | `movingAverage` |
| Rare Event: G / T charts | **есть** | `gChart`, `tChart` |
| Multivariate: T² / MEWMA / Generalized Variance | **есть** | `t2Chart`, `mewma`, `generalizedVarianceChart` |
| Box-Cox / Johnson transform in chart | **MVP** | transforms exist separately |
| Nelson rules 1–8 | **есть** | in `controlChart` |

## Stat › Time Series

| Minitab | Status | columna |
|---|---|---|
| Trend Analysis | **есть** | `trendAnalysis` |
| Decomposition | **есть** | `decompose` |
| STL | **MVP** | `stl` Cleveland LOESS; multi-season via `{ periods: number[] }` |
| Moving Average / Single Exp Smoothing | **есть** | via `ets` SES / MA helpers |
| Double Exp Smoothing | **есть** | `ets` DES |
| Winters’ Method | **MVP** | `ets` winters-add/mul + PI |
| ACF / PACF / CCF | **есть** | `acf`, `pacf`, `ccf` |
| Autocorrelation (Ljung–Box) | **есть** | `ljungBox` |
| ARIMA | **MVP** | `arima` CSS-ML; `method: 'ML'` Kalman (incl. seasonal expand); `p,q,P,Q ≤ 5` |
| Seasonal ARIMA | **MVP** | `arima({ seasonal })` CSS-ML or ML |
| ARIMAX / with predictors | **MVP** | `xreg` / `transfer` ω(B)+δ(B); `transferIdentify`; **ML + transfer** (Kalman mean = μ + TF) |
| Auto ARIMA | **MVP** | `autoArima` stepwise AIC/BIC |
| Difference / Lag | **MVP** | via Expr / internal diffs |
| Spectral Analysis | **есть** | `periodogram`, `cumulativePeriodogram` |

## Stat › DOE

| Minitab | Status | columna |
|---|---|---|
| Create Factorial Design | **есть** | `fullFactorial`, `fractionalFactorial` |
| Create Plackett-Burman | **есть** | `plackettBurman` |
| Create Response Surface (CCD/BBD) | **есть** | `ccd`, `boxBehnken` |
| Create Taguchi Design | **MVP** | `taguchi` L4–L243 (incl. L20–L48 rare + L108/L121/L128) |
| Analyze Factorial / RSM | **MVP** | `analyzeEffects`, `analyzeDoe`, `fitDesign` |
| Analyze Taguchi Design | **MVP** | `analyzeTaguchi` (S/N, response tables) |
| Alias Structure | **есть** | `aliasStructure(k, generators)` — defining relation, resolution, alias chains |
| Definitive Screening | **MVP** | `definitiveScreening` |
| Mixture Designs | **MVP** | `mixtureDesign` + `analyzeMixture` (Scheffé ± process vars) |
| Response Optimizer | **MVP** | `responseOptimizer` (Derringer–Suich) |
| Full OA catalog (rare arrays) | **MVP** | `taguchi` L4–L243 (standard + extended set); exotic arrays beyond that are not tabulated |

## Stat › Reliability / Survival

| Minitab | Status | columna |
|---|---|---|
| Distribution Analysis (Right Censoring) | **MVP** | `reliabilityFit` Weibull/lognormal/exp |
| Parametric Distribution Analysis | **MVP** | MLE + SE/CI; left/interval partial |
| Nonparametric (Kaplan–Meier) | **есть** | `kaplanMeier` |
| Distribution ID Plot | **MVP** | `probabilityPlot` + IDI elsewhere |
| Warranty Analysis | **MVP** | `warrantyPrediction` |
| Test Plans | **есть** | `demonstrationTestPlan`, `estimationTestPlan` |
| Regression with Life Data | **есть** | `lifeRegression` — AFT (Weibull / lognormal / loglogistic / exponential / SEV / normal / logistic), censoring, percentiles |
| Probit Analysis | **есть** | `probitAnalysis` — ED_p with Fieller limits, natural response |
| Repairable Systems | **есть** | `powerLawNHPP` — Crow–AMSAA, trend tests, TTT |
| Survival: Log-Rank | **есть** | `logRank` (logrank / Wilcoxon / Tarone–Ware + strata) |
| Cox Regression | **MVP** | `coxPH` (+ cluster SE; cause-specific; piecewise / AR(1) TV frailty) |
| Fine–Gray / CIF | **MVP** | `fineGray` subdistribution + CIF |
| Accelerated Life Testing | **есть** | `altRegression` — Arrhenius / inverse power / exponential, use-condition percentiles |

## Stat › Multivariate

| Minitab | Status | columna |
|---|---|---|
| PCA | **есть** | `pca` |
| Factor Analysis | **есть** | `factorAnalysis` PCA + ML + varimax, `promax` |
| Cluster Observations | **MVP** | `kmeans`, `hclust` |
| Cluster Variables | **есть** | `clusterVariables` (+ `cutTree`) |
| Discriminant Analysis | **есть** | `discriminant` LDA/QDA |
| Simple Correspondence | **есть** | `correspondence` |
| Multiple Correspondence | **есть** | `multipleCorrespondence` (indicator / Burt) |
| Item Analysis / Cronbach | **есть** | `itemAnalysis` |
| MANOVA | **MVP** | one-way `manova` (also under ANOVA) |

## Stat › Predictive Analytics

| Minitab | Status | columna |
|---|---|---|
| CART Classification / Regression | **MVP** | `cart` |
| Random Forests | **MVP** | `randomForest` |
| TreeNet | **MVP** | `treeNet` reg + classification |
| MARS | **MVP** | `mars` + GCV prune; no interactions degree |
| Automated Machine Learning | **есть** | `autoModel` — CV-ranked OLS / CART / RF / TreeNet / MARS (logistic for binary) |
| Model validation / Discover reports | **есть** | `crossValidate` — k-fold metrics, out-of-fold predictions |

## Assistant / Graphs / Product

| Minitab | Status | columna |
|---|---|---|
| Assistant (guided analyses) | **нет** | — |
| Graph Builder / Graph menu | **MVP** | `plotSeries` + `renderPlotSeries` (static + interactive SVG lite); no product Graph Builder |
| Session / ReportPad | **нет** | — |
| Worksheet / Project files | **нет** | DataFrame IO instead |
| macros / exec | **нет** | JS/TS API |

---

## Summary counts (analytic tools only)

Approximate tallies from the tables above (Assistant/Graphs excluded from “stats core” denominator):

| Status | ~Count | Share of listed analytic tools |
|---|---|---|
| **есть** | ~101 | ~76% |
| **MVP** | ~28 | ~21% |
| **нет** | ~4 | ~3% (Full OA catalog beyond L243, Multiple-season STL, transfer-function noise joint ML, crossed-AGQ GLMM) |

(Assistant/Graphs/Product rows excluded; those remain **нет** as product features. Every remaining **MVP** has a working core.)

Weighted by everyday QC use (Basic + ANOVA + Regression + SPC + Capability + survival basics), effective coverage is closer to **80%+**.

## Coverage by menu (quick)

| Menu | Level |
|---|---|
| Basic Statistics / Nonparametrics / Power | high (**есть**-heavy) |
| ANOVA / Regression / GLM | high; Mixed/GLMM **MVP** (nested + NB); MANOVA **MVP** |
| Quality Tools / Control Charts | high; `plotSeries` + interactive SVG lite |
| Time Series | medium–high (ARIMA ML/xreg/δ TF + `transferIdentify`; no spectral/joint ML+TF) |
| DOE | medium–high (factorial/RSM/Taguchi L4–L243/DSD/mixture×process/optimizer) |
| Reliability / Survival | medium–high (parametric + KM + log-rank + Cox MVP + Fine–Gray + AR(1) frailty) |
| Multivariate / Predictive | medium (core tools + one-way MANOVA; no AutoML) |
| Assistant / Graphs / Product UI | interactive SVG lite + export; no Session/Assistant |

## Highest-value gaps (next)

1. Chart renderer beyond `renderPlotSeries` (all plot data — boxplot, ECDF, dotplot, interval / main-effects / interaction, fishbone SVG — is produced)
2. Crossed random effects with AGQ; multi-season STL; joint ML for transfer-function noise  
3. Product surface: Assistant, Session / ReportPad, Worksheet / Project files, macros (out of library scope)  

**Closed recently**

- Gap closure (Ярус 8): descriptive statistics / graphical summary, Poisson GOF, main-effects / interaction / interval plot data, boxplot / dotplot / ECDF, cause-and-effect, stability study, G / T charts, T² / MEWMA / generalized variance, spectral analysis, alias structure, life regression + ALT + test plans + repairable systems + probit, cluster variables, MCA, item analysis, promax, general MANOVA, AutoML / cross-validation, random & patterned data; continuous distributions (beta, gamma, weibull, lognormal, exponential, logistic, SEV)

- Interactive SVG lite; AGQ `glmm` + random slope; AR(1) frailty bands; `transferIdentify`; Taguchi L20/L24/L28/L40/L44/L48  
- SciPy parity Wave 1–3: Kendall/`partialCorr`, normality extras, scale/table tests, effect sizes/KDE/bootstrap, nbinom/hypergeom/gumbel/pareto/invgauss; two-way ANOVA/ANCOVA, ADF/KPSS, multi-season STL, ML+transfer ARIMA, AGQ correlated ρ; KS2/AD-k/energy, LOWESS/isotonic, ridge/lasso/QR, DBSCAN, nnls, Welch/SG, brentq  
- Earlier: `renderPlotSeries` SVG; AGQ RI + correlated LMM `rho`; piecewise frailty bands; TF `delta` + ML `xreg`; Taguchi L108/L121/L128/L243  
- Earlier: `capabilitySixpack`; crossed LMM `group2` + Laplace `glmm`; `fineGray` + TV frailty; seasonal Kalman ML + `transfer`; Taguchi L25/L81 + `nestedAnova`  
- Earlier: `plotSeries`; nested LMM + NB `glmm`; Cox cluster/cause-specific; nonseasonal ML; mixture×process; `gamesHowell`; one-way `manova`; Taguchi L50/L54/L64  
