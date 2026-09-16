# pandas → columna

| pandas | columna |
|---|---|
| `df[cols]` | `df.select(...cols)` |
| `df.query("age > 18")` | `df.filter(col('age').gt(18))` |
| `df.assign(x=...)` | `df.withColumn('x', expr)` |
| `df.sort_values('a')` | `df.sort('a')` / `df.sort(col('a').desc())` |
| `df.head(5)` | `df.head(5)` |
| `df.groupby('c').agg(...)` | `df.groupBy('c').agg({ ... })` |
| `df.merge(other, on=...)` | `df.join(other, { on })` / `leftJoin` / `innerJoin` |
| `df.fillna(0)` | `df.fillNull(0)` |
| `df.dropna()` | `df.dropNull()` |
| `pd.melt` | `df.melt({ idVars, valueVars })` |
| `df.pivot_table` | `df.pivot({ index, columns, values })` |
| `pd.concat` | `LazyFrame.concat([a, b])` |
| `df.describe()` | `df.describe()` — `df.describe({ quantileMethod: 'minitab' })` for Minitab's quartiles |
| `df.x.quantile(0.25)` (`interpolation='linear'`) | `col('x').quantile(0.25)` — type 7, position (n − 1)q |
| `np.percentile(x, 25, method='weibull')` / Minitab / SPSS / R `type = 6` | `col('x').quantile(0.25, 'minitab')`, `col('x').median('minitab')` — position q(n + 1), clamped; also `quantile(x, p)` from `columna/advanced` (Minitab by default, `{ method: 'linear' }` for pandas) |
| `df.corr()` / `df.corr(method="spearman")` | `df.corr()` / `df.corr({ method: 'spearman' })` — колонка `column` + матрица |
| `df.cov()` | `df.cov()` |
| `df['c'].value_counts()` | `df.valueCounts('c')` |
| `df.drop_duplicates()` | `df.unique()` |
| `df.rolling(3).mean()` | `df.rolling('out', 'col', 3, 'mean')` |
| `df['x'].rank(method='average')` | `df.withWindow('r', 'rank', { orderBy: ['x'], method: 'average' \| 'min' \| 'max' \| 'dense' \| 'ordinal' })` |
| `df['x'].shift(1)` / lead | `df.withWindow('prev', 'lag' \| 'lead', { expr: col('x'), offset: 1 })` |
| `df.iloc[a:b]` | `df.slice(a, b)` |
| `df.iloc[[i,j]]` | `df.take([i, j])` |
| `pd.read_csv` | `DataFrame.fromCSV(text)` / `fromCSVUrl` |
| `pd.read_json` | `DataFrame.fromJSON(...)` |

Notes:
- Chains are **lazy**; call `.collect()` (or `.toArray()`) to execute.
- Prefer `col` / `lit` expressions so WASM/WebGPU can accelerate.
- Force engine: `.engine('webgpu' | 'wasm' | 'cpu')`.
- Inspect plan: `.explain()`.

Broadcast aggregates (scalar context):

| pandas | columna |
|---|---|
| `(df.x - df.x.mean()) / df.x.std()` | `col('x').sub(col('x').mean()).div(col('x').std())` |
| `df.x / df.x.sum()` | `col('x').div(col('x').sum())` |
| `df[df.x > df.x.median()]` | `df.filter(col('x').gt(col('x').median()))` |

Window aggregates (pandas `transform`):

| pandas | columna |
|---|---|
| `df.groupby('g').x.transform('mean')` | `col('x').mean().over('g')` |
| `df.x - df.groupby('g').x.transform('mean')` | `col('x').sub(col('x').mean()).over('g')` |
| `df.x / df.groupby(['g','h']).x.transform('sum')` | `col('x').div(col('x').sum().over(['g', 'h']))` |
| `df[df.x > df.groupby('g').x.transform('mean')]` | `df.filter(col('x').gt(col('x').mean().over('g')))` |
| `df.sort_values('t').groupby('g').x.cumsum()` (back in original order) | `col('x').sum().over('g', { orderBy: 't' })` |
| `df.sort_values('t').groupby('g').x.expanding().mean()` | `col('x').mean().over('g', { orderBy: 't' })` |
| `df.sort_values('t').x.cumsum()` | `col('x').sum().over([], { orderBy: 't' })` |

Math (element-wise, numeric):

| pandas / numpy | columna |
|---|---|
| `np.sqrt(df.x)` / `np.log(df.x)` / `np.exp(df.x)` | `col('x').sqrt()` / `.log()` / `.exp()` |
| `np.log10(df.x)`, `np.log2`, `np.log(x)/np.log(b)` | `col('x').log10()`, `.log2()`, `.log(b)` |
| `df.x ** 2`, `np.power(a, b)` | `col('x').pow(2)`, `col('a').pow(col('b'))` |
| `df.x.round(2)` (half-to-even) | `col('x').round(2)` (half away from zero, as Minitab / Excel) |
| `np.floor` / `np.ceil` / `np.sign` | `.floor()` / `.ceil()` / `.sign()` |
| `df.a + ' ' + df.b`, `df.a.str.cat(df.b, sep)` | `col('a').str.concat(col('b'), ' ')` |
| `df.s.str.zfill(4)` / `.str.pad(6, side='right')` | `col('s').str.padStart(4, '0')` / `.str.padEnd(6)` |

Everything below this line is in **`columna/advanced`** (`import … from 'columna/advanced'`).

Distributions (scipy.stats):

| scipy | columna |
|---|---|
| `norm(mu, sd).cdf(x)` / `.sf` / `.ppf` / `.isf` / `.pdf` | `dist.normal(mu, sd).cdf(x)` … |
| `t(df)`, `chi2(k)`, `f(d1, d2)` | `dist.t(df)`, `dist.chi2(k)`, `dist.f(d1, d2)` |
| `binom(n, p).pmf/cdf/ppf`, `poisson(mu)`, `hypergeom.pmf` | `dist.binomial(n, p)`, `dist.poisson(mu)`, `hypergeomPmf(k, N, K, n)` |
| `nct.cdf(x, df, nc)`, `ncf.cdf`, `ncx2.cdf` | `nctCdf(x, df, nc)`, `ncfCdf(x, d1, d2, nc)`, `ncChi2Cdf(x, k, nc)` |
| `scipy.special.gammaln / gammainc / betainc / erf` | `lgamma / gammainc / betainc / erf` |

Hypothesis tests (scipy.stats):

| scipy | columna |
|---|---|
| `ttest_1samp(x, mu)` | `ttest1(x, { mu })` / `df.ttest('x', { mu })` |
| `ttest_ind(a, b, equal_var=False)` | `ttest2(a, b)` / `df.ttest('x', { by: 'g' })` (Welch default; `equalVar: true` pools) |
| `ttest_rel(a, b)` | `ttestPaired(a, b)` / `df.ttest('a', { paired: 'b' })` |
| `f_oneway(*groups)` | `anova({ g1, g2, … })` / `df.anova('x', 'g')` |
| `tukey_hsd(*groups)` | `tukeyHSD({ g1, g2, … })` / `df.tukey('x', 'g')` — plus Minitab grouping letters |
| `dunnett(*samples, control=…, alternative)` | `dunnett(groups, { control, alternative })` / `df.dunnett('x', 'g', { control })` |
| — (statsmodels `pairwise_tukeyhsd` has no LSD) | `fisherLSD(groups)` / `df.fisher('x', 'g')` |
| — (Minitab only) | `hsuMCB(groups, { best })` / `df.hsu('x', 'g')` |
| `levene(*groups, center='median')` / `bartlett(*groups)` | `levene(groups)` / `bartlett(groups)` / `bonett(groups)` / `df.equalVariances('x', 'g', method)` |
| — (Minitab 2 Variances, Bonett) | `bonett2(a, b)` |
| `studentized_range.cdf(q, k, df)` / `.ppf` | `ptukey(q, k, df)` / `qtukey(p, k, df)` |
| `chi2_contingency(table, correction=False)` | `chi2test(table)` / `df.chi2test('a', 'b')` |
| `chisquare(obs, f_exp, ddof)` | `chi2gof(obs, expected, { ddof })` |
| `pd.crosstab(a, b)` | `crosstab(a, b)` |
| `mannwhitneyu(a, b, alternative, method)` | `mannWhitney(a, b, { alternative, method })` / `df.mannWhitney('x', 'g')` — plus Hodges–Lehmann estimate and CI |
| `kruskal(*groups)` | `kruskal(groups)` / `df.kruskal('x', 'g')` — plus per-group medians, average ranks, z |
| `shapiro(x)` | `shapiroWilk(x)` / `df.normalityTest('x', 'shapiro-wilk')` |
| `anderson(x, 'norm')` (statistic only) | `andersonDarling(x)` / `df.normalityTest('x')` — with Minitab's p-value |
| `kstest(x, 'norm', args=(mean, sd))` statistic; statsmodels `lilliefors` p | `kolmogorovSmirnov(x)` / `df.normalityTest('x', 'kolmogorov-smirnov')` |
| — (Minitab only) | `ryanJoiner(x)` / `df.normalityTest('x', 'ryan-joiner')` |
| statsmodels `ztest(x, value=mu)` | `ztest1(x, { sigma, mu })` / `df.ztest('x', { sigma, mu })` |
| `binomtest(k, n, p).pvalue` + `.proportion_ci(method='exact')` | `propTest1(k, n, { p0 })` (exact + Clopper–Pearson; `method: 'normal'` Wald) |
| statsmodels `proportions_ztest` / `fisher_exact(table)` | `propTest2(e1, n1, e2, n2)` / `{ method: 'fisher' }` |
| `poisson.cdf` by hand / statsmodels `test_poisson` | `poissonRateTest1(events, exposure, { lambda0 })`, `poissonRateTest2(...)` |
| `chi2` by hand (1 variance) | `varTest1(x, { sigma0 })` / `df.varTest`; `{ method: 'bonett' }` for non-normal data |
| `pearsonr(a, b)` + `.confidence_interval()`, `spearmanr(a, b)` | `corrTest(a, b)` / `df.corrTest('a', 'b', { method })` |
| `outliers::grubbs.test` / `dixon.test` (R) | `grubbs(x)`, `dixon(x)` / `df.outlierTest('x', { method })` |
| `binomtest(above, above+below)` (sign) | `signTest(x, { median })` / `df.signTest` — plus Minitab's interpolated CI |
| `wilcoxon(x - mu, method, zero_method='wilcox')` | `wilcoxonSigned(x, { median, method })` / `df.wilcoxon` — plus Walsh-average estimate and CI |
| `median_test(*groups, ties='below', correction=False)` | `moodMedian(groups)` / `df.mood('x', 'g')` |
| `friedmanchisquare(*columns)` | `friedman(table)` / `df.friedman('y', 'treatment', 'block')` |
| statsmodels `runstest_1samp(x, cutoff='mean')` | `runsTest(x, { k, correction })` / `df.runsTest('x')` |
| statsmodels `ttost_paired` / `ttost_ind` | `tost1`, `tost2`, `tostPaired` / `df.equivalence('y', { limits, by | paired })` |
| statsmodels `TTestPower().solve_power`, `NormalIndPower`, `FTestAnovaPower` | `power({ test, effect, n, power, alpha, alternative })` |

Regression and models (statsmodels / sklearn / scipy):

| Python | columna |
|---|---|
| `sm.OLS(y, sm.add_constant(X)).fit()` + `.summary()`, `OLSInfluence` | `ols(y, X)` / `df.regress('y', [...])` — coefficients, ANOVA, VIF, leverage, Cook's D, DFFITS, studentized residuals, unusual observations |
| `np.polyfit(x, y, deg)` | `fittedLine(x, y, { degree })` / `df.fittedLine('x', 'y')` |
| `sm.WLS` | `ols(y, X, { weights })` |
| `.get_prediction(Xnew).summary_frame()` | `result.predict(xNew)` → fit, se, ci, pi |
| `statsmodels` stepwise recipes / R `leaps::regsubsets` | `stepwise(y, X, { method, alphaIn, alphaOut })`, `bestSubsets(y, X)` |
| `sm.Logit(y, X).fit()`, `sm.GLM(family=Binomial())` | `logit(y, X)` / `df.logistic('y', [...])`; `{ trials }` for events / trials, `link: 'probit' \| 'cloglog'` |
| `sm.GLM(y, X, family=Poisson(), offset=np.log(t))` | `poissonRegression(y, X, { offset })` / `df.glm('y', [...], { family: 'poisson', offset })` |
| `OrderedModel(y, X, distr='logit')` (note: opposite sign convention) | `ologit(y, X)` / `df.ologit('y', [...])` — Minitab's logit P(Y ≤ k) = θ_k + x'β |
| `sm.MNLogit(y, X)` | `mlogit(y, X, { reference })` / `df.mlogit` |
| `smf.ols('y ~ C(a)*C(b) + x').fit()` + `anova_lm(typ=3)` (with sum contrasts) | `linearModel(data, 'y ~ a*b + x')` / `df.linearModel('y ~ a*b + x')` |
| `scipy.optimize.curve_fit(f, x, y, p0)` | `nls(f, x, y, { start })` / `df.nls(f, 'x', 'y', { start })` — plus t / p / CI, correlation matrix, history, predict() |
| `scipy.odr` with `sx`, `sy` | `orthogonalRegression(x, y, { errorVarianceRatio })` |
| `sklearn.cross_decomposition.PLSRegression(n_components)` | `pls(y, X, { components, standardize })` — plus R²X / R²Y per component and LOO predicted R² |

Quality tools / SPC (qcc, SixSigma, AIAG MSA):

| Python / R | columna |
|---|---|
| ASTM / Minitab chart constants | `spcConstants(n)` → A2, A3, B3, B4, D3, D4, d2, c4 |
| `qcc::qcc` (xbar.one / xbar / R / S / p / np / c / u) | `controlChart(x, { type, subgroup, sizes })` / `df.controlChart` — Nelson rules 1–8 |
| `qcc::ewma`, `qcc::cusum` | `ewma({ lambda, L })`, `cusum({ h, k })`, `movingAverage({ span })` |
| `qcc::process.capability` / SixSigma | `capability(x, { lsl, usl, subgroup, target })` / `df.capability` — Cp, Cpk, Pp, Ppk, Cpm, PPM, Z.Bench |
| `scipy.stats.boxcox`, `weibull_min.fit` | `boxCoxLambda`, `johnsonFit`, `weibullFit` |
| R `tolerance` | `toleranceInterval(x, { coverage, confidence, method })` |
| AIAG / Minitab Gage R&R | `gageRR({ part, operator, measurement })`, `gageLinearity`, `gageType1` |
| `statsmodels.stats.inter_rater` | `attributeAgreement` — Cohen / Fleiss κ, Kendall W / τ |
| ANSI Z1.4 OC | `acceptanceSampling({ type, n, c \| k })` — Pa, AOQ, ATI |
| Pareto / run chart / IDI | `pareto`, `runChart`, `multiVari`, `symmetryTest`, `individualDistributionID` |

Time series / DOE / reliability / multivariate / predictive:

| Python / R | columna |
|---|---|
| `np.polyfit` / Minitab Trend Analysis | `trendAnalysis(y, { model: 'linear'\|'quadratic'\|'exponential'\|'s-curve' })` |
| `statsmodels.tsa.seasonal_decompose` | `decompose(y, { seasonLength })` |
| `statsmodels.tsa.seasonal.STL` | `stl(y, { seasonLength, robust? })` |
| `ExponentialSmoothing` / Holt–Winters | `ets(y, { method: 'ses'\|'des'\|'winters-add'\|'winters-mul' })` — `forecastLower`/`forecastUpper` |
| `acf` / `pacf` / `ccf`, `acorr_ljungbox` | `acf`, `pacf`, `ccf`, `ljungBox` |
| `ARIMA` / SARIMA / ARIMAX / auto_arima | `arima(...)`, **`autoArima`** (stepwise AIC/BIC; orders ≤5) |
| `pyDOE2` / Minitab DOE + Taguchi S/N | `fullFactorial`…`taguchi` (L4–L36), `definitiveScreening`, `mixtureDesign`/`analyzeMixture`, `responseOptimizer`, `analyzeTaguchi` |
| `lifelines` / `weibull_min.fit` (censored) | `reliabilityFit` (MLE SE/CI), `kaplanMeier`, `probabilityPlot`, `warrantyPrediction` |
| `lifelines.statistics.logrank_test` / `survdiff` / CoxPHFitter | `logRank(..., { weight, strata })`, **`coxPH`** (strata, frailty, counting-process) |
| `statsmodels.MixedLM` / lme4 / `glmer` | **`mixedModel`** (RI+RS), **`glmm`** (binomial/Poisson PQL) |
| `sklearn.decomposition.PCA`, `FactorAnalysis` | `pca`, `factorAnalysis` (`method: 'pca'\|'ml'` + varimax) |
| `KMeans`, hierarchical, `LDA`/`QDA` | `kmeans`, `hclust`, `discriminant` |
| `sklearn.tree` / `RandomForest` / GBM / `GradientBoostingClassifier` / `py-earth` | `cart`, `randomForest`, `treeNet` (`task: 'regression'\|'classification'`), `mars` (GCV prune) |
| `sklearn.model_selection.cross_val_score`, AutoML | `crossValidate(X, y, { model, folds })`, `autoModel(X, y)` |

Remaining Minitab menu items:

| Python / R | columna |
|---|---|
| `df.describe()` + `scipy.stats.describe` / Minitab Display Descriptive Statistics | `descriptiveStats(x \| { a, b }, { by })` — Minitab's full table; `graphicalSummary(x)` |
| `scipy.stats.chisquare` on Poisson expected counts | `poissonGof(counts)` (tail pooling, df = k − 2) |
| `np.percentile(method='weibull')`, `plt.boxplot` stats | `boxplotStats`, `dotplot`, `ecdf`, `intervalPlot`, `mainEffectsPlot`, `interactionPlot` |
| ICH Q1E shelf-life (R `stability`) | `stabilityStudy(y, time, batch, { lsl, usl })` |
| `qcc` g / t charts; `MSQC::mult.chart` (T², MEWMA) | `gChart`, `tChart`, `t2Chart`, `mewma`, `generalizedVarianceChart` |
| `scipy.signal.periodogram`, `spec.pgram` (R), `cpgram` | `periodogram(x, { spans, taper, detrend })`, `cumulativePeriodogram` |
| `FrF2::aliases` (R) | `aliasStructure(k, generators)` |
| `survreg` (R) / `lifelines.WeibullAFTFitter` | `lifeRegression(time, X, { distribution, censor, time2 })` |
| ALTA / `survreg` with Arrhenius | `altRegression(time, stress, { relation, useStress })` |
| Minitab Test Plans | `demonstrationTestPlan`, `estimationTestPlan` |
| `reliability::crow_amsaa` / Crow–AMSAA | `powerLawNHPP(times, { endTime })` |
| R `MASS::dose.p` / Minitab Probit Analysis | `probitAnalysis(events, trials, stress, { distribution, naturalResponse })` |
| `scipy.cluster.hierarchy.linkage(1 − r)` | `clusterVariables(data, { method, distance, nClusters })` |
| `prince.MCA` / `FactoMineR::MCA` | `multipleCorrespondence(data, { method })` |
| `pingouin.cronbach_alpha` / `psych::alpha` | `itemAnalysis(items)` |
| `psych::promax` | `promax(loadings, { power })` |
| `statsmodels.multivariate.manova.MANOVA` | `manovaModel(data, responses, 'a*b + x')` |
| `np.random.default_rng(seed)` | `random(seed).normal(n, mean, sd)` … (16 distributions), `patterned(from, to)` |
| `scipy.stats.weibull_min / gamma / beta / lognorm / logistic / gumbel_l` | `dist.weibull / gamma / beta / lognormal / logistic / smallestExtremeValue` |
