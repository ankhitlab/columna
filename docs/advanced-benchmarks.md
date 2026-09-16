# Benchmarks: `columna/advanced` vs pandas / polars / scipy / numpy

Generated 2026-09-15 · Node v24.19.0 · scale 1 · median of 5 runs after a warm-up (heavy cases fewer).
Python: numpy 1.26.4, scipy 1.14.0, pandas 2.2.2, polars 1.44.1.

How to read: every row is one `columna/advanced` function on synthetic data of the stated size. The Python column times the closest
equivalent on data of the same size and distribution — **not** the same numbers (numerical agreement: [`advanced-convergence.md`](advanced-convergence.md)).
Where Python has no equivalent (statsmodels / sklearn / lifelines are not installed, or the tool is Minitab-only) the cell is "—".
Comparisons are throughput only: columna often computes more per call (full Minitab table: SE, CI, diagnostics, unusual observations),
and the scipy.optimize rows use a generic BFGS on the same likelihood rather than a purpose-built estimator.

Regenerate: `pnpm run bench:advanced` (TS → Python → this file). Quick pass: `BENCH_SCALE=0.1 pnpm run bench:advanced`.

## Distributions

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| normal.cdf × 1000000 | 108.9 ms | scipy | 42.5 ms | 2.6× slower |
| normal.ppf × 1000000 | 288.2 ms | scipy | 50.2 ms | 5.7× slower |
| t(10).cdf × 100000 | 23.0 ms | scipy | 20.7 ms | 1.1× slower |
| t(10).ppf × 100000 | 653.9 ms | scipy | 186.9 ms | 3.5× slower |
| chi2(5).sf × 100000 | 12.4 ms | scipy | 15.9 ms | 1.3× faster |
| f(3,20).ppf × 100000 | 770.6 ms | scipy | 133.0 ms | 5.8× slower |
| gamma(2.5,3).cdf × 100000 | 15.1 ms | scipy | 15.1 ms | 1.0× faster |
| beta(2,5).ppf × 10000 | 106.6 ms | scipy | 20.8 ms | 5.1× slower |
| weibull(1.8,50).ppf × 1000000 | 39.4 ms | scipy | 67.2 ms | 1.7× faster |
| binomial(50,0.3).cdf × 100000 | 46.4 ms | scipy | 8.90 ms | 5.2× slower |
| poisson(4).pmf × 100000 | 2.91 ms | scipy | 7.68 ms | 2.6× faster |
| nctCdf × 10000 | 6.76 ms | scipy | 9.63 ms | 1.4× faster |
| ptukey × 1000 | 1.07 s | scipy | 7.77 s | 7.2× faster |
| qtukey × 200 | 1.86 s | scipy | 22.17 s | 12× faster |

## Basic statistics

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| ttest1 n=100000 | 0.52 ms | scipy | 1.01 ms | 1.9× faster |
| ttest2 (Welch) n=100000+100000 | 1.47 ms | scipy | 0.96 ms | 1.5× slower |
| ttestPaired n=100000 | 1.31 ms | scipy | 1.32 ms | 1.0× faster |
| ztest1 n=100000 | 0.41 ms | numpy | 0.049 ms | 8.5× slower |
| propTest1 (exact) 350/1000 | 0.19 ms | scipy | 0.38 ms | 2.0× faster |
| propTest2 Fisher 120/400 vs 90/380 | 0.085 ms | scipy | 0.56 ms | 6.6× faster |
| poissonRateTest1 (exact) 120 events | 0.072 ms | scipy | 0.11 ms | 1.5× faster |
| varTest1 (χ²) n=100000 | 0.39 ms | scipy | 0.20 ms | 2.0× slower |
| varTest1 (Bonett) n=100000 | 9.29 ms | — | — | — |
| corrTest Pearson n=100000 | 2.90 ms | scipy | 2.13 ms | 1.4× slower |
| corrTest Spearman n=100000 | 61.8 ms | scipy | 20.4 ms | 3.0× slower |
| grubbs n=100000 | 0.83 ms | numpy | 0.32 ms | 2.6× slower |
| dixon n=25 (cached null distribution) | 0.050 ms | — | — | — |

## ANOVA

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| anova 5 × 20000 | 0.50 ms | scipy | 0.59 ms | 1.2× faster |
| levene 5 × 20000 | 15.2 ms | scipy | 1.49 ms | 10× slower |
| bartlett 5 × 20000 | 0.65 ms | scipy | 0.43 ms | 1.5× slower |
| bonett 5 × 20000 | 8.12 ms | — | — | — |
| tukeyHSD 6 × 5000 | 0.34 ms | scipy | 333.6 ms | 987× faster |
| fisherLSD 6 × 5000 | 0.21 ms | — | — | — |
| dunnett 6 × 5000 | 6.86 ms | scipy | 22.7 ms | 3.3× faster |
| hsuMCB 6 × 5000 | 3.63 ms | — | — | — |
| gamesHowell 6 × 5000 | 48.7 ms | — | — | — |
| equalVariances (Levene) 6 × 5000 | 3.48 ms | scipy | 0.60 ms | 5.8× slower |

## Tables

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| chi2test 20×20 table | 0.039 ms | scipy | 0.088 ms | 2.3× faster |
| chi2gof 50 categories | 0.004 ms | scipy | 0.025 ms | 5.7× faster |
| crosstab n=1000000 | 60.4 ms | pandas | 291.9 ms | 4.8× faster |
|  |  | polars | 8.60 ms | 7.0× slower |

## Normality

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| andersonDarling n=100000 | 32.9 ms | scipy | 20.3 ms | 1.6× slower |
| shapiroWilk n=5000 | 1.90 ms | scipy | 0.22 ms | 8.5× slower |
| ryanJoiner n=100000 | 34.0 ms | — | — | — |
| kolmogorovSmirnov (Lilliefors) n=100000 | 18.5 ms | scipy | 8.72 ms | 2.1× slower |
| individualDistributionID n=10000 | 104.2 ms | scipy | 107.2 ms | 1.0× faster |

## Nonparametrics

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| mannWhitney (asymptotic) 20000+20000 | 42.5 ms | scipy | 2.58 ms | 16× slower |
| mannWhitney (exact) 40+40 | 3.17 ms | scipy | 0.24 ms | 13× slower |
| kruskal 5 × 20000 | 36.6 ms | scipy | 14.2 ms | 2.6× slower |
| signTest n=100000 | 18.7 ms | scipy | 0.68 ms | 28× slower |
| wilcoxonSigned (asymptotic) n=20000 | 18.4 ms | scipy | 1.41 ms | 13× slower |
| wilcoxonSigned (exact) n=50 | 0.43 ms | scipy | 0.43 ms | 1.0× faster |
| moodMedian 5 × 20000 | 18.1 ms | scipy | 1.25 ms | 15× slower |
| friedman 5000 blocks × 5 | 6.91 ms | scipy | 152.2 ms | 22× faster |
| runsTest n=1000000 | 9.76 ms | numpy | 1.21 ms | 8.1× slower |

## Equivalence / power

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| tost1 n=100000 | 0.40 ms | numpy | 0.16 ms | 2.5× slower |
| tost2 n=100000+100000 | 1.41 ms | numpy | 0.34 ms | 4.1× slower |
| power 2-sample t (solve n) × 50 | 18.3 ms | — | — | — |
| power one-way ANOVA (solve effect) × 20 | 16.8 ms | — | — | — |

## Descriptive

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| descriptiveStats n=1000000 | 108.8 ms | pandas | 415.2 ms | 3.8× faster |
|  |  | polars | 12.8 ms | 8.5× slower |
| descriptiveStats by 10 groups n=1000000 | 139.3 ms | pandas | 97.9 ms | 1.4× slower |
|  |  | polars | 20.2 ms | 6.9× slower |
| graphicalSummary n=100000 | 73.2 ms | — | — | — |
| poissonGof n=1000000 | 24.9 ms | numpy | 21.8 ms | 1.1× slower |
| boxplotStats n=1000000 | 115.6 ms | numpy | 18.4 ms | 6.3× slower |
| mainEffectsPlot 3 factors n=1000000 | 156.2 ms | pandas | 160.6 ms | 1.0× faster |
|  |  | polars | 49.7 ms | 3.1× slower |
| interactionPlot 4×5 n=1000000 | 186.8 ms | pandas | 356.8 ms | 1.9× faster |
|  |  | polars | 19.6 ms | 9.5× slower |
| intervalPlot 10 groups n=1000000 | 45.3 ms | pandas | 65.9 ms | 1.5× faster |
|  |  | polars | 24.6 ms | 1.8× slower |
| ecdf n=1000000 | 171.6 ms | numpy | 59.6 ms | 2.9× slower |
| dotplot n=1000000 | 69.9 ms | numpy | 42.3 ms | 1.7× slower |
| causeAndEffect 6 categories × 5 causes | 0.041 ms | — | — | — |

## Regression

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| ols n=100000, p=10 (full diagnostics) | 136.1 ms | numpy | 43.9 ms | 3.1× slower |
| fittedLine (quadratic) n=100000 | 40.3 ms | numpy | 8.94 ms | 4.5× slower |
| stepwise n=10000, p=8 | 61.3 ms | — | — | — |
| bestSubsets n=10000, p=8 (255 fits) | 455.2 ms | — | — | — |
| logit n=100000, p=5 | 393.4 ms | scipy.optimize | 119.3 ms | 3.3× slower |
| poissonRegression n=100000, p=5 | 439.2 ms | scipy.optimize | 117.7 ms | 3.7× slower |
| ologit n=10000, p=3, 4 levels | 80.1 ms | — | — | — |
| mlogit n=10000, p=3, 3 classes | 57.6 ms | — | — | — |
| linearModel 'y ~ a*b + x' n=100000 | 242.8 ms | numpy | 131.8 ms | 1.8× slower |
| nls (2 params, LM) n=10000 | 86.5 ms | scipy | 5.14 ms | 17× slower |
| orthogonalRegression (jackknife SE) n=5000 | 1.35 ms | scipy | 5.50 ms | 4.1× faster |
| pls n=2000, p=10, 3 comp (LOO CV) | 936.3 ms | — | — | — |
| stabilityStudy 5 batches × 8 times | 13.6 ms | — | — | — |

## Control charts

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| controlChart I-MR (Nelson 1–8) n=1000000 | 623.7 ms | numpy | 17.9 ms | 35× slower |
| controlChart X̄-R subgroups of 5 n=1000000 | 168.6 ms | numpy | 12.2 ms | 14× slower |
| controlChart P n=100000 | 32.6 ms | numpy | 0.13 ms | 252× slower |
| ewma n=1000000 | 116.5 ms | pandas | 15.3 ms | 7.6× slower |
|  |  | polars | 8.25 ms | 14× slower |
| cusum n=1000000 | 106.8 ms | numpy | 19.6 ms | 5.4× slower |
| movingAverage span 5 n=1000000 | 103.9 ms | pandas | 33.0 ms | 3.1× slower |
|  |  | polars | 5.52 ms | 19× slower |
| gChart n=100000 | 10.2 ms | scipy | 0.11 ms | 95× slower |
| tChart (Weibull) n=100000 | 32.5 ms | scipy | 253.9 ms | 7.8× faster |
| t2Chart p=5 n=100000 | 74.1 ms | numpy | 14.4 ms | 5.1× slower |
| mewma p=3 n=100000 (h given) | 16.3 ms | numpy | 319.2 ms | 20× faster |
| mewma h calibration (ARL₀ 200, p=3) | 441.3 ms | — | — | — |
| generalizedVarianceChart p=3, 10000 subgroups × 5 | 20.1 ms | numpy | 153.9 ms | 7.7× faster |

## Capability

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| capability n=1000000 | 126.7 ms | numpy | 14.6 ms | 8.7× slower |
| capabilitySixpack n=100000 | 87.6 ms | — | — | — |
| boxCoxLambda n=100000 | 551.5 ms | scipy | 497.0 ms | 1.1× slower |
| johnsonFit n=100000 | 42.2 ms | — | — | — |
| weibullFit n=100000 | 15.0 ms | scipy | 290.1 ms | 19× faster |
| toleranceInterval n=100000 | 33.5 ms | — | — | — |

## Measurement systems

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| gageRR crossed 10 parts × 3 operators × 3 | 0.11 ms | — | — | — |
| gageLinearity 5 refs × 12 | 0.010 ms | — | — | — |
| gageType1 n=50 | 0.009 ms | — | — | — |
| attributeAgreement Fleiss κ 3 raters × 500 | 0.15 ms | — | — | — |
| acceptanceSampling n=125,c=3 OC curve | 0.011 ms | scipy | 0.054 ms | 4.9× faster |

## Quality tools

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| pareto n=1000000, 30 categories | 17.1 ms | pandas | 100.2 ms | 5.8× faster |
|  |  | polars | 14.4 ms | 1.2× slower |
| runChart n=100000 | 49.3 ms | — | — | — |
| multiVari 3 factors n=100000 | 40.4 ms | pandas | 30.1 ms | 1.3× slower |
| symmetryTest n=100000 | 5.61 ms | — | — | — |

## Time series

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| trendAnalysis quadratic n=100000 | 22.7 ms | numpy | 7.60 ms | 3.0× slower |
| decompose (12) n=100000 | 13.1 ms | — | — | — |
| stl (12) n=10000 | 3.73 s | — | — | — |
| ets winters-add n=10000 | 48.8 ms | — | — | — |
| acf 40 lags n=100000 | 13.3 ms | numpy | 0.48 ms | 27× slower |
| pacf 40 lags n=100000 | 7.86 ms | — | — | — |
| ccf ±20 lags n=100000 | 17.4 ms | numpy | 18.5 ms | 1.1× faster |
| ljungBox 20 lags n=100000 | 8.88 ms | — | — | — |
| arima (1,1,1) CSS-ML n=2000 | 5.38 ms | — | — | — |
| arima (1,1,1)(1,1,1)₁₂ n=600 | 5.64 ms | — | — | — |
| autoArima n=500 | 82.2 ms | — | — | — |
| periodogram n=20000 | 12.8 ms | scipy | 0.44 ms | 29× slower |
| cumulativePeriodogram n=20000 | 15.8 ms | — | — | — |

## DOE

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| fullFactorial 12 factors (4096 runs) | 0.39 ms | — | — | — |
| fractionalFactorial 2^(8-3) | 0.050 ms | — | — | — |
| aliasStructure 2^(10-4), order ≤ 3 | 0.80 ms | — | — | — |
| plackettBurman 23 factors | 0.11 ms | — | — | — |
| ccd 6 factors | 0.072 ms | — | — | — |
| boxBehnken 7 factors | 0.036 ms | — | — | — |
| taguchi L243 | 0.37 ms | — | — | — |
| definitiveScreening 10 factors | 0.013 ms | — | — | — |
| mixtureDesign 4 components lattice degree 3 | 0.18 ms | — | — | — |
| analyzeEffects 2^7 with interactions | 0.087 ms | — | — | — |
| analyzeDoe 2^6 × 2 replicates (linearModel) | 5.09 ms | — | — | — |
| analyzeTaguchi L27 × 3 replicates | 0.042 ms | — | — | — |
| analyzeMixture 3 components quadratic | 0.020 ms | — | — | — |
| responseOptimizer 2 responses, 3 factors | 0.29 ms | — | — | — |
| nestedAnova 3 levels n=10000 | 3.27 ms | — | — | — |

## Reliability

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| reliabilityFit Weibull censored n=10000 | 58.6 ms | scipy | 6.55 ms | 9.0× slower |
| kaplanMeier n=100000 | 89.6 ms | numpy | 9.35 ms | 9.6× slower |
| logRank 3 groups n=100000 | 76.0 ms | — | — | — |
| coxPH n=10000, p=3 | 38.7 ms | — | — | — |
| fineGray n=3000, p=2 | 881.4 ms | — | — | — |
| lifeRegression Weibull n=10000, p=2 | 49.4 ms | scipy.optimize | 47.8 ms | 1.0× slower |
| altRegression Arrhenius n=10000 | 30.6 ms | — | — | — |
| powerLawNHPP n=10000 failures | 1.33 ms | numpy | 0.040 ms | 34× slower |
| probitAnalysis 8 doses × 100 | 0.15 ms | scipy.optimize | 2.52 ms | 16× faster |
| demonstrationTestPlan × 100 | 0.068 ms | — | — | — |
| estimationTestPlan × 20 | 2.30 ms | — | — | — |
| probabilityPlot n=10000 | 3.66 ms | — | — | — |

## Multivariate

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| pca n=100000, p=10 | 44.5 ms | numpy | 9.17 ms | 4.9× slower |
| factorAnalysis ML + varimax n=10000, p=8, 2 factors | 12.9 ms | — | — | — |
| promax 20 × 3 loadings | 0.024 ms | — | — | — |
| kmeans k=4 n=100000, p=5 | 494.2 ms | — | — | — |
| hclust average n=1500, p=5 | 37.3 ms | scipy | 35.1 ms | 1.1× slower |
| clusterVariables 30 variables n=10000 | 14.7 ms | scipy | 3.00 ms | 4.9× slower |
| discriminant LDA 3 classes n=10000, p=5 | 7.25 ms | — | — | — |
| correspondence 30×30 table | 0.98 ms | numpy | 0.089 ms | 11× slower |
| multipleCorrespondence 4 variables n=5000 | 26.8 ms | numpy | 2.01 ms | 13× slower |
| itemAnalysis 20 items n=10000 | 40.1 ms | numpy | 6.05 ms | 6.6× slower |
| manova one-way 5 groups n=10000, p=4 | 3.63 ms | — | — | — |
| manovaModel 'a*b' n=10000, p=3 | 80.9 ms | numpy | 4.98 ms | 16× slower |
| mixedModel RI 100 groups n=10000 | 56.3 ms | — | — | — |
| glmm binomial PQL 50 groups n=5000 | 37.6 ms | — | — | — |

## Predictive

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| cart regression n=10000, p=5 | 293.2 ms | — | — | — |
| randomForest 30 trees n=3000, p=5 | 1.00 s | — | — | — |
| treeNet 50 trees n=3000, p=5 | 2.24 s | — | — | — |
| mars n=3000, p=5 | 1.21 s | — | — | — |
| crossValidate ols 5-fold n=10000 | 324.2 ms | — | — | — |
| autoModel (ols, cart, rf 20) 3-fold n=2000 | 2.45 s | — | — | — |

## Random data

| Function (size) | columna | Python equivalent | Python | columna vs Python |
|---|---:|---|---:|---|
| random.normal × 1000000 | 39.5 ms | numpy | 9.00 ms | 4.4× slower |
| random.gamma × 1000000 | 68.6 ms | numpy | 13.6 ms | 5.1× slower |
| random.poisson × 1000000 | 51.3 ms | numpy | 23.3 ms | 2.2× slower |

## Summary

- 172 functions timed; 110 comparisons against Python libraries: columna faster or equal in 34, slower in 76.
- Largest gaps vs the fastest Python library: controlChart.p (251.9× slower), gChart (94.5× slower), controlChart.imr (34.8× slower), powerLawNHPP (33.7× slower), periodogram (29.3× slower), signTest (27.5× slower), acf (27.5× slower), movingAverage (18.8× slower).
- Largest wins: tukeyHSD (986.5× faster), friedman (22.0× faster), mewma (19.6× faster), weibullFit (19.3× faster), probitAnalysis (16.3× faster), dist.qtukey (11.9× faster), tChart (7.8× faster), generalizedVarianceChart (7.7× faster).

Notes: all timings are single-threaded JavaScript (V8 JIT, no WebAssembly / GPU) versus compiled C / Fortran / Rust kernels on the Python side.
