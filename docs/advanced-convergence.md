# Numerical convergence: `columna/advanced` vs SciPy / NumPy

Generated 2026-09-15 · seed 20260915 · numpy 1.26.4, scipy 1.14.0.

Companion to [`advanced-benchmarks.md`](advanced-benchmarks.md) (throughput). Here the **same input arrays** are evaluated in columna and in SciPy/NumPy; we report max absolute and relative differences on the compared scalars/vectors.

### Verdict thresholds

| Verdict | Criterion (max abs **or** max rel error) |
|---|---|
| **agree** | ≤ 1e-10 |
| **close** | ≤ 1e-06 |
| **soft** | ≤ 0.001 |
| **diverge** | above soft |

Signed test statistics are compared on absolute values when that reduces the gap (Welch t / Mann–Whitney U sign conventions).

Regenerate: `pnpm --filter @columna/bench run advanced:convergence`.

## Summary

- Cases: **35** · agree **31** · close **2** · soft **1** · diverge **1** · error **0**
- Within close (≤1e-6): **33/35** (94%).

## Distributions

| Case | Library | Verdict | max ‖Δ‖ | max rel | Notes |
|---|---|---|---:|---:|---|
| `dist.normal.cdf` — normal.cdf (200-point grid) | scipy | **agree** | 4.44e-16 | 4.44e-16 | values: abs=4.44e-16 rel=4.44e-16 |
| `dist.normal.ppf` — normal.ppf (200-point grid) | scipy | **agree** | 2.89e-15 | 1.71e-15 | values: abs=2.89e-15 rel=1.71e-15 |
| `dist.t.cdf` — t(10).cdf | scipy | **agree** | 5.66e-15 | 5.66e-15 | values: abs=5.66e-15 rel=5.66e-15 |
| `dist.t.ppf` — t(10).ppf | scipy | **agree** | 4.34e-11 | 1.86e-11 | values: abs=4.34e-11 rel=1.86e-11 |
| `dist.chi2.sf` — chi2(5).sf | scipy | **agree** | 2.22e-15 | 2.22e-15 | values: abs=2.22e-15 rel=2.22e-15 |
| `dist.gamma.cdf` — gamma(2.5, scale=3).cdf | scipy | **agree** | 2.33e-15 | 2.33e-15 | values: abs=2.33e-15 rel=2.33e-15 |
| `dist.beta.ppf` — beta(2,5).ppf | scipy | **agree** | 2.33e-15 | 2.33e-15 | values: abs=2.33e-15 rel=2.33e-15 |
| `dist.weibull.ppf` — weibull(1.8,50).ppf | scipy | **agree** | 1.42e-14 | 2.72e-16 | values: abs=1.42e-14 rel=2.72e-16 |
| `dist.ptukey` — ptukey(q; k=4, df=20) | scipy | **agree** | 3.47e-12 | 3.47e-12 | values: abs=3.47e-12 rel=3.47e-12 |
| `dist.qtukey` — qtukey(p; k=4, df=20) | scipy | **agree** | 2.54e-11 | 7.11e-12 | values: abs=2.54e-11 rel=7.11e-12 |

## Basic statistics

| Case | Library | Verdict | max ‖Δ‖ | max rel | Notes |
|---|---|---|---:|---:|---|
| `ttest1` — one-sample t (n=500) | scipy | **agree** | 1.13e-13 | 1.13e-13 | statistic: abs=1.13e-13 rel=1.13e-13; pValue: abs=4.36e-14 rel=4.36e-14; esti… |
| `ttest2` — Welch two-sample t (n=400+400) | scipy | **agree** | 1.87e-15 | 1.87e-15 | statistic: abs=0.00e+00 rel=0.00e+00; pValue: abs=1.87e-15 rel=1.87e-15; esti… |
| `ztest1` — one-sample z (known σ) | numpy | **agree** | 1.11e-13 | 1.11e-13 | statistic: abs=1.11e-13 rel=1.11e-13; pValue: abs=5.66e-14 rel=5.66e-14 |
| `propTest1` — binomial exact 350/1000 vs 0.3 | scipy | **agree** | 1.70e-16 | 1.70e-16 | pValue: abs=1.70e-16 rel=1.70e-16; estimate: abs=0.00e+00 rel=0.00e+00 |
| `propTest2.fisher` — Fisher exact 2×2 | scipy | **agree** | 1.42e-14 | 1.42e-14 | pValue: abs=1.42e-14 rel=1.42e-14 |
| `corrTest.pearson` — Pearson r (n=300) | scipy | **agree** | 7.11e-15 | 6.33e-16 | estimate: abs=2.22e-16 rel=2.22e-16; statistic: abs=7.11e-15 rel=6.33e-16; pV… |
| `corrTest.spearman` — Spearman ρ (n=300) | scipy | **agree** | 1.11e-16 | 1.11e-16 | estimate: abs=1.11e-16 rel=1.11e-16; pValue: abs=2.04e-34 rel=2.04e-34 |
| `corrTest.kendall` — Kendall τ-b MVP | scipy | **MVP** | — | — | tau-b + normal approx; see `tier11-scipy1` |
| `partialCorr` — Pearson residuals | scipy | **MVP** | — | — | residual OLS partial r |
| `jarqueBera` / `dagostinoK2` / `cramerVonMises` | scipy | **MVP** | — | — | hooked into `normalityTest` |
| `fligner` / `ansariBradley` / `brunnerMunzel` | scipy | **MVP** | — | — | scale / stochastic equality |
| `mcnemar` / `cochranQ` / `bowker` | scipy | **MVP** | — | — | paired categorical tables |
| `cohensD` / `hedgesG` / `glassDelta` | scipy | **MVP** | — | — | standardized effect sizes |
| `gaussianKde` / `bootstrap` / `permutationTest` | scipy | **MVP** | — | — | Scott/Silverman KDE; percentile CI |
| `negativeBinomial` / `hypergeometric` / `gumbel` / `pareto` / `invgauss` | scipy | **MVP** | — | — | full Discrete/continuous Distribution API |
| `varTest1` — χ² variance test | scipy | **agree** | 5.68e-14 | 3.09e-15 | statistic: abs=5.68e-14 rel=3.61e-16; pValue: abs=3.09e-15 rel=3.09e-15 |

## ANOVA

| Case | Library | Verdict | max ‖Δ‖ | max rel | Notes |
|---|---|---|---:|---:|---|
| `anova` — one-way ANOVA 4×80 | scipy | **agree** | 1.78e-14 | 5.57e-16 | statistic: abs=1.78e-14 rel=5.57e-16; pValue: abs=5.12e-31 rel=5.12e-31 |
| `levene` — Levene | scipy | **agree** | 4.51e-15 | 4.51e-15 | statistic: abs=4.44e-16 rel=1.89e-16; pValue: abs=4.51e-15 rel=4.51e-15 |
| `bartlett` — Bartlett | scipy | **agree** | 1.42e-14 | 1.81e-15 | statistic: abs=1.42e-14 rel=1.81e-15; pValue: abs=3.19e-16 rel=3.19e-16 |

## Tables

| Case | Library | Verdict | max ‖Δ‖ | max rel | Notes |
|---|---|---|---:|---:|---|
| `chi2test` — χ² contingency 5×5 | scipy | **agree** | 2.49e-18 | 2.49e-18 | statistic: abs=0.00e+00 rel=0.00e+00; pValue: abs=2.49e-18 rel=2.49e-18 |
| `chi2gof` — χ² GOF (equal expected) | scipy | **agree** | 7.11e-15 | 2.16e-16 | statistic: abs=7.11e-15 rel=2.16e-16; pValue: abs=2.17e-19 rel=2.17e-19 |

## Normality

| Case | Library | Verdict | max ‖Δ‖ | max rel | Notes |
|---|---|---|---:|---:|---|
| `andersonDarling` — Anderson–Darling A² (n=500) | scipy | **agree** | 0.00e+00 | 0.00e+00 | Compare A²; p-value tables may differ from scipy |
| `shapiroWilk` — Shapiro–Wilk (n=100) | scipy | **close** | 8.34e-10 | 8.34e-10 | statistic: abs=1.33e-11 rel=1.33e-11; pValue: abs=8.34e-10 rel=8.34e-10 |

## Nonparametrics

| Case | Library | Verdict | max ‖Δ‖ | max rel | Notes |
|---|---|---|---:|---:|---|
| `mannWhitney` — Mann–Whitney U | scipy | **agree** | 7.49e-16 | 7.49e-16 | statistic: abs=0.00e+00 rel=0.00e+00; pValue: abs=7.49e-16 rel=7.49e-16 |
| `wilcoxonSigned` — Wilcoxon signed-rank | scipy | **soft** | 7.93e-04 | 7.93e-04 | columna exposes W+; comparison uses min(W+,W−) to match scipy |
| `kruskal` — Kruskal–Wallis | scipy | **agree** | 4.34e-19 | 4.34e-19 | statistic: abs=0.00e+00 rel=0.00e+00; pValue: abs=4.34e-19 rel=4.34e-19 |

## Regression

| Case | Library | Verdict | max ‖Δ‖ | max rel | Notes |
|---|---|---|---:|---:|---|
| `ols` — OLS + intercept (n=200, p=2) | numpy | **agree** | 1.55e-15 | 1.06e-15 | coefficients: abs=1.55e-15 rel=1.06e-15; r2: abs=1.11e-16 rel=1.11e-16; s: ab… |
| `glm.binomial` — GLM logit (n=150) | scipy | **close** | 3.25e-07 | 3.25e-07 | Python: scipy.optimize BFGS on Bernoulli log-likelihood |

## Time series

| Case | Library | Verdict | max ‖Δ‖ | max rel | Notes |
|---|---|---|---:|---:|---|
| `acf` — ACF lags 0…5 | numpy | **agree** | 5.55e-17 | 5.55e-17 | acf: abs=5.55e-17 rel=5.55e-17 |
| `arima.ar1` — ARIMA(1,0,0) φ | numpy | **diverge** | 1.42e-01 | 1.42e-01 | numpy reference = OLS y_t ~ y_{t-1}; columna uses CSS-ML |

## Multivariate

| Case | Library | Verdict | max ‖Δ‖ | max rel | Notes |
|---|---|---|---:|---:|---|
| `pca` — PCA eigenvalues (correlation) | numpy | **agree** | 6.66e-16 | 4.41e-16 | eigenvalues: abs=6.66e-16 rel=4.41e-16 |

## Reliability

| Case | Library | Verdict | max ‖Δ‖ | max rel | Notes |
|---|---|---|---:|---:|---|
| `kaplanMeier` — Kaplan–Meier S(t) | numpy | **agree** | 0.00e+00 | 0.00e+00 | numpy: product-limit formula (no lifelines) |

## Linear algebra

| Case | Library | Verdict | max ‖Δ‖ | max rel | Notes |
|---|---|---|---:|---:|---|
| `lstsq` — 3×3 lstsq | numpy | **agree** | 4.44e-16 | 4.44e-16 | coef: abs=4.44e-16 rel=4.44e-16 |

## SciPy parity Wave 2–3 (MVP)

| Case | Library | Verdict | Notes |
|---|---|---|---|
| `anovaTwoWay` / `ancova` | scipy | **MVP** | Type III via `linearModel` |
| `adfTest` / `kpssTest` | statsmodels | **MVP** | unit-root lite |
| `stl({ periods })` | statsmodels | **MVP** | multi-season sequential peel |
| `arima` ML + transfer | statsmodels | **MVP** | Kalman mean = μ + TF |
| `glmm` AGQ correlated ρ | — | **MVP** | bivariate GH |
| `eppsSingleton` / `moodTwoSample` | scipy | **MVP** | Wave 2 extras |
| `ksTwoSample` / `andersonKSample` / `energyDistance` | scipy | **MVP** | Wave 3 GOF |
| `lowess` / `isotonicRegression` | statsmodels | **MVP** | smoothing |
| `ridge` / `lasso` / `elasticNet` / `quantileRegression` | sklearn | **MVP** | penalized / QR |
| `dbscan` / `pdist` / `nnls` / `eigh` | scipy | **MVP** | cluster + linalg |
| `welchPsd` / `savitzkyGolay` / `brentq` / `trapz` | scipy | **MVP** | signal + numerics |

## Soft / diverge / error detail

### `wilcoxonSigned` (soft)

- statistic: abs=0.00e+00 rel=0.00e+00; pValue: abs=7.93e-04 rel=7.93e-04
- Note: columna exposes W+; comparison uses min(W+,W−) to match scipy

### `arima.ar1` (diverge)

- ar: abs=3.77e-07 rel=3.77e-07; intercept: abs=1.42e-01 rel=1.42e-01
- Note: numpy reference = OLS y_t ~ y_{t-1}; columna uses CSS-ML

## Method notes

- Distributions: SciPy `scipy.stats` CDF/PPF/SF on the same grid as columna.
- Hypothesis tests: SciPy defaults closest to columna (Welch `ttest_ind`, `binomtest`, `fisher_exact`, asymptotic Mann–Whitney).
- OLS / ACF / AR(1) / PCA / KM / lstsq: NumPy reference formulas.
- GLM logit: SciPy BFGS on Bernoulli NLL (not statsmodels); expect soft agreement.
- ARIMA CSS-ML vs OLS AR(1): intentional estimator difference — soft/diverge is informative, not a bug by itself.
