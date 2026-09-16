# Reference generators

Each `*_ref.py` regenerates one fixture in `../fixtures/*.json` from scipy 1.14 (plus direct
formula evaluation where scipy has no counterpart). Run with `py -3 <script>` from the repo root.
Fixtures are committed so the test suite does not need Python; regenerate only when the grid changes.

| script | fixture | validates |
|---|---|---|
| dist_ref.py | dist-scipy.json | normal / t / chi2 / F pdf, cdf, sf, ppf, isf; lgamma, erf |
| tests_ref.py | tests-scipy.json | t-tests, ANOVA, chi-square |
| normality_ref.py | normality-scipy.json | Shapiro–Wilk, Anderson–Darling, Ryan–Joiner R, KS D |
| tukey_ref.py | tukey-scipy.json | studentized range, Tukey HSD, Bartlett, Levene, F-test |
| multcomp_ref.py | multcomp-scipy.json | Fisher LSD, Dunnett (QMC → loose tolerances) |
| nonparametric_ref.py | nonparametric-scipy.json | Mann–Whitney, Kruskal–Wallis, Hodges–Lehmann |
| tier8_ref.py | tier8-scipy.json | descriptive stats (type-6 quartiles, adjusted skew/kurt), Poisson GOF, periodogram / Bartlett, Weibull & lognormal AFT (scipy.optimize), Crow–AMSAA, probit, cluster variables (scipy linkage), MCA (SVD), Cronbach α, promax, T² / generalized variance limits, G / T chart limits, stability shelf life, general MANOVA (eigh) |
| tier3_ref.py | tier3-scipy.json | OLS + diagnostics (numpy), best subsets / stepwise, logistic (binary, events/trials, probit), Poisson, ordinal, multinomial (scipy.optimize), GLM Type III, NLS (curve_fit + NIST Misra1a), orthogonal (scipy.odr), PLS (NIPALS) |
| tier2_ref.py | tier2-scipy.json | 1-Sample Z, proportions (binomtest, fisher_exact), Poisson rates, 1 Variance, pearsonr / spearmanr, Grubbs, sign, wilcoxon, median_test, friedmanchisquare, runs, TOST, power (nct / ncf / ncx2), binom / poisson / hypergeom |
| tier4_ref.py | tier4-scipy.json | SPC constants (ASTM + c4), I-MR / X̄-R / P limits, EWMA recursion, capability / Howe tolerance, Box–Cox / Weibull (scipy), Cohen / Fleiss κ, acceptance OC (binom), Pareto |
| tier5_ref.py | tier5-scipy.json | Trend OLS, ACF₁, SES, 2³ DOE, Weibull/KM, PCA/k-means/LDA, RF; deepenings: complete Weibull MLE, Taguchi L9 S/N, MA(1), TreeNet clf |

Bonett's tests, Hsu's MCB, Dixon's Q and the sign-test interpolated CI have no scipy reference: they are
validated by formula cross-checks, published tables and seeded Monte-Carlo calibration inside the tests
(see docs/advanced-roadmap.md, «Протокол»).
