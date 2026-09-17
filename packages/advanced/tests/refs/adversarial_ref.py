"""Adversarial reference values (scipy 1.14 / numpy): degenerate and ill-conditioned regressions, missing
values, ties, constant samples, extreme tail probabilities, one-sided alternatives, and the specialised
reference method for each (fisher_exact, kendalltau tau-b, chi2_contingency with/without Yates, …).
Every case is one the happy-path fixtures do not exercise."""
import io, json, sys
import numpy as np
import scipy.stats as st
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
rng = np.random.default_rng(4242)
out = {}
ALTS = ["two-sided", "less", "greater"]
L = lambda a: [float(v) for v in np.asarray(a).ravel()]
nan2none = lambda v: None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else float(v)

# ---- 1. regression: exact collinearity, ill conditioning, saturation, constant predictor ----------------
n = 40
x1 = rng.normal(size=n); x2 = rng.normal(size=n); x3 = x1 + x2          # exact linear dependence
y = 1 + 2 * x1 - 3 * x2 + rng.normal(scale=0.5, size=n)
X = np.column_stack([np.ones(n), x1, x2, x3])
coef_mn, *_ = np.linalg.lstsq(X, y, rcond=None)                          # minimum-norm solution
fitted = X @ coef_mn
Xr = np.column_stack([np.ones(n), x1, x2])                                # what a rank-detecting fit keeps
coef_r, *_ = np.linalg.lstsq(Xr, y, rcond=None)
out["collinear"] = {"x1": L(x1), "x2": L(x2), "x3": L(x3), "y": L(y), "rank": int(np.linalg.matrix_rank(X)),
    "fitted": L(fitted), "sse": float(np.sum((y - fitted) ** 2)), "coef_reduced": L(coef_r)}

# ill-conditioned: polynomial columns of x ∈ [1, 2] up to x^6 (cond ~ 1e9)
xi = np.linspace(1, 2, 60)
Xi = np.column_stack([xi ** k for k in range(7)])
beta_true = np.array([1, -2, 0.5, 0.1, -0.05, 0.01, 0.002])
yi = Xi @ beta_true + rng.normal(scale=1e-3, size=60)
coef_i, *_ = np.linalg.lstsq(Xi, yi, rcond=None)
out["illcond"] = {"x": L(xi), "y": L(yi), "cond": float(np.linalg.cond(Xi)), "coef": L(coef_i), "fitted": L(Xi @ coef_i),
    "sse": float(np.sum((yi - Xi @ coef_i) ** 2))}

# saturated: p = n (intercept + 4 predictors, 5 rows) → exact fit
xs = rng.normal(size=(5, 4)); ys = rng.normal(size=5)
cs, *_ = np.linalg.lstsq(np.column_stack([np.ones(5), xs]), ys, rcond=None)
out["saturated"] = {"X": [L(r) for r in xs], "y": L(ys), "coef": L(cs)}

# constant predictor (aliased with the intercept) and constant response
xc = rng.normal(size=30); yc = 3 + 0.5 * xc + rng.normal(scale=0.2, size=30)
cc, *_ = np.linalg.lstsq(np.column_stack([np.ones(30), xc]), yc, rcond=None)
out["constant_predictor"] = {"x": L(xc), "const": [7.0] * 30, "y": L(yc), "coef": L(cc)}
out["constant_response"] = {"x": L(xc), "y": [2.5] * 30}

# ---- 2. missing values: pairwise / complete-case deletion must match scipy on the cleaned data ------------
a = rng.normal(1, 1, 25).tolist(); b = rng.normal(1.4, 1.5, 22).tolist()
a_n = a[:]; b_n = b[:]
for i in (2, 7, 19): a_n[i] = None
for i in (0, 11): b_n[i] = None
ac = [v for v in a_n if v is not None]; bc = [v for v in b_n if v is not None]
t2 = st.ttest_ind(ac, bc, equal_var=False)
out["missing_ttest2"] = {"a": a_n, "b": b_n, "statistic": float(t2.statistic), "pValue": float(t2.pvalue), "df": float(t2.df)}
px = rng.normal(size=30); py = 0.6 * px + rng.normal(size=30)
px_n = px.tolist(); py_n = py.tolist()
px_n[3] = None; py_n[8] = None; px_n[8] = None; py_n[15] = None
keep = [i for i in range(30) if px_n[i] is not None and py_n[i] is not None]
pr = st.pearsonr([px_n[i] for i in keep], [py_n[i] for i in keep])
out["missing_pearson"] = {"x": px_n, "y": py_n, "n": len(keep), "r": float(pr.statistic), "pValue": float(pr.pvalue)}

# ---- 3. ties -----------------------------------------------------------------------------------------------
ta = [1, 2, 2, 2, 3, 3, 1, 2, 3, 3, 2, 1]; tb = [2, 3, 3, 3, 3, 2, 3, 3, 2, 3]
mw = {alt: st.mannwhitneyu(ta, tb, alternative=alt, method="asymptotic", use_continuity=True) for alt in ALTS}
out["ties_mannwhitney"] = {"a": ta, "b": tb, "U": float(mw["two-sided"].statistic), "p": {alt: float(mw[alt].pvalue) for alt in ALTS}}
tw = [1.5, -1.5, 0, 2, 2, -2, 3, 0, 1.5, -3, 2, 2, 4, -1.5]
wx = {alt: st.wilcoxon(tw, alternative=alt, zero_method="wilcox", method="approx", correction=True) for alt in ALTS}
out["ties_wilcoxon"] = {"x": tw, "statistic": float(wx["two-sided"].statistic), "p": {alt: float(wx[alt].pvalue) for alt in ALTS}}
kg = [[1, 2, 2, 3, 3, 3], [2, 2, 3, 3, 4, 4], [1, 1, 2, 2, 2, 3, 3]]
kw = st.kruskal(*kg)
out["ties_kruskal"] = {"groups": kg, "H": float(kw.statistic), "pValue": float(kw.pvalue)}
sx = [1, 2, 2, 3, 4, 4, 4, 5, 6, 6]; sy = [2, 1, 3, 3, 5, 4, 6, 6, 6, 7]
sp = st.spearmanr(sx, sy); kt = st.kendalltau(sx, sy)  # tau-b handles ties on both sides
out["ties_rank_corr"] = {"x": sx, "y": sy, "spearman": float(sp.statistic), "spearman_p": float(sp.pvalue),
    "kendall_b": float(kt.statistic), "kendall_p": float(kt.pvalue)}

# ---- 4. constant samples -------------------------------------------------------------------------------------
c5 = [5.0] * 12
t1c = st.ttest_1samp(c5, 4.0); t1e = st.ttest_1samp(c5, 5.0)
out["constant"] = {"x": c5,
    "ttest1_mu4": {"statistic": nan2none(float(t1c.statistic)), "pValue": nan2none(float(t1c.pvalue))},
    "ttest1_mu5": {"statistic": nan2none(float(t1e.statistic)), "pValue": nan2none(float(t1e.pvalue))},
    "anova_with_constant_group": {"groups": [c5, [4.5, 5.5, 5.0, 4.8, 5.2, 5.1], [6, 6.5, 5.5, 6.2, 5.9]]},
    "levene_with_constant_group": {"groups": [c5[:6], [4.5, 5.5, 5.0, 4.8, 5.2, 5.1], [6, 6.5, 5.5, 6.2, 5.9, 6.1]]}}
fa = st.f_oneway(*out["constant"]["anova_with_constant_group"]["groups"])
out["constant"]["anova_with_constant_group"].update({"F": float(fa.statistic), "pValue": float(fa.pvalue)})
lv = st.levene(*out["constant"]["levene_with_constant_group"]["groups"], center="median")
out["constant"]["levene_with_constant_group"].update({"W": float(lv.statistic), "pValue": float(lv.pvalue)})

# ---- 5. extreme tail probabilities ----------------------------------------------------------------------------
out["tails"] = {
    "norm": [{"x": x, "cdf": float(st.norm.cdf(x)), "sf": float(st.norm.sf(x)), "logsf": float(st.norm.logsf(x))} for x in [-38, -30, -20, -10, 10, 20, 30, 38]],
    "norm_ppf": [{"q": q, "ppf": float(st.norm.ppf(q))} for q in [1e-300, 1e-200, 1e-100, 1e-16, 1 - 1e-16, 1 - 1e-12]],
    "t": [{"df": df, "x": x, "cdf": float(st.t.cdf(x, df)), "sf": float(st.t.sf(x, df))} for df, x in [(1, -1e5), (1, 1e8), (2, -1e4), (3, 3e3), (30, -40), (1e6, -8)]],
    "t_ppf": [{"df": df, "q": q, "ppf": float(st.t.ppf(q, df))} for df, q in [(2, 1e-12), (1, 1e-9), (5, 1 - 1e-13), (100, 1e-15)]],
    "chi2": [{"df": df, "x": x, "sf": float(st.chi2.sf(x, df)), "logsf": float(st.chi2.logsf(x, df))} for df, x in [(3, 300), (1, 600), (10, 500), (2, 1e-8), (0.5, 1e-6)]],
    "chi2_ppf": [{"df": df, "q": q, "ppf": float(st.chi2.ppf(q, df))} for df, q in [(3, 1e-14), (1, 1e-300), (50, 1 - 1e-15), (0.5, 0.5)]],
    "f": [{"d1": d1, "d2": d2, "x": x, "sf": float(st.f.sf(x, d1, d2))} for d1, d2, x in [(2, 5, 1e3), (1, 1, 1e6), (10, 200, 30), (5, 3, 1e-6)]],
    "f_ppf": [{"d1": d1, "d2": d2, "q": q, "ppf": float(st.f.ppf(q, d1, d2))} for d1, d2, q in [(3, 4, 1 - 1e-12), (1, 1, 1e-10), (20, 20, 0.999999)]],
    "binom": [{"n": n_, "p": p_, "k": k_, "pmf": float(st.binom.pmf(k_, n_, p_)), "cdf": float(st.binom.cdf(k_, n_, p_)), "sf": float(st.binom.sf(k_, n_, p_))}
              for n_, p_, k_ in [(1000, 0.999, 0), (1000, 0.999, 990), (500, 1e-6, 0), (500, 1e-6, 3), (10, 0.5, 10), (10000, 0.5, 4600)]],
    "poisson": [{"lam": lam, "k": k_, "cdf": float(st.poisson.cdf(k_, lam)), "sf": float(st.poisson.sf(k_, lam)), "pmf": float(st.poisson.pmf(k_, lam))}
                for lam, k_ in [(1e5, 99000), (1e5, 101000), (1e-8, 0), (1e-8, 2), (700, 800), (0.5, 20)]],
    "beta": [{"a": a_, "b": b_, "q": q, "ppf": float(st.beta.ppf(q, a_, b_))} for a_, b_, q in [(0.05, 0.05, 0.5), (0.05, 0.05, 1e-6), (0.5, 200, 0.999), (200, 0.5, 1e-6), (1e-3, 1, 0.5)]],
    "gamma": [{"a": a_, "x": x, "cdf": float(st.gamma.cdf(x, a_)), "sf": float(st.gamma.sf(x, a_))} for a_, x in [(1e-3, 1e-6), (1e-3, 5), (500, 400), (500, 600), (2, 1e-8)]],
}

# ---- 6. one-sided alternatives against the specialised reference -----------------------------------------------
oa = rng.normal(0.3, 1, 15).tolist(); ob = rng.normal(0, 1.2, 18).tolist()
out["onesided"] = {
    "a": oa, "b": ob,
    "ttest1": {alt: {"statistic": float(st.ttest_1samp(oa, 0, alternative=alt).statistic), "pValue": float(st.ttest_1samp(oa, 0, alternative=alt).pvalue),
                     "ci": [nan2none(float(v)) for v in st.ttest_1samp(oa, 0, alternative=alt).confidence_interval(0.95)]} for alt in ALTS},
    "ttest2_welch": {alt: {"pValue": float(st.ttest_ind(oa, ob, equal_var=False, alternative=alt).pvalue),
                           "ci": [nan2none(float(v)) for v in st.ttest_ind(oa, ob, equal_var=False, alternative=alt).confidence_interval(0.95)]} for alt in ALTS},
    "ttest2_pooled": {alt: {"pValue": float(st.ttest_ind(oa, ob, equal_var=True, alternative=alt).pvalue)} for alt in ALTS},
    "paired": {alt: {"pValue": float(st.ttest_rel(oa, ob[:15], alternative=alt).pvalue)} for alt in ALTS},
    "mannwhitney_exact": {alt: {"pValue": float(st.mannwhitneyu(oa, ob, alternative=alt, method="exact").pvalue)} for alt in ALTS},
    "wilcoxon_exact": {alt: {"pValue": float(st.wilcoxon(oa, alternative=alt, method="exact").pvalue)} for alt in ALTS},
    "pearson": {alt: {"r": float(st.pearsonr(oa, ob[:15], alternative=alt).statistic), "pValue": float(st.pearsonr(oa, ob[:15], alternative=alt).pvalue),
                      "ci": [nan2none(float(v)) for v in st.pearsonr(oa, ob[:15], alternative=alt).confidence_interval(0.95)]} for alt in ALTS},
    "prop1_exact": {f"{k}/{n_}/{alt}": {"pValue": float(st.binomtest(k, n_, 0.3, alternative=alt).pvalue),
                                        "ci": list(map(float, st.binomtest(k, n_, 0.3, alternative=alt).proportion_ci(0.95, method="exact")))}
                    for k, n_ in [(0, 20), (20, 20), (3, 20), (19, 20)] for alt in ALTS},
    "sign": {alt: {"pValue": float(st.binomtest(sum(1 for v in oa if v > 0.2), sum(1 for v in oa if v != 0.2), 0.5, alternative=alt).pvalue)} for alt in ALTS},
    "var1_chi2": {alt: {"statistic": float((len(oa) - 1) * np.var(oa, ddof=1) / 1.2 ** 2),
                        "pValue": float({"two-sided": 2 * min(st.chi2.cdf((len(oa) - 1) * np.var(oa, ddof=1) / 1.44, len(oa) - 1), st.chi2.sf((len(oa) - 1) * np.var(oa, ddof=1) / 1.44, len(oa) - 1)),
                                         "less": st.chi2.cdf((len(oa) - 1) * np.var(oa, ddof=1) / 1.44, len(oa) - 1),
                                         "greater": st.chi2.sf((len(oa) - 1) * np.var(oa, ddof=1) / 1.44, len(oa) - 1)}[alt])} for alt in ALTS},
    "poisson_rate_exact": {alt: {"pValue": float({"two-sided": min(1.0, 2 * min(st.poisson.cdf(37, 50 * 1.0), st.poisson.sf(36, 50 * 1.0))),
                                                  "less": st.poisson.cdf(37, 50.0), "greater": st.poisson.sf(36, 50.0)}[alt])} for alt in ALTS},
}

# ---- 7. specialised reference methods for contingency tables -------------------------------------------------
tables = {"extreme_diag": [[0, 10], [10, 0]], "extreme_big": [[100, 0], [0, 100]], "sparse": [[1, 0], [0, 1]], "skewed": [[50, 2], [3, 60]], "zeros_row_ok": [[0, 5], [8, 7]]}
out["tables"] = {}
for name, tbl in tables.items():
    arr = np.array(tbl)
    fe = st.fisher_exact(arr)
    rec = {"table": tbl, "fisher_two_sided": float(fe.pvalue), "fisher_less": float(st.fisher_exact(arr, alternative="less").pvalue),
           "fisher_greater": float(st.fisher_exact(arr, alternative="greater").pvalue), "odds_ratio": nan2none(float(fe.statistic))}
    try:
        c_y = st.chi2_contingency(arr, correction=True); c_n = st.chi2_contingency(arr, correction=False)
        rec.update({"chi2_yates": float(c_y.statistic), "p_yates": float(c_y.pvalue), "chi2": float(c_n.statistic), "p": float(c_n.pvalue)})
    except ValueError as e:
        rec["chi2_error"] = str(e)
    out["tables"][name] = rec

Path(__file__).resolve().parents[1].joinpath("fixtures", "adversarial-scipy.json").write_text(json.dumps(out, indent=1, allow_nan=True), encoding="utf-8")
print("ok", len(out))
