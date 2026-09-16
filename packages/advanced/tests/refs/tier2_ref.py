"""Reference values for Tier 2 (scipy 1.14): basic tests, nonparametrics, TOST, power."""
import io, json, sys
import numpy as np
import scipy.stats as st
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
out = {}
ALTS = ["two-sided", "less", "greater"]

# ---- 2.1 one-sample z ------------------------------------------------------------------------------------
x = [5.1, 4.9, 6.2, 5.8, 6.0, 5.5, 5.35, 6.4, 5.9, 5.05, 5.6, 5.2]
sigma, mu0 = 0.6, 5.3
z = (np.mean(x) - mu0) / (sigma / np.sqrt(len(x)))
out["ztest"] = {"x": x, "sigma": sigma, "mu": mu0, "z": float(z),
    "p": {"two-sided": float(2 * st.norm.sf(abs(z))), "less": float(st.norm.cdf(z)), "greater": float(st.norm.sf(z))},
    "ci95": [float(np.mean(x) - 1.959963984540054 * sigma / np.sqrt(len(x))), float(np.mean(x) + 1.959963984540054 * sigma / np.sqrt(len(x)))]}

# ---- 2.2 proportions ---------------------------------------------------------------------------------------
def prop1(k, n, p0):
    r = {}
    for alt in ALTS:
        b = st.binomtest(k, n, p0, alternative=alt)
        ci = b.proportion_ci(0.95, method="exact")
        r[alt] = {"p": float(b.pvalue), "ci": [float(ci.low), float(ci.high)]}
    return r
out["prop1"] = [{"k": k, "n": n, "p0": p0, "exact": prop1(k, n, p0)} for k, n, p0 in [(7, 20, 0.5), (3, 25, 0.3), (18, 22, 0.6), (0, 10, 0.2), (10, 10, 0.5)]]
def fisher(e1, n1, e2, n2):
    return {alt: float(st.fisher_exact([[e1, n1 - e1], [e2, n2 - e2]], alternative=alt)[1]) for alt in ALTS}
def prop2(e1, n1, e2, n2):
    p1, p2 = e1 / n1, e2 / n2
    pb = (e1 + e2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(pb * (1 - pb) * (1 / n1 + 1 / n2))
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return {"z": float(z), "p": {"two-sided": float(2 * st.norm.sf(abs(z))), "less": float(st.norm.cdf(z)), "greater": float(st.norm.sf(z))},
            "ci95": [float(p1 - p2 - 1.959963984540054 * se), float(p1 - p2 + 1.959963984540054 * se)], "fisher": fisher(e1, n1, e2, n2)}
out["prop2"] = [{"e1": e1, "n1": n1, "e2": e2, "n2": n2, **prop2(e1, n1, e2, n2)} for e1, n1, e2, n2 in [(12, 40, 6, 35), (3, 12, 9, 14), (20, 50, 20, 50), (1, 30, 8, 28)]]

# ---- 2.3 Poisson rates ------------------------------------------------------------------------------------
def rate1(k, t, lam0):
    m = lam0 * t
    d = st.poisson(m)
    pobs = d.pmf(k)
    ks = np.arange(0, int(m + 40 * np.sqrt(m) + 60) + max(k, 0) + 1)
    two = float(min(1, d.pmf(ks)[d.pmf(ks) <= pobs * (1 + 1e-7)].sum()))
    lo = 0.0 if k == 0 else st.chi2.ppf(0.025, 2 * k) / 2
    hi = st.chi2.ppf(0.975, 2 * k + 2) / 2
    return {"k": k, "t": t, "lambda0": lam0, "p": {"two-sided": two, "less": float(d.cdf(k)), "greater": float(d.sf(k - 1))},
            "ci95": [float(lo / t), float(hi / t)]}
out["rate1"] = [rate1(k, t, l) for k, t, l in [(12, 10, 1.0), (3, 4.5, 1.5), (0, 2, 1.0), (45, 20, 2.0)]]
def rate2(k1, t1, k2, t2):
    tot = k1 + k2
    p0 = t1 / (t1 + t2)
    b = {alt: float(st.binomtest(k1, tot, p0, alternative=alt).pvalue) for alt in ALTS}
    r1, r2 = k1 / t1, k2 / t2
    pooled = tot / (t1 + t2)
    z = (r1 - r2) / np.sqrt(pooled / t1 + pooled / t2)
    return {"k1": k1, "t1": t1, "k2": k2, "t2": t2, "exact": b, "z": float(z), "pNormal": float(2 * st.norm.sf(abs(z)))}
out["rate2"] = [rate2(*a) for a in [(12, 10, 5, 8), (30, 15, 45, 15), (2, 3, 9, 4)]]

# ---- 2.4 one-sample variance (chi-square) --------------------------------------------------------------------
v = [10.2, 9.8, 11.1, 10.5, 9.4, 10.9, 10.0, 9.6, 11.4, 10.3, 9.9, 10.7, 10.1, 9.2]
s2 = float(np.var(v, ddof=1)); n = len(v); sigma0 = 0.5
stat = (n - 1) * s2 / sigma0 ** 2
out["var1"] = {"x": v, "sigma0": sigma0, "s2": s2, "stat": float(stat),
    "p": {"two-sided": float(min(1, 2 * min(st.chi2.cdf(stat, n - 1), st.chi2.sf(stat, n - 1)))), "less": float(st.chi2.cdf(stat, n - 1)), "greater": float(st.chi2.sf(stat, n - 1))},
    "ci95": [float((n - 1) * s2 / st.chi2.ppf(0.975, n - 1)), float((n - 1) * s2 / st.chi2.ppf(0.025, n - 1))]}

# ---- 2.5 correlation ------------------------------------------------------------------------------------------
a = [1.2, 2.3, 2.9, 4.1, 5.5, 5.9, 7.2, 8.1, 8.8, 10.3, 11.0, 12.4]
b = [2.1, 2.0, 3.9, 3.5, 6.2, 5.1, 7.9, 7.0, 9.8, 9.1, 12.0, 11.3]
pr = st.pearsonr(a, b)
ci = pr.confidence_interval(0.95)
out["corr"] = {"a": a, "b": b, "pearson": {"r": float(pr.statistic), "p": float(pr.pvalue), "ci95": [float(ci.low), float(ci.high)],
    "less": float(st.pearsonr(a, b, alternative="less").pvalue), "greater": float(st.pearsonr(a, b, alternative="greater").pvalue)}}
sp = st.spearmanr(a, b)
out["corr"]["spearman"] = {"r": float(sp.statistic), "p": float(sp.pvalue)}
# ties in spearman
c = [1, 2, 2, 3, 5, 5, 5, 7, 8, 9, 9, 12]
d = [2, 1, 4, 4, 5, 7, 6, 6, 9, 8, 11, 10]
sp2 = st.spearmanr(c, d)
out["corr_ties"] = {"a": c, "b": d, "r": float(sp2.statistic), "p": float(sp2.pvalue)}

# ---- 2.6 Grubbs by formula ------------------------------------------------------------------------------------
g = [199.31, 199.53, 200.19, 200.82, 201.92, 201.95, 202.18, 245.57]  # NIST Tietjen-Moore / Grubbs example
n = len(g)
G = (max(g) - np.mean(g)) / np.std(g, ddof=1)
tc = st.t.ppf(1 - 0.05 / (2 * n), n - 2)
crit = (n - 1) / np.sqrt(n) * np.sqrt(tc ** 2 / (n - 2 + tc ** 2))
tt = np.sqrt((n - 2) * n * G ** 2 / ((n - 1) ** 2 - n * G ** 2))
out["grubbs"] = {"x": g, "G": float(G), "crit": float(crit), "p": float(min(1, 2 * n * st.t.sf(tt, n - 2)))}

# ---- 2.7 sign / Wilcoxon ------------------------------------------------------------------------------------------
w = [1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30, 0.71, 2.11, 2.75, 1.04, 1.93]  # distinct |x − 1.5|?
eta = 1.5
dd = np.array(w) - eta
assert len(set(np.abs(dd))) == len(dd) and (dd != 0).all()
ab = int((dd > 0).sum()); be = int((dd < 0).sum())
out["sign"] = {"x": w, "median": eta, "above": ab, "below": be,
    "p": {alt: float(st.binomtest(ab, ab + be, 0.5, alternative=alt).pvalue) for alt in ALTS}}
out["wilcoxon_exact"] = {"x": w, "median": eta, "W": float(st.wilcoxon(dd, alternative="greater", method="exact").statistic),
    "p": {alt: float(st.wilcoxon(dd, alternative=alt, method="exact").pvalue) for alt in ALTS}}
out["wilcoxon_approx"] = {"p": {alt: float(st.wilcoxon(dd, alternative=alt, method="approx", correction=True).pvalue) for alt in ALTS}}
# ties and zeros
wt = [2.0, 3.5, 3.5, 1.0, 5.0, 2.5, 2.5, 6.0, 4.0, 4.0, 0.5, 3.0, 5.5, 3.0]
eta2 = 3.0
dt = np.array(wt) - eta2
out["wilcoxon_ties"] = {"x": wt, "median": eta2, "nNonzero": int((dt != 0).sum()),
    "p": {alt: float(st.wilcoxon(dt, alternative=alt, method="approx", correction=True, zero_method="wilcox").pvalue) for alt in ALTS}}
# Walsh-average CI (Minitab): k = floor(mu − z·sd), interval [W(k), W(M−k+1)]
vals = np.array(w); m = len(vals)
walsh = np.sort([(vals[i] + vals[j]) / 2 for i in range(m) for j in range(i, m)])
mu = m * (m + 1) / 4; sd = np.sqrt(m * (m + 1) * (2 * m + 1) / 24)
kk = int(np.floor(mu - 1.959963984540054 * sd))
out["wilcoxon_ci"] = {"estimate": float(np.median(walsh)), "k": kk, "ci": [float(walsh[kk - 1]), float(walsh[len(walsh) - kk])]}

# ---- 2.8 Mood / Friedman ------------------------------------------------------------------------------------------
g1 = [5.1, 4.9, 6.2, 5.8, 6.0, 5.5, 5.3]; g2 = [4.8, 5.2, 5.0, 4.6, 5.15, 4.95]; g3 = [6.1, 6.5, 6.3, 6.8, 6.05, 6.6, 5.9, 6.4]
stat, p, med, tbl = st.median_test(g1, g2, g3, ties="below", correction=False)
out["mood"] = {"groups": {"g1": g1, "g2": g2, "g3": g3}, "stat": float(stat), "p": float(p), "median": float(med), "table": tbl.tolist()}
stat2, p2, med2, tbl2 = st.median_test(g1, g2, ties="below", correction=False)
out["mood2"] = {"stat": float(stat2), "p": float(p2), "median": float(med2), "table": tbl2.tolist()}
table = [[3.1, 2.7, 4.0, 3.5], [2.9, 2.5, 3.8, 3.2], [3.4, 3.0, 4.2, 3.9], [2.6, 2.8, 3.6, 3.1], [3.2, 2.9, 4.1, 3.3], [3.0, 2.6, 3.9, 3.4]]
fr = st.friedmanchisquare(*np.array(table).T)
out["friedman"] = {"table": table, "stat": float(fr.statistic), "p": float(fr.pvalue)}
table_t = [[1, 2, 2, 3], [2, 2, 3, 1], [1, 1, 2, 3], [3, 2, 2, 1], [2, 3, 3, 1], [1, 2, 3, 3]]
frt = st.friedmanchisquare(*np.array(table_t).T)
out["friedman_ties"] = {"table": table_t, "stat": float(frt.statistic), "p": float(frt.pvalue)}
# runs test by formula
r = [12.1, 13.5, 11.8, 14.2, 12.9, 11.5, 13.8, 12.2, 12.6, 14.0, 11.9, 13.1, 12.4, 13.9, 11.7, 12.8, 13.3, 12.0]
k = float(np.mean(r))
above = np.array(r) > k
runs = 1 + int((above[1:] != above[:-1]).sum())
n1 = int(above.sum()); n2 = len(r) - n1; nn = len(r)
mu_r = 2 * n1 * n2 / nn + 1
var_r = 2 * n1 * n2 * (2 * n1 * n2 - nn) / (nn ** 2 * (nn - 1))
zr = (runs - mu_r) / np.sqrt(var_r)
out["runs"] = {"x": r, "k": k, "runs": runs, "n1": n1, "n2": n2, "expected": mu_r, "z": float(zr), "p": float(2 * st.norm.sf(abs(zr)))}

# ---- 2.9 TOST by formula --------------------------------------------------------------------------------------
tt1 = [49.6, 50.3, 50.1, 49.8, 50.5, 49.9, 50.2, 50.0, 49.7, 50.4]
lim = [-0.5, 0.5]
mn = np.mean(tt1); se = np.std(tt1, ddof=1) / np.sqrt(len(tt1)); df = len(tt1) - 1
tl = (mn - lim[0]) / se; tu = (mn - lim[1]) / se
out["tost1"] = {"x": tt1, "limits": lim, "mean": float(mn), "se": float(se), "tLower": float(tl), "tUpper": float(tu),
    "pLower": float(st.t.sf(tl, df)), "pUpper": float(st.t.cdf(tu, df)), "ci90": [float(mn - st.t.ppf(0.95, df) * se), float(mn + st.t.ppf(0.95, df) * se)]}
ta = [10.2, 9.8, 10.5, 10.1, 9.9, 10.4, 10.0, 9.7, 10.3, 10.6, 9.6, 10.2]
tb = [10.0, 10.3, 9.9, 10.2, 9.8, 10.1, 10.4, 9.7, 10.0, 10.5]
d2 = np.mean(ta) - np.mean(tb); va = np.var(ta, ddof=1) / len(ta); vb = np.var(tb, ddof=1) / len(tb)
se2 = np.sqrt(va + vb); dfw = (va + vb) ** 2 / (va ** 2 / (len(ta) - 1) + vb ** 2 / (len(tb) - 1))
lim2 = [-0.4, 0.4]
tl2 = (d2 - lim2[0]) / se2; tu2 = (d2 - lim2[1]) / se2
out["tost2"] = {"a": ta, "b": tb, "limits": lim2, "diff": float(d2), "se": float(se2), "df": float(dfw), "pLower": float(st.t.sf(tl2, dfw)), "pUpper": float(st.t.cdf(tu2, dfw))}

# ---- 2.10 power via noncentral distributions ------------------------------------------------------------------
def t_power(delta, df, alpha=0.05, alt="two-sided"):
    if alt == "greater":
        return float(st.nct.sf(st.t.ppf(1 - alpha, df), df, delta))
    if alt == "less":
        return float(st.nct.cdf(-st.t.ppf(1 - alpha, df), df, delta))
    c = st.t.ppf(1 - alpha / 2, df)
    return float(st.nct.sf(c, df, delta) + st.nct.cdf(-c, df, delta))
out["power"] = {
    "t1": [{"d": d, "n": n, "power": t_power(d * np.sqrt(n), n - 1)} for d, n in [(1.0, 10), (0.5, 30), (0.2, 100), (0.8, 5)]],
    "t1_greater": [{"d": d, "n": n, "power": t_power(d * np.sqrt(n), n - 1, alt="greater")} for d, n in [(0.5, 20), (0.3, 50)]],
    "t2": [{"d": d, "n": n, "power": t_power(d * np.sqrt(n / 2), 2 * n - 2)} for d, n in [(1.0, 17), (0.5, 64), (0.25, 200)]],
    "anova": [{"maxdiff": md, "k": k, "n": n, "power": float(st.ncf.sf(st.f.ppf(0.95, k - 1, k * (n - 1)), k - 1, k * (n - 1), n * md ** 2 / 2))} for md, k, n in [(1.0, 3, 10), (0.5, 4, 30), (2.0, 5, 4)]],
    "z1": [{"d": d, "n": n, "power": float(st.norm.sf(1.959963984540054 - d * np.sqrt(n)) + st.norm.cdf(-1.959963984540054 - d * np.sqrt(n)))} for d, n in [(0.5, 20), (1.0, 8)]],
    "var1": [{"ratio": r, "n": n, "power": float(st.chi2.sf(st.chi2.ppf(0.975, n - 1) / r ** 2, n - 1) + st.chi2.cdf(st.chi2.ppf(0.025, n - 1) / r ** 2, n - 1))} for r, n in [(1.5, 30), (2.0, 10)]],
    "var2": [{"ratio": r, "n": n, "power": float(st.f.sf(st.f.ppf(0.975, n - 1, n - 1) / r ** 2, n - 1, n - 1) + st.f.cdf(st.f.ppf(0.025, n - 1, n - 1) / r ** 2, n - 1, n - 1))} for r, n in [(2.0, 20), (1.5, 50)]],
}
# noncentral cdf spot checks and discrete distributions
out["nct"] = [{"x": x, "df": df, "nc": nc, "cdf": float(st.nct.cdf(x, df, nc))} for x, df, nc in [(1.5, 10, 1.0), (-0.5, 5, 0.8), (3.0, 30, 2.5), (0.0, 8, -1.2), (6.0, 3, 4.0)]]
out["ncf"] = [{"x": x, "d1": d1, "d2": d2, "nc": nc, "cdf": float(st.ncf.cdf(x, d1, d2, nc))} for x, d1, d2, nc in [(2.0, 3, 20, 5.0), (1.0, 2, 10, 1.5), (4.0, 4, 8, 12.0), (0.5, 1, 30, 0.3)]]
out["ncx2"] = [{"x": x, "k": k, "nc": nc, "cdf": float(st.ncx2.cdf(x, k, nc))} for x, k, nc in [(5.0, 3, 2.0), (12.0, 6, 8.0), (1.0, 1, 0.5)]]
out["binom"] = [{"n": n, "p": p, "k": k, "pmf": float(st.binom.pmf(k, n, p)), "cdf": float(st.binom.cdf(k, n, p)), "ppf": int(st.binom.ppf(q, n, p)), "q": q} for n, p, k, q in [(20, 0.3, 5, 0.4), (50, 0.9, 47, 0.95), (7, 0.5, 0, 0.01), (100, 0.02, 3, 0.7)]]
out["poisson"] = [{"lam": l, "k": k, "pmf": float(st.poisson.pmf(k, l)), "cdf": float(st.poisson.cdf(k, l)), "ppf": int(st.poisson.ppf(q, l)), "q": q} for l, k, q in [(3.5, 2, 0.5), (0.4, 0, 0.9), (25.0, 30, 0.99), (120.0, 100, 0.1)]]
out["hypergeom"] = [{"k": k, "N": N, "K": K, "n": n, "pmf": float(st.hypergeom.pmf(k, N, K, n))} for k, N, K, n in [(3, 20, 7, 12), (0, 15, 5, 5), (5, 15, 5, 5), (10, 100, 30, 40)]]

txt = json.dumps(out, indent=1, default=lambda o: o.item()).replace("Infinity", "null")
Path("C:/devops/JS Pandas/packages/advanced/tests/fixtures/tier2-scipy.json").write_text(txt)
print("ok")
