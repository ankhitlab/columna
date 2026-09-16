import json, numpy as np, scipy.stats as st
from pathlib import Path
out = {}
a = [5.1, 4.9, 6.2, 5.8, 6.0, 5.5, 5.35, 6.4, 5.9, 5.05]      # all 18 values distinct → no ties
b = [4.8, 5.2, 5.0, 4.6, 5.15, 4.95, 5.3, 4.7]
def mw(x, y, alt, method):
    r = st.mannwhitneyu(x, y, alternative=alt, method=method)
    return {"U": float(r.statistic), "p": float(r.pvalue)}
out["mw_noties"] = {"a": a, "b": b,
    "exact": {alt: mw(a, b, alt, "exact") for alt in ["two-sided", "less", "greater"]},
    "asymptotic": {alt: mw(a, b, alt, "asymptotic") for alt in ["two-sided", "less", "greater"]}}
# ties (repeated values across samples)
c = [3, 5, 5, 7, 8, 8, 9, 12, 12, 15, 4]
d = [2, 5, 6, 8, 8, 10, 11, 12, 3]
out["mw_ties"] = {"a": c, "b": d, "asymptotic": {alt: mw(c, d, alt, "asymptotic") for alt in ["two-sided", "less", "greater"]}}
# Hodges–Lehmann estimate and Minitab-style CI ordering for a, b
diffs = np.sort(np.subtract.outer(np.array(a), np.array(b)).ravel())
n1, n2 = len(a), len(b); m = n1 * n2; N = n1 + n2
sd = np.sqrt(n1 * n2 * (N + 1) / 12)
kk = int(np.floor(m / 2 - st.norm.ppf(0.975) * sd))
out["hl"] = {"estimate": float(np.median(diffs)), "k": kk, "lo": float(diffs[kk - 1]), "hi": float(diffs[m - kk]),
             "achieved": float(1 - 2 * st.norm.sf((m / 2 - kk + 0.5) / sd))}
# Kruskal–Wallis with and without ties
g1 = [5.1, 4.9, 6.2, 5.8, 6.0, 5.5]; g2 = [4.8, 5.2, 5.0, 4.6, 5.15]; g3 = [6.1, 6.5, 6.3, 6.8, 6.05, 6.6, 5.9]
r = st.kruskal(g1, g2, g3)
out["kw"] = {"groups": {"g1": g1, "g2": g2, "g3": g3}, "H": float(r.statistic), "p": float(r.pvalue)}
t1 = [1, 2, 2, 3, 5, 5, 5]; t2 = [2, 3, 3, 6, 7]; t3 = [5, 5, 8, 9, 9, 10]
r = st.kruskal(t1, t2, t3)
out["kw_ties"] = {"groups": {"t1": t1, "t2": t2, "t3": t3}, "H": float(r.statistic), "p": float(r.pvalue)}
Path("C:/devops/JS Pandas/packages/advanced/tests/fixtures/nonparametric-scipy.json").write_text(json.dumps(out, indent=1))
print("ok")
