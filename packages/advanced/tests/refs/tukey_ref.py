import json, numpy as np, scipy.stats as st
from pathlib import Path
out = {}
a = [5.1, 4.9, 6.2, 5.8, 6.0, 5.5, 5.3, 6.4, 5.9, 5.0]
b = [4.8, 5.2, 5.0, 4.6, 5.1, 4.9, 5.3, 4.7]
c = [6.1, 6.5, 6.3, 6.8, 6.0, 6.6]
d = [3.0, 9.5, 5.2, 7.7, 1.9, 8.8, 4.4]  # much larger spread
groups = {"a": a, "b": b, "c": c, "d": d}
out["groups"] = groups
out["bartlett"] = dict(zip(["stat", "p"], map(float, st.bartlett(a, b, c, d))))
out["levene_median"] = dict(zip(["stat", "p"], map(float, st.levene(a, b, c, d, center="median"))))
out["levene_mean"] = dict(zip(["stat", "p"], map(float, st.levene(a, b, c, d, center="mean"))))
# F-test for two variances (no scipy function; compute directly)
va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
F = va / vb
p2 = 2 * min(st.f.cdf(F, 9, 7), st.f.sf(F, 9, 7))
out["ftest"] = {"ratio": float(F), "p": float(p2), "lo": float(F / st.f.ppf(0.975, 9, 7)), "hi": float(F / st.f.ppf(0.025, 9, 7))}
# Tukey HSD
res = st.tukey_hsd(a, b, c, d)
ci = res.confidence_interval(0.95)
names = ["a", "b", "c", "d"]
comps = []
for i in range(4):
    for j in range(i + 1, 4):
        comps.append({"a": names[i], "b": names[j], "diff": float(res.statistic[i, j]), "p": float(res.pvalue[i, j]), "lo": float(ci.low[i, j]), "hi": float(ci.high[i, j])})
out["tukey"] = comps
# studentized range cdf / ppf grid
grid = []
for k, df in [(2, 5), (3, 21), (4, 27), (5, 50), (8, 120), (10, 200), (20, 1000), (3, 10000)]:
    for q in [0.5, 1, 2, 3, 3.5, 4, 5, 7]:
        grid.append({"k": k, "df": df, "q": q, "cdf": float(st.studentized_range.cdf(q, k, df))})
    for p in [0.5, 0.9, 0.95, 0.99]:
        grid.append({"k": k, "df": df, "p": p, "ppf": float(st.studentized_range.ppf(p, k, df))})
out["srange"] = grid
Path("C:/devops/JS Pandas/packages/advanced/tests/fixtures/tukey-scipy.json").write_text(json.dumps(out, indent=1))
print("ok", len(grid))
