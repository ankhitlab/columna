import json, numpy as np, scipy.stats as st, scipy.special as sp
from pathlib import Path
out = {"normal": [], "t": [], "chi2": [], "f": [], "special": []}
xs_norm = [-40, -10, -6, -3, -1.96, -1, -0.5, -1e-3, 0, 1e-3, 0.5, 1, 1.96, 3, 6, 10, 40]
for mu, sd in [(0, 1), (50, 10), (-2, 0.25)]:
    for z in xs_norm:
        x = mu + sd * z
        d = st.norm(mu, sd)
        out["normal"].append({"mu": mu, "sd": sd, "x": x, "pdf": d.pdf(x), "cdf": d.cdf(x), "sf": d.sf(x)})
ps = [1e-15, 1e-10, 1e-6, 1e-3, 0.01, 0.025, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.975, 0.99, 0.999, 1 - 1e-6, 1 - 1e-10]
for mu, sd in [(0, 1), (50, 10)]:
    for p in ps:
        d = st.norm(mu, sd)
        out["normal"].append({"mu": mu, "sd": sd, "p": p, "ppf": d.ppf(p), "isf": d.isf(p)})
for df in [0.5, 1, 2, 3, 5, 10, 30, 100, 1000]:
    d = st.t(df)
    for x in [-50, -10, -4, -2.228, -1, -0.3, 0, 0.3, 1, 2.228, 4, 10, 50]:
        out["t"].append({"df": df, "x": x, "pdf": d.pdf(x), "cdf": d.cdf(x), "sf": d.sf(x)})
    for p in ps:
        out["t"].append({"df": df, "p": p, "ppf": d.ppf(p), "isf": d.isf(p)})
for k in [0.5, 1, 2, 3, 5, 10, 30, 100, 500]:
    d = st.chi2(k)
    for x in [1e-6, 0.01, 0.1, 0.5, 1, 2, 5, 11.07, 20, 50, 120, 600]:
        out["chi2"].append({"k": k, "x": x, "pdf": d.pdf(x), "cdf": d.cdf(x), "sf": d.sf(x)})
    for p in ps:
        out["chi2"].append({"k": k, "p": p, "ppf": d.ppf(p), "isf": d.isf(p)})
for d1, d2 in [(1, 1), (1, 10), (2, 5), (3, 10), (5, 5), (10, 20), (30, 30), (100, 200), (2, 1000)]:
    d = st.f(d1, d2)
    for x in [1e-6, 0.01, 0.1, 0.5, 1, 2, 3.708, 5, 10, 50, 1000]:
        out["f"].append({"d1": d1, "d2": d2, "x": x, "pdf": d.pdf(x), "cdf": d.cdf(x), "sf": d.sf(x)})
    for p in ps:
        out["f"].append({"d1": d1, "d2": d2, "p": p, "ppf": d.ppf(p), "isf": d.isf(p)})
for x in [0.1, 0.5, 1, 2.5, 10, 100, 1000.5]:
    out["special"].append({"fn": "lgamma", "x": x, "v": float(sp.gammaln(x))})
for x in [-5, -1, -0.1, 0, 0.1, 1, 3, 6, 10]:
    out["special"].append({"fn": "erf", "x": x, "v": float(sp.erf(x))})
    out["special"].append({"fn": "erfc", "x": x, "v": float(sp.erfc(x))})
Path("C:/devops/JS Pandas/packages/advanced/tests/fixtures").mkdir(exist_ok=True)
Path("C:/devops/JS Pandas/packages/advanced/tests/fixtures/dist-scipy.json").write_text(json.dumps(out))
print("written", sum(len(v) for v in out.values()), "reference values")
