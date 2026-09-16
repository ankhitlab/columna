import json, numpy as np, scipy.stats as st
from pathlib import Path
a = [5.1, 4.9, 6.2, 5.8, 6.0, 5.5, 5.3, 6.4, 5.9, 5.0]
b = [4.8, 5.2, 5.0, 4.6, 5.1, 4.9, 5.3, 4.7]
c = [6.1, 6.5, 6.3, 6.8, 6.0, 6.6]
d = [3.0, 9.5, 5.2, 7.7, 1.9, 8.8, 4.4]
groups = {"a": a, "b": b, "c": c, "d": d}
out = {"groups": groups}
# Fisher LSD by hand: pooled MSW, df = N - k
arrs = [np.asarray(v, float) for v in groups.values()]
N = sum(len(x) for x in arrs); k = len(arrs); df = N - k
msw = sum(((x - x.mean()) ** 2).sum() for x in arrs) / df
names = list(groups)
tcrit = st.t.ppf(0.975, df)
fisher = []
for i in range(k):
    for j in range(i + 1, k):
        x, y = arrs[i], arrs[j]
        diff = x.mean() - y.mean(); se = np.sqrt(msw * (1 / len(x) + 1 / len(y))); t = diff / se
        fisher.append({"a": names[i], "b": names[j], "diff": float(diff), "t": float(t), "p": float(2 * st.t.sf(abs(t), df)), "lo": float(diff - tcrit * se), "hi": float(diff + tcrit * se)})
out["fisher"] = {"df": df, "msw": float(msw), "tcrit": float(tcrit), "comparisons": fisher}
# Dunnett (scipy >= 1.11), control = 'b'; scipy uses QMC → results are stochastic at ~1e-3–1e-4 level; fix the seed
rng = np.random.default_rng(12345)
def dn(alt):
    res = st.dunnett(np.asarray(a), np.asarray(c), np.asarray(d), control=np.asarray(b), alternative=alt, random_state=rng)
    ci = res.confidence_interval(0.95)
    return {"stat": [float(v) for v in res.statistic], "p": [float(v) for v in res.pvalue], "lo": [float(v) for v in ci.low], "hi": [float(v) for v in ci.high]}
out["dunnett"] = {"control": "b", "treatments": ["a", "c", "d"], "two-sided": dn("two-sided"), "greater": dn("greater"), "less": dn("less")}
# Dunnett's classical table check: equal n, k-1 = 3 treatments, df = 20, two-sided 5% critical value = 2.54 (Dunnett 1955); one-sided 2.19
out["table"] = {"k1": 3, "df": 20, "two_sided_05": 2.54, "one_sided_05": 2.19}
Path("C:/devops/JS Pandas/packages/advanced/tests/fixtures/multcomp-scipy.json").write_text(json.dumps(out, indent=1))
print("ok")
