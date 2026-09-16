import json, numpy as np, scipy.stats as st
from pathlib import Path
rng = np.random.default_rng(7)
cases = {}
def add(name, x):
    x = np.asarray(x, dtype=float)
    w, p = st.shapiro(x)
    ad = st.anderson(x, "norm")
    cases[name] = {"x": x.tolist(), "sw_w": float(w), "sw_p": float(p), "ad_a2": float(ad.statistic)}
add("n3", [1.2, 3.4, 2.2])
add("n5", [10.1, 9.7, 10.5, 9.9, 10.3])
add("n8_skewed", [0.2, 0.5, 0.9, 1.4, 2.6, 4.1, 7.3, 12.8])
add("n10", [2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 2.8, 4.9, 3.7, 3.1])
add("n11", rng.normal(50, 10, 11))
add("n12", rng.normal(0, 1, 12))
add("n30_normal", rng.normal(100, 15, 30))
add("n30_exp", rng.exponential(2.0, 30))
add("n50_uniform", rng.uniform(0, 1, 50))
add("n200_normal", rng.normal(0, 1, 200))
add("n200_t3", rng.standard_t(3, 200))
add("n1000_normal", rng.normal(5, 2, 1000))
add("n1000_lognormal", rng.lognormal(0, 0.5, 1000))
add("n5000_normal", rng.normal(0, 1, 5000))
# D'Agostino–Stephens AD p-value (same formula as Minitab / R nortest) for cross-checking arithmetic
def ad_p(a2, n):
    a = a2 * (1 + 0.75 / n + 2.25 / n**2)
    if a >= 0.6: return np.exp(1.2937 - 5.709 * a + 0.0186 * a * a)
    if a >= 0.34: return np.exp(0.9177 - 4.279 * a - 1.38 * a * a)
    if a >= 0.2: return 1 - np.exp(-8.318 + 42.796 * a - 59.938 * a * a)
    return 1 - np.exp(-13.436 + 101.14 * a - 223.73 * a * a)
for c in cases.values():
    c["ad_p"] = float(ad_p(c["ad_a2"], len(c["x"])))
Path("C:/devops/JS Pandas/packages/advanced/tests/fixtures/normality-scipy.json").write_text(json.dumps(cases))
print("ok", len(cases))
import json, numpy as np, scipy.stats as st
from pathlib import Path
p = Path("C:/devops/JS Pandas/packages/advanced/tests/fixtures/normality-scipy.json")
cases = json.loads(p.read_text())
for name, c in cases.items():
    x = np.asarray(c["x"], dtype=float)
    n = len(x)
    if n < 4:
        continue
    mean = x.mean(); sd = x.std(ddof=1)
    # Lilliefors D = KS statistic against N(mean, sd) estimated from the data
    c["ks_d"] = float(st.kstest(x, "norm", args=(mean, sd)).statistic)
    # Ryan–Joiner R = Pearson correlation of ordered data with normal scores Φ⁻¹((i − 3/8)/(n + 1/4))
    xs = np.sort(x)
    b = st.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    c["rj_r"] = float(st.pearsonr(xs, b)[0])
p.write_text(json.dumps(cases))
print("ok")
