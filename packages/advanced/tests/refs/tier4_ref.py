"""Reference values for Tier 4 (SPC, capability, MSA, quality tools).
scipy / numpy formulas matching ASTM constants, Minitab capability, Box–Cox, Weibull MLE,
tolerance k-factor (Howe), Cohen/Fleiss kappa, acceptance OC, Pareto counts.
"""
import io, json, sys, math
import numpy as np
import scipy.stats as st
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
rng = np.random.default_rng(20240914)
out = {}
def L(a): return np.asarray(a, dtype=float).tolist()

# ---- 4.1 SPC constants (ASTM table excerpt + c4 closed form) ----------------------------------------
def c4(n):
    return math.sqrt(2 / (n - 1)) * math.exp(math.lgamma(n / 2) - math.lgamma((n - 1) / 2))

D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
D3 = {2: 0.8525, 3: 0.8884, 4: 0.8798, 5: 0.8641, 6: 0.848, 7: 0.8332, 8: 0.8198, 9: 0.8078, 10: 0.7971}
consts = []
for n in range(2, 11):
    c = c4(n); d2 = D2[n]; d3 = D3[n]
    A2 = 3 / (d2 * math.sqrt(n)); A3 = 3 / (c * math.sqrt(n))
    se = math.sqrt(1 - c * c) / c
    B3 = max(0, 1 - 3 * se); B4 = 1 + 3 * se
    D3lim = max(0, 1 - 3 * d3 / d2); D4lim = 1 + 3 * d3 / d2
    consts.append({"n": n, "d2": d2, "d3": d3, "c4": c, "A2": A2, "A3": A3, "B3": B3, "B4": B4, "D3": D3lim, "D4": D4lim})
out["spcConstants"] = consts

# ---- 4.2 I-MR example -------------------------------------------------------------------------------
x = np.round(rng.normal(50, 2, 30), 3)
mr = np.abs(np.diff(x)); mr_bar = float(mr.mean()); d2_2 = 1.128
sigma = mr_bar / d2_2; center = float(x.mean())
out["imr"] = {"x": L(x), "center": center, "sigma": sigma, "ucl": center + 3 * sigma, "lcl": center - 3 * sigma,
              "mrBar": mr_bar, "mrUcl": (1 + 3 * 0.8525 / 1.128) * mr_bar, "mrLcl": max(0, 1 - 3 * 0.8525 / 1.128) * mr_bar}

# Xbar-R subgroups of 5
raw = np.round(rng.normal(10, 1.5, 50), 3)
groups = raw.reshape(10, 5)
means = groups.mean(axis=1); ranges = groups.max(axis=1) - groups.min(axis=1)
xbar = float(means.mean()); rbar = float(ranges.mean()); c5 = consts[3]  # n=5
out["xbarR"] = {"groups": L(groups), "center": xbar, "rBar": rbar,
                "ucl": xbar + c5["A2"] * rbar, "lcl": xbar - c5["A2"] * rbar,
                "rUcl": c5["D4"] * rbar, "rLcl": c5["D3"] * rbar}

# P chart
sizes = np.full(20, 50); defects = rng.binomial(50, 0.08, 20)
pbar = defects.sum() / sizes.sum()
sig = np.sqrt(pbar * (1 - pbar) / sizes)
out["pChart"] = {"counts": L(defects), "sizes": L(sizes), "center": float(pbar),
                 "ucl": L(np.minimum(1, pbar + 3 * sig)), "lcl": L(np.maximum(0, pbar - 3 * sig))}

# ---- 4.4 EWMA / CUSUM -------------------------------------------------------------------------------
x2 = np.round(rng.normal(0, 1, 40), 4)
lam, Llim = 0.2, 3.0
z = []; ucl = []; lcl = []; prev = 0.0; sigma = 1.0
for t, xi in enumerate(x2):
    zt = lam * xi + (1 - lam) * prev
    vf = (lam / (2 - lam)) * (1 - (1 - lam) ** (2 * (t + 1)))
    half = Llim * sigma * math.sqrt(vf)
    z.append(zt); ucl.append(half); lcl.append(-half); prev = zt
out["ewma"] = {"x": L(x2), "lambda": lam, "L": Llim, "z": z, "ucl": ucl, "lcl": lcl}

# ---- 4.5 Capability ---------------------------------------------------------------------------------
# known: N(0,1), LSL=-3, USL=3 → Cp = 1, Cpk ≈ 1
xc = np.round(rng.normal(0, 1, 200), 5)
mu = float(xc.mean()); s = float(xc.std(ddof=1))
lsl, usl = -3.0, 3.0
Cp = (usl - lsl) / (6 * s); Cpl = (mu - lsl) / (3 * s); Cpu = (usl - mu) / (3 * s)
Cpk = min(Cpl, Cpu)
# within with subgroup 5
g = xc[:200].reshape(40, 5); rbar = float((g.max(1) - g.min(1)).mean()); sw = rbar / 2.326
out["capability"] = {"x": L(xc), "lsl": lsl, "usl": usl, "mean": mu, "sigmaOverall": s,
                     "Pp": (usl - lsl) / (6 * s), "Ppk": Cpk, "Cp_overall_as_pp": Cp,
                     "sigmaWithin_R": sw, "Cp": (usl - lsl) / (6 * sw),
                     "Cpk": min((mu - lsl) / (3 * sw), (usl - mu) / (3 * sw)),
                     "subgroup": 5}

# ---- 4.6 Box-Cox / Weibull --------------------------------------------------------------------------
xb = np.round(rng.lognormal(0, 0.5, 80), 5)
# profile LL grid
def boxcox_ll(lam, x):
    n = len(x); lx = np.log(x)
    if abs(lam) < 1e-12: y = lx
    else: y = (x ** lam - 1) / lam
    sse = np.sum((y - y.mean()) ** 2)
    return -n / 2 * np.log(sse / n) + (lam - 1) * lx.sum()
grid = np.arange(-2, 2.001, 0.01)
lls = [boxcox_ll(lam, xb) for lam in grid]
best = float(grid[int(np.argmax(lls))])
# scipy reference
from scipy.stats import boxcox as sp_boxcox
y_sp, lam_sp = sp_boxcox(xb)
out["boxcox"] = {"x": L(xb), "lambda_grid": best, "lambda_scipy": float(lam_sp)}

# Weibull MLE via scipy
xw = st.weibull_min.rvs(c=1.5, scale=10, size=60, random_state=rng)
shape, loc, scale = st.weibull_min.fit(xw, floc=0)
out["weibull"] = {"x": L(xw), "shape": float(shape), "scale": float(scale)}

# ---- 4.7 Tolerance (Howe k) -------------------------------------------------------------------------
xt = np.round(rng.normal(100, 5, 30), 4)
n = len(xt); coverage, conf = 0.95, 0.95
z = st.norm.ppf(0.5 + coverage / 2)
chi = st.chi2.ppf(1 - conf, n - 1)
u = z * math.sqrt(1 + 1 / n); k = u * math.sqrt((n - 1) / chi)
m = float(xt.mean()); s = float(xt.std(ddof=1))
out["tolerance"] = {"x": L(xt), "coverage": coverage, "confidence": conf, "k": k,
                    "interval": [m - k * s, m + k * s]}

# ---- 4.8 Gage R&R (synthetic crossed) ---------------------------------------------------------------
# 10 parts × 3 operators × 2 replicates; σ_repeat=0.5, σ_op=0.8, σ_part=2
parts, ops, ys = [], [], []
for p in range(10):
    part_eff = rng.normal(0, 2)
    for o in range(3):
        op_eff = rng.normal(0, 0.8)
        for r in range(2):
            parts.append(f"P{p}"); ops.append(f"O{o}")
            ys.append(float(50 + part_eff + op_eff + rng.normal(0, 0.5)))
out["gage"] = {"part": parts, "operator": ops, "measurement": ys}

# ---- 4.9 Kappa --------------------------------------------------------------------------------------
# Cohen: two raters
a = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 1, 0, 2, 1, 1, 0, 2, 2, 0, 1, 1])
b = np.array([0, 1, 1, 1, 2, 1, 0, 1, 2, 2, 0, 2, 1, 0, 0, 2, 1, 0, 1, 1])
# confusion
cats = [0, 1, 2]; mat = np.zeros((3, 3))
for i, j in zip(a, b): mat[i, j] += 1
N = len(a); po = np.trace(mat) / N
pe = np.sum(mat.sum(0) * mat.sum(1)) / N ** 2
kappa = (po - pe) / (1 - pe)
out["cohen"] = {"a": a.tolist(), "b": b.tolist(), "kappa": float(kappa), "po": float(po), "pe": float(pe)}

# Fleiss: 10 subjects × 4 raters, 3 cats
fleiss = []
for i in range(10):
    true = i % 3
    row = [true if rng.random() > 0.2 else (true + 1) % 3 for _ in range(4)]
    fleiss.append(row)
# compute fleiss kappa
fleiss = np.array(fleiss); N, m = fleiss.shape; k = 3
counts = np.zeros((N, k))
for i in range(N):
    for r in fleiss[i]: counts[i, r] += 1
P = (np.sum(counts * (counts - 1), axis=1) / (m * (m - 1))).mean()
pj = counts.sum(0) / (N * m); Pe = np.sum(pj ** 2)
out["fleiss"] = {"ratings": fleiss.tolist(), "kappa": float((P - Pe) / (1 - Pe))}

# ---- 4.10 Acceptance --------------------------------------------------------------------------------
n, c = 50, 2
ps = [0, 0.01, 0.02, 0.05, 0.1, 0.2]
curve = []
for p in ps:
    Pa = float(st.binom.cdf(c, n, p)) if p > 0 else 1.0
    if p == 0: Pa = 1.0
    curve.append({"p": p, "Pa": Pa, "AOQ": Pa * p})
out["acceptance"] = {"n": n, "c": c, "curve": curve}

# ---- 4.11 Pareto / IDI ------------------------------------------------------------------------------
cats = ["A"] * 40 + ["B"] * 25 + ["C"] * 15 + ["D"] * 10 + ["E"] * 5
out["pareto"] = {"categories": cats, "counts": {"A": 40, "B": 25, "C": 15, "D": 10, "E": 5}}

# normal IDI — AD should prefer normal
xn = rng.normal(5, 2, 50)
out["idiNormal"] = L(xn)

path = Path(__file__).resolve().parents[1] / "fixtures" / "tier4-scipy.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"wrote {path}")
