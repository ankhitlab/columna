"""Reference values for Tier 5 (time series, DOE, reliability, multivariate, predictive).
numpy / scipy only — formulas match Minitab/statsmodels where noted.
"""
import io, json, sys, math
import numpy as np
import scipy.stats as st
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
rng = np.random.default_rng(20250914)
out = {}
def L(a): return np.asarray(a, dtype=float).tolist()

# ---- 5.1 Trend linear / ACF / AR(1) -----------------------------------------------------------------
n = 40
t = np.arange(1, n + 1)
y_lin = np.round(10 + 0.5 * t + rng.normal(0, 0.3, n), 4)
X = np.column_stack([np.ones(n), t])
b, *_ = np.linalg.lstsq(X, y_lin, rcond=None)
out["trend"] = {"y": L(y_lin), "coef": L(b)}

# AR(1) series φ=0.7
phi = 0.7
e = rng.normal(0, 1, 80)
ar = np.zeros(80); ar[0] = e[0]
for i in range(1, 80): ar[i] = phi * ar[i - 1] + e[i]
ar = np.round(ar, 5)
# sample acf lag 1
m = ar.mean(); c0 = np.sum((ar - m) ** 2); c1 = np.sum((ar[1:] - m) * (ar[:-1] - m))
out["ar1"] = {"y": L(ar), "phi": phi, "acf1": float(c1 / c0)}

# SES on first 40
ys = ar[:40]
alpha = 0.3
level = [ys[0]]; fitted = [ys[0]]; sse = 0.0
for i in range(1, len(ys)):
    fitted.append(level[-1]); sse += (ys[i] - level[-1]) ** 2
    level.append(alpha * ys[i] + (1 - alpha) * level[-1])
out["ses"] = {"y": L(ys), "alpha": alpha, "sse": float(sse), "fitted": fitted, "last": float(level[-1])}

# ---- 5.2 Full factorial 2^3 ------------------------------------------------------------------------
# effects A=2, B=-1, AB=0.5
factors = []
resp = []
for a in (-1, 1):
    for b in (-1, 1):
        for c in (-1, 1):
            factors.append([a, b, c])
            resp.append(5 + 2 * a - 1 * b + 0.5 * a * b + 0.1 * rng.normal())
out["doe"] = {"matrix": factors, "y": L(resp), "effects": {"A": 4.0, "B": -2.0}}  # effect = 2*coef

# ---- 5.3 Weibull with 30% right censoring ----------------------------------------------------------
shape, scale = 1.5, 100.0
raw = st.weibull_min.rvs(c=shape, scale=scale, size=60, random_state=rng)
censor_time = np.quantile(raw, 0.7)
time = np.minimum(raw, censor_time)
cens = (raw > censor_time).astype(int)
out["weibull"] = {"time": L(time), "censor": cens.tolist(), "shape": shape, "scale": scale}

# KM on exponential
te = rng.exponential(50, 40); ce = (te > 60).astype(int); te = np.minimum(te, 60)
out["km"] = {"time": L(te), "censor": ce.tolist()}

# ---- 5.4 PCA on known correlation ------------------------------------------------------------------
# X1, X2 correlated ρ=0.8; X3 independent
z = rng.normal(size=(50, 3))
X1 = z[:, 0]; X2 = 0.8 * z[:, 0] + 0.6 * z[:, 1]; X3 = z[:, 2]
out["pca"] = {"x1": L(X1), "x2": L(X2), "x3": L(X3)}

# k-means separable
c1 = rng.normal(loc=[-3, -3], scale=0.4, size=(30, 2))
c2 = rng.normal(loc=[3, 3], scale=0.4, size=(30, 2))
out["kmeans"] = {"X": L(np.vstack([c1, c2])), "labels": [0] * 30 + [1] * 30}

# LDA
out["lda"] = {"X": L(np.vstack([c1, c2])), "y": [0] * 30 + [1] * 30}

# ---- 5.5 CART / RF regression ----------------------------------------------------------------------
Xr = rng.normal(size=(80, 3))
yr = 2 * Xr[:, 0] - 1.5 * Xr[:, 1] + rng.normal(0, 0.3, 80)
out["rf"] = {"X": L(Xr), "y": L(yr)}

# ---- Deepenings: reliability SE, Taguchi S/N, ARMA CSS-ML, TreeNet clf -----------------------------
# Complete Weibull — scipy MLE (floc=0) as layer-1 anchor for SE/CI presence
shape_c, scale_c = 2.0, 50.0
w_c = st.weibull_min.rvs(c=shape_c, scale=scale_c, size=100, random_state=rng)
c_fit, _loc, s_fit = st.weibull_min.fit(w_c, floc=0)
out["weibull_complete"] = {
    "time": L(np.round(w_c, 6)),
    "shape_true": shape_c,
    "scale_true": scale_c,
    "shape_mle": float(c_fit),
    "scale_mle": float(s_fit),
}

# Exponential with left + interval censoring (for likelihood smoke / MLE)
# times: failures + left-censored + interval-censored encoded via censor codes in TS
exp_times = rng.exponential(40.0, 50)
out["exp_mixed"] = {"time": L(np.round(exp_times, 5)), "mean": 40.0}

# Taguchi L9 larger-the-better with 3 outer replicates; A dominates
L9 = [
    [1, 1, 1, 1],
    [1, 2, 2, 2],
    [1, 3, 3, 3],
    [2, 1, 2, 3],
    [2, 2, 3, 1],
    [2, 3, 1, 2],
    [3, 1, 3, 2],
    [3, 2, 1, 3],
    [3, 3, 2, 1],
]
outer = []
sn_larger = []
mean_row = []
for row in L9:
    a, b, c, d = row
    # planted mean: level of A is dominant (larger better)
    mu = 20 + 8 * (a - 1) + 1.5 * (b - 1) + 0.5 * (c - 1)
    reps = (mu + rng.normal(0, 0.8, 3)).tolist()
    outer.append([round(x, 4) for x in reps])
    mean_row.append(float(np.mean(reps)))
    s = np.mean([1.0 / (y * y) for y in reps])
    sn_larger.append(float(-10 * np.log10(s)))
# Average S/N by A level
sn_by_A = {}
for lev in (1, 2, 3):
    idx = [i for i, r in enumerate(L9) if r[0] == lev]
    sn_by_A[str(lev)] = float(np.mean([sn_larger[i] for i in idx]))
out["taguchi_l9"] = {
    "matrix": L9,
    "factors": ["A", "B", "C", "D"],
    "outer": outer,
    "sn_larger": sn_larger,
    "mean": mean_row,
    "sn_by_A": sn_by_A,
    "best_A": int(max(sn_by_A, key=lambda k: sn_by_A[k])),
}

# MA(1) θ=0.5 for CSS-ML recovery
theta = 0.5
e_ma = rng.normal(0, 1, 120)
ma = np.zeros(120)
for i in range(1, 120):
    ma[i] = e_ma[i] + theta * e_ma[i - 1]
out["ma1"] = {"y": L(np.round(ma, 5)), "theta": theta}

# Separable 2-class for TreeNet classification
Xc = np.vstack([
    rng.normal(loc=[-2, -2], scale=0.5, size=(40, 2)),
    rng.normal(loc=[2, 2], scale=0.5, size=(40, 2)),
])
yc = [0] * 40 + [1] * 40
out["treenet_clf"] = {"X": L(Xc), "y": yc}

path = Path(__file__).resolve().parents[1] / "fixtures" / "tier5-scipy.json"
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"wrote {path}")
