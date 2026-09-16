"""Reference values for the gap-closing modules (descriptive, Poisson GOF, spectral, alias structure, life
regression / ALT / probit / NHPP / test plans, cluster variables, MCA, item analysis, promax, T² / G / T
charts, generalized variance, stability study, general MANOVA). scipy 1.14 / numpy only."""
import io, json, sys
import numpy as np
import scipy.stats as st
from scipy import signal, optimize, linalg
from scipy.cluster.hierarchy import linkage
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
rng = np.random.default_rng(8080)
out = {}
def L(a): return np.asarray(a, dtype=float).tolist()

# ---- descriptive statistics (n = 40 so that round(0.05 n) == floor) ----------------------------------------------
x = np.round(rng.normal(50, 5, 40), 2); x[3] = x[7]  # a repeated value → mode
q = lambda p: float(np.percentile(x, 100 * p, method="weibull"))
out["descriptive"] = {"x": L(x), "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "seMean": float(x.std(ddof=1) / np.sqrt(40)),
    "q1": q(0.25), "median": q(0.5), "q3": q(0.75), "skew": float(st.skew(x, bias=False)), "kurt": float(st.kurtosis(x, bias=False)),
    "trimmed": float(st.trim_mean(x, 0.05)), "mssd": float(np.sum(np.diff(x) ** 2) / (2 * 39)), "coefVar": float(100 * x.std(ddof=1) / x.mean()),
    "mode": float(x[3]), "ci_mean": [float(x.mean() - st.t.ppf(0.975, 39) * x.std(ddof=1) / np.sqrt(40)), float(x.mean() + st.t.ppf(0.975, 39) * x.std(ddof=1) / np.sqrt(40))],
    "ci_sd": [float(np.sqrt(39 * x.var(ddof=1) / st.chi2.ppf(0.975, 39))), float(np.sqrt(39 * x.var(ddof=1) / st.chi2.ppf(0.025, 39)))]}
# quantile definitions: Minitab (type 6, numpy "weibull") vs pandas / numpy default (type 7, "linear")
ps = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
out["quantiles"] = {"x": L(x), "p": ps, "minitab": [float(np.percentile(x, 100 * p, method="weibull")) for p in ps],
    "linear": [float(np.percentile(x, 100 * p, method="linear")) for p in ps]}
# boxplot
b = np.array([1, 2, 2, 3, 4, 5, 6, 7, 8, 30, 12, 9, 11, 6.5, 5.5])
q1, q3 = np.percentile(b, [25, 75], method="weibull"); iqr = q3 - q1
inside = b[(b >= q1 - 1.5 * iqr) & (b <= q3 + 1.5 * iqr)]
out["boxplot"] = {"x": L(b), "q1": float(q1), "q3": float(q3), "median": float(np.percentile(b, 50, method="weibull")), "wlo": float(inside.min()), "whi": float(inside.max()), "outliers": L(np.sort(b[(b < q1 - 1.5 * iqr) | (b > q3 + 1.5 * iqr)]))}

# ---- Poisson goodness-of-fit -------------------------------------------------------------------------------------
counts = rng.poisson(2.3, 200)
lam = counts.mean(); vals, freq = np.unique(counts, return_counts=True)
kmax = int(vals.max()); obs = np.array([int(freq[vals == k].sum()) if k in vals else 0 for k in range(kmax + 1)])
exp_ = np.array([200 * (st.poisson.pmf(k, lam) if k < kmax else st.poisson.sf(kmax - 1, lam)) for k in range(kmax + 1)])
# same pooling rule as the implementation: scan upward, accumulate while expected < 5
pooled = []; acc = None
for o, e in zip(obs, exp_):
    if acc is not None:
        acc = (acc[0] + o, acc[1] + e)
        if acc[1] >= 5: pooled.append(acc); acc = None
    elif e < 5: acc = (o, e)
    else: pooled.append((o, e))
if acc is not None:
    if pooled: pooled[-1] = (pooled[-1][0] + acc[0], pooled[-1][1] + acc[1])
    else: pooled.append(acc)
chi = sum((o - e) ** 2 / e for o, e in pooled)
out["poissonGof"] = {"counts": L(counts), "mean": float(lam), "stat": float(chi), "df": len(pooled) - 2, "p": float(st.chi2.sf(chi, len(pooled) - 2)), "categories": len(pooled)}

# ---- spectral ---------------------------------------------------------------------------------------------------------
tt = np.arange(200); sig = np.round(np.sin(2 * np.pi * tt / 12) + 0.5 * np.cos(2 * np.pi * tt / 5) + rng.normal(0, 0.5, 200), 4)
f, P = signal.periodogram(sig, fs=1.0, window="boxcar", detrend="constant", scaling="density")
out["periodogram"] = {"x": L(sig), "freq": L(f[1:]), "spectrum": L(P[1:]), "domFreq": float(f[1:][np.argmax(P[1:])])}
white = rng.normal(size=150)
Pw = np.abs(np.fft.fft(white - white.mean()))[1:76] ** 2 / 150; C = np.cumsum(Pw) / Pw.sum(); m = len(Pw); ks = np.arange(1, m + 1)
D = max(np.max(np.abs(C - ks / m)), np.max(np.abs(C - (ks - 1) / m)))
out["cumper"] = {"x": L(white), "D": float(D), "crit05": float(1.358 / np.sqrt(m - 1))}

# ---- life regression: Weibull AFT with right censoring, direct likelihood ------------------------------------------------
n = 120; xa = np.round(rng.normal(0, 1, n), 3)
sigma_true = 0.5; T = np.exp(3 + 0.7 * xa + sigma_true * np.log(rng.weibull(1, n)))  # SEV error: log(Weibull(1)) is SEV(0,1)
cens = rng.uniform(20, 120, n); obsT = np.minimum(T, cens); c = (T > cens).astype(int)
obsT = np.round(obsT, 4)
def nll(th):
    b0, b1, ls = th; s = np.exp(ls); z = (np.log(obsT) - b0 - b1 * xa) / s
    return -np.sum(np.where(c == 0, z - np.exp(z) - np.log(s), -np.exp(z)))
res = optimize.minimize(nll, [3, 0.5, np.log(0.5)], method="BFGS", options={"gtol": 1e-10})
res = optimize.minimize(nll, res.x, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
def numhess(fn, t, h=1e-4):
    k = len(t); H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ei = np.zeros(k); ej = np.zeros(k); ei[i] = h; ej[j] = h
            H[i, j] = (fn(t + ei + ej) - fn(t + ei - ej) - fn(t - ei + ej) + fn(t - ei - ej)) / (4 * h * h)
    return H
cov = np.linalg.inv(numhess(nll, res.x))
out["lifereg"] = {"t": L(obsT), "x": L(xa), "censor": L(c), "coef": L(res.x[:2]), "sigma": float(np.exp(res.x[2])), "se": L(np.sqrt(np.diag(cov))[:2]), "seLogSigma": float(np.sqrt(cov[2, 2])), "logLik": float(-res.fun),
    "p10_at0": float(np.exp(res.x[0] + np.exp(res.x[2]) * np.log(-np.log(0.9))))}
# lognormal AFT, exact only
Tl = np.round(np.exp(2 + 0.4 * xa + 0.3 * rng.normal(size=n)), 4)
def nll2(th):
    b0, b1, ls = th; s = np.exp(ls); z = (np.log(Tl) - b0 - b1 * xa) / s
    return -np.sum(st.norm.logpdf(z) - np.log(s))
r2 = optimize.minimize(nll2, [2, 0.4, np.log(0.3)], method="BFGS", options={"gtol": 1e-11})
out["lifereg_lognormal"] = {"t": L(Tl), "coef": L(r2.x[:2]), "sigma": float(np.exp(r2.x[2])), "logLik": float(-r2.fun)}

# ---- demonstration test plan (closed form, c = 0) --------------------------------------------------------------------
R, C, beta, t0, Tt = 0.9, 0.95, 1.5, 1000, 1500
qf = 1 - R ** ((Tt / t0) ** beta)
out["demo"] = {"R": R, "C": C, "beta": beta, "t0": t0, "T": Tt, "n": int(np.ceil(np.log(1 - C) / np.log(1 - qf)))}

# ---- power-law NHPP MLE (single system, time truncated) ---------------------------------------------------------------
Tend = 500.0; lam0, b0 = 0.05, 1.4
# simulate via inverse of the cumulative intensity
u = rng.uniform(size=60); Nt = np.cumsum(-np.log(rng.uniform(size=60))); times = (Nt / lam0) ** (1 / b0); times = np.round(times[times < Tend], 3)
nn = len(times); bhat = nn / np.sum(np.log(Tend / times)); lhat = nn / Tend ** bhat
lap = (np.sum(times / Tend) - nn / 2) / np.sqrt(nn / 12)
out["nhpp"] = {"times": L(times), "T": Tend, "n": nn, "beta": float(bhat), "lambda": float(lhat), "laplace": float(lap), "mil": float(2 * np.sum(np.log(Tend / times)))}

# ---- probit analysis: events/trials by dose, direct likelihood -----------------------------------------------------------
dose = np.array([1, 2, 3, 4, 5, 6.], dtype=float); tr = np.array([50, 50, 50, 50, 50, 50]); pt = st.norm.cdf(-3 + 0.9 * dose); ev = rng.binomial(tr, pt)
def nllp(bb): p_ = st.norm.cdf(bb[0] + bb[1] * dose); p_ = np.clip(p_, 1e-12, 1 - 1e-12); return float(-np.sum(ev * np.log(p_) + (tr - ev) * np.log(1 - p_)))
rp = optimize.minimize(nllp, [-2, 0.5], method="BFGS", options={"gtol": 1e-11})
out["probit"] = {"dose": L(dose), "trials": L(tr), "events": L(ev), "coef": L(rp.x), "ed50": float(-rp.x[0] / rp.x[1]), "ed90": float((st.norm.ppf(0.9) - rp.x[0]) / rp.x[1])}

# ---- cluster variables (average linkage on 1 − r) --------------------------------------------------------------------
Zc = rng.normal(size=(60, 5)); Zc[:, 1] = Zc[:, 0] + 0.3 * rng.normal(size=60); Zc[:, 3] = -Zc[:, 2] + 0.5 * rng.normal(size=60)
Zc = np.round(Zc, 3); Rm = np.corrcoef(Zc.T)
Dm = 1 - Rm; iu = np.triu_indices(5, 1)
Zl = linkage(Dm[iu], method="average")
out["clusterVars"] = {"Z": L(Zc), "heights": L(Zl[:, 2]), "R": L(Rm)}

# ---- hclust on observations: merge heights per linkage method
Ho = np.round(rng.normal(size=(40, 3)), 3)
out["hclust"] = {"X": L(Ho), **{m: L(linkage(Ho, method=m)[:, 2]) for m in ["single", "complete", "average", "ward"]}}

# ---- MCA (indicator): singular values of the standardized residual matrix ------------------------------------------------
ca = rng.choice(["a", "b", "c"], 80); cb = rng.choice(["x", "y"], 80); cc = rng.choice(["p", "q", "r", "s"], 80)
cols = []
for var in [ca, cb, cc]:
    for lv in sorted(set(var)): cols.append((var == lv).astype(float))
Zind = np.column_stack(cols); tot = Zind.sum(); Pm = Zind / tot; r_ = Pm.sum(1); c_ = Pm.sum(0)
S = (Pm - np.outer(r_, c_)) / np.sqrt(np.outer(r_, c_)); sv = np.linalg.svd(S, compute_uv=False)
out["mca"] = {"a": ca.tolist(), "b": cb.tolist(), "c": cc.tolist(), "singular": L(sv[:4]), "inertia": float(np.sum(sv ** 2))}

# ---- item analysis --------------------------------------------------------------------------------------------------------
latent = rng.normal(size=100); items = np.round(np.column_stack([latent + rng.normal(0, s_, 100) for s_ in [0.5, 0.7, 0.9, 1.2]]), 3)
k = 4; alpha = k / (k - 1) * (1 - items.var(axis=0, ddof=1).sum() / items.sum(1).var(ddof=1))
total = items.sum(1)
def alpha_del(j): sub = np.delete(items, j, axis=1); return (k - 1) / (k - 2) * (1 - sub.var(axis=0, ddof=1).sum() / sub.sum(1).var(ddof=1))
Ri = np.corrcoef(items.T); rbar = Ri[np.triu_indices(4, 1)].mean()
out["item"] = {"items": L(items), "alpha": float(alpha), "stdAlpha": float(k * rbar / (1 + (k - 1) * rbar)), "alphaDel": [float(alpha_del(j)) for j in range(4)],
    "itemTotal": [float(np.corrcoef(items[:, j], total)[0, 1]) for j in range(4)], "adjItemTotal": [float(np.corrcoef(items[:, j], total - items[:, j])[0, 1]) for j in range(4)],
    "smc": [float(1 - 1 / np.linalg.inv(Ri)[j, j]) for j in range(4)]}

# ---- promax (Hendrickson–White) on a varimax loading matrix --------------------------------------------------------------
A = np.array([[0.8, 0.1], [0.75, 0.2], [0.7, 0.15], [0.1, 0.85], [0.2, 0.8], [0.15, 0.7]])
h = np.sqrt((A ** 2).sum(1, keepdims=True)); An = A / h; target = np.sign(An) * np.abs(An) ** 4 * h
Tm = np.linalg.solve(A.T @ A, A.T @ target); d = np.sqrt(np.diag(np.linalg.inv(Tm.T @ Tm))); Tm = Tm * d
Pp = A @ Tm; Phi = np.linalg.inv(Tm) @ np.linalg.inv(Tm).T
out["promax"] = {"A": L(A), "P": L(Pp), "Phi": L(Phi)}

# ---- T² chart (individuals, phase I) & generalized variance ---------------------------------------------------------------
Xt = np.round(rng.multivariate_normal([10, 20], [[1, 0.6], [0.6, 2]], 40), 3)
mu = Xt.mean(0); Sc = np.cov(Xt.T); Si = np.linalg.inv(Sc)
t2 = np.array([(r - mu) @ Si @ (r - mu) for r in Xt])
nn2 = 40; pp = 2; a = 0.00135
ucl1 = (nn2 - 1) ** 2 / nn2 * st.beta.ppf(1 - a, pp / 2, (nn2 - pp - 1) / 2)
ucl2 = pp * (nn2 + 1) * (nn2 - 1) / (nn2 * (nn2 - pp)) * st.f.ppf(1 - a, pp, nn2 - pp)
out["t2"] = {"X": L(Xt), "t2": L(t2), "ucl1": float(ucl1), "ucl2": float(ucl2)}
# subgrouped: 10 subgroups of 4
Xs = np.round(rng.multivariate_normal([0, 0, 0], np.eye(3), 40), 3); sg = np.repeat(np.arange(10), 4)
means = np.array([Xs[sg == g].mean(0) for g in range(10)]); grand = means.mean(0)
Sp = np.mean([np.cov(Xs[sg == g].T) for g in range(10)], axis=0); Spi = np.linalg.inv(Sp)
t2s = np.array([4 * (mm - grand) @ Spi @ (mm - grand) for mm in means])
k_, m_, p_ = 10, 4, 3; df2 = k_ * m_ - k_ - p_ + 1
ucls1 = p_ * (m_ - 1) * (k_ - 1) / df2 * st.f.ppf(1 - a, p_, df2)
dets = [float(np.linalg.det(np.cov(Xs[sg == g].T))) for g in range(10)]
out["t2sub"] = {"X": L(Xs), "sg": L(sg), "t2": L(t2s), "ucl1": float(ucls1), "dets": dets, "detPooled": float(np.linalg.det(Sp))}

# ---- G chart: geometric quantiles ----------------------------------------------------------------------------------------
gc = rng.geometric(0.02, 50) - 1  # failures before success
gbar = gc.mean(); pg = 1 / (gbar + 1)
# geometric quantile (failures before first success): smallest k with 1-(1-p)^(k+1) >= q
def gq(qq): return int(np.ceil(np.log(1 - qq) / np.log(1 - pg) - 1))
out["gchart"] = {"g": L(gc), "p": float(pg), "ucl": gq(1 - 0.00135), "lcl": max(0, gq(0.00135)), "sigmaUcl": float((1 - pg) / pg + 3 * np.sqrt((1 - pg) / pg ** 2))}
# T chart: Weibull fit
tc = np.round(rng.weibull(1.8, 40) * 50, 3)
sh, loc, sc = st.weibull_min.fit(tc, floc=0)
out["tchart"] = {"t": L(tc), "shape": float(sh), "scale": float(sc), "ucl": float(sc * (-np.log(0.00135)) ** (1 / sh)), "lcl": float(sc * (-np.log(1 - 0.00135)) ** (1 / sh)), "center": float(sc * np.log(2) ** (1 / sh))}

# ---- alias structure (hand) -------------------------------------------------------------------------------------------------
out["alias"] = {"k5": {"defining": ["ABCDE"], "resolution": 5, "A": ["BCDE"]}, "k6": {"defining": sorted(["ABCE", "BCDF", "ADEF"], key=lambda w: (len(w), w)), "resolution": 4, "AB": ["CE", "ACDF", "BDEF"]}}

# ---- stability study: batch + time (parallel lines), shelf life with lower spec --------------------------------------------
tm = np.tile([0, 3, 6, 9, 12, 18, 24], 3); bt = np.repeat(["B1", "B2", "B3"], 7)
inter = {"B1": 100.5, "B2": 99.8, "B3": 100.9}
ys = np.round(np.array([inter[b_] for b_ in bt]) - 0.25 * tm + rng.normal(0, 0.3, 21), 3)
# effects coding B1, B2 (B3 = -1,-1); full model with interactions
def design(with_int, with_batch):
    colsd = [np.ones(21), tm]
    if with_batch:
        e1 = np.where(bt == "B1", 1.0, np.where(bt == "B3", -1.0, 0.0)); e2 = np.where(bt == "B2", 1.0, np.where(bt == "B3", -1.0, 0.0))
        colsd += [e1, e2]
        if with_int: colsd += [e1 * tm, e2 * tm]
    return np.column_stack(colsd)
def sse(X): b_, *_ = np.linalg.lstsq(X, ys, rcond=None); return float(np.sum((ys - X @ b_) ** 2)), X.shape[1]
sF, pF = sse(design(True, True)); sB, pB = sse(design(False, True)); sT, pT = sse(design(False, False))
F_int = ((sB - sF) / 2) / (sF / (21 - pF)); p_int = float(st.f.sf(F_int, 2, 21 - pF))
F_b = ((sT - sB) / 2) / (sB / (21 - pB)); p_b = float(st.f.sf(F_b, 2, 21 - pB))
out["stability"] = {"time": L(tm), "batch": bt.tolist(), "y": L(ys), "pInt": p_int, "pBatch": p_b}
# shelf life under the selected model (assume batch+time retained if p_b <= 0.25): one-sided 95% lower bound crossing lsl=95
Xsel = design(False, True) if p_b <= 0.25 else design(False, False)
bsel, *_ = np.linalg.lstsq(Xsel, ys, rcond=None); res_ = ys - Xsel @ bsel; dfe = 21 - Xsel.shape[1]; mse = res_ @ res_ / dfe; XtXi = np.linalg.inv(Xsel.T @ Xsel)
def lower(t, batch):
    if Xsel.shape[1] == 2: row = np.array([1, t])
    else: row = np.array([1, t, 1 if batch == "B1" else -1 if batch == "B3" else 0, 1 if batch == "B2" else -1 if batch == "B3" else 0])
    fit = row @ bsel; se = np.sqrt(row @ XtXi @ row * mse); return fit - st.t.ppf(0.95, dfe) * se
def shelf(batch):
    lo, hi = 0.0, 200.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if lower(mid, batch) < 95: hi = mid
        else: lo = mid
    return hi
out["stability"]["shelf"] = {b_: float(shelf(b_)) for b_ in ["B1", "B2", "B3"]} if Xsel.shape[1] > 2 else {"all": float(shelf("B1"))}
out["stability"]["model"] = "batch + time" if Xsel.shape[1] > 2 else "time"

# ---- general MANOVA: two-factor, Type III via reduced-model SSCP; eigen via scipy.linalg.eigh(H, E) --------------------------
fa = np.array(["L", "M", "H"] * 10); fb = np.array(["a", "b"] * 15)
Y1 = np.round(10 + np.where(fa == "H", 1.5, 0) + np.where(fb == "b", 0.8, 0) + rng.normal(0, 1, 30), 3)
Y2 = np.round(20 + np.where(fa == "L", -1.0, 0) + rng.normal(0, 1.5, 30), 3)
Ym = np.column_stack([Y1, Y2])
def code(levels, vals): return np.column_stack([np.array([1.0 if v == levels[j] else -1.0 if v == levels[-1] else 0.0 for v in vals]) for j in range(len(levels) - 1)])
Am = code(["H", "L", "M"], fa); Bm = code(["a", "b"], fb); ABm = np.column_stack([Am[:, i] * Bm[:, 0] for i in range(2)])
Xm = np.column_stack([np.ones(30), Am, Bm, ABm]); groups = {"a": [1, 2], "b": [3], "a*b": [4, 5]}
def sscp(M):
    bb, *_ = np.linalg.lstsq(M, Ym, rcond=None); Rr = Ym - M @ bb; return Rr.T @ Rr
E = sscp(Xm); dfE = 30 - 6
terms = {}
for name, cidx in groups.items():
    keepc = [c for c in range(6) if c not in cidx]; H = sscp(Xm[:, keepc]) - E
    ev = np.sort(linalg.eigh(H, E, eigvals_only=True))[::-1]
    terms[name] = {"pillai": float(np.sum(ev / (1 + ev))), "wilks": float(np.prod(1 / (1 + ev))), "hotelling": float(np.sum(ev)), "roy": float(ev[0] / (1 + ev[0])), "dfH": len(cidx)}
out["manovaModel"] = {"a": fa.tolist(), "b": fb.tolist(), "y1": L(Y1), "y2": L(Y2), "terms": terms}

txt = json.dumps(out, indent=1, default=lambda o: o.item()).replace("Infinity", "null")
Path("C:/devops/JS Pandas/packages/advanced/tests/fixtures/tier8-scipy.json").write_text(txt)
print("ok", out["stability"]["model"], out["stability"]["shelf"])
