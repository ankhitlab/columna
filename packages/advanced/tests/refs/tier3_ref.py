"""Reference values for Tier 3 (regression & models). scipy 1.14 / numpy only: OLS by lstsq + closed
forms, logistic / Poisson / ordinal / multinomial by direct likelihood maximisation (scipy.optimize),
GLM Type III by reduced-model SSE, nonlinear by curve_fit, orthogonal by scipy.odr, PLS by NIPALS."""
import io, json, sys
import numpy as np
import scipy.stats as st
from scipy import optimize, odr
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
rng = np.random.default_rng(20240914)
out = {}
def L(a): return np.asarray(a, dtype=float).tolist()

# ---- 3.1 / 3.2 OLS with diagnostics -------------------------------------------------------------------------
n = 25
x1 = np.round(rng.normal(10, 2, n), 2); x2 = np.round(rng.normal(5, 1, n), 2); x3 = np.round(x1 * 0.5 + rng.normal(0, 1.5, n), 2)
y = np.round(3 + 1.5 * x1 - 2 * x2 + 0.3 * x3 + rng.normal(0, 1.2, n), 3)
y[7] += 6  # an outlier for the unusual-observation table
X = np.column_stack([np.ones(n), x1, x2, x3]); p = X.shape[1]
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
fit = X @ beta; e = y - fit; sse = e @ e; dfe = n - p; mse = sse / dfe
XtXi = np.linalg.inv(X.T @ X); se = np.sqrt(np.diag(XtXi) * mse); tt = beta / se
H = X @ XtXi @ X.T; h = np.diag(H)
std = e / np.sqrt(mse * (1 - h))
s2i = (sse - e ** 2 / (1 - h)) / (dfe - 1); stud = e / np.sqrt(s2i * (1 - h))
cook = std ** 2 * h / (p * (1 - h)); dffits = stud * np.sqrt(h / (1 - h))
press = np.sum((e / (1 - h)) ** 2); sst = np.sum((y - y.mean()) ** 2)
vif = []
for j in range(1, p):
    others = np.delete(X, j, axis=1); b2, *_ = np.linalg.lstsq(others, X[:, j], rcond=None)
    r2j = 1 - np.sum((X[:, j] - others @ b2) ** 2) / np.sum((X[:, j] - X[:, j].mean()) ** 2); vif.append(1 / (1 - r2j))
seq = []; prev = sst
for k in range(2, p + 1):
    bk, *_ = np.linalg.lstsq(X[:, :k], y, rcond=None); ssek = np.sum((y - X[:, :k] @ bk) ** 2); seq.append(prev - ssek); prev = ssek
xnew = np.array([1, 10.5, 5.2, 4.9]); fitn = xnew @ beta; sen = np.sqrt(xnew @ XtXi @ xnew * mse)
tc = st.t.ppf(0.975, dfe)
ll = -n / 2 * (np.log(2 * np.pi * sse / n) + 1)
out["ols"] = {"x1": L(x1), "x2": L(x2), "x3": L(x3), "y": L(y), "coef": L(beta), "se": L(se), "t": L(tt), "p": L(2 * st.t.sf(np.abs(tt), dfe)),
    "ci": [L(beta - tc * se), L(beta + tc * se)], "s": float(np.sqrt(mse)), "r2": float(1 - sse / sst), "r2adj": float(1 - mse / (sst / (n - 1))),
    "r2pred": float(1 - press / sst), "press": float(press), "ssr": float(sst - sse), "sse": float(sse), "F": float((sst - sse) / (p - 1) / mse),
    "pF": float(st.f.sf((sst - sse) / (p - 1) / mse, p - 1, dfe)), "vif": vif, "seqSS": L(seq), "adjSS": L(tt[1:] ** 2 * mse),
    "leverage": L(h), "std": L(std), "stud": L(stud), "cook": L(cook), "dffits": L(dffits), "dw": float(np.sum(np.diff(e) ** 2) / sse),
    "logLik": float(ll), "aic": float(-2 * ll + 2 * (p + 1)), "bic": float(-2 * ll + (p + 1) * np.log(n)),
    "predict": {"x": [10.5, 5.2, 4.9], "fit": float(fitn), "se": float(sen), "ci": [float(fitn - tc * sen), float(fitn + tc * sen)],
                "pi": [float(fitn - tc * np.sqrt(sen ** 2 + mse)), float(fitn + tc * np.sqrt(sen ** 2 + mse))]}}
# fitted line: quadratic
xq = np.linspace(1, 10, 15); yq = np.round(2 + 0.5 * xq - 0.08 * xq ** 2 + rng.normal(0, 0.1, 15), 4)
Xq = np.column_stack([np.ones(15), xq, xq ** 2]); bq, *_ = np.linalg.lstsq(Xq, yq, rcond=None)
out["fittedLine"] = {"x": L(xq), "y": L(yq), "coef": L(bq), "r2": float(1 - np.sum((yq - Xq @ bq) ** 2) / np.sum((yq - yq.mean()) ** 2))}

# ---- 3.3 stepwise / best subsets ------------------------------------------------------------------------------
n2 = 40
Z = np.round(rng.normal(size=(n2, 5)), 3); Z[:, 3] = np.round(Z[:, 0] * 0.8 + rng.normal(0, 0.6, n2), 3)
yz = np.round(1 + 2 * Z[:, 0] - 1.5 * Z[:, 1] + 0.7 * Z[:, 2] + rng.normal(0, 1, n2), 3)
def ssefor(cols):
    Xs = np.column_stack([np.ones(n2)] + [Z[:, j] for j in cols]); b, *_ = np.linalg.lstsq(Xs, yz, rcond=None); return float(np.sum((yz - Xs @ b) ** 2))
sstz = float(np.sum((yz - yz.mean()) ** 2)); mseFull = ssefor([0, 1, 2, 3, 4]) / (n2 - 6)
subsets = []
from itertools import combinations
for k in range(1, 6):
    rows = []
    for c in combinations(range(5), k):
        s_ = ssefor(list(c)); rows.append({"vars": [f"z{j+1}" for j in c], "r2": 1 - s_ / sstz, "cp": s_ / mseFull - (n2 - 2 * (k + 1)), "s": float(np.sqrt(s_ / (n2 - k - 1)))})
    rows.sort(key=lambda r: -r["r2"]); subsets.extend(rows[:2])
# forward stepwise by p-value (alpha 0.15)
def pvals(cols):
    Xs = np.column_stack([np.ones(n2)] + [Z[:, j] for j in cols]); b, *_ = np.linalg.lstsq(Xs, yz, rcond=None)
    r = yz - Xs @ b; mse_ = r @ r / (n2 - Xs.shape[1]); se_ = np.sqrt(np.diag(np.linalg.inv(Xs.T @ Xs)) * mse_)
    return 2 * st.t.sf(np.abs(b / se_), n2 - Xs.shape[1])
cur = []; steps = []
while True:
    best = None
    for j in range(5):
        if j in cur: continue
        pv = pvals(cur + [j])[-1]
        if best is None or pv < best[1]: best = (j, pv)
    if best is None or best[1] >= 0.15: break
    cur.append(best[0]); steps.append({"add": f"z{best[0]+1}", "p": float(best[1])})
    # removal check
    pv = pvals(cur)[1:]
    worst = int(np.argmax(pv))
    if pv[worst] > 0.15:
        removed = cur.pop(worst); steps.append({"remove": f"z{removed+1}", "p": float(pv[worst])})
out["subsets"] = {"Z": L(Z), "y": L(yz), "best": subsets, "mseFull": mseFull, "stepwise": steps, "final": [f"z{j+1}" for j in cur]}

# ---- 3.4 logistic (binary) via direct likelihood -----------------------------------------------------------------
n3 = 120
a = np.round(rng.normal(0, 1, n3), 3); b = np.round(rng.normal(0, 1, n3), 3)
eta = -0.5 + 1.2 * a - 0.8 * b; pr = 1 / (1 + np.exp(-eta)); yb = (rng.uniform(size=n3) < pr).astype(float)
Xb = np.column_stack([np.ones(n3), a, b])
def nll(bb): e_ = Xb @ bb; return float(np.sum(np.logaddexp(0, e_) - yb * e_))
def grad(bb): e_ = Xb @ bb; return Xb.T @ (1 / (1 + np.exp(-e_)) - yb)
res = optimize.minimize(nll, np.zeros(3), jac=grad, method="BFGS", options={"gtol": 1e-12})
bb = res.x; mu = 1 / (1 + np.exp(-(Xb @ bb))); Wm = mu * (1 - mu); cov = np.linalg.inv(Xb.T @ (Xb * Wm[:, None])); seb = np.sqrt(np.diag(cov))
dev = -2 * -nll(bb); p0 = yb.mean(); nulldev = -2 * np.sum(yb * np.log(p0) + (1 - yb) * np.log(1 - p0))
pearson = float(np.sum((yb - mu) ** 2 / (mu * (1 - mu))))
order = np.argsort(mu); hl = 0; groups = []
for g in range(10):
    idx = order[int(g * n3 / 10):int((g + 1) * n3 / 10)]; o = yb[idx].sum(); ex = mu[idx].sum(); m = len(idx)
    groups.append([float(o), float(ex), m]); hl += (o - ex) ** 2 / (ex * (1 - ex / m))
out["logit"] = {"a": L(a), "b": L(b), "y": L(yb), "coef": L(bb), "se": L(seb), "or": L(np.exp(bb[1:])), "deviance": float(dev), "nullDeviance": float(nulldev),
    "G": float(nulldev - dev), "pG": float(st.chi2.sf(nulldev - dev, 2)), "pearson": pearson, "hl": float(hl), "pHL": float(st.chi2.sf(hl, 8)), "logLik": float(-nll(bb)),
    "predict": {"x": [0.5, -0.3], "p": float(1 / (1 + np.exp(-(bb[0] + 0.5 * bb[1] - 0.3 * bb[2]))))}}
# events/trials form
tr = rng.integers(5, 20, 15); xa = np.round(np.linspace(-2, 2, 15), 3); pe = 1 / (1 + np.exp(-(0.3 + 1.1 * xa))); ev = rng.binomial(tr, pe)
Xe = np.column_stack([np.ones(15), xa])
def nlle(bb): e_ = Xe @ bb; return float(np.sum(tr * np.logaddexp(0, e_) - ev * e_))
def grade(bb): e_ = Xe @ bb; return Xe.T @ (tr / (1 + np.exp(-e_)) - ev)
rese = optimize.minimize(nlle, np.zeros(2), jac=grade, method="BFGS", options={"gtol": 1e-12}); be = rese.x
mue = 1 / (1 + np.exp(-(Xe @ be))); cove = np.linalg.inv(Xe.T @ (Xe * (tr * mue * (1 - mue))[:, None]))
ye = ev / tr
deve = 2 * np.sum(np.where(ev > 0, ev * np.log(ye / mue), 0) + np.where(tr - ev > 0, (tr - ev) * np.log((1 - ye) / (1 - mue)), 0))
out["logit_trials"] = {"x": L(xa), "events": L(ev), "trials": L(tr), "coef": L(be), "se": L(np.sqrt(np.diag(cove))), "deviance": float(deve),
    "pearson": float(np.sum((ev - tr * mue) ** 2 / (tr * mue * (1 - mue))))}
# probit
def nllp(bb): e_ = Xb @ bb; P = st.norm.cdf(e_); return float(-np.sum(yb * np.log(P) + (1 - yb) * np.log(1 - P)))
resp = optimize.minimize(nllp, np.zeros(3), method="BFGS", options={"gtol": 1e-10}); out["probit"] = {"coef": L(resp.x), "logLik": float(-nllp(resp.x))}

# ---- 3.5 Poisson regression ------------------------------------------------------------------------------------
n4 = 80
c1 = np.round(rng.normal(0, 1, n4), 3); c2 = np.round(rng.uniform(0, 2, n4), 3); expo = np.round(rng.uniform(0.5, 3, n4), 2)
lam = np.exp(0.2 + 0.6 * c1 - 0.4 * c2) * expo; yc = rng.poisson(lam).astype(float)
Xc = np.column_stack([np.ones(n4), c1, c2]); off = np.log(expo)
def nllc(bb): e_ = Xc @ bb + off; return float(np.sum(np.exp(e_) - yc * e_))
def gradc(bb): e_ = Xc @ bb + off; return Xc.T @ (np.exp(e_) - yc)
resc = optimize.minimize(nllc, np.zeros(3), jac=gradc, method="BFGS", options={"gtol": 1e-12}); bc = resc.x
muc = np.exp(Xc @ bc + off); covc = np.linalg.inv(Xc.T @ (Xc * muc[:, None]))
devc = 2 * np.sum(np.where(yc > 0, yc * np.log(yc / muc), 0) - (yc - muc))
# null with offset
def nll0(b0): e_ = b0[0] + off; return float(np.sum(np.exp(e_) - yc * e_))
r0 = optimize.minimize(nll0, [0.0], method="BFGS", options={"gtol": 1e-12}); mu0 = np.exp(r0.x[0] + off)
dev0 = 2 * np.sum(np.where(yc > 0, yc * np.log(yc / mu0), 0) - (yc - mu0))
out["poisson"] = {"c1": L(c1), "c2": L(c2), "exposure": L(expo), "y": L(yc), "coef": L(bc), "se": L(np.sqrt(np.diag(covc))), "deviance": float(devc), "nullDeviance": float(dev0),
    "pearson": float(np.sum((yc - muc) ** 2 / muc)), "logLik": float(np.sum(st.poisson.logpmf(yc, muc)))}

# ---- 3.4 ordinal logistic (Minitab: logit P(Y<=k) = theta_k + x'beta) ------------------------------------------
n5 = 150
o1 = np.round(rng.normal(0, 1, n5), 3); o2 = np.round(rng.normal(0, 1, n5), 3)
lin = -1.0 * o1 + 0.6 * o2; th = np.array([-1.0, 0.5])
cum = lambda k, e: 0 if k < 0 else 1 if k >= 2 else 1 / (1 + np.exp(-(th[k] + e)))
u = rng.uniform(size=n5); yo = np.zeros(n5, int)
for i in range(n5):
    c_ = [cum(k, lin[i]) for k in range(2)]; yo[i] = 0 if u[i] < c_[0] else 1 if u[i] < c_[1] else 2
Xo = np.column_stack([o1, o2])
def nllo(t):
    thr = t[:2]; bo = t[2:]; e_ = Xo @ bo; s = 0.0
    for i in range(n5):
        k = yo[i]; hi = 1.0 if k >= 2 else 1 / (1 + np.exp(-(thr[k] + e_[i]))); lo = 0.0 if k == 0 else 1 / (1 + np.exp(-(thr[k - 1] + e_[i])))
        s -= np.log(hi - lo)
    return s
reso = optimize.minimize(nllo, np.array([-1, 0.5, 0, 0]), method="BFGS", options={"gtol": 1e-9})
reso = optimize.minimize(nllo, reso.x, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
def numhess(fn, t, h=1e-4):
    k = len(t); Hm = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ei = np.zeros(k); ej = np.zeros(k); ei[i] = h; ej[j] = h
            Hm[i, j] = (fn(t + ei + ej) - fn(t + ei - ej) - fn(t - ei + ej) + fn(t - ei - ej)) / (4 * h * h)
    return Hm
covo = np.linalg.inv(numhess(nllo, reso.x))
null_o = optimize.minimize(lambda t: nllo(np.r_[t, 0, 0]), np.array([-1, 0.5]), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12})
out["ologit"] = {"x1": L(o1), "x2": L(o2), "y": L(yo), "theta": L(reso.x[:2]), "coef": L(reso.x[2:]), "se": L(np.sqrt(np.diag(covo))), "logLik": float(-reso.fun),
    "G": float(2 * (null_o.fun - reso.fun))}

# ---- 3.4 nominal logistic (reference = first level) ----------------------------------------------------------------
n6 = 150
m1 = np.round(rng.normal(0, 1, n6), 3)
Xm = np.column_stack([np.ones(n6), m1]); B = np.array([[0.3, 1.0], [-0.4, -0.8]])  # levels 1, 2 vs 0
lp = np.column_stack([np.zeros(n6), Xm @ B[0], Xm @ B[1]]); Pm = np.exp(lp) / np.exp(lp).sum(1, keepdims=True)
ym = np.array([rng.choice(3, p=Pm[i]) for i in range(n6)])
def nllm(t):
    B_ = t.reshape(2, 2); lp_ = np.column_stack([np.zeros(n6), Xm @ B_[0], Xm @ B_[1]]); lse = np.logaddexp.reduce(lp_, axis=1)
    return float(np.sum(lse - lp_[np.arange(n6), ym]))
resm = optimize.minimize(nllm, np.zeros(4), method="BFGS", options={"gtol": 1e-11})
covm = np.linalg.inv(numhess(nllm, resm.x, 1e-4))
counts = np.bincount(ym, minlength=3); null_ll = float(np.sum(counts * np.log(counts / n6)))
out["mlogit"] = {"x": L(m1), "y": L(ym), "coef": [L(resm.x[:2]), L(resm.x[2:])], "se": [L(np.sqrt(np.diag(covm))[:2]), L(np.sqrt(np.diag(covm))[2:])],
    "logLik": float(-resm.fun), "G": float(2 * (-resm.fun - null_ll))}

# ---- 3.6 GLM Type III (effects coding), unbalanced 2-factor with covariate ---------------------------------------------
fa = np.array(["L", "M", "H"] * 12)[:34]; fb = np.array(["a", "b"] * 17); cv = np.round(rng.normal(10, 2, 34), 2)
effA = {"L": -1.0, "M": 0.3, "H": 0.7}; effB = {"a": -0.5, "b": 0.5}
yg = np.round(20 + np.array([effA[v] for v in fa]) + np.array([effB[v] for v in fb]) + 0.4 * cv + np.where((fa == "H") & (fb == "b"), 1.5, 0) + rng.normal(0, 1, 34), 3)
def code(levels, vals):
    m = len(levels); cols = []
    for j in range(m - 1):
        cols.append(np.array([1.0 if v == levels[j] else -1.0 if v == levels[-1] else 0.0 for v in vals]))
    return np.column_stack(cols)
lvA = ["H", "L", "M"]; lvB = ["a", "b"]   # sorted order, as the implementation sorts levels
A_ = code(lvA, fa); B_ = code(lvB, fb); AB = np.column_stack([A_[:, i] * B_[:, j] for i in range(2) for j in range(1)])
Xg = np.column_stack([np.ones(34), A_, B_, AB, cv]); groups_ = {"a": [1, 2], "b": [3], "a*b": [4, 5], "cv": [6]}
def sse_of(M): bb_, *_ = np.linalg.lstsq(M, yg, rcond=None); return float(np.sum((yg - M @ bb_) ** 2))
sseF = sse_of(Xg); dfeF = 34 - 7; mseF = sseF / dfeF
adj = {}
for name, cols in groups_.items():
    keepc = [c for c in range(7) if c not in cols]; adj[name] = sse_of(Xg[:, keepc]) - sseF
seqg = {}; prevs = float(np.sum((yg - yg.mean()) ** 2)); used = [0]
for name, cols in groups_.items():
    used += cols; s_ = sse_of(Xg[:, used]); seqg[name] = prevs - s_; prevs = s_
bg, *_ = np.linalg.lstsq(Xg, yg, rcond=None); seg = np.sqrt(np.diag(np.linalg.inv(Xg.T @ Xg)) * mseF)
out["glm"] = {"a": fa.tolist(), "b": fb.tolist(), "cv": L(cv), "y": L(yg), "adjSS": adj, "seqSS": seqg, "sse": sseF, "coef": L(bg), "se": L(seg),
    "F": {k: v / len(groups_[k]) / mseF for k, v in adj.items()}, "p": {k: float(st.f.sf(v / len(groups_[k]) / mseF, len(groups_[k]), dfeF)) for k, v in adj.items()}}

# ---- 3.7 nonlinear: NIST Misra1a (curve_fit) ------------------------------------------------------------------------
mx = [77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4, 434.8, 477.3, 536.8, 593.1, 689.1, 760.0]
my = [10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 50.76, 55.05, 61.01, 66.40, 75.47, 81.78]
fm = lambda x, b1, b2: b1 * (1 - np.exp(-b2 * x))
popt, pcov = optimize.curve_fit(fm, mx, my, p0=[500, 1e-4], xtol=1e-14, ftol=1e-14, gtol=1e-14, maxfev=20000)
res_ = np.array(my) - fm(np.array(mx), *popt)
out["nls"] = {"x": mx, "y": my, "params": L(popt), "se": L(np.sqrt(np.diag(pcov))), "sse": float(res_ @ res_), "nist": {"b": [2.3894212918e2, 5.5015643181e-4], "se": [2.7070075241, 7.2668688436e-6], "sse": 1.2455138894e-1}}

# ---- 3.8 orthogonal regression (scipy.odr, delta = Var(ey)/Var(ex)) --------------------------------------------------
n7 = 30
xi = np.linspace(1, 20, n7); xo = np.round(xi + rng.normal(0, 0.8, n7), 3); yo2 = np.round(2 + 1.1 * xi + rng.normal(0, 0.8 * np.sqrt(2), n7), 3)
def refs_for(delta):
    data = odr.RealData(xo, yo2, sx=np.ones(n7), sy=np.full(n7, np.sqrt(delta)))
    o = odr.ODR(data, odr.Model(lambda B, x: B[0] + B[1] * x), beta0=[0, 1]).run()
    return {"delta": delta, "intercept": float(o.beta[0]), "slope": float(o.beta[1]), "sd": L(o.sd_beta)}
out["orthogonal"] = {"x": L(xo), "y": L(yo2), "fits": [refs_for(1.0), refs_for(2.0), refs_for(0.5)]}

# ---- 3.8 PLS (NIPALS in numpy) ----------------------------------------------------------------------------------------
n8 = 30
P1 = np.round(rng.normal(size=(n8, 4)), 3); P1[:, 2] = np.round(P1[:, 0] + 0.3 * rng.normal(size=n8), 3)
yp = np.round(1 + P1 @ np.array([1.0, -0.5, 0.8, 0.2]) + rng.normal(0, 0.3, n8), 3)
def nipals(X0, y0, A):
    Xc = X0 - X0.mean(0); yc = y0 - y0.mean(); W = []; Pl = []; Q = []; T = []
    for a in range(A):
        w = Xc.T @ yc; w /= np.linalg.norm(w); t = Xc @ w; q = (yc @ t) / (t @ t); pl = (Xc.T @ t) / (t @ t)
        Xc = Xc - np.outer(t, pl); yc = yc - t * q; W.append(w); Pl.append(pl); Q.append(q); T.append(t)
    W = np.array(W).T; Pl = np.array(Pl).T; Q = np.array(Q); B = W @ np.linalg.inv(Pl.T @ W) @ Q
    return B, np.array(T).T, W, Pl, Q
pls_out = {}
for A in [1, 2, 4]:
    B, T, W, Pl, Q = nipals(P1, yp, A); const = yp.mean() - P1.mean(0) @ B
    fitp = const + P1 @ B; pls_out[str(A)] = {"coef": L(B), "constant": float(const), "r2y": float(1 - np.sum((yp - fitp) ** 2) / np.sum((yp - yp.mean()) ** 2)),
        "r2x": float(1 - np.sum((P1 - P1.mean(0) - T @ Pl.T) ** 2) / np.sum((P1 - P1.mean(0)) ** 2))}
# LOO PRESS for A = 2
press2 = 0.0
for i in range(n8):
    m_ = np.ones(n8, bool); m_[i] = False; B, *_ = nipals(P1[m_], yp[m_], 2); c_ = yp[m_].mean() - P1[m_].mean(0) @ B; press2 += (yp[i] - (c_ + P1[i] @ B)) ** 2
pls_out["press2"] = float(press2); pls_out["r2pred2"] = float(1 - press2 / np.sum((yp - yp.mean()) ** 2))
out["pls"] = {"X": L(P1), "y": L(yp), **pls_out}

txt = json.dumps(out, indent=1, default=lambda o: o.item()).replace("Infinity", "null")
Path("C:/devops/JS Pandas/packages/advanced/tests/fixtures/tier3-scipy.json").write_text(txt)
print("ok", "nls", popt, "nist diff", np.abs(popt - np.array([2.3894212918e2, 5.5015643181e-4])))
