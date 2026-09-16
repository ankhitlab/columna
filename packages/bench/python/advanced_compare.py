"""Python side of the advanced (Minitab-parity) benchmark: times the closest scipy / numpy / pandas /
polars equivalent of each columna/advanced function, on data of the same size and distribution as
src/advanced-bench.ts (ids must match). Writes results/advanced-python.json.

    py -3 packages/bench/python/advanced_compare.py
    BENCH_SCALE=0.1 py -3 packages/bench/python/advanced_compare.py
"""
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import cluster, linalg, odr, optimize, signal

try:
    import polars as pl
except Exception:  # pragma: no cover
    pl = None

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "advanced-python.json"
SCALE = float(os.environ.get("BENCH_SCALE", "1"))
REPEAT = int(os.environ.get("BENCH_REPEAT", "5"))
ONLY = os.environ.get("BENCH_ONLY")
rng = np.random.default_rng(12345)


def N(base: int) -> int:
    return max(50, int(round(base * SCALE)))


rows = []


def bench(id_, lib, setup, repeat=None):
    """setup() builds inputs and returns the timed closure."""
    if ONLY and ONLY not in id_:
        return
    try:
        fn = setup()
    except Exception as e:  # noqa: BLE001
        print(f"{id_:50s} [{lib}] setup failed: {e}")
        return
    try:
        fn()
        times = []
        for _ in range(repeat or REPEAT):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        med = times[len(times) // 2]
        rows.append({"id": id_, "lib": lib, "msMedian": med, "msMin": times[0]})
        print(f"{id_:50s} [{lib:8s}] {med:12.2f} ms")
    except Exception as e:  # noqa: BLE001
        print(f"{id_:50s} [{lib}] failed: {e}")


def levels(n, k, prefix="L"):
    return np.array([f"{prefix}{i % k}" for i in range(n)])


# ---- distributions ---------------------------------------------------------------------------------------
n = N(1_000_000)
m = N(100_000)
bench("dist.normal.cdf", "scipy", lambda: (lambda x=rng.normal(size=n): lambda: st.norm.cdf(x))())
bench("dist.normal.ppf", "scipy", lambda: (lambda x=rng.uniform(size=n): lambda: st.norm.ppf(x))())
bench("dist.t.cdf", "scipy", lambda: (lambda x=rng.normal(size=m): lambda: st.t.cdf(x, 10))())
bench("dist.t.ppf", "scipy", lambda: (lambda x=rng.uniform(size=m): lambda: st.t.ppf(x, 10))())
bench("dist.chi2.sf", "scipy", lambda: (lambda x=rng.chisquare(5, m): lambda: st.chi2.sf(x, 5))())
bench("dist.f.ppf", "scipy", lambda: (lambda x=rng.uniform(size=m): lambda: st.f.ppf(x, 3, 20))())
bench("dist.gamma.cdf", "scipy", lambda: (lambda x=rng.gamma(2.5, 3, m): lambda: st.gamma.cdf(x, 2.5, scale=3))())
bench("dist.beta.ppf", "scipy", lambda: (lambda x=rng.uniform(size=N(10_000)): lambda: st.beta.ppf(x, 2, 5))())
bench("dist.weibull.ppf", "scipy", lambda: (lambda x=rng.uniform(size=n): lambda: st.weibull_min.ppf(x, 1.8, scale=50))())
bench("dist.binomial.cdf", "scipy", lambda: (lambda k=rng.binomial(50, 0.3, m): lambda: st.binom.cdf(k, 50, 0.3).sum())())
bench("dist.poisson.pmf", "scipy", lambda: (lambda k=rng.poisson(4, m): lambda: st.poisson.pmf(k, 4).sum())())
bench("dist.nct.cdf", "scipy", lambda: (lambda x=rng.normal(1, 1, N(10_000)): lambda: st.nct.cdf(x, 12, 0.8).sum())())
bench("dist.ptukey", "scipy", lambda: (lambda x=rng.uniform(1, 5, N(1000)): lambda: st.studentized_range.cdf(x, 4, 20).sum())())
bench("dist.qtukey", "scipy", lambda: (lambda x=rng.uniform(0.5, 0.99, N(200)): lambda: st.studentized_range.ppf(x, 4, 20).sum())(), repeat=2)

# ---- basic tests --------------------------------------------------------------------------------------------
n = N(100_000)
bench("ttest1", "scipy", lambda: (lambda x=rng.normal(50, 10, n): lambda: st.ttest_1samp(x, 50))())
bench("ttest2", "scipy", lambda: (lambda a=rng.normal(size=n), b=rng.normal(0.1, 1.2, n): lambda: st.ttest_ind(a, b, equal_var=False))())
bench("ttestPaired", "scipy", lambda: (lambda a=rng.normal(size=n), b=rng.normal(0.1, 1, n): lambda: st.ttest_rel(a, b))())
bench("ztest1", "numpy", lambda: (lambda x=rng.normal(50, 10, n): lambda: (lambda z=(x.mean() - 50) / (10 / np.sqrt(n)): 2 * st.norm.sf(abs(z)))())())
bench("propTest1", "scipy", lambda: lambda: st.binomtest(350, 1000, 0.3))
bench("propTest2.fisher", "scipy", lambda: lambda: st.fisher_exact([[120, 280], [90, 290]]))
bench("poissonRateTest1", "scipy", lambda: lambda: (st.poisson.cdf(120, 100), st.chi2.ppf([0.025, 0.975], [240, 242])))
bench("varTest1", "scipy", lambda: (lambda x=rng.normal(0, 2, n): lambda: st.chi2.sf((n - 1) * x.var(ddof=1) / 4, n - 1))())
bench("corrTest.pearson", "scipy", lambda: (lambda a=rng.normal(size=n), b=rng.normal(size=n): lambda: st.pearsonr(a, b))())
bench("corrTest.spearman", "scipy", lambda: (lambda a=rng.normal(size=n), b=rng.normal(size=n): lambda: st.spearmanr(a, b))())
bench("grubbs", "numpy", lambda: (lambda x=rng.normal(size=n): lambda: (lambda G=np.abs(x - x.mean()).max() / x.std(ddof=1): st.t.ppf(1 - 0.05 / (2 * n), n - 2))())())
gs = [rng.normal(i * 0.3, 1, N(20_000)) for i in range(5)]
bench("anova", "scipy", lambda: lambda: st.f_oneway(*gs))
bench("levene", "scipy", lambda: lambda: st.levene(*gs))
bench("bartlett", "scipy", lambda: lambda: st.bartlett(*gs))
bench("kruskal", "scipy", lambda: lambda: st.kruskal(*gs))
bench("moodMedian", "scipy", lambda: lambda: st.median_test(*gs))
bench("chi2test", "scipy", lambda: (lambda t=rng.integers(5, 100, (20, 20)): lambda: st.chi2_contingency(t, correction=False))())
bench("chi2gof", "scipy", lambda: (lambda o=rng.integers(20, 100, 50): lambda: st.chisquare(o))())
nc = N(1_000_000)
ca = levels(nc, 10, "a")
cb = levels(nc, 8, "b")
bench("crosstab", "pandas", lambda: lambda: pd.crosstab(ca, cb))
if pl is not None:
    bench("crosstab", "polars", lambda: (lambda df=pl.DataFrame({"a": ca, "b": cb}): lambda: df.group_by(["a", "b"]).len().pivot(on="b", index="a", values="len"))())

# ---- normality ------------------------------------------------------------------------------------------------
n = N(100_000)
bench("andersonDarling", "scipy", lambda: (lambda x=rng.normal(size=n): lambda: st.anderson(x))())
bench("shapiroWilk", "scipy", lambda: (lambda x=rng.normal(size=5000): lambda: st.shapiro(x))())
bench("kolmogorovSmirnov", "scipy", lambda: (lambda x=rng.normal(size=n): lambda: st.kstest(x, "norm", args=(x.mean(), x.std(ddof=1))))())
def idi_setup():
    x = rng.weibull(1.8, N(10_000)) * 12
    def run():
        for d in [st.norm, st.lognorm, st.weibull_min, st.expon, st.gamma, st.logistic]:
            params = d.fit(x) if d is not st.lognorm else d.fit(x, floc=0)
            st.anderson(x, "norm")  # scipy has no AD for the fitted families; nearest is the fit itself
        return params
    return run
bench("individualDistributionID", "scipy", idi_setup, repeat=2)

# ---- multiple comparisons ---------------------------------------------------------------------------------
g6 = [rng.normal(i * 0.3, 1, N(5000)) for i in range(6)]
bench("tukeyHSD", "scipy", lambda: lambda: st.tukey_hsd(*g6))
bench("dunnett", "scipy", lambda: lambda: st.dunnett(*g6[1:], control=g6[0]), repeat=2)
bench("equalVariances", "scipy", lambda: lambda: st.levene(*g6))

# ---- nonparametrics ---------------------------------------------------------------------------------------
n = N(20_000)
bench("mannWhitney", "scipy", lambda: (lambda a=rng.normal(size=n), b=rng.normal(0.1, 1, n): lambda: st.mannwhitneyu(a, b, method="asymptotic"))())
bench("mannWhitney.exact", "scipy", lambda: (lambda a=rng.normal(size=40), b=rng.normal(0.3, 1, 40): lambda: st.mannwhitneyu(a, b, method="exact"))())
bench("signTest", "scipy", lambda: (lambda x=rng.normal(size=N(100_000)): lambda: st.binomtest(int((x > 0).sum()), int((x != 0).sum())))())
bench("wilcoxonSigned", "scipy", lambda: (lambda x=rng.normal(0.05, 1, n): lambda: st.wilcoxon(x, method="approx"))())
bench("wilcoxonSigned.exact", "scipy", lambda: (lambda x=rng.normal(0.2, 1, 50): lambda: st.wilcoxon(x, method="exact"))())
bench("friedman", "scipy", lambda: (lambda t=rng.normal(size=(N(5000), 5)): lambda: st.friedmanchisquare(*t.T))())
def runs_setup():
    x = rng.normal(size=N(1_000_000))
    def run():
        above = x > x.mean()
        r = 1 + int((above[1:] != above[:-1]).sum()); n1 = int(above.sum()); n2 = len(x) - n1; nn = len(x)
        mu = 2 * n1 * n2 / nn + 1; var = 2 * n1 * n2 * (2 * n1 * n2 - nn) / (nn ** 2 * (nn - 1))
        return 2 * st.norm.sf(abs((r - mu) / np.sqrt(var)))
    return run
bench("runsTest", "numpy", runs_setup)

# ---- equivalence -------------------------------------------------------------------------------------------
n = N(100_000)
def tost1_setup():
    x = rng.normal(0.1, 1, n)
    def run():
        mn = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
        return max(st.t.sf((mn + 0.5) / se, n - 1), st.t.cdf((mn - 0.5) / se, n - 1))
    return run
bench("tost1", "numpy", tost1_setup)
def tost2_setup():
    a = rng.normal(size=n); b = rng.normal(0.1, 1, n)
    def run():
        d = a.mean() - b.mean(); va = a.var(ddof=1) / n; vb = b.var(ddof=1) / n; se = np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (n - 1) + vb ** 2 / (n - 1))
        return max(st.t.sf((d + 0.5) / se, df), st.t.cdf((d - 0.5) / se, df))
    return run
bench("tost2", "numpy", tost2_setup)

# ---- descriptive --------------------------------------------------------------------------------------------
n = N(1_000_000)
x1m = rng.normal(50, 5, n)
s1m = pd.Series(x1m)
def desc_pandas():
    d = s1m.describe()
    return d, s1m.skew(), s1m.kurt(), s1m.sem(), s1m.quantile([0.25, 0.5, 0.75], interpolation="linear"), s1m.mode()
bench("descriptiveStats", "pandas", lambda: desc_pandas)
if pl is not None:
    ps = pl.Series(x1m)
    bench("descriptiveStats", "polars", lambda: lambda: (ps.describe(), ps.skew(), ps.kurtosis(), ps.quantile(0.25), ps.quantile(0.75)))
by10 = levels(n, 10)
dfg = pd.DataFrame({"x": x1m, "g": by10})
bench("descriptiveStats.by", "pandas", lambda: lambda: dfg.groupby("g")["x"].agg(["count", "mean", "std", "min", "median", "max", "skew"]))
if pl is not None:
    plg = pl.DataFrame({"x": x1m, "g": by10})
    bench("descriptiveStats.by", "polars", lambda: lambda: plg.group_by("g").agg(pl.col("x").count().alias("n"), pl.col("x").mean().alias("mean"), pl.col("x").std().alias("sd"), pl.col("x").min().alias("min"), pl.col("x").median().alias("med"), pl.col("x").max().alias("max"), pl.col("x").skew().alias("skew")))
def pgof_setup():
    c = rng.poisson(3, n)
    def run():
        lam = c.mean(); k, f = np.unique(c, return_counts=True)
        e = n * st.poisson.pmf(k, lam); return st.chisquare(f, e * f.sum() / e.sum())
    return run
bench("poissonGof", "numpy", pgof_setup)
bench("boxplotStats", "numpy", lambda: (lambda x=rng.normal(size=n): lambda: (np.percentile(x, [25, 50, 75], method="weibull"), x.min(), x.max(), x.mean()))())
fa3 = pd.DataFrame({"y": rng.normal(size=n), "a": levels(n, 3, "a"), "b": levels(n, 4, "b"), "c": levels(n, 5, "c")})
bench("mainEffectsPlot", "pandas", lambda: lambda: [fa3.groupby(k)["y"].agg(["mean", "count"]) for k in ["a", "b", "c"]])
if pl is not None:
    pfa3 = pl.from_pandas(fa3)
    bench("mainEffectsPlot", "polars", lambda: lambda: [pfa3.group_by(k).agg(pl.col("y").mean().alias("mean"), pl.col("y").count().alias("n")) for k in ["a", "b", "c"]])
bench("interactionPlot", "pandas", lambda: lambda: fa3.pivot_table(index="a", columns="b", values="y", aggfunc=["mean", "count"]))
if pl is not None:
    bench("interactionPlot", "polars", lambda: lambda: pfa3.group_by(["a", "b"]).agg(pl.col("y").mean().alias("mean"), pl.col("y").count().alias("n")))
bench("intervalPlot", "pandas", lambda: lambda: dfg.groupby("g")["x"].agg(["mean", "sem", "count"]))
if pl is not None:
    bench("intervalPlot", "polars", lambda: lambda: plg.group_by("g").agg(pl.col("x").mean().alias("mean"), pl.col("x").std().alias("sd"), pl.col("x").count().alias("n")))
bench("ecdf", "numpy", lambda: (lambda x=rng.normal(size=n): lambda: (np.sort(x), np.arange(1, n + 1) / n))())
bench("dotplot", "numpy", lambda: (lambda x=rng.normal(size=n): lambda: np.unique(np.round((x - x.min()) / ((x.max() - x.min()) / 40)), return_counts=True))())

# ---- regression --------------------------------------------------------------------------------------------------
n = N(100_000)
X10 = rng.normal(size=(n, 10)); y10 = X10 @ (0.5 - 0.1 * np.arange(10)) + 1 + rng.normal(size=n)
def ols_numpy():
    Xc = np.column_stack([np.ones(n), X10]); b, *_ = np.linalg.lstsq(Xc, y10, rcond=None)
    r = y10 - Xc @ b; mse = r @ r / (n - 11); cov = np.linalg.inv(Xc.T @ Xc) * mse
    h = np.einsum("ij,jk,ik->i", Xc, np.linalg.inv(Xc.T @ Xc), Xc)  # leverage
    return b, np.sqrt(np.diag(cov)), h
bench("ols", "numpy", lambda: ols_numpy, repeat=3)
bench("fittedLine", "numpy", lambda: (lambda x=rng.normal(size=n): lambda: np.polyfit(x, 1 + 2 * x - 0.5 * x ** 2 + rng.normal(size=n), 2, full=True))())
def logit_setup():
    X = rng.normal(size=(n, 5)); Xc = np.column_stack([np.ones(n), X]); yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.3 + X[:, 0] - 0.5 * X[:, 1])))).astype(float)
    def nll(b): e = Xc @ b; return np.sum(np.logaddexp(0, e) - yb * e)
    def grad(b): e = Xc @ b; return Xc.T @ (1 / (1 + np.exp(-e)) - yb)
    return lambda: optimize.minimize(nll, np.zeros(6), jac=grad, method="BFGS", options={"gtol": 1e-8})
bench("logit", "scipy.optimize", logit_setup, repeat=3)
def pois_setup():
    X = rng.normal(size=(n, 5)); Xc = np.column_stack([np.ones(n), X]); yc = rng.poisson(np.exp(1 + 0.3 * X[:, 0] - 0.2 * X[:, 1])).astype(float)
    def nll(b): e = Xc @ b; return np.sum(np.exp(e) - yc * e)
    def grad(b): e = Xc @ b; return Xc.T @ (np.exp(e) - yc)
    return lambda: optimize.minimize(nll, np.zeros(6), jac=grad, method="BFGS", options={"gtol": 1e-8})
bench("poissonRegression", "scipy.optimize", pois_setup, repeat=3)
def lm_setup():
    a = levels(n, 3, "a"); b = levels(n, 4, "b"); x = rng.normal(size=n); y = x + (a == "a0") + rng.normal(size=n)
    def code(v, lv): return np.column_stack([np.where(v == lv[j], 1.0, np.where(v == lv[-1], -1.0, 0.0)) for j in range(len(lv) - 1)])
    A = code(a, sorted(set(a))); B = code(b, sorted(set(b))); AB = np.column_stack([A[:, i] * B[:, j] for i in range(A.shape[1]) for j in range(B.shape[1])])
    Xf = np.column_stack([np.ones(n), A, B, AB, x]); groups_ = [range(1, 3), range(3, 6), range(6, 12), [12]]
    def run():
        bf, *_ = np.linalg.lstsq(Xf, y, rcond=None); sse = np.sum((y - Xf @ bf) ** 2)
        out = []
        for gcols in groups_:
            keep = [c for c in range(13) if c not in gcols]; br, *_ = np.linalg.lstsq(Xf[:, keep], y, rcond=None); out.append(np.sum((y - Xf[:, keep] @ br) ** 2) - sse)
        return out
    return run
bench("linearModel", "numpy", lm_setup, repeat=3)
def nls_setup():
    x = rng.uniform(0, 800, N(10_000)); y = 238 * (1 - np.exp(-5.5e-4 * x)) + 0.1 * rng.normal(size=len(x))
    return lambda: optimize.curve_fit(lambda t, a, b: a * (1 - np.exp(-b * t)), x, y, p0=[500, 1e-4])
bench("nls", "scipy", nls_setup)
def odr_setup():
    x = rng.normal(10, 3, N(5000)); y = 2 + 1.1 * x + rng.normal(size=len(x))
    data = odr.RealData(x, y, sx=np.ones(len(x)), sy=np.ones(len(x)))
    return lambda: odr.ODR(data, odr.Model(lambda B, t: B[0] + B[1] * t), beta0=[0, 1]).run()
bench("orthogonalRegression", "scipy", odr_setup)

# ---- SPC ---------------------------------------------------------------------------------------------------------
n = N(1_000_000)
def imr_setup():
    x = rng.normal(10, 1, n)
    def run():
        mr = np.abs(np.diff(x)); s = mr.mean() / 1.128; c = x.mean()
        return (x > c + 3 * s) | (x < c - 3 * s), (mr > 3.267 * mr.mean())
    return run
bench("controlChart.imr", "numpy", imr_setup)
def xbar_setup():
    x = rng.normal(10, 1, n).reshape(-1, 5)
    def run():
        xb = x.mean(1); r = x.max(1) - x.min(1); c = xb.mean(); rb = r.mean()
        return (xb > c + 0.577 * rb) | (xb < c - 0.577 * rb), r > 2.114 * rb
    return run
bench("controlChart.xbar", "numpy", xbar_setup)
def p_setup():
    d = rng.binomial(100, 0.05, N(100_000))
    def run():
        p = d / 100; pb = p.mean(); s = np.sqrt(pb * (1 - pb) / 100)
        return (p > pb + 3 * s) | (p < pb - 3 * s)
    return run
bench("controlChart.p", "numpy", p_setup)
xs1m = rng.normal(size=n)
bench("ewma", "pandas", lambda: (lambda s=pd.Series(xs1m): lambda: s.ewm(alpha=0.2, adjust=False).mean())())
if pl is not None:
    bench("ewma", "polars", lambda: (lambda s=pl.Series(xs1m): lambda: s.ewm_mean(alpha=0.2, adjust=False))())
bench("cusum", "numpy", lambda: lambda: (np.maximum.accumulate(np.cumsum(xs1m - 0.5)), np.cumsum(xs1m)))
bench("movingAverage", "pandas", lambda: (lambda s=pd.Series(xs1m): lambda: s.rolling(5).mean())())
if pl is not None:
    bench("movingAverage", "polars", lambda: (lambda s=pl.Series(xs1m): lambda: s.rolling_mean(5))())
bench("gChart", "scipy", lambda: (lambda gcount=rng.integers(0, 200, N(100_000)): lambda: (lambda p=1 / (gcount.mean() + 1): st.geom.ppf([0.00135, 0.99865], p))())())
bench("tChart", "scipy", lambda: (lambda t=rng.weibull(1.5, N(100_000)) * 20: lambda: st.weibull_min.fit(t, floc=0))(), repeat=2)
def t2_setup():
    X = rng.normal(size=(N(100_000), 5))
    def run():
        mu = X.mean(0); Si = np.linalg.inv(np.cov(X.T)); d = X - mu; t2 = np.einsum("ij,jk,ik->i", d, Si, d)
        return t2, st.beta.ppf(1 - 0.00135, 2.5, (len(X) - 6) / 2)
    return run
bench("t2Chart", "numpy", t2_setup)
def mewma_setup():
    X = rng.normal(size=(N(100_000), 3)); lam = 0.1
    def run():
        mu = X.mean(0); Si = np.linalg.inv(np.cov(X.T)); Z = np.zeros(3); out = np.empty(len(X))
        for i in range(len(X)):
            Z = lam * (X[i] - mu) + (1 - lam) * Z; v = lam / (2 - lam) * (1 - (1 - lam) ** (2 * (i + 1))); out[i] = Z @ Si @ Z / v
        return out
    return run
bench("mewma", "numpy", mewma_setup, repeat=2)
def gv_setup():
    X = rng.normal(size=(N(50_000), 3)).reshape(-1, 5, 3)
    return lambda: np.array([np.linalg.det(np.cov(g.T)) for g in X])
bench("generalizedVarianceChart", "numpy", gv_setup)
def cap_setup():
    x = rng.normal(10, 1, n)
    def run():
        sub = x.reshape(-1, 5); rbar = (sub.max(1) - sub.min(1)).mean(); sw = rbar / 2.326; so = x.std(ddof=1); mu = x.mean()
        return (6 / (6 * sw), min(13 - mu, mu - 7) / (3 * sw), 6 / (6 * so), min(13 - mu, mu - 7) / (3 * so), st.norm.sf((13 - mu) / so) + st.norm.cdf((7 - mu) / so))
    return run
bench("capability", "numpy", cap_setup)
bench("boxCoxLambda", "scipy", lambda: (lambda x=rng.lognormal(1, 0.4, N(100_000)): lambda: st.boxcox(x))())
bench("weibullFit", "scipy", lambda: (lambda x=rng.weibull(1.8, N(100_000)) * 12: lambda: st.weibull_min.fit(x, floc=0))(), repeat=2)
bench("acceptanceSampling", "scipy", lambda: lambda: st.binom.cdf(3, 125, np.linspace(0.001, 0.2, 100)))
catn = levels(n, 30, "cat")
bench("pareto", "pandas", lambda: lambda: pd.Series(catn).value_counts().cumsum())
if pl is not None:
    bench("pareto", "polars", lambda: (lambda s=pl.Series(catn): lambda: s.value_counts(sort=True))())
mv = pd.DataFrame({"m": rng.normal(size=N(100_000)), "a": levels(N(100_000), 3, "a"), "b": levels(N(100_000), 4, "b"), "c": levels(N(100_000), 2, "c")})
bench("multiVari", "pandas", lambda: lambda: (mv.groupby(["a", "b", "c"])["m"].mean(), mv.groupby(["a", "b"])["m"].mean(), mv.groupby("a")["m"].mean()))

# ---- time series ------------------------------------------------------------------------------------------------------
n = N(100_000)
tt = np.arange(n); ser = 10 + 0.01 * tt + 2 * np.sin(2 * np.pi * tt / 12) + rng.normal(size=n)
bench("trendAnalysis", "numpy", lambda: lambda: np.polyfit(tt, ser, 2))
def acf_np(x, maxlag):
    xc = x - x.mean(); den = xc @ xc
    return np.array([(xc[:-k] @ xc[k:]) / den if k else 1.0 for k in range(maxlag + 1)])
bench("acf", "numpy", lambda: lambda: acf_np(ser, 40))
ser2 = np.roll(ser, 2) + rng.normal(size=n)
bench("ccf", "numpy", lambda: lambda: [np.corrcoef(ser[:-k] if k else ser, ser2[k:] if k else ser2)[0, 1] for k in range(21)])
bench("periodogram", "scipy", lambda: (lambda y=ser[:N(20_000)]: lambda: signal.periodogram(y, detrend="constant"))())

# ---- reliability ----------------------------------------------------------------------------------------------------
n = N(10_000)
def life_setup():
    t = rng.weibull(1.5, n) * 100; c = t > 150; t = np.minimum(t, 150)
    def nll(th):
        b, ls = th; s = np.exp(ls); z = (np.log(t) - b) / s
        return -np.sum(np.where(~c, z - np.exp(z) - np.log(s), -np.exp(z)))
    return lambda: optimize.minimize(nll, [4.5, np.log(0.7)], method="BFGS")
bench("reliabilityFit", "scipy", life_setup)
def km_setup():
    m_ = N(100_000); t = rng.weibull(1.5, m_) * 100; c = (t > 150).astype(int); t = np.minimum(t, 150)
    def run():
        order = np.argsort(t); ts = t[order]; cs = c[order]; ut, idx = np.unique(ts, return_index=True)
        d = np.add.reduceat(1 - cs, idx); n_at = m_ - idx
        return np.cumprod(1 - d / n_at)
    return run
bench("kaplanMeier", "numpy", km_setup)
def lifereg_setup():
    X = rng.normal(size=(n, 2)); t = np.exp(4 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.5 * np.log(-np.log(rng.uniform(size=n)))); c = t > 200; t = np.minimum(t, 200)
    Xc = np.column_stack([np.ones(n), X])
    def nll(th):
        b = th[:3]; s = np.exp(th[3]); z = (np.log(t) - Xc @ b) / s
        return -np.sum(np.where(~c, z - np.exp(z) - np.log(s), -np.exp(z)))
    return lambda: optimize.minimize(nll, [5, 0, 0, np.log(0.5)], method="BFGS")
bench("lifeRegression", "scipy.optimize", lifereg_setup)
def nhpp_setup():
    t = np.sort(1000 * np.sqrt(rng.uniform(size=n)))
    return lambda: (n / np.sum(np.log(1000 / t)), (np.sum(t / 1000) - n / 2) / np.sqrt(n / 12))
bench("powerLawNHPP", "numpy", nhpp_setup)
def probit_setup():
    dose = np.arange(1, 9, dtype=float); tr = np.full(8, 100); ev = rng.binomial(tr, st.norm.cdf(-3 + 0.8 * dose))
    def nll(bb): p_ = np.clip(st.norm.cdf(bb[0] + bb[1] * dose), 1e-12, 1 - 1e-12); return float(-np.sum(ev * np.log(p_) + (tr - ev) * np.log(1 - p_)))
    return lambda: optimize.minimize(nll, [-2, 0.5], method="BFGS")
bench("probitAnalysis", "scipy.optimize", probit_setup)

# ---- multivariate ------------------------------------------------------------------------------------------------
n = N(100_000)
bench("pca", "numpy", lambda: (lambda X=rng.normal(size=(n, 10)): lambda: np.linalg.eigh(np.cov(X.T)))())
bench("hclust", "scipy", lambda: (lambda X=rng.normal(size=(1500, 5)): lambda: cluster.hierarchy.linkage(X, method="average"))())
bench("clusterVariables", "scipy", lambda: (lambda X=rng.normal(size=(N(10_000), 30)): lambda: cluster.hierarchy.linkage((1 - np.corrcoef(X.T))[np.triu_indices(30, 1)], method="average"))())
def ca_setup():
    T = rng.integers(1, 100, (30, 30)).astype(float)
    def run():
        P = T / T.sum(); r = P.sum(1); c = P.sum(0); S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c)); return np.linalg.svd(S)
    return run
bench("correspondence", "numpy", ca_setup)
def mca_setup():
    m_ = N(5000); cols = []
    for k in [3, 4, 5, 2]:
        v = levels(m_, k)
        for lv in sorted(set(v)): cols.append((v == lv).astype(float))
    Z = np.column_stack(cols)
    def run():
        P = Z / Z.sum(); r = P.sum(1); c = P.sum(0); S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c)); return np.linalg.svd(S, full_matrices=False)
    return run
bench("multipleCorrespondence", "numpy", mca_setup, repeat=3)
def item_setup():
    lat = rng.normal(size=N(10_000)); items = np.column_stack([lat + (0.5 + 0.05 * j) * rng.normal(size=len(lat)) for j in range(20)])
    def run():
        k = 20; alpha = k / (k - 1) * (1 - items.var(axis=0, ddof=1).sum() / items.sum(1).var(ddof=1)); R = np.corrcoef(items.T)
        return alpha, np.diag(np.linalg.inv(R)), [np.corrcoef(items[:, j], items.sum(1) - items[:, j])[0, 1] for j in range(k)]
    return run
bench("itemAnalysis", "numpy", item_setup)
def manova_setup():
    m_ = N(10_000); Y = rng.normal(size=(m_, 3)); a = levels(m_, 3, "a"); b = levels(m_, 4, "b")
    def code(v, lv): return np.column_stack([np.where(v == lv[j], 1.0, np.where(v == lv[-1], -1.0, 0.0)) for j in range(len(lv) - 1)])
    A = code(a, sorted(set(a))); B = code(b, sorted(set(b))); AB = np.column_stack([A[:, i] * B[:, j] for i in range(2) for j in range(3)])
    Xf = np.column_stack([np.ones(m_), A, B, AB]); groups_ = [range(1, 3), range(3, 6), range(6, 12)]
    def sscp(M): bb, *_ = np.linalg.lstsq(M, Y, rcond=None); R = Y - M @ bb; return R.T @ R
    def run():
        E = sscp(Xf); out = []
        for gcols in groups_:
            keep = [c for c in range(12) if c not in gcols]; H = sscp(Xf[:, keep]) - E; out.append(linalg.eigh(H, E, eigvals_only=True))
        return out
    return run
bench("manovaModel", "numpy", manova_setup)

# ---- random data ------------------------------------------------------------------------------------------------
n = N(1_000_000)
bench("random.normal", "numpy", lambda: lambda: rng.normal(size=n))
bench("random.gamma", "numpy", lambda: lambda: rng.gamma(2.5, 2, n))
bench("random.poisson", "numpy", lambda: lambda: rng.poisson(4, n))

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({"scale": SCALE, "repeat": REPEAT, "numpy": np.__version__, "pandas": pd.__version__, "polars": getattr(pl, "__version__", None), "scipy": __import__("scipy").__version__, "rows": rows}, indent=1))
print(f"wrote {OUT}")
