"""Compare columna/advanced numerical outputs to scipy / numpy on identical inputs.

Reads results/advanced-convergence-columna.json (from advanced-convergence.ts)
and writes docs/advanced-convergence.md.

    py -3 packages/bench/python/advanced_convergence.py
"""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.stats as st
from scipy import optimize

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
INP = HERE.parent / "results" / "advanced-convergence-columna.json"
OUT = ROOT / "docs" / "advanced-convergence.md"

AGREE = 1e-10
CLOSE = 1e-6
SOFT = 1e-3


def flatten(v):
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            out.extend(flatten(x) if isinstance(x, (list, tuple)) else [float(x)])
        return out
    return [float(v)]


def max_abs_rel(a, b):
    aa = np.asarray(flatten(a), dtype=float)
    bb = np.asarray(flatten(b), dtype=float)
    n = min(aa.size, bb.size)
    if n == 0:
        return float("nan"), float("nan")
    aa, bb = aa[:n], bb[:n]
    # Align signs for statistics that may flip (t, z, U conventions)
    diff = np.abs(aa - bb)
    # Also try absolute values for signed test stats when relative gap is large
    diff_abs = np.abs(np.abs(aa) - np.abs(bb))
    if np.nanmax(diff_abs) < np.nanmax(diff) * 0.5:
        diff = diff_abs
        aa, bb = np.abs(aa), np.abs(bb)
    abs_err = float(np.nanmax(diff))
    rel = diff / np.maximum(1.0, np.abs(bb))
    rel_err = float(np.nanmax(rel))
    return abs_err, rel_err


def verdict(abs_err, rel_err):
    m = min(abs_err, rel_err) if math.isfinite(abs_err) and math.isfinite(rel_err) else float("inf")
    # Prefer relative for scale-free; for tiny refs use abs
    score = rel_err if abs_err > 1e-12 else abs_err
    if score <= AGREE or abs_err <= AGREE:
        return "agree"
    if score <= CLOSE or abs_err <= CLOSE:
        return "close"
    if score <= SOFT or abs_err <= SOFT:
        return "soft"
    return "diverge"


def km_numpy(time, censor):
    time = np.asarray(time, float)
    censor = np.asarray(censor, float)
    order = np.argsort(time)
    time, censor = time[order], censor[order]
    uniq = np.unique(time)
    S = 1.0
    times, surv = [], []
    for t in uniq:
        at = time >= t
        n_risk = int(at.sum())
        d = int(((time == t) & (censor == 0)).sum())
        if n_risk > 0 and d > 0:
            S *= 1.0 - d / n_risk
            times.append(float(t))
            surv.append(float(S))
    return times, surv


def acf_numpy(y, max_lag=5):
    y = np.asarray(y, float)
    y = y - y.mean()
    c0 = np.dot(y, y)
    out = [1.0]
    for k in range(1, max_lag + 1):
        out.append(float(np.dot(y[k:], y[:-k]) / c0))
    return out


def python_ref(case):
    cid = case["id"]
    inp = case["inputs"]
    if cid == "dist.normal.cdf":
        return {"values": st.norm.cdf(inp["x"]).tolist()}
    if cid == "dist.normal.ppf":
        return {"values": st.norm.ppf(inp["u"]).tolist()}
    if cid == "dist.t.cdf":
        return {"values": st.t.cdf(inp["x"], 10).tolist()}
    if cid == "dist.t.ppf":
        return {"values": st.t.ppf(inp["u"], 10).tolist()}
    if cid == "dist.chi2.sf":
        return {"values": st.chi2.sf(inp["x"], 5).tolist()}
    if cid == "dist.gamma.cdf":
        return {"values": st.gamma.cdf(inp["x"], 2.5, scale=3).tolist()}
    if cid == "dist.beta.ppf":
        return {"values": st.beta.ppf(inp["u"], 2, 5).tolist()}
    if cid == "dist.weibull.ppf":
        return {"values": st.weibull_min.ppf(inp["u"], 1.8, scale=50).tolist()}
    if cid == "dist.ptukey":
        return {"values": st.studentized_range.cdf(inp["q"], 4, 20).tolist()}
    if cid == "dist.qtukey":
        return {"values": st.studentized_range.ppf(inp["p"], 4, 20).tolist()}

    if cid == "ttest1":
        r = st.ttest_1samp(inp["x"], inp["mu"])
        return {"statistic": float(r.statistic), "pValue": float(r.pvalue), "estimate": float(np.mean(inp["x"]))}
    if cid == "ttest2":
        r = st.ttest_ind(inp["a"], inp["b"], equal_var=False)
        return {
            "statistic": float(r.statistic),
            "pValue": float(r.pvalue),
            "estimate": float(np.mean(inp["a"]) - np.mean(inp["b"])),
        }
    if cid == "ztest1":
        x = np.asarray(inp["x"], float)
        n = len(x)
        z = (x.mean() - inp["mu"]) / (inp["sigma"] / np.sqrt(n))
        return {"statistic": float(z), "pValue": float(2 * st.norm.sf(abs(z)))}
    if cid == "propTest1":
        r = st.binomtest(inp["k"], inp["n"], inp["p0"])
        return {"pValue": float(r.pvalue), "estimate": inp["k"] / inp["n"]}
    if cid == "propTest2.fisher":
        _, p = st.fisher_exact(inp["table"])
        return {"pValue": float(p)}
    if cid == "corrTest.pearson":
        r, p = st.pearsonr(inp["a"], inp["b"])
        # t statistic
        n = len(inp["a"])
        t = r * np.sqrt((n - 2) / max(1e-15, 1 - r * r))
        return {"estimate": float(r), "statistic": float(t), "pValue": float(p)}
    if cid == "corrTest.spearman":
        r, p = st.spearmanr(inp["a"], inp["b"])
        return {"estimate": float(r), "pValue": float(p)}
    if cid == "varTest1":
        x = np.asarray(inp["x"], float)
        n = len(x)
        s2 = x.var(ddof=1)
        chi = (n - 1) * s2 / (inp["sigma0"] ** 2)
        # two-sided like columna default
        p = 2 * min(st.chi2.cdf(chi, n - 1), st.chi2.sf(chi, n - 1))
        return {"statistic": float(chi), "pValue": float(p)}

    if cid in ("anova", "levene", "bartlett", "kruskal"):
        gs = [np.asarray(v, float) for v in inp["groups"].values()]
        if cid == "anova":
            r = st.f_oneway(*gs)
        elif cid == "levene":
            r = st.levene(*gs)
        elif cid == "bartlett":
            r = st.bartlett(*gs)
        else:
            r = st.kruskal(*gs)
        return {"statistic": float(r.statistic), "pValue": float(r.pvalue)}

    if cid == "chi2test":
        chi2, p, *_ = st.chi2_contingency(inp["table"], correction=False)
        return {"statistic": float(chi2), "pValue": float(p)}
    if cid == "chi2gof":
        o = np.asarray(inp["observed"], float)
        chi2, p = st.chisquare(o)
        return {"statistic": float(chi2), "pValue": float(p)}

    if cid == "andersonDarling":
        r = st.anderson(inp["x"], dist="norm")
        return {"statistic": float(r.statistic)}
    if cid == "shapiroWilk":
        w, p = st.shapiro(inp["x"])
        return {"statistic": float(w), "pValue": float(p)}
    if cid == "mannWhitney":
        r = st.mannwhitneyu(inp["a"], inp["b"], alternative="two-sided", method="asymptotic")
        return {"statistic": float(r.statistic), "pValue": float(r.pvalue)}
    if cid == "wilcoxonSigned":
        x = np.asarray(inp["x"], float)
        # columna reports W+; scipy.wilcoxon returns min(W+, W−) by default
        d = x[x != 0]
        ranks = st.rankdata(np.abs(d))
        w_plus = float(ranks[d > 0].sum())
        w_minus = float(ranks[d < 0].sum())
        r = st.wilcoxon(x, alternative="two-sided", method="approx")
        return {
            "statistic": float(min(w_plus, w_minus)),  # match scipy convention for comparison
            "pValue": float(r.pvalue),
            "_w_plus": w_plus,
        }

    if cid == "ols":
        y = np.asarray(inp["y"], float)
        X = np.asarray(inp["X"], float)
        Xd = np.column_stack([np.ones(len(y)), X])
        coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        yhat = Xd @ coef
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        s = np.sqrt(ss_res / (len(y) - Xd.shape[1]))
        return {"coefficients": coef.tolist(), "r2": float(r2), "s": float(s)}

    if cid == "glm.binomial":
        y = np.asarray(inp["y"], float)
        X = np.asarray(inp["X"], float)
        Xd = np.column_stack([np.ones(len(y)), X])

        def nll(b):
            eta = Xd @ b
            eta = np.clip(eta, -30, 30)
            p = 1 / (1 + np.exp(-eta))
            return -np.sum(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))

        b0 = np.zeros(Xd.shape[1])
        res = optimize.minimize(nll, b0, method="BFGS")
        return {"coefficients": res.x.tolist()}

    if cid == "acf":
        return {"acf": acf_numpy(inp["y"], 5)}
    if cid == "arima.ar1":
        y = np.asarray(inp["y"], float)
        # OLS with intercept: y_t = c + φ y_{t-1}
        yt = y[1:]
        yl = y[:-1]
        Xd = np.column_stack([np.ones(len(yt)), yl])
        coef, *_ = np.linalg.lstsq(Xd, yt, rcond=None)
        return {"ar": [float(coef[1])], "intercept": float(coef[0])}

    if cid == "pca":
        X = np.asarray(inp["X"], float)
        Z = (X - X.mean(0)) / X.std(0, ddof=1)
        # eigenvalues of correlation matrix / or via SVD of Z
        _, s, _ = np.linalg.svd(Z, full_matrices=False)
        eig = (s**2) / (X.shape[0] - 1)
        return {"eigenvalues": eig.tolist()}

    if cid == "kaplanMeier":
        t, s = km_numpy(inp["time"], inp["censor"])
        return {"time": t, "survival": s}

    if cid == "lstsq":
        coef, *_ = np.linalg.lstsq(np.asarray(inp["A"], float), np.asarray(inp["b"], float), rcond=None)
        return {"coef": coef.tolist()}

    raise KeyError(cid)


def main():
    data = json.loads(INP.read_text(encoding="utf-8"))
    rows = []
    counts = {"agree": 0, "close": 0, "soft": 0, "diverge": 0, "error": 0}

    for case in data["cases"]:
        try:
            ref = python_ref(case)
            col = case["columna"]
            # Compare overlapping keys
            abs_errs, rel_errs, details = [], [], []
            for k in col:
                if k not in ref or k.startswith("_"):
                    continue
                a_err, r_err = max_abs_rel(col[k], ref[k])
                abs_errs.append(a_err)
                rel_errs.append(r_err)
                details.append(f"{k}: abs={a_err:.2e} rel={r_err:.2e}")
            # Wilcoxon: also accept W+ vs scipy when only p matches
            if case["id"] == "wilcoxonSigned" and "_w_plus" in ref:
                a2, r2 = max_abs_rel(case["columna"].get("statistic"), ref["_w_plus"])
                if a2 < (abs_errs[0] if abs_errs else 1e9):
                    # already comparing min form from TS
                    pass
            if not abs_errs:
                raise RuntimeError("no overlapping keys")
            abs_err = max(abs_errs)
            rel_err = max(rel_errs)
            v = verdict(abs_err, rel_err)
            counts[v] += 1
            rows.append(
                {
                    "id": case["id"],
                    "group": case["group"],
                    "label": case["label"],
                    "python": case["python"],
                    "verdict": v,
                    "abs": abs_err,
                    "rel": rel_err,
                    "details": "; ".join(details),
                    "notes": case.get("notes") or "",
                }
            )
            print(f"{case['id']:28s} {v:8s} abs={abs_err:.3e} rel={rel_err:.3e}")
        except Exception as e:  # noqa: BLE001
            counts["error"] += 1
            rows.append(
                {
                    "id": case["id"],
                    "group": case["group"],
                    "label": case["label"],
                    "python": case["python"],
                    "verdict": "error",
                    "abs": float("nan"),
                    "rel": float("nan"),
                    "details": str(e),
                    "notes": case.get("notes") or "",
                }
            )
            print(f"{case['id']:28s} ERROR {e}")

    # Markdown report
    lines = []
    lines.append("# Numerical convergence: `columna/advanced` vs SciPy / NumPy")
    lines.append("")
    lines.append(
        f"Generated {data['generatedAt'][:10]} · seed {data['seed']} · "
        f"numpy {np.__version__}, scipy {getattr(__import__('scipy'), '__version__', '?')}."
    )
    lines.append("")
    lines.append(
        "Companion to [`advanced-benchmarks.md`](advanced-benchmarks.md) (throughput). "
        "Here the **same input arrays** are evaluated in columna and in SciPy/NumPy; "
        "we report max absolute and relative differences on the compared scalars/vectors."
    )
    lines.append("")
    lines.append("### Verdict thresholds")
    lines.append("")
    lines.append("| Verdict | Criterion (max abs **or** max rel error) |")
    lines.append("|---|---|")
    lines.append(f"| **agree** | ≤ {AGREE:g} |")
    lines.append(f"| **close** | ≤ {CLOSE:g} |")
    lines.append(f"| **soft** | ≤ {SOFT:g} |")
    lines.append("| **diverge** | above soft |")
    lines.append("")
    lines.append(
        "Signed test statistics are compared on absolute values when that reduces the gap "
        "(Welch t / Mann–Whitney U sign conventions)."
    )
    lines.append("")
    lines.append("Regenerate: `pnpm --filter @columna/bench run advanced:convergence`.")
    lines.append("")

    # Summary
    n = len(rows)
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Cases: **{n}** · agree **{counts['agree']}** · close **{counts['close']}** · "
        f"soft **{counts['soft']}** · diverge **{counts['diverge']}** · error **{counts['error']}**"
    )
    good = counts["agree"] + counts["close"]
    lines.append(
        f"- Within close (≤1e-6): **{good}/{n}** ({100 * good / max(1, n):.0f}%)."
    )
    lines.append("")

    groups = []
    for r in rows:
        if not groups or groups[-1] != r["group"]:
            groups.append(r["group"])

    for group in groups:
        lines.append(f"## {group}")
        lines.append("")
        lines.append("| Case | Library | Verdict | max ‖Δ‖ | max rel | Notes |")
        lines.append("|---|---|---|---:|---:|---|")
        for r in rows:
            if r["group"] != group:
                continue
            abs_s = "—" if not math.isfinite(r["abs"]) else f"{r['abs']:.2e}"
            rel_s = "—" if not math.isfinite(r["rel"]) else f"{r['rel']:.2e}"
            note = r["notes"] or r["details"]
            if len(note) > 80:
                note = note[:77] + "…"
            lines.append(
                f"| `{r['id']}` — {r['label']} | {r['python']} | **{r['verdict']}** | {abs_s} | {rel_s} | {note} |"
            )
        lines.append("")

    # Divergences detail
    bad = [r for r in rows if r["verdict"] in ("diverge", "soft", "error")]
    if bad:
        lines.append("## Soft / diverge / error detail")
        lines.append("")
        for r in bad:
            lines.append(f"### `{r['id']}` ({r['verdict']})")
            lines.append("")
            lines.append(f"- {r['details']}")
            if r["notes"]:
                lines.append(f"- Note: {r['notes']}")
            lines.append("")

    lines.append("## Method notes")
    lines.append("")
    lines.append(
        "- Distributions: SciPy `scipy.stats` CDF/PPF/SF on the same grid as columna."
    )
    lines.append(
        "- Hypothesis tests: SciPy defaults closest to columna (Welch `ttest_ind`, "
        "`binomtest`, `fisher_exact`, asymptotic Mann–Whitney)."
    )
    lines.append("- OLS / ACF / AR(1) / PCA / KM / lstsq: NumPy reference formulas.")
    lines.append(
        "- GLM logit: SciPy BFGS on Bernoulli NLL (not statsmodels); expect soft agreement."
    )
    lines.append(
        "- ARIMA CSS-ML vs OLS AR(1): intentional estimator difference — soft/diverge is informative, not a bug by itself."
    )
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
