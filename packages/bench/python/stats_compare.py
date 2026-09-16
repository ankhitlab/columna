"""pandas side of the statistics bench.

1. Correctness: reads results/stats-sample.csv (written by stats-bench.ts) and prints reference
   values for rank / z-score / math / corr / cov that stats-bench.ts is compared against.
2. Speed: times the same operations on synthetic data of the same sizes (numbers differ from the
   TS sample — only distributions match — so this is a throughput comparison, not a value check).

    py -3 packages/bench/python/stats_compare.py
    BENCH_SIZES=100000,1000000 py -3 packages/bench/python/stats_compare.py
"""
import io
import json
import sys
import os
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
SIZES = [int(s) for s in os.environ.get("BENCH_SIZES", "100000,1000000,10000000").split(",")]
REPEAT = int(os.environ.get("BENCH_REPEAT", "3"))


def reference_values() -> dict:
    df = pd.read_csv(RESULTS / "stats-sample.csv")
    out = {"n": len(df)}
    r = df["w"].rank(method="average")
    out["rank_w_sum"] = float(r.sum())
    out["rank_w_first10"] = r.head(10).tolist()
    z = (df["x"] - df["x"].mean()) / df["x"].std()
    out["zscore_first5"] = z.head(5).tolist()
    out["geo_mean_x"] = float(np.exp(np.log(df["x"]).mean()))
    out["round2_first5"] = df["x"].round(2).head(5).tolist()  # note: pandas = half-to-even
    out["log_first3"] = np.log(df["x"]).head(3).tolist()
    out["corr_pearson"] = df.corr().round(12).to_dict()
    out["corr_spearman"] = df.corr(method="spearman").round(12).to_dict()
    out["cov"] = df.cov().round(9).to_dict()
    out["median_x"] = float(df["x"].median())
    out["rows_gt_median"] = int((df["x"] > df["x"].median()).sum())
    return out


def make(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x = 50 + 10 * (rng.random((n, 4)).sum(axis=1) - 2) * 1.7320508
    y = 0.8 * x + 5 * (rng.random(n) - 0.5)
    y[7::100] = np.nan
    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "z": rng.random(n) * 100,
            "w": np.floor(rng.random(n) * 101).astype("int32"),
            "v": (x - 50) ** 2 + 20 * (rng.random(n) - 0.5),
        }
    )


def bench(n: int, name: str, fn, rows: list) -> None:
    fn()  # warm-up
    times = []
    peak = 0
    for _ in range(REPEAT):
        tracemalloc.start()
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
        _, p = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak = max(peak, p)
    times.sort()
    med = times[len(times) // 2]
    rows.append({"n": n, "op": name, "msMedian": med, "msMin": times[0], "peakAllocMB": peak / 1048576})
    print(f"  {name:<34} {med:9.1f} ms  {n / (med / 1000) / 1e6:6.1f} M rows/s  peak alloc {peak / 1048576:8.1f} MB")


def main() -> None:
    ref = reference_values()
    (RESULTS / "stats-pandas-reference.json").write_text(json.dumps(ref, indent=2))
    print(f"pandas {pd.__version__} · numpy {np.__version__} · reference → results/stats-pandas-reference.json")

    rows: list = []
    for n in SIZES:
        print(f"\n=== n = {n:,} (pandas) ===")
        t0 = time.perf_counter()
        df = make(n)
        print(f"  build DataFrame (5 cols)              {(time.perf_counter() - t0) * 1000:9.0f} ms  frame ≈ {df.memory_usage().sum() / 1048576:8.1f} MB")
        bench(n, "rank average (w, ties)", lambda: df["w"].rank(method="average"), rows)
        bench(n, "rank average (x, no ties)", lambda: df["x"].rank(method="average"), rows)
        bench(n, "rank dense (w)", lambda: df["w"].rank(method="dense"), rows)
        bench(n, "zscore (x−mean)/std", lambda: (df["x"] - df["x"].mean()) / df["x"].std(), rows)
        bench(n, "share x / sum(x)", lambda: df["x"] / df["x"].sum(), rows)
        bench(n, "filter x > median(x)", lambda: df[df["x"] > df["x"].median()], rows)
        bench(n, "nunique(w)", lambda: df["w"].nunique(), rows)
        bench(n, "math: log, sqrt, pow2, round2 (4 cols)", lambda: df.assign(l=np.log(df["x"]), s=np.sqrt(df["x"]), p=df["x"] ** 2, r=df["x"].round(2)), rows)
        bench(n, "math: exp(mean(log x)) geo-mean", lambda: np.exp(np.log(df["x"]).mean()), rows)
        bench(n, "corr pearson 5×5 (1% nulls in y)", lambda: df.corr(), rows)
        bench(n, "corr pearson 2 cols no nulls", lambda: df[["x", "v"]].corr(), rows)
        bench(n, "cov 5×5", lambda: df.cov(), rows)
        if n <= 1_000_000 or os.environ.get("BENCH_SPEARMAN_10M"):
            bench(n, "corr spearman 5×5", lambda: df.corr(method="spearman"), rows)
        bench(n, "describe() [baseline]", lambda: df.describe(), rows)

    (RESULTS / "stats-pandas-latest.json").write_text(json.dumps({"pandas": pd.__version__, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
