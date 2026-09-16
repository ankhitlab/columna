"""
Comparative benchmark: pandas vs polars vs columna (via subprocess JSON).

Classic + API-parity ops (str/dt/when/is_in, sample/tail/explode, join variants,
timeseries, write/profile).

Usage:
  py -3 -m pip install pandas polars numpy
  py -3 packages/bench/python/compare.py --sizes 1000000 --runs 3
  BENCH_OPS=filter,str_contains,groupby_stats  (optional subset)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl


CITIES = [
    "Berlin",
    "Paris",
    "London",
    "Madrid",
    "Rome",
    "Vienna",
    "Warsaw",
    "Prague",
    "Lisbon",
    "Dublin",
]
CATEGORIES = [f"c{i}" for i in range(50)]
SEGMENTS = [f"s{i}" for i in range(20)]
TAG_VARIANTS = ['["a"]', '["a","b"]', '["x","y","z"]', "[]"]
PAYLOAD_VARIANTS = [
    '{"user":{"id":1},"n":10}',
    '{"user":{"id":2},"n":20}',
    '{"user":{"id":3},"n":30}',
]

ALL_OPS = [
    "filter",
    "groupby_agg",
    "groupby_multi",
    "sort",
    "join",
    "pipeline",
    "select",
    "with_column",
    "unique",
    "value_counts",
    "rolling",
    "describe",
    "melt",
    "with_columns",
    "str_contains",
    "str_lower",
    "dt_parts",
    "when_then",
    "is_in",
    "is_between",
    "groupby_stats",
    "tail",
    "sample",
    "explode",
    "semi_join",
    "anti_join",
    "cross_join",
    "join_asof",
    "shift_diff",
    "pct_change",
    "expanding",
    "interpolate",
    "unnest",
    "transpose",
    "map_elements",
    "to_csv",
    "profile",
]


@dataclass
class Timing:
    library: str
    n: int
    op: str
    ms: float
    extra: dict[str, Any] | None = None


def make_fact_pandas(n: int, seed: int = 42) -> pd.DataFrame:
    t0 = time.perf_counter()
    rng = np.arange(n, dtype=np.int32)
    user_mod = max(n // 10, 1)
    base_ts = np.datetime64("2020-01-01", "ms").astype("datetime64[ms]").astype(np.int64)
    city = pd.Categorical.from_codes(rng % len(CITIES), categories=CITIES)
    df = pd.DataFrame(
        {
            "id": rng,
            "user_id": (rng.astype(np.int64) * 7 + seed) % user_mod,
            "city": city,
            "category": pd.Categorical.from_codes(rng % 50, categories=CATEGORIES),
            "age": 18 + (rng % 50),
            "salary": (40_000 + (rng % 1000) * 10 + (rng % 17)).astype(np.float64),
            "ts": (base_ts + rng.astype(np.int64) * 60_000).astype(np.int64),
            "tags": [TAG_VARIANTS[i % len(TAG_VARIANTS)] for i in range(n)],
            "payload": [PAYLOAD_VARIANTS[i % len(PAYLOAD_VARIANTS)] for i in range(n)],
        }
    )
    print(f"    pandas fact n={n:,} in {(time.perf_counter() - t0) * 1000:.0f} ms", flush=True)
    return df


def make_dim_pandas(n_fact: int) -> pd.DataFrame:
    n = max(n_fact // 10, 1)
    rng = np.arange(n, dtype=np.int32)
    return pd.DataFrame(
        {
            "user_id": rng,
            "segment": pd.Categorical.from_codes(rng % 20, categories=SEGMENTS),
            "score": (rng.astype(np.int64) * 13) % 100,
        }
    )


def make_asof_dim_pandas(n_fact: int) -> pd.DataFrame:
    n = max(n_fact // 20, 1)
    base_ts = np.datetime64("2020-01-01", "ms").astype("datetime64[ms]").astype(np.int64)
    rng = np.arange(n, dtype=np.int64)
    return pd.DataFrame({"ts": base_ts + rng * 120_000, "rate": 1 + (rng % 17) * 0.01})


def make_cross_dim_pandas() -> pd.DataFrame:
    return pd.DataFrame({"kid": np.arange(5, dtype=np.int32), "label": [f"L{i}" for i in range(5)]})


def make_fact_polars(n: int, seed: int = 42) -> pl.DataFrame:
    t0 = time.perf_counter()
    user_mod = max(n // 10, 1)
    rng = pl.arange(0, n, eager=True, dtype=pl.Int32)
    city_col = pl.Series("city", CITIES, dtype=pl.Categorical).gather(rng % len(CITIES))
    category_col = pl.Series("category", CATEGORIES, dtype=pl.Categorical).gather(rng % 50)
    base_ts = int(np.datetime64("2020-01-01", "ms").astype("datetime64[ms]").astype(np.int64))
    tags = pl.Series("tags", [TAG_VARIANTS[i % len(TAG_VARIANTS)] for i in range(n)])
    payload = pl.Series("payload", [PAYLOAD_VARIANTS[i % len(PAYLOAD_VARIANTS)] for i in range(n)])
    df = pl.DataFrame(
        {
            "id": rng,
            "user_id": (rng.cast(pl.Int64) * 7 + seed) % user_mod,
            "city": city_col,
            "category": category_col,
            "age": 18 + (rng % 50),
            "salary": (40_000 + (rng % 1000) * 10 + (rng % 17)).cast(pl.Float64),
            "ts": (rng.cast(pl.Int64) * 60_000 + base_ts),
            "tags": tags,
            "payload": payload,
        }
    )
    print(f"    polars fact n={n:,} in {(time.perf_counter() - t0) * 1000:.0f} ms", flush=True)
    return df


def make_dim_polars(n_fact: int) -> pl.DataFrame:
    n = max(n_fact // 10, 1)
    rng = pl.arange(0, n, eager=True, dtype=pl.Int32)
    segment_col = pl.Series("segment", SEGMENTS, dtype=pl.Categorical).gather(rng % 20)
    return pl.DataFrame(
        {
            "user_id": rng,
            "segment": segment_col,
            "score": (rng.cast(pl.Int64) * 13) % 100,
        }
    )


def make_asof_dim_polars(n_fact: int) -> pl.DataFrame:
    n = max(n_fact // 20, 1)
    base_ts = int(np.datetime64("2020-01-01", "ms").astype("datetime64[ms]").astype(np.int64))
    rng = pl.arange(0, n, eager=True, dtype=pl.Int64)
    return pl.DataFrame({"ts": rng * 120_000 + base_ts, "rate": 1 + (rng % 17) * 0.01})


def make_cross_dim_polars() -> pl.DataFrame:
    return pl.DataFrame({"kid": pl.arange(0, 5, eager=True, dtype=pl.Int32), "label": [f"L{i}" for i in range(5)]})


def timed(fn: Callable[[], Any], runs: int) -> tuple[float, Any]:
    samples: list[float] = []
    result: Any = None
    result = fn()
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples), result


def runs_for(n: int, base: int) -> int:
    if n >= 50_000_000:
        return min(base, 2)
    if n >= 10_000_000:
        return min(base, 3)
    return base


def parse_ops(raw: str | None) -> set[str]:
    if not raw or not raw.strip():
        return set(ALL_OPS)
    chosen = {x.strip() for x in raw.split(",") if x.strip()}
    unknown = chosen - set(ALL_OPS)
    if unknown:
        raise SystemExit(f"Unknown BENCH_OPS: {sorted(unknown)}; allowed={ALL_OPS}")
    return chosen


def want(ops: set[str], name: str, n: int) -> bool:
    if name not in ops:
        return False
    if name in {"describe", "melt"} and n > 2_000_000:
        return False
    if name == "rolling" and n > 10_000_000:
        return False
    if name in {"explode", "unnest", "transpose", "to_csv", "cross_join", "map_elements"} and n > 500_000:
        return False
    if name in {"interpolate", "profile"} and n > 2_000_000:
        return False
    return True


def bench_pandas(n: int, runs: int, ops: set[str]) -> list[Timing]:
    fact = make_fact_pandas(n)
    dim = make_dim_pandas(n)
    asof_dim = make_asof_dim_pandas(n)
    cross_dim = make_cross_dim_pandas()
    out: list[Timing] = []
    r = runs_for(n, runs)
    city_str = fact["city"].astype(str)

    if want(ops, "filter", n):
        ms, _ = timed(lambda: fact[(fact["age"] > 30) & (fact["salary"] > 45_000)].copy(), r)
        out.append(Timing("pandas", n, "filter", ms))

    if want(ops, "groupby_agg", n):
        ms, _ = timed(
            lambda: fact.groupby("city", observed=True, as_index=False).agg(
                salary_mean=("salary", "mean"), age_count=("age", "count")
            ),
            r,
        )
        out.append(Timing("pandas", n, "groupby_agg", ms))

    if want(ops, "groupby_multi", n):
        ms, _ = timed(
            lambda: fact.groupby(["city", "category"], observed=True, as_index=False).agg(
                salary_mean=("salary", "mean"), n=("id", "count")
            ),
            r,
        )
        out.append(Timing("pandas", n, "groupby_multi", ms))

    if want(ops, "sort", n):
        ms, _ = timed(lambda: fact.sort_values("salary", ascending=False).head(1000), r)
        out.append(Timing("pandas", n, "sort", ms))

    if want(ops, "join", n):
        ms, joined = timed(lambda: fact.merge(dim, on="user_id", how="inner"), r)
        out.append(Timing("pandas", n, "join", ms, {"rows": int(len(joined))}))

    if want(ops, "pipeline", n):
        def pipeline() -> pd.DataFrame:
            return (
                fact[(fact["age"] > 25) & (fact["salary"] > 42_000)]
                .groupby("city", observed=True, as_index=False)
                .agg(salary_mean=("salary", "mean"), n=("id", "count"))
                .sort_values("salary_mean", ascending=False)
            )

        ms, _ = timed(pipeline, r)
        out.append(Timing("pandas", n, "pipeline", ms))

    if want(ops, "select", n):
        ms, _ = timed(lambda: fact[["id", "city", "salary"]].copy(), r)
        out.append(Timing("pandas", n, "select", ms))

    if want(ops, "with_column", n):
        ms, _ = timed(lambda: fact.assign(bonus=fact["salary"] * 1.1), r)
        out.append(Timing("pandas", n, "with_column", ms))

    if want(ops, "unique", n):
        ms, res = timed(lambda: fact.drop_duplicates(subset=["user_id"], keep="first"), r)
        out.append(Timing("pandas", n, "unique", ms, {"rows": int(len(res))}))

    if want(ops, "value_counts", n):
        ms, _ = timed(lambda: fact["city"].value_counts(), r)
        out.append(Timing("pandas", n, "value_counts", ms))

    if want(ops, "rolling", n):
        ms, _ = timed(lambda: fact["salary"].rolling(32, min_periods=1).mean(), r)
        out.append(Timing("pandas", n, "rolling", ms))

    if want(ops, "describe", n):
        ms, _ = timed(lambda: fact.describe(include="all"), r)
        out.append(Timing("pandas", n, "describe", ms))

    if want(ops, "melt", n):
        ms, res = timed(lambda: fact.melt(id_vars=["id", "city"], value_vars=["age", "salary"]), r)
        out.append(Timing("pandas", n, "melt", ms, {"rows": int(len(res))}))

    if want(ops, "with_columns", n):
        ms, _ = timed(lambda: fact.assign(bonus=fact["salary"] * 1.1, age1=fact["age"] + 1), r)
        out.append(Timing("pandas", n, "with_columns", ms))

    if want(ops, "str_contains", n):
        ms, _ = timed(lambda: fact[city_str.str.contains("a", regex=False)].copy(), r)
        out.append(Timing("pandas", n, "str_contains", ms))

    if want(ops, "str_lower", n):
        ms, _ = timed(lambda: fact.assign(city_lo=city_str.str.lower()), r)
        out.append(Timing("pandas", n, "str_lower", ms))

    if want(ops, "dt_parts", n):
        ts = pd.to_datetime(fact["ts"], unit="ms", utc=True)
        ms, _ = timed(lambda: fact.assign(y=ts.dt.year, m=ts.dt.month), r)
        out.append(Timing("pandas", n, "dt_parts", ms))

    if want(ops, "when_then", n):
        ms, _ = timed(
            lambda: fact.assign(
                band=np.where(fact["age"] < 30, "young", np.where(fact["age"] < 50, "mid", "senior"))
            ),
            r,
        )
        out.append(Timing("pandas", n, "when_then", ms))

    if want(ops, "is_in", n):
        ms, _ = timed(lambda: fact[fact["age"].isin([25, 30, 35, 40, 45])].copy(), r)
        out.append(Timing("pandas", n, "is_in", ms))

    if want(ops, "is_between", n):
        ms, _ = timed(lambda: fact[fact["salary"].between(42_000, 48_000)].copy(), r)
        out.append(Timing("pandas", n, "is_between", ms))

    if want(ops, "groupby_stats", n):
        ms, _ = timed(
            lambda: fact.groupby("city", observed=True, as_index=False).agg(
                std=("salary", "std"), med=("salary", "median"), q75=("salary", lambda s: s.quantile(0.75))
            ),
            r,
        )
        out.append(Timing("pandas", n, "groupby_stats", ms))

    if want(ops, "tail", n):
        ms, _ = timed(lambda: fact.tail(1000).copy(), r)
        out.append(Timing("pandas", n, "tail", ms))

    if want(ops, "sample", n):
        k = min(10_000, n)
        ms, _ = timed(lambda: fact.sample(n=k, random_state=7), r)
        out.append(Timing("pandas", n, "sample", ms))

    if want(ops, "explode", n):
        def explode() -> pd.DataFrame:
            tmp = fact.copy()
            tmp["tags"] = tmp["tags"].map(json.loads)
            return tmp.explode("tags")

        ms, res = timed(explode, r)
        out.append(Timing("pandas", n, "explode", ms, {"rows": int(len(res))}))

    if want(ops, "semi_join", n):
        ms, res = timed(lambda: fact.merge(dim[["user_id"]], on="user_id", how="inner"), r)
        out.append(Timing("pandas", n, "semi_join", ms, {"rows": int(len(res))}))

    if want(ops, "anti_join", n):
        ms, res = timed(
            lambda: fact.merge(dim[["user_id"]], on="user_id", how="left", indicator=True)
            .query('_merge == "left_only"')
            .drop(columns="_merge"),
            r,
        )
        out.append(Timing("pandas", n, "anti_join", ms, {"rows": int(len(res))}))

    if want(ops, "cross_join", n):
        ms, res = timed(lambda: fact.merge(cross_dim, how="cross"), r)
        out.append(Timing("pandas", n, "cross_join", ms, {"rows": int(len(res))}))

    if want(ops, "join_asof", n):
        left = fact.sort_values("ts")
        right = asof_dim.sort_values("ts")
        ms, res = timed(lambda: pd.merge_asof(left, right, on="ts", direction="backward"), r)
        out.append(Timing("pandas", n, "join_asof", ms, {"rows": int(len(res))}))

    if want(ops, "shift_diff", n):
        ms, _ = timed(lambda: fact.assign(s=fact["salary"].shift(1), d=fact["salary"].diff(1)), r)
        out.append(Timing("pandas", n, "shift_diff", ms))

    if want(ops, "pct_change", n):
        ms, _ = timed(lambda: fact.assign(pct=fact["salary"].pct_change(1)), r)
        out.append(Timing("pandas", n, "pct_change", ms))

    if want(ops, "expanding", n):
        ms, _ = timed(lambda: fact.assign(csum=fact["salary"].expanding().sum()), r)
        out.append(Timing("pandas", n, "expanding", ms))

    if want(ops, "interpolate", n):
        m = min(n, 200_000)
        a = pd.Series(np.arange(m, dtype=np.float64))
        a[a.index % 17 == 0] = np.nan
        ms, _ = timed(lambda: a.interpolate(method="linear"), r)
        out.append(Timing("pandas", n, "interpolate", ms))

    if want(ops, "unnest", n):
        def unnest() -> pd.DataFrame:
            parsed = fact["payload"].map(json.loads)
            flat = pd.json_normalize(parsed)
            return pd.concat([fact.drop(columns=["payload"]), flat], axis=1)

        ms, _ = timed(unnest, r)
        out.append(Timing("pandas", n, "unnest", ms))

    if want(ops, "transpose", n):
        small = fact.head(20).set_index("id")
        ms, _ = timed(lambda: small.T, r)
        out.append(Timing("pandas", n, "transpose", ms))

    if want(ops, "map_elements", n):
        ms, _ = timed(lambda: fact.assign(x2=fact["age"].map(lambda v: int(v) * 2)), r)
        out.append(Timing("pandas", n, "map_elements", ms))

    if want(ops, "to_csv", n):
        ms, text = timed(lambda: fact.to_csv(index=False), r)
        out.append(Timing("pandas", n, "to_csv", ms, {"chars": len(text)}))

    if want(ops, "profile", n):
        def profile() -> dict[str, Any]:
            return {
                "rows": len(fact),
                "nulls": fact.isna().sum().to_dict(),
                "nunique": {c: int(fact[c].nunique(dropna=True)) for c in fact.columns},
            }

        ms, _ = timed(profile, r)
        out.append(Timing("pandas", n, "profile", ms))

    return out


def bench_polars(n: int, runs: int, ops: set[str]) -> list[Timing]:
    fact = make_fact_polars(n)
    dim = make_dim_polars(n)
    asof_dim = make_asof_dim_polars(n)
    cross_dim = make_cross_dim_polars()
    out: list[Timing] = []
    r = runs_for(n, runs)

    if want(ops, "filter", n):
        ms, _ = timed(lambda: fact.filter((pl.col("age") > 30) & (pl.col("salary") > 45_000)), r)
        out.append(Timing("polars", n, "filter", ms))

    if want(ops, "groupby_agg", n):
        ms, _ = timed(
            lambda: fact.group_by("city").agg(
                pl.col("salary").mean().alias("salary_mean"),
                pl.col("age").count().alias("age_count"),
            ),
            r,
        )
        out.append(Timing("polars", n, "groupby_agg", ms))

    if want(ops, "groupby_multi", n):
        ms, _ = timed(
            lambda: fact.group_by(["city", "category"]).agg(
                pl.col("salary").mean().alias("salary_mean"),
                pl.col("id").count().alias("n"),
            ),
            r,
        )
        out.append(Timing("polars", n, "groupby_multi", ms))

    if want(ops, "sort", n):
        ms, _ = timed(lambda: fact.sort("salary", descending=True).head(1000), r)
        out.append(Timing("polars", n, "sort", ms))

    if want(ops, "join", n):
        ms, joined = timed(lambda: fact.join(dim, on="user_id", how="inner"), r)
        out.append(Timing("polars", n, "join", ms, {"rows": int(joined.height)}))

    if want(ops, "pipeline", n):
        def pipeline() -> pl.DataFrame:
            return (
                fact.lazy()
                .filter((pl.col("age") > 25) & (pl.col("salary") > 42_000))
                .group_by("city")
                .agg(
                    pl.col("salary").mean().alias("salary_mean"),
                    pl.col("id").count().alias("n"),
                )
                .sort("salary_mean", descending=True)
                .collect()
            )

        ms, _ = timed(pipeline, r)
        out.append(Timing("polars", n, "pipeline", ms))

    if want(ops, "select", n):
        ms, _ = timed(lambda: fact.select(["id", "city", "salary"]), r)
        out.append(Timing("polars", n, "select", ms))

    if want(ops, "with_column", n):
        ms, _ = timed(lambda: fact.with_columns((pl.col("salary") * 1.1).alias("bonus")), r)
        out.append(Timing("polars", n, "with_column", ms))

    if want(ops, "unique", n):
        ms, res = timed(lambda: fact.unique(subset=["user_id"], keep="first"), r)
        out.append(Timing("polars", n, "unique", ms, {"rows": int(res.height)}))

    if want(ops, "value_counts", n):
        ms, _ = timed(lambda: fact.get_column("city").value_counts(), r)
        out.append(Timing("polars", n, "value_counts", ms))

    if want(ops, "rolling", n):
        ms, _ = timed(lambda: fact.select(pl.col("salary").rolling_mean(window_size=32, min_samples=1)), r)
        out.append(Timing("polars", n, "rolling", ms))

    if want(ops, "describe", n):
        ms, _ = timed(lambda: fact.describe(), r)
        out.append(Timing("polars", n, "describe", ms))

    if want(ops, "melt", n):
        ms, res = timed(
            lambda: fact.unpivot(
                index=["id", "city"], on=["age", "salary"], variable_name="variable", value_name="value"
            ),
            r,
        )
        out.append(Timing("polars", n, "melt", ms, {"rows": int(res.height)}))

    if want(ops, "with_columns", n):
        ms, _ = timed(
            lambda: fact.with_columns(
                (pl.col("salary") * 1.1).alias("bonus"),
                (pl.col("age") + 1).alias("age1"),
            ),
            r,
        )
        out.append(Timing("polars", n, "with_columns", ms))

    if want(ops, "str_contains", n):
        ms, _ = timed(lambda: fact.filter(pl.col("city").cast(pl.Utf8).str.contains("a", literal=True)), r)
        out.append(Timing("polars", n, "str_contains", ms))

    if want(ops, "str_lower", n):
        ms, _ = timed(
            lambda: fact.with_columns(pl.col("city").cast(pl.Utf8).str.to_lowercase().alias("city_lo")),
            r,
        )
        out.append(Timing("polars", n, "str_lower", ms))

    if want(ops, "dt_parts", n):
        ms, _ = timed(
            lambda: fact.with_columns(
                pl.from_epoch(pl.col("ts"), time_unit="ms").dt.year().alias("y"),
                pl.from_epoch(pl.col("ts"), time_unit="ms").dt.month().alias("m"),
            ),
            r,
        )
        out.append(Timing("polars", n, "dt_parts", ms))

    if want(ops, "when_then", n):
        ms, _ = timed(
            lambda: fact.with_columns(
                pl.when(pl.col("age") < 30)
                .then(pl.lit("young"))
                .when(pl.col("age") < 50)
                .then(pl.lit("mid"))
                .otherwise(pl.lit("senior"))
                .alias("band")
            ),
            r,
        )
        out.append(Timing("polars", n, "when_then", ms))

    if want(ops, "is_in", n):
        ms, _ = timed(lambda: fact.filter(pl.col("age").is_in([25, 30, 35, 40, 45])), r)
        out.append(Timing("polars", n, "is_in", ms))

    if want(ops, "is_between", n):
        ms, _ = timed(lambda: fact.filter(pl.col("salary").is_between(42_000, 48_000)), r)
        out.append(Timing("polars", n, "is_between", ms))

    if want(ops, "groupby_stats", n):
        ms, _ = timed(
            lambda: fact.group_by("city").agg(
                pl.col("salary").std().alias("std"),
                pl.col("salary").median().alias("med"),
                pl.col("salary").quantile(0.75).alias("q75"),
            ),
            r,
        )
        out.append(Timing("polars", n, "groupby_stats", ms))

    if want(ops, "tail", n):
        ms, _ = timed(lambda: fact.tail(1000), r)
        out.append(Timing("polars", n, "tail", ms))

    if want(ops, "sample", n):
        k = min(10_000, n)
        ms, _ = timed(lambda: fact.sample(n=k, seed=7), r)
        out.append(Timing("polars", n, "sample", ms))

    if want(ops, "explode", n):
        ms, res = timed(
            lambda: fact.with_columns(pl.col("tags").str.json_decode(pl.List(pl.Utf8))).explode("tags"),
            r,
        )
        out.append(Timing("polars", n, "explode", ms, {"rows": int(res.height)}))

    if want(ops, "semi_join", n):
        ms, res = timed(lambda: fact.join(dim, on="user_id", how="semi"), r)
        out.append(Timing("polars", n, "semi_join", ms, {"rows": int(res.height)}))

    if want(ops, "anti_join", n):
        ms, res = timed(lambda: fact.join(dim, on="user_id", how="anti"), r)
        out.append(Timing("polars", n, "anti_join", ms, {"rows": int(res.height)}))

    if want(ops, "cross_join", n):
        ms, res = timed(lambda: fact.join(cross_dim, how="cross"), r)
        out.append(Timing("polars", n, "cross_join", ms, {"rows": int(res.height)}))

    if want(ops, "join_asof", n):
        ms, res = timed(
            lambda: fact.sort("ts").join_asof(asof_dim.sort("ts"), on="ts", strategy="backward"),
            r,
        )
        out.append(Timing("polars", n, "join_asof", ms, {"rows": int(res.height)}))

    if want(ops, "shift_diff", n):
        ms, _ = timed(
            lambda: fact.with_columns(
                pl.col("salary").shift(1).alias("s"),
                pl.col("salary").diff(1).alias("d"),
            ),
            r,
        )
        out.append(Timing("polars", n, "shift_diff", ms))

    if want(ops, "pct_change", n):
        ms, _ = timed(lambda: fact.with_columns(pl.col("salary").pct_change(1).alias("pct")), r)
        out.append(Timing("polars", n, "pct_change", ms))

    if want(ops, "expanding", n):
        ms, _ = timed(
            lambda: fact.with_columns(pl.col("salary").cum_sum().alias("csum")),
            r,
        )
        out.append(Timing("polars", n, "expanding", ms))

    if want(ops, "interpolate", n):
        m = min(n, 200_000)
        s = pl.Series("a", [None if i % 17 == 0 else float(i) for i in range(m)])
        df = pl.DataFrame({"a": s})
        ms, _ = timed(lambda: df.with_columns(pl.col("a").interpolate()), r)
        out.append(Timing("polars", n, "interpolate", ms))

    if want(ops, "unnest", n):
        payload_dtype = pl.Struct({"user": pl.Struct({"id": pl.Int64}), "n": pl.Int64})
        ms, _ = timed(
            lambda: fact.with_columns(pl.col("payload").str.json_decode(payload_dtype).alias("payload")).unnest(
                "payload"
            ),
            r,
        )
        out.append(Timing("polars", n, "unnest", ms))

    if want(ops, "transpose", n):
        small = fact.head(20).select(["id", "age", "salary"])
        ms, _ = timed(lambda: small.transpose(include_header=True, header_name="column"), r)
        out.append(Timing("polars", n, "transpose", ms))

    if want(ops, "map_elements", n):
        ms, _ = timed(
            lambda: fact.with_columns(pl.col("age").map_elements(lambda v: int(v) * 2, return_dtype=pl.Int64).alias("x2")),
            r,
        )
        out.append(Timing("polars", n, "map_elements", ms))

    if want(ops, "to_csv", n):
        ms, text = timed(lambda: fact.write_csv(), r)
        out.append(Timing("polars", n, "to_csv", ms, {"chars": len(text)}))

    if want(ops, "profile", n):
        def profile() -> pl.DataFrame:
            return fact.describe()

        ms, _ = timed(profile, r)
        out.append(Timing("polars", n, "profile", ms))

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="1000000")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", default="")
    parser.add_argument("--ops", default=os.environ.get("BENCH_OPS", ""))
    args = parser.parse_args()
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    ops = parse_ops(args.ops)

    results: list[dict[str, Any]] = []
    for n in sizes:
        print(f"[python] n={n:,} (runs={runs_for(n, args.runs)}) ops={sorted(ops)}", flush=True)
        for t in bench_pandas(n, args.runs, ops) + bench_polars(n, args.runs, ops):
            row = asdict(t)
            results.append(row)
            print(f"  {t.library:7} {t.op:14} {t.ms:10.2f} ms", flush=True)

    payload = {
        "engine": "python",
        "pandas": pd.__version__,
        "polars": pl.__version__,
        "runs": args.runs,
        "ops": sorted(ops),
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
