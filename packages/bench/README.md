# @columna/bench

Comparative benchmarks: **pandas vs polars vs columna**.

## Ops

### Classic
| Op | Description |
|---|---|
| `filter` | `age > 30 && salary > 45000` |
| `groupby_agg` | group by `city` → mean(salary), count(age) |
| `groupby_multi` | group by `city, category` |
| `sort` | sort by salary desc, head 1000 |
| `join` | inner join fact ↔ dim on `user_id` |
| `pipeline` | filter → groupby → sort (polars lazy) |
| `select` | project `id, city, salary` |
| `with_column` | `bonus = salary * 1.1` |
| `unique` | dedupe on `user_id` |
| `value_counts` | counts for `city` |
| `rolling` | rolling mean window 32 on `salary` |
| `describe` | summary stats (skip if n > 2M) |
| `melt` | unpivot age/salary (skip if n > 2M) |

### API parity / new
| Op | Description | Skip if |
|---|---|---|
| `with_columns` | multi-column assign | |
| `str_contains` / `str_lower` | string namespace | |
| `dt_parts` | year/month from epoch-ms `ts` | |
| `when_then` | conditional column | |
| `is_in` / `is_between` | membership / range filter | |
| `groupby_stats` | std / median / quantile(0.75) | |
| `tail` / `sample` | last N / sample 10k | |
| `explode` | JSON-array `tags` | n > 500k |
| `semi_join` / `anti_join` | join variants | |
| `cross_join` | × 5-row dim | n > 500k |
| `join_asof` | backward asof on `ts` | |
| `shift_diff` / `pct_change` | timeseries helpers | |
| `expanding` | expanding sum | |
| `interpolate` | linear fill (≤200k series) | n > 2M |
| `unnest` | JSON object `payload` | n > 500k |
| `transpose` | head(20) transpose | n > 500k |
| `map_elements` | per-element UDF | n > 500k |
| `to_csv` | serialize CSV | n > 500k |
| `profile` | nulls / approx unique | n > 2M |

Default size: **1M** rows (heavy reshape ops auto-skipped). Median of 3 runs after 1 warmup.

## Run

```bash
py -3 -m pip install pandas polars numpy
pnpm build
pnpm bench:compare
```

Env overrides:

```bash
# PowerShell — full suite at 200k (includes explode/unnest/cross/…)
$env:BENCH_SIZES='200000'
$env:BENCH_RUNS='3'
$env:BENCH_ENGINES='cpu'
pnpm bench:compare

# Focus on new ops
$env:BENCH_OPS='str_contains,str_lower,dt_parts,when_then,is_in,is_between,groupby_stats,sample,tail,semi_join,anti_join,join_asof,shift_diff,pct_change,expanding,with_columns'
$env:BENCH_SIZES='1000000'
pnpm bench:compare
```

Outputs: [`results/compare-latest.json`](results/compare-latest.json)

## Separate runners

```bash
pnpm --filter @columna/bench compare:python
pnpm --filter @columna/bench compare:columna
```
