# columna vs Arquero / DuckDB-Wasm / Polars / DuckDB — same data, same operations, checked results

Generated 2026-09-17 · Node v24.19.0 · win32 x64 · 2 000 000 rows × 8 columns (2 int, 4 float, 2 low-cardinality text), CSV 107 MB.
Each library in its own process. **Cold total** = sum of timed operation durations on the first run in that process (module import / DuckDB-Wasm init are **outside** these timers). **Warm** = median of 2 further runs. **Peak RSS** sampled every 20 ms (synchronous work can hide intermediate peaks). Times in ms.

## Summary

| Library | Version | Cold total | Warm total | Peak RSS | Notes |
|---|---|---:|---:|---:|---|
| columna | 0.2.1 | 1728 | 1694 | 1102 MB |  |
| arquero | 8.0.3 | 9199 | 10362 | 3546 MB | reads CSV from a string (fs.readFile + fromCSV); filter/derive/orderby/join/unique reified so the work is inside the timing; corr is JS pearson over column arrays |
| duckdb-wasm | 1.33.1-dev57.0 | 4104 | 3962 | 1847 MB | single-threaded WebAssembly build (eh), blocking Node bindings; the browser deployment target |
| polars | 0.26.1 | 2203 | 2303 | 4327 MB | eager API; native multi-threaded core |
| polars-lazy | 0.26.1 | 3235 | 3828 | 2532 MB | lazy API: each op = scanCSV + op + collect (read is inside every timing; query optimiser + streaming) |
| duckdb | 1.4.4 | 968 | 1118 | 1706 MB | native addon, multi-threaded |

## Warm time by operation (ms)

Rows are operations; columns are libraries. Lower is better. `polars-lazy` includes CSV scan inside every op (its `read` cell is 0 by design).

| Operation | columna | arquero | duckdb-wasm | polars | polars-lazy | duckdb |
|---|---:|---:|---:|---:|---:|---:|
| read CSV | 269 | 1752 | 790 | 47 | 0 | 252 |
| filter (`age>30 ∧ salary>45k`) | 35 | 169 | 149 | 40 | 201 | 296 |
| select 4 columns | 0 | 0 | 106 | 0 | 313 | 19 |
| drop (`z`,`y`) | 0 | 0 | 166 | 0 | 106 | 23 |
| rename city→town | 0 | 0 | 192 | 0 | 95 | 29 |
| withColumn (`xy = x+y`) | 6 | 18 | 212 | 8 | 113 | 31 |
| groupBy city + agg | 30 | 230 | 33 | 41 | 145 | 7 |
| sort salary desc | 175 | 1509 | 530 | 119 | 206 | 96 |
| sort city asc, salary desc | 236 | 2743 | 640 | 1140 | 301 | 109 |
| unique (`age`,`city`) | 180 | 291 | 60 | 53 | 193 | 15 |
| head 1000 | 0 | 0 | 3 | 0 | 2 | 1 |
| tail 1000 | 0 | 0 | 5 | 0 | 95 | 11 |
| sample n=1000 | 1 | 13 | 2 | 2 | 99 | 4 |
| inner join regions on city | 47 | 234 | 36 | 39 | 192 | 5 |
| left join regions on city | 62 | 357 | 35 | 23 | 155 | 4 |
| semi join regions on city | 43 | 215 | 33 | 37 | 161 | 3 |
| anti join regions on city | 21 | 123 | 35 | 30 | 148 | 4 |
| melt x,y (id=id,city) | 15 | 729 | 230 | 43 | 124 | 59 |
| pivot segment × mean(salary) | 257 | 704 | 49 | 106 | 186 | 8 |
| valueCounts(city) | 9 | 112 | 28 | 41 | 142 | 6 |
| describe (age mean) | 257 | 85 | 3 | 82 | 205 | 1 |
| corr(x, y) | 37 | 61 | 10 | 399 | 468 | 3 |
| filter age>30 → groupBy city sum(salary) | 29 | 144 | 32 | 53 | 169 | 5 |
| writeCsv (filtered) | 44 | 1288 | 603 | 51 | 44 | 137 |

## Cold time by operation (ms)

| Operation | columna | arquero | duckdb-wasm | polars | polars-lazy | duckdb |
|---|---:|---:|---:|---:|---:|---:|
| read CSV | 248 | 1638 | 801 | 44 | 0 | 253 |
| filter (`age>30 ∧ salary>45k`) | 36 | 171 | 171 | 40 | 109 | 203 |
| select 4 columns | 0 | 0 | 127 | 1 | 78 | 20 |
| drop (`z`,`y`) | 0 | 0 | 167 | 0 | 86 | 19 |
| rename city→town | 0 | 0 | 192 | 0 | 105 | 22 |
| withColumn (`xy = x+y`) | 5 | 13 | 214 | 10 | 98 | 24 |
| groupBy city + agg | 31 | 127 | 38 | 44 | 126 | 4 |
| sort salary desc | 163 | 1037 | 556 | 100 | 211 | 97 |
| sort city asc, salary desc | 219 | 2502 | 655 | 1128 | 298 | 107 |
| unique (`age`,`city`) | 173 | 252 | 64 | 53 | 171 | 14 |
| head 1000 | 0 | 0 | 7 | 0 | 1 | 1 |
| tail 1000 | 0 | 0 | 6 | 0 | 84 | 11 |
| sample n=1000 | 1 | 13 | 4 | 1 | 94 | 4 |
| inner join regions on city | 42 | 233 | 39 | 32 | 150 | 5 |
| left join regions on city | 58 | 421 | 36 | 22 | 121 | 3 |
| semi join regions on city | 43 | 207 | 34 | 39 | 138 | 3 |
| anti join regions on city | 20 | 107 | 35 | 26 | 130 | 3 |
| melt x,y (id=id,city) | 15 | 668 | 225 | 30 | 116 | 28 |
| pivot segment × mean(salary) | 278 | 260 | 49 | 68 | 181 | 5 |
| valueCounts(city) | 6 | 115 | 28 | 36 | 116 | 3 |
| describe (age mean) | 259 | 85 | 3 | 80 | 195 | 1 |
| corr(x, y) | 58 | 63 | 10 | 363 | 457 | 2 |
| filter age>30 → groupBy city sum(salary) | 28 | 148 | 30 | 43 | 125 | 4 |
| writeCsv (filtered) | 43 | 1139 | 614 | 44 | 44 | 133 |

## Result validation

Reference (columna): filter 1240671, select 4, drop 6, rename 1, Σxy 199965448.286, groupBy 100355262.877, sort 139999.95/20000.08, sortMulti first salary 139999.95, unique 300, head Σ 499500, tail Σ 1999499500, sample rows 1000, join 1600000/127980215810.788, left 2000000, semi 1600000, anti 400000, melt 4000000/199965448.286, pivot Σ 800113.544, valueCounts Σ 2000000, describe age μ 47.525, corr(x,y) 0, filter→groupBy Σ 125361576887.11.
- arquero: summary-match
- duckdb-wasm: summary-match
- polars: summary-match
- polars-lazy: summary-match
- duckdb: summary-match

## How to read this

- Wall-clock of a single process on one machine; Polars / native DuckDB use every core end-to-end. columna engages its worker pool / native Rayon kernels on heavy ops above per-op thresholds (see README → Engines); Arquero and DuckDB-Wasm stay single-threaded.
- polars-lazy pays the CSV scan inside every operation (that is the point of a streaming optimiser); compare its per-op numbers with `read + op` of the eager rows.
- Peak RSS includes the runtime itself (a WebAssembly heap, Polars’ thread pool) and the CSV read; it is the number to budget, not the column bytes. Same-process 20 ms sampling underestimates peaks during long synchronous stretches.
- SQL engines (DuckDB / DuckDB-Wasm) time **aggregate-over-join** (`count`/`sum` on a join), not full join materialization — do not compare that join cell directly to columna/Arquero/Polars full join output.
- **unique** keeps the first row per (`age`,`city`) key (~300 groups from 2M rows) — a hash-dedup stress with a tiny result.
- **head** / **tail** are first/last 1 000 rows in storage order (head id Σ = 499500 on this fixture).
- **sample** checks only row count (RNG differs across libraries). **corr** is Pearson of `x` and `y`.
- **summary-match** checks aggregates (row counts, sums, sort extremes) with tolerance — not byte-identical tables.
- columna numbers come from `collect()` on the CPU engine; `collectWithReport()` confirms no other backend was involved.
