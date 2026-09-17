# columna vs Arquero / DuckDB-Wasm / Polars / DuckDB — same data, same operations, checked results

Generated 2026-09-17 · Node v24.19.0 · win32 x64 · 2 000 000 rows × 8 columns (2 int, 4 float, 2 low-cardinality text), CSV 107 MB.
Each library in its own process. **Cold total** = sum of timed operation durations on the first run in that process (module import / DuckDB-Wasm init are **outside** these timers). **Warm** = median of 2 further runs. **Peak RSS** sampled every 20 ms (synchronous work can hide intermediate peaks). Times in ms.

## Summary

| Library | Version | Cold total | Warm total | Peak RSS | Notes |
|---|---|---:|---:|---:|---|
| columna | 0.1.0 | 1781 | 1737 | 1084 MB |  |
| arquero | 8.0.3 | 11274 | 10829 | 3525 MB | reads CSV from a string (fs.readFile + fromCSV); filter/derive/orderby/join/unique reified so the work is inside the timing; corr is JS pearson over column arrays |
| duckdb-wasm | 1.33.1-dev57.0 | 4154 | 4079 | 1852 MB | single-threaded WebAssembly build (eh), blocking Node bindings; the browser deployment target |
| polars | 0.26.1 | 2241 | 2370 | 4330 MB | eager API; native multi-threaded core |
| polars-lazy | 0.26.1 | 3509 | 3980 | 2550 MB | lazy API: each op = scanCSV + op + collect (read is inside every timing; query optimiser + streaming) |
| duckdb | 1.4.4 | 1004 | 1190 | 1695 MB | native addon, multi-threaded |

## Warm time by operation (ms)

Rows are operations; columns are libraries. Lower is better. `polars-lazy` includes CSV scan inside every op (its `read` cell is 0 by design).

| Operation | columna | arquero | duckdb-wasm | polars | polars-lazy | duckdb |
|---|---:|---:|---:|---:|---:|---:|
| read CSV | 280 | 1831 | 875 | 50 | 0 | 261 |
| filter (`age>30 ∧ salary>45k`) | 30 | 182 | 160 | 40 | 218 | 325 |
| select 4 columns | 0 | 0 | 108 | 0 | 336 | 21 |
| drop (`z`,`y`) | 0 | 0 | 168 | 0 | 109 | 24 |
| rename city→town | 0 | 0 | 188 | 0 | 100 | 26 |
| withColumn (`xy = x+y`) | 6 | 19 | 212 | 9 | 110 | 29 |
| groupBy city + agg | 31 | 170 | 34 | 48 | 148 | 6 |
| sort salary desc | 182 | 1906 | 526 | 125 | 241 | 100 |
| sort city asc, salary desc | 233 | 2707 | 655 | 1159 | 327 | 118 |
| unique (`age`,`city`) | 182 | 297 | 60 | 76 | 183 | 16 |
| head 1000 | 0 | 0 | 2 | 0 | 3 | 2 |
| tail 1000 | 0 | 0 | 4 | 0 | 109 | 12 |
| sample n=1000 | 1 | 13 | 2 | 1 | 105 | 5 |
| inner join regions on city | 45 | 388 | 38 | 41 | 210 | 5 |
| left join regions on city | 59 | 481 | 35 | 27 | 162 | 5 |
| semi join regions on city | 43 | 237 | 34 | 47 | 168 | 4 |
| anti join regions on city | 22 | 130 | 34 | 32 | 153 | 4 |
| melt x,y (id=id,city) | 16 | 749 | 226 | 61 | 126 | 65 |
| pivot segment × mean(salary) | 275 | 330 | 50 | 97 | 214 | 10 |
| valueCounts(city) | 8 | 141 | 28 | 37 | 147 | 5 |
| describe (age mean) | 265 | 88 | 2 | 83 | 237 | 1 |
| corr(x, y) | 38 | 67 | 10 | 382 | 505 | 3 |
| filter age>30 → groupBy city sum(salary) | 30 | 132 | 29 | 59 | 173 | 6 |
| writeCsv (filtered) | 44 | 1304 | 600 | 46 | 46 | 141 |

## Cold time by operation (ms)

| Operation | columna | arquero | duckdb-wasm | polars | polars-lazy | duckdb |
|---|---:|---:|---:|---:|---:|---:|
| read CSV | 259 | 1664 | 826 | 48 | 0 | 274 |
| filter (`age>30 ∧ salary>45k`) | 28 | 190 | 171 | 33 | 129 | 211 |
| select 4 columns | 0 | 0 | 128 | 0 | 87 | 21 |
| drop (`z`,`y`) | 0 | 0 | 169 | 0 | 99 | 19 |
| rename city→town | 0 | 0 | 198 | 0 | 96 | 23 |
| withColumn (`xy = x+y`) | 6 | 19 | 216 | 9 | 108 | 26 |
| groupBy city + agg | 31 | 133 | 39 | 50 | 144 | 5 |
| sort salary desc | 163 | 1998 | 546 | 109 | 238 | 88 |
| sort city asc, salary desc | 242 | 3166 | 661 | 1093 | 348 | 107 |
| unique (`age`,`city`) | 188 | 291 | 63 | 58 | 191 | 13 |
| head 1000 | 0 | 0 | 7 | 0 | 2 | 1 |
| tail 1000 | 0 | 0 | 6 | 0 | 96 | 11 |
| sample n=1000 | 1 | 15 | 3 | 2 | 105 | 4 |
| inner join regions on city | 42 | 398 | 40 | 40 | 153 | 3 |
| left join regions on city | 55 | 359 | 36 | 23 | 141 | 4 |
| semi join regions on city | 41 | 230 | 35 | 37 | 154 | 4 |
| anti join regions on city | 21 | 117 | 36 | 27 | 141 | 4 |
| melt x,y (id=id,city) | 14 | 809 | 230 | 37 | 133 | 28 |
| pivot segment × mean(salary) | 270 | 267 | 51 | 85 | 178 | 6 |
| valueCounts(city) | 6 | 104 | 30 | 35 | 131 | 4 |
| describe (age mean) | 265 | 84 | 4 | 91 | 189 | 1 |
| corr(x, y) | 76 | 63 | 11 | 367 | 475 | 2 |
| filter age>30 → groupBy city sum(salary) | 28 | 132 | 31 | 48 | 123 | 3 |
| writeCsv (filtered) | 45 | 1235 | 616 | 47 | 47 | 139 |

## Result validation

Reference (columna): filter 1240671, select 4, drop 6, rename 1, Σxy 199965448.286, groupBy 100355262.877, sort 139999.95/20000.08, sortMulti first salary 139999.95, unique 300, head Σ 499500, tail Σ 1999499500, sample rows 1000, join 1600000/127980215810.788, left 2000000, semi 1600000, anti 400000, melt 4000000/199965448.286, pivot Σ 800113.544, valueCounts Σ 2000000, describe age μ 47.525, corr(x,y) 0, filter→groupBy Σ 125361576887.11.
- arquero: summary-match
- duckdb-wasm: summary-match
- polars: summary-match
- polars-lazy: summary-match
- duckdb: summary-match

## How to read this

- Wall-clock of a single process on one machine; the multi-threaded engines (Polars, native DuckDB) use every core, columna / Arquero / DuckDB-Wasm one.
- polars-lazy pays the CSV scan inside every operation (that is the point of a streaming optimiser); compare its per-op numbers with `read + op` of the eager rows.
- Peak RSS includes the runtime itself (a WebAssembly heap, Polars’ thread pool) and the CSV read; it is the number to budget, not the column bytes. Same-process 20 ms sampling underestimates peaks during long synchronous stretches.
- SQL engines (DuckDB / DuckDB-Wasm) time **aggregate-over-join** (`count`/`sum` on a join), not full join materialization — do not compare that join cell directly to columna/Arquero/Polars full join output.
- **unique** keeps the first row per (`age`,`city`) key (~300 groups from 2M rows) — a hash-dedup stress with a tiny result.
- **head** / **tail** are first/last 1 000 rows in storage order (head id Σ = 499500 on this fixture).
- **sample** checks only row count (RNG differs across libraries). **corr** is Pearson of `x` and `y`.
- **summary-match** checks aggregates (row counts, sums, sort extremes) with tolerance — not byte-identical tables.
- columna numbers come from `collect()` on the CPU engine; `collectWithReport()` confirms no other backend was involved.
