# columna vs Arquero / DuckDB-Wasm / Polars / DuckDB — same data, same operations, checked results

Generated 2026-09-17 · Node v24.19.0 · win32 x64 · 2 000 000 rows × 8 columns (2 int, 4 float, 2 low-cardinality text), CSV 107 MB.
Each library in its own process: **cold** = module load + first run; **warm** = median of 2 further runs in the same process; peak RSS sampled every 20 ms. Times in ms.

| Library | Version | Cold total | Warm total | read (warm) | filter (warm) | groupBy (warm) | sort (warm) | join (warm) | writeCsv (warm) | Peak RSS | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| columna | 0.1.0 | 4191 | 4254 | 3221 | 34 | 32 | 199 | 43 | 726 | 728 MB |  |
| arquero | 8.0.3 | 5266 | 5121 | 1777 | 160 | 171 | 1512 | 408 | 1099 | 2211 MB | reads CSV from a string (fs.readFile + fromCSV); filter/orderby/join reified so the work is inside the timing |
| duckdb-wasm | 1.33.1-dev57.0 | 2168 | 2173 | 765 | 159 | 33 | 533 | 38 | 644 | 1072 MB | single-threaded WebAssembly build (eh), blocking Node bindings; the browser deployment target |
| polars | 0.26.1 | 375 | 310 | 45 | 33 | 44 | 104 | 36 | 52 | 1932 MB | eager API; native multi-threaded core |
| polars-lazy | 0.26.1 | 733 | 749 | 0 | 150 | 192 | 225 | 158 | 44 | 848 MB | lazy API: each op = scanCSV + op + collect (read is inside every timing; query optimiser + streaming) |
| duckdb | 1.4.4 | 686 | 701 | 247 | 228 | 6 | 89 | 5 | 127 | 753 MB | native addon, multi-threaded |

## Result validation

Reference (columna): filter rows 1240671, groupBy checksum 100355262.877, sort first/last 139999.95/20000.08, join rows 1600000, join checksum 127980215810.788.
- arquero: identical
- duckdb-wasm: identical
- polars: identical
- polars-lazy: identical
- duckdb: identical

## How to read this

- Wall-clock of a single process on one machine; the multi-threaded engines (Polars, native DuckDB) use every core, columna / Arquero / DuckDB-Wasm one.
- polars-lazy pays the CSV scan inside every operation (that is the point of a streaming optimiser); compare its per-op numbers with `read + op` of the eager rows.
- Peak RSS includes the runtime itself (a WebAssembly heap, Polars’ thread pool) and the CSV read; it is the number to budget, not the column bytes.
- columna numbers come from `collect()` on the CPU engine; `collectWithReport()` confirms no other backend was involved.
