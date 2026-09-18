<p align="center">
  <img src="assets/columna-mark.png" alt="columna" width="96" />
</p>

# Where columna fits — and where it does not

"Fluent API, runs in JS" describes Arquero as well as columna, and DuckDB-Wasm and Polars both have JavaScript
interfaces. This page states the actual trade-offs, with numbers from `pnpm bench:compare:js`
([comparison-js.md](comparison-js.md): same 2M-row file, 24 operations, results cross-checked to be identical).
For a scale where worker / native thresholds actually fire, see [comparison-js-10m.md](comparison-js-10m.md)
(`E2E_ROWS=10000000`, 10M rows · 539 MB CSV).

## The alternatives, by scenario

| Scenario | Strongest option | Where columna stands |
|---|---|---|
| Table transformations directly in JS (browser or Node), no native binary, no WASM heap | **Arquero** or **columna** | Same operation set. On 2M rows columna is typically several× faster per transform (filter 35 vs 169 ms, groupBy 30 vs 230, sort 175 vs 1509, sortMulti 236 vs 2743, unique 180 vs 291, join 47 vs 234) and holds ~3× less memory (1102 vs 3546 MB peak) because columns are typed arrays rather than JS arrays; Arquero has a larger ecosystem (Vega, Observable). |
| SQL analytics and Parquet in the browser | **DuckDB-Wasm** | It is a real query engine: optimiser, SQL, Parquet/Arrow natively, CSV read is in the same ballpark here (~0.8 vs 0.3 s with columna's optional native Rayon path). Cost: a 5–10 MB WASM bundle, a separate heap (~1.8 GB peak here), SQL as the interface. columna's per-op speed is comparable on filter/groupBy/join with a ~150 kB core and no SQL. |
| Heavy tabular processing outside the browser | **Polars** (`nodejs-polars`) or **DuckDB** (native) | Native multi-threaded cores still win on some ops, but on this 24-op suite columna's warm total (~1.7 s) is competitive with Polars eager (~2.3 s) when `@columna/native` is present; DuckDB remains fastest (~1.1 s). At **10M rows** ([comparison-js-10m.md](comparison-js-10m.md)) columna's warm total (~10 s) beats Polars eager (~19 s) and DuckDB (~12 s) on the same op mix — worker-pool sort and native kernels engage above their thresholds. If the workload is server-side and a full SQL/streaming optimiser is required, prefer Polars/DuckDB. columna's distinct role is the statistics layer (below). |

Measured on one Windows x64 machine, Node 24; single process per library; multi-threaded engines use every core.
columna's worker pool / native kernels engage on the parallel ops listed above; cold numbers (module load + first run)
are in the linked report.

## What columna actually offers that the others do not

1. **Minitab-level statistics in the same package.** One-sample to MANOVA, DOE, reliability, SPC, capability,
   time series, regression with full diagnostics — ~250 functions, each checked against scipy / numpy / NIST
   fixtures, property tests and seeded Monte-Carlo ([advanced-roadmap.md](advanced-roadmap.md),
   [minitab-coverage.md](minitab-coverage.md)). Arquero, DuckDB and Polars stop at descriptive aggregates;
   the JS statistics libraries stop at t-tests. This is the reason to pick columna.
2. **Exactness as a contract.** Type inference widens instead of coercing (a late 2³¹ or "n/a" changes the
   column, never the value); `cast` refuses overflow; GPU kernels compare integers bit-exactly or decline;
   `engine(kind, { strict: true })` and `collectWithReport()` say which backend actually ran each node.
3. **Small and dependency-free where it matters.** The engine (`@columna/arrow`, `@columna/runtime`) and
   `@columna/advanced` have no third-party runtime dependencies; CSV / JSON / Arrow IPC IO is built in.
   `@columna/core` / `columna` declare four IO packages — Parquet read/write (`hyparquet`,
   `hyparquet-compressors`, `hyparquet-writer`) and Excel (`xlsx`) — loaded lazily on first use; SQL / Kafka
   drivers are optional peers. The same code runs in Node and the browser.
5. **Apache Arrow IPC without a dependency.** `toArrowIpc()` / `fromArrowIpc()` write and read the real IPC
   stream / file formats (dictionary-encoded categories, timestamps, Utf8View on input), verified against
   `apache-arrow`, Polars and DuckDB-Wasm. DataFrames move to DuckDB / Polars / pyarrow and back as bytes.
6. **Cancellable, deadline-bounded execution.** `collect({ signal, timeoutMs })` checks between operators and
   yields to the event loop so a UI stays responsive and an abort lands within one operator.
4. **Explicit boundaries for server use.** IO policy (allowed hosts / directories, byte caps, timeouts,
   cancellation), formula-safe CSV export, prototype-safe row construction — documented in
   [SECURITY.md](../SECURITY.md).
5. **Rule-based rewrite with light join costing.** `collect` / `explain` / `executeCpu` run
   `optimizePlan`: merge consecutive filters; push predicates past simple projects, under `withColumn` /
   `drop` / `rename` / `sort`, and into join sides when column refs are local; push limit under project;
   fold nested projects/drops/renames; prune groupBy inputs; projection pushdown; then **inner-join
   reorder** using sample/NDV-backed `estimatePlanRows` (filter selectivity from column samples;
   inner ≈ `L·R/max(NDV)`; smaller side as hash build / right; greedy left-deep equi graph across
   differing key names; trailing project keeps output column order — execute fuses project→join `keep`).
   **Left / semi / anti never swap** (output schema is left-driven). `explain` prints `rows≈N` per node;
   `collectWithReport` may emit `optimized:joinReorder` and notes native join thresholds.
6. **A real multi-threaded engine, not just a fast single thread.** Heavy ops run on a reusable worker
   pool with SharedArrayBuffer-backed, zero-copy chunks: Node `worker_threads` and browser Web Workers
   share one `engine-worker` protocol (filter, sort, groupBy, unique, gather, dual-gt). `CpuBackend.execute`
   routes large sort / unique / filter / gather through workers above per-op thresholds; `@columna/native`
   adds Rayon kernels for multi-key sort, parallel unique, generic multi-column filter, hash-join build,
   single-key argsort, dense groupBy, and dense join probe/semi. Every parallel path falls back to a
   single-threaded kernel below its threshold or when workers / SharedArrayBuffer are unavailable, so
   results are identical with or without threads. `collectWithReport` records the kernel that ran
   (`workers:sort`, `native:joinBuildDenseI32`, `js:unique`, …).

## What columna does not have — do not read the fluent API as a promise of these

- **Not a full cost-based optimiser.** No histograms, no bushy join trees, no outer/right-join reorder.
  Polars-lazy and DuckDB still plan more aggressively; columna's rewrite is rule-based with sample/NDV
  cardinality heuristics only.
- **Not a morsel-driven parallel runtime.** Threads accelerate individual heavy ops (filter, sort,
  groupBy, unique, gather, joins) above per-op thresholds, but columna is not a pipelined,
  work-stealing scheduler like Polars/DuckDB: a plan still runs one operator at a time, and small ops
  stay single-threaded. The browser Web Worker pool needs cross-origin isolation (COOP/COEP) for
  SharedArrayBuffer; without it, the browser falls back to one thread. WASM / WebGPU remain narrow
  accelerations with CPU fallback (see README → Engines).
- **CSV reading without the native addon is the slower fused-JS path.** `@columna/native`'s Rayon parser
  brings the 107 MB compare-js load to ~280 ms; without it the fused single-thread scanner is markedly
  slower (see [comparison-js.md](comparison-js.md)). Both paths stay streaming and memory-bounded.
- **Arrow IPC is a copy, not shared memory.** `toArrowIpc()` serialises into a fresh buffer and `fromArrowIpc()`
  copies into columna's own columns; there is no zero-copy Arrow memory model behind `DataFrame` (columns are
  TypedArrays + validity bitmaps + dictionaries, close to Arrow but not Arrow buffers). Nested / decimal / binary
  Arrow types and LZ4 / ZSTD-compressed batches are refused, not approximated.
- **Cancellation is per operator, not per row.** A running kernel finishes before `collect({ signal })` rejects.
- **No SQL.** Everything is method chains and expressions.
- **Scale ceiling is process memory by default.** On Node you can set a soft budget (`MemoryPolicy.maxBytes`) so sort / unique / join / groupBy spill intermediates to disk instead of growing without bound; the browser has no spill path.

## Choosing

- Statistics beyond aggregates, in JavaScript, with defensible numbers → columna.
- Interactive in-browser transforms of up to a few million rows, minimal bundle, no SQL → columna or Arquero;
  columna if speed and memory matter, Arquero if the Vega / Observable ecosystem matters.
- Browser analytics over Parquet, SQL, or tens of millions of rows → DuckDB-Wasm.
- Server-side bulk processing → Polars or DuckDB; put columna's `advanced` on the result if you need the tests.
