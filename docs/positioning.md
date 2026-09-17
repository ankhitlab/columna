# Where columna fits — and where it does not

"Fluent API, runs in JS" describes Arquero as well as columna, and DuckDB-Wasm and Polars both have JavaScript
interfaces. This page states the actual trade-offs, with numbers from `pnpm bench:compare:js`
([comparison-js.md](comparison-js.md): same 2M-row file, 24 operations, results cross-checked to be identical).

## The alternatives, by scenario

| Scenario | Strongest option | Where columna stands |
|---|---|---|
| Table transformations directly in JS (browser or Node), no native binary, no WASM heap | **Arquero** or **columna** | Same operation set. On 2M rows columna is typically several× faster per transform (filter 30 vs 182 ms, groupBy 31 vs 170, sort 182 vs 1906, sortMulti 233 vs 2707, unique 182 vs 297, join 45 vs 388) and holds ~3× less memory (1084 vs 3525 MB peak) because columns are typed arrays rather than JS arrays; Arquero has a larger ecosystem (Vega, Observable). |
| SQL analytics and Parquet in the browser | **DuckDB-Wasm** | It is a real query engine: optimiser, SQL, Parquet/Arrow natively, CSV read is in the same ballpark here (~0.9 vs 0.3 s with columna's optional native Rayon path). Cost: a 5–10 MB WASM bundle, a separate heap (~1.9 GB peak here), SQL as the interface. columna's per-op speed is comparable on filter/groupBy/join with a ~150 kB core and no SQL. |
| Heavy tabular processing outside the browser | **Polars** (`nodejs-polars`) or **DuckDB** (native) | Native multi-threaded cores still win on some ops, but on this 24-op suite columna's warm total (~1.7 s) is competitive with Polars eager (~2.4 s) when `@columna/native` is present; DuckDB remains fastest (~1.2 s). If the workload is server-side and a full SQL/streaming optimiser is required, prefer them. columna's distinct role is the statistics layer (below). |

Measured on one Windows x64 machine, Node 24; single process per library; multi-threaded engines use every core, the
three JS-side libraries use one. Cold numbers (module load + first run) are in the linked report.

## What columna actually offers that the others do not

1. **Minitab-level statistics in the same package.** One-sample to MANOVA, DOE, reliability, SPC, capability,
   time series, regression with full diagnostics — ~250 functions, each checked against scipy / numpy / NIST
   fixtures, property tests and seeded Monte-Carlo ([advanced-roadmap.md](advanced-roadmap.md),
   [minitab-coverage.md](minitab-coverage.md)). Arquero, DuckDB and Polars stop at descriptive aggregates;
   the JS statistics libraries stop at t-tests. This is the reason to pick columna.
2. **Exactness as a contract.** Type inference widens instead of coercing (a late 2³¹ or "n/a" changes the
   column, never the value); `cast` refuses overflow; GPU kernels compare integers bit-exactly or decline;
   `engine(kind, { strict: true })` and `collectWithReport()` say which backend actually ran each node.
3. **Small and dependency-free where it matters.** Core + runtime + advanced have zero runtime dependencies;
   CSV / JSON / Arrow-like IO is built in; only Parquet (`hyparquet`) and Excel (`xlsx`) pull packages, and
   SQL / Kafka drivers are optional peers. The same code runs in Node and the browser.
4. **Explicit boundaries for server use.** IO policy (allowed hosts / directories, byte caps, timeouts,
   cancellation), formula-safe CSV export, prototype-safe row construction — documented in
   [SECURITY.md](../SECURITY.md).

## What columna does not have — do not read the fluent API as a promise of these

- **No query optimiser.** Operations execute in the order written; `filter → groupBy` is fused, most other
  chains materialise intermediates. Polars-lazy and DuckDB plan; columna does not.
- **No multi-threaded engine.** Node workers apply to two kernels above 50M / 2M rows; the browser is one
  thread; WASM / WebGPU are narrow accelerations with CPU fallback (see README → Engines). A 10M-row groupBy in
  the browser blocks the UI for as long as it takes.
- **CSV reading is the slow part.** 3.2 s for 107 MB, single-threaded JavaScript parsing; the operations after
  it take 0.3 s. The reader is streaming and memory-bounded, not fast.
- **Arrow interop is a JSON-shaped `ArrowLike`, not Arrow IPC.** Zero-copy exchange with DuckDB-Wasm or
  Arrow-based tools is not available; conversion goes through JavaScript values.
- **No SQL.** Everything is method chains and expressions.
- **Scale ceiling is process memory by default.** On Node you can set a soft budget (`MemoryPolicy.maxBytes`) so sort / unique / join / groupBy spill intermediates to disk instead of growing without bound; the browser has no spill path.

## Choosing

- Statistics beyond aggregates, in JavaScript, with defensible numbers → columna.
- Interactive in-browser transforms of up to a few million rows, minimal bundle, no SQL → columna or Arquero;
  columna if speed and memory matter, Arquero if the Vega / Observable ecosystem matters.
- Browser analytics over Parquet, SQL, or tens of millions of rows → DuckDB-Wasm.
- Server-side bulk processing → Polars or DuckDB; put columna's `advanced` on the result if you need the tests.
