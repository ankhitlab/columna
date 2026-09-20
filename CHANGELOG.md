# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-09-20

### Added

- **Apache Arrow IPC** (no dependency): `DataFrame.toArrowIpc({ format?: 'stream' | 'file', batchRows? })`,
  `DataFrame.fromArrowIpc(bytes)`, `readArrowIpc(source)` / `writeArrowIpc(path)`, `LazyFrame.toArrowIpc()`;
  `toArrowIpc` / `fromArrowIpc` on `TableView` from `@columna/arrow`. Writes Float64/32, Int32, UInt32, Bool, Utf8,
  Dictionary<Int32, Utf8> (category), Timestamp(ms) (datetime); reads those plus Int8…Int64 / UInt8…UInt64,
  Float16, LargeUtf8, Utf8View, Timestamp in any unit, Date32/64, Null, dictionary indices of any width, several
  record batches and dictionary deltas. Interop suite against `apache-arrow` (dev dependency), committed Polars
  fixtures (`tests/fixtures/polars-ipc.*`) and DuckDB-Wasm in the browser smoke.
- `collect({ signal, timeoutMs })` / `collectWithReport(...)` / `Runtime.execute(plan, opts)`: cooperative
  cancellation and deadlines checked before every operator; the CPU engine yields to the event loop between
  execution units when a guard is present (fused units stay fused, each unit traced). Rejects with
  `ExecutionAbortedError { reason: 'signal' | 'timeout', node, elapsedMs }`; never triggers an engine fallback.
- `pnpm test:coverage` (v8; CI prints the summary on the job page and uploads the HTML report),
  `pnpm docs:api` (typedoc over the `columna` entry points; CI artifact + GitHub Pages workflow),
  [docs/migrating.md](docs/migrating.md) (Arquero / Polars / pandas → columna).
- Non-blocking spill: under a memory budget the runtime executes unit by unit and sort / unique / join spill through
  `fs/promises` (`spillWriteAsync` / `spillReadAsync`, chunked writes); the event loop keeps turning, `{ signal }` is
  honoured between units, kernels are reported as `js:sort+spill` / `js:unique+spill` / `js:join+spill`.
- SQL `nRows` pushdown: single SELECT / WITH statements get `SELECT * FROM (…) AS __columna_q LIMIT n` (PostgreSQL,
  MySQL, SQLite, ClickHouse) or `SET ROWCOUNT n` (SQL Server); `pushdown: false` opts out; unknown dialect / several
  statements keep the client-side slice.
- Documentation set: [compatibility.md](docs/compatibility.md) (tiers, 0.x SemVer rules, deprecation, format
  and runtime support promises) enforced by `docs/api-surface.json` + `api-surface.test.ts` / `pnpm api:check`;
  [operations.md](docs/operations.md) (startup, sessions, timeouts, memory, spill directory, workers, observability,
  shutdown, upgrade gate); [troubleshooting.md](docs/troubleshooting.md) (error index + the incident corpus with
  commit references); versioned documentation site (`pnpm docs:site`: `latest/` + every release tag, version
  switcher, `versions.json`) published by the `pages` workflow on pushes and tags; guides included in the typedoc site.
- `columna` umbrella now exports `setMemoryPolicy` / `getMemoryPolicy` / `clearMemoryPolicy` / `clearPersistCache` /
  `persistCacheStats` / `estimateTableBytes` / `closeParallelPool` (the README already used them).
- `Session` / `createSession({ io, runtime, persist, backends })`: per-tenant runtime (engine, strict, memory), own
  `PersistCache` and an IO policy floor layered over `setIoPolicy()` (`IoLoadOptions.floors`); `df.withRuntime(rt)` /
  `getRuntime()`; `Runtime.listBackends()`; `PersistCache` class exported (module-level persist helpers now delegate
  to `defaultPersistCache`). `new Runtime({ memory })` no longer sets the process memory policy.
- `denyPrivateHosts` now also resolves hostnames on Node (`dns.lookup`, all addresses, every redirect hop) and
  refuses names that resolve to private addresses; `IoPolicy.resolveHost` swaps the resolver (process-wide one is a
  floor). SECURITY.md documents connection pinning with an undici Agent for the rebinding TOCTOU.
- Release tooling: `pnpm release:prepare <version>` (bumps versions, moves [Unreleased] → [version]),
  `.github/workflows/release.yml` on `v*` tags (gates → GitHub Release with tarball + SHA-256 + CHANGELOG section →
  npm publish with provenance when `NPM_TOKEN` exists).
- README badges (CI, coverage via the Pages-hosted shields endpoint, npm, API reference, license).
- Arrow interop tests moved to `packages/arrow-interop` (`pnpm test:interop`); `@columna/core` has no Arrow dev dependency.
- Browser smoke: DuckDB-Wasm Arrow IPC exchange check; the strict WebGPU check uses exact i32 columns
  (the f64 refusal is its own check) so the browser job is green on runners that expose an adapter.

- Analyst DX: sync `DataFrame.head` / `tail` (materialized peeks), `show` / `print` (markdown to console).
- `ffill` / `bfill` plan nodes; `fillNull({ col: value })` per-column map overload.
- `GroupBy.count` / `sum` / `mean` / `min` / `max` / `std` / `var` / `nunique` / `median` shortcuts; `agg({ col: ['sum','mean'] })` expands to `col_sum` / `col_mean`.
- Join `suffix` / `lSuffix` / `rSuffix`; `rightJoin` / `outerJoin` / `fullJoin`; `validate: '1:1' | '1:m' | 'm:1'`; right/outer shared-key coalesce.
- Sort `nullsLast` per key (default true); `col('x').asc({ nullsLast: false })`.
- Sync `DataFrame.describe()`; filter aliases `where` / `isNull` / `notNull` / `between`; `exclude` / `selectNumeric` / `selectDtypes`; `assign`; `rename(fn)`.
- `DataFrame.col` / `nunique()`; Series `fillNull` / `ffill` / `bfill` / `valueCounts` / `nunique`.
- Real Apache Parquet write: `writeParquet` / `writeParquetBytes` via `hyparquet-writer` (round-trips with `readParquet`); `writeParquetLike` remains the JSON container.
- Plan fusions: `filter → unique`, `filter → sort → limit`; projection through join with execute-time column keep.
- Rule-based `optimizePlan` (merge filters; push filter past project / under withColumn·drop·rename·sort / into join sides; limit under project; fold project/drop/rename; groupBy input prune; sample/NDV `estimatePlanRows`; inner-join greedy graph reorder + build-side swap; left/semi/anti never swap); runs on `collect` / `explain` / `executeCpu` before backend dispatch. `explain` shows `rows≈`; `collectWithReport` traces `optimized:joinReorder` and native join threshold notes.
- Native Rayon: single-key `argsort` (f64/i32), dual-cmp filter beyond gt∧gt (incl. f64∧f64), dense groupBy min/max.
- **Multi-threaded engine.** Generalized `engine-worker` protocol v2 (filter, sort, groupBy, unique, gather, dual-gt) shared by Node `worker_threads` and browser Web Workers over `SharedArrayBuffer` chunks; reusable worker pool with `parallelFilter` / `parallelSort` / `parallelGroupBy` / `parallelUnique` / `parallelGather` dispatch and per-op thresholds (filter / dual-gt ≥ 1M when native is absent, sort/groupBy/unique ≥ 5M, gather ≥ 2M). Native Rayon is preferred over workers when both are available. `CpuBackend.execute` routes large sort / unique / filter / gather through workers; every parallel path falls back to a single-threaded kernel below its threshold or when workers / SAB are unavailable. Native Rayon gaps filled: multi-key `argsortMultiF64`, parallel `uniqueF64`, generic multi-column `filterF64`, hash-join build `joinBuildDenseI32`. `collectWithReport` records the kernel that ran (`workers:sort`, `native:dualFilter`, `native:joinBuildDenseI32`, `js:unique`, …).
- `columna/advanced` DataFrame wrappers: `propTest`, `adfTest`, `kpssTest`, `ridge`, `lasso`, `ancova`, `ksTwoSample`; `formatReport(result, 'markdown' | 'html')` for t-test / ANOVA / OLS.

### Changed

- `toArrow()` / `fromArrow()` are deprecated (they return / take columna's JSON `ArrowLike`, not Arrow):
  use `toArrowLike()` / `fromArrowLike()`, or `toArrowIpc()` / `fromArrowIpc()` for Apache Arrow.
- README / positioning no longer say "no runtime dependencies in the core": the engine and statistics packages have
  none; `@columna/core` / `columna` declare the four lazily-loaded IO packages.

- Positioning / Engines docs: native gather threshold 50k (not 250k); CSV compare-js ~280 ms with native; Parquet write and fused plan list updated; rule-based rewrite listed under offers (not full CBO under gaps); "multi-threaded engine" moved from a gap to an offer (threads accelerate individual heavy ops; not a morsel-driven parallel runtime).

### Fixed

- Optimizer semantics (review 1684ac4): do not push right-side filters under `LEFT`/`OUTER` joins (or left-side under `RIGHT`/`OUTER`); keep the outer `project` after pushing columns under `sort` so `select` cannot re-expose sort-only fields; treat `agg` and `rowOffset` as filter pushdown barriers so `mean` / `shift` see the correct input; skip inner-join build-side swap and graph reorder when non-key column names collide (avoids swapping `value` vs `value_right` provenance); merge composite equi-key components into one join-graph edge instead of dropping all but the first.
- `hashPlan` / persist: distinct UDF identities via process-local function tokens; distinguish `null` / `NaN` / `±Infinity` literals so cache lookup cannot return another expression's result.
- Spill temp files: per-process private subdirectory under the spill parent, `0700`/`0600` modes, exclusive create (`wx`), unlink on write failure.
- `columna/advanced` DTS: discriminated `propTest` one-/two-sample options (`exact`|`normal` vs `normal`|`fisher`) with runtime method checks; `ancova` coerces boolean group labels to `"true"`/`"false"` instead of passing `Series.toArray()` booleans through.
- Integration regressions: `packages/core/tests/review-1684ac4.test.ts`.
- Conservative optimizer / AST: shared `expr_walk` so `str.concat` `other` is renamed and scanned like every other child; refuse join projection pushdown when the projection asks for a collision suffix (`score_right`); skip final `project` after join keep only for identity column picks (aliases like `col('score').alias('x')` keep the output name).
- Fast-path kernels aligned with generic semantics: top-k ties break by original row index and NaN keys take the full radix sort; `agg({ x: 'count' })` counts non-null values on dense/Map paths; multi-key category `unique` keeps null-key rows; numeric `unique` distinguishes `null` from `NaN`; dense join / semi / anti only for integer build keys.
- Composite Map keys for slow join / groupBy / unique use typed length-prefixed encoding (no `∅`+`\0` collisions).
- CJS spill: `createRequire` falls back when `import.meta.url` is empty in the CJS bundle; browser `MemoryPolicy` scopes refuse a plain `collect` while another override is active.
- CI: cooperative cancel parity tests get a 30s timeout so `pnpm test:coverage` on Node 22 no longer flakes at the Vitest 5s default.

## [0.2.1] - 2026-09-17

Republish of the 0.2.0 contents after the initial `0.2.0` tarball stalled in the npm registry staging queue.

## [0.2.0] - 2026-09-17

### Added

- `MemoryPolicy` (`setMemoryPolicy` / `collect({ memory })`): soft `maxBytes` budget; on Node, sort / unique / join spill columnar temp files (`spillDir`, default `os.tmpdir()/columna-spill`) and report `spilledBytes` / `peakBytes`.
- Chunked one-pass `groupBy` under the memory budget (partial Acc maps merged across row batches).
- Explicit `persist()` / `unpersist()` LRU cache keyed by plan hash (`maxCacheBytes`); `collectWithReport().report.cacheHit`.
- Filter views (shared column buffers + selection index until gather/sort/join) and light projection pushdown (project below sort / through filter / into join sides).
- Native `writeCsvUnquoted` for unquoted Node path writes.
- Stress / load identity suite (`pnpm test:stress`; `STRESS_HEAVY=1` for larger N): adversarial nulls, joins, sortMulti, CSV round-trip, spill vs in-memory.

### Changed

- Faster CSV read: fused unquoted scan (numbers/nulls/bools without per-cell strings), char-code number parse, capacity reserve from file size / newline estimate, 4 MiB stream chunks.
- CSV acceleration: optional `@columna/native` Rayon `parseCsvUnquoted` (typed column builders, no per-cell `Vec`), then worker_threads chunk pool (≥8 MiB), then single-thread fused path; public `columna` package lists `@columna/native` as an optionalDependency so Node benches hit the native path.
- JS CSV writer specializes ncols 4–8 and skips safe-int substrings in the fused parser; chunk size 32 768 rows.
- Sort gather: native threshold 50k rows; always gather through a single `Uint32Array` index buffer.
- Fast multi-key / string sort: successive stable radix via `sortKeyCodes` (category/utf8 ranks) and `argsortNumeric(order)`; single-key category/utf8 uses the same path; slow path prefetches keys and gathers.
- `compare:js` / `docs/comparison-js.md`: 24 operations (added drop, rename, sortMulti, tail, sample, left/semi/anti join, melt, pivot, valueCounts, describe, corr, filter→groupBy); report is ops×libraries with warm and cold tables.
- Adopted external recheck suites (`review-regressions`, `review-types`); datetime `toArray` type asserted as epoch `number` (not `Date`).

### Fixed

- `hashPlan` / persist lookup no longer `JSON.stringify`s scan table payloads (identity token + column names); empty-cache collect skips hashing.
- Category `str.*` transforms canonicalize the dictionary and remap codes after level collapse (toLowerCase → valueCounts / unique).
- GPU map/arith requires `gpuLossyF32: true`; exact mode always matches CPU f64 results (including for f32 inputs).
- `denyPrivateHosts` blocks IPv4-mapped IPv6 literals (e.g. `[::ffff:127.0.0.1]`).
- `Expr.shift` / `diff` / `pctChange` typed as allowing null at the window edge.
- `toArray()` returns datetime as epoch milliseconds (aligned with `InferColumns`).
- `fromRows` reads cells via own-property `getRowField` (missing `constructor` is null, not `Object`).
- `writeCsvText` uses one stream `error` listener for the whole write.
- compare-js report wording: cold totals, aggregate-over-join, summary-match, RSS sampling limits.

## [0.1.0] - 2026-09-16

### Added

- Fluent DataFrame API (`DataFrame`, `LazyFrame`, `Series`, `Expr` / `col`) with CPU, WASM, and WebGPU backends.
- Public umbrella package `columna` with subpath entry points `columna/core` and `columna/advanced`.
- IO helpers: CSV, JSON, Excel, Parquet, optional SQL dialects, and bounded Kafka batch reads.
- Advanced statistics layer (`columna/advanced`) aimed at Minitab-parity workflows.
- Optional native Node kernels (`@columna/native`, private workspace package) and browser IDE (`columna-studio`, private).
- Benchmark and comparison tooling under `@columna/bench` (private).
