# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
