# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Public npm distribution is the single package `columna` (workspace `@columna/*` packages remain private and are bundled into the published tarball).
- `fromRows()` infers dtypes from **all** rows (no 256-row sample).
- GPU numeric filters skip `f64` / `datetime` (f32 upload was lossy); CPU fallback used instead.
- `DataFrame.writeParquet()` now throws: it never wrote Apache Parquet. Use `writeParquetLike()` for the JSON format.
- `toBlob('parquet')` renamed to `toBlob('parquet-like')`.

### Fixed

- CSV write dense-integer cache no longer truncates fractional values (e.g. `100.5` → `100`).
- `setValue` / `cast('i32'|'u32')` reject non-integers and out-of-range values instead of silent wrap/truncation.
- CI: remove duplicate pnpm version pin so `pnpm/action-setup` can use `packageManager` from `package.json`.

## [0.1.0] - 2026-09-16

### Added

- Fluent DataFrame API (`DataFrame`, `LazyFrame`, `Series`, `Expr` / `col`) with CPU, WASM, and WebGPU backends.
- Public umbrella package `columna` with subpath entry points `columna/core` and `columna/advanced`.
- IO helpers: CSV, JSON, Excel, Parquet, optional SQL dialects, and bounded Kafka batch reads.
- Advanced statistics layer (`columna/advanced`) aimed at Minitab-parity workflows.
- Optional native Node kernels (`@columna/native`, private workspace package) and browser IDE (`columna-studio`, private).
- Benchmark and comparison tooling under `@columna/bench` (private).
