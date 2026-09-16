# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Public npm distribution is the single package `columna` (workspace `@columna/*` packages remain private and are bundled into the published tarball).

## [0.1.0] - 2026-09-16

### Added

- Fluent DataFrame API (`DataFrame`, `LazyFrame`, `Series`, `Expr` / `col`) with CPU, WASM, and WebGPU backends.
- Public umbrella package `columna` with subpath entry points `columna/core` and `columna/advanced`.
- IO helpers: CSV, JSON, Excel, Parquet, optional SQL dialects, and bounded Kafka batch reads.
- Advanced statistics layer (`columna/advanced`) aimed at Minitab-parity workflows.
- Optional native Node kernels (`@columna/native`, private workspace package) and browser IDE (`columna-studio`, private).
- Benchmark and comparison tooling under `@columna/bench` (private).
