# Compatibility promises

What you can rely on when you pin `columna@x.y.z`, what may change between versions, and how a change is announced.
The API part of this document is **machine-checked**: [`docs/api-surface.json`](api-surface.json) records every
public export and class member, and `packages/columna/tests/api-surface.test.ts` fails the build if one disappears.
Removing or renaming a public name is therefore always a deliberate edit of that file, in the same commit as the
CHANGELOG entry that explains it.

## Versioning

`columna` follows [SemVer 2.0](https://semver.org). While the major version is **0**:

| Change | Allowed in | Announced by |
|---|---|---|
| Bug fix, performance change with identical results, new export / method / option | patch (`0.3.x`) | CHANGELOG *Fixed* / *Added* |
| Behaviour change that alters results for existing inputs (a numeric convention, an inference rule, a default) | minor (`0.x.0`) | CHANGELOG *Changed*, with the old and the new behaviour spelled out |
| Removing or renaming a **Tier 1** name, changing a Tier 1 signature incompatibly | minor, only after one minor of `@deprecated` | CHANGELOG *Breaking* + the api-surface snapshot diff |
| Removing or changing a **Tier 2** name | minor | CHANGELOG *Breaking* |
| Anything in **Tier 3** | any release | not announced |

From `1.0.0` on, Tier 1 and Tier 2 breaking changes move to major versions. Patch releases never break anything
recorded in the snapshot.

## Tiers

**Tier 1 — stable API.** Everything exported from `columna`, `columna/core` and `columna/advanced` that is not
listed under Tier 2, in particular: `DataFrame`, `LazyFrame`, `GroupBy`, `Series`, `Expr` and their public
methods; `col` / `cols` / `lit` / `when` / `dt`; the readers and writers (`readCsv`, `readParquet`, `readArrowIpc`,
`toArrowIpc`, `writeCsv`, …) and their option names; `IoPolicy` / `setIoPolicy`; `Session` / `createSession`;
`collect()` options (`signal`, `timeoutMs`, `memory`); `ExecutionReport` field names; `EngineStrictError` /
`ExecutionAbortedError` and their public fields; every statistical function of `columna/advanced` — its
signature, the **names** of its result fields and the conventions documented in
[advanced-roadmap.md](advanced-roadmap.md) (which reference each procedure is checked against).

**Tier 2 — evolving.** Usable, documented, may change between minors with a CHANGELOG *Breaking* line:
`Runtime` and `RuntimeOptions`, `PersistCache`, the `Backend` interface, `PlanNode` / `ExprNode` shapes,
`optimizePlan` / `executeCpu` / `hashPlan`, `ExecutionEvent.kernel` strings (they name implementation details:
`native:dualFilter`, `js:sort+spill`, …), `formatExecutionReport` text, the exact wording of error messages
(match on the stable prefixes listed in [troubleshooting.md](troubleshooting.md), not on full strings), the
numeric thresholds at which engines / workers / native kernels engage.

**Tier 3 — internal.** The `@columna/*` workspace packages are private and bundled into `columna`; their module
paths, the spill file format (`.cspill`), the parquet-like JSON container's internals, worker protocol messages,
`__`-prefixed and `_`-prefixed members, anything under `packages/bench`, `packages/browser-smoke`,
`packages/studio`.

## Deprecation

A Tier 1 name is retired in three steps: it is marked `@deprecated` in the declarations (editors strike it
through) with the replacement in the message and a CHANGELOG *Changed* entry; it keeps working for at least one
minor release; it is then removed in a minor release with a *Breaking* entry and a snapshot diff. Current
deprecations: `toArrow()` / `fromArrow()` (→ `toArrowLike()` / `fromArrowLike()`; for Apache Arrow use
`toArrowIpc()` / `fromArrowIpc()`), `SqlConnectionConfig.keepAlive` (→ `openSqlClient()`).

## Data and format stability

| Format | Promise |
|---|---|
| Apache Arrow IPC (`toArrowIpc` / `fromArrowIpc`) | The standard: bytes written by version *n* are read by any Arrow implementation and by every later columna. The type mapping (category → `Dictionary<Int32, Utf8>`, datetime → `Timestamp(ms)`) is Tier 1. |
| Apache Parquet (`writeParquet` / `readParquet`) | The standard (hyparquet); codec defaults may change (currently SNAPPY). |
| CSV / JSON writers | Output shape (quoting, `""` for empty strings, datetimes as epoch milliseconds, null as an empty field, `NaN` written literally) is Tier 1; a change is a *Changed* entry. |
| `writeParquetLike` (JSON container) | Read by later versions; not for exchange with other tools. |
| Spill files, persist cache | Process-local, never read across versions or processes. |
| `hashPlan` keys | Process-local; not a portable identifier (UDF identities are per process). |

## Results and numerics

**64-bit integers are exact or refused.** Without an i64 dtype, a 64-bit column is an f64 column only when every
value is within ±(2^53 − 1); otherwise readers raise `PrecisionLossError` unless `int64: 'string'` (exact) or
`int64: 'number'` (declared lossy) is chosen. This default is Tier 1: no future version reads a 64-bit value as
a different number without that explicit opt-in.

Statistical results are compared against scipy / numpy / NIST / Minitab conventions in the test suite; a fix that
changes a result (as the Poisson tail and beta quantile fixes did) is a *Fixed* entry that names the affected
inputs and the size of the change. Dtype inference rules (`fromRows` full-column inference, widening instead of
wrapping, `""` ≠ null through CSV) are Tier 1 behaviour. Floating-point results are **not** promised bit-identical
across engines: the CPU, worker and native paths are checked to agree within the tolerances of the cross-engine
tests; WebGPU is bit-exact for integer kernels and refuses f64 unless `gpuLossyF32` is set.

## Runtime support

| Environment | Status |
|---|---|
| Node.js 18, 20, 22 | Tested in CI on Linux (and 22 on Windows): lint, build, typecheck, the test suite and the Arrow interop suite. Node 18 is past its upstream end of life; it stays supported while `engines` says `>=18`, and leaving it is a minor-release change announced one minor ahead. On 18 the `node:sqlite` test is skipped (the module does not exist there); `process.getBuiltinModule` / `AbortSignal.any` have fallbacks. |
| Node.js 24 | Used in development; not yet in the CI matrix. |
| Chromium (headless, current stable) | Built bundle tested in CI, including the DuckDB-Wasm Arrow exchange and — where an adapter exists — WebGPU. |
| Firefox, Safari | Best effort: the CPU path has no browser-specific code; WebGPU and `SharedArrayBuffer` workers depend on the browser's support and on COOP/COEP. |
| Bundlers | Vite is tested (browser smoke, no aliases). The package ships ESM + CJS with `exports`; Node-only modules are loaded through opaque dynamic imports so browser builds stay clean. |
| Optional native addon (`@columna/native`) | Same version as `columna`; absence is always tolerated (JS kernels run). |

## What is explicitly *not* promised

- Timing, memory and the choice of kernel for a given input. Thresholds move as kernels improve; read the
  execution report, do not encode kernel names in application logic.
- Text of error messages beyond the prefixes listed in [troubleshooting.md](troubleshooting.md).
- Iteration order of `unique` / `groupBy` output without an explicit `sort`.
- Behaviour on inputs outside the documented domain of a statistical procedure (they throw, and the message may change).

## Checking your own usage

`pnpm api:check` (after `pnpm build`) verifies the built package against the snapshot; the CI `check` job runs the
same through vitest. Consumers who want the same guarantee for the subset they use can copy
`scripts/api-surface.mjs`, record their own list and run it against each new version before upgrading — the
[operations guide](operations.md) describes the upgrade gate this fits into.
