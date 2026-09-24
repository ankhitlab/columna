# Troubleshooting

Symptom → cause → fix, in two parts: **A.** an index of the errors the library throws (the prefixes are stable
across versions, the full text is not), and **B.** the incident corpus — defects that were actually found in
this repository, by whom, how they showed up and what now prevents them. Part B is kept because the failure
modes recur in user code and in other libraries; the commit hashes let you read the fix.

When something is slow rather than wrong, start with `collectWithReport()` — see the
[operations guide §7](operations.md#7-observability).

## A. Error index

| Message starts with | Thrown by | Meaning and what to do |
|---|---|---|
| `IO policy: host "…" is not in allowedHosts` | any URL read | The host (after every redirect hop) is not covered by `allowedHosts` of the process floor, a session floor or the call. Check the *effective* layers — a per-call list cannot widen a floor. |
| `IO policy: host "…" is a loopback / private / link-local address` | URL read | Name check of `denyPrivateHosts` (literal IPs, `localhost`, `*.internal`, metadata hosts). Intentional for user-supplied URLs. |
| `IO policy: host "…" resolves to …` | URL read, Node | `denyPrivateHosts` resolved the name to a private address. If the host is legitimately internal, it does not belong behind a policy that denies private hosts; give it its own session without the flag. |
| `IO policy: cannot resolve host` | URL read, Node | DNS failure before the request; fail-closed by design. Supply `resolveHost` for offline / split-horizon setups. |
| `IO policy: path "…" is outside allowedDirs` | path read / write | Real paths are compared; a symlink inside an allowed directory that points outside is refused. |
| `IO policy: … above maxBytes` / `exceeds maxBytes` | any read | The smallest `maxBytes` of all layers won. Responses are cut off while streaming, so the byte count in the message may be the cap, not the file size. |
| `IO policy: too many redirects` | URL read | `maxRedirects` (default 5, minimum over layers) exceeded. |
| `IO policy: credentials in URLs are not allowed` | URL read | `user:pass@host` is refused; use `fetch` with headers. |
| `Local path IO requires Node.js` | path read in the browser | A bare string looked like a path. Pass `{ text }` for content or `{ url }` for a URL. |
| `Failed to fetch …: <status>` | URL read | Non-2xx after policy checks; the response body is not read. |
| `Unknown column "…"` (with `Did you mean:`) | any column reference | Typo or a column renamed / dropped earlier in the chain; `explain()` prints the plan with column names. |
| `Column length mismatch` | `fromColumns`, `tableFromColumns` | Ragged input; every column must have the same length. |
| `Cannot store … in i32` / `in u32` | `cast`, `setValue` | Overflow is refused, never wrapped. Cast to `f64`, or fix the data. |
| `engine "…" (strict): …` (`EngineStrictError`) | `collect` with `strict: true` | The requested engine did not run the plan; `reasons` says why (`does not support this plan`, `column "x" is f64: float32 would merge neighbouring values`, `no adapter`). Either use the CPU engine or make the plan eligible. `init({ gpuLossyF32: true })` is the explicit opt-in for f64 on WebGPU. |
| `execution aborted at "…"` / `execution exceeded timeoutMs at "…"` (`ExecutionAbortedError`) | `collect({ signal, timeoutMs })` | Observed between operators; `node` names the operator that would have run next. Not a fallback — the plan produced nothing. |
| `timeoutMs must be > 0` | `collect` | Validation. |
| `MemoryPolicy.spill is only supported on Node.js` | `setMemoryPolicy` / `collect({ memory })` in a browser | Leave `spill` unset in browser code; `maxBytes` alone is accepted. |
| `Concurrent collect({ memory }) overrides are not supported in the browser` | browser | Two overlapping collects with different per-call memory policies. Use one session-level policy (`createSession({ runtime: { memory } })`) instead of per-call overrides. |
| `spill: Node support not initialised` | direct `spillWrite` / `executeCpu` under a budget | You called the synchronous spill path before the runtime initialised Node support; go through `collect()` or `await ensureSpillSupport()`. |
| `spillRead: bad magic` / `unsupported version` | spill | A spill file from another process or version, or a foreign file at the path. Clean the spill directory. |
| `worker pool destroyed` / `worker threads need Node` / `web workers unavailable` | parallel paths | Informational when a job was in flight at shutdown; otherwise the environment has no workers and the single-threaded kernel is used automatically. |
| `fromArrowIpc: …` | Arrow reader | Names the column and the Arrow type it refuses (nested, decimal, binary, compressed batches, big-endian) or a truncated buffer. Write uncompressed IPC; flatten nested columns upstream. |
| `toArrowIpc: utf8 column exceeds 2 GiB per batch` | Arrow writer | Use `{ batchRows }`. |
| `join validate '…': … keys are not unique` | `join({ validate })` | The relationship you asserted does not hold; the join is not performed. |
| `<dialect> support requires the optional "<pkg>" package` | `readSql` | Install the driver as a peer dependency; nothing is bundled. |
| `rolling: window must be a positive integer` | `rolling` | Validation. |
| `<test> needs at least N observations` / `alpha must be in (0, 1)` / `alternative must be …` | `columna/advanced` | Domain checks of statistical procedures; the reference implementations reject the same inputs. |
| `Session is closed` | `Session` | `close()` was called; frames it produced stay usable, the session does not. |
| `<reader>: column "…" row N holds …, which a double cannot represent exactly` (`PrecisionLossError`) | Arrow IPC, Parquet, JSON, SQL, `fromRows` | A 64-bit integer beyond ±(2^53 − 1). IDs / join keys: `{ int64: 'string' }`. Measurements where the nearest double is acceptable: `{ int64: 'number' }`. |
| `…: the inputs are bound to N different runtimes (sessions)` (`RuntimeMismatchError`) | concat, join, as-of join | Frames of two sessions were combined. `session.bind(frame)` one of them, or pass `{ runtime }`. |
| `simpson: x must be finite and strictly monotonic` / `x has N samples, y has M` | `simpson`, `trapz` | Repeated or unsorted sample positions; sort by x and drop duplicates first. |
| report `cacheSkipped: reads caller-owned buffers…` / `…calls a UDF…` | `persist()` | Not an error: the plan ran and was not cached. Build the frame with the default `fromColumns` copy, or `persist({ trustUdfs: true })` for pure functions. |

## B. Incident corpus

Each entry: how it surfaced, root cause, the fix, and the guard that now exists. Dates are the commit dates on
`main`.

### Query correctness

**Outer-join filter pushed below the join** (2026-09-17, `d395417`). A `filter` on a right-side column above a
`leftJoin` was pushed into the right input, turning the left join into an inner join for rows with nulls.
Surfaced in a review of the optimizer, not in tests: the fixtures used inner joins. Fix: pushdown rules track
which side a column came from and never push a predicate past a join that can produce nulls for it; rules that
cannot prove a column's provenance leave the plan alone (`3ffdf40`). Guard: `optimize-differential.test.ts`
executes 2000 seeded random plans optimized and unoptimized and compares rows.

**Projection through sort dropped the sort key** (`d395417`). `project` folded under `sort` removed a column the
sort still needed. Same fix family; same guard.

**Aggregation and `shift` barriers** (`d395417`). A filter was pushed below a window `shift` / an aggregation it
depended on. The optimizer now treats aggregation, window and shift nodes as barriers.

**Fast paths disagreeing with the generic kernel** (`f2184a9`, `bf2f5e7`): top-k (`sort → limit`) heap path
ordered nulls differently from the full sort; dense join / unique / count fast paths diverged on duplicate keys
and on nulls. Fix: each fast path shares the key encoding (`sortKeyCodes`) and null policy of the generic kernel.
Guard: fast-path tests run every input through both paths and compare.

**Composite key collisions** (`740d243`). Multi-column group / join keys were joined with a separator that could
appear inside a string, so `("a|b", "c")` and `("a", "b|c")` grouped together. Fix: typed length-prefixed key
encoding. Guard: keys with separators and empty strings in the invariants suite.

**`persist()` cache collisions** (`d817149`). Plan hashes stringified `NaN` / `±Infinity` / `-0` as `null`, and
two different `mapElements` functions hashed identically, so a second plan could be served the first plan's
result. Fix: explicit tokens for non-finite numbers and per-function identities. Guard: `persist.test.ts`.

**Category join by code, not by value** (`0fa6d84`). Joining two frames whose category columns had different
dictionaries compared dictionary codes. Surfaced in the Arquero/DuckDB comparison bench (wrong row counts). Fix:
join keys decode categories or remap the smaller dictionary. Guard: `join-dictionaries.test.ts`.

**`transpose` emitted category codes** (`0fa6d84`). Same class: a kernel read the code array as values.

### Numerics

**`simpson(y, x)` was the trapezoid** (fixed in the correctness release after 0.3.0, external review). With an
explicit `x`, the function returned `trapz(y, x)`; `∫x² dx` on any grid was off by O(h²). The suite stayed green because the
only test used the `x`-less path on a symmetric input. Fix: SciPy's algorithm. Guard: SciPy fixtures plus polynomial
exactness and convergence-order properties (`quadrature.test.ts`) — the property tests would have caught it on day one.

**Int64 → Number, JSON.parse and ns timestamps rounded silently** (same release). `fromArrowIpc` converted Int64 with
`Number()`; `JSON.parse` rounds integer literals beyond 2^53; ns timestamps were rounded before scaling to ms. Fix:
one `Int64Policy` across readers (error by default), an exact JSON integer pass, BigInt timestamp arithmetic. Guard:
boundary suites in `int64-precision.test.ts` and `packages/arrow-interop`.

**Poisson upper tail wrong by 2·10⁻⁷ at λ = 10⁵** (`14f55eb`, adversarial suite). The incomplete-gamma series
stopped after 1000 terms. Fix: iteration budget grows with √a. Guard: `adversarial.test.ts` tails to 10⁻³⁰⁰.

**Beta quantile with tiny shapes returned 4·10⁻¹⁶ instead of 10⁻¹¹⁵** (`14f55eb`). Bisection in x underflowed.
Fix: inversion in log space. Same guard.

**CSV fast decimal parser off by 1 ulp** (`14f55eb`, invariants suite). `intPart + frac / scale` rounds twice.
Fix: exact path for ≤ 15 significant digits, `Number()` beyond. Guard: 200 000 literals fuzzed against `Number()`.

### Data fidelity

**Empty string became null through CSV** (`14f55eb`). Writer emitted nothing for `""`; reader mapped empty
fields to null. Fix: writer emits `""`, reader keeps a quoted empty as `''`. Guard: invariants round-trip.

**parquet-like reader collapsed categories to the first value** (`14f55eb`). Strings were written into
`setValue(…, 'category')` and all got code 0. Guard: same.

**Fused CSV scanner split records on newlines inside quotes** (`14f55eb`). A fast chunk scanner was not
quote-aware. Guard: same, plus `csv-columnar.test.ts`.

**`fromRows` inferred dtypes from a 256-row sample** (`0fa6d84`). A late `2^31` in an i32 column wrapped; a late
string in a numeric column threw. Fix: full-column inference with widening. Guard: browser smoke check "a late
2^31 widens".

### Runtime and memory

**`LazyFrame.concat` dropped the session runtime** (external review). The static constructor built its result on
the process default, so a tenant's concatenated plan used the process cache and memory policy. Fix: `resolveRuntime`
for every multi-input operator; mixing sessions is an error. Guard: `runtime-propagation.test.ts` runs every
multi-input operator from both receivers, same-session / unbound / cross-session.

**Strict engine requests answered from the cache** (external review). The persist lookup ran before engine
dispatch, so `engine('webgpu', { strict: true })` could return a CPU-computed table with `cacheHit: true`; a failed
strict run also stored its CPU result first. Fix: provenance on entries, strict-aware probe, store only after the
strict check. Guard: `persist-semantics.test.ts` with a stand-in accelerator backend.

**Mutable inputs and UDF closures behind cached results** (external review). `fromColumns` aliased caller arrays and
`persist()` keyed plans by table identity, so writing to the array after the first `collect()` left a stale cached
result; a UDF reading mutable state had the same problem. Fix: copy by default, refuse zero-copy and untrusted-UDF
plans. Guard: same file.

**Memory policy leaked across concurrent requests** (`ac58937`). A per-call `collect({ memory })` set a module
variable that another in-flight request read. Fix: `AsyncLocalStorage` scope on Node; the browser refuses
concurrent overrides. Guard: `memory-spill.test.ts` "keeps overlapping Node async scopes isolated".

**Spill files world-readable / overwritable** (`d395417`, `d817149`). Fix: per-process `0700` directory, `0600`
files, `wx` create, unlink only what this invocation created. Guard: `memory-spill.test.ts` EEXIST case.

**Synchronous spill blocked the event loop** (`ec059b0`, due-diligence report). Fix: async twins through
`fs/promises` under the unit-by-unit executor. Guard: `spill-async.test.ts` asserts an interval keeps ticking.

**CJS build resolved spill's `createRequire` against an empty `import.meta.url`** (`73886e9`). Surfaced only in
the clean-consumer CJS test. Guard: `scripts/consumer-test.mjs` runs a CJS spill.

### Packaging and CI

**Browser bundle failed: static `os` / `worker_threads` / `node:module` imports** (`14f55eb`). Worker and spill
modules were reachable through static re-exports. Fix: opaque dynamic imports and separate build entries.
Guard: the browser smoke builds the real bundle with Vite and no aliases.

**Umbrella did not export `setIoPolicy` / `openSqlClient` / `cols`** (`14f55eb`); later `setMemoryPolicy` and
`clearPersistCache` (this document's review). Surfaced by the clean-consumer test and by writing the operations
guide. Guard: `api-surface.test.ts` snapshot — a name can no longer vanish silently, and the consumer script
imports what the README shows.

**CI never reached the tests** (`14f55eb`). Two declarations of the pnpm version (`packageManager` and the
action's `version`) made `pnpm/action-setup` fail. Fix: one source of truth. The comment in `ci.yml` records it.

**Browser smoke red after the WebGPU exactness change** (`ec059b0`). The strict-GPU check filtered an f64 column
that the new contract refuses. The check itself was stale, not the runtime; it now uses exact i32 columns and
the refusal is its own check.

**`better-sqlite3` "Could not locate the bindings file"** (SQL pushdown tests). pnpm does not run install
scripts by default, so the native driver had no binary. The test uses `node:sqlite` instead; for the driver, add
the package to `pnpm.onlyBuiltDependencies` or run `pnpm rebuild better-sqlite3`.

**`apache-arrow` version mismatch with DuckDB-Wasm** (browser smoke). `tableToIPC` from apache-arrow 21 over a
Table produced by DuckDB's bundled apache-arrow 17 yielded bytes without a schema message. Fix: take DuckDB's
own IPC bytes (`conn.useUnsafe((db, id) => db.runQuery(id, sql))`) — no apache-arrow in the loop.

**Node 18 CI jobs took 20+ minutes** (dropped from the matrix in `cafa5ed`, restored after this investigation).
The published package was never involved — `npm install columna` pulls 7 pure-JS packages in seconds on any Node
version, and the SQL / Kafka drivers are optional peers npm does not install. The time went into `pnpm install` of
the *workspace*: `packages/bench` (private) depends on `duckdb`, which has no Node 18 prebuild, so its install
script compiled DuckDB from source. Fix: `pnpm.neverBuiltDependencies` for the bench-only natives (`duckdb`,
`nodejs-polars`) and for `better-sqlite3` (no test needs its binary), so no install step depends on a prebuild
existing for the running Node. Rebuild them locally (`pnpm rebuild duckdb`) only to run the comparison benches.

**CRLF line endings on Windows contributors' machines**. Tools that rewrite files (`sed -i`) strip `\r` and
produce whole-file diffs. `.gitattributes` normalises on commit; edit with CR-aware tooling.

## Reporting a new one

Open an issue with: the version (`npm ls columna`), Node / browser, the plan (`lf.explain()`), the execution
report (`formatExecutionReport(report)`), and — for a wrong result — the smallest input that reproduces it.
Security-relevant reports go through [SECURITY.md](../SECURITY.md).
