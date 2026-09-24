# Production operations guide

How to run columna inside a long-lived service: what to configure before the first request, what to watch, how
to size memory and disk, how to shut down, how to upgrade. Everything here points at a concrete API; the
[troubleshooting corpus](troubleshooting.md) covers what to do when one of these steps misbehaves.

## 1. Startup

```ts
import { init, createSession, setIoPolicy, setMemoryPolicy, closeParallelPool } from 'columna'

// 1. process-wide floors — the part per-request code cannot loosen
setIoPolicy({ allowedDirs: ['/srv/data'], denyPrivateHosts: true, maxBytes: 500e6, timeoutMs: 30_000, maxRedirects: 3 })
setMemoryPolicy({ maxBytes: 1.5 * 1024 ** 3, spillDir: '/var/lib/app/columna-spill', maxCacheBytes: 256 * 1024 ** 2 })

// 2. accelerators — once, before serving; nothing is detected at import time
const runtime = await init({ native: true, rust: false })
```

- `init()` is the only place WebGPU detection, the native addon and the WASM kernels are loaded. Call it once at
  boot; until then everything runs on the CPU engine, which is a valid production configuration on its own.
- The native addon (`@columna/native`) is optional. Log `isNativeKernelsLoaded()` at boot so a deployment that
  silently lost the binary (wrong platform, missing build) is visible in the logs rather than in latency graphs.
- `setIoPolicy` and `setMemoryPolicy` are **floors**: per-call and per-session options can only narrow them.
  Set them before the first request; they are not re-read from the environment.

## 2. Per-tenant / per-request isolation

Use a `Session` per tenant (or per request when tenants are not a concept). It owns a runtime (engine,
strictness, memory policy), a `persist()` cache and an IO policy layered over the process floor:

```ts
const session = createSession({
  io: { allowedHosts: [`${tenant}.data.example`], maxBytes: 200e6 },
  runtime: { engine: 'cpu', strict: true, memory: { maxBytes: 512 * 1024 ** 2 } },
  persist: { maxBytes: 64 * 1024 ** 2 },
})
try {
  const df = await session.readParquet({ url })
  return await df.lazy().filter(...).groupBy(...).agg(...).collect({ signal: req.signal, timeoutMs: 10_000 })
} finally {
  session.close() // frees the tenant's cache; frames already returned stay valid
}
```

`engine('cpu', { strict: true })` (or `runtime: { engine: 'cpu', strict: true }`) makes results deterministic and
independent of what accelerators the host happens to have; use it unless you have measured a benefit from
`'auto'` and read the execution report to confirm the accelerator actually ran.

## 3. Request timeouts and cancellation

`collect({ signal, timeoutMs })` checks the signal and the deadline **before every operator** and yields to the
event loop between operators, so an aborted HTTP request stops a plan within one operator and the service stays
responsive while a plan runs. What it cannot do is interrupt a single kernel: a 50M-row sort finishes before the
abort is observed. Size `timeoutMs` for the largest single operator you allow, and cap input size with the IO
policy so no request can submit an operator that large. IO readers honour the same `signal` while streaming.

Catch `ExecutionAbortedError` and map it to 499 / 504; it carries `reason` (`'signal' | 'timeout'`), `node` (the
operator) and `elapsedMs`. An abort never falls back to another engine.

## 4. Memory

Rule of thumb for a plan's peak: the input columns (8 B per f64 / datetime cell, 4 per i32 / u32 / f32 / category,
1 per bool, plus a JS string per utf8 cell) **plus the largest intermediate** (a sort or join holds index arrays;
a groupBy its accumulators). Put a `maxBytes` budget in the memory policy: when live estimates exceed it, sort /
unique / join spill runs to disk through `fs/promises` (the event loop keeps turning) and the report shows
`spilledBytes` / `peakBytes`. Budgets are per session, so a tenant cannot consume another's headroom.

Node flags still matter: the budget is *columna's* estimate of *its* tables, not the heap. Run with
`--max-old-space-size` sized for budget + application + headroom, and watch RSS, not only the report.

`persist()` caches are LRU-capped by `maxCacheBytes` (process) or `persist: { maxBytes }` (session); an entry is
evicted whole. Cached tables count toward RSS, not toward the execution budget. The cache's own bookkeeping is
bounded too: `maxEntries` (default 256), `maxPending` (default 1024 marks of plans not yet collected), optional
`ttlMs` / `pendingTtlMs`; watch `cache.stats()` (`hits`, `misses`, `evictions`, `pendingEvictions`, `expired`,
`skipped`, `strictBypasses`).

What the cache will not do: serve a plan over zero-copy frames (`fromColumns(…, { copy: false })`), serve a plan
that calls `mapElements` unless `persist({ trustUdfs: true })`, or answer a strict engine request with a table
another engine produced — the report's `cacheSkipped` / `cachedFrom` fields say which case applied.

## 5. Spill directory

- Default: `os.tmpdir()/columna-spill/columna-p<pid>-<random>/`, created `0700`; files are `0600`, created with
  `wx` (never overwritten), unlinked as soon as the operator finishes — also on abort.
- Point `spillDir` at a volume with room for **the largest spilling input, twice** (sorted runs + merge) and no
  cross-tenant readers. `tmpfs` is fine when RAM allows; it defeats the purpose otherwise.
- After a crash the per-process directory stays behind: clean `columna-p*` directories whose pid is dead in your
  boot script. Nothing in a spill file is meaningful across processes or versions.
- Spill is Node-only. In the browser a `maxBytes` budget is still enforced as an estimate; `spill: true` throws.

## 6. Workers, threads and the browser

- Node: a worker pool of `min(cpus, 4)` threads starts lazily on the first operator above its threshold (filter /
  dual-gt ≥ 1M rows without the native addon, gather ≥ 2M, sort / groupBy / unique ≥ 5M) and stays alive. Call
  `closeParallelPool()` in your shutdown hook or the process waits for the workers. `COLUMNA_SHARED=0` disables
  SharedArrayBuffer-backed columns (then workers are not used).
- Browser: Web Workers over `SharedArrayBuffer` need cross-origin isolation (`Cross-Origin-Opener-Policy:
  same-origin`, `Cross-Origin-Embedder-Policy: require-corp`); without it the same plans run single-threaded,
  correctly, with no warning beyond the report's kernel names. WebGPU needs `init()` and an adapter; the
  strict engine reports the refusal reason when there is none, or when a column is f64 (exactness).
- Heavy browser plans block the main thread for their duration unless `collect({ signal })` is used (yields
  between operators) — for real off-main-thread execution run columna inside your own Web Worker and post
  Arrow IPC bytes across.

## 7. Observability

Log the execution report of anything slow or unexpected — it is the only source of truth for what ran:

```ts
const { frame, report } = await plan.collectWithReport()
log.info({ requested: report.requested, dispatched: report.dispatched, used: report.backendsUsed,
           ms: report.totalMs, peak: report.peakBytes, spilled: report.spilledBytes, cacheHit: report.cacheHit,
           fallbacks: report.fallbacks, nodes: report.events.map((e) => [e.node, e.backend, e.kernel, e.ms, e.rows]) })
```

Good alerts: `fallbacks.length > 0` when you asked for an accelerator; `spilledBytes > 0` on a service that
should never spill; `peakBytes` near the budget; `totalMs` per node type. `formatExecutionReport(report)` is the
human-readable form for tickets. Kernel names are Tier 2 (see [compatibility.md](compatibility.md)): plot them,
do not branch on them.

## 8. Shutdown

```ts
process.on('SIGTERM', async () => {
  server.close()
  await closeParallelPool()       // worker threads
  clearPersistCache()             // process-wide cache (sessions clear their own on close())
  process.exit(0)
})
```

Spill files of operators still running are removed by their `finally` blocks; an abrupt kill leaves the
per-process spill directory (see §5). SQL clients from `openSqlClient()` are yours to `close()`.

## 9. Upgrading

1. Pin an exact version (or the release tarball's SHA-256 from the GitHub Release) — never `main`.
2. Read the CHANGELOG section: *Breaking* lines reference the API-surface snapshot diff; *Changed* lines name
   behaviour that produces different results for the same input.
3. Run your own differential gate: the same inputs through the old and the new version, compare row-for-row
   (`toArray()` / Arrow IPC bytes) and compare `collectWithReport()` kernels for the plans that matter.
4. Only then roll out; keep the previous tarball to roll back.

`pnpm api:check` against the built package verifies nothing you recorded disappeared; consumers can run the same
script over their own list of names.

## 10. Security checklist (details in [SECURITY.md](../SECURITY.md))

- Every untrusted source is `{ url }` / `{ path }` / `{ text }`, never a bare string.
- `allowedHosts` + `denyPrivateHosts` + a pinning `fetch` (or an egress proxy) for URL loads; `allowedDirs` for
  paths; `maxBytes` and `timeoutMs` everywhere.
- CSV exports of user text use `escapeFormulas: true`.
- SQL text is yours: parameters through `params`, never string concatenation; `nRows` pushdown wraps a single
  SELECT and leaves anything else untouched.
- `mapElements(fn)` runs arbitrary code in your process — it is an escape hatch for your code, not a sandbox for
  someone else's.
