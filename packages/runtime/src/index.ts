import type { TableView } from '@columna/arrow'
import { CpuBackend } from './cpu.js'
import {
  getExecMemoryStats,
  getMemoryPolicy,
  resetExecMemoryStats,
  withMemoryPolicyAsync,
  type MemoryPolicy,
} from './memory.js'
import { ensureSpillSupport } from './spill.js'
import { ExecutionAbortedError, checkGuard, makeGuard } from './cancel.js'
import { PersistCache, defaultPersistCache } from './persist.js'
import { optimizePlan, joinOrderChanged } from './optimize.js'
import {
  DEFAULT_WASM_MIN_ROWS,
  DEFAULT_WEBGPU_MIN_ROWS,
  EngineStrictError,
  estimateRows,
  explainPlan,
  type Backend,
  type EngineKind,
  type ExecContext,
  type ExecuteOptions,
  type ExecutionEvent,
  type ExecutionReport,
  type PlanNode,
  type RuntimeOptions,
} from './types.js'

export class Runtime {
  private backends = new Map<EngineKind, Backend>()
  private options: Required<Omit<RuntimeOptions, 'memory' | 'persist'>> & { memory?: MemoryPolicy }
  /** This runtime's `persist()` cache: the process default unless the constructor was given its own. */
  readonly persist: PersistCache
  /**
   * Lineage: `withEngine()` forks share their origin's root (same cache, backends and tenant). Frames on runtimes
   * with different roots belong to different owners and are not combined implicitly — see `resolveRuntime`.
   */
  readonly root: Runtime = this

  constructor(options: RuntimeOptions = {}) {
    this.options = {
      engine: options.engine ?? 'auto',
      webgpuMinRows: options.webgpuMinRows ?? DEFAULT_WEBGPU_MIN_ROWS,
      wasmMinRows: options.wasmMinRows ?? DEFAULT_WASM_MIN_ROWS,
      preferGpu: options.preferGpu ?? true,
      strict: options.strict ?? false,
      memory: options.memory,
    }
    this.persist = options.persist ?? defaultPersistCache
    this.register(new CpuBackend())
    // `memory` is this runtime's policy, applied per execution; it never touches the process default.
  }

  /** The memory policy this runtime applies to every execution (call-site `collect({ memory })` overrides it). */
  get memoryPolicy(): MemoryPolicy | undefined {
    return this.options.memory
  }

  register(backend: Backend): void {
    this.backends.set(backend.name, backend)
  }

  /** Registered backends (CPU first). Backends are shareable: `withEngine` forks and sessions reuse them. */
  listBackends(): Backend[] {
    return [...this.backends.values()]
  }

  setEngine(engine: EngineKind): void {
    this.options.engine = engine
  }

  /** Replace this runtime's memory policy (instance only; `setMemoryPolicy()` from the package sets the process default). */
  setMemoryPolicy(policy: MemoryPolicy): void {
    this.options.memory = policy
  }

  /** Immutable-ish fork with a forced engine (does not mutate this instance). */
  withEngine(engine: EngineKind, options: { strict?: boolean } = {}): Runtime {
    const rt = new Runtime({
      ...this.options,
      engine,
      strict: options.strict ?? this.options.strict,
      persist: this.persist,
    })
    for (const backend of this.backends.values()) {
      if (backend.name !== 'cpu') rt.register(backend)
    }
    ;(rt as { root: Runtime }).root = this.root
    return rt
  }

  explain(plan: PlanNode): string {
    const optimized = optimizePlan(plan)
    const chosen = this.chooseBackend(optimized)
    return `Engine: ${chosen.name} (planned — the backend a node actually ran on is only known after execution; see collectWithReport())\n${explainPlan(optimized)}`
  }

  chooseBackend(plan: PlanNode): Backend {
    if (this.options.engine !== 'auto') {
      const forced = this.backends.get(this.options.engine)
      if (!forced) throw new Error(`Engine "${this.options.engine}" is not registered`)
      if (!forced.supports(plan)) {
        if (this.options.strict) throw new EngineStrictError(this.options.engine, ['backend does not support this plan'])
        const cpu = this.backends.get('cpu')!
        return cpu
      }
      return forced
    }

    const rows = estimateRows(plan)
    const gpu = this.backends.get('webgpu')
    // Hybrid WebGPU: accelerate numeric filter/map subtrees even when other columns are utf8.
    if (this.options.preferGpu && gpu && rows >= this.options.webgpuMinRows && gpu.supports(plan)) {
      return gpu
    }

    const wasm = this.backends.get('wasm')
    if (wasm && rows >= this.options.wasmMinRows && wasm.supports(plan)) {
      return wasm
    }

    return this.backends.get('cpu')!
  }

  async execute(plan: PlanNode, opts?: ExecuteOptions): Promise<TableView> {
    return (await this.executeWithReport(plan, opts)).table
  }

  /**
   * Execute and return what actually ran: per-node backend, kernels, fallback reasons and timings.
   * In strict mode a plan whose requested engine executed no node throws `EngineStrictError`.
   */
  async executeWithReport(
    plan: PlanNode,
    opts?: ExecuteOptions,
  ): Promise<{ table: TableView; report: ExecutionReport }> {
    const memory = opts?.memory ?? this.options.memory
    const guard = makeGuard(opts)
    checkGuard(guard, 'start')
    if (memory?.maxBytes || getMemoryPolicy().maxBytes) await ensureSpillSupport()
    return withMemoryPolicyAsync(memory, async () => {
      resetExecMemoryStats()
      const rawPlan = plan
      plan = optimizePlan(plan)
      checkGuard(guard, 'optimize')
      const requested = this.options.engine
      const strict = this.options.strict
      // A strict request is answered from the cache only with a table the requested engine produced;
      // otherwise the plan runs and either that engine executes it or EngineStrictError is raised.
      const probe = this.persist.probe(plan, { requested, strict })
      if (probe.status === 'hit') {
        const mem = getExecMemoryStats()
        return {
          table: probe.table,
          report: {
            requested,
            dispatched: probe.provenance.dispatched,
            strict,
            events: [],
            fallbacks: [],
            totalMs: 0,
            backendsUsed: [...probe.provenance.backendsUsed],
            spilledBytes: mem.spilledBytes,
            peakBytes: mem.peakBytes,
            cacheHit: true,
            cachedFrom: { ...probe.provenance, backendsUsed: [...probe.provenance.backendsUsed] },
          },
        }
      }
      const out = await this.executeWithReportInner(plan, rawPlan, guard)
      if (probe.status === 'skip') out.report.cacheSkipped = probe.reason
      return out
    })
  }

  private async executeWithReportInner(
    plan: PlanNode,
    rawPlan?: PlanNode,
    guard?: ExecContext['guard'],
  ): Promise<{ table: TableView; report: ExecutionReport }> {
    const requested = this.options.engine
    const strict = this.options.strict
    const backend = this.chooseBackend(plan)
    const events: ExecutionEvent[] = []
    const fallbacks: ExecutionReport['fallbacks'] = []
    const ctx: ExecContext = { requested, strict, trace: (e) => void events.push(e), guard }
    if (rawPlan && joinOrderChanged(rawPlan, plan)) {
      events.push({
        node: 'join',
        backend: 'cpu',
        kernel: 'optimized:joinReorder',
        reason: 'inner join build/order changed by optimizePlan',
      })
    }
    const t0 = now()
    const finish = (table: TableView, dispatched: EngineKind): { table: TableView; report: ExecutionReport } => {
      const backendsUsed = [...new Set(events.map((e) => e.backend))]
      const mem = getExecMemoryStats()
      const report: ExecutionReport = {
        requested,
        dispatched,
        strict,
        events,
        fallbacks,
        totalMs: now() - t0,
        backendsUsed,
        spilledBytes: mem.spilledBytes,
        peakBytes: mem.peakBytes,
        cacheHit: false,
      }
      if (strict && requested !== 'auto' && requested !== 'cpu' && !backendsUsed.includes(requested)) {
        const reasons = [
          ...fallbacks.map((f) => `${f.from} → ${f.to}: ${f.reason}`),
          ...events.filter((e) => e.reason).map((e) => `${e.node}: ${e.reason}`),
        ]
        throw new EngineStrictError(requested, reasons)
      }
      // stored only after the strict contract held, with what produced it
      this.persist.maybeStore(plan, table, { requested, dispatched, strict, backendsUsed })
      return { table, report }
    }
    const run = async (b: Backend): Promise<TableView> => {
      checkGuard(guard, b.name)
      const r = b.execute(plan, ctx)
      const table = r instanceof Promise ? await r : r
      checkGuard(guard, 'finish')
      return table
    }
    try {
      return finish(await run(backend), backend.name)
    } catch (err) {
      if (err instanceof EngineStrictError || err instanceof ExecutionAbortedError) throw err
      if (backend.name === 'cpu') throw err
      const reason = err instanceof Error ? err.message : String(err)
      if (strict) throw new EngineStrictError(backend.name, [`backend threw: ${reason}`])
      // Fallback chain: webgpu → wasm → cpu
      if (backend.name === 'webgpu') {
        const wasm = this.backends.get('wasm')
        if (wasm?.supports(plan)) {
          fallbacks.push({ from: 'webgpu', to: 'wasm', reason })
          try {
            return finish(await run(wasm), 'wasm')
          } catch (err2) {
            fallbacks.push({ from: 'wasm', to: 'cpu', reason: err2 instanceof Error ? err2.message : String(err2) })
            return finish(await run(this.backends.get('cpu')!), 'cpu')
          }
        }
      }
      fallbacks.push({ from: backend.name, to: 'cpu', reason })
      return finish(await run(this.backends.get('cpu')!), 'cpu')
    }
  }
}

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now())

let defaultRuntime: Runtime | null = null
/** Roots of every runtime that has served as the process default (frames built without a runtime point there). */
const processDefaultRoots = new WeakSet<Runtime>()

export function getDefaultRuntime(): Runtime {
  if (!defaultRuntime) defaultRuntime = new Runtime()
  processDefaultRoots.add(defaultRuntime.root)
  return defaultRuntime
}

export function setDefaultRuntime(runtime: Runtime): void {
  defaultRuntime = runtime
  processDefaultRoots.add(runtime.root)
}

/** True for the process default runtime and its `withEngine` forks — i.e. a frame nobody bound to a session. */
export function isProcessDefaultRuntime(runtime: Runtime): boolean {
  return processDefaultRoots.has(runtime.root)
}

/** Frames bound to different runtimes (sessions / tenants) were combined without saying which runtime runs the result. */
export class RuntimeMismatchError extends Error {
  constructor(
    readonly operation: string,
    readonly runtimes: readonly Runtime[],
  ) {
    super(
      `${operation}: the inputs are bound to ${runtimes.length} different runtimes (sessions). Rebind them first — ` +
        `session.bind(frame) — or pass { runtime } to choose the runtime that executes the result.`,
    )
    this.name = 'RuntimeMismatchError'
  }
}

/**
 * The runtime a multi-input operation (concat, join, cross / as-of join) runs on:
 *  1. an explicit `runtime` wins;
 *  2. inputs on the process default runtime are unbound and adopt the others' runtime;
 *  3. bound inputs must share one lineage (`root`), else `RuntimeMismatchError` — two tenants never mix implicitly;
 *  4. within that lineage the receiver's runtime (its `engine()` fork) is kept when it belongs to it;
 *  5. all inputs unbound: the receiver's runtime, else the first input's.
 * The result never falls back to the process default when any input is bound.
 */
export function resolveRuntime(
  operation: string,
  inputs: readonly Runtime[],
  options: { receiver?: Runtime; explicit?: Runtime } = {},
): Runtime {
  if (options.explicit) return options.explicit
  const bound = inputs.filter((r) => !isProcessDefaultRuntime(r))
  const roots = [...new Set(bound.map((r) => r.root))]
  if (roots.length > 1) throw new RuntimeMismatchError(operation, roots)
  if (roots.length === 1) {
    if (options.receiver && options.receiver.root === roots[0]) return options.receiver
    return bound[0]!
  }
  return options.receiver ?? inputs[0] ?? getDefaultRuntime()
}

export * from './types.js'
export * from './cpu.js'
export { ExecutionAbortedError, yieldToEventLoop, type ExecGuard } from './cancel.js'
export {
  setMemoryPolicy,
  getMemoryPolicy,
  clearMemoryPolicy,
  estimateTableBytes,
} from './memory_api.js'
export type { MemoryPolicy } from './memory.js'
export {
  PersistCache,
  defaultPersistCache,
  type PersistCacheOptions,
  type PersistCacheStats,
  type PersistOptions,
  type PersistProvenance,
  type PersistProbe,
  hashPlan,
  storePersistCache,
  dropPersistCache,
  clearPersistCache,
  persistCacheStats,
  markPlanPersist,
  unmarkPlanPersist,
} from './persist.js'
export { tryLoadNativeKernels, isNativeKernelsLoaded, setNativeKernels, NATIVE_FILTER_MIN_ROWS, NATIVE_SORT_MIN_ROWS, NATIVE_SORT_MULTI_MIN_ROWS, NATIVE_FILTER_GENERIC_MIN_ROWS, NATIVE_UNIQUE_MIN_ROWS, NATIVE_JOIN_BUILD_MIN_ROWS } from './native_kernels.js'
export { optimizePlan, estimatePlanRows, planOutputColumns, joinOrderChanged } from './optimize.js'
export { pushdownProjections, exprColumnRefs } from './pushdown.js'
export {
  approxNdv,
  estimateFilterSelectivity,
  sampleIndices,
} from './stats.js'
export {
  PARALLEL_MIN_ROWS,
  PARALLEL_GATHER_MIN_ROWS,
  PARALLEL_FILTER_MIN_ROWS,
  PARALLEL_SORT_MIN_ROWS,
  PARALLEL_GROUPBY_MIN_ROWS,
  PARALLEL_UNIQUE_MIN_ROWS,
  parallelDualGtIndices,
  parallelTakeTable,
  parallelFilter,
  parallelSort,
  parallelGroupBy,
  parallelUnique,
  closeParallelPool,
} from './parallel.js'
