import type { TableView } from '@columna/arrow'
import { CpuBackend } from './cpu.js'
import {
  getExecMemoryStats,
  getMemoryPolicy,
  resetExecMemoryStats,
  setMemoryPolicy,
  withMemoryPolicyAsync,
  type MemoryPolicy,
} from './memory.js'
import { ensureSpillSupport } from './spill.js'
import { lookupPersistCache, maybeStorePersist, type PersistLookup } from './persist.js'
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
  type ExecutionEvent,
  type ExecutionReport,
  type PlanNode,
  type RuntimeOptions,
} from './types.js'

export class Runtime {
  private backends = new Map<EngineKind, Backend>()
  private options: Required<Omit<RuntimeOptions, 'memory'>> & { memory?: MemoryPolicy }

  constructor(options: RuntimeOptions = {}) {
    this.options = {
      engine: options.engine ?? 'auto',
      webgpuMinRows: options.webgpuMinRows ?? DEFAULT_WEBGPU_MIN_ROWS,
      wasmMinRows: options.wasmMinRows ?? DEFAULT_WASM_MIN_ROWS,
      preferGpu: options.preferGpu ?? true,
      strict: options.strict ?? false,
      memory: options.memory,
    }
    this.register(new CpuBackend())
    if (options.memory) setMemoryPolicy(options.memory)
  }

  register(backend: Backend): void {
    this.backends.set(backend.name, backend)
  }

  setEngine(engine: EngineKind): void {
    this.options.engine = engine
  }

  setMemoryPolicy(policy: MemoryPolicy): void {
    this.options.memory = policy
    setMemoryPolicy(policy)
  }

  /** Immutable-ish fork with a forced engine (does not mutate this instance). */
  withEngine(engine: EngineKind, options: { strict?: boolean } = {}): Runtime {
    const rt = new Runtime({
      ...this.options,
      engine,
      strict: options.strict ?? this.options.strict,
    })
    for (const backend of this.backends.values()) {
      if (backend.name !== 'cpu') rt.register(backend)
    }
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

  async execute(plan: PlanNode, opts?: { memory?: MemoryPolicy }): Promise<TableView> {
    return (await this.executeWithReport(plan, opts)).table
  }

  /**
   * Execute and return what actually ran: per-node backend, kernels, fallback reasons and timings.
   * In strict mode a plan whose requested engine executed no node throws `EngineStrictError`.
   */
  async executeWithReport(
    plan: PlanNode,
    opts?: { memory?: MemoryPolicy },
  ): Promise<{ table: TableView; report: ExecutionReport }> {
    const memory = opts?.memory ?? this.options.memory
    if (memory?.maxBytes || getMemoryPolicy().maxBytes) await ensureSpillSupport()
    return withMemoryPolicyAsync(memory, async () => {
      resetExecMemoryStats()
      const rawPlan = plan
      plan = optimizePlan(plan)
      const cached = lookupPersistCache(plan)
      if (cached) {
        const mem = getExecMemoryStats()
        return {
          table: cached.table,
          report: {
            requested: this.options.engine,
            dispatched: 'cpu',
            strict: this.options.strict,
            events: [],
            fallbacks: [],
            totalMs: 0,
            backendsUsed: ['cpu'],
            spilledBytes: mem.spilledBytes,
            peakBytes: mem.peakBytes,
            cacheHit: true,
          },
        }
      }
      return this.executeWithReportInner(plan, rawPlan)
    })
  }

  private async executeWithReportInner(
    plan: PlanNode,
    rawPlan?: PlanNode,
  ): Promise<{ table: TableView; report: ExecutionReport }> {
    const requested = this.options.engine
    const strict = this.options.strict
    const backend = this.chooseBackend(plan)
    const events: ExecutionEvent[] = []
    const fallbacks: ExecutionReport['fallbacks'] = []
    const ctx: ExecContext = { requested, strict, trace: (e) => void events.push(e) }
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
      maybeStorePersist(plan, table)
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
      return { table, report }
    }
    const run = async (b: Backend): Promise<TableView> => {
      const r = b.execute(plan, ctx)
      return r instanceof Promise ? r : Promise.resolve(r)
    }
    try {
      return finish(await run(backend), backend.name)
    } catch (err) {
      if (err instanceof EngineStrictError) throw err
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

export function getDefaultRuntime(): Runtime {
  if (!defaultRuntime) defaultRuntime = new Runtime()
  return defaultRuntime
}

export function setDefaultRuntime(runtime: Runtime): void {
  defaultRuntime = runtime
}

export * from './types.js'
export * from './cpu.js'
export {
  setMemoryPolicy,
  getMemoryPolicy,
  clearMemoryPolicy,
  estimateTableBytes,
} from './memory_api.js'
export type { MemoryPolicy } from './memory.js'
export {
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
