import type { TableView } from '@columna/arrow'
import { CpuBackend } from './cpu.js'
import {
  DEFAULT_WASM_MIN_ROWS,
  DEFAULT_WEBGPU_MIN_ROWS,
  estimateRows,
  explainPlan,
  type Backend,
  type EngineKind,
  type PlanNode,
  type RuntimeOptions,
} from './types.js'

export class Runtime {
  private backends = new Map<EngineKind, Backend>()
  private options: Required<RuntimeOptions>

  constructor(options: RuntimeOptions = {}) {
    this.options = {
      engine: options.engine ?? 'auto',
      webgpuMinRows: options.webgpuMinRows ?? DEFAULT_WEBGPU_MIN_ROWS,
      wasmMinRows: options.wasmMinRows ?? DEFAULT_WASM_MIN_ROWS,
      preferGpu: options.preferGpu ?? true,
    }
    this.register(new CpuBackend())
  }

  register(backend: Backend): void {
    this.backends.set(backend.name, backend)
  }

  setEngine(engine: EngineKind): void {
    this.options.engine = engine
  }

  /** Immutable-ish fork with a forced engine (does not mutate this instance). */
  withEngine(engine: EngineKind): Runtime {
    const rt = new Runtime({ ...this.options, engine })
    for (const backend of this.backends.values()) {
      if (backend.name !== 'cpu') rt.register(backend)
    }
    return rt
  }

  explain(plan: PlanNode): string {
    const chosen = this.chooseBackend(plan)
    return `Engine: ${chosen.name}\n${explainPlan(plan)}`
  }

  chooseBackend(plan: PlanNode): Backend {
    if (this.options.engine !== 'auto') {
      const forced = this.backends.get(this.options.engine)
      if (!forced) throw new Error(`Engine "${this.options.engine}" is not registered`)
      if (!forced.supports(plan)) {
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

  async execute(plan: PlanNode): Promise<TableView> {
    const backend = this.chooseBackend(plan)
    try {
      return await backend.execute(plan)
    } catch (err) {
      if (backend.name === 'cpu') throw err
      // Fallback chain: webgpu → wasm → cpu
      if (backend.name === 'webgpu') {
        const wasm = this.backends.get('wasm')
        if (wasm?.supports(plan)) {
          try {
            return await wasm.execute(plan)
          } catch {
            return this.backends.get('cpu')!.execute(plan)
          }
        }
      }
      return this.backends.get('cpu')!.execute(plan)
    }
  }
}

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
export { tryLoadNativeKernels, isNativeKernelsLoaded, setNativeKernels, NATIVE_FILTER_MIN_ROWS } from './native_kernels.js'
