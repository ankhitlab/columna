/**
 * Soft memory budget for live tables and spill / persist behaviour.
 * Spill is Node-only; browsers ignore spill or refuse `spill: true`.
 */
export type MemoryPolicy = {
  /** Soft budget for live TableView bytes (estimateTableBytes). */
  maxBytes?: number
  /** Directory for spill files. Default: os.tmpdir()/columna-spill */
  spillDir?: string
  /**
   * When true (default if `maxBytes` is set on Node), oversized sort/unique/join
   * intermediates may be written to disk. Browser: leave false / unset.
   */
  spill?: boolean
  /** LRU cap for explicit `persist()` cache entries (Phase 3). */
  maxCacheBytes?: number
}

type MemoryExecutionState = {
  policy: MemoryPolicy
  spilledBytes: number
  peakBytes: number
}

interface AsyncLocalStorageLike<T> {
  getStore(): T | undefined
  run<R>(store: T, callback: () => R): R
}

type AsyncLocalStorageConstructor = new <T>() => AsyncLocalStorageLike<T>

let globalPolicy: MemoryPolicy = {}

let fallbackSpilledBytes = 0
let fallbackPeakBytes = 0

let nodeStorage: AsyncLocalStorageLike<MemoryExecutionState> | null = null
let nodeStorageInit: Promise<AsyncLocalStorageLike<MemoryExecutionState>> | null = null

let browserOverrideState: MemoryExecutionState | null = null

async function ensureNodeStorage(): Promise<AsyncLocalStorageLike<MemoryExecutionState>> {
  if (nodeStorage) return nodeStorage

  if (!nodeStorageInit) {
    nodeStorageInit = (async () => {
      const id = 'node:async_hooks'

      const mod = (await import(
        /* @vite-ignore */
        id
      )) as unknown as {
        AsyncLocalStorage: AsyncLocalStorageConstructor
      }

      nodeStorage = new mod.AsyncLocalStorage<MemoryExecutionState>()

      return nodeStorage
    })()
  }

  return nodeStorageInit
}

function activeState(): MemoryExecutionState | undefined {
  return nodeStorage?.getStore() ?? browserOverrideState ?? undefined
}

function activePolicy(): MemoryPolicy {
  return activeState()?.policy ?? globalPolicy
}

export function setMemoryPolicy(policy: MemoryPolicy): void {
  if (policy.spill === true && !isNode()) {
    throw new Error('MemoryPolicy.spill is only supported on Node.js')
  }
  globalPolicy = { ...policy }
}

export function getMemoryPolicy(): MemoryPolicy {
  return { ...activePolicy() }
}

export function clearMemoryPolicy(): void {
  globalPolicy = {}
}

function mergePolicy(base: MemoryPolicy, override: MemoryPolicy): MemoryPolicy {
  const merged: MemoryPolicy = { ...base, ...override }
  if (merged.spill === true && !isNode()) {
    throw new Error('MemoryPolicy.spill is only supported on Node.js')
  }
  if (merged.maxBytes != null && merged.spill === undefined) {
    merged.spill = isNode()
  }
  return merged
}

/** Merge call-site overrides onto the process policy for one execution. */
export function withMemoryPolicy<T>(override: MemoryPolicy | undefined, fn: () => T): T {
  if (!override || Object.keys(override).length === 0) return fn()
  const prev = globalPolicy
  globalPolicy = mergePolicy(prev, override)
  try {
    return fn()
  } finally {
    globalPolicy = prev
  }
}

export async function withMemoryPolicyAsync<T>(
  override: MemoryPolicy | undefined,
  fn: () => Promise<T>,
): Promise<T> {
  const base = activePolicy()
  const policy = mergePolicy(base, override ?? {})

  const state: MemoryExecutionState = {
    policy,
    spilledBytes: 0,
    peakBytes: 0,
  }

  if (isNode()) {
    const storage = await ensureNodeStorage()
    return storage.run(state, fn)
  }

  const hasOverride = override !== undefined && Object.keys(override).length > 0

  // Browser has no portable AsyncLocalStorage equivalent.
  // Do not silently mix per-call overrides.
  if (!hasOverride) {
    return fn()
  }

  if (browserOverrideState) {
    throw new Error('Concurrent collect({ memory }) overrides are not supported in the browser')
  }

  browserOverrideState = state

  try {
    return await fn()
  } finally {
    browserOverrideState = null
  }
}

export function resetExecMemoryStats(): void {
  const state = activeState()

  if (state) {
    state.spilledBytes = 0
    state.peakBytes = 0
    return
  }

  fallbackSpilledBytes = 0
  fallbackPeakBytes = 0
}

export function recordLiveBytes(bytes: number): void {
  const state = activeState()

  if (state) {
    if (bytes > state.peakBytes) {
      state.peakBytes = bytes
    }
    return
  }

  if (bytes > fallbackPeakBytes) {
    fallbackPeakBytes = bytes
  }
}

export function recordSpilledBytes(bytes: number): void {
  const state = activeState()

  if (state) {
    state.spilledBytes += bytes
    return
  }

  fallbackSpilledBytes += bytes
}

export function getExecMemoryStats(): {
  spilledBytes: number
  peakBytes: number
} {
  const state = activeState()

  if (state) {
    return {
      spilledBytes: state.spilledBytes,
      peakBytes: state.peakBytes,
    }
  }

  return {
    spilledBytes: fallbackSpilledBytes,
    peakBytes: fallbackPeakBytes,
  }
}

export function isNode(): boolean {
  return typeof process !== 'undefined' && !!process.versions?.node
}

/** Spill is active when maxBytes is set and spill is not explicitly false (Node). */
export function spillEnabled(): boolean {
  const policy = activePolicy()

  if (policy.maxBytes == null || policy.maxBytes <= 0) return false
  if (policy.spill === false) return false
  if (!isNode()) return false

  return policy.spill === true || policy.spill === undefined
}

export function memoryBudget(): number | undefined {
  const value = activePolicy().maxBytes
  return value != null && value > 0 ? value : undefined
}

/** Empty string means spill.ts should use os.tmpdir()/columna-spill. */
export function resolveSpillDir(): string {
  return activePolicy().spillDir ?? ''
}
