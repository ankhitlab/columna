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

let globalPolicy: MemoryPolicy = {}
let spilledBytesThisExec = 0
let peakBytesThisExec = 0

export function setMemoryPolicy(policy: MemoryPolicy): void {
  if (policy.spill === true && !isNode()) {
    throw new Error('MemoryPolicy.spill is only supported on Node.js')
  }
  globalPolicy = { ...policy }
}

export function getMemoryPolicy(): MemoryPolicy {
  return { ...globalPolicy }
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
  if (!override || Object.keys(override).length === 0) return fn()
  const prev = globalPolicy
  globalPolicy = mergePolicy(prev, override)
  try {
    return await fn()
  } finally {
    globalPolicy = prev
  }
}

export function resetExecMemoryStats(): void {
  spilledBytesThisExec = 0
  peakBytesThisExec = 0
}

export function recordLiveBytes(bytes: number): void {
  if (bytes > peakBytesThisExec) peakBytesThisExec = bytes
}

export function recordSpilledBytes(bytes: number): void {
  spilledBytesThisExec += bytes
}

export function getExecMemoryStats(): { spilledBytes: number; peakBytes: number } {
  return { spilledBytes: spilledBytesThisExec, peakBytes: peakBytesThisExec }
}

export function isNode(): boolean {
  return typeof process !== 'undefined' && !!process.versions?.node
}

/** Spill is active when maxBytes is set and spill is not explicitly false (Node). */
export function spillEnabled(): boolean {
  const p = globalPolicy
  if (p.maxBytes == null || p.maxBytes <= 0) return false
  if (p.spill === false) return false
  if (!isNode()) return false
  return p.spill === true || p.spill === undefined
}

export function memoryBudget(): number | undefined {
  const b = globalPolicy.maxBytes
  return b != null && b > 0 ? b : undefined
}

/** Empty string means spill.ts should use os.tmpdir()/columna-spill. */
export function resolveSpillDir(): string {
  return globalPolicy.spillDir ?? ''
}
