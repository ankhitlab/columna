/**
 * Chunked parallel dual-gt filter + typed gather.
 * Browser / non-Node always uses sync kernels. Node optionally loads
 * parallel-node.js (worker_threads) via a dynamic import that bundlers ignore.
 *
 * Prefer `@columna/native` Rayon kernels (via dualGtIndices) when loaded;
 * worker_threads cover the no-native case from ~1M rows up (SharedArrayBuffer,
 * zero-copy — cheaper than the old structured-clone path that needed ≥50M).
 */
import {
  takeColumn,
  tableFromColumns,
  type Column,
  type TableView,
} from '@columna/arrow'
import { dualGtIndices } from './fast.js'
import { isNativeKernelsLoaded, NATIVE_FILTER_MIN_ROWS } from './native_kernels.js'

/** Dual-gt via workers when native is unavailable. SAB zero-copy; 1M amortizes pool spin-up. */
export const PARALLEL_MIN_ROWS = 1_000_000

/** Opaque specifier: bundlers must not inline the worker_threads module into browser builds. */
async function loadParallelNode(): Promise<typeof import('./parallel-node.js')> {
  if (typeof process === 'undefined' || !process.versions?.node) throw new Error('worker threads need Node')
  const id = './parallel-node.js'
  return (await import(/* @vite-ignore */ id)) as typeof import('./parallel-node.js')
}

/** Browser Web Worker pool — opaque specifier so Node bundles never pull DOM APIs. */
async function loadParallelWeb(): Promise<typeof import('./parallel-web.js')> {
  if (typeof Worker === 'undefined') throw new Error('web workers unavailable')
  const id = './parallel-web.js'
  return (await import(/* @vite-ignore */ id)) as typeof import('./parallel-web.js')
}

type ParallelModule = {
  runParallelDualGt: (a: NumArr, b: NumArr, la: number, lb: number) => Promise<Uint32Array>
  runParallelGather: (table: unknown, indices: Uint32Array, specs: unknown) => Promise<void>
  runParallelFilter: (cols: NumArr[], ops: CmpOp[], lits: number[], n: number) => Promise<Uint32Array>
  runParallelSort: (keys: unknown, n: number) => Promise<Uint32Array>
  runParallelGroupBy: (keyCols: NumArr[], aggCols: NumArr[], aggs: unknown, n: number) => Promise<unknown>
  runParallelUnique: (cols: NumArr[], n: number) => Promise<Uint32Array>
  closePool: () => void
}

/** Load whichever parallel backend the current runtime supports (Node or browser). */
async function loadParallel(): Promise<ParallelModule | null> {
  if (isNodeRuntime()) {
    try { return (await loadParallelNode()) as unknown as ParallelModule } catch { return null }
  }
  if (typeof Worker !== 'undefined') {
    try { return (await loadParallelWeb()) as unknown as ParallelModule } catch { return null }
  }
  return null
}

/** Parallel typed gather via workers — only for huge takes. */
export const PARALLEL_GATHER_MIN_ROWS = 2_000_000

/** Generic multi-cmp filter via workers (when native is unavailable). */
export const PARALLEL_FILTER_MIN_ROWS = 1_000_000

/** Parallel sort via workers. */
export const PARALLEL_SORT_MIN_ROWS = 5_000_000

/** Parallel groupBy via workers. */
export const PARALLEL_GROUPBY_MIN_ROWS = 5_000_000

/** Parallel unique via workers. */
export const PARALLEL_UNIQUE_MIN_ROWS = 5_000_000

type NumArr = Float64Array | Float32Array | Int32Array | Uint32Array
type CmpOp = 'eq' | 'neq' | 'gt' | 'gte' | 'lt' | 'lte'

function isNodeRuntime(): boolean {
  return typeof process !== 'undefined' && Boolean(process.versions?.node)
}

/** True when any worker backend (Node threads or browser Web Workers) could run. */
function canParallel(): boolean {
  return isNodeRuntime() || typeof Worker !== 'undefined'
}

export type DualFilterKernel = 'workers:dualFilter' | 'native:dualFilter' | 'js:dualFilter'

/**
 * Build match indices for a > la && b > lb.
 * Order: native Rayon (if loaded) → worker pool (≥ PARALLEL_MIN_ROWS) → JS.
 * `minRows` overrides PARALLEL_MIN_ROWS (useful in tests).
 */
export async function parallelDualGtIndices(
  a: NumArr,
  b: NumArr,
  la: number,
  lb: number,
  opts?: { minRows?: number },
): Promise<{ indices: Uint32Array; kernel: DualFilterKernel }> {
  const n = a.length
  const minRows = opts?.minRows ?? PARALLEL_MIN_ROWS

  // In-process Rayon beats worker IPC whenever the addon is present.
  if (n >= NATIVE_FILTER_MIN_ROWS && isNativeKernelsLoaded()) {
    return { indices: dualGtIndices(a, b, la, lb), kernel: 'native:dualFilter' }
  }

  if (n >= minRows && canParallel()) {
    try {
      const mod = await loadParallel()
      if (mod) {
        const indices = await mod.runParallelDualGt(a, b, la, lb)
        return { indices, kernel: 'workers:dualFilter' }
      }
    } catch {
      // fall through
    }
  }

  return { indices: dualGtIndices(a, b, la, lb), kernel: 'js:dualFilter' }
}

function isTypedNoNull(col: Column): boolean {
  if (col.nullBitmap) return false
  const d = col.field.dtype
  return (
    d === 'f64' ||
    d === 'f32' ||
    d === 'i32' ||
    d === 'u32' ||
    d === 'bool' ||
    d === 'category' ||
    d === 'datetime'
  )
}

function allocLike(col: Column, n: number): NumArr | Uint8Array {
  const d = col.field.dtype
  const canSab = typeof SharedArrayBuffer !== 'undefined'
  if (canSab) {
    if (d === 'f64' || d === 'datetime') return new Float64Array(new SharedArrayBuffer(n * 8))
    if (d === 'f32') return new Float32Array(new SharedArrayBuffer(n * 4))
    if (d === 'i32') return new Int32Array(new SharedArrayBuffer(n * 4))
    if (d === 'bool') return new Uint8Array(new SharedArrayBuffer(n))
    return new Uint32Array(new SharedArrayBuffer(n * 4))
  }
  if (d === 'f64' || d === 'datetime') return new Float64Array(n)
  if (d === 'f32') return new Float32Array(n)
  if (d === 'i32') return new Int32Array(n)
  if (d === 'bool') return new Uint8Array(n)
  return new Uint32Array(n)
}

/**
 * takeTable with parallel typed gathers on Node when output is large enough.
 */
export async function parallelTakeTable(
  table: TableView,
  indices: ArrayLike<number>,
  keep?: readonly string[],
  opts?: { minRows?: number },
): Promise<TableView> {
  const cols =
    keep && keep.length > 0
      ? keep.map((name) => {
          const col = table.columns.find((c) => c.field.name === name)
          if (!col) throw new Error(`Unknown column "${name}" in parallelTakeTable`)
          return col
        })
      : table.columns

  const n = indices.length
  if (n === table.numRows && cols.length === table.columns.length) {
    let identity = true
    for (let i = 0; i < n; i++) {
      if ((indices[i] as number) !== i) {
        identity = false
        break
      }
    }
    if (identity) return table
  }

  const minRows = opts?.minRows ?? PARALLEL_GATHER_MIN_ROWS
  const typed = cols.filter(isTypedNoNull)
  if (n < minRows || typed.length < 2 || !isNodeRuntime()) {
    return tableFromColumns(cols.map((c) => takeColumn(c, indices)))
  }

  const idx =
    indices instanceof Uint32Array
      ? indices
      : Uint32Array.from({ length: n }, (_, i) => indices[i] as number)

  try {
    const mod = await loadParallelNode()
    const outs = new Map<Column, NumArr | Uint8Array>()
    const specs = typed.map((col) => {
      const out = allocLike(col, n)
      outs.set(col, out)
      return { src: col.data as NumArr | Uint8Array, out }
    })
    await mod.runParallelGather(idx, specs)

    return tableFromColumns(
      cols.map((c) => {
        if (!outs.has(c)) return takeColumn(c, idx)
        return {
          field: c.field,
          data: outs.get(c)!,
          dictionary: c.dictionary,
        }
      }),
    )
  } catch {
    return tableFromColumns(cols.map((c) => takeColumn(c, indices)))
  }
}

// --- Protocol v2 API: parallelFilter, parallelSort, parallelGroupBy, parallelUnique ---

/** Parallel generic filter via workers; falls back to sync below threshold or no workers. */
export async function parallelFilter(
  cols: NumArr[],
  ops: CmpOp[],
  lits: number[],
  n: number,
  opts?: { minRows?: number },
): Promise<Uint32Array> {
  const minRows = opts?.minRows ?? PARALLEL_FILTER_MIN_ROWS
  if (n < minRows || !canParallel()) {
    return syncFilterIndices(cols, ops, lits, n)
  }
  const mod = await loadParallel()
  if (!mod) return syncFilterIndices(cols, ops, lits, n)
  try {
    return await mod.runParallelFilter(cols, ops, lits, n)
  } catch {
    return syncFilterIndices(cols, ops, lits, n)
  }
}

function syncFilterIndices(cols: NumArr[], ops: CmpOp[], lits: number[], n: number): Uint32Array {
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    let hit = true
    for (let c = 0; c < cols.length; c++) {
      if (!syncCmp(ops[c]!, cols[c]![i]!, lits[c]!)) { hit = false; break }
    }
    if (hit) out.push(i)
  }
  return new Uint32Array(out)
}

function syncCmp(op: CmpOp, v: number, lit: number): boolean {
  switch (op) {
    case 'eq': return v === lit
    case 'neq': return v !== lit
    case 'gt': return v > lit
    case 'gte': return v >= lit
    case 'lt': return v < lit
    case 'lte': return v <= lit
  }
}

/** Parallel sort via workers; falls back to a real sync sort below threshold or no workers. */
export async function parallelSort(
  keys: Array<{ buffer: SharedArrayBuffer; kind: string; length: number; descending: boolean; nullsLast: boolean; nullBitmap?: SharedArrayBuffer | null }>,
  n: number,
  opts?: { minRows?: number },
): Promise<Uint32Array> {
  const minRows = opts?.minRows ?? PARALLEL_SORT_MIN_ROWS
  if (n < minRows || !canParallel()) {
    return syncSortFromSpecs(keys, n)
  }
  const mod = await loadParallel()
  if (!mod) return syncSortFromSpecs(keys, n)
  try {
    return await mod.runParallelSort(keys as never, n)
  } catch {
    return syncSortFromSpecs(keys, n)
  }
}

function viewSpec(kind: string, buffer: SharedArrayBuffer, length: number): ArrayLike<number> {
  switch (kind) {
    case 'Float64Array': return new Float64Array(buffer, 0, length)
    case 'Float32Array': return new Float32Array(buffer, 0, length)
    case 'Int32Array': return new Int32Array(buffer, 0, length)
    case 'Uint32Array': return new Uint32Array(buffer, 0, length)
    case 'Uint8Array': return new Uint8Array(buffer, 0, length)
    default: return new Float64Array(buffer, 0, length)
  }
}

function syncSortFromSpecs(
  keys: Array<{ buffer: SharedArrayBuffer; kind: string; length: number; descending: boolean; nullsLast: boolean; nullBitmap?: SharedArrayBuffer | null }>,
  n: number,
): Uint32Array {
  const idx = new Uint32Array(n)
  for (let i = 0; i < n; i++) idx[i] = i
  const views = keys.map((k) => ({
    data: viewSpec(k.kind, k.buffer, k.length),
    nullBitmap: k.nullBitmap ? new Uint8Array(k.nullBitmap, 0, k.length) : null,
    descending: k.descending,
    nullsLast: k.nullsLast,
  }))
  idx.sort((a, b) => {
    for (let k = 0; k < views.length; k++) {
      const kv = views[k]!
      const aNull = kv.nullBitmap ? !kv.nullBitmap[a] : false
      const bNull = kv.nullBitmap ? !kv.nullBitmap[b] : false
      if (aNull || bNull) {
        if (aNull && bNull) continue
        return kv.nullsLast ? (aNull ? 1 : -1) : (aNull ? -1 : 1)
      }
      const va = kv.data[a]!
      const vb = kv.data[b]!
      if (va === vb) continue
      const cmp = va < vb ? -1 : 1
      return kv.descending ? -cmp : cmp
    }
    return 0
  })
  return idx
}

/** Parallel groupBy via workers; falls back to sync (null) below threshold or no workers. */
export async function parallelGroupBy(
  keyCols: NumArr[],
  aggCols: NumArr[],
  aggs: Array<{ op: 'sum' | 'mean' | 'count' | 'min' | 'max'; colIdx: number }>,
  n: number,
  opts?: { minRows?: number },
) {
  const minRows = opts?.minRows ?? PARALLEL_GROUPBY_MIN_ROWS
  if (n < minRows || !canParallel()) return null
  const mod = await loadParallel()
  if (!mod) return null
  try {
    return await mod.runParallelGroupBy(keyCols, aggCols, aggs as never, n)
  } catch {
    return null
  }
}

/** Parallel unique via workers; falls back to sync (null) below threshold or no workers. */
export async function parallelUnique(
  cols: NumArr[],
  n: number,
  opts?: { minRows?: number },
): Promise<Uint32Array | null> {
  const minRows = opts?.minRows ?? PARALLEL_UNIQUE_MIN_ROWS
  if (n < minRows || !canParallel()) return null
  const mod = await loadParallel()
  if (!mod) return null
  try {
    return await mod.runParallelUnique(cols, n)
  } catch {
    return null
  }
}

/** Graceful pool shutdown (for tests / process exit). */
export async function closeParallelPool(): Promise<void> {
  if (!canParallel()) return
  try {
    const mod = await loadParallel()
    mod?.closePool()
  } catch {
    // not loaded — nothing to close
  }
}
