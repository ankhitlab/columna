/**
 * Chunked parallel dual-gt filter + typed gather.
 * Browser / non-Node always uses sync kernels. Node optionally loads
 * parallel-node.js (worker_threads) via a dynamic import that bundlers ignore.
 *
 * Prefer `@columna/native` Rayon kernels (via dualGtIndices) for large dual-gt;
 * worker_threads stay as a last resort for enormous n.
 */
import {
  takeColumn,
  tableFromColumns,
  type Column,
  type TableView,
} from '@columna/arrow'
import { dualGtIndices } from './fast.js'

/** Dual-gt via worker_threads — IPC usually loses; keep threshold very high. */
export const PARALLEL_MIN_ROWS = 50_000_000

/** Opaque specifier: bundlers must not inline the worker_threads module into browser builds. */
async function loadParallelNode(): Promise<typeof import('./parallel-node.js')> {
  if (typeof process === 'undefined' || !process.versions?.node) throw new Error('worker threads need Node')
  const id = './parallel-node.js'
  return (await import(/* @vite-ignore */ id)) as typeof import('./parallel-node.js')
}

/** Parallel typed gather via workers — only for huge takes. */
export const PARALLEL_GATHER_MIN_ROWS = 2_000_000

type NumArr = Float64Array | Float32Array | Int32Array | Uint32Array

function isNodeRuntime(): boolean {
  return typeof process !== 'undefined' && Boolean(process.versions?.node)
}

/**
 * Build match indices for a > la && b > lb in parallel chunks when possible.
 * `minRows` overrides PARALLEL_MIN_ROWS (useful in tests).
 */
export async function parallelDualGtIndices(
  a: NumArr,
  b: NumArr,
  la: number,
  lb: number,
  opts?: { minRows?: number },
): Promise<Uint32Array> {
  const n = a.length
  const minRows = opts?.minRows ?? PARALLEL_MIN_ROWS
  if (n < minRows || !isNodeRuntime()) {
    return dualGtIndices(a, b, la, lb)
  }

  try {
    const mod = await loadParallelNode()
    return await mod.runParallelDualGt(a, b, la, lb)
  } catch {
    return dualGtIndices(a, b, la, lb)
  }
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
