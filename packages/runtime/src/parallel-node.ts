/**
 * Node-only parallel kernels via a reusable worker_threads pool + SharedArrayBuffer.
 * Imported dynamically from parallel.ts so browser bundles never pull node:*.
 */
import { cpus } from 'node:os'
import { Worker } from 'node:worker_threads'
import { fileURLToPath } from 'node:url'
import { dualGtIndices } from './fast.js'

type NumArr = Float64Array | Float32Array | Int32Array | Uint32Array | Uint8Array
type ArrKind = 'Float64Array' | 'Float32Array' | 'Int32Array' | 'Uint32Array' | 'Uint8Array'

type Pending = {
  resolve: (v: unknown) => void
  reject: (e: Error) => void
}

type Pool = {
  workers: Worker[]
  rr: number
  pending: Map<number, Pending>
  nextId: number
}

let pool: Pool | null = null

function workerUrl(): string | null {
  try {
    return fileURLToPath(new URL('./filter-worker.js', import.meta.url))
  } catch {
    return null
  }
}

function arrKind(arr: NumArr): ArrKind {
  if (arr instanceof Float64Array) return 'Float64Array'
  if (arr instanceof Float32Array) return 'Float32Array'
  if (arr instanceof Int32Array) return 'Int32Array'
  if (arr instanceof Uint8Array) return 'Uint8Array'
  return 'Uint32Array'
}

function toShared(arr: NumArr): { buffer: SharedArrayBuffer; kind: ArrKind; length: number } {
  const kind = arrKind(arr)
  const buf = arr.buffer
  if (
    buf instanceof SharedArrayBuffer &&
    arr.byteOffset === 0 &&
    arr.byteLength === buf.byteLength
  ) {
    return { buffer: buf, kind, length: arr.length }
  }
  const sab = new SharedArrayBuffer(arr.length * arr.BYTES_PER_ELEMENT)
  const Ctor = arr.constructor as new (buffer: SharedArrayBuffer) => NumArr
  new Ctor(sab).set(arr as never)
  return { buffer: sab, kind, length: arr.length }
}

function destroyPool(): void {
  if (!pool) return
  for (const w of pool.workers) void w.terminate()
  for (const [, p] of pool.pending) p.reject(new Error('worker pool destroyed'))
  pool = null
}

function ensurePool(): Pool | null {
  if (pool) return pool
  const url = workerUrl()
  if (!url) return null
  const threads = Math.min(Math.max(1, cpus().length), 4)
  if (threads <= 1) return null

  const workers: Worker[] = []
  const pending = new Map<number, Pending>()
  const nextPool: Pool = { workers, rr: 0, pending, nextId: 1 }

  for (let t = 0; t < threads; t++) {
    const worker = new Worker(url)
    worker.on('message', (msg: { id: number; ok: boolean; indices?: Uint32Array; count?: number; error?: string }) => {
      const pend = pending.get(msg.id)
      if (!pend) return
      pending.delete(msg.id)
      if (msg.ok) pend.resolve(msg.indices ?? msg.count ?? true)
      else pend.reject(new Error(msg.error ?? 'worker job failed'))
    })
    worker.on('error', (err) => {
      destroyPool()
      for (const [, p] of pending) p.reject(err instanceof Error ? err : new Error(String(err)))
    })
    workers.push(worker)
  }
  pool = nextPool
  return pool
}

function postJob<T>(job: Record<string, unknown>, transfer?: ArrayBuffer[]): Promise<T> {
  const p = ensurePool()
  if (!p) return Promise.reject(new Error('no worker pool'))
  const id = p.nextId++
  const worker = p.workers[p.rr++ % p.workers.length]!
  return new Promise<T>((resolve, reject) => {
    p.pending.set(id, {
      resolve: resolve as (v: unknown) => void,
      reject,
    })
    worker.postMessage({ ...job, id }, transfer ?? [])
  })
}

export async function runParallelDualGt(
  a: NumArr,
  b: NumArr,
  la: number,
  lb: number,
): Promise<Uint32Array> {
  const n = a.length
  const p = ensurePool()
  if (!p) {
    return dualGtIndices(
      a as Float64Array | Float32Array | Int32Array | Uint32Array,
      b as Float64Array | Float32Array | Int32Array | Uint32Array,
      la,
      lb,
    )
  }

  const threads = p.workers.length
  const chunk = Math.ceil(n / threads)
  const sharedA = toShared(a)
  const sharedB = toShared(b)
  const maskBuf = new SharedArrayBuffer(n)
  const mask = new Uint8Array(maskBuf)

  try {
    const counts = await Promise.all(
      Array.from({ length: threads }, (_, t) => {
        const start = t * chunk
        const end = Math.min(n, start + chunk)
        if (start >= end) return Promise.resolve(0)
        return postJob<number>({
          type: 'dualGt',
          aBuffer: sharedA.buffer,
          bBuffer: sharedB.buffer,
          aKind: sharedA.kind,
          bKind: sharedB.kind,
          aLength: sharedA.length,
          bLength: sharedB.length,
          maskBuffer: maskBuf,
          start,
          end,
          la,
          lb,
        })
      }),
    )

    let total = 0
    for (const c of counts) total += c
    const out = new Uint32Array(total)
    let j = 0
    for (let i = 0; i < n; i++) {
      out[j] = i
      j += mask[i]!
    }
    return out
  } catch {
    destroyPool()
    return dualGtIndices(
      a as Float64Array | Float32Array | Int32Array | Uint32Array,
      b as Float64Array | Float32Array | Int32Array | Uint32Array,
      la,
      lb,
    )
  }
}

export type GatherSpec = {
  src: NumArr
  /** Pre-allocated output buffer (same length as indices). */
  out: NumArr
}

/**
 * Gather many typed columns in parallel. `indices` shared once; each column written into `out`.
 * Prefer SharedArrayBuffer-backed `out` (zero copy back). Falls back to sync on failure.
 */
export async function runParallelGather(indices: Uint32Array, specs: GatherSpec[]): Promise<void> {
  if (specs.length === 0) return
  const p = ensurePool()
  if (!p) {
    for (const s of specs) {
      for (let i = 0; i < indices.length; i++) s.out[i] = s.src[indices[i]!]! as never
    }
    return
  }

  const idxShared = toShared(indices)
  try {
    await Promise.all(
      specs.map(async (s) => {
        const src = toShared(s.src)
        let outBuf: SharedArrayBuffer
        const outKind = arrKind(s.out)
        if (s.out.buffer instanceof SharedArrayBuffer && s.out.byteOffset === 0) {
          outBuf = s.out.buffer
        } else {
          outBuf = new SharedArrayBuffer(s.out.length * s.out.BYTES_PER_ELEMENT)
          const tmp = new (s.out.constructor as new (buffer: SharedArrayBuffer) => NumArr)(outBuf)
          await postJob({
            type: 'gather',
            srcBuffer: src.buffer,
            srcKind: src.kind,
            srcLength: src.length,
            idxBuffer: idxShared.buffer,
            idxLength: indices.length,
            outBuffer: outBuf,
            outKind,
          })
          s.out.set(tmp as never)
          return
        }
        await postJob({
          type: 'gather',
          srcBuffer: src.buffer,
          srcKind: src.kind,
          srcLength: src.length,
          idxBuffer: idxShared.buffer,
          idxLength: indices.length,
          outBuffer: outBuf,
          outKind,
        })
      }),
    )
  } catch {
    destroyPool()
    for (const s of specs) {
      for (let i = 0; i < indices.length; i++) s.out[i] = s.src[indices[i]!]! as never
    }
  }
}
