/**
 * Browser-only parallel kernels via a reusable Web Worker pool + SharedArrayBuffer.
 * Imported dynamically from parallel.ts so Node bundles never pull DOM APIs.
 * Mirrors parallel-node.ts dispatch but uses `new Worker(url)` and `self`-style
 * messaging. Requires cross-origin isolation (COOP/COEP) for SharedArrayBuffer;
 * falls back to single-threaded sync when SAB or workers are unavailable.
 */
import { dualGtIndices } from './fast.js'
import {
  type EngineJob,
  type ArrKind,
  type CmpOp,
  type SortKeySpec,
  type AggSpec,
} from './engine-jobs.js'

type NumArr = Float64Array | Float32Array | Int32Array | Uint32Array | Uint8Array

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void }
type Pool = { workers: Worker[]; rr: number; pending: Map<number, Pending>; nextId: number }

let pool: Pool | null = null

function sabAvailable(): boolean {
  try {
    // eslint-disable-next-line no-new
    new SharedArrayBuffer(1)
    return true
  } catch {
    return false
  }
}

function workerUrl(): string | null {
  try {
    const url = new URL('./engine-worker-web.js', import.meta.url)
    return url.href
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
  if (buf instanceof SharedArrayBuffer && arr.byteOffset === 0 && arr.byteLength === buf.byteLength) {
    return { buffer: buf, kind, length: arr.length }
  }
  const sab = new SharedArrayBuffer(arr.length * arr.BYTES_PER_ELEMENT)
  const Ctor = arr.constructor as new (buffer: SharedArrayBuffer) => NumArr
  new Ctor(sab).set(arr as never)
  return { buffer: sab, kind, length: arr.length }
}

function viewLocal(kind: ArrKind, buffer: SharedArrayBuffer, length: number): ArrayLike<number> & { [i: number]: number } {
  switch (kind) {
    case 'Float64Array': return new Float64Array(buffer, 0, length)
    case 'Float32Array': return new Float32Array(buffer, 0, length)
    case 'Int32Array': return new Int32Array(buffer, 0, length)
    case 'Uint32Array': return new Uint32Array(buffer, 0, length)
    case 'Uint8Array': return new Uint8Array(buffer, 0, length)
  }
}

function cmpHolds(op: CmpOp, v: number, lit: number): boolean {
  switch (op) {
    case 'eq': return v === lit
    case 'neq': return v !== lit
    case 'gt': return v > lit
    case 'gte': return v >= lit
    case 'lt': return v < lit
    case 'lte': return v <= lit
  }
}

export function destroyPool(): void {
  if (!pool) return
  for (const w of pool.workers) void w.terminate()
  for (const [, p] of pool.pending) p.reject(new Error('worker pool destroyed'))
  pool = null
}

function ensurePool(): Pool | null {
  if (pool) return pool
  if (typeof Worker === 'undefined') return null
  if (!sabAvailable()) return null
  const url = workerUrl()
  if (!url) return null
  const cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4
  const threads = Math.min(Math.max(1, cores - 1), 4)
  if (threads <= 1) return null

  const workers: Worker[] = []
  const pending = new Map<number, Pending>()
  const nextPool: Pool = { workers, rr: 0, pending, nextId: 1 }

  for (let t = 0; t < threads; t++) {
    const worker = new Worker(url, { type: 'module' })
    worker.onmessage = (ev: MessageEvent) => {
      const msg = ev.data as { id: number; ok: boolean; indices?: Uint32Array; count?: number; data?: Float64Array; error?: string }
      const pend = pending.get(msg.id)
      if (!pend) return
      pending.delete(msg.id)
      if (msg.ok) pend.resolve(msg.indices ?? msg.data ?? msg.count ?? true)
      else pend.reject(new Error(msg.error ?? 'worker job failed'))
    }
    worker.onerror = (err) => {
      destroyPool()
      for (const [, p] of pending) p.reject(err instanceof Error ? err : new Error(String(err)))
    }
    workers.push(worker)
  }
  pool = nextPool
  return pool
}

function postJob<T>(job: Omit<EngineJob, 'id'>, transfer: Transferable[] = []): Promise<T> {
  const p = ensurePool()
  if (!p) return Promise.reject(new Error('no worker pool'))
  const id = p.nextId++
  const worker = p.workers[p.rr++ % p.workers.length]!
  return new Promise<T>((resolve, reject) => {
    p.pending.set(id, { resolve: resolve as (v: unknown) => void, reject })
    worker.postMessage({ ...job, id }, transfer)
  })
}

type NumArrNoU8 = Float64Array | Float32Array | Int32Array | Uint32Array

export async function runParallelDualGt(
  a: NumArr,
  b: NumArr,
  la: number,
  lb: number,
): Promise<Uint32Array> {
  const n = a.length
  const p = ensurePool()
  if (!p) return dualGtIndices(a as NumArrNoU8, b as NumArrNoU8, la, lb)

  const threads = p.workers.length
  const chunk = Math.ceil(n / threads)
  const aShared = toShared(a)
  const bShared = toShared(b)
  const maskBuf = new SharedArrayBuffer(n)
  const mask = new Uint8Array(maskBuf)

  try {
    await Promise.all(
      Array.from({ length: threads }, (_, t) => {
        const start = t * chunk
        const end = Math.min(n, start + chunk)
        if (start >= end) return Promise.resolve()
        return postJob<number>({
          type: 'dualGt',
          aBuffer: aShared.buffer, aKind: aShared.kind, aLength: aShared.length,
          bBuffer: bShared.buffer, bKind: bShared.kind, bLength: bShared.length,
          la, lb, maskBuffer: maskBuf, start, end,
        })
      }),
    )
    let total = 0
    for (let i = 0; i < n; i++) if (mask[i]) total++
    const out = new Uint32Array(total)
    let j = 0
    for (let i = 0; i < n; i++) if (mask[i]) out[j++] = i
    return out
  } catch {
    destroyPool()
    return dualGtIndices(a as NumArrNoU8, b as NumArrNoU8, la, lb)
  }
}

export async function runParallelGather(
  _table: unknown,
  indices: Uint32Array,
  specs: Array<{ src: NumArr; out: NumArr }>,
): Promise<void> {
  const p = ensurePool()
  if (!p) {
    for (const s of specs) for (let i = 0; i < indices.length; i++) s.out[i] = s.src[indices[i]!]! as never
    return
  }
  const idxBuf = new SharedArrayBuffer(indices.length * 4)
  new Uint32Array(idxBuf).set(indices)
  await Promise.all(
    specs.map((s) => {
      const srcShared = toShared(s.src)
      const outBuf = new SharedArrayBuffer(s.out.length * s.out.BYTES_PER_ELEMENT)
      return postJob<void>({
        type: 'gather',
        srcBuffer: srcShared.buffer, srcKind: srcShared.kind, srcLength: srcShared.length,
        idxBuffer: idxBuf, idxLength: indices.length,
        outBuffer: outBuf, outKind: arrKind(s.src),
      }).then(() => {
        const Ctor = s.out.constructor as new (buffer: SharedArrayBuffer) => NumArr
        const view = new Ctor(outBuf)
        for (let i = 0; i < s.out.length; i++) s.out[i] = view[i]! as never
      })
    }),
  )
}

export async function runParallelFilter(
  cols: NumArr[],
  ops: CmpOp[],
  lits: number[],
  n: number,
): Promise<Uint32Array> {
  const p = ensurePool()
  if (!p) return syncFilter(cols, ops, lits, n)

  const threads = p.workers.length
  const chunk = Math.ceil(n / threads)
  const sharedCols = cols.map((c) => toShared(c))
  const maskBuf = new SharedArrayBuffer(n)
  const mask = new Uint8Array(maskBuf)

  try {
    await Promise.all(
      Array.from({ length: threads }, (_, t) => {
        const start = t * chunk
        const end = Math.min(n, start + chunk)
        if (start >= end) return Promise.resolve(0)
        return postJob<number>({
          type: 'filter',
          cols: sharedCols.map((s) => ({ buffer: s.buffer, kind: s.kind, length: s.length })),
          ops, lits, maskBuffer: maskBuf, start, end,
        })
      }),
    )
    let total = 0
    for (let i = 0; i < n; i++) if (mask[i]) total++
    const out = new Uint32Array(total)
    let j = 0
    for (let i = 0; i < n; i++) if (mask[i]) out[j++] = i
    return out
  } catch {
    destroyPool()
    return syncFilter(cols, ops, lits, n)
  }
}

function syncFilter(cols: NumArr[], ops: CmpOp[], lits: number[], n: number): Uint32Array {
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    let hit = true
    for (let c = 0; c < cols.length; c++) {
      if (!cmpHolds(ops[c]!, cols[c]![i]!, lits[c]!)) { hit = false; break }
    }
    if (hit) out.push(i)
  }
  return new Uint32Array(out)
}

export async function runParallelSort(keys: SortKeySpec[], n: number): Promise<Uint32Array> {
  const p = ensurePool()
  if (!p) return syncSort(keys, n)

  const threads = p.workers.length
  const chunk = Math.ceil(n / threads)

  try {
    const chunkResults = await Promise.all(
      Array.from({ length: threads }, async (_, t) => {
        const start = t * chunk
        const end = Math.min(n, start + chunk)
        if (start >= end) return new Uint32Array(0)
        const outBuf = new SharedArrayBuffer((end - start) * 4)
        const r = await postJob<Uint32Array>({
          type: 'sortChunk', keys, outIdxBuffer: outBuf, start, end,
        })
        return (r as unknown as Uint32Array) ?? new Uint32Array(outBuf, 0, end - start)
      }),
    )
    return kWayMerge(chunkResults, keys)
  } catch {
    destroyPool()
    return syncSort(keys, n)
  }
}

function syncSort(keys: SortKeySpec[], n: number): Uint32Array {
  const idx = new Uint32Array(n)
  for (let i = 0; i < n; i++) idx[i] = i
  const keyViews = keys.map((k) => ({
    data: viewLocal(k.kind, k.buffer, k.length),
    nullBitmap: k.nullBitmap ? new Uint8Array(k.nullBitmap, 0, k.length) : null,
    descending: k.descending, nullsLast: k.nullsLast,
  }))
  idx.sort((a, b) => compareKeys(keyViews, a, b))
  return idx
}

function compareKeys(
  keyViews: Array<{ data: ArrayLike<number>; nullBitmap: Uint8Array | null; descending: boolean; nullsLast: boolean }>,
  a: number, b: number,
): number {
  for (let k = 0; k < keyViews.length; k++) {
    const kv = keyViews[k]!
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
}

function kWayMerge(chunks: Uint32Array[], keys: SortKeySpec[]): Uint32Array {
  const total = chunks.reduce((s, c) => s + c.length, 0)
  const out = new Uint32Array(total)
  const keyViews = keys.map((k) => ({
    data: viewLocal(k.kind, k.buffer, k.length),
    nullBitmap: k.nullBitmap ? new Uint8Array(k.nullBitmap, 0, k.length) : null,
    descending: k.descending, nullsLast: k.nullsLast,
  }))
  const pos = new Int32Array(chunks.length)
  let outPos = 0
  while (outPos < total) {
    let bestChunk = -1
    let bestIdx = -1
    for (let c = 0; c < chunks.length; c++) {
      if (pos[c]! >= chunks[c]!.length) continue
      const idx = chunks[c]![pos[c]!]!
      if (bestChunk < 0) { bestChunk = c; bestIdx = idx; continue }
      if (compareKeys(keyViews, idx, bestIdx) < 0) { bestChunk = c; bestIdx = idx }
    }
    if (bestChunk < 0) break
    out[outPos++] = bestIdx
    pos[bestChunk]!++
  }
  return out
}

export type ParallelGroupResult = {
  data: Float64Array
  groupCount: number
  keyCount: number
  aggCount: number
}

const PER_AGG = 5

export async function runParallelGroupBy(
  keyCols: NumArr[],
  aggCols: NumArr[],
  aggs: AggSpec[],
  n: number,
): Promise<ParallelGroupResult> {
  const p = ensurePool()
  if (!p) return syncGroupBy(keyCols, aggCols, aggs, n)

  const threads = p.workers.length
  const chunk = Math.ceil(n / threads)
  const sharedKeys = keyCols.map((c) => toShared(c))
  const sharedAggs = aggCols.map((c) => toShared(c))

  try {
    const results = await Promise.all(
      Array.from({ length: threads }, (_, t) => {
        const start = t * chunk
        const end = Math.min(n, start + chunk)
        if (start >= end) return Promise.resolve({ data: new Float64Array(0), count: 0 })
        return postJob<{ data: Float64Array; count: number }>({
          type: 'groupByChunk',
          keyCols: sharedKeys.map((s) => ({ buffer: s.buffer, kind: s.kind, length: s.length })),
          aggCols: sharedAggs.map((s) => ({ buffer: s.buffer, kind: s.kind, length: s.length })),
          aggs, start, end,
        })
      }),
    )
    return mergePartialGroups(results, keyCols.length, aggs.length)
  } catch {
    destroyPool()
    return syncGroupBy(keyCols, aggCols, aggs, n)
  }
}

function syncGroupBy(
  keyCols: NumArr[], aggCols: NumArr[], aggs: AggSpec[], n: number,
): ParallelGroupResult {
  const groups = new Map<string, Float64Array>()
  for (let i = 0; i < n; i++) {
    let key = ''
    for (const kc of keyCols) key += kc[i]! + '\0'
    let accs = groups.get(key)
    if (!accs) {
      accs = new Float64Array(aggs.length * PER_AGG)
      for (let a = 0; a < aggs.length; a++) {
        accs[a * PER_AGG + 2] = Infinity
        accs[a * PER_AGG + 3] = -Infinity
      }
      groups.set(key, accs)
    }
    accumulateRow(accs, aggs, aggCols, i)
  }
  return serializeGroups(groups, keyCols.length, aggs.length)
}

function accumulateRow(accs: Float64Array, aggs: AggSpec[], aggCols: NumArr[], i: number): void {
  for (let a = 0; a < aggs.length; a++) {
    const spec = aggs[a]!
    const base = a * PER_AGG
    if (spec.op === 'count') { accs[base]! += 1; continue }
    const v = spec.colIdx >= 0 ? aggCols[spec.colIdx]![i]! : 0
    const c = accs[base]!
    const sum = accs[base + 1]!
    const oldMean = c > 0 ? sum / c : 0
    accs[base]! = c + 1
    accs[base + 1]! = sum + v
    if (v < accs[base + 2]!) accs[base + 2]! = v
    if (v > accs[base + 3]!) accs[base + 3]! = v
    accs[base + 4]! = accs[base + 4]! + (v - oldMean) * (v - (sum + v) / (c + 1))
  }
}

function serializeGroups(
  groups: Map<string, Float64Array>, keyCount: number, aggCount: number,
): ParallelGroupResult {
  const rowLen = 1 + keyCount + aggCount * PER_AGG
  const data = new Float64Array(groups.size * rowLen)
  let pos = 0
  for (const [keyStr, accs] of groups) {
    const parts = keyStr.split('\0').filter((p) => p !== '')
    data[pos++] = parts.length
    for (const part of parts) data[pos++] = Number(part)
    for (let a = 0; a < aggCount; a++) {
      const base = a * PER_AGG
      data[pos++] = accs[base]!
      data[pos++] = accs[base + 1]!
      data[pos++] = accs[base + 2]!
      data[pos++] = accs[base + 3]!
      data[pos++] = accs[base + 4]!
    }
  }
  return { data, groupCount: groups.size, keyCount, aggCount }
}

function mergePartialGroups(
  results: Array<{ data: Float64Array; count: number }>,
  keyCount: number, aggCount: number,
): ParallelGroupResult {
  const merged = new Map<string, Float64Array>()
  for (const { data } of results) {
    let pos = 0
    while (pos < data.length) {
      const parts = data[pos++]! as number
      const keyParts: number[] = []
      for (let p = 0; p < parts; p++) keyParts.push(data[pos++]!)
      const key = keyParts.join('\0')
      let accs = merged.get(key)
      if (!accs) {
        accs = new Float64Array(aggCount * PER_AGG)
        for (let a = 0; a < aggCount; a++) {
          accs[a * PER_AGG + 2] = Infinity
          accs[a * PER_AGG + 3] = -Infinity
        }
        merged.set(key, accs)
      }
      for (let a = 0; a < aggCount; a++) {
        const base = a * PER_AGG
        const count = data[pos++]!
        const sum = data[pos++]!
        const min = data[pos++]!
        const max = data[pos++]!
        const m2 = data[pos++]!
        if (count === 0) continue
        const oldCount = accs[base]!
        if (oldCount === 0) {
          accs[base]! = count
          accs[base + 1]! = sum
          accs[base + 2]! = min
          accs[base + 3]! = max
          accs[base + 4]! = m2
        } else {
          const oldSum = accs[base + 1]!
          const n1 = oldCount, n2 = count
          const mean1 = oldSum / n1, mean2 = sum / n2
          accs[base]! = n1 + n2
          accs[base + 1]! = oldSum + sum
          if (min < accs[base + 2]!) accs[base + 2]! = min
          if (max > accs[base + 3]!) accs[base + 3]! = max
          accs[base + 4]! = accs[base + 4]! + m2 + ((mean1 - mean2) * (mean1 - mean2) * n1 * n2) / (n1 + n2)
        }
      }
    }
  }
  return { data: serializeGroupsData(merged, keyCount, aggCount), groupCount: merged.size, keyCount, aggCount }
}

function serializeGroupsData(
  groups: Map<string, Float64Array>, keyCount: number, aggCount: number,
): Float64Array {
  const rowLen = 1 + keyCount + aggCount * PER_AGG
  const data = new Float64Array(groups.size * rowLen)
  let pos = 0
  for (const [keyStr, accs] of groups) {
    const parts = keyStr.split('\0').filter((p) => p !== '')
    data[pos++] = parts.length
    for (const part of parts) data[pos++] = Number(part)
    for (let a = 0; a < aggCount; a++) {
      const base = a * PER_AGG
      data[pos++] = accs[base]!
      data[pos++] = accs[base + 1]!
      data[pos++] = accs[base + 2]!
      data[pos++] = accs[base + 3]!
      data[pos++] = accs[base + 4]!
    }
  }
  return data
}

export async function runParallelUnique(cols: NumArr[], n: number): Promise<Uint32Array> {
  const p = ensurePool()
  if (!p) return syncUnique(cols, n)

  const threads = p.workers.length
  const chunk = Math.ceil(n / threads)
  const sharedCols = cols.map((c) => toShared(c))

  try {
    const results = await Promise.all(
      Array.from({ length: threads }, (_, t) => {
        const start = t * chunk
        const end = Math.min(n, start + chunk)
        if (start >= end) return Promise.resolve(new Uint32Array(0))
        return postJob<Uint32Array>({
          type: 'uniqueChunk',
          uniqCols: sharedCols.map((s) => ({ buffer: s.buffer, kind: s.kind, length: s.length })),
          start, end,
        })
      }),
    )
    const seen = new Set<string>()
    const out: number[] = []
    for (const chunkIdx of results) {
      for (let i = 0; i < chunkIdx.length; i++) {
        const row = chunkIdx[i]!
        let key = ''
        for (const c of cols) key += c[row]! + '\0'
        if (!seen.has(key)) { seen.add(key); out.push(row) }
      }
    }
    return new Uint32Array(out)
  } catch {
    destroyPool()
    return syncUnique(cols, n)
  }
}

function syncUnique(cols: NumArr[], n: number): Uint32Array {
  const seen = new Set<string>()
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    let key = ''
    for (const c of cols) key += c[i]! + '\0'
    if (!seen.has(key)) { seen.add(key); out.push(i) }
  }
  return new Uint32Array(out)
}

export function closePool(): void {
  destroyPool()
}
