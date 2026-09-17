/**
 * Shared job handlers for the parallel engine — environment-agnostic.
 * Used by both the Node `engine-worker.ts` (parentPort) and the browser
 * `engine-worker-web.ts` (self) wrappers. Protocol v2.
 */

export type ArrKind = 'Float64Array' | 'Float32Array' | 'Int32Array' | 'Uint32Array' | 'Uint8Array'
export type CmpOp = 'eq' | 'neq' | 'gt' | 'gte' | 'lt' | 'lte'

export type BufferSpec = {
  buffer: SharedArrayBuffer
  kind: ArrKind
  length: number
}

export type SortKeySpec = {
  buffer: SharedArrayBuffer
  kind: ArrKind
  length: number
  descending: boolean
  nullsLast: boolean
  nullBitmap?: SharedArrayBuffer | null
}

export type AggSpec = {
  op: 'sum' | 'mean' | 'count' | 'min' | 'max'
  /** Column index in cols array that this agg reads; -1 for count. */
  colIdx: number
}

export type EngineJob = {
  id: number
  type: 'dualGt' | 'gather' | 'filter' | 'sortChunk' | 'groupByChunk' | 'uniqueChunk' | 'ping'
  // dualGt
  aBuffer?: SharedArrayBuffer
  bBuffer?: SharedArrayBuffer
  aKind?: ArrKind
  bKind?: ArrKind
  aLength?: number
  bLength?: number
  start?: number
  end?: number
  la?: number
  lb?: number
  maskBuffer?: SharedArrayBuffer
  // gather
  srcBuffer?: SharedArrayBuffer
  srcKind?: ArrKind
  srcLength?: number
  idxBuffer?: SharedArrayBuffer
  idxLength?: number
  outBuffer?: SharedArrayBuffer
  outKind?: ArrKind
  // filter (generic)
  cols?: BufferSpec[]
  ops?: CmpOp[]
  lits?: number[]
  // sortChunk
  keys?: SortKeySpec[]
  outIdxBuffer?: SharedArrayBuffer
  // groupByChunk
  keyCols?: BufferSpec[]
  aggCols?: BufferSpec[]
  aggs?: AggSpec[]
  // uniqueChunk
  uniqCols?: BufferSpec[]
  outIdxStart?: number
}

export interface WorkerPort {
  postMessage(msg: unknown, transfer?: Transferable[]): void
}

export function view(kind: ArrKind, buffer: SharedArrayBuffer, length: number): ArrayLike<number> & { [i: number]: number } {
  switch (kind) {
    case 'Float64Array':
      return new Float64Array(buffer, 0, length)
    case 'Float32Array':
      return new Float32Array(buffer, 0, length)
    case 'Int32Array':
      return new Int32Array(buffer, 0, length)
    case 'Uint32Array':
      return new Uint32Array(buffer, 0, length)
    case 'Uint8Array':
      return new Uint8Array(buffer, 0, length)
  }
}

export function cmpHolds(op: CmpOp, v: number, lit: number): boolean {
  switch (op) {
    case 'eq':
      return v === lit
    case 'neq':
      return v !== lit
    case 'gt':
      return v > lit
    case 'gte':
      return v >= lit
    case 'lt':
      return v < lit
    case 'lte':
      return v <= lit
  }
}

/** Dispatch a single job; replies via `port.postMessage`. */
export function handleJob(job: EngineJob, port: WorkerPort): void {
  try {
    if (job.type === 'ping') {
      port.postMessage({ id: job.id, ok: true })
      return
    }

    if (job.type === 'dualGt') {
      const aFull = view(job.aKind!, job.aBuffer!, job.aLength!)
      const bFull = view(job.bKind!, job.bBuffer!, job.bLength!)
      const mask = new Uint8Array(job.maskBuffer!, 0, job.aLength!)
      const start = job.start!
      const end = job.end!
      const la = job.la!
      const lb = job.lb!
      let count = 0
      for (let i = start; i < end; i++) {
        const hit = (aFull[i]! > la ? 1 : 0) & (bFull[i]! > lb ? 1 : 0)
        mask[i] = hit
        count += hit
      }
      port.postMessage({ id: job.id, ok: true, count })
      return
    }

    if (job.type === 'gather') {
      const src = view(job.srcKind!, job.srcBuffer!, job.srcLength!)
      const idx = new Uint32Array(job.idxBuffer!, 0, job.idxLength!)
      const out = view(job.outKind!, job.outBuffer!, job.idxLength!)
      const n = job.idxLength!
      for (let i = 0; i < n; i++) out[i] = src[idx[i]!]!
      port.postMessage({ id: job.id, ok: true })
      return
    }

    if (job.type === 'filter') {
      const cols = job.cols!
      const ops = job.ops!
      const lits = job.lits!
      const mask = new Uint8Array(job.maskBuffer!, 0, cols[0]!.length)
      const start = job.start!
      const end = job.end!
      let count = 0
      for (let i = start; i < end; i++) {
        let hit = 1
        for (let c = 0; c < cols.length; c++) {
          const col = view(cols[c]!.kind, cols[c]!.buffer, cols[c]!.length)
          const v = col[i]!
          if (!cmpHolds(ops[c]!, v, lits[c]!)) {
            hit = 0
            break
          }
        }
        mask[i] = hit
        count += hit
      }
      port.postMessage({ id: job.id, ok: true, count })
      return
    }

    if (job.type === 'sortChunk') {
      const keys = job.keys!
      const start = job.start!
      const end = job.end!
      const outIdx = new Uint32Array(job.outIdxBuffer!, 0, end - start)
      for (let i = 0; i < end - start; i++) outIdx[i] = start + i
      const keyViews = keys.map((k) => ({
        data: view(k.kind, k.buffer, k.length),
        nullBitmap: k.nullBitmap ? new Uint8Array(k.nullBitmap, 0, k.length) : null,
        descending: k.descending,
        nullsLast: k.nullsLast,
      }))
      outIdx.sort((a, b) => {
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
      })
      port.postMessage({ id: job.id, ok: true, indices: outIdx })
      return
    }

    if (job.type === 'groupByChunk') {
      const keyCols = job.keyCols!
      const aggCols = job.aggCols!
      const aggs = job.aggs!
      const start = job.start!
      const end = job.end!
      const keyViews = keyCols.map((c) => view(c.kind, c.buffer, c.length))
      const aggViews = aggCols.map((c) => view(c.kind, c.buffer, c.length))

      const groups = new Map<string, Float64Array>()
      for (let i = start; i < end; i++) {
        let key = ''
        for (const kv of keyViews) key += kv[i]! + '\0'
        let accs = groups.get(key)
        if (!accs) {
          accs = new Float64Array(aggs.length * 5)
          for (let a = 0; a < aggs.length; a++) {
            accs[a * 5 + 2] = Infinity
            accs[a * 5 + 3] = -Infinity
          }
          groups.set(key, accs)
        }
        for (let a = 0; a < aggs.length; a++) {
          const spec = aggs[a]!
          const base = a * 5
          if (spec.op === 'count') {
            accs[base]! += 1
            continue
          }
          const v = spec.colIdx >= 0 ? aggViews[spec.colIdx]![i]! : 0
          const c = accs[base]!
          const sum = accs[base + 1]!
          const oldMean = c > 0 ? sum / c : 0
          accs[base]! = c + 1
          accs[base + 1]! = sum + v
          if (v < accs[base + 2]!) accs[base + 2]! = v
          if (v > accs[base + 3]!) accs[base + 3]! = v
          const newMean = (sum + v) / (c + 1)
          accs[base + 4]! = accs[base + 4]! + (v - oldMean) * (v - newMean)
        }
      }

      const entries: Array<[string, Float64Array]> = [...groups]
      const keyPartCount = keyCols.length
      const perAgg = 5
      const rowLen = 1 + keyPartCount + aggs.length * perAgg
      const data = new Float64Array(entries.length * rowLen)
      let pos = 0
      for (const [keyStr, accs] of entries) {
        const parts = keyStr.split('\0').filter((p) => p !== '')
        data[pos++] = parts.length
        for (const p of parts) data[pos++] = Number(p)
        for (let a = 0; a < aggs.length; a++) {
          const base = a * 5
          data[pos++] = accs[base]!
          data[pos++] = accs[base + 1]!
          data[pos++] = accs[base + 2]!
          data[pos++] = accs[base + 3]!
          data[pos++] = accs[base + 4]!
        }
      }
      port.postMessage({ id: job.id, ok: true, data, count: entries.length })
      return
    }

    if (job.type === 'uniqueChunk') {
      const cols = job.uniqCols!
      const start = job.start!
      const end = job.end!
      const colViews = cols.map((c) => view(c.kind, c.buffer, c.length))
      const seen = new Set<string>()
      const indices: number[] = []
      for (let i = start; i < end; i++) {
        const key = colViews.map((v) => v[i]!).join('\0')
        if (!seen.has(key)) {
          seen.add(key)
          indices.push(i)
        }
      }
      const out = new Uint32Array(indices)
      port.postMessage({ id: job.id, ok: true, indices: out })
      return
    }

    port.postMessage({ id: job.id, ok: false, error: 'unknown job' })
  } catch (err) {
    port.postMessage({ id: job.id, ok: false, error: String(err) })
  }
}
