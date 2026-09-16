/**
 * Long-lived worker: dual-gt writes a shared Uint8 mask; typed gather into SAB.
 */
import { parentPort } from 'node:worker_threads'

type ArrKind = 'Float64Array' | 'Float32Array' | 'Int32Array' | 'Uint32Array' | 'Uint8Array'

function view(kind: ArrKind, buffer: SharedArrayBuffer, length: number): ArrayLike<number> & { [i: number]: number } {
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

parentPort!.on('message', (job: {
  id: number
  type: 'dualGt' | 'gather' | 'ping'
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
  srcBuffer?: SharedArrayBuffer
  srcKind?: ArrKind
  srcLength?: number
  idxBuffer?: SharedArrayBuffer
  idxLength?: number
  outBuffer?: SharedArrayBuffer
  outKind?: ArrKind
}) => {
  try {
    if (job.type === 'ping') {
      parentPort!.postMessage({ id: job.id, ok: true })
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
      parentPort!.postMessage({ id: job.id, ok: true, count })
      return
    }

    if (job.type === 'gather') {
      const src = view(job.srcKind!, job.srcBuffer!, job.srcLength!)
      const idx = new Uint32Array(job.idxBuffer!, 0, job.idxLength!)
      const out = view(job.outKind!, job.outBuffer!, job.idxLength!)
      const n = job.idxLength!
      for (let i = 0; i < n; i++) out[i] = src[idx[i]!]!
      parentPort!.postMessage({ id: job.id, ok: true })
      return
    }

    parentPort!.postMessage({ id: job.id, ok: false, error: 'unknown job' })
  } catch (err) {
    parentPort!.postMessage({ id: job.id, ok: false, error: String(err) })
  }
})
