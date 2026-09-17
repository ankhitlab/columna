/**
 * Browser Web Worker entry for the parallel engine (Protocol v2).
 * Thin wrapper around the shared `engine-jobs` handler — the Node
 * equivalent lives in `engine-worker.ts`. Uses the global `self` port.
 */
import { handleJob, type EngineJob } from './engine-jobs.js'

const ctx = self as unknown as Worker & { onmessage: ((ev: MessageEvent) => void) | null }

const port = {
  postMessage: (msg: unknown, transfer?: Transferable[]) => {
    if (transfer && transfer.length > 0) ctx.postMessage(msg, transfer)
    else ctx.postMessage(msg)
  },
}

ctx.onmessage = (ev: MessageEvent) => {
  handleJob(ev.data as EngineJob, port)
}
