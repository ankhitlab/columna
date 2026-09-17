/**
 * Node worker_threads entry for the parallel engine (Protocol v2).
 * Thin wrapper around the shared `engine-jobs` handler — the browser
 * equivalent lives in `engine-worker-web.ts`.
 */
import { parentPort } from 'node:worker_threads'
import { handleJob, type EngineJob } from './engine-jobs.js'

const port = parentPort
if (port) {
  const sender = {
    postMessage: (msg: unknown, transfer?: Transferable[]) => {
      if (transfer && transfer.length > 0) port.postMessage(msg, transfer as never)
      else port.postMessage(msg)
    },
  }
  port.on('message', (job: EngineJob) => handleJob(job, sender))
}
