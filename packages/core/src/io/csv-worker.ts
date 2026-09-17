/**
 * Worker thread: parse a CSV chunk with the columnar reader and return transferable columns.
 */
import { parentPort } from 'node:worker_threads'
import { parseCsvToTable } from './csv-columnar.js'
import type { ReadCsvOptions } from './types.js'

parentPort!.on('message', (job: { text: string; options: ReadCsvOptions }) => {
  try {
    const table = parseCsvToTable(job.text, job.options)
    const columns = table.columns.map((c) => ({
      name: c.field.name,
      dtype: c.field.dtype,
      nullable: Boolean(c.nullBitmap),
      data: c.data,
      nullBitmap: c.nullBitmap,
      dictionary: c.dictionary,
    }))
    const transfer: ArrayBuffer[] = []
    for (const col of columns) {
      const d = col.data
      if (ArrayBuffer.isView(d) && d.buffer instanceof ArrayBuffer) transfer.push(d.buffer)
      if (col.nullBitmap && col.nullBitmap.buffer instanceof ArrayBuffer) transfer.push(col.nullBitmap.buffer)
    }
    parentPort!.postMessage({ ok: true, columns }, transfer)
  } catch (err) {
    parentPort!.postMessage({ ok: false, error: err instanceof Error ? err.message : String(err) })
  }
})
