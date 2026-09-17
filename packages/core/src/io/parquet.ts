import { setRowField } from '@columna/arrow'
import type { ReadParquetOptions } from './types.js'

/** Parse Parquet bytes into row objects. */
export async function parseParquetToRows(
  bytes: Uint8Array,
  options: ReadParquetOptions = {},
): Promise<Record<string, unknown>[]> {
  const [{ parquetReadObjects }, { compressors }] = await Promise.all([
    import('hyparquet'),
    import('hyparquet-compressors'),
  ])

  const rowStart = options.rowStart ?? 0
  let rowEnd = options.rowEnd
  if (rowEnd === undefined && options.nRows !== undefined) rowEnd = rowStart + options.nRows

  // hyparquet wants an ArrayBuffer: hand over the existing one when the view covers it exactly, else copy
  const ab =
    bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength && bytes.buffer instanceof ArrayBuffer
      ? bytes.buffer
      : bytes.slice().buffer

  const rows = await parquetReadObjects({
    file: ab,
    columns: options.columns,
    rowStart,
    rowEnd,
    compressors,
  })

  return (rows as Record<string, unknown>[]).map(normalizeParquetRow)
}

/** BigInt → number when safe (else decimal string); column names are data, never prototype keys. */
export function normalizeParquetRow(row: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(row)) {
    if (typeof v === 'bigint') {
      const n = Number(v)
      setRowField(out, k, Number.isSafeInteger(n) ? n : v.toString())
    } else setRowField(out, k, v)
  }
  return out
}
