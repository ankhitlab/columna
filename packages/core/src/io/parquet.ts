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

  const ab = new ArrayBuffer(bytes.byteLength)
  new Uint8Array(ab).set(bytes)

  const rows = await parquetReadObjects({
    file: ab,
    columns: options.columns,
    rowStart,
    rowEnd,
    compressors,
  })

  return (rows as Record<string, unknown>[]).map((row) => {
    const out: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(row)) {
      if (typeof v === 'bigint') {
        const n = Number(v)
        out[k] = Number.isSafeInteger(n) ? n : v.toString()
      } else out[k] = v
    }
    return out
  })
}
