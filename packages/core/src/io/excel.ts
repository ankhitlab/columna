import { setRowField } from '@columna/arrow'
import type { ReadExcelOptions } from './types.js'

function asNullSet(v: string | string[] | undefined): Set<string> {
  const base = ['', 'null', 'NULL', 'na', 'NA', 'NaN']
  const extra = v ? (Array.isArray(v) ? v : [v]) : []
  return new Set([...base, ...extra])
}

/** Parse Excel workbook bytes (.xls / .xlsx) into row objects. */
export async function parseExcelToRows(
  bytes: Uint8Array,
  options: ReadExcelOptions = {},
): Promise<Record<string, unknown>[]> {
  const XLSX = await import('xlsx')
  const wb = XLSX.read(bytes, { type: 'array', cellDates: true })
  const sheetRef = options.sheetName ?? options.sheet ?? 0
  const sheetName =
    typeof sheetRef === 'number'
      ? wb.SheetNames[sheetRef]
      : wb.SheetNames.includes(sheetRef)
        ? sheetRef
        : undefined
  if (!sheetName) throw new Error(`Excel sheet not found: ${String(sheetRef)}`)
  const sheet = wb.Sheets[sheetName]
  if (!sheet) throw new Error(`Excel sheet missing: ${sheetName}`)

  const matrix = XLSX.utils.sheet_to_json(sheet, {
    header: 1,
    defval: null,
    raw: true,
    blankrows: false,
  }) as unknown[][]

  const skipRows = options.skipRows ?? 0
  const rows = matrix.slice(skipRows)
  if (rows.length === 0) return []

  const nulls = asNullSet(options.nullValues)
  const headerOpt = options.header === undefined ? 0 : options.header

  let headers: string[]
  let body: unknown[][]
  if (headerOpt === false) {
    const width = rows[0]?.length ?? 0
    headers = options.names ?? Array.from({ length: width }, (_, i) => `column_${i}`)
    body = rows
  } else {
    const headerRow = typeof headerOpt === 'number' ? headerOpt : 0
    const headerCells = rows[headerRow] ?? []
    headers =
      options.names ??
      headerCells.map((c, i) => (c === null || c === undefined || c === '' ? `column_${i}` : String(c)))
    body = rows.slice(headerRow + 1)
  }

  if (options.nRows !== undefined) body = body.slice(0, options.nRows)

  let colIdx = headers.map((_, i) => i)
  if (options.usecols && options.usecols.length > 0) {
    colIdx = options.usecols.map((c) => {
      if (typeof c === 'number') return c
      const i = headers.indexOf(c)
      if (i < 0) throw new Error(`usecols: unknown column "${c}"`)
      return i
    })
  }

  return body.map((cells) => {
    const row: Record<string, unknown> = {}
    for (const i of colIdx) {
      const name = headers[i] ?? `column_${i}`
      let v = cells[i] ?? null
      if (v instanceof Date) v = v.getTime()
      if (typeof v === 'string' && nulls.has(v)) v = null
      setRowField(row, name, v)
    }
    return row
  })
}
