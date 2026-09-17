import { setRowField } from '@columna/arrow'
import type { ReadJsonOptions } from './types.js'

function takeSlice<T>(arr: T[], skipRows: number, nRows?: number): T[] {
  const start = Math.max(0, skipRows)
  const end = nRows === undefined ? arr.length : start + Math.max(0, nRows)
  return arr.slice(start, end)
}

/** Parse JSON text or already-parsed value into row objects. */
export function parseJsonToRows(data: unknown, options: ReadJsonOptions = {}): Record<string, unknown>[] {
  const skipRows = options.skipRows ?? 0
  const linesMode = options.lines === true || options.orient === 'lines'
  const orient = linesMode ? 'lines' : (options.orient ?? 'records')

  if (typeof data === 'string') {
    if (orient === 'lines') {
      const parsed: Record<string, unknown>[] = []
      for (const line of data.replace(/^\uFEFF/, '').split(/\r?\n/)) {
        if (!line.trim()) continue
        parsed.push(JSON.parse(line) as Record<string, unknown>)
      }
      return takeSlice(parsed, skipRows, options.nRows)
    }
    data = JSON.parse(data)
  }

  if (orient === 'records') {
    if (!Array.isArray(data)) throw new Error('readJson orient=records expects an array of objects')
    return takeSlice(data as Record<string, unknown>[], skipRows, options.nRows)
  }

  if (orient === 'columns') {
    if (!data || typeof data !== 'object' || Array.isArray(data)) {
      throw new Error('readJson orient=columns expects { column: values[] }')
    }
    const cols = data as Record<string, unknown[]>
    const names = Object.keys(cols)
    const n = names.reduce((m, k) => Math.max(m, cols[k]?.length ?? 0), 0)
    const rows: Record<string, unknown>[] = []
    for (let i = skipRows; i < n; i++) {
      if (options.nRows !== undefined && rows.length >= options.nRows) break
      const row: Record<string, unknown> = {}
      for (const name of names) setRowField(row, name, cols[name]?.[i] ?? null)
      rows.push(row)
    }
    return rows
  }

  if (orient === 'values') {
    if (!Array.isArray(data)) throw new Error('readJson orient=values expects an array of arrays')
    const matrix = data as unknown[][]
    const sliced = takeSlice(matrix, skipRows, options.nRows)
    if (sliced.length === 0) return []
    const width = sliced.reduce((m, r) => Math.max(m, r.length), 0)
    const names = Array.from({ length: width }, (_, i) => `column_${i}`)
    return sliced.map((cells) => {
      const row: Record<string, unknown> = {}
      for (let i = 0; i < width; i++) setRowField(row, names[i]!, cells[i] ?? null)
      return row
    })
  }

  throw new Error(`Unsupported JSON orient: ${String(orient)}`)
}
