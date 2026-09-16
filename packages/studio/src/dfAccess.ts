import type { DataFrame } from 'columna'
import { isDataFrame } from './types'

function bitmapValid(bitmap: Uint8Array | undefined, i: number): boolean {
  if (!bitmap) return true
  return (bitmap[i >> 3]! & (1 << (i & 7))) !== 0
}

function cellValue(df: DataFrame, colIndex: number, row: number): unknown {
  const col = df.table.columns[colIndex]!
  if (!bitmapValid(col.nullBitmap, row)) return null
  const raw = col.data[row as number]
  if (col.field.dtype === 'bool') return Boolean(raw)
  if (col.field.dtype === 'category' && col.dictionary) {
    return col.dictionary[Number(raw)] ?? null
  }
  return raw
}

/** Materialize only the first `limit` rows (no full-frame toArray). */
export function headRows(df: DataFrame, limit: number): Record<string, unknown>[] {
  const n = Math.min(Math.max(0, limit), df.table.numRows)
  const names = df.columns
  const rows: Record<string, unknown>[] = new Array(n)
  for (let i = 0; i < n; i++) {
    const row: Record<string, unknown> = {}
    for (let c = 0; c < names.length; c++) row[names[c]!] = cellValue(df, c, i)
    rows[i] = row
  }
  return rows
}

/** CSV for up to `limit` rows without building a full row-object array. */
export function toCsvText(df: DataFrame, limit?: number): string {
  const names = df.columns
  const n = Math.min(limit ?? df.table.numRows, df.table.numRows)
  const lines: string[] = [names.join(',')]
  for (let i = 0; i < n; i++) {
    lines.push(
      names
        .map((_, c) => {
          const v = cellValue(df, c, i)
          if (v == null) return ''
          const s = String(v)
          return s.includes(',') || s.includes('"') || s.includes('\n')
            ? `"${s.replaceAll('"', '""')}"`
            : s
        })
        .join(','),
    )
  }
  return lines.join('\n')
}

const MAX_PLOT_POINTS = 50_000

export function plotSeries(
  data: unknown,
  xName: string,
  yName: string,
): { x: (number | string)[]; y: number[] } {
  if (Array.isArray(data)) {
    const rows = data as Record<string, unknown>[]
    const step = Math.max(1, Math.ceil(rows.length / MAX_PLOT_POINTS))
    const x: (number | string)[] = []
    const y: number[] = []
    for (let i = 0; i < rows.length; i += step) {
      const r = rows[i]!
      x.push(r[xName] as number | string)
      y.push(Number(r[yName] ?? NaN))
    }
    return { x, y }
  }
  if (!isDataFrame(data)) throw new Error('plot/hist expects a DataFrame or array of row objects')

  const xi = data.columns.indexOf(xName)
  const yi = data.columns.indexOf(yName)
  if (xi < 0) throw new Error(`Unknown column "${xName}"`)
  if (yi < 0) throw new Error(`Unknown column "${yName}"`)

  const n = data.table.numRows
  const step = Math.max(1, Math.ceil(n / MAX_PLOT_POINTS))
  const x: (number | string)[] = []
  const y: number[] = []
  for (let i = 0; i < n; i += step) {
    x.push(cellValue(data, xi, i) as number | string)
    y.push(Number(cellValue(data, yi, i) ?? NaN))
  }
  return { x, y }
}

/** Finite numeric values for a column (optional downsample for large frames). */
export function numericColumnValues(data: unknown, column: string): Float64Array {
  if (Array.isArray(data)) {
    const rows = data as Record<string, unknown>[]
    const out = new Float64Array(rows.length)
    let j = 0
    for (const r of rows) {
      const v = Number(r[column] ?? NaN)
      if (Number.isFinite(v)) out[j++] = v
    }
    return out.subarray(0, j)
  }
  if (!isDataFrame(data)) throw new Error('plot/hist expects a DataFrame or array of row objects')

  const ci = data.columns.indexOf(column)
  if (ci < 0) throw new Error(`Unknown column "${column}"`)
  const n = data.table.numRows
  const step = Math.max(1, Math.ceil(n / MAX_PLOT_POINTS))
  const out = new Float64Array(Math.ceil(n / step))
  let j = 0
  for (let i = 0; i < n; i += step) {
    const v = Number(cellValue(data, ci, i) ?? NaN)
    if (Number.isFinite(v)) out[j++] = v
  }
  return out.subarray(0, j)
}
