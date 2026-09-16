import type { DType } from '@columna/arrow'
import { setRowField } from '@columna/arrow'
import type { ReadCsvOptions } from './types.js'

function asList(v: string | string[] | undefined): string[] {
  if (!v) return []
  return Array.isArray(v) ? v : [v]
}

/** Split CSV text into records: a newline inside a quoted field does not end the record (RFC 4180). */
export function splitCsvRecords(csv: string, quoteChar = '"'): string[] {
  const out: string[] = []
  let start = 0
  let inQuotes = false
  for (let i = 0; i < csv.length; i++) {
    const ch = csv[i]!
    if (ch === quoteChar) inQuotes = !inQuotes
    else if (!inQuotes && (ch === '\n' || ch === '\r')) {
      out.push(csv.slice(start, i))
      if (ch === '\r' && csv[i + 1] === '\n') i++
      start = i + 1
    }
  }
  out.push(csv.slice(start))
  return out
}

const escapeRegExp = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** pandas-style mangling of repeated header names: a, a.1, a.2 … */
function dedupeHeaders(headers: string[]): string[] {
  const seen = new Map<string, number>()
  return headers.map((h) => {
    const k = seen.get(h)
    if (k === undefined) {
      seen.set(h, 0)
      return h
    }
    let n = k + 1
    while (seen.has(`${h}.${n}`)) n++
    seen.set(h, n)
    seen.set(`${h}.${n}`, 0)
    return `${h}.${n}`
  })
}

export function parseCsvLine(line: string, delimiter: string, quoteChar = '"'): string[] {
  const out: string[] = []
  let cur = ''
  let inQuotes = false
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]!
    if (ch === quoteChar) {
      if (inQuotes && line[i + 1] === quoteChar) {
        cur += quoteChar
        i++
      } else inQuotes = !inQuotes
      continue
    }
    if (ch === delimiter && !inQuotes) {
      out.push(cur)
      cur = ''
      continue
    }
    cur += ch
  }
  out.push(cur)
  return out
}

function coerceValue(
  raw: string,
  opts: {
    nullValues: Set<string>
    trueValues: Set<string>
    falseValues: Set<string>
    decimal: string
    thousands?: string
  },
): unknown {
  let s = raw.trim()
  if (opts.nullValues.has(s) || opts.nullValues.has(s.toLowerCase())) return null
  if (opts.trueValues.has(s) || opts.trueValues.has(s.toLowerCase())) return true
  if (opts.falseValues.has(s) || opts.falseValues.has(s.toLowerCase())) return false
  if (opts.thousands) s = s.split(opts.thousands).join('')
  if (opts.decimal !== '.') s = s.replace(opts.decimal, '.')
  if (s === '') return null
  if (/^[+-]?\d+$/.test(s)) return Number(s)
  if (/^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(s)) return Number(s)
  return raw.trim() === '' ? null : raw
}

function applyDtypes(rows: Record<string, unknown>[], dtypes: Record<string, DType>): void {
  for (const row of rows) {
    for (const [name, dtype] of Object.entries(dtypes)) {
      if (!(name in row)) continue
      const v = row[name]
      if (v === null || v === undefined) continue
      if (dtype === 'bool') row[name] = Boolean(v)
      else if (dtype === 'utf8' || dtype === 'category') row[name] = String(v)
      else if (dtype === 'datetime') row[name] = typeof v === 'number' ? v : Date.parse(String(v))
      else row[name] = Number(v)
    }
  }
}

/** Sync CSV string → row objects (shared by fromCSV / readCsv). */
export function parseCsvToRows(csv: string, options: ReadCsvOptions = {}): Record<string, unknown>[] {
  const delimiter = options.separator ?? options.delimiter ?? ','
  const quoteChar = options.quoteChar ?? '"'
  const skipRows = options.skipRows ?? 0
  const skipBlankLines = options.skipBlankLines !== false
  const comment = options.comment
  const decimal = options.decimal ?? '.'
  const nullValues = new Set(
    [...asList(options.nullValues), '', 'null', 'NULL', 'na', 'NA', 'NaN', 'nan'].map((s) => s),
  )
  const trueValues = new Set(asList(options.trueValues).map((s) => s))
  const falseValues = new Set(asList(options.falseValues).map((s) => s))
  if (trueValues.size === 0) {
    trueValues.add('true')
    trueValues.add('True')
    trueValues.add('TRUE')
  }
  if (falseValues.size === 0) {
    falseValues.add('false')
    falseValues.add('False')
    falseValues.add('FALSE')
  }

  let lines = splitCsvRecords(csv.replace(/^\uFEFF/, ''), quoteChar)
  if (skipRows > 0) lines = lines.slice(skipRows)

  const filtered: string[] = []
  for (const line of lines) {
    if (skipBlankLines && line.trim() === '') continue
    if (comment !== undefined && line.trimStart().startsWith(comment)) continue
    filtered.push(options.skipInitialSpace ? line.replace(new RegExp(`${escapeRegExp(delimiter)}\\s+`, 'g'), delimiter) : line)
  }
  if (filtered.length === 0) return []

  const hasHeader =
    options.hasHeader !== undefined
      ? options.hasHeader
      : options.header === undefined
        ? true
        : options.header !== false

  let headerRowIndex = 0
  if (typeof options.header === 'number') headerRowIndex = options.header

  let headers: string[]
  let dataStart: number
  if (!hasHeader) {
    const width = parseCsvLine(filtered[0]!, delimiter, quoteChar).length
    headers = options.names ?? Array.from({ length: width }, (_, i) => `column_${i}`)
    dataStart = 0
  } else {
    headers = options.names ?? dedupeHeaders(parseCsvLine(filtered[headerRowIndex]!, delimiter, quoteChar))
    dataStart = headerRowIndex + 1
  }

  let usecolsIdx: number[] | null = null
  if (options.usecols && options.usecols.length > 0) {
    usecolsIdx = options.usecols.map((c) => {
      if (typeof c === 'number') return c
      const i = headers.indexOf(c)
      if (i < 0) throw new Error(`usecols: unknown column "${c}"`)
      return i
    })
  }

  const coerceOpts = { nullValues, trueValues, falseValues, decimal, thousands: options.thousands }
  const rows: Record<string, unknown>[] = []
  const limit = options.nRows
  for (let li = dataStart; li < filtered.length; li++) {
    if (limit !== undefined && rows.length >= limit) break
    let cells = parseCsvLine(filtered[li]!, delimiter, quoteChar)
    if (options.skipInitialSpace) cells = cells.map((c) => c.trimStart())
    const row: Record<string, unknown> = {}
    const idxs = usecolsIdx ?? headers.map((_, i) => i)
    for (const i of idxs) {
      const name = headers[i] ?? `column_${i}`
      setRowField(row, name, coerceValue(cells[i] ?? '', coerceOpts))
    }
    rows.push(row)
  }

  if (options.dtypes) applyDtypes(rows, options.dtypes)
  return rows
}
