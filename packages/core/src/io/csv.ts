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

/**
 * Split one CSV record into fields. Unquoted rows (the common bench path) use `slice` only — no
 * per-character string growth. Quoted fields still support RFC 4180 escapes.
 */
export function parseCsvLine(line: string, delimiter: string, quoteChar = '"'): string[] {
  const out: string[] = []
  parseCsvLineInto(line, delimiter, quoteChar, out)
  return out
}

/** Fill `out` with fields of `line` (clears `out` first). Reuse the same array across rows. */
export function parseCsvLineInto(line: string, delimiter: string, quoteChar: string, out: string[], quoted?: boolean[]): void {
  out.length = 0
  if (quoted) quoted.length = 0
  // Fast path: no quotes → delimiter splits via slices (2M×8 bench CSV never quotes).
  if (line.indexOf(quoteChar) < 0) {
    if (delimiter.length === 1) {
      const d = delimiter.charCodeAt(0)
      let start = 0
      for (let i = 0; i < line.length; i++) {
        if (line.charCodeAt(i) === d) {
          out.push(line.slice(start, i))
          start = i + 1
        }
      }
      out.push(line.slice(start))
      return
    }
    let start = 0
    let i = line.indexOf(delimiter)
    while (i >= 0) {
      out.push(line.slice(start, i))
      start = i + delimiter.length
      i = line.indexOf(delimiter, start)
    }
    out.push(line.slice(start))
    return
  }
  let start = 0
  let inQuotes = false
  let escaped = false
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]!
    if (ch === quoteChar) {
      if (inQuotes && line[i + 1] === quoteChar) {
        escaped = true
        i++
      } else inQuotes = !inQuotes
      continue
    }
    if (ch === delimiter && !inQuotes) {
      out.push(extractField(line, start, i, quoteChar, escaped))
      if (quoted) quoted.push(line[start] === quoteChar)
      start = i + 1
      escaped = false
    }
  }
  out.push(extractField(line, start, line.length, quoteChar, escaped))
  if (quoted) quoted.push(line[start] === quoteChar)
}

/** Strip surrounding quote toggles; unescape doubled quotes when needed. */
function extractField(line: string, start: number, end: number, quoteChar: string, escaped: boolean): string {
  if (!escaped) {
    // Quotes were toggled out of the slice range in the scanner — rebuild without them.
    let out = ''
    let inQuotes = false
    for (let i = start; i < end; i++) {
      const ch = line[i]!
      if (ch === quoteChar) {
        inQuotes = !inQuotes
        continue
      }
      out += ch
    }
    return out
  }
  let out = ''
  let inQuotes = false
  for (let i = start; i < end; i++) {
    const ch = line[i]!
    if (ch === quoteChar) {
      if (inQuotes && line[i + 1] === quoteChar) {
        out += quoteChar
        i++
      } else inQuotes = !inQuotes
      continue
    }
    out += ch
  }
  return out
}

export type CoerceOptions = {
  nullValues: Set<string>
  trueValues: Set<string>
  falseValues: Set<string>
  decimal: string
  thousands?: string
}

export function coerceValue(raw: string, opts: CoerceOptions): unknown {
  let s = trimAscii(raw)
  if (opts.nullValues.has(s) || (s !== s.toLowerCase() && opts.nullValues.has(s.toLowerCase()))) return null
  if (opts.trueValues.has(s) || (s !== s.toLowerCase() && opts.trueValues.has(s.toLowerCase()))) return true
  if (opts.falseValues.has(s) || (s !== s.toLowerCase() && opts.falseValues.has(s.toLowerCase()))) return false
  if (opts.thousands) s = s.split(opts.thousands).join('')
  if (opts.decimal !== '.') s = s.replace(opts.decimal, '.')
  if (s === '') return null
  // Default locale (`.` decimal, no thousands): scan char codes — avoids two RegExp tests per cell.
  if (!opts.thousands && opts.decimal === '.') {
    const num = tryParsePlainNumber(s)
    if (num !== undefined) return num
    return s === '' ? null : raw
  }
  if (/^[+-]?\d+$/.test(s)) {
    const n = Number(s)
    // An integer beyond 2^53 cannot be held exactly in a JS number (there is no int64 dtype): keep the digits
    // as text rather than silently rounding an identifier. Pass dtypes: { col: 'f64' } to force a number.
    return Number.isSafeInteger(n) ? n : s
  }
  if (/^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(s)) return Number(s)
  return trimAscii(raw) === '' ? null : raw
}

function trimAscii(raw: string): string {
  let start = 0
  let end = raw.length
  while (start < end && raw.charCodeAt(start) <= 32) start++
  while (end > start && raw.charCodeAt(end - 1) <= 32) end--
  return start === 0 && end === raw.length ? raw : raw.slice(start, end)
}

/**
 * Parse a plain number matching `^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$`. Returns the number, or the digit
 * string when an integer is outside the safe-integer range (same contract as the RegExp path).
 */
function tryParsePlainNumber(s: string): number | string | undefined {
  return tryParsePlainNumberRange(s, 0, s.length)
}

/**
 * Same as {@link tryParsePlainNumber} on `s[start..end)`. Safe integers and simple decimals are parsed
 * without allocating a substring; only unsafe integers / scientific notation slice.
 */
export function tryParsePlainNumberRange(s: string, start: number, end: number): number | string | undefined {
  if (end <= start) return undefined
  let i = start
  let sign = 1
  const c0 = s.charCodeAt(i)
  if (c0 === 45) {
    sign = -1
    i++
  } else if (c0 === 43) i++
  if (i >= end) return undefined

  let intDigits = 0
  let intVal = 0
  while (i < end) {
    const c = s.charCodeAt(i)
    if (c < 48 || c > 57) break
    intDigits++
    intVal = intVal * 10 + (c - 48)
    i++
  }
  if (intDigits === 0) return undefined

  if (i >= end) {
    // integer, no fraction/exponent — avoid allocating a substring for safe ints
    if (intDigits > 15) {
      const token = s.slice(start, end)
      const n = Number(token)
      return Number.isSafeInteger(n) ? n : token
    }
    const n = sign < 0 ? -intVal : intVal
    return Number.isSafeInteger(n) ? n : s.slice(start, end)
  }

  if (s.charCodeAt(i) === 46) {
    i++
    let fracDigits = 0
    let fracVal = 0
    while (i < end) {
      const c = s.charCodeAt(i)
      if (c < 48 || c > 57) break
      fracDigits++
      fracVal = fracVal * 10 + (c - 48)
      i++
    }
    if (fracDigits === 0) return undefined
    if (i >= end) {
      // Plain decimal. Clinger's exact fast path: with ≤ 15 significant digits the integer mantissa
      // intVal·10^k + fracVal is exact in a double and so is 10^k, so ONE division gives the correctly
      // rounded value. `intVal + fracVal / scale` (two roundings) was off by an ulp for values such as
      // 0.09626300842501223 — caught by the CSV round-trip invariant test. Longer literals go to Number().
      if (intDigits + fracDigits <= 15) {
        let scale = 1
        for (let k = 0; k < fracDigits; k++) scale *= 10
        return sign * ((intVal * scale + fracVal) / scale)
      }
      return Number(s.slice(start, end))
    }
    // exponent follows
  }

  const e = s.charCodeAt(i)
  if (e !== 101 && e !== 69) return undefined
  i++
  if (i < end) {
    const sgn = s.charCodeAt(i)
    if (sgn === 43 || sgn === 45) i++
  }
  let expDigits = 0
  while (i < end) {
    const c = s.charCodeAt(i)
    if (c < 48 || c > 57) break
    expDigits++
    i++
  }
  if (expDigits === 0 || i !== end) return undefined
  const token = s.slice(start, end)
  return Number(token)
}

function applyDtypes(rows: Record<string, unknown>[], dtypes: Record<string, DType>): void {
  for (const row of rows) {
    for (const [name, dtype] of Object.entries(dtypes)) {
      if (!(name in row)) continue
      const v = row[name]
      if (v === null || v === undefined) continue
      if (dtype === 'bool') setRowField(row, name, Boolean(v))
      else if (dtype === 'utf8' || dtype === 'category') setRowField(row, name, String(v))
      else if (dtype === 'datetime') setRowField(row, name, typeof v === 'number' ? v : Date.parse(String(v)))
      else setRowField(row, name, Number(v))
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
