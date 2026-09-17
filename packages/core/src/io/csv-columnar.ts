/**
 * Columnar CSV ingestion: records go straight into growable typed column builders — no row objects, no
 * second pass — and a Node path is read as a stream of chunks, so peak memory is the columns plus one
 * chunk, not the whole file plus an object per cell. Type inference is exactly `fromRows`'s: a late
 * fractional / out-of-Int32 / text value widens the column instead of being coerced.
 */
import { setValue, tableFromColumns, type Column, type DType, type TableView } from '@columna/arrow'
import { coerceValue, parseCsvLine, type CoerceOptions } from './csv.js'
import type { ReadCsvOptions } from './types.js'

// ---- growable column builder -----------------------------------------------------------------------

class ColumnBuilder {
  /** Numeric values, or dictionary codes once the column has turned into text. */
  private nums = new Float64Array(1024)
  private valid = new Uint8Array(128)
  private n = 0
  private sawInt = false
  private sawFloat = false
  private sawBool = false
  private intOutsideI32 = false
  private anyNull = false
  /** Text mode: values are dictionary-encoded on the fly so no per-cell string is retained. */
  private dict: string[] | null = null
  private codes: Map<string, number> | null = null

  constructor(
    readonly name: string,
    private readonly forced: DType | undefined,
  ) {
    if (forced === 'utf8' || forced === 'category') this.enterTextMode(0)
  }

  push(v: unknown): void {
    const i = this.n++
    if (i >= this.nums.length) this.grow()
    if (v === null || v === undefined) {
      this.anyNull = true
      return
    }
    this.valid[i >> 3]! |= 1 << (i & 7)
    if (this.codes) {
      this.nums[i] = this.code(typeof v === 'string' ? v : String(v))
      return
    }
    if (typeof v === 'number') {
      this.nums[i] = v
      if (Number.isInteger(v)) {
        this.sawInt = true
        if (v > 2147483647 || v < -2147483648) this.intOutsideI32 = true
      } else this.sawFloat = true
      return
    }
    if (typeof v === 'boolean') {
      this.nums[i] = v ? 1 : 0
      this.sawBool = true
      return
    }
    // first text value: widen every earlier value to its text form (same as fromRows → String(v))
    this.enterTextMode(i) // only the rows before this one hold numeric values
    this.nums[i] = this.code(typeof v === 'string' ? v : String(v))
  }

  private enterTextMode(count: number): void {
    if (this.codes) return
    this.dict = []
    this.codes = new Map()
    for (let k = 0; k < count; k++) {
      if ((this.valid[k >> 3]! >> (k & 7)) & 1) this.nums[k] = this.code(this.numToString(this.nums[k]!))
    }
  }

  private code(s: string): number {
    let c = this.codes!.get(s)
    if (c === undefined) {
      c = this.dict!.length
      this.dict!.push(s)
      this.codes!.set(s, c)
    }
    return c
  }

  private numToString(x: number): string {
    // booleans were stored as 1 / 0 before a text value appeared; keep fromRows semantics (String(true) → "true")
    return this.sawBool && !this.sawInt && !this.sawFloat ? (x ? 'true' : 'false') : String(x)
  }

  private grow(): void {
    const next = new Float64Array(this.nums.length * 2)
    next.set(this.nums)
    this.nums = next
    if (this.valid.length * 8 < next.length) {
      const nv = new Uint8Array(this.valid.length * 2)
      nv.set(this.valid)
      this.valid = nv
    }
  }

  finish(): Column {
    const n = this.n
    const nullBitmap = this.anyNull ? this.valid.slice(0, Math.ceil(n / 8) || 1) : undefined
    const forced = this.forced
    if (this.codes) {
      const dict = this.dict!
      const nullable = this.anyNull
      if (forced !== 'utf8' && dict.length <= Math.max(1024, n / 4)) {
        const codes = new Uint32Array(n)
        for (let k = 0; k < n; k++) codes[k] = this.nums[k]!
        return { field: { name: this.name, dtype: 'category', nullable }, data: codes, nullBitmap, dictionary: dict }
      }
      const data = new Array<string>(n)
      for (let k = 0; k < n; k++) data[k] = (this.valid[k >> 3]! >> (k & 7)) & 1 ? dict[this.nums[k]!]! : ''
      return { field: { name: this.name, dtype: 'utf8', nullable }, data, nullBitmap }
    }
    let dtype: DType
    if (forced) dtype = forced
    else if (this.sawFloat || this.intOutsideI32) dtype = 'f64'
    else if (this.sawInt) dtype = 'i32'
    else if (this.sawBool) dtype = 'bool'
    else dtype = 'f64'
    let data: Column['data']
    if (dtype === 'f64' || dtype === 'datetime') data = this.nums.slice(0, n)
    else if (dtype === 'bool') {
      const b = new Uint8Array(n)
      for (let k = 0; k < n; k++) b[k] = this.nums[k] ? 1 : 0
      data = b
    } else {
      const typed = dtype === 'i32' ? new Int32Array(n) : dtype === 'u32' ? new Uint32Array(n) : new Float32Array(n)
      for (let k = 0; k < n; k++) if ((this.valid[k >> 3]! >> (k & 7)) & 1) setValue(typed, k, this.nums[k]!, dtype)
      data = typed
    }
    return { field: { name: this.name, dtype, nullable: this.anyNull }, data, nullBitmap }
  }
}

// ---- record processing (shared by the string and the stream path) ---------------------------------

/** Consumes CSV records one at a time and builds columns; mirrors every option of `parseCsvToRows`. */
export class CsvTableBuilder {
  private readonly delimiter: string
  private readonly quoteChar: string
  private readonly skipBlankLines: boolean
  private readonly comment: string | undefined
  private readonly skipInitialSpace: boolean
  private readonly hasHeader: boolean
  private readonly headerRowIndex: number
  private readonly coerce: CoerceOptions
  private readonly limit: number | undefined
  private readonly dtypes: Record<string, DType> | undefined
  private readonly delimRe: RegExp | null
  private toSkip: number
  private seenFiltered = 0
  private headers: string[] | null = null
  private builders: ColumnBuilder[] = []
  private colIdx: number[] = []
  private rows = 0
  /** True once `nRows` is reached: callers may stop feeding records. */
  done = false

  constructor(private readonly options: ReadCsvOptions = {}) {
    this.delimiter = options.separator ?? options.delimiter ?? ','
    this.quoteChar = options.quoteChar ?? '"'
    this.toSkip = options.skipRows ?? 0
    this.skipBlankLines = options.skipBlankLines !== false
    this.comment = options.comment
    this.skipInitialSpace = Boolean(options.skipInitialSpace)
    this.hasHeader = options.hasHeader !== undefined ? options.hasHeader : options.header === undefined ? true : options.header !== false
    this.headerRowIndex = typeof options.header === 'number' ? options.header : 0
    this.limit = options.nRows
    this.dtypes = options.dtypes
    this.coerce = makeCoerceOptions(options)
    this.delimRe = this.skipInitialSpace ? new RegExp(`${escapeRegExp(this.delimiter)}\\s+`, 'g') : null
    if (options.names) this.initColumns(options.names)
  }

  /** Feed one record (a physical line with quoted newlines already joined). */
  push(line: string): void {
    if (this.done) return
    if (this.toSkip > 0) {
      this.toSkip--
      return
    }
    if (this.skipBlankLines && line.trim() === '') return
    if (this.comment !== undefined && line.trimStart().startsWith(this.comment)) return
    const rec = this.delimRe ? line.replace(this.delimRe, this.delimiter) : line
    const idx = this.seenFiltered++
    if (!this.headers) {
      if (!this.hasHeader) {
        const width = parseCsvLine(rec, this.delimiter, this.quoteChar).length
        this.initColumns(Array.from({ length: width }, (_, i) => `column_${i}`))
        // fall through: this record is data
      } else {
        if (idx < this.headerRowIndex) return
        this.initColumns(dedupeHeaders(parseCsvLine(rec, this.delimiter, this.quoteChar)))
        return
      }
    } else if (this.hasHeader && idx <= this.headerRowIndex) {
      // names were given up front: the header row of the file is still skipped (parseCsvToRows semantics)
      return
    }
    let cells = parseCsvLine(rec, this.delimiter, this.quoteChar)
    if (this.skipInitialSpace) cells = cells.map((c) => c.trimStart())
    for (let k = 0; k < this.colIdx.length; k++) {
      const b = this.builders[k]!
      const raw = cells[this.colIdx[k]!] ?? ''
      let v = coerceValue(raw, this.coerce)
      const forced = this.dtypes?.[b.name]
      if (forced !== undefined && v !== null) v = coerceForced(v, forced)
      b.push(v)
    }
    this.rows++
    if (this.limit !== undefined && this.rows >= this.limit) this.done = true
  }

  private initColumns(headers: string[]): void {
    this.headers = headers
    let idxs: number[]
    if (this.options.usecols && this.options.usecols.length > 0) {
      idxs = this.options.usecols.map((c) => {
        if (typeof c === 'number') return c
        const i = headers.indexOf(c)
        if (i < 0) throw new Error(`usecols: unknown column "${c}"`)
        return i
      })
    } else idxs = headers.map((_, i) => i)
    this.colIdx = idxs
    this.builders = idxs.map((i) => new ColumnBuilder(headers[i] ?? `column_${i}`, this.dtypes?.[headers[i] ?? `column_${i}`]))
  }

  finish(): TableView {
    if (!this.headers) return tableFromColumns([])
    return tableFromColumns(this.builders.map((b) => b.finish()))
  }
}

function coerceForced(v: unknown, dtype: DType): unknown {
  if (dtype === 'bool') return Boolean(v)
  if (dtype === 'utf8' || dtype === 'category') return String(v)
  if (dtype === 'datetime') return typeof v === 'number' ? v : Date.parse(String(v))
  return Number(v)
}

function makeCoerceOptions(options: ReadCsvOptions): CoerceOptions {
  const list = (v: string | string[] | undefined) => (v ? (Array.isArray(v) ? v : [v]) : [])
  const nullValues = new Set([...list(options.nullValues), '', 'null', 'NULL', 'na', 'NA', 'NaN', 'nan'])
  const trueValues = new Set(list(options.trueValues))
  const falseValues = new Set(list(options.falseValues))
  if (trueValues.size === 0) for (const t of ['true', 'True', 'TRUE']) trueValues.add(t)
  if (falseValues.size === 0) for (const f of ['false', 'False', 'FALSE']) falseValues.add(f)
  return { nullValues, trueValues, falseValues, decimal: options.decimal ?? '.', thousands: options.thousands }
}

const escapeRegExp = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** pandas-style mangling of repeated header names: a, a.1, a.2 … */
export function dedupeHeaders(headers: string[]): string[] {
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

// ---- record splitting with carry (quoted newlines never end a record) -----------------------------

export class CsvRecordSplitter {
  private carry = ''
  /** Characters of `carry` already scanned (a trailing CR is left unscanned until the next chunk arrives). */
  private scanned = 0
  private inQuotes = false
  constructor(private readonly quoteChar = '"') {}

  /** Split a chunk into complete records; an incomplete trailing record is kept for the next chunk. */
  push(chunk: string, sink: (record: string) => void): void {
    const text = this.carry + chunk
    let start = 0
    let inQuotes = this.inQuotes
    let i = this.scanned
    for (; i < text.length; i++) {
      const ch = text[i]!
      if (ch === this.quoteChar) inQuotes = !inQuotes
      else if (!inQuotes && (ch === '\n' || ch === '\r')) {
        if (ch === '\r' && i + 1 >= text.length) break // may be the first half of CRLF — wait for the next chunk
        sink(text.slice(start, i))
        if (ch === '\r' && text[i + 1] === '\n') i++
        start = i + 1
      }
    }
    this.carry = text.slice(start)
    this.scanned = i - start // everything before `i` is scanned; `inQuotes` is the state there
    this.inQuotes = inQuotes
  }

  /** Flush the last record (called at end of input). */
  end(sink: (record: string) => void): void {
    if (this.carry.length > 0) sink(this.carry.replace(/\r$/, ''))
    this.carry = ''
    this.scanned = 0
    this.inQuotes = false
  }
}

// ---- entry points ---------------------------------------------------------------------------------

/** Whole CSV text → table (records split quote-aware, no row objects). */
export function parseCsvToTable(csv: string, options: ReadCsvOptions = {}): TableView {
  const builder = new CsvTableBuilder(options)
  const splitter = new CsvRecordSplitter(options.quoteChar ?? '"')
  splitter.push(csv.replace(/^﻿/, ''), (r) => builder.push(r))
  splitter.end((r) => builder.push(r))
  return builder.finish()
}

/**
 * Stream a Node file into a table chunk by chunk. `maxBytes` is enforced on bytes read; `nRows` stops the
 * read early. Falls back to the caller for non-Node environments.
 */
export async function streamCsvFileToTable(
  path: string,
  options: ReadCsvOptions & { maxBytes?: number; signal?: AbortSignal; encoding?: string },
): Promise<TableView> {
  const fs = await import('node:fs')
  const builder = new CsvTableBuilder(options)
  const splitter = new CsvRecordSplitter(options.quoteChar ?? '"')
  const decoder = new TextDecoder(options.encoding ?? 'utf-8')
  const stream = fs.createReadStream(path, { highWaterMark: 1 << 20, signal: options.signal })
  let bytes = 0
  let first = true
  try {
    for await (const chunk of stream as AsyncIterable<Uint8Array>) {
      bytes += chunk.byteLength
      if (options.maxBytes !== undefined && bytes > options.maxBytes) {
        throw new Error(`IO policy: file "${path}" exceeds maxBytes = ${options.maxBytes}`)
      }
      let text = decoder.decode(chunk, { stream: true })
      if (first) {
        text = text.replace(/^﻿/, '')
        first = false
      }
      splitter.push(text, (r) => builder.push(r))
      if (builder.done) break
    }
    if (!builder.done) {
      const tail = decoder.decode()
      if (tail) splitter.push(tail, (r) => builder.push(r))
      splitter.end((r) => builder.push(r))
    }
  } finally {
    stream.destroy()
  }
  return builder.finish()
}
