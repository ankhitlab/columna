/**
 * Columnar CSV ingestion: records go straight into growable typed column builders — no row objects, no
 * second pass — and a Node path is read as a stream of chunks, so peak memory is the columns plus one
 * chunk, not the whole file plus an object per cell. Type inference is exactly `fromRows`'s: a late
 * fractional / out-of-Int32 / text value widens the column instead of being coerced.
 */
import { setValue, tableFromColumns, type Column, type DType, type TableView } from '@columna/arrow'
import { coerceValue, parseCsvLine, parseCsvLineInto, tryParsePlainNumberRange, type CoerceOptions } from './csv.js'
import type { ReadCsvOptions } from './types.js'

// ---- growable column builder -----------------------------------------------------------------------

class ColumnBuilder {
  /** Numeric values, or dictionary codes once the column has turned into text. */
  private nums = new Float64Array(4096)
  private valid = new Uint8Array(512)
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

  /** Pre-size for an estimated row count (avoids repeated geometric copies on large files). */
  reserve(rows: number): void {
    if (rows <= this.nums.length) return
    const next = new Float64Array(rows)
    next.set(this.nums.subarray(0, this.n))
    this.nums = next
    const needValid = Math.ceil(rows / 8) || 1
    if (needValid > this.valid.length) {
      const nv = new Uint8Array(needValid)
      nv.set(this.valid)
      this.valid = nv
    }
  }

  push(v: unknown): void {
    if (v === null || v === undefined) {
      this.pushNull()
      return
    }
    if (typeof v === 'number') {
      this.pushNumber(v)
      return
    }
    if (typeof v === 'boolean') {
      this.pushBool(v)
      return
    }
    this.pushText(typeof v === 'string' ? v : String(v))
  }

  pushNull(): void {
    const i = this.n++
    if (i >= this.nums.length) this.grow()
    this.anyNull = true
  }

  pushNumber(v: number): void {
    const i = this.n++
    if (i >= this.nums.length) this.grow()
    if (this.codes) {
      this.valid[i >> 3]! |= 1 << (i & 7)
      this.nums[i] = this.code(String(v))
      return
    }
    this.valid[i >> 3]! |= 1 << (i & 7)
    this.nums[i] = v
    if (Number.isInteger(v)) {
      this.sawInt = true
      if (v > 2147483647 || v < -2147483648) this.intOutsideI32 = true
    } else this.sawFloat = true
  }

  pushBool(v: boolean): void {
    const i = this.n++
    if (i >= this.nums.length) this.grow()
    if (this.codes) {
      this.valid[i >> 3]! |= 1 << (i & 7)
      this.nums[i] = this.code(v ? 'true' : 'false')
      return
    }
    this.valid[i >> 3]! |= 1 << (i & 7)
    this.nums[i] = v ? 1 : 0
    this.sawBool = true
  }

  pushText(s: string): void {
    const i = this.n++
    if (i >= this.nums.length) this.grow()
    this.valid[i >> 3]! |= 1 << (i & 7)
    if (!this.codes) this.enterTextMode(i)
    this.nums[i] = this.code(s)
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
    const next = new Float64Array(Math.max(this.nums.length * 2, this.n))
    next.set(this.nums)
    this.nums = next
    if (this.valid.length * 8 < next.length) {
      const nv = new Uint8Array(Math.ceil(next.length / 8))
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
    } else if (dtype === 'i32') {
      const typed = new Int32Array(n)
      const nums = this.nums
      if (this.anyNull) {
        for (let k = 0; k < n; k++) if ((this.valid[k >> 3]! >> (k & 7)) & 1) typed[k] = nums[k]!
      } else {
        for (let k = 0; k < n; k++) typed[k] = nums[k]!
      }
      data = typed
    } else {
      const typed = dtype === 'u32' ? new Uint32Array(n) : new Float32Array(n)
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
  private readonly delimCode: number
  private readonly quoteChar: string
  private readonly quoteCode: number
  private readonly skipBlankLines: boolean
  private readonly comment: string | undefined
  private readonly skipInitialSpace: boolean
  private readonly hasHeader: boolean
  private readonly headerRowIndex: number
  private readonly coerce: CoerceOptions
  private readonly limit: number | undefined
  private readonly dtypes: Record<string, DType> | undefined
  private readonly delimRe: RegExp | null
  /** Default locale, single-char delimiter, no custom null/bool tokens → fused field scan. */
  private readonly useFused: boolean
  private toSkip: number
  private seenFiltered = 0
  private headers: string[] | null = null
  private builders: ColumnBuilder[] = []
  private colIdx: number[] = []
  private rows = 0
  private reserveHint = 0
  /** Leftover partial record when ingesting via {@link pushChunk}. */
  private fusedCarry = ''
  /** Reused across rows so parseCsvLineInto does not allocate a fresh string[] each record. */
  private readonly cellBuf: string[] = []
  private readonly quotedBuf: boolean[] = []
  /** Fused chunk scanner state carried across chunks. */
  private fusedScanned = 0
  private fusedInQuotes = false
  /** True once `nRows` is reached: callers may stop feeding records. */
  done = false

  constructor(private readonly options: ReadCsvOptions = {}) {
    this.delimiter = options.separator ?? options.delimiter ?? ','
    this.delimCode = this.delimiter.length === 1 ? this.delimiter.charCodeAt(0) : -1
    this.quoteChar = options.quoteChar ?? '"'
    this.quoteCode = this.quoteChar.charCodeAt(0)
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
    this.useFused =
      this.delimCode >= 0 &&
      !this.skipInitialSpace &&
      !options.thousands &&
      (options.decimal ?? '.') === '.' &&
      options.nullValues === undefined &&
      options.trueValues === undefined &&
      options.falseValues === undefined
    if (options.names) this.initColumns(options.names)
  }

  /**
   * True when callers may feed the file via {@link pushChunk} (no per-line strings for data rows).
   * Disabled when comments are enabled — those need the quote-aware record splitter.
   */
  get fusedChunkMode(): boolean {
    return this.useFused && this.comment === undefined
  }

  /** Hint an expected row count so column buffers are sized once (e.g. from file size). */
  estimateRows(n: number): void {
    this.reserveHint = Math.max(0, Math.floor(n))
    if (this.builders.length) for (const b of this.builders) b.reserve(this.reserveHint)
  }

  /**
   * Ingest a decoded chunk without allocating one string per CSV record. Only valid when
   * {@link fusedChunkMode} is true. Pass `endOfInput` on the final call to flush a trailing record.
   */
  pushChunk(chunk: string, endOfInput = false): void {
    if (this.done) return
    const buf = this.fusedCarry.length > 0 ? this.fusedCarry + chunk : chunk
    // Resume scanning after the already-scanned carry with the quote state it ended in: a newline inside a
    // quoted field must not end the record (RFC 4180), even when the field spans two chunks.
    let i = this.fusedScanned
    this.fusedCarry = ''
    let start = 0
    let inQuotes = this.fusedInQuotes
    const q = this.quoteCode
    for (; i < buf.length; i++) {
      const c = buf.charCodeAt(i)
      if (c === q) {
        inQuotes = !inQuotes
        continue
      }
      if (inQuotes || (c !== 10 && c !== 13)) continue
      if (c === 13 && i + 1 >= buf.length && !endOfInput) break // may be the first half of CRLF
      this.pushRecordRange(buf, start, i)
      if (this.done) return
      if (c === 13 && i + 1 < buf.length && buf.charCodeAt(i + 1) === 10) i++
      start = i + 1
    }
    if (endOfInput) {
      if (start < buf.length) this.pushRecordRange(buf, start, buf.length)
      this.fusedScanned = 0
      this.fusedInQuotes = false
    } else {
      this.fusedCarry = buf.slice(start)
      this.fusedScanned = Math.max(0, i - start)
      this.fusedInQuotes = inQuotes
    }
  }

  /** Feed one record (a physical line with quoted newlines already joined). */
  push(line: string): void {
    if (this.done) return
    this.pushRecordRange(line, 0, line.length)
  }

  private pushRecordRange(s: string, lo: number, hi: number): void {
    if (this.done) return
    if (this.toSkip > 0) {
      this.toSkip--
      return
    }
    if (this.skipBlankLines) {
      let empty = true
      for (let j = lo; j < hi; j++) {
        if (s.charCodeAt(j) > 32) {
          empty = false
          break
        }
      }
      if (empty) return
    }
    if (this.comment !== undefined) {
      let j = lo
      while (j < hi && s.charCodeAt(j) <= 32) j++
      if (s.startsWith(this.comment, j)) return
    }

    // skipInitialSpace path still needs a materialised record string for delimRe
    let recLo = lo
    let recHi = hi
    let rec = s
    if (this.delimRe) {
      rec = s.slice(lo, hi).replace(this.delimRe, this.delimiter)
      recLo = 0
      recHi = rec.length
    }

    const idx = this.seenFiltered++
    if (!this.headers) {
      const line = rec.slice(recLo, recHi)
      if (!this.hasHeader) {
        const width = parseCsvLine(line, this.delimiter, this.quoteChar).length
        this.initColumns(Array.from({ length: width }, (_, i) => `column_${i}`))
        // fall through: this record is data
      } else {
        if (idx < this.headerRowIndex) return
        this.initColumns(dedupeHeaders(parseCsvLine(line, this.delimiter, this.quoteChar)))
        return
      }
    } else if (this.hasHeader && idx <= this.headerRowIndex) {
      return
    }

    if (this.useFused && this.pushFusedRange(rec, recLo, recHi)) {
      this.rows++
      if (this.limit !== undefined && this.rows >= this.limit) this.done = true
      return
    }

    const line = rec.slice(recLo, recHi)
    parseCsvLineInto(line, this.delimiter, this.quoteChar, this.cellBuf, this.quotedBuf)
    const cells = this.cellBuf
    if (this.skipInitialSpace) {
      for (let i = 0; i < cells.length; i++) cells[i] = cells[i]!.trimStart()
    }
    for (let k = 0; k < this.colIdx.length; k++) {
      const b = this.builders[k]!
      const raw = cells[this.colIdx[k]!] ?? ''
      // a quoted empty field ("") is an empty string, a bare empty field is null
      let v = raw === '' && this.quotedBuf[this.colIdx[k]!] ? '' : coerceValue(raw, this.coerce)
      const forced = this.dtypes?.[b.name]
      if (forced !== undefined && v !== null) v = coerceForced(v, forced)
      b.push(v)
    }
    this.rows++
    if (this.limit !== undefined && this.rows >= this.limit) this.done = true
  }

  /**
   * Unquoted default-locale record → builders without per-field strings for numbers/nulls/bools.
   * Returns false when the line contains a quote (caller falls back to the generic path).
   */
  private pushFusedRange(s: string, lo: number, hi: number): boolean {
    const q = this.quoteCode
    const d = this.delimCode
    for (let i = lo; i < hi; i++) if (s.charCodeAt(i) === q) return false

    const builders = this.builders
    const colIdx = this.colIdx
    const nWant = colIdx.length
    let filled = 0
    let field = 0
    let start = lo

    for (let i = lo; i <= hi; i++) {
      if (i < hi && s.charCodeAt(i) !== d) continue
      for (let k = 0; k < nWant; k++) {
        if (colIdx[k] === field) {
          this.acceptField(builders[k]!, s, start, i)
          filled++
        }
      }
      if (filled >= nWant) return true
      field++
      start = i + 1
    }
    for (let k = 0; k < nWant; k++) {
      if (colIdx[k]! >= field) this.acceptField(builders[k]!, s, hi, hi)
    }
    return true
  }

  private acceptField(b: ColumnBuilder, line: string, start: number, end: number): void {
    const rawStart = start
    const rawEnd = end
    while (start < end && line.charCodeAt(start) <= 32) start++
    while (end > start && line.charCodeAt(end - 1) <= 32) end--
    const forced = this.dtypes?.[b.name]

    if (start === end) {
      b.pushNull()
      return
    }

    if (isDefaultNullRange(line, start, end)) {
      b.pushNull()
      return
    }
    const truth = isDefaultBoolRange(line, start, end)
    if (truth !== undefined) {
      if (forced !== undefined) b.push(coerceForced(truth, forced))
      else b.pushBool(truth)
      return
    }

    const num = tryParsePlainNumberRange(line, start, end)
    if (typeof num === 'number') {
      if (forced !== undefined) b.push(coerceForced(num, forced))
      else b.pushNumber(num)
      return
    }
    if (typeof num === 'string') {
      // integer outside the safe range — keep digits as text (same as coerceValue)
      if (forced !== undefined) b.push(coerceForced(num, forced))
      else b.pushText(num)
      return
    }

    // Non-numeric text: coerceValue returns the untrimmed raw field.
    const raw = line.slice(rawStart, rawEnd)
    if (forced !== undefined) b.push(coerceForced(raw, forced))
    else b.pushText(raw)
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
    if (this.reserveHint > 0) for (const b of this.builders) b.reserve(this.reserveHint)
  }

  finish(): TableView {
    if (!this.headers) return tableFromColumns([])
    return tableFromColumns(this.builders.map((b) => b.finish()))
  }
}

/** Default null tokens from makeCoerceOptions, matched without allocating. */
function isDefaultNullRange(s: string, start: number, end: number): boolean {
  const n = end - start
  if (n === 0) return true
  if (n === 2) {
    const a = s.charCodeAt(start)
    const b = s.charCodeAt(start + 1)
    // na / NA
    return (a === 110 || a === 78) && (b === 97 || b === 65)
  }
  if (n === 3) {
    // nan / NaN / NAN
    const a = s.charCodeAt(start)
    const b = s.charCodeAt(start + 1)
    const c = s.charCodeAt(start + 2)
    return (
      (a === 110 || a === 78) &&
      (b === 97 || b === 65) &&
      (c === 110 || c === 78)
    )
  }
  if (n === 4) {
    // null / NULL / Null — coerceValue lowercases against the default set
    const a = s.charCodeAt(start)
    const b = s.charCodeAt(start + 1)
    const c = s.charCodeAt(start + 2)
    const d = s.charCodeAt(start + 3)
    return (
      (a === 110 || a === 78) &&
      (b === 117 || b === 85) &&
      (c === 108 || c === 76) &&
      (d === 108 || d === 76)
    )
  }
  return false
}

function isDefaultBoolRange(s: string, start: number, end: number): boolean | undefined {
  const n = end - start
  if (n === 4) {
    // true / True / TRUE
    const a = s.charCodeAt(start)
    const b = s.charCodeAt(start + 1)
    const c = s.charCodeAt(start + 2)
    const d = s.charCodeAt(start + 3)
    if (
      (a === 116 || a === 84) &&
      (b === 114 || b === 82) &&
      (c === 117 || c === 85) &&
      (d === 101 || d === 69)
    )
      return true
    return undefined
  }
  if (n === 5) {
    // false / False / FALSE
    const a = s.charCodeAt(start)
    const b = s.charCodeAt(start + 1)
    const c = s.charCodeAt(start + 2)
    const d = s.charCodeAt(start + 3)
    const e = s.charCodeAt(start + 4)
    if (
      (a === 102 || a === 70) &&
      (b === 97 || b === 65) &&
      (c === 108 || c === 76) &&
      (d === 115 || d === 83) &&
      (e === 101 || e === 69)
    )
      return false
  }
  return undefined
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
  // Rough row estimate from newlines (header included); better than repeated 2× copies.
  let lines = 1
  for (let i = 0; i < csv.length; i++) {
    const c = csv.charCodeAt(i)
    if (c === 10) lines++
  }
  builder.estimateRows(lines)
  const text = csv.replace(/^﻿/, '')
  if (builder.fusedChunkMode) {
    builder.pushChunk(text, true)
  } else {
    const splitter = new CsvRecordSplitter(options.quoteChar ?? '"')
    splitter.push(text, (r) => builder.push(r))
    splitter.end((r) => builder.push(r))
  }
  return builder.finish()
}

/**
 * Stream a Node file into a table chunk by chunk. `maxBytes` is enforced on bytes read; `nRows` stops the
 * read early. Falls back to the caller for non-Node environments.
 *
 * Acceleration (when options allow the fused path): try native Rayon CSV, then a worker_threads pool,
 * then the single-thread fused scanner.
 */
type CsvAccelerators = typeof import('./csv-parallel.js')
async function loadCsvAccelerators(): Promise<CsvAccelerators | null> {
  if (typeof process === 'undefined' || !process.versions?.node) return null
  try {
    const id = './csv-parallel.js'
    return (await import(/* @vite-ignore */ id)) as CsvAccelerators
  } catch {
    return null
  }
}

export async function streamCsvFileToTable(
  path: string,
  options: ReadCsvOptions & { maxBytes?: number; signal?: AbortSignal; encoding?: string },
): Promise<TableView> {
  // Node-only accelerators live in a sibling module loaded through an opaque specifier: bundlers cannot
  // inline it (it imports worker_threads / os / fs), so a browser build of the library stays clean and the
  // import simply fails there → single-thread path.
  const accel = await loadCsvAccelerators()
  if (accel && accel.canAccelerateCsv(options) && options.maxBytes === undefined && !options.signal) {
    const native = await accel.tryParseCsvNative(path, options)
    if (native) return native
    const parallel = await accel.tryParseCsvParallel(path, options)
    if (parallel) return parallel
  }

  const fs = await import('node:fs')
  const builder = new CsvTableBuilder(options)
  try {
    const size = fs.statSync(path).size
    // ~40–60 bytes/row on the compare-js fixture; underestimate slightly so we rarely over-allocate.
    builder.estimateRows(Math.min(50_000_000, Math.max(4096, Math.ceil(size / 48))))
  } catch {
    // ignore — reserve is optional
  }
  const decoder = new TextDecoder(options.encoding ?? 'utf-8')
  const highWaterMark =
    options.maxBytes !== undefined ? Math.min(4 << 20, Math.max(64 << 10, options.maxBytes)) : 4 << 20
  const stream = fs.createReadStream(path, { highWaterMark, signal: options.signal })
  let bytes = 0
  let first = true
  const fused = builder.fusedChunkMode
  const splitter = fused ? null : new CsvRecordSplitter(options.quoteChar ?? '"')
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
      if (fused) builder.pushChunk(text, false)
      else splitter!.push(text, (r) => builder.push(r))
      if (builder.done) break
    }
    if (!builder.done) {
      const tail = decoder.decode()
      if (fused) {
        builder.pushChunk(tail, true)
      } else {
        if (tail) splitter!.push(tail, (r) => builder.push(r))
        splitter!.end((r) => builder.push(r))
      }
    }
  } finally {
    stream.destroy()
  }
  return builder.finish()
}
