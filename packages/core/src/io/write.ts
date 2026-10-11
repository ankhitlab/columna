import type { Column, TableView } from '@columna/arrow'
import { getColumn, isValid, tableFromColumns, toRowObjects } from '@columna/arrow'

/**
 * Sync `require` without a static `node:module` import (that breaks browser bundles).
 * Available in CJS builds and some Node loaders; ESM/browser callers should use the async writers.
 */
function syncNodeRequire(id: string): unknown {
  try {
    // eslint-disable-next-line @typescript-eslint/no-implied-eval, no-new-func
    const req = new Function('return typeof require === "function" ? require : undefined')() as
      | ((specifier: string) => unknown)
      | undefined
    if (typeof req === 'function') return req(id)
  } catch {
    /* ignore */
  }
  throw new Error(
    `${id}: sync require is unavailable in this environment — use the async writer (e.g. writeParquet) instead`,
  )
}

/** Project a table to named columns (order preserved). */
export function projectTable(table: TableView, columns?: string[]): TableView {
  if (!columns || columns.length === 0) return table
  return tableFromColumns(columns.map((name) => getColumn(table, name)))
}

// ───────────────────────────── CSV ─────────────────────────────

export type CsvWriteOptions = {
  /**
   * Formula-injection protection for exports that will be opened in Excel-like applications: text cells
   * (and header names) starting with = + - @ tab or CR are prefixed with "'" and quoted. Off by default —
   * it alters the data (pandas / polars do not do this either); turn it on for exports of untrusted text.
   */
  escapeFormulas?: boolean
  /** Column separator (default `,`). Alias of `delimiter`. */
  separator?: string
  /** Alias of `separator` (pandas-style). */
  delimiter?: string
  /** Text encoding when writing files (default `utf-8`). Ignored by `toCsv()` string return. */
  encoding?: string
  /** Quote character (default `"`). */
  quoteChar?: string
  /** Write header row (default true). */
  header?: boolean
  /** Representation for null/missing (default `''`). pandas `na_rep`. */
  naRep?: string
  /** Decimal mark for numbers (default `.`). */
  decimal?: string
  /** Line terminator (default `\n`). pandas `lineterminator`. */
  lineTerminator?: string
  /** Subset of columns to write, in order. */
  columns?: string[]
}

type ResolvedCsv = {
  sep: string
  quoteChar: string
  header: boolean
  naRep: string
  decimal: string
  lineTerminator: string
  escapeFormulas: boolean
  encoding: string
}

function resolveCsvOptions(options: CsvWriteOptions): ResolvedCsv {
  const sep = options.separator ?? options.delimiter ?? ','
  if (sep.length === 0) throw new RangeError('CSV separator must not be empty')
  const quoteChar = options.quoteChar ?? '"'
  if (quoteChar.length !== 1) throw new RangeError('CSV quoteChar must be a single character')
  const decimal = options.decimal ?? '.'
  if (decimal.length !== 1) throw new RangeError('CSV decimal must be a single character')
  return {
    sep,
    quoteChar,
    header: options.header !== false,
    naRep: options.naRep ?? '',
    decimal,
    lineTerminator: options.lineTerminator ?? '\n',
    escapeFormulas: Boolean(options.escapeFormulas),
    encoding: options.encoding ?? 'utf-8',
  }
}

function csvOptionsAreNativeDefaults(options: CsvWriteOptions, resolved: ResolvedCsv): boolean {
  if (resolved.escapeFormulas) return false
  if (resolved.sep !== ',' || resolved.quoteChar !== '"' || !resolved.header) return false
  if (resolved.naRep !== '' || resolved.decimal !== '.' || resolved.lineTerminator !== '\n') return false
  if (options.columns && options.columns.length > 0) return false
  const enc = resolved.encoding.toLowerCase().replace('-', '')
  if (enc !== 'utf8') return false
  return true
}

function needsCsvEscape(s: string, sep: string, quoteChar: string): boolean {
  if (s.includes(sep) || s.includes(quoteChar)) return true
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i)
    if (c === 10 || c === 13) return true
  }
  return false
}

function escapeCsv(s: string, sep: string, quoteChar: string): string {
  // "" for an empty string: the reader maps a bare empty field to null, a quoted one to "" (polars convention)
  if (s === '') return quoteChar + quoteChar
  if (!needsCsvEscape(s, sep, quoteChar)) return s
  const doubled = s.split(quoteChar).join(quoteChar + quoteChar)
  return `${quoteChar}${doubled}${quoteChar}`
}

/**
 * Leading characters a spreadsheet interprets as a formula (OWASP CSV injection list: = + - @, tab, CR).
 * Only text cells are at risk; numeric columns are written as numbers and a leading "-" there is a sign.
 */
const FORMULA_LEAD = /^[=+\-@\t\r]/

function escapeCsvSafe(s: string, sep: string, quoteChar: string): string {
  if (FORMULA_LEAD.test(s)) {
    const body = `'` + s
    const doubled = body.split(quoteChar).join(quoteChar + quoteChar)
    return `${quoteChar}${doubled}${quoteChar}`
  }
  return escapeCsv(s, sep, quoteChar)
}

type NumArr = Float64Array | Float32Array | Int32Array | Uint32Array

type ColWriter =
  | { kind: 'utf8'; data: string[]; bm: Uint8Array | undefined; cache: Map<string, string> }
  | { kind: 'cat'; codes: Uint32Array; escaped: string[]; bm: Uint8Array | undefined }
  | { kind: 'bool'; data: Uint8Array; bm: Uint8Array | undefined }
  | {
      kind: 'num'
      data: NumArr
      bm: Uint8Array | undefined
      dense: string[] | null
      denseMin: number
    }

const SMALL_INT_STR: string[] = Array.from({ length: 1024 }, (_, i) => String(i))

function prepareWriter(col: Column, esc: (s: string) => string): ColWriter {
  const bm = col.nullBitmap
  const dtype = col.field.dtype

  if (dtype === 'utf8') {
    return { kind: 'utf8', data: col.data as string[], bm, cache: new Map() }
  }
  if (dtype === 'bool') {
    return { kind: 'bool', data: col.data as Uint8Array, bm }
  }
  if (dtype === 'category' && col.dictionary) {
    return {
      kind: 'cat',
      codes: col.data as Uint32Array,
      escaped: col.dictionary.map((s) => esc(s)),
      bm,
    }
  }

  const data = col.data as NumArr
  const n = data.length
  const sample = Math.min(n, 4096)
  let lo = Infinity
  let hi = -Infinity
  let allInt = true
  let seen = 0
  for (let i = 0; i < sample; i++) {
    if (bm && !isValid(bm, i)) continue
    const v = data[i]!
    seen++
    if (!Number.isInteger(v)) {
      allInt = false
      break
    }
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  let dense: string[] | null = null
  let denseMin = 0
  if (allInt && seen > 0 && Number.isFinite(lo) && Number.isFinite(hi)) {
    const span = hi - lo + 1
    if (span > 0 && span <= 20_000 && span <= Math.max(seen * 4, 256)) {
      denseMin = lo
      dense = new Array(span)
      for (let o = 0; o < span; o++) dense[o] = String(lo + o)
    }
  }
  return { kind: 'num', data, bm, dense, denseMin }
}

function formatNumber(v: number, decimal: string, sep: string, quoteChar: string): string {
  let s: string
  if (v >= 0 && v < SMALL_INT_STR.length && Number.isInteger(v)) s = SMALL_INT_STR[v]!
  else s = String(v)
  if (decimal !== '.' && s.includes('.')) s = s.replace('.', decimal)
  if (needsCsvEscape(s, sep, quoteChar)) return escapeCsv(s, sep, quoteChar)
  return s
}

/** Cell string for a single row index — used by the row-wise CSV emitter. */
function cellAt(
  w: ColWriter,
  i: number,
  esc: (s: string) => string,
  resolved: ResolvedCsv,
  naOut: string,
): string {
  const bm = w.bm
  if (bm && !isValid(bm, i)) return naOut
  if (w.kind === 'utf8') {
    const s = w.data[i] ?? ''
    let e = w.cache.get(s)
    if (e === undefined) {
      e = esc(s)
      if (w.cache.size < 65_536) w.cache.set(s, e)
    }
    return e
  }
  if (w.kind === 'cat') return w.escaped[w.codes[i]!] ?? naOut
  if (w.kind === 'bool') return w.data[i] ? 'true' : 'false'
  const v = w.data[i]!
  if (w.dense && Number.isInteger(v) && resolved.decimal === '.') {
    const o = v - w.denseMin
    if (o >= 0 && o < w.dense.length) return w.dense[o]!
  }
  return formatNumber(v, resolved.decimal, resolved.sep, resolved.quoteChar)
}

/** Rows per emitted chunk: bounds the working set alive at once. */
export const CSV_CHUNK_ROWS = 32_768

function needsQuoteInDict(s: string): boolean {
  return needsCsvEscape(s, ',', '"')
}

/** True when every text/category value is free of `,` `"` CR LF (safe for native unquoted write). */
function tableIsUnquotedSafe(table: TableView): boolean {
  for (const col of table.columns) {
    if (needsQuoteInDict(col.field.name)) return false
    const dtype = col.field.dtype
    if (dtype === 'utf8') {
      for (const s of col.data as string[]) {
        if (needsQuoteInDict(s)) return false
      }
    } else if (dtype === 'category' && col.dictionary) {
      for (const s of col.dictionary) {
        if (needsQuoteInDict(s)) return false
      }
    }
  }
  return true
}

type NativeWriteCol = {
  name: string
  dtype: string
  nullBitmap?: Uint8Array
  f64Data?: Float64Array
  i32Data?: Int32Array
  boolData?: Uint8Array
  catCodes?: Uint32Array
  dictionary?: string[]
  utf8Data?: string[]
}

function toNativeWriteColumns(table: TableView): NativeWriteCol[] {
  return table.columns.map((c) => {
    const dtype = c.field.dtype
    const base: NativeWriteCol = { name: c.field.name, dtype, nullBitmap: c.nullBitmap }
    if (dtype === 'i32') base.i32Data = c.data as Int32Array
    else if (dtype === 'bool') base.boolData = c.data as Uint8Array
    else if (dtype === 'category') {
      base.catCodes = c.data as Uint32Array
      base.dictionary = c.dictionary
    } else if (dtype === 'utf8') base.utf8Data = c.data as string[]
    else if (dtype === 'f32') {
      const src = c.data as Float32Array
      const f = new Float64Array(src.length)
      for (let i = 0; i < src.length; i++) f[i] = src[i]!
      base.dtype = 'f64'
      base.f64Data = f
    } else {
      base.dtype = dtype === 'datetime' ? 'datetime' : 'f64'
      base.f64Data = c.data as Float64Array
    }
    return base
  })
}

async function tryWriteCsvNative(table: TableView, path: string, options: CsvWriteOptions): Promise<boolean> {
  const resolved = resolveCsvOptions(options)
  if (!csvOptionsAreNativeDefaults(options, resolved)) return false
  if (!tableIsUnquotedSafe(table)) return false
  try {
    const id = '@columna/' + 'native'
    const mod = (await import(/* webpackIgnore: true */ /* @vite-ignore */ id)) as {
      isNativeLoaded?: boolean
      writeCsvUnquoted?: (path: string, numRows: number, columns: NativeWriteCol[]) => void
    }
    if (!mod.isNativeLoaded || typeof mod.writeCsvUnquoted !== 'function') return false
    mod.writeCsvUnquoted(path, table.numRows, toNativeWriteColumns(table))
    return true
  } catch {
    return false
  }
}

function joinCells(parts: string[], sep: string): string {
  return parts.join(sep)
}

/**
 * Serialize a table to CSV as a sequence of chunks (header first, then blocks of rows). Cells are
 * formatted row-wise (no ncols × chunk temporary string grid), so `writeCsv(path)` streams with memory
 * bounded by the chunk, not the table.
 */
export function* tableToCsvChunks(table: TableView, options: CsvWriteOptions = {}): Generator<string, void, undefined> {
  const resolved = resolveCsvOptions(options)
  const view = projectTable(table, options.columns)
  const n = view.numRows
  const cols = view.columns
  const ncols = cols.length
  const { sep, quoteChar } = resolved
  const esc = (s: string) =>
    resolved.escapeFormulas ? escapeCsvSafe(s, sep, quoteChar) : escapeCsv(s, sep, quoteChar)
  const naOut = resolved.naRep === '' ? '' : esc(resolved.naRep)

  if (resolved.header) {
    yield cols.map((c) => esc(c.field.name)).join(sep)
  }
  if (n === 0) return
  if (ncols === 0) {
    for (let start = 0; start < n; start += CSV_CHUNK_ROWS) {
      const len = Math.min(n, start + CSV_CHUNK_ROWS) - start
      yield new Array(len).fill('').join(resolved.lineTerminator)
    }
    return
  }

  const writers = cols.map((c) => prepareWriter(c, esc))
  const chunk = new Array<string>(CSV_CHUNK_ROWS)
  for (let start = 0; start < n; start += CSV_CHUNK_ROWS) {
    const end = Math.min(n, start + CSV_CHUNK_ROWS)
    const len = end - start
    if (ncols === 1) {
      const w0 = writers[0]!
      for (let i = 0; i < len; i++) chunk[i] = cellAt(w0, start + i, esc, resolved, naOut)
    } else if (ncols === 2) {
      const w0 = writers[0]!
      const w1 = writers[1]!
      for (let i = 0; i < len; i++) {
        const r = start + i
        chunk[i] = joinCells([cellAt(w0, r, esc, resolved, naOut), cellAt(w1, r, esc, resolved, naOut)], sep)
      }
    } else if (ncols === 3) {
      const w0 = writers[0]!
      const w1 = writers[1]!
      const w2 = writers[2]!
      for (let i = 0; i < len; i++) {
        const r = start + i
        chunk[i] = joinCells(
          [cellAt(w0, r, esc, resolved, naOut), cellAt(w1, r, esc, resolved, naOut), cellAt(w2, r, esc, resolved, naOut)],
          sep,
        )
      }
    } else if (ncols === 4) {
      const [w0, w1, w2, w3] = writers as [ColWriter, ColWriter, ColWriter, ColWriter]
      for (let i = 0; i < len; i++) {
        const r = start + i
        chunk[i] = joinCells(
          [
            cellAt(w0, r, esc, resolved, naOut),
            cellAt(w1, r, esc, resolved, naOut),
            cellAt(w2, r, esc, resolved, naOut),
            cellAt(w3, r, esc, resolved, naOut),
          ],
          sep,
        )
      }
    } else if (ncols === 5) {
      const [w0, w1, w2, w3, w4] = writers as [ColWriter, ColWriter, ColWriter, ColWriter, ColWriter]
      for (let i = 0; i < len; i++) {
        const r = start + i
        chunk[i] = joinCells(
          [
            cellAt(w0, r, esc, resolved, naOut),
            cellAt(w1, r, esc, resolved, naOut),
            cellAt(w2, r, esc, resolved, naOut),
            cellAt(w3, r, esc, resolved, naOut),
            cellAt(w4, r, esc, resolved, naOut),
          ],
          sep,
        )
      }
    } else if (ncols === 6) {
      const [w0, w1, w2, w3, w4, w5] = writers as [
        ColWriter,
        ColWriter,
        ColWriter,
        ColWriter,
        ColWriter,
        ColWriter,
      ]
      for (let i = 0; i < len; i++) {
        const r = start + i
        chunk[i] = joinCells(
          [
            cellAt(w0, r, esc, resolved, naOut),
            cellAt(w1, r, esc, resolved, naOut),
            cellAt(w2, r, esc, resolved, naOut),
            cellAt(w3, r, esc, resolved, naOut),
            cellAt(w4, r, esc, resolved, naOut),
            cellAt(w5, r, esc, resolved, naOut),
          ],
          sep,
        )
      }
    } else if (ncols === 7) {
      const [w0, w1, w2, w3, w4, w5, w6] = writers as [
        ColWriter,
        ColWriter,
        ColWriter,
        ColWriter,
        ColWriter,
        ColWriter,
        ColWriter,
      ]
      for (let i = 0; i < len; i++) {
        const r = start + i
        chunk[i] = joinCells(
          [
            cellAt(w0, r, esc, resolved, naOut),
            cellAt(w1, r, esc, resolved, naOut),
            cellAt(w2, r, esc, resolved, naOut),
            cellAt(w3, r, esc, resolved, naOut),
            cellAt(w4, r, esc, resolved, naOut),
            cellAt(w5, r, esc, resolved, naOut),
            cellAt(w6, r, esc, resolved, naOut),
          ],
          sep,
        )
      }
    } else if (ncols === 8) {
      const [w0, w1, w2, w3, w4, w5, w6, w7] = writers as [
        ColWriter,
        ColWriter,
        ColWriter,
        ColWriter,
        ColWriter,
        ColWriter,
        ColWriter,
        ColWriter,
      ]
      for (let i = 0; i < len; i++) {
        const r = start + i
        chunk[i] = joinCells(
          [
            cellAt(w0, r, esc, resolved, naOut),
            cellAt(w1, r, esc, resolved, naOut),
            cellAt(w2, r, esc, resolved, naOut),
            cellAt(w3, r, esc, resolved, naOut),
            cellAt(w4, r, esc, resolved, naOut),
            cellAt(w5, r, esc, resolved, naOut),
            cellAt(w6, r, esc, resolved, naOut),
            cellAt(w7, r, esc, resolved, naOut),
          ],
          sep,
        )
      }
    } else {
      for (let i = 0; i < len; i++) {
        const r = start + i
        let row = cellAt(writers[0]!, r, esc, resolved, naOut)
        for (let c = 1; c < ncols; c++) row += sep + cellAt(writers[c]!, r, esc, resolved, naOut)
        chunk[i] = row
      }
    }
    yield chunk.length === len
      ? chunk.join(resolved.lineTerminator)
      : chunk.slice(0, len).join(resolved.lineTerminator)
  }
}

/** Serialize a table to CSV text. For files prefer `writeCsvText(path)`, which streams the chunks. */
export function tableToCsv(table: TableView, options: CsvWriteOptions = {}): string {
  const lt = resolveCsvOptions(options).lineTerminator
  const parts: string[] = []
  for (const chunk of tableToCsvChunks(table, options)) parts.push(chunk)
  return parts.join(lt)
}

async function writeNodeFile(path: string, data: string | Uint8Array, encoding?: string): Promise<void> {
  const fs = await import('node:fs/promises')
  if (typeof data === 'string' && encoding) {
    await fs.writeFile(path, data, { encoding: nodeBufferEncoding(encoding) })
  } else {
    await fs.writeFile(path, data)
  }
}

function nodeBufferEncoding(encoding: string): BufferEncoding {
  const key = encoding.toLowerCase().replace(/_/g, '-')
  const map: Record<string, BufferEncoding> = {
    'utf-8': 'utf8',
    utf8: 'utf8',
    'utf-16le': 'utf16le',
    utf16le: 'utf16le',
    ucs2: 'utf16le',
    'ucs-2': 'utf16le',
    latin1: 'latin1',
    binary: 'latin1',
    ascii: 'ascii',
    hex: 'hex',
    base64: 'base64',
    'base64url': 'base64url' as BufferEncoding,
  }
  const enc = map[key]
  if (!enc) throw new Error(`Unsupported CSV/JSON encoding: ${encoding}`)
  return enc
}

/**
 * Write CSV to a Node path (streamed chunk by chunk, honouring back-pressure) or return the CSV string
 * when the path is omitted. With a path the function resolves to '' — the whole text is never built.
 */
export async function writeCsvText(table: TableView, path?: string, options: CsvWriteOptions = {}): Promise<string> {
  if (!path) return tableToCsv(table, options)
  if (await tryWriteCsvNative(table, path, options)) return ''
  const resolved = resolveCsvOptions(options)
  const enc = nodeBufferEncoding(resolved.encoding)
  const fs = await import('node:fs')
  const out = fs.createWriteStream(path, { encoding: enc })
  let rejectWrite: ((err: Error) => void) | null = null
  const onError = (err: Error) => {
    if (rejectWrite) rejectWrite(err)
  }
  out.on('error', onError)
  const write = (s: string) =>
    new Promise<void>((resolve, reject) => {
      rejectWrite = reject
      if (out.write(s)) {
        rejectWrite = null
        resolve()
      } else {
        out.once('drain', () => {
          rejectWrite = null
          resolve()
        })
      }
    })
  try {
    let first = true
    for (const chunk of tableToCsvChunks(table, options)) {
      await write(first ? chunk : resolved.lineTerminator + chunk)
      first = false
    }
    await new Promise<void>((resolve, reject) => {
      rejectWrite = reject
      out.end((err?: Error | null) => {
        rejectWrite = null
        if (err) reject(err)
        else resolve()
      })
    })
  } catch (err) {
    out.destroy()
    throw err
  } finally {
    out.off('error', onError)
  }
  return ''
}

// ───────────────────────────── Parquet ─────────────────────────────

export type ParquetWriteOptions = {
  columns?: string[]
  /** Default `SNAPPY`. `GZIP` uses Node `zlib` (Node only). */
  codec?: 'SNAPPY' | 'UNCOMPRESSED' | 'GZIP'
  rowGroupSize?: number | number[]
  statistics?: boolean
}

export type ParquetLikeWriteOptions = {
  columns?: string[]
}

/** Minimal parquet-like JSON payload (same as `@columna/wasm` writeParquetLike).
 *  This is NOT Apache Parquet — use {@link writeParquetLikeBytes} / `writeParquetLike`. */
export function tableToParquetLike(table: TableView, options: ParquetLikeWriteOptions = {}): Uint8Array {
  const view = projectTable(table, options.columns)
  const payload = {
    format: 'columna-parquet-like-v1',
    numRows: view.numRows,
    schema: view.schema,
    columns: view.columns.map((c) => {
      const values: Array<number | string | boolean | null> = []
      for (let i = 0; i < view.numRows; i++) {
        if (!isValid(c.nullBitmap, i)) values.push(null)
        else {
          const dtype = c.field.dtype
          if (dtype === 'utf8') values.push((c.data as string[])[i]!)
          else if (dtype === 'bool') values.push(Boolean((c.data as Uint8Array)[i]))
          else if (dtype === 'category' && c.dictionary) {
            values.push(c.dictionary[(c.data as Uint32Array)[i]!] ?? null)
          } else values.push(Number((c.data as Float64Array)[i]!))
        }
      }
      return { name: c.field.name, dtype: c.field.dtype, values, dictionary: c.dictionary }
    }),
  }
  return new TextEncoder().encode(JSON.stringify(payload))
}

/** Write the custom JSON "parquet-like" format (round-trips with `readParquetLike`). */
export async function writeParquetLikeBytes(
  table: TableView,
  path?: string,
  options: ParquetLikeWriteOptions = {},
): Promise<Uint8Array> {
  const bytes = tableToParquetLike(table, options)
  if (path) await writeNodeFile(path, bytes)
  return bytes
}

type ParquetBasicType = 'BOOLEAN' | 'INT32' | 'INT64' | 'FLOAT' | 'DOUBLE' | 'STRING' | 'TIMESTAMP'

/** Expand a columna column into hyparquet-writer `columnData` (nulls as JS null). */
export function tableToParquetColumnData(
  table: TableView,
): Array<{ name: string; data: Array<number | string | boolean | null>; type: ParquetBasicType; nullable: boolean }> {
  const n = table.numRows
  return table.columns.map((c) => {
    const dtype = c.field.dtype
    const nullable = Boolean(c.nullBitmap) || c.field.nullable
    let type: ParquetBasicType
    const data: Array<number | string | boolean | null> = new Array(n)
    switch (dtype) {
      case 'bool': {
        type = 'BOOLEAN'
        const src = c.data as Uint8Array
        for (let i = 0; i < n; i++) {
          data[i] = !isValid(c.nullBitmap, i) ? null : Boolean(src[i])
        }
        break
      }
      case 'i32': {
        type = 'INT32'
        const src = c.data as Int32Array
        for (let i = 0; i < n; i++) {
          data[i] = !isValid(c.nullBitmap, i) ? null : src[i]!
        }
        break
      }
      case 'u32': {
        // Parquet INT32 is signed; values above 2^31-1 widen to INT64.
        let needsI64 = false
        const src = c.data as Uint32Array
        for (let i = 0; i < n; i++) {
          if (isValid(c.nullBitmap, i) && src[i]! > 0x7fffffff) {
            needsI64 = true
            break
          }
        }
        type = needsI64 ? 'INT64' : 'INT32'
        for (let i = 0; i < n; i++) {
          data[i] = !isValid(c.nullBitmap, i) ? null : src[i]!
        }
        break
      }
      case 'f32': {
        type = 'FLOAT'
        const src = c.data as Float32Array
        for (let i = 0; i < n; i++) {
          data[i] = !isValid(c.nullBitmap, i) ? null : src[i]!
        }
        break
      }
      case 'f64': {
        type = 'DOUBLE'
        const src = c.data as Float64Array
        for (let i = 0; i < n; i++) {
          data[i] = !isValid(c.nullBitmap, i) ? null : src[i]!
        }
        break
      }
      case 'datetime': {
        type = 'TIMESTAMP'
        const src = c.data as Float64Array
        for (let i = 0; i < n; i++) {
          data[i] = !isValid(c.nullBitmap, i) ? null : src[i]!
        }
        break
      }
      case 'category': {
        type = 'STRING'
        const codes = c.data as Uint32Array
        const dict = c.dictionary ?? []
        for (let i = 0; i < n; i++) {
          if (!isValid(c.nullBitmap, i)) data[i] = null
          else data[i] = dict[codes[i]!] ?? null
        }
        break
      }
      case 'utf8':
      default: {
        type = 'STRING'
        const src = c.data as string[]
        for (let i = 0; i < n; i++) {
          data[i] = !isValid(c.nullBitmap, i) ? null : src[i]!
        }
        break
      }
    }
    return { name: c.field.name, data, type, nullable }
  })
}

/** Sync Apache Parquet bytes (hyparquet-writer). Uses `createRequire` for CJS deps. */
export function tableToParquetBytes(table: TableView, options: ParquetWriteOptions = {}): Uint8Array {
  const { parquetWriteBuffer } = syncNodeRequire('hyparquet-writer') as typeof import('hyparquet-writer')
  const view = projectTable(table, options.columns)
  const columnData = tableToParquetColumnData(view)
  const codec = options.codec ?? 'SNAPPY'
  const writeOpts: {
    columnData: typeof columnData
    codec: typeof codec
    rowGroupSize?: number | number[]
    statistics?: boolean
    compressors?: Record<string, (input: Uint8Array) => Uint8Array>
  } = { columnData, codec }
  if (options.rowGroupSize !== undefined) writeOpts.rowGroupSize = options.rowGroupSize
  if (options.statistics !== undefined) writeOpts.statistics = options.statistics
  if (codec === 'GZIP') {
    const { gzipSync } = syncNodeRequire('node:zlib') as typeof import('node:zlib')
    writeOpts.compressors = { GZIP: (u: Uint8Array) => gzipSync(u) }
  }
  const ab = parquetWriteBuffer(writeOpts)
  return new Uint8Array(ab)
}

/**
 * Write Apache Parquet (via hyparquet-writer). Round-trips with {@link DataFrame.readParquet}.
 * Default codec is SNAPPY.
 */
export async function writeParquetBytes(
  table: TableView,
  path?: string,
  options: ParquetWriteOptions = {},
): Promise<Uint8Array> {
  const { parquetWriteBuffer } = await import('hyparquet-writer')
  const view = projectTable(table, options.columns)
  const columnData = tableToParquetColumnData(view)
  const codec = options.codec ?? 'SNAPPY'
  const writeOpts: Parameters<typeof parquetWriteBuffer>[0] = { columnData, codec }
  if (options.rowGroupSize !== undefined) writeOpts.rowGroupSize = options.rowGroupSize
  if (options.statistics !== undefined) writeOpts.statistics = options.statistics
  if (codec === 'GZIP') {
    const { gzipSync } = await import('node:zlib')
    writeOpts.compressors = { GZIP: (u: Uint8Array) => gzipSync(u) }
  }
  const ab = parquetWriteBuffer(writeOpts)
  const bytes = new Uint8Array(ab)
  if (path) await writeNodeFile(path, bytes)
  return bytes
}

// ───────────────────────────── JSON ─────────────────────────────

export type JsonWriteOptions = {
  /** `records` (default), `columns`, `values`, or `lines` (NDJSON). */
  orient?: 'records' | 'columns' | 'values' | 'lines'
  /** Alias of `orient: 'lines'`. */
  lines?: boolean
  columns?: string[]
  /** File encoding for `writeJson(path)` (default `utf-8`). */
  encoding?: string
  /** Pretty-print indent; default `null` (compact). Ignored for `lines`. */
  indent?: number | null
}

function cellJson(table: TableView, colIdx: number, row: number): unknown {
  const col = table.columns[colIdx]!
  if (!isValid(col.nullBitmap, row)) return null
  const dtype = col.field.dtype
  if (dtype === 'utf8') return (col.data as string[])[row] ?? null
  if (dtype === 'bool') return Boolean((col.data as Uint8Array)[row])
  if (dtype === 'category' && col.dictionary) {
    return col.dictionary[(col.data as Uint32Array)[row]!] ?? null
  }
  return Number((col.data as Float64Array | Int32Array | Uint32Array | Float32Array)[row]!)
}

export function tableToJson(table: TableView, options: JsonWriteOptions = {}): string {
  const view = projectTable(table, options.columns)
  const linesMode = options.lines === true || options.orient === 'lines'
  const orient = linesMode ? 'lines' : (options.orient ?? 'records')
  const indent = options.indent ?? null
  const space = indent == null ? undefined : indent

  if (orient === 'lines') {
    const rows = toRowObjects(view)
    return rows.map((r) => JSON.stringify(r)).join('\n') + (rows.length ? '\n' : '')
  }
  if (orient === 'records') {
    return JSON.stringify(toRowObjects(view), null, space)
  }
  if (orient === 'columns') {
    const obj: Record<string, unknown[]> = {}
    for (let c = 0; c < view.columns.length; c++) {
      const name = view.schema[c]!.name
      const values: unknown[] = new Array(view.numRows)
      for (let i = 0; i < view.numRows; i++) values[i] = cellJson(view, c, i)
      obj[name] = values
    }
    return JSON.stringify(obj, null, space)
  }
  if (orient === 'values') {
    const matrix: unknown[][] = new Array(view.numRows)
    for (let i = 0; i < view.numRows; i++) {
      const row: unknown[] = new Array(view.columns.length)
      for (let c = 0; c < view.columns.length; c++) row[c] = cellJson(view, c, i)
      matrix[i] = row
    }
    return JSON.stringify(matrix, null, space)
  }
  throw new Error(`Unsupported JSON orient: ${String(orient)}`)
}

export async function writeJsonText(
  table: TableView,
  path?: string,
  options: JsonWriteOptions = {},
): Promise<string> {
  const text = tableToJson(table, options)
  if (!path) return text
  await writeNodeFile(path, text, options.encoding ?? 'utf-8')
  return ''
}

// ───────────────────────────── Excel ─────────────────────────────

export type ExcelWriteOptions = {
  /** Sheet name or 0-based index when replacing (default `'Sheet1'`). */
  sheet?: string | number
  /** Alias of `sheet`. */
  sheetName?: string | number
  columns?: string[]
  /** Write header row (default true). */
  header?: boolean
}

export async function tableToExcelBytes(table: TableView, options: ExcelWriteOptions = {}): Promise<Uint8Array> {
  const XLSX = await import('xlsx')
  const view = projectTable(table, options.columns)
  const header = options.header !== false
  const names = view.schema.map((f) => f.name)
  const aoa: unknown[][] = []
  if (header) aoa.push(names)
  for (let i = 0; i < view.numRows; i++) {
    const row: unknown[] = new Array(names.length)
    for (let c = 0; c < view.columns.length; c++) {
      const col = view.columns[c]!
      if (!isValid(col.nullBitmap, i)) {
        row[c] = null
        continue
      }
      const dtype = col.field.dtype
      if (dtype === 'utf8') row[c] = (col.data as string[])[i]!
      else if (dtype === 'bool') row[c] = Boolean((col.data as Uint8Array)[i])
      else if (dtype === 'category' && col.dictionary) {
        row[c] = col.dictionary[(col.data as Uint32Array)[i]!] ?? null
      } else if (dtype === 'datetime') {
        row[c] = new Date(Number((col.data as Float64Array)[i]!))
      } else {
        row[c] = Number((col.data as Float64Array | Int32Array | Uint32Array | Float32Array)[i]!)
      }
    }
    aoa.push(row)
  }
  const ws = XLSX.utils.aoa_to_sheet(aoa)
  const wb = XLSX.utils.book_new()
  const sheetRef = options.sheetName ?? options.sheet ?? 'Sheet1'
  const sheetName = typeof sheetRef === 'number' ? `Sheet${sheetRef + 1}` : String(sheetRef)
  XLSX.utils.book_append_sheet(wb, ws, sheetName)
  const buf = XLSX.write(wb, { type: 'array', bookType: 'xlsx' }) as ArrayBuffer | Uint8Array
  return buf instanceof Uint8Array ? buf : new Uint8Array(buf)
}

export async function writeExcelBytes(
  table: TableView,
  path?: string,
  options: ExcelWriteOptions = {},
): Promise<Uint8Array> {
  const bytes = await tableToExcelBytes(table, options)
  if (path) await writeNodeFile(path, bytes)
  return bytes
}
