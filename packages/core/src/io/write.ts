import type { Column, TableView } from '@columna/arrow'
import { isValid } from '@columna/arrow'

function needsCsvEscape(s: string): boolean {
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i)
    if (c === 34 || c === 44 || c === 10 || c === 13) return true
  }
  return false
}

function escapeCsv(s: string): string {
  // "" for an empty string: the reader maps a bare empty field to null, a quoted one to "" (polars convention)
  if (s === '') return '""'
  if (!needsCsvEscape(s)) return s
  return `"${s.replace(/"/g, '""')}"`
}

/**
 * Leading characters a spreadsheet interprets as a formula (OWASP CSV injection list: = + - @, tab, CR).
 * Only text cells are at risk; numeric columns are written as numbers and a leading "-" there is a sign.
 */
const FORMULA_LEAD = /^[=+\-@\t\r]/

/**
 * Neutralise a text cell for spreadsheet consumers: a leading formula trigger gets a "'" prefix (Excel /
 * LibreOffice / Sheets then treat the cell as text) and the cell is always quoted so the quote survives.
 */
function escapeCsvSafe(s: string): string {
  if (FORMULA_LEAD.test(s)) return `"'${s.replace(/"/g, '""')}"`
  return escapeCsv(s)
}

export type CsvWriteOptions = {
  /**
   * Formula-injection protection for exports that will be opened in Excel-like applications: text cells
   * (and header names) starting with = + - @ tab or CR are prefixed with "'" and quoted. Off by default —
   * it alters the data (pandas / polars do not do this either); turn it on for exports of untrusted text.
   */
  escapeFormulas?: boolean
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

/** Cell string for a single row index — used by the row-wise CSV emitter. */
function cellAt(w: ColWriter, i: number, esc: (s: string) => string): string {
  const bm = w.bm
  if (bm && !isValid(bm, i)) return ''
  if (w.kind === 'utf8') {
    const s = w.data[i] ?? ''
    let e = w.cache.get(s)
    if (e === undefined) {
      e = esc(s)
      if (w.cache.size < 65_536) w.cache.set(s, e)
    }
    return e
  }
  if (w.kind === 'cat') return w.escaped[w.codes[i]!] ?? ''
  if (w.kind === 'bool') return w.data[i] ? 'true' : 'false'
  const v = w.data[i]!
  if (w.dense && Number.isInteger(v)) {
    const o = v - w.denseMin
    if (o >= 0 && o < w.dense.length) return w.dense[o]!
  }
  if (v >= 0 && v < SMALL_INT_STR.length && Number.isInteger(v)) return SMALL_INT_STR[v]!
  return String(v)
}

/** Rows per emitted chunk: bounds the working set alive at once. */
export const CSV_CHUNK_ROWS = 32_768

function needsQuoteInDict(s: string): boolean {
  return needsCsvEscape(s)
}

/** True when every text/category value is free of `,` `"` CR LF (safe for native unquoted write). */
function tableIsUnquotedSafe(table: TableView): boolean {
  for (const col of table.columns) {
    if (needsCsvEscape(col.field.name)) return false
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
  if (options.escapeFormulas) return false
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

/**
 * Serialize a table to CSV as a sequence of chunks (header first, then blocks of rows). Cells are
 * formatted row-wise (no ncols × chunk temporary string grid), so `writeCsv(path)` streams with memory
 * bounded by the chunk, not the table.
 */
export function* tableToCsvChunks(table: TableView, options: CsvWriteOptions = {}): Generator<string, void, undefined> {
  const n = table.numRows
  const cols = table.columns
  const ncols = cols.length
  const esc = options.escapeFormulas ? escapeCsvSafe : escapeCsv
  yield cols.map((c) => esc(c.field.name)).join(',')
  if (n === 0) return
  const writers = cols.map((c) => prepareWriter(c, esc))
  const chunk = new Array<string>(CSV_CHUNK_ROWS)
  for (let start = 0; start < n; start += CSV_CHUNK_ROWS) {
    const end = Math.min(n, start + CSV_CHUNK_ROWS)
    const len = end - start
    if (ncols === 1) {
      const w0 = writers[0]!
      for (let i = 0; i < len; i++) chunk[i] = cellAt(w0, start + i, esc)
    } else if (ncols === 2) {
      const w0 = writers[0]!
      const w1 = writers[1]!
      for (let i = 0; i < len; i++) {
        const r = start + i
        chunk[i] = `${cellAt(w0, r, esc)},${cellAt(w1, r, esc)}`
      }
    } else if (ncols === 3) {
      const w0 = writers[0]!
      const w1 = writers[1]!
      const w2 = writers[2]!
      for (let i = 0; i < len; i++) {
        const r = start + i
        chunk[i] = `${cellAt(w0, r, esc)},${cellAt(w1, r, esc)},${cellAt(w2, r, esc)}`
      }
    } else if (ncols === 4) {
      const [w0, w1, w2, w3] = writers as [ColWriter, ColWriter, ColWriter, ColWriter]
      for (let i = 0; i < len; i++) {
        const r = start + i
        chunk[i] = `${cellAt(w0, r, esc)},${cellAt(w1, r, esc)},${cellAt(w2, r, esc)},${cellAt(w3, r, esc)}`
      }
    } else if (ncols === 5) {
      const [w0, w1, w2, w3, w4] = writers as [ColWriter, ColWriter, ColWriter, ColWriter, ColWriter]
      for (let i = 0; i < len; i++) {
        const r = start + i
        chunk[i] =
          `${cellAt(w0, r, esc)},${cellAt(w1, r, esc)},${cellAt(w2, r, esc)},${cellAt(w3, r, esc)},${cellAt(w4, r, esc)}`
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
        chunk[i] =
          `${cellAt(w0, r, esc)},${cellAt(w1, r, esc)},${cellAt(w2, r, esc)},${cellAt(w3, r, esc)},${cellAt(w4, r, esc)},${cellAt(w5, r, esc)}`
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
        chunk[i] =
          `${cellAt(w0, r, esc)},${cellAt(w1, r, esc)},${cellAt(w2, r, esc)},${cellAt(w3, r, esc)},${cellAt(w4, r, esc)},${cellAt(w5, r, esc)},${cellAt(w6, r, esc)}`
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
        chunk[i] =
          `${cellAt(w0, r, esc)},${cellAt(w1, r, esc)},${cellAt(w2, r, esc)},${cellAt(w3, r, esc)},${cellAt(w4, r, esc)},${cellAt(w5, r, esc)},${cellAt(w6, r, esc)},${cellAt(w7, r, esc)}`
      }
    } else {
      for (let i = 0; i < len; i++) {
        const r = start + i
        let row = cellAt(writers[0]!, r, esc)
        for (let c = 1; c < ncols; c++) row += `,${cellAt(writers[c]!, r, esc)}`
        chunk[i] = row
      }
    }
    yield chunk.length === len ? chunk.join('\n') : chunk.slice(0, len).join('\n')
  }
}

/** Serialize a table to CSV text. For files prefer `writeCsvText(path)`, which streams the chunks. */
export function tableToCsv(table: TableView, options: CsvWriteOptions = {}): string {
  const parts: string[] = []
  for (const chunk of tableToCsvChunks(table, options)) parts.push(chunk)
  return parts.join('\n')
}

async function writeNodeFile(path: string, data: string | Uint8Array): Promise<void> {
  const fs = await import('node:fs/promises')
  await fs.writeFile(path, data)
}

/**
 * Write CSV to a Node path (streamed chunk by chunk, honouring back-pressure) or return the CSV string
 * when the path is omitted. With a path the function resolves to '' — the whole text is never built.
 */
export async function writeCsvText(table: TableView, path?: string, options: CsvWriteOptions = {}): Promise<string> {
  if (!path) return tableToCsv(table, options)
  if (await tryWriteCsvNative(table, path, options)) return ''
  const fs = await import('node:fs')
  const out = fs.createWriteStream(path)
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
      await write(first ? chunk : '\n' + chunk)
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

/** Minimal parquet-like JSON payload (same as `@columna/wasm` writeParquetLike).
 *  This is NOT Apache Parquet — use {@link writeParquetLikeBytes} / `writeParquetLike`. */
export function tableToParquetLike(table: TableView): Uint8Array {
  const payload = {
    format: 'columna-parquet-like-v1',
    numRows: table.numRows,
    schema: table.schema,
    columns: table.columns.map((c) => {
      const values: Array<number | string | boolean | null> = []
      for (let i = 0; i < table.numRows; i++) {
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
export async function writeParquetLikeBytes(table: TableView, path?: string): Promise<Uint8Array> {
  const bytes = tableToParquetLike(table)
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

/**
 * Write Apache Parquet (via hyparquet-writer). Round-trips with {@link DataFrame.readParquet}.
 * Default codec is SNAPPY.
 */
export async function writeParquetBytes(table: TableView, path?: string): Promise<Uint8Array> {
  const { parquetWriteBuffer } = await import('hyparquet-writer')
  const columnData = tableToParquetColumnData(table)
  const ab = parquetWriteBuffer({ columnData, codec: 'SNAPPY' })
  const bytes = new Uint8Array(ab)
  if (path) await writeNodeFile(path, bytes)
  return bytes
}
