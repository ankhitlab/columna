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

/** Cell strings for rows [start, end) of one column — only this window is materialised. */
function cellsFor(w: ColWriter, start: number, end: number, esc: (s: string) => string): string[] {
  const len = end - start
  const out = new Array<string>(len)
  const bm = w.bm
  if (w.kind === 'utf8') {
    const data = w.data
    const cache = w.cache
    for (let i = start; i < end; i++) {
      if (bm && !isValid(bm, i)) {
        out[i - start] = ''
        continue
      }
      const s = data[i] ?? ''
      let e = cache.get(s)
      if (e === undefined) {
        e = esc(s)
        if (cache.size < 65_536) cache.set(s, e)
      }
      out[i - start] = e
    }
    return out
  }
  if (w.kind === 'cat') {
    const codes = w.codes
    const escaped = w.escaped
    for (let i = start; i < end; i++) out[i - start] = bm && !isValid(bm, i) ? '' : (escaped[codes[i]!] ?? '')
    return out
  }
  if (w.kind === 'bool') {
    const data = w.data
    for (let i = start; i < end; i++) out[i - start] = bm && !isValid(bm, i) ? '' : data[i] ? 'true' : 'false'
    return out
  }
  const data = w.data
  const dense = w.dense
  const denseMin = w.denseMin
  for (let i = start; i < end; i++) {
    if (bm && !isValid(bm, i)) {
      out[i - start] = ''
      continue
    }
    const v = data[i]!
    if (dense && Number.isInteger(v)) {
      const o = v - denseMin
      if (o >= 0 && o < dense.length) {
        out[i - start] = dense[o]!
        continue
      }
    }
    if (v >= 0 && v < SMALL_INT_STR.length && Number.isInteger(v)) out[i - start] = SMALL_INT_STR[v]!
    else out[i - start] = String(v)
  }
  return out
}

/** Rows per emitted chunk: bounds the cell strings alive at once to ncols × CSV_CHUNK_ROWS. */
export const CSV_CHUNK_ROWS = 16_384

/**
 * Serialize a table to CSV as a sequence of chunks (header first, then blocks of rows). Only one block of
 * cell strings exists at a time, so `writeCsv(path)` streams with memory bounded by the chunk, not the table.
 */
export function* tableToCsvChunks(table: TableView, options: CsvWriteOptions = {}): Generator<string, void, undefined> {
  const n = table.numRows
  const cols = table.columns
  const ncols = cols.length
  const esc = options.escapeFormulas ? escapeCsvSafe : escapeCsv
  yield cols.map((c) => esc(c.field.name)).join(',')
  if (n === 0) return
  const writers = cols.map((c) => prepareWriter(c, esc))
  for (let start = 0; start < n; start += CSV_CHUNK_ROWS) {
    const end = Math.min(n, start + CSV_CHUNK_ROWS)
    const len = end - start
    const cells = writers.map((w) => cellsFor(w, start, end, esc))
    const chunk = new Array<string>(len)
    if (ncols === 1) {
      const a = cells[0]!
      for (let i = 0; i < len; i++) chunk[i] = a[i]!
    } else if (ncols === 2) {
      const a = cells[0]!
      const b = cells[1]!
      for (let i = 0; i < len; i++) chunk[i] = `${a[i]!},${b[i]!}`
    } else if (ncols === 3) {
      const a = cells[0]!
      const b = cells[1]!
      const c = cells[2]!
      for (let i = 0; i < len; i++) chunk[i] = `${a[i]!},${b[i]!},${c[i]!}`
    } else {
      for (let i = 0; i < len; i++) {
        let row = cells[0]![i]!
        for (let c = 1; c < ncols; c++) row += `,${cells[c]![i]!}`
        chunk[i] = row
      }
    }
    yield chunk.join('\n')
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
  const fs = await import('node:fs')
  const out = fs.createWriteStream(path)
  const write = (s: string) =>
    new Promise<void>((resolve, reject) => {
      if (out.write(s)) resolve()
      else out.once('drain', resolve)
      out.once('error', reject)
    })
  try {
    let first = true
    for (const chunk of tableToCsvChunks(table, options)) {
      await write(first ? chunk : '\n' + chunk)
      first = false
    }
    await new Promise<void>((resolve, reject) => out.end((err?: Error | null) => (err ? reject(err) : resolve())))
  } catch (err) {
    out.destroy()
    throw err
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

/**
 * @deprecated This never wrote Apache Parquet — it wrote JSON (`columna-parquet-like-v1`).
 * Use {@link writeParquetLikeBytes} explicitly. A real Parquet writer is not implemented yet.
 */
export async function writeParquetBytes(table: TableView, path?: string): Promise<Uint8Array> {
  throw new Error(
    'DataFrame.writeParquet() does not write Apache Parquet files. ' +
      'It previously wrote a JSON "columna-parquet-like-v1" payload which cannot be read by ' +
      'DataFrame.readParquet() (hyparquet). Use writeParquetLike() / writeParquetLikeBytes() for that format, ' +
      'or write CSV / JSON until a real Parquet writer ships.',
  )
}
