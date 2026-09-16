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

function prepareWriter(col: Column): ColWriter {
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
      escaped: col.dictionary.map((s) => escapeCsv(s)),
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

/** Serialize a table to CSV text (column format + chunked row join). */
export function tableToCsv(table: TableView): string {
  const n = table.numRows
  const cols = table.columns
  const ncols = cols.length
  const header = cols.map((c) => escapeCsv(c.field.name)).join(',')
  if (n === 0) return header

  const writers = cols.map(prepareWriter)
  const parts: string[] = [header]
  const chunkRows = 16_384

  // Hot path: materialize dense/cat/utf8 caches then build rows with minimal dispatch.
  const cells = writers.map((w) => {
    if (w.kind === 'utf8') {
      const data = w.data
      const bm = w.bm
      const cache = w.cache
      const out = new Array<string>(n)
      for (let i = 0; i < n; i++) {
        if (bm && !isValid(bm, i)) {
          out[i] = ''
          continue
        }
        const s = data[i] ?? ''
        let esc = cache.get(s)
        if (esc === undefined) {
          esc = escapeCsv(s)
          cache.set(s, esc)
        }
        out[i] = esc
      }
      return out
    }
    if (w.kind === 'cat') {
      const codes = w.codes
      const escaped = w.escaped
      const bm = w.bm
      const out = new Array<string>(n)
      for (let i = 0; i < n; i++) {
        out[i] = bm && !isValid(bm, i) ? '' : (escaped[codes[i]!] ?? '')
      }
      return out
    }
    if (w.kind === 'bool') {
      const data = w.data
      const bm = w.bm
      const out = new Array<string>(n)
      for (let i = 0; i < n; i++) {
        out[i] = bm && !isValid(bm, i) ? '' : data[i] ? 'true' : 'false'
      }
      return out
    }
    // num
    const data = w.data
    const bm = w.bm
    const dense = w.dense
    const denseMin = w.denseMin
    const out = new Array<string>(n)
    for (let i = 0; i < n; i++) {
      if (bm && !isValid(bm, i)) {
        out[i] = ''
        continue
      }
      const v = data[i]!
      if (dense) {
        const o = v - denseMin
        if (o >= 0 && o < dense.length) {
          out[i] = dense[o]!
          continue
        }
      }
      if (v >= 0 && v < SMALL_INT_STR.length && Number.isInteger(v)) out[i] = SMALL_INT_STR[v]!
      else out[i] = String(v)
    }
    return out
  })

  for (let start = 0; start < n; start += chunkRows) {
    const end = Math.min(n, start + chunkRows)
    const len = end - start
    const chunk = new Array<string>(len)

    if (ncols === 9) {
      const a = cells[0]!
      const b = cells[1]!
      const c = cells[2]!
      const d = cells[3]!
      const e = cells[4]!
      const f = cells[5]!
      const g = cells[6]!
      const h = cells[7]!
      const i9 = cells[8]!
      for (let i = start; i < end; i++) {
        chunk[i - start] = `${a[i]!},${b[i]!},${c[i]!},${d[i]!},${e[i]!},${f[i]!},${g[i]!},${h[i]!},${i9[i]!}`
      }
    } else if (ncols === 1) {
      const a = cells[0]!
      for (let i = start; i < end; i++) chunk[i - start] = a[i]!
    } else if (ncols === 2) {
      const a = cells[0]!
      const b = cells[1]!
      for (let i = start; i < end; i++) chunk[i - start] = `${a[i]!},${b[i]!}`
    } else if (ncols === 3) {
      const a = cells[0]!
      const b = cells[1]!
      const c = cells[2]!
      for (let i = start; i < end; i++) chunk[i - start] = `${a[i]!},${b[i]!},${c[i]!}`
    } else {
      for (let i = start; i < end; i++) {
        let row = cells[0]![i]!
        for (let c = 1; c < ncols; c++) row += `,${cells[c]![i]!}`
        chunk[i - start] = row
      }
    }
    parts.push(chunk.join('\n'))
  }
  return parts.join('\n')
}

async function writeNodeFile(path: string, data: string | Uint8Array): Promise<void> {
  const fs = await import('node:fs/promises')
  await fs.writeFile(path, data)
}

/** Write CSV to a Node path, or return the CSV string when path is omitted. */
export async function writeCsvText(table: TableView, path?: string): Promise<string> {
  const text = tableToCsv(table)
  if (path) await writeNodeFile(path, text)
  return text
}

/** Minimal parquet-like JSON payload (same as `@columna/wasm` writeParquetLike). */
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

export async function writeParquetBytes(table: TableView, path?: string): Promise<Uint8Array> {
  const bytes = tableToParquetLike(table)
  if (path) await writeNodeFile(path, bytes)
  return bytes
}
