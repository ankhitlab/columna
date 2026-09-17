/**
 * Optional native / worker-pool CSV ingestion for large unquoted files.
 * Falls back quietly when the addon or workers are unavailable.
 */
import { tableFromColumns, type Column, type DType, type TableView } from '@columna/arrow'
import { cpus } from 'node:os'
import { Worker } from 'node:worker_threads'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import type { ReadCsvOptions } from './types.js'
import { parseCsvToTable } from './csv-columnar.js'

export function canAccelerateCsv(options: ReadCsvOptions = {}): boolean {
  const delim = options.separator ?? options.delimiter ?? ','
  return (
    delim.length === 1 &&
    !options.skipInitialSpace &&
    !options.thousands &&
    (options.decimal ?? '.') === '.' &&
    options.nullValues === undefined &&
    options.trueValues === undefined &&
    options.falseValues === undefined &&
    options.comment === undefined &&
    options.usecols === undefined &&
    options.nRows === undefined &&
    options.skipRows === undefined &&
    (options.quoteChar === undefined || options.quoteChar === '"')
  )
}

type NativeCol = {
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

type NativeTable = { numRows: number; columns: NativeCol[] }

function nativeColToColumn(c: NativeCol): Column {
  const dtype = c.dtype as DType
  const nullBitmap = c.nullBitmap
  const nullable = Boolean(nullBitmap)
  if (dtype === 'f64' || dtype === 'datetime') {
    return { field: { name: c.name, dtype, nullable }, data: c.f64Data ?? new Float64Array(0), nullBitmap }
  }
  if (dtype === 'i32') {
    return { field: { name: c.name, dtype, nullable }, data: c.i32Data ?? new Int32Array(0), nullBitmap }
  }
  if (dtype === 'bool') {
    return { field: { name: c.name, dtype, nullable }, data: c.boolData ?? new Uint8Array(0), nullBitmap }
  }
  if (dtype === 'category') {
    return {
      field: { name: c.name, dtype, nullable },
      data: c.catCodes ?? new Uint32Array(0),
      nullBitmap,
      dictionary: c.dictionary,
    }
  }
  return { field: { name: c.name, dtype: 'utf8', nullable }, data: c.utf8Data ?? [], nullBitmap }
}

/** Try `@columna/native` Rayon CSV parse. Returns null when unavailable / unsupported. */
export async function tryParseCsvNative(path: string, options: ReadCsvOptions = {}): Promise<TableView | null> {
  if (!canAccelerateCsv(options)) return null
  try {
    const id = '@columna/' + 'native'
    const mod = (await import(/* webpackIgnore: true */ /* @vite-ignore */ id)) as {
      isNativeLoaded?: boolean
      parseCsvUnquoted?: (bytes: Buffer, delimiter: number, hasHeader: boolean) => NativeTable
    }
    if (!mod.isNativeLoaded || typeof mod.parseCsvUnquoted !== 'function') return null
    const fs = await import('node:fs/promises')
    const bytes = await fs.readFile(path)
    // Quick reject: quoted files belong on the JS path.
    if (bytes.includes(0x22)) return null
    const delim = (options.separator ?? options.delimiter ?? ',').charCodeAt(0)
    const hasHeader =
      options.hasHeader !== undefined
        ? options.hasHeader
        : options.header === undefined
          ? true
          : options.header !== false
    if (typeof options.header === 'number' && options.header !== 0) return null
    if (options.names) return null
    if (options.dtypes) return null
    const parsed = mod.parseCsvUnquoted(bytes, delim, hasHeader)
    return tableFromColumns(parsed.columns.map(nativeColToColumn))
  } catch {
    return null
  }
}

function csvWorkerUrl(): string | null {
  try {
    for (const name of ['./csv-worker.js', './csv-worker.ts']) {
      const p = fileURLToPath(new URL(name, import.meta.url))
      if (existsSync(p)) return p
    }
  } catch {
    // ignore
  }
  return null
}

/** Split UTF-8 text into `parts` record-aligned chunks (first keeps the header). */
export function splitCsvTextChunks(text: string, parts: number): string[] {
  if (parts <= 1 || text.length < 1 << 20) return [text]
  const n = Math.min(parts, 8)
  const targets: number[] = []
  for (let i = 1; i < n; i++) targets.push(Math.floor((text.length * i) / n))
  const cuts: number[] = [0]
  for (const t of targets) {
    let i = t
    while (i < text.length && text.charCodeAt(i) !== 10) i++
    if (i < text.length) i++ // include the newline with the previous chunk
    if (i > cuts[cuts.length - 1]! && i < text.length) cuts.push(i)
  }
  cuts.push(text.length)
  const out: string[] = []
  for (let c = 0; c < cuts.length - 1; c++) {
    const slice = text.slice(cuts[c]!, cuts[c + 1]!)
    if (slice.length) out.push(slice)
  }
  return out.length ? out : [text]
}

type WorkerCol = {
  name: string
  dtype: DType
  nullable: boolean
  data: Float64Array | Int32Array | Uint32Array | Uint8Array | string[]
  nullBitmap?: Uint8Array
  dictionary?: string[]
}

function serializeTable(table: TableView): WorkerCol[] {
  return table.columns.map((c) => ({
    name: c.field.name,
    dtype: c.field.dtype,
    nullable: Boolean(c.nullBitmap),
    data: c.data as WorkerCol['data'],
    nullBitmap: c.nullBitmap,
    dictionary: c.dictionary,
  }))
}

function concatWorkerCols(parts: WorkerCol[][]): TableView {
  if (parts.length === 0) return tableFromColumns([])
  const ncols = parts[0]!.length
  // Promote i32→f64 when any chunk widened (late fractional value).
  for (let ci = 0; ci < ncols; ci++) {
    const kinds = new Set(parts.map((p) => p[ci]!.dtype))
    if (kinds.has('f64') && kinds.has('i32')) {
      for (const p of parts) {
        const c = p[ci]!
        if (c.dtype !== 'i32') continue
        const src = c.data as Int32Array
        const f = new Float64Array(src.length)
        for (let i = 0; i < src.length; i++) f[i] = src[i]!
        c.data = f
        c.dtype = 'f64'
      }
    }
    if (kinds.has('utf8') && kinds.has('category')) {
      for (const p of parts) {
        const c = p[ci]!
        if (c.dtype !== 'category') continue
        const codes = c.data as Uint32Array
        const dict = c.dictionary ?? []
        const data = Array.from({ length: codes.length }, (_, i) => dict[codes[i]!] ?? '')
        c.data = data
        c.dtype = 'utf8'
        c.dictionary = undefined
      }
    }
  }
  const cols: Column[] = []
  for (let ci = 0; ci < ncols; ci++) {
    const head = parts[0]![ci]!
    const total = parts.reduce((s, p) => s + (p[ci]!.data as { length: number }).length, 0)
    const anyNull = parts.some((p) => p[ci]!.nullBitmap)
    const dtype = head.dtype

    if (dtype === 'utf8') {
      const data = new Array<string>(total)
      let o = 0
      let nullBitmap: Uint8Array | undefined
      if (anyNull) nullBitmap = new Uint8Array(Math.ceil(total / 8) || 1)
      for (const p of parts) {
        const c = p[ci]!
        const src = c.data as string[]
        for (let i = 0; i < src.length; i++) {
          data[o] = src[i]!
          if (nullBitmap && c.nullBitmap && (c.nullBitmap[i >> 3]! >> (i & 7)) & 1) {
            nullBitmap[o >> 3]! |= 1 << (o & 7)
          } else if (nullBitmap && !c.nullBitmap) {
            nullBitmap[o >> 3]! |= 1 << (o & 7)
          }
          o++
        }
      }
      cols.push({ field: { name: head.name, dtype, nullable: anyNull }, data, nullBitmap })
      continue
    }

    if (dtype === 'category') {
      const dict: string[] = []
      const map = new Map<string, number>()
      const code = (s: string) => {
        let c = map.get(s)
        if (c === undefined) {
          c = dict.length
          dict.push(s)
          map.set(s, c)
        }
        return c
      }
      const data = new Uint32Array(total)
      let o = 0
      let nullBitmap: Uint8Array | undefined
      if (anyNull) nullBitmap = new Uint8Array(Math.ceil(total / 8) || 1)
      for (const p of parts) {
        const c = p[ci]!
        const src = c.data as Uint32Array
        const local = c.dictionary ?? []
        for (let i = 0; i < src.length; i++) {
          const valid = !c.nullBitmap || ((c.nullBitmap[i >> 3]! >> (i & 7)) & 1) !== 0
          if (valid) {
            data[o] = code(local[src[i]!] ?? '')
            if (nullBitmap) nullBitmap[o >> 3]! |= 1 << (o & 7)
          }
          o++
        }
      }
      cols.push({
        field: { name: head.name, dtype, nullable: anyNull },
        data,
        nullBitmap,
        dictionary: dict,
      })
      continue
    }

    const Ctor =
      dtype === 'i32' ? Int32Array : dtype === 'bool' ? Uint8Array : dtype === 'f32' ? Float32Array : Float64Array
    const data = new Ctor(total) as Float64Array | Int32Array | Uint8Array | Float32Array
    let o = 0
    let nullBitmap: Uint8Array | undefined
    if (anyNull) nullBitmap = new Uint8Array(Math.ceil(total / 8) || 1)
    for (const p of parts) {
      const c = p[ci]!
      const src = c.data as ArrayLike<number>
      for (let i = 0; i < src.length; i++) {
        ;(data as { [i: number]: number })[o] = src[i]!
        if (nullBitmap) {
          const valid = !c.nullBitmap || ((c.nullBitmap[i >> 3]! >> (i & 7)) & 1) !== 0
          if (valid) nullBitmap[o >> 3]! |= 1 << (o & 7)
        }
        o++
      }
    }
    cols.push({ field: { name: head.name, dtype, nullable: anyNull }, data, nullBitmap })
  }
  return tableFromColumns(cols)
}

/**
 * Multi-threaded CSV parse via worker_threads (2–4 chunks). Used when native CSV is unavailable
 * and the file is large enough to amortize worker startup.
 */
export async function tryParseCsvParallel(path: string, options: ReadCsvOptions = {}): Promise<TableView | null> {
  if (!canAccelerateCsv(options)) return null
  if (typeof options.header === 'number' && options.header !== 0) return null
  if (options.names || options.dtypes) return null

  const url = csvWorkerUrl()
  if (!url) return null

  const fs = await import('node:fs/promises')
  const st = await fs.stat(path)
  // Workers cost ~50–100 ms to spawn; only worth it on larger files.
  if (st.size < 8 << 20) return null

  const threads = Math.min(Math.max(2, cpus().length), 4)
  const text = await fs.readFile(path, (options.encoding as BufferEncoding | undefined) ?? 'utf8')
  if (text.includes('"')) {
    // Quoted fields may span newlines — refuse parallel split.
    return null
  }
  const chunks = splitCsvTextChunks(text.replace(/^\uFEFF/, ''), threads)
  if (chunks.length < 2) return parseCsvToTable(text, options)

  const hasHeader =
    options.hasHeader !== undefined
      ? options.hasHeader
      : options.header === undefined
        ? true
        : options.header !== false

  // Parse header chunk on main thread to get column names for subsequent chunks.
  const first = parseCsvToTable(chunks[0]!, { ...options, hasHeader })
  const headers = first.columns.map((c) => c.field.name)
  const restJobs = chunks.slice(1).map(
    (chunk) =>
      new Promise<WorkerCol[]>((resolve, reject) => {
        const worker = new Worker(url, {
          execArgv: url.endsWith('.ts') ? ['--import', 'tsx'] : undefined,
        })
        const payload = chunk.startsWith('\n') || chunk.startsWith('\r') ? chunk : chunk
        worker.once('message', (msg: { ok: boolean; columns?: WorkerCol[]; error?: string }) => {
          void worker.terminate()
          if (msg.ok && msg.columns) resolve(msg.columns)
          else reject(new Error(msg.error ?? 'csv worker failed'))
        })
        worker.once('error', (err) => {
          void worker.terminate()
          reject(err)
        })
        worker.postMessage({
          text: payload,
          options: { ...options, hasHeader: false, names: headers, header: false },
        })
      }),
  )

  try {
    const rest = await Promise.all(restJobs)
    return concatWorkerCols([serializeTable(first), ...rest])
  } catch {
    // Worker failure → let caller use the single-thread path.
    return null
  }
}
