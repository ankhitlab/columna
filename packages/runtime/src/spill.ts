import {
  allocateData,
  getValue,
  isValid,
  setValid,
  setValue,
  tableFromColumns,
  type Column,
  type DType,
  type TableView,
  type TypedData,
} from '@columna/arrow'
import { recordSpilledBytes, resolveSpillDir } from './memory.js'

/**
 * Node's `require`, obtained lazily and only on Node. A static `import { createRequire } from 'node:module'`
 * evaluated at module load would break every browser bundle of the library, although spilling can never
 * run there. `ensureSpillSupport()` is awaited by the runtime before executing under a memory budget.
 */
let nodeRequire: NodeJS.Require | null = null

/**
 * Filename / URL passed to `createRequire`.
 * tsup's CJS build leaves `import.meta` empty (`{}`), so `import.meta.url` is undefined —
 * fall back to `process.execPath` (any absolute path works for `node:` builtins).
 */
function createRequireFilename(): string {
  try {
    const url = import.meta.url
    if (typeof url === 'string' && url.length > 0) return url
  } catch {
    /* CJS / non-module */
  }
  return process.execPath
}

function bindNodeRequire(createRequire: typeof import('node:module').createRequire): NodeJS.Require {
  return createRequire(createRequireFilename())
}

export async function ensureSpillSupport(): Promise<boolean> {
  if (nodeRequire) return true
  if (typeof process === 'undefined' || !process.versions?.node) return false
  const proc = process as NodeJS.Process & { getBuiltinModule?: (id: string) => unknown }
  if (typeof proc.getBuiltinModule === 'function') {
    const mod = proc.getBuiltinModule('node:module') as typeof import('node:module')
    nodeRequire = bindNodeRequire(mod.createRequire)
    return true
  }
  const id = 'node:module' // opaque to bundlers
  const mod = (await import(/* @vite-ignore */ id)) as typeof import('node:module')
  nodeRequire = bindNodeRequire(mod.createRequire)
  return true
}

function require(id: string): unknown {
  if (!nodeRequire && typeof process !== 'undefined' && process.versions?.node) {
    // synchronous path on Node ≥ 20.16 / 22.3; older Node needs the awaited ensureSpillSupport()
    const proc = process as NodeJS.Process & { getBuiltinModule?: (id: string) => unknown }
    if (typeof proc.getBuiltinModule === 'function') {
      nodeRequire = bindNodeRequire(
        (proc.getBuiltinModule('node:module') as typeof import('node:module')).createRequire,
      )
    }
  }
  if (!nodeRequire) throw new Error('spill: Node support not initialised (the runtime awaits ensureSpillSupport() before running under a memory budget)')
  return nodeRequire(id)
}

const MAGIC = 0x434f4c53 // 'COLS'
const VERSION = 1

type Fs = typeof import('node:fs')
type Path = typeof import('node:path')

function nodeFs(): Fs {
  return require('node:fs') as Fs
}

function nodePath(): Path {
  return require('node:path') as Path
}

let spillSeq = 0
/** Per-process private spill subdirectory (under configured or default parent). */
let spillSessionDir: string | null = null
let spillSessionParent: string | null = null

function chmodPrivate(fs: Fs, target: string, mode: number): void {
  try {
    fs.chmodSync(target, mode)
  } catch {
    /* Windows / unsupported chmod — best-effort */
  }
}

function ensureSpillDir(): string {
  const fs = nodeFs()
  const path = nodePath()
  const os = require('node:os') as typeof import('node:os')

  const configured = resolveSpillDir()
  const parent = configured || path.join(os.tmpdir(), 'columna-spill')

  fs.mkdirSync(parent, { recursive: true })

  if (spillSessionDir && spillSessionParent === parent && fs.existsSync(spillSessionDir)) {
    return spillSessionDir
  }

  const pid = typeof process !== 'undefined' ? process.pid : 0

  const sessionDir = fs.mkdtempSync(path.join(parent, `columna-p${pid}-`))

  chmodPrivate(fs, sessionDir, 0o700)

  spillSessionDir = sessionDir
  spillSessionParent = parent

  return sessionDir
}

export function spillTempPath(prefix = 'run'): string {
  const path = nodePath()
  const dir = ensureSpillDir()
  spillSeq += 1
  const pid = typeof process !== 'undefined' ? process.pid : 0
  return path.join(dir, `${prefix}-${pid}-${Date.now()}-${spillSeq}.cspill`)
}

/**
 * Write a table as a simple columnar blob (header + typed arrays / utf8 lengths).
 * Returns the absolute path. Caller owns cleanup via `spillUnlink`.
 */
export function spillWrite(table: TableView, path?: string): string {
  const fs = nodeFs()
  const outPath = path ?? spillTempPath('tbl')
  const parts = encodeTable(table)

  const total = parts.reduce((a, p) => a + p.byteLength, 0)
  const buf = new Uint8Array(total)
  let off = 0
  for (const p of parts) {
    buf.set(p, off)
    off += p.byteLength
  }
  // Exclusive create + private mode. Only unlink a file this invocation created.
  let fd: number | undefined

  try {
    fd = fs.openSync(outPath, 'wx', 0o600)

    try {
      fs.writeFileSync(fd, buf)
      try {
        fs.fchmodSync(fd, 0o600)
      } catch {
        /* Windows / unsupported fchmod — best-effort */
      }
    } finally {
      fs.closeSync(fd)
    }
  } catch (err) {
    // Only clean up a path that THIS invocation successfully created.
    if (fd !== undefined) {
      try {
        fs.unlinkSync(outPath)
      } catch {
        // Ignore cleanup failure. Preserve original exception.
      }
    }

    throw err
  }
  recordSpilledBytes(total)
  return outPath
}

/** Header + schema + columns as separate parts (the async writer streams them; the sync writer concatenates). */
function encodeTable(table: TableView): Uint8Array[] {
  const parts: Uint8Array[] = []
  const schemaJson = JSON.stringify(
    table.schema.map((f) => ({ name: f.name, dtype: f.dtype, nullable: f.nullable })),
  )
  const schemaBytes = new TextEncoder().encode(schemaJson)
  const header = new ArrayBuffer(20)
  const hv = new DataView(header)
  hv.setUint32(0, MAGIC, true)
  hv.setUint32(4, VERSION, true)
  hv.setUint32(8, table.numRows, true)
  hv.setUint32(12, table.columns.length, true)
  hv.setUint32(16, schemaBytes.byteLength, true)
  parts.push(new Uint8Array(header), schemaBytes)
  for (const col of table.columns) parts.push(...encodeColumn(col, table.numRows))
  return parts
}

/**
 * Non-blocking spill write: `fs/promises`, exclusive create + private mode, parts written one after another
 * (no contiguous copy of the whole table). Used by the async spill operators the runtime takes when a memory
 * budget is active; the event loop keeps turning while the disk works.
 */
export async function spillWriteAsync(table: TableView, path?: string): Promise<string> {
  const fsp = require('node:fs/promises') as typeof import('node:fs/promises')
  const outPath = path ?? spillTempPath('tbl')
  const parts = encodeTable(table)
  let total = 0
  const handle = await fsp.open(outPath, 'wx', 0o600)
  try {
    for (const p of parts) {
      // a part may sit on a SharedArrayBuffer (worker paths) — write a plain view of it
      const view = p.buffer instanceof ArrayBuffer ? p : new Uint8Array(p)
      let off = 0
      while (off < view.byteLength) {
        const { bytesWritten } = await handle.write(view, off, view.byteLength - off)
        off += bytesWritten
      }
      total += p.byteLength
    }
    try {
      await handle.chmod(0o600)
    } catch {
      /* Windows / unsupported */
    }
  } catch (err) {
    await handle.close().catch(() => undefined)
    await fsp.unlink(outPath).catch(() => undefined)
    throw err
  }
  await handle.close()
  recordSpilledBytes(total)
  return outPath
}

export async function spillReadAsync(path: string): Promise<TableView> {
  const fsp = require('node:fs/promises') as typeof import('node:fs/promises')
  const file = await fsp.readFile(path)
  return decodeTable(file instanceof Uint8Array ? file : new Uint8Array(file), path)
}

export async function spillUnlinkAsync(path: string): Promise<void> {
  const fsp = require('node:fs/promises') as typeof import('node:fs/promises')
  await fsp.unlink(path).catch(() => undefined)
}

export async function spillUnlinkManyAsync(paths: string[]): Promise<void> {
  await Promise.all(paths.map((p) => spillUnlinkAsync(p)))
}

export function spillRead(path: string): TableView {
  const fs = nodeFs()
  const file = fs.readFileSync(path)
  return decodeTable(file instanceof Uint8Array ? file : new Uint8Array(file), path)
}

function decodeTable(buf: Uint8Array, path: string): TableView {
  let off = 0
  const hv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  if (hv.getUint32(off, true) !== MAGIC) throw new Error(`spillRead: bad magic at ${path}`)
  off += 4
  const version = hv.getUint32(off, true)
  off += 4
  if (version !== VERSION) throw new Error(`spillRead: unsupported version ${version}`)
  const numRows = hv.getUint32(off, true)
  off += 4
  const numCols = hv.getUint32(off, true)
  off += 4
  const schemaLen = hv.getUint32(off, true)
  off += 4
  const schemaJson = new TextDecoder().decode(buf.subarray(off, off + schemaLen))
  off += schemaLen
  const schema = JSON.parse(schemaJson) as Array<{ name: string; dtype: DType; nullable: boolean }>
  if (schema.length !== numCols) throw new Error('spillRead: schema/column count mismatch')

  const columns: Column[] = []
  for (let c = 0; c < numCols; c++) {
    const field = schema[c]!
    const decoded = decodeColumn(buf, off, field.dtype, field.name, field.nullable, numRows)
    off = decoded.next
    columns.push(decoded.column)
  }
  return tableFromColumns(columns)
}

export function spillUnlink(path: string): void {
  try {
    nodeFs().unlinkSync(path)
  } catch {
    /* ignore missing */
  }
}

export function spillUnlinkMany(paths: string[]): void {
  for (const p of paths) spillUnlink(p)
}

function encodeColumn(col: Column, numRows: number): Uint8Array[] {
  const parts: Uint8Array[] = []
  let flags = 0
  if (col.nullBitmap) flags |= 1
  if (col.dictionary) flags |= 2
  const meta = new ArrayBuffer(4)
  new DataView(meta).setUint32(0, flags, true)
  parts.push(new Uint8Array(meta))

  if (col.nullBitmap) {
    const len = new ArrayBuffer(4)
    new DataView(len).setUint32(0, col.nullBitmap.byteLength, true)
    parts.push(new Uint8Array(len), col.nullBitmap)
  }

  if (col.dictionary) {
    const dictJson = JSON.stringify(col.dictionary)
    const dictBytes = new TextEncoder().encode(dictJson)
    const len = new ArrayBuffer(4)
    new DataView(len).setUint32(0, dictBytes.byteLength, true)
    parts.push(new Uint8Array(len), dictBytes)
  }

  const dtype = col.field.dtype
  if (dtype === 'utf8') {
    const strings = col.data as string[]
    const encoded = strings.map((s) => new TextEncoder().encode(s ?? ''))
    const lens = new Uint32Array(numRows)
    let total = 0
    for (let i = 0; i < numRows; i++) {
      lens[i] = encoded[i]!.byteLength
      total += encoded[i]!.byteLength
    }
    parts.push(new Uint8Array(lens.buffer, lens.byteOffset, lens.byteLength))
    const payload = new Uint8Array(total)
    let o = 0
    for (const e of encoded) {
      payload.set(e, o)
      o += e.byteLength
    }
    parts.push(payload)
  } else {
    const view = col.data as ArrayBufferView
    const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength)
    const len = new ArrayBuffer(4)
    new DataView(len).setUint32(0, bytes.byteLength, true)
    parts.push(new Uint8Array(len), bytes)
  }
  return parts
}

function decodeColumn(
  buf: Uint8Array,
  off: number,
  dtype: DType,
  name: string,
  nullable: boolean,
  numRows: number,
): { column: Column; next: number } {
  const hv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  const flags = hv.getUint32(off, true)
  off += 4
  let nullBitmap: Uint8Array | undefined
  if (flags & 1) {
    const nlen = hv.getUint32(off, true)
    off += 4
    nullBitmap = buf.subarray(off, off + nlen).slice()
    off += nlen
  }
  let dictionary: string[] | undefined
  if (flags & 2) {
    const dlen = hv.getUint32(off, true)
    off += 4
    dictionary = JSON.parse(new TextDecoder().decode(buf.subarray(off, off + dlen))) as string[]
    off += dlen
  }

  let data: TypedData
  if (dtype === 'utf8') {
    const lens = new Uint32Array(numRows)
    for (let i = 0; i < numRows; i++) {
      lens[i] = hv.getUint32(off, true)
      off += 4
    }
    const strings: string[] = new Array(numRows)
    const dec = new TextDecoder()
    for (let i = 0; i < numRows; i++) {
      const L = lens[i]!
      strings[i] = dec.decode(buf.subarray(off, off + L))
      off += L
    }
    data = strings
  } else {
    const blen = hv.getUint32(off, true)
    off += 4
    const raw = buf.subarray(off, off + blen)
    off += blen
    data = allocateData(dtype, numRows)
    const dest = data as ArrayBufferView
    new Uint8Array(dest.buffer, dest.byteOffset, dest.byteLength).set(
      raw.subarray(0, Math.min(raw.byteLength, dest.byteLength)),
    )
  }

  return {
    next: off,
    column: {
      field: { name, dtype, nullable },
      data,
      nullBitmap,
      dictionary,
    },
  }
}

/** Concatenate tables with identical schemas (row-wise). */
export function concatTables(tables: TableView[]): TableView {
  if (tables.length === 0) return tableFromColumns([])
  if (tables.length === 1) return tables[0]!
  const schema = tables[0]!.schema
  const total = tables.reduce((a, t) => a + t.numRows, 0)
  const columns: Column[] = []

  for (let ci = 0; ci < schema.length; ci++) {
    const field = schema[ci]!
    const data = allocateData(field.dtype, total)
    const nullBitmap = new Uint8Array(Math.ceil(total / 8) || 1)
    let anyNull = false
    let dict = tables[0]!.columns[ci]!.dictionary
    let row = 0

    for (const t of tables) {
      const src = t.columns[ci]!
      if (src.field.dtype !== field.dtype) {
        throw new Error(`concatTables: dtype mismatch on ${field.name}`)
      }
      if (src.dictionary) {
        if (!dict) dict = src.dictionary
        else if (JSON.stringify(src.dictionary) !== JSON.stringify(dict)) {
          throw new Error(`concatTables: dictionary mismatch on column ${field.name}`)
        }
      }
      for (let i = 0; i < t.numRows; i++) {
        if (!isValid(src.nullBitmap, i)) {
          anyNull = true
        } else {
          setValid(nullBitmap, row, true)
          setValue(data, row, getValue(src.data, i) as never, field.dtype)
        }
        row++
      }
    }

    columns.push({
      field: { ...field },
      data,
      nullBitmap: anyNull ? nullBitmap : undefined,
      dictionary: dict ? [...dict] : undefined,
    })
  }

  return tableFromColumns(columns)
}
