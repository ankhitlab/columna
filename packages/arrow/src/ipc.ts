/**
 * Apache Arrow IPC — the real thing, not `ArrowLike`.
 *
 * `toArrowIpc()` writes the Arrow IPC *streaming* format (or the *file* format, a.k.a. Feather v2) and
 * `fromArrowIpc()` reads both. The bytes are what `apache-arrow`'s `tableFromIPC`, DuckDB(-Wasm)'s
 * `insertArrowFromIPCStream`, pyarrow's `ipc.open_stream / open_file`, Polars' `read_ipc_stream / read_ipc`
 * and pandas' `read_feather` consume, and what they produce is what this reader takes. No dependency: the
 * FlatBuffers encoding of the Schema / RecordBatch / DictionaryBatch messages is small enough to write by hand
 * (the reference-style back-to-front builder below) and the interop suite pins it against `apache-arrow`.
 *
 * Type mapping (columna → Arrow):  f64 → Float64, f32 → Float32, i32 → Int32, u32 → UInt32, bool → Bool,
 * utf8 → Utf8, category → Dictionary<Int32, Utf8>, datetime → Timestamp(MILLISECOND).
 * Arrow → columna additionally accepts Int8/16/64 and UInt8/16/64 (64-bit → f64, exact below 2^53),
 * Float16 (→ f32), LargeUtf8 and Utf8View (the string layout Polars ≥ 1.0 and DuckDB emit), Timestamp in any
 * unit and Date32/Date64 (→ datetime, milliseconds), Null (→ all-null f64) and dictionary-encoded strings with
 * any index width. Nested, decimal, binary, interval
 * and compressed batches are refused with the field name in the error rather than decoded wrongly.
 */
import {
  PrecisionLossError,
  isSafeBigInt,
  isValid,
  setValid,
  tableFromColumns,
  type Column,
  type DType,
  type Field,
  type Int64Policy,
  type TableView,
} from './index.js'

// ───────────────────────────── FlatBuffers builder (back-to-front, as flatbuffers.Builder) ─────────────────────────────

class FbBuilder {
  private bb: Uint8Array
  private dv: DataView
  private space: number
  private minalign = 1
  private vtable: number[] = []
  private objectStart = 0
  private vectorNumElems = 0

  constructor(initial = 1024) {
    this.bb = new Uint8Array(initial)
    this.dv = new DataView(this.bb.buffer)
    this.space = initial
  }

  offset(): number {
    return this.bb.length - this.space
  }

  private grow(): void {
    const old = this.bb
    const next = new Uint8Array(old.length * 2)
    next.set(old, old.length)
    this.bb = next
    this.dv = new DataView(next.buffer)
    this.space += old.length
  }

  private pad(n: number): void {
    for (let i = 0; i < n; i++) this.bb[--this.space] = 0
  }

  /** Align so that after `additional` bytes are written the position is `size`-aligned. */
  prep(size: number, additional: number): void {
    if (size > this.minalign) this.minalign = size
    const alignSize = (~(this.bb.length - this.space + additional) + 1) & (size - 1)
    while (this.space < alignSize + size + additional) this.grow()
    this.pad(alignSize)
  }

  addI8(v: number): void {
    this.prep(1, 0)
    this.dv.setInt8(--this.space, v)
  }
  addU8(v: number): void {
    this.prep(1, 0)
    this.bb[--this.space] = v
  }
  addI16(v: number): void {
    this.prep(2, 0)
    this.space -= 2
    this.dv.setInt16(this.space, v, true)
  }
  addI32(v: number): void {
    this.prep(4, 0)
    this.space -= 4
    this.dv.setInt32(this.space, v, true)
  }
  addI64(v: number): void {
    this.prep(8, 0)
    this.space -= 8
    this.dv.setBigInt64(this.space, BigInt(v), true)
  }
  /** uoffset: relative distance from this field forward to the target object. */
  addOffset(off: number): void {
    this.prep(4, 0)
    this.space -= 4
    this.dv.setUint32(this.space, this.offset() - off, true)
  }

  startObject(numFields: number): void {
    this.vtable = new Array<number>(numFields).fill(0)
    this.objectStart = this.offset()
  }
  private slot(i: number): void {
    this.vtable[i] = this.offset()
  }
  addFieldI8(i: number, v: number, def: number): void {
    if (v !== def) {
      this.addI8(v)
      this.slot(i)
    }
  }
  addFieldI16(i: number, v: number, def: number): void {
    if (v !== def) {
      this.addI16(v)
      this.slot(i)
    }
  }
  addFieldI32(i: number, v: number, def: number): void {
    if (v !== def) {
      this.addI32(v)
      this.slot(i)
    }
  }
  addFieldI64(i: number, v: number, def: number): void {
    if (v !== def) {
      this.addI64(v)
      this.slot(i)
    }
  }
  addFieldOffset(i: number, off: number): void {
    if (off !== 0) {
      this.addOffset(off)
      this.slot(i)
    }
  }
  endObject(): number {
    this.addI32(0) // soffset placeholder
    const vtableloc = this.offset()
    let i = this.vtable.length - 1
    while (i >= 0 && this.vtable[i] === 0) i--
    const trimmed = i + 1
    for (; i >= 0; i--) this.addI16(this.vtable[i] !== 0 ? vtableloc - this.vtable[i]! : 0)
    this.addI16(vtableloc - this.objectStart)
    this.addI16((trimmed + 2) * 2)
    this.dv.setInt32(this.bb.length - vtableloc, this.offset() - vtableloc, true)
    this.vtable = []
    return vtableloc
  }

  startVector(elemSize: number, num: number, alignment: number): void {
    this.prep(4, elemSize * num)
    this.prep(alignment, elemSize * num)
    this.vectorNumElems = num
  }
  endVector(): number {
    this.addI32(this.vectorNumElems)
    return this.offset()
  }
  createString(s: string): number {
    const bytes = new TextEncoder().encode(s)
    this.addI8(0)
    this.startVector(1, bytes.length, 1)
    this.space -= bytes.length
    this.bb.set(bytes, this.space)
    return this.endVector()
  }
  createOffsetVector(offs: number[]): number {
    this.startVector(4, offs.length, 4)
    for (let i = offs.length - 1; i >= 0; i--) this.addOffset(offs[i]!)
    return this.endVector()
  }
  /** Vector of `{i64, i64}` structs (FieldNode, Buffer), written in reverse so memory order is a, b. */
  createI64PairVector(pairs: Array<[number, number]>): number {
    this.startVector(16, pairs.length, 8)
    for (let i = pairs.length - 1; i >= 0; i--) {
      this.addI64(pairs[i]![1])
      this.addI64(pairs[i]![0])
    }
    return this.endVector()
  }
  /** Vector of Block structs `{i64 offset, i32 metaDataLength, pad, i64 bodyLength}` (24 bytes). */
  createBlockVector(blocks: Array<[number, number, number]>): number {
    this.startVector(24, blocks.length, 8)
    for (let i = blocks.length - 1; i >= 0; i--) {
      this.addI64(blocks[i]![2])
      this.pad(4)
      this.addI32(blocks[i]![1])
      this.addI64(blocks[i]![0])
    }
    return this.endVector()
  }
  finish(root: number): Uint8Array {
    this.prep(this.minalign, 4)
    this.addOffset(root)
    return this.bb.slice(this.space)
  }
}

// ───────────────────────────── FlatBuffers reader ─────────────────────────────

function fbFieldPos(dv: DataView, table: number, slot: number): number {
  const vt = table - dv.getInt32(table, true)
  const vsize = dv.getUint16(vt, true)
  const off = 4 + slot * 2
  if (off + 2 > vsize) return 0
  const fo = dv.getUint16(vt + off, true)
  return fo === 0 ? 0 : table + fo
}
function fbI8(dv: DataView, table: number, slot: number, def: number): number {
  const p = fbFieldPos(dv, table, slot)
  return p ? dv.getInt8(p) : def
}
function fbI16(dv: DataView, table: number, slot: number, def: number): number {
  const p = fbFieldPos(dv, table, slot)
  return p ? dv.getInt16(p, true) : def
}
function fbI32(dv: DataView, table: number, slot: number, def: number): number {
  const p = fbFieldPos(dv, table, slot)
  return p ? dv.getInt32(p, true) : def
}
function fbI64(dv: DataView, table: number, slot: number, def: number): number {
  const p = fbFieldPos(dv, table, slot)
  return p ? Number(dv.getBigInt64(p, true)) : def
}
function fbTable(dv: DataView, table: number, slot: number): number {
  const p = fbFieldPos(dv, table, slot)
  return p ? p + dv.getUint32(p, true) : 0
}
/** Returns [elementsStart, length] of a vector field, or null when absent. */
function fbVector(dv: DataView, table: number, slot: number): [number, number] | null {
  const p = fbFieldPos(dv, table, slot)
  if (!p) return null
  const v = p + dv.getUint32(p, true)
  return [v + 4, dv.getUint32(v, true)]
}
function fbString(dv: DataView, table: number, slot: number): string {
  const p = fbFieldPos(dv, table, slot)
  if (!p) return ''
  const s = p + dv.getUint32(p, true)
  const len = dv.getUint32(s, true)
  return new TextDecoder().decode(new Uint8Array(dv.buffer, dv.byteOffset + s + 4, len))
}

// ───────────────────────────── Arrow schema constants ─────────────────────────────

const enum ArrowType {
  Null = 1,
  Int = 2,
  FloatingPoint = 3,
  Binary = 4,
  Utf8 = 5,
  Bool = 6,
  Date = 8,
  Timestamp = 10,
  LargeUtf8 = 20,
  Utf8View = 24,
}
const enum Header {
  Schema = 1,
  DictionaryBatch = 2,
  RecordBatch = 3,
}
const METADATA_V5 = 4
const CONTINUATION = 0xffffffff
const FILE_MAGIC = 'ARROW1'
const ARROW_TYPE_NAMES: Record<number, string> = {
  0: 'NONE', 1: 'Null', 2: 'Int', 3: 'FloatingPoint', 4: 'Binary', 5: 'Utf8', 6: 'Bool', 7: 'Decimal', 8: 'Date',
  9: 'Time', 10: 'Timestamp', 11: 'Interval', 12: 'List', 13: 'Struct', 14: 'Union', 15: 'FixedSizeBinary',
  16: 'FixedSizeList', 17: 'Map', 18: 'Duration', 19: 'LargeBinary', 20: 'LargeUtf8', 21: 'LargeList',
  22: 'RunEndEncoded', 23: 'BinaryView', 24: 'Utf8View', 25: 'ListView', 26: 'LargeListView',
}

const align8 = (n: number): number => (n + 7) & ~7

// ───────────────────────────── Writer ─────────────────────────────

export interface ArrowIpcWriteOptions {
  /** `'stream'` (default; what DuckDB `insertArrowFromIPCStream` / `pl.read_ipc_stream` expect) or `'file'` (Feather v2, random access). */
  format?: 'stream' | 'file'
  /** Split the table into record batches of this many rows (default: one batch). */
  batchRows?: number
}

interface EncodedBuffers {
  nodes: Array<[number, number]>
  buffers: Array<[number, number]>
  parts: Uint8Array[]
  bodyLength: number
}

function writeType(b: FbBuilder, dtype: DType): { typeType: number; typeOff: number } {
  switch (dtype) {
    case 'f64':
    case 'f32': {
      b.startObject(1)
      b.addFieldI16(0, dtype === 'f64' ? 2 : 1, 0)
      return { typeType: ArrowType.FloatingPoint, typeOff: b.endObject() }
    }
    case 'i32':
    case 'u32': {
      b.startObject(2)
      b.addFieldI32(0, 32, 0)
      b.addFieldI8(1, dtype === 'i32' ? 1 : 0, 0)
      return { typeType: ArrowType.Int, typeOff: b.endObject() }
    }
    case 'bool': {
      b.startObject(0)
      return { typeType: ArrowType.Bool, typeOff: b.endObject() }
    }
    case 'utf8':
    case 'category': {
      b.startObject(0)
      return { typeType: ArrowType.Utf8, typeOff: b.endObject() }
    }
    case 'datetime': {
      b.startObject(2)
      b.addFieldI16(0, 1, 0) // MILLISECOND
      return { typeType: ArrowType.Timestamp, typeOff: b.endObject() }
    }
  }
}

function writeField(b: FbBuilder, field: Field, dictionaryId: number): number {
  const nameOff = b.createString(field.name)
  const { typeType, typeOff } = writeType(b, field.dtype)
  let dictOff = 0
  if (field.dtype === 'category') {
    b.startObject(2)
    b.addFieldI32(0, 32, 0)
    b.addFieldI8(1, 1, 0)
    const indexType = b.endObject()
    b.startObject(4)
    b.addFieldI64(0, dictionaryId, 0)
    b.addFieldOffset(1, indexType)
    dictOff = b.endObject()
  }
  const children = b.createOffsetVector([])
  b.startObject(7)
  b.addFieldOffset(0, nameOff)
  b.addFieldI8(1, field.nullable ? 1 : 0, 0)
  b.addFieldI8(2, typeType, 0)
  b.addFieldOffset(3, typeOff)
  b.addFieldOffset(4, dictOff)
  b.addFieldOffset(5, children)
  return b.endObject()
}

function writeSchemaTable(b: FbBuilder, schema: readonly Field[], dictIds: Map<string, number>): number {
  const fieldOffs = schema.map((f) => writeField(b, f, dictIds.get(f.name) ?? 0))
  const fields = b.createOffsetVector(fieldOffs)
  b.startObject(4)
  b.addFieldOffset(1, fields)
  return b.endObject()
}

function finishMessage(b: FbBuilder, headerType: number, headerOff: number, bodyLength: number): Uint8Array {
  b.startObject(5)
  b.addFieldI16(0, METADATA_V5, 0)
  b.addFieldI8(1, headerType, 0)
  b.addFieldOffset(2, headerOff)
  b.addFieldI64(3, bodyLength, 0)
  return b.finish(b.endObject())
}

function writeRecordBatchTable(b: FbBuilder, length: number, enc: EncodedBuffers): number {
  const nodes = b.createI64PairVector(enc.nodes)
  const buffers = b.createI64PairVector(enc.buffers)
  b.startObject(5)
  b.addFieldI64(0, length, 0)
  b.addFieldOffset(1, nodes)
  b.addFieldOffset(2, buffers)
  return b.endObject()
}

function schemaMessage(schema: readonly Field[], dictIds: Map<string, number>): Uint8Array {
  const b = new FbBuilder()
  return finishMessage(b, Header.Schema, writeSchemaTable(b, schema, dictIds), 0)
}

function recordBatchMessage(length: number, enc: EncodedBuffers): Uint8Array {
  const b = new FbBuilder()
  return finishMessage(b, Header.RecordBatch, writeRecordBatchTable(b, length, enc), enc.bodyLength)
}

function dictionaryBatchMessage(id: number, length: number, enc: EncodedBuffers): Uint8Array {
  const b = new FbBuilder()
  const rb = writeRecordBatchTable(b, length, enc)
  b.startObject(3)
  b.addFieldI64(0, id, 0)
  b.addFieldOffset(1, rb)
  return finishMessage(b, Header.DictionaryBatch, b.endObject(), enc.bodyLength)
}

/** Body buffers: each padded to 8 bytes; offsets are relative to the body start. */
class BodyEncoder implements EncodedBuffers {
  nodes: Array<[number, number]> = []
  buffers: Array<[number, number]> = []
  parts: Uint8Array[] = []
  bodyLength = 0

  push(bytes: Uint8Array | null): void {
    if (!bytes || bytes.length === 0) {
      this.buffers.push([this.bodyLength, 0])
      return
    }
    const padded = align8(bytes.length)
    this.buffers.push([this.bodyLength, bytes.length])
    this.parts.push(bytes)
    if (padded > bytes.length) this.parts.push(new Uint8Array(padded - bytes.length))
    this.bodyLength += padded
  }
}

function rawBytes(arr: ArrayBufferView): Uint8Array {
  // copies (the source may live on a SharedArrayBuffer; the IPC buffer must be a plain ArrayBuffer)
  return new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength).slice()
}

function sliceBitmap(bitmap: Uint8Array | undefined, start: number, end: number): { bytes: Uint8Array | null; nulls: number } {
  if (!bitmap) return { bytes: null, nulls: 0 }
  const n = end - start
  const out = new Uint8Array(Math.ceil(n / 8) || 1)
  let nulls = 0
  for (let i = 0; i < n; i++) {
    if (isValid(bitmap, start + i)) out[i >> 3]! |= 1 << (i & 7)
    else nulls++
  }
  return nulls === 0 ? { bytes: null, nulls: 0 } : { bytes: out, nulls }
}

function encodeUtf8Values(values: ArrayLike<string>, start: number, end: number, valid: (i: number) => boolean): { offsets: Int32Array; data: Uint8Array } {
  const n = end - start
  const encoder = new TextEncoder()
  const chunks: Uint8Array[] = []
  const offsets = new Int32Array(n + 1)
  let total = 0
  for (let i = 0; i < n; i++) {
    if (valid(start + i)) {
      const bytes = encoder.encode(values[start + i]!)
      chunks.push(bytes)
      total += bytes.length
      if (total > 0x7fffffff) throw new RangeError('toArrowIpc: utf8 column exceeds 2 GiB per batch — use batchRows to split it')
    }
    offsets[i + 1] = total
  }
  const data = new Uint8Array(total)
  let pos = 0
  for (const c of chunks) {
    data.set(c, pos)
    pos += c.length
  }
  return { offsets, data }
}

function encodeColumn(col: Column, start: number, end: number, body: BodyEncoder): void {
  const n = end - start
  const validity = sliceBitmap(col.nullBitmap, start, end)
  body.nodes.push([n, validity.nulls])
  body.push(validity.bytes)
  const valid = (i: number): boolean => isValid(col.nullBitmap, i)
  switch (col.field.dtype) {
    case 'f64':
    case 'f32':
    case 'i32':
    case 'u32': {
      const data = col.data as Float64Array | Float32Array | Int32Array | Uint32Array
      body.push(rawBytes(data.subarray(start, end)))
      return
    }
    case 'category': {
      const codes = col.data as Uint32Array
      const out = new Int32Array(n)
      for (let i = 0; i < n; i++) out[i] = valid(start + i) ? codes[start + i]! : 0
      body.push(rawBytes(out))
      return
    }
    case 'datetime': {
      const ms = col.data as Float64Array
      const out = new BigInt64Array(n)
      for (let i = 0; i < n; i++) {
        const v = ms[start + i]!
        out[i] = valid(start + i) && Number.isFinite(v) ? BigInt(Math.trunc(v)) : 0n
      }
      body.push(rawBytes(out))
      return
    }
    case 'bool': {
      const flags = col.data as Uint8Array
      const out = new Uint8Array(Math.ceil(n / 8) || 1)
      for (let i = 0; i < n; i++) if (flags[start + i]) out[i >> 3]! |= 1 << (i & 7)
      body.push(out)
      return
    }
    case 'utf8': {
      const { offsets, data } = encodeUtf8Values(col.data as string[], start, end, valid)
      body.push(rawBytes(offsets))
      body.push(data)
      return
    }
  }
}

function encapsulate(meta: Uint8Array, body: EncodedBuffers | null): Uint8Array {
  const metaPadded = align8(meta.length)
  const out = new Uint8Array(8 + metaPadded + (body?.bodyLength ?? 0))
  const dv = new DataView(out.buffer)
  dv.setUint32(0, CONTINUATION, true)
  dv.setInt32(4, metaPadded, true)
  out.set(meta, 8)
  if (body) {
    let pos = 8 + metaPadded
    for (const p of body.parts) {
      out.set(p, pos)
      pos += p.length
    }
  }
  return out
}

/**
 * Serialize a table as Apache Arrow IPC bytes. See the module comment for the type mapping.
 */
export function toArrowIpc(table: TableView, options: ArrowIpcWriteOptions = {}): Uint8Array {
  const format = options.format ?? 'stream'
  const batchRows = options.batchRows && options.batchRows > 0 ? Math.floor(options.batchRows) : table.numRows
  const dictIds = new Map<string, number>()
  for (const col of table.columns) if (col.field.dtype === 'category') dictIds.set(col.field.name, dictIds.size)

  const messages: Uint8Array[] = []
  const dictBlocks: Array<[number, number, number]> = []
  const batchBlocks: Array<[number, number, number]> = []
  let fileOffset = format === 'file' ? 8 : 0
  const emit = (msg: Uint8Array, blocks?: Array<[number, number, number]>): void => {
    const metaLen = 8 + new DataView(msg.buffer, msg.byteOffset).getInt32(4, true)
    blocks?.push([fileOffset, metaLen, msg.length - metaLen])
    messages.push(msg)
    fileOffset += msg.length
  }

  emit(encapsulate(schemaMessage(table.schema, dictIds), null))

  for (const col of table.columns) {
    if (col.field.dtype !== 'category') continue
    const dict = col.dictionary ?? []
    const body = new BodyEncoder()
    body.nodes.push([dict.length, 0])
    body.push(null)
    const { offsets, data } = encodeUtf8Values(dict, 0, dict.length, () => true)
    body.push(rawBytes(offsets))
    body.push(data)
    emit(encapsulate(dictionaryBatchMessage(dictIds.get(col.field.name)!, dict.length, body), body), dictBlocks)
  }

  const batches = table.numRows === 0 ? 1 : Math.ceil(table.numRows / batchRows)
  for (let bi = 0; bi < batches; bi++) {
    const start = bi * batchRows
    const end = Math.min(table.numRows, start + batchRows)
    const body = new BodyEncoder()
    for (const col of table.columns) encodeColumn(col, start, end, body)
    emit(encapsulate(recordBatchMessage(end - start, body), body), batchBlocks)
  }

  const eos = new Uint8Array(8)
  new DataView(eos.buffer).setUint32(0, CONTINUATION, true)
  messages.push(eos)
  fileOffset += 8

  if (format === 'stream') return concatBytes(messages)

  // File format: magic + stream messages + footer (schema + block index) + footer length + magic
  const fb = new FbBuilder()
  const schemaOff = writeSchemaTable(fb, table.schema, dictIds)
  const dictVec = fb.createBlockVector(dictBlocks)
  const batchVec = fb.createBlockVector(batchBlocks)
  fb.startObject(5)
  fb.addFieldI16(0, METADATA_V5, 0)
  fb.addFieldOffset(1, schemaOff)
  fb.addFieldOffset(2, dictVec)
  fb.addFieldOffset(3, batchVec)
  const footer = fb.finish(fb.endObject())
  const magic = new TextEncoder().encode(FILE_MAGIC)
  const head = new Uint8Array(8)
  head.set(magic)
  const tail = new Uint8Array(4 + 6)
  new DataView(tail.buffer).setInt32(0, footer.length, true)
  tail.set(magic, 4)
  return concatBytes([head, ...messages, footer, tail])
}

function concatBytes(parts: Uint8Array[]): Uint8Array {
  let total = 0
  for (const p of parts) total += p.length
  const out = new Uint8Array(total)
  let pos = 0
  for (const p of parts) {
    out.set(p, pos)
    pos += p.length
  }
  return out
}

// ───────────────────────────── Reader ─────────────────────────────

interface ArrowFieldInfo {
  name: string
  nullable: boolean
  typeType: number
  /** Int: bit width; FloatingPoint: precision; Timestamp/Date: unit */
  a: number
  /** Int: signed */
  b: number
  dictionaryId: number | null
  indexBits: number
  indexSigned: boolean
  hasChildren: boolean
}

interface DecodedColumn {
  dtype: DType
  data: Float64Array | Float32Array | Int32Array | Uint32Array | Uint8Array | string[]
  nullBitmap?: Uint8Array
  /** for dictionary-encoded fields: the codes (data) reference this id's values */
  dictionaryId?: number
}

function parseField(dv: DataView, table: number): ArrowFieldInfo {
  const typeType = fbI8(dv, table, 2, 0) & 0xff
  const typeTable = fbTable(dv, table, 3)
  const children = fbVector(dv, table, 5)
  let a = 0
  let b = 0
  if (typeTable) {
    if (typeType === ArrowType.Int) {
      a = fbI32(dv, typeTable, 0, 0)
      b = fbI8(dv, typeTable, 1, 0)
    } else if (typeType === ArrowType.FloatingPoint || typeType === ArrowType.Timestamp || typeType === ArrowType.Date) {
      a = fbI16(dv, typeTable, 0, 0)
    }
  }
  const dictTable = fbTable(dv, table, 4)
  let dictionaryId: number | null = null
  let indexBits = 32
  let indexSigned = true
  if (dictTable) {
    dictionaryId = fbI64(dv, dictTable, 0, 0)
    const idx = fbTable(dv, dictTable, 1)
    if (idx) {
      indexBits = fbI32(dv, idx, 0, 32)
      indexSigned = fbI8(dv, idx, 1, 0) !== 0
    }
  }
  return {
    name: fbString(dv, table, 0),
    nullable: fbI8(dv, table, 1, 0) !== 0,
    typeType,
    a,
    b,
    dictionaryId,
    indexBits,
    indexSigned,
    hasChildren: children !== null && children[1] > 0,
  }
}

function parseSchema(dv: DataView, schemaTable: number): ArrowFieldInfo[] {
  if (fbI16(dv, schemaTable, 0, 0) !== 0) throw new Error('fromArrowIpc: big-endian Arrow data is not supported')
  const vec = fbVector(dv, schemaTable, 1)
  if (!vec) return []
  const [start, len] = vec
  const fields: ArrowFieldInfo[] = []
  for (let i = 0; i < len; i++) {
    const p = start + i * 4
    fields.push(parseField(dv, p + dv.getUint32(p, true)))
  }
  return fields
}

function unsupported(field: ArrowFieldInfo, why: string): Error {
  const t = ARROW_TYPE_NAMES[field.typeType] ?? `type#${field.typeType}`
  return new Error(`fromArrowIpc: column "${field.name}" (${t}): ${why}`)
}

function readBitmap(bytes: Uint8Array, n: number): Uint8Array {
  const out = new Uint8Array(Math.ceil(n / 8) || 1)
  out.set(bytes.subarray(0, Math.min(bytes.length, out.length)))
  // clear any trailing bits past n so equality checks stay stable
  if (n % 8 !== 0 && out.length) out[out.length - 1]! &= (1 << (n % 8)) - 1
  return out
}

function readInts(bytes: Uint8Array, bits: number, signed: boolean, n: number): { data: Int32Array | Uint32Array | Float64Array; dtype: DType } {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  if (bits === 8 || bits === 16) {
    const out = new Int32Array(n)
    for (let i = 0; i < n; i++) {
      out[i] =
        bits === 8 ? (signed ? dv.getInt8(i) : dv.getUint8(i)) : signed ? dv.getInt16(i * 2, true) : dv.getUint16(i * 2, true)
    }
    return { data: out, dtype: 'i32' }
  }
  if (bits === 32) {
    if (signed) return { data: new Int32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + n * 4)), dtype: 'i32' }
    return { data: new Uint32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + n * 4)), dtype: 'u32' }
  }
  if (bits === 64) {
    // dictionary indices only (a dictionary cannot have 2^53 entries); value columns go through readInt64Column
    const out = new Float64Array(n)
    for (let i = 0; i < n; i++) out[i] = Number(signed ? dv.getBigInt64(i * 8, true) : dv.getBigUint64(i * 8, true))
    return { data: out, dtype: 'f64' }
  }
  throw new Error(`fromArrowIpc: unsupported integer width ${bits}`)
}

/** Int64 / UInt64 value column under the reader's Int64Policy — never a silently rounded value. */
function readInt64Column(
  bytes: Uint8Array,
  signed: boolean,
  n: number,
  nullBitmap: Uint8Array | undefined,
  column: string,
  policy: Int64Policy,
  rowBase: number,
): { data: Float64Array | string[]; dtype: DType } {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const at = (i: number): bigint => (signed ? dv.getBigInt64(i * 8, true) : dv.getBigUint64(i * 8, true))
  if (policy === 'string') {
    const out = new Array<string>(n)
    for (let i = 0; i < n; i++) out[i] = isValid(nullBitmap, i) ? at(i).toString() : ''
    return { data: out, dtype: 'utf8' }
  }
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    if (!isValid(nullBitmap, i)) continue
    const v = at(i)
    if (policy === 'error' && !isSafeBigInt(v)) throw new PrecisionLossError(column, rowBase + i, v.toString(), 'fromArrowIpc')
    out[i] = Number(v)
  }
  return { data: out, dtype: 'f64' }
}

/** Epoch value in `unitsPerMs` units (BigInt) → milliseconds as a double, without rounding the integer part first. */
function toEpochMs(v: bigint, unitsPerMs: bigint): number {
  if (unitsPerMs === 1n) return Number(v)
  const q = v / unitsPerMs
  const r = v % unitsPerMs
  return Number(q) + Number(r) / Number(unitsPerMs)
}

function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1
  const e = (h >> 10) & 0x1f
  const f = h & 0x3ff
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024)
  if (e === 31) return f ? NaN : s * Infinity
  return s * Math.pow(2, e - 15) * (1 + f / 1024)
}

function readOffsets(bytes: Uint8Array, n: number, large: boolean): number[] {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const out = new Array<number>(n + 1)
  for (let i = 0; i <= n; i++) out[i] = large ? Number(dv.getBigInt64(i * 8, true)) : dv.getInt32(i * 4, true)
  return out
}

function readStrings(offsetsBytes: Uint8Array, dataBytes: Uint8Array, n: number, large: boolean): string[] {
  const offsets = readOffsets(offsetsBytes, n, large)
  const decoder = new TextDecoder()
  const out = new Array<string>(n)
  for (let i = 0; i < n; i++) {
    const a = offsets[i]!
    const b = offsets[i + 1]!
    out[i] = b > a ? decoder.decode(dataBytes.subarray(a, b)) : ''
  }
  return out
}

/** Decode one flat field from a record batch body; advances the node/buffer cursors. */
/**
 * Utf8View ("German strings"): 16-byte views — length, then either 12 inline bytes or prefix + buffer index +
 * offset into one of the variadic data buffers.
 */
function readStringViews(views: Uint8Array, dataBuffers: Uint8Array[], n: number): string[] {
  const dv = new DataView(views.buffer, views.byteOffset, views.byteLength)
  const decoder = new TextDecoder()
  const out = new Array<string>(n)
  for (let i = 0; i < n; i++) {
    const p = i * 16
    const len = dv.getInt32(p, true)
    if (len <= 12) {
      out[i] = len > 0 ? decoder.decode(views.subarray(p + 4, p + 4 + len)) : ''
    } else {
      const bufIdx = dv.getInt32(p + 8, true)
      const off = dv.getInt32(p + 12, true)
      const buf = dataBuffers[bufIdx]
      if (!buf) throw new Error(`fromArrowIpc: Utf8View references data buffer ${bufIdx} which the batch does not carry`)
      out[i] = decoder.decode(buf.subarray(off, off + len))
    }
  }
  return out
}

function decodeField(
  field: ArrowFieldInfo,
  cur: { node: number; buffer: number; variadic: number },
  nodes: Array<[number, number]>,
  buffers: Array<[number, number]>,
  body: Uint8Array,
  variadic: number[] = [],
  opts: { int64: Int64Policy; rowBase: number } = { int64: 'error', rowBase: 0 },
): DecodedColumn {
  if (field.hasChildren) throw unsupported(field, 'nested columns are not supported')
  const node = nodes[cur.node++]
  if (!node) throw unsupported(field, 'record batch has fewer nodes than schema fields')
  const [n, nullCount] = node
  const buf = (): Uint8Array => {
    const b = buffers[cur.buffer++]
    if (!b) throw unsupported(field, 'record batch has fewer buffers than the schema needs')
    return body.subarray(b[0], b[0] + b[1])
  }
  const t = field.typeType
  if (t === ArrowType.Null) {
    return { dtype: 'f64', data: new Float64Array(n), nullBitmap: new Uint8Array(Math.ceil(n / 8) || 1) }
  }
  const validityBytes = buf()
  const nullBitmap = nullCount !== 0 && validityBytes.length > 0 ? readBitmap(validityBytes, n) : undefined
  if (field.dictionaryId !== null) {
    const { data } = readInts(buf(), field.indexBits, field.indexSigned, n)
    const codes = new Uint32Array(n)
    for (let i = 0; i < n; i++) codes[i] = isValid(nullBitmap, i) ? Number(data[i]) : 0
    return { dtype: 'category', data: codes, nullBitmap, dictionaryId: field.dictionaryId }
  }
  switch (t) {
    case ArrowType.Int: {
      if (field.a === 64) {
        const r = readInt64Column(buf(), field.b !== 0, n, nullBitmap, field.name, opts.int64, opts.rowBase)
        return { dtype: r.dtype, data: r.data, nullBitmap }
      }
      const r = readInts(buf(), field.a, field.b !== 0, n)
      return { dtype: r.dtype, data: r.data, nullBitmap }
    }
    case ArrowType.FloatingPoint: {
      const bytes = buf()
      if (field.a === 2) return { dtype: 'f64', data: new Float64Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + n * 8)), nullBitmap }
      if (field.a === 1) return { dtype: 'f32', data: new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + n * 4)), nullBitmap }
      const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
      const out = new Float32Array(n)
      for (let i = 0; i < n; i++) out[i] = halfToFloat(dv.getUint16(i * 2, true))
      return { dtype: 'f32', data: out, nullBitmap }
    }
    case ArrowType.Bool: {
      const bits = buf()
      const out = new Uint8Array(n)
      for (let i = 0; i < n; i++) out[i] = (bits[i >> 3]! >> (i & 7)) & 1
      return { dtype: 'bool', data: out, nullBitmap }
    }
    case ArrowType.Utf8:
    case ArrowType.LargeUtf8: {
      const offsets = buf()
      const data = buf()
      return { dtype: 'utf8', data: readStrings(offsets, data, n, t === ArrowType.LargeUtf8), nullBitmap }
    }
    case ArrowType.Utf8View: {
      const views = buf()
      const count = variadic[cur.variadic++] ?? 0
      const dataBuffers: Uint8Array[] = []
      for (let i = 0; i < count; i++) dataBuffers.push(buf())
      return { dtype: 'utf8', data: readStringViews(views, dataBuffers, n), nullBitmap }
    }
    case ArrowType.Timestamp:
    case ArrowType.Date: {
      const bytes = buf()
      const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
      const out = new Float64Array(n)
      if (t === ArrowType.Date && field.a === 0) {
        for (let i = 0; i < n; i++) out[i] = dv.getInt32(i * 4, true) * 86_400_000
      } else {
        // Timestamp: SECOND=0, MILLISECOND=1, MICROSECOND=2, NANOSECOND=3; Date64 is milliseconds.
        // Converted exactly in BigInt (ns epoch values exceed 2^53); the ms double keeps ~0.1 µs today.
        const unit = t === ArrowType.Date ? 1 : field.a
        for (let i = 0; i < n; i++) {
          if (!isValid(nullBitmap, i)) continue
          const v = dv.getBigInt64(i * 8, true)
          if (unit === 0) {
            if (!isSafeBigInt(v * 1000n)) throw new PrecisionLossError(field.name, opts.rowBase + i, `${v} s`, 'fromArrowIpc')
            out[i] = Number(v) * 1000
          } else {
            const perMs = unit === 1 ? 1n : unit === 2 ? 1000n : 1_000_000n
            if (!isSafeBigInt(v / perMs)) throw new PrecisionLossError(field.name, opts.rowBase + i, `${v}`, 'fromArrowIpc')
            out[i] = toEpochMs(v, perMs)
          }
        }
      }
      return { dtype: 'datetime', data: out, nullBitmap }
    }
    default:
      throw unsupported(field, 'this Arrow type has no columna dtype (supported: Int, FloatingPoint, Bool, Utf8, LargeUtf8, Utf8View, Timestamp, Date, Null, dictionary-encoded strings)')
  }
}

function parseRecordBatch(dv: DataView, rb: number): { length: number; nodes: Array<[number, number]>; buffers: Array<[number, number]>; variadic: number[] } {
  if (fbTable(dv, rb, 3)) throw new Error('fromArrowIpc: compressed record batches (LZ4 / ZSTD) are not supported — write uncompressed IPC')
  const length = fbI64(dv, rb, 0, 0)
  const nodes: Array<[number, number]> = []
  const nv = fbVector(dv, rb, 1)
  if (nv) for (let i = 0; i < nv[1]; i++) nodes.push([Number(dv.getBigInt64(nv[0] + i * 16, true)), Number(dv.getBigInt64(nv[0] + i * 16 + 8, true))])
  const buffers: Array<[number, number]> = []
  const bv = fbVector(dv, rb, 2)
  if (bv) for (let i = 0; i < bv[1]; i++) buffers.push([Number(dv.getBigInt64(bv[0] + i * 16, true)), Number(dv.getBigInt64(bv[0] + i * 16 + 8, true))])
  const variadic: number[] = []
  const vv = fbVector(dv, rb, 4)
  if (vv) for (let i = 0; i < vv[1]; i++) variadic.push(Number(dv.getBigInt64(vv[0] + i * 8, true)))
  return { length, nodes, buffers, variadic }
}

function concatDecoded(dtype: DType, parts: DecodedColumn[], total: number): { data: Column['data']; nullBitmap?: Uint8Array } {
  if (parts.length === 1) return { data: parts[0]!.data as Column['data'], nullBitmap: parts[0]!.nullBitmap }
  const anyNull = parts.some((p) => p.nullBitmap)
  const nullBitmap = anyNull ? new Uint8Array(Math.ceil(total / 8) || 1) : undefined
  let data: Column['data']
  if (dtype === 'utf8') data = new Array<string>(total)
  else if (dtype === 'bool') data = new Uint8Array(total)
  else if (dtype === 'i32') data = new Int32Array(total)
  else if (dtype === 'u32' || dtype === 'category') data = new Uint32Array(total)
  else if (dtype === 'f32') data = new Float32Array(total)
  else data = new Float64Array(total)
  let pos = 0
  for (const p of parts) {
    const n = Array.isArray(p.data) ? p.data.length : p.data.length
    if (Array.isArray(data)) for (let i = 0; i < n; i++) data[pos + i] = (p.data as string[])[i]!
    else (data as Float64Array).set(p.data as Float64Array, pos)
    if (nullBitmap) for (let i = 0; i < n; i++) setValid(nullBitmap, pos + i, isValid(p.nullBitmap, i))
    pos += n
  }
  return { data, nullBitmap }
}

/**
 * Parse Apache Arrow IPC bytes (stream or file format) into a TableView. Several record batches are
 * concatenated; dictionary deltas are applied. Big-endian, nested, compressed and exotic types throw with the
 * column name — nothing is silently coerced.
 */
export interface ArrowIpcReadOptions {
  /** Int64 / UInt64 columns: `'error'` (default) `| 'string' | 'number'` — see {@link Int64Policy}. */
  int64?: Int64Policy
}

export function fromArrowIpc(input: Uint8Array | ArrayBuffer, options: ArrowIpcReadOptions = {}): TableView {
  const int64: Int64Policy = options.int64 ?? 'error'
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input)
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const isFile = bytes.length >= 8 && new TextDecoder().decode(bytes.subarray(0, 6)) === FILE_MAGIC

  let fields: ArrowFieldInfo[] | null = null
  const dictionaries = new Map<number, string[]>()
  const batches: DecodedColumn[][] = []
  let totalRows = 0

  // One encapsulated message at `pos`; returns null at end of stream / footer.
  const readMessage = (pos: number): { headerType: number; header: number; body: Uint8Array; next: number } | null => {
    if (pos + 4 > bytes.length) return null
    let metaLen = dv.getInt32(pos, true)
    if ((metaLen >>> 0) === CONTINUATION) {
      if (pos + 8 > bytes.length) return null
      metaLen = dv.getInt32(pos + 4, true)
      pos += 8
    } else {
      if (isFile) return null
      pos += 4 // legacy (pre-0.15) encapsulation without the continuation marker
    }
    if (metaLen === 0) return null
    if (metaLen < 0 || pos + metaLen > bytes.length) throw new Error('fromArrowIpc: truncated IPC message')
    const msg = pos + dv.getUint32(pos, true)
    const headerType = fbI8(dv, msg, 1, 0)
    const header = fbTable(dv, msg, 2)
    const bodyLength = fbI64(dv, msg, 3, 0)
    const bodyStart = pos + metaLen
    if (bodyStart + bodyLength > bytes.length) throw new Error('fromArrowIpc: truncated IPC body')
    return { headerType, header, body: bytes.subarray(bodyStart, bodyStart + bodyLength), next: bodyStart + bodyLength }
  }

  const handle = (m: { headerType: number; header: number; body: Uint8Array }): void => {
    const { headerType, header, body } = m
    if (headerType === Header.Schema) {
      if (!fields) fields = parseSchema(dv, header)
    } else if (headerType === Header.DictionaryBatch) {
      if (!fields) throw new Error('fromArrowIpc: dictionary batch before schema')
      const id = fbI64(dv, header, 0, 0)
      const isDelta = fbI8(dv, header, 2, 0) !== 0
      const rb = parseRecordBatch(dv, fbTable(dv, header, 1))
      const owner = fields.find((f) => f.dictionaryId === id)
      if (!owner) throw new Error(`fromArrowIpc: dictionary batch for unknown id ${id}`)
      // dictionary values are decoded as a plain (non-dictionary) column of the field's value type
      const valueField: ArrowFieldInfo = { ...owner, dictionaryId: null }
      const decoded = decodeField(valueField, { node: 0, buffer: 0, variadic: 0 }, rb.nodes, rb.buffers, body, rb.variadic)
      if (decoded.dtype !== 'utf8') throw unsupported(owner, 'only dictionary-encoded strings map to category')
      const values = decoded.data as string[]
      if (isDelta && dictionaries.has(id)) dictionaries.get(id)!.push(...values)
      else dictionaries.set(id, values)
    } else if (headerType === Header.RecordBatch) {
      if (!fields) throw new Error('fromArrowIpc: record batch before schema')
      const rb = parseRecordBatch(dv, header)
      const cur = { node: 0, buffer: 0, variadic: 0 }
      const cols = fields.map((f) => decodeField(f, cur, rb.nodes, rb.buffers, body, rb.variadic, { int64, rowBase: totalRows }))
      batches.push(cols)
      totalRows += rb.length
    }
  }

  if (isFile) {
    // File format: the footer carries the schema and the block index; the schema message inline is optional
    // (pyarrow writes it, apache-arrow JS does not).
    const footerLen = dv.getInt32(bytes.length - 10, true)
    const footerStart = bytes.length - 10 - footerLen
    if (footerLen <= 0 || footerStart < 8) throw new Error('fromArrowIpc: corrupt Arrow file footer')
    const footer = footerStart + dv.getUint32(footerStart, true)
    fields = parseSchema(dv, fbTable(dv, footer, 1))
    const blocks = (slot: number): number[] => {
      const v = fbVector(dv, footer, slot)
      const out: number[] = []
      if (v) for (let i = 0; i < v[1]; i++) out.push(Number(dv.getBigInt64(v[0] + i * 24, true)))
      return out
    }
    for (const off of [...blocks(2), ...blocks(3)]) {
      const m = readMessage(off)
      if (!m) throw new Error('fromArrowIpc: block offset points past the end of the file')
      handle(m)
    }
  } else {
    const pos = 0
    for (let m = readMessage(pos); m; m = readMessage(m.next)) handle(m)
  }
  if (!fields) throw new Error('fromArrowIpc: no schema message found (not Arrow IPC bytes?)')
  const schema: ArrowFieldInfo[] = fields

  const columns: Column[] = schema.map((f, ci) => {
    const parts = batches.map((b) => b[ci]!)
    const first = parts[0]
    const dtype: DType = first ? first.dtype : dtypeOfField(f)
    const merged = first ? concatDecoded(dtype, parts, totalRows) : emptyData(dtype)
    const col: Column = { field: { name: f.name, dtype, nullable: f.nullable }, data: merged.data, nullBitmap: merged.nullBitmap }
    if (dtype === 'category') {
      const id = f.dictionaryId ?? -1
      col.dictionary = dictionaries.get(id) ?? []
    }
    return col
  })
  return tableFromColumns(columns)
}

/** columna dtype of an Arrow field when no batch tells us (empty table). */
function dtypeOfField(f: ArrowFieldInfo): DType {
  if (f.dictionaryId !== null) return 'category'
  switch (f.typeType) {
    case ArrowType.Int:
      return f.a === 32 ? (f.b !== 0 ? 'i32' : 'u32') : f.a === 64 ? 'f64' : 'i32'
    case ArrowType.FloatingPoint:
      return f.a === 2 ? 'f64' : 'f32'
    case ArrowType.Bool:
      return 'bool'
    case ArrowType.Utf8:
    case ArrowType.LargeUtf8:
    case ArrowType.Utf8View:
      return 'utf8'
    case ArrowType.Timestamp:
    case ArrowType.Date:
      return 'datetime'
    default:
      return 'f64'
  }
}

function emptyData(dtype: DType): { data: Column['data']; nullBitmap?: Uint8Array } {
  switch (dtype) {
    case 'utf8':
      return { data: [] }
    case 'bool':
      return { data: new Uint8Array(0) }
    case 'i32':
      return { data: new Int32Array(0) }
    case 'u32':
    case 'category':
      return { data: new Uint32Array(0) }
    case 'f32':
      return { data: new Float32Array(0) }
    default:
      return { data: new Float64Array(0) }
  }
}
