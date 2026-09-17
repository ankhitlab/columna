export type DType =
  | 'f64'
  | 'f32'
  | 'i32'
  | 'u32'
  | 'bool'
  | 'utf8'
  | 'category'
  | 'datetime'

export type NumericDType = 'f64' | 'f32' | 'i32' | 'u32'
export type GpuFriendlyDType = 'f32' | 'i32' | 'u32' | 'bool' | 'category'

export interface Field {
  name: string
  dtype: DType
  nullable: boolean
}

export type Schema = readonly Field[]

export type TypedData =
  | Float64Array
  | Float32Array
  | Int32Array
  | Uint32Array
  | Uint8Array
  | string[]

export interface Column {
  field: Field
  data: TypedData
  /** Bit-packed validity; 1 = valid. Length ceil(n/8). Undefined = all valid. */
  nullBitmap?: Uint8Array
  /** Dictionary for category columns */
  dictionary?: string[]
}

export interface TableView {
  readonly schema: Schema
  readonly numRows: number
  readonly columns: readonly Column[]
}

export function createNullBitmap(numRows: number, isNull?: (i: number) => boolean): Uint8Array | undefined {
  if (!isNull) return undefined
  const bytes = new Uint8Array(Math.ceil(numRows / 8) || 1)
  let anyNull = false
  for (let i = 0; i < numRows; i++) {
    if (isNull(i)) {
      anyNull = true
    } else {
      bytes[i >> 3] |= 1 << (i & 7)
    }
  }
  return anyNull ? bytes : undefined
}

export function isValid(bitmap: Uint8Array | undefined, index: number): boolean {
  if (!bitmap) return true
  return (bitmap[index >> 3]! & (1 << (index & 7))) !== 0
}

export function setValid(bitmap: Uint8Array, index: number, valid: boolean): void {
  const byte = index >> 3
  const bit = 1 << (index & 7)
  if (valid) bitmap[byte]! |= bit
  else bitmap[byte]! &= ~bit
}

export function makeAllValidBitmap(numRows: number): Uint8Array {
  const bytes = new Uint8Array(Math.ceil(numRows / 8) || 1)
  bytes.fill(0xff)
  if (numRows % 8 !== 0) {
    const last = bytes.length - 1
    bytes[last] = (1 << (numRows % 8)) - 1
  }
  return bytes
}

export function getField(schema: Schema, name: string): Field {
  const field = schema.find((f) => f.name === name)
  if (!field) {
    const suggestions = suggestColumns(schema, name)
    const hint = suggestions.length ? ` Did you mean: ${suggestions.join(', ')}?` : ''
    throw new Error(`Unknown column "${name}".${hint}`)
  }
  return field
}

export function suggestColumns(schema: Schema, name: string, limit = 3): string[] {
  return schema
    .map((f) => ({ name: f.name, d: levenshtein(f.name.toLowerCase(), name.toLowerCase()) }))
    .filter((x) => x.d <= Math.max(2, Math.floor(name.length / 2)))
    .sort((a, b) => a.d - b.d)
    .slice(0, limit)
    .map((x) => x.name)
}

function levenshtein(a: string, b: string): number {
  const m = a.length
  const n = b.length
  const dp = Array.from({ length: m + 1 }, () => new Array<number>(n + 1).fill(0))
  for (let i = 0; i <= m; i++) dp[i]![0] = i
  for (let j = 0; j <= n; j++) dp[0]![j] = j
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1
      dp[i]![j] = Math.min(dp[i - 1]![j]! + 1, dp[i]![j - 1]! + 1, dp[i - 1]![j - 1]! + cost)
    }
  }
  return dp[m]![n]!
}

export function isGpuFriendly(dtype: DType): dtype is GpuFriendlyDType {
  return dtype === 'f32' || dtype === 'i32' || dtype === 'u32' || dtype === 'bool' || dtype === 'category'
}

export function isNumeric(dtype: DType): dtype is NumericDType {
  return dtype === 'f64' || dtype === 'f32' || dtype === 'i32' || dtype === 'u32'
}

export function allocateData(dtype: DType, length: number): TypedData {
  // Node: prefer SharedArrayBuffer so worker_threads can attach without copying.
  // Opt out with COLUMNA_SHARED=0. Browsers keep ArrayBuffer (COOP/COEP required for SAB).
  const shared =
    typeof SharedArrayBuffer !== 'undefined' &&
    typeof process !== 'undefined' &&
    Boolean(process.versions?.node) &&
    process.env.COLUMNA_SHARED !== '0'

  const buf = (bytes: number): ArrayBufferLike =>
    shared ? new SharedArrayBuffer(bytes) : new ArrayBuffer(bytes)

  switch (dtype) {
    case 'f64':
    case 'datetime':
      return new Float64Array(buf(length * 8))
    case 'f32':
      return new Float32Array(buf(length * 4))
    case 'i32':
      return new Int32Array(buf(length * 4))
    case 'u32':
    case 'category':
      return new Uint32Array(buf(length * 4))
    case 'bool':
      return new Uint8Array(buf(length))
    case 'utf8':
      return new Array<string>(length).fill('')
  }
}

/** Optionally copy a typed array onto a SharedArrayBuffer (Node workers / COLUMNA_SHARED). */
export function maybeShareTyped<T extends Float64Array | Float32Array | Int32Array | Uint32Array | Uint8Array>(
  arr: T,
): T {
  if (typeof SharedArrayBuffer === 'undefined') return arr
  if (typeof process === 'undefined' || !process.versions?.node) return arr
  if (process.env.COLUMNA_SHARED === '0') return arr
  if (arr.buffer instanceof SharedArrayBuffer && arr.byteOffset === 0 && arr.byteLength === arr.buffer.byteLength) {
    return arr
  }
  const sab = new SharedArrayBuffer(arr.byteLength)
  new Uint8Array(sab).set(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength))
  return new (arr.constructor as new (buffer: SharedArrayBuffer) => T)(sab)
}

export function getColumn(table: TableView, name: string): Column {
  const idx = table.schema.findIndex((f) => f.name === name)
  if (idx < 0) getField(table.schema, name)
  return table.columns[idx]!
}

/** Build once per hot plan node to avoid repeated schema scans. */
export function columnIndex(table: TableView): Map<string, number> {
  const map = new Map<string, number>()
  for (let i = 0; i < table.schema.length; i++) map.set(table.schema[i]!.name, i)
  return map
}

/** Gather rows by index. Fast paths: no-null numeric / category, shared dictionary. */
export function takeColumn(column: Column, indices: ArrayLike<number>): Column {
  const n = indices.length
  const dtype = column.field.dtype
  const srcBitmap = column.nullBitmap
  const dict = column.dictionary

  if (dtype === 'utf8') {
    const src = column.data as string[]
    const data = new Array<string>(n)
    if (!srcBitmap) {
      for (let i = 0; i < n; i++) data[i] = src[indices[i] as number]!
      return { field: column.field, data, dictionary: dict }
    }
    let anyNull = false
    const nullBitmap = new Uint8Array(Math.ceil(n / 8) || 1)
    for (let i = 0; i < n; i++) {
      const srcIdx = indices[i] as number
      data[i] = src[srcIdx]!
      if (isValid(srcBitmap, srcIdx)) setValid(nullBitmap, i, true)
      else anyNull = true
    }
    return {
      field: { ...column.field },
      data,
      nullBitmap: anyNull ? nullBitmap : undefined,
      dictionary: dict,
    }
  }

  // No-null typed gather (bench hot path)
  if (!srcBitmap) {
    if (dtype === 'f64' || dtype === 'datetime') {
      const src = column.data as Float64Array
      const data = new Float64Array(n)
      for (let i = 0; i < n; i++) data[i] = src[indices[i] as number]!
      return { field: column.field, data, dictionary: dict }
    }
    if (dtype === 'f32') {
      const src = column.data as Float32Array
      const data = new Float32Array(n)
      for (let i = 0; i < n; i++) data[i] = src[indices[i] as number]!
      return { field: column.field, data, dictionary: dict }
    }
    if (dtype === 'i32') {
      const src = column.data as Int32Array
      const data = new Int32Array(n)
      for (let i = 0; i < n; i++) data[i] = src[indices[i] as number]!
      return { field: column.field, data, dictionary: dict }
    }
    if (dtype === 'u32' || dtype === 'category') {
      const src = column.data as Uint32Array
      const data = new Uint32Array(n)
      for (let i = 0; i < n; i++) data[i] = src[indices[i] as number]!
      return { field: column.field, data, dictionary: dict }
    }
    if (dtype === 'bool') {
      const src = column.data as Uint8Array
      const data = new Uint8Array(n)
      for (let i = 0; i < n; i++) data[i] = src[indices[i] as number]!
      return { field: column.field, data, dictionary: dict }
    }
  }

  const src = column.data as Float64Array | Float32Array | Int32Array | Uint32Array | Uint8Array
  const data = allocateData(dtype, n) as typeof src
  let anyNull = false
  let nullBitmap: Uint8Array | undefined
  if (srcBitmap) nullBitmap = new Uint8Array(Math.ceil(n / 8) || 1)
  for (let i = 0; i < n; i++) {
    const srcIdx = indices[i] as number
    data[i] = src[srcIdx]!
    if (srcBitmap) {
      if (isValid(srcBitmap, srcIdx)) setValid(nullBitmap!, i, true)
      else anyNull = true
    }
  }
  return {
    field: { ...column.field },
    data,
    nullBitmap: anyNull ? nullBitmap : undefined,
    dictionary: dict,
  }
}

/** Gather every column; identity indices → shallow column reuse. */
export function takeTable(table: TableView, indices: ArrayLike<number>, keep?: readonly string[]): TableView {
  const cols =
    keep && keep.length > 0
      ? keep.map((name) => {
          const col = table.columns.find((c) => c.field.name === name)
          if (!col) throw new Error(`Unknown column "${name}" in takeTable`)
          return col
        })
      : table.columns

  const n = indices.length
  if (n === table.numRows && cols.length === table.columns.length) {
    let identity = true
    for (let i = 0; i < n; i++) {
      if ((indices[i] as number) !== i) {
        identity = false
        break
      }
    }
    if (identity) return table
  }
  return tableFromColumns(cols.map((c) => takeColumn(c, indices)))
}

/** Contiguous row slice; typed columns without nulls share via subarray. */
export function sliceColumn(column: Column, start: number, end: number): Column {
  const len = columnLength(column)
  const s = Math.max(0, Math.min(start, len))
  const e = Math.max(s, Math.min(end, len))
  if (s === 0 && e === len) return column

  const n = e - s
  const dtype = column.field.dtype
  const dict = column.dictionary

  if (dtype === 'utf8') {
    const src = column.data as string[]
    const data = src.slice(s, e)
    if (!column.nullBitmap) return { field: column.field, data, dictionary: dict }
    let anyNull = false
    const nullBitmap = new Uint8Array(Math.ceil(n / 8) || 1)
    for (let i = 0; i < n; i++) {
      if (isValid(column.nullBitmap, s + i)) setValid(nullBitmap, i, true)
      else anyNull = true
    }
    return {
      field: { ...column.field },
      data,
      nullBitmap: anyNull ? nullBitmap : undefined,
      dictionary: dict,
    }
  }

  const src = column.data as Float64Array | Float32Array | Int32Array | Uint32Array | Uint8Array
  const data = src.subarray(s, e) as typeof src
  if (!column.nullBitmap) return { field: column.field, data, dictionary: dict }

  let anyNull = false
  const nullBitmap = new Uint8Array(Math.ceil(n / 8) || 1)
  for (let i = 0; i < n; i++) {
    if (isValid(column.nullBitmap, s + i)) setValid(nullBitmap, i, true)
    else anyNull = true
  }
  return {
    field: { ...column.field },
    data,
    nullBitmap: anyNull ? nullBitmap : undefined,
    dictionary: dict,
  }
}

export function sliceTable(table: TableView, start: number, end: number): TableView {
  return tableFromColumns(table.columns.map((c) => sliceColumn(c, start, end)))
}

export function tableFromColumns(columns: Column[]): TableView {
  if (columns.length === 0) {
    return { schema: [], numRows: 0, columns: [] }
  }
  const numRows = columnLength(columns[0]!)
  for (const col of columns) {
    if (columnLength(col) !== numRows) {
      throw new Error(`Column length mismatch: ${col.field.name} has ${columnLength(col)}, expected ${numRows}`)
    }
  }
  return {
    schema: columns.map((c) => c.field),
    numRows,
    columns,
  }
}

export function columnLength(column: Column): number {
  return column.data.length
}

/** Bytes per numeric/bool element for size estimates (utf8/category handled separately). */
function dtypeElementBytes(dtype: DType): number {
  switch (dtype) {
    case 'f64':
    case 'datetime':
      return 8
    case 'f32':
    case 'i32':
    case 'u32':
      return 4
    case 'bool':
    case 'category':
      return 1
    case 'utf8':
      return 0
  }
}

/**
 * Rough live footprint of a column: typed buffer + null bitmap + dictionary/utf8 string payloads
 * (UTF-16 code units × 2, plus a small per-string overhead).
 */
export function estimateColumnBytes(column: Column): number {
  const n = columnLength(column)
  let bytes = 64
  if (column.nullBitmap) bytes += column.nullBitmap.byteLength
  if (column.field.dtype === 'utf8') {
    const data = column.data as string[]
    bytes += n * 24
    for (let i = 0; i < n; i++) {
      const s = data[i]
      if (s != null) bytes += s.length * 2
    }
  } else if (column.field.dtype === 'category') {
    bytes += n * dtypeElementBytes('category')
    if (column.dictionary) {
      bytes += column.dictionary.length * 24
      for (const s of column.dictionary) bytes += s.length * 2
    }
  } else {
    const data = column.data as ArrayBufferView
    bytes += data.byteLength || n * dtypeElementBytes(column.field.dtype)
  }
  return bytes
}

/** Rough live footprint of a table (sum of columns + schema overhead). */
export function estimateTableBytes(table: TableView): number {
  let bytes = 128 + table.schema.length * 48
  for (const col of table.columns) bytes += estimateColumnBytes(col)
  return bytes
}

export function cloneColumn(column: Column, indices?: number[]): Column {
  if (!indices) {
    return {
      field: { ...column.field },
      data: copyData(column.data),
      nullBitmap: column.nullBitmap ? new Uint8Array(column.nullBitmap) : undefined,
      dictionary: column.dictionary ? [...column.dictionary] : undefined,
    }
  }
  return takeColumn(column, indices)
}

export function getValue(data: TypedData, index: number): number | string | boolean {
  if (Array.isArray(data)) return data[index]!
  return data[index]!
}

export function setValue(data: TypedData, index: number, value: number | string | boolean, dtype: DType): void {
  if (dtype === 'utf8') {
    ;(data as string[])[index] = String(value)
    return
  }
  if (dtype === 'bool') {
    ;(data as Uint8Array)[index] = value ? 1 : 0
    return
  }
  const n = Number(value)
  if (dtype === 'i32') {
    if (!Number.isInteger(n) || n < -2147483648 || n > 2147483647) {
      throw new RangeError(`Cannot store ${String(value)} in i32 (need integer in Int32 range)`)
    }
    ;(data as Int32Array)[index] = n
    return
  }
  if (dtype === 'u32') {
    if (!Number.isInteger(n) || n < 0 || n > 4294967295) {
      throw new RangeError(`Cannot store ${String(value)} in u32 (need integer in Uint32 range)`)
    }
    ;(data as Uint32Array)[index] = n
    return
  }
  ;(data as Float64Array | Float32Array)[index] = n
}

/** Streaming dtype inference over a column accessor (avoids 256-row sample traps). */
export function inferDtypeFromValues(length: number, at: (i: number) => unknown): DType {
  let sawFloat = false
  let sawInt = false
  let sawBool = false
  let sawString = false
  let sawDate = false
  let intOutsideI32 = false
  for (let i = 0; i < length; i++) {
    const v = at(i)
    if (v === null || v === undefined) continue
    if (typeof v === 'boolean') {
      sawBool = true
      continue
    }
    if (typeof v === 'number') {
      if (!Number.isInteger(v)) sawFloat = true
      else {
        sawInt = true
        if (v > 2147483647 || v < -2147483648) intOutsideI32 = true
      }
      continue
    }
    if (v instanceof Date) {
      sawDate = true
      continue
    }
    sawString = true
  }
  if (sawString) return 'utf8'
  if (sawDate) return 'datetime'
  if (sawBool && !sawFloat && !sawInt) return 'bool'
  if (sawFloat || intOutsideI32) return 'f64'
  if (sawInt) return 'i32'
  if (sawBool) return 'bool'
  return 'f64'
}

export function inferDtype(values: unknown[]): DType {
  return inferDtypeFromValues(values.length, (i) => values[i])
}

export function copyData(data: TypedData): TypedData {
  if (Array.isArray(data)) return [...data]
  return data.slice() as TypedData
}

export function encodeCategory(values: Array<string | null | undefined>): {
  codes: Uint32Array
  dictionary: string[]
  nullBitmap?: Uint8Array
} {
  const dict: string[] = []
  const map = new Map<string, number>()
  const codes = new Uint32Array(values.length)
  let anyNull = false
  const nullBitmap = new Uint8Array(Math.ceil(values.length / 8) || 1)
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (v === null || v === undefined) {
      anyNull = true
      codes[i] = 0
      continue
    }
    setValid(nullBitmap, i, true)
    let code = map.get(v)
    if (code === undefined) {
      code = dict.length
      dict.push(v)
      map.set(v, code)
    }
    codes[i] = code
  }
  return { codes, dictionary: dict, nullBitmap: anyNull ? nullBitmap : undefined }
}

/**
 * Assign a field on a plain row object. A column named "__proto__" would otherwise rewrite the row's
 * prototype (or vanish when the value is a primitive) instead of becoming a property.
 */
export function setRowField(row: Record<string, unknown>, name: string, value: unknown): void {
  if (name === '__proto__') Object.defineProperty(row, name, { value, enumerable: true, writable: true, configurable: true })
  else row[name] = value
}

/** Read a cell from a row object: only own properties (missing `constructor` must not become Object). */
export function getRowField(row: Record<string, unknown>, name: string): unknown {
  if (!Object.hasOwn(row, name)) return undefined
  if (name === '__proto__') return Object.getOwnPropertyDescriptor(row, name)?.value
  return row[name]
}

export function toRowObjects(table: TableView): Record<string, unknown>[] {
  const rows: Record<string, unknown>[] = []
  for (let i = 0; i < table.numRows; i++) {
    const row: Record<string, unknown> = {}
    for (const col of table.columns) {
      const name = col.field.name
      if (!isValid(col.nullBitmap, i)) {
        setRowField(row, name, null)
        continue
      }
      const raw = getValue(col.data, i)
      if (col.field.dtype === 'bool') setRowField(row, name, Boolean(raw))
      else if (col.field.dtype === 'category' && col.dictionary) {
        setRowField(row, name, col.dictionary[Number(raw)] ?? null)
      } else if (col.field.dtype === 'datetime') {
        // Epoch milliseconds — matches InferColumns / DTypeValue, not Date objects.
        setRowField(row, name, Number(raw))
      } else setRowField(row, name, raw)
    }
    rows.push(row)
  }
  return rows
}

/** Minimal Arrow-like IPC JSON for interop (schema + columnar arrays). */
export interface ArrowLike {
  schema: Schema
  columns: Array<{
    name: string
    dtype: DType
    data: Array<number | string | boolean | null>
    dictionary?: string[]
  }>
}

export function toArrowLike(table: TableView): ArrowLike {
  return {
    schema: table.schema.map((f) => ({ ...f })),
    columns: table.columns.map((col) => {
      const data: Array<number | string | boolean | null> = []
      for (let i = 0; i < table.numRows; i++) {
        if (!isValid(col.nullBitmap, i)) {
          data.push(null)
          continue
        }
        const raw = getValue(col.data, i)
        if (col.field.dtype === 'bool') data.push(Boolean(raw))
        else if (col.field.dtype === 'category' && col.dictionary) data.push(col.dictionary[Number(raw)] ?? null)
        else data.push(raw as number | string)
      }
      return {
        name: col.field.name,
        dtype: col.field.dtype,
        data,
        dictionary: col.dictionary ? [...col.dictionary] : undefined,
      }
    }),
  }
}

export function fromArrowLike(arrow: ArrowLike): TableView {
  const columns: Column[] = arrow.columns.map((c, idx) => {
    const field = arrow.schema[idx] ?? { name: c.name, dtype: c.dtype, nullable: true }
    if (field.dtype === 'category' || (c.dictionary && c.dictionary.length)) {
      const encoded = encodeCategory(c.data.map((v) => (v === null ? null : String(v))))
      return {
        field: { ...field, dtype: 'category' },
        data: encoded.codes,
        nullBitmap: encoded.nullBitmap,
        dictionary: encoded.dictionary,
      }
    }
    const data = allocateData(field.dtype, c.data.length)
    let anyNull = false
    const nullBitmap = new Uint8Array(Math.ceil(c.data.length / 8) || 1)
    for (let i = 0; i < c.data.length; i++) {
      const v = c.data[i]
      if (v === null || v === undefined) {
        anyNull = true
        continue
      }
      setValid(nullBitmap, i, true)
      setValue(data, i, v as number | string | boolean, field.dtype)
    }
    return {
      field: { ...field },
      data,
      nullBitmap: anyNull ? nullBitmap : undefined,
    }
  })
  return tableFromColumns(columns)
}
