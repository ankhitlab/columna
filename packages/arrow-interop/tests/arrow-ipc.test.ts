/**
 * Apache Arrow IPC interop, pinned against the reference implementation (`apache-arrow`, dev-only):
 *  - what columna writes, apache-arrow reads back value-for-value (stream and file format, every dtype, nulls,
 *    dictionaries, several record batches);
 *  - what apache-arrow writes (types columna does not even have: Int8/Int64/UInt16, Float16, LargeUtf8,
 *    nanosecond timestamps, Date32, Null, dictionary with Int8 indices, multi-batch tables) columna reads;
 *  - unsupported Arrow types are refused by column name instead of being decoded wrongly.
 */
import { describe, expect, it } from 'vitest'
import {
  Bool,
  DateDay,
  Dictionary,
  Field,
  Float16,
  Float32,
  Float64,
  Int8,
  Int32,
  Int64,
  LargeUtf8,
  List,
  Null,
  RecordBatchStreamWriter,
  Schema,
  Table,
  TimestampMillisecond,
  TimestampNanosecond,
  Uint16,
  Uint32,
  Uint64,
  Utf8,
  makeData,
  makeVector,
  tableFromIPC,
  tableToIPC,
  vectorFromArray,
} from 'apache-arrow'
import { DataFrame, PrecisionLossError } from '@columna/core'

const rows = [
  { f: 1.5, i: -3, u: 7, b: true, s: 'alpha', c: 'x', d: Date.UTC(2024, 0, 1) },
  { f: null, i: 0, u: 0, b: false, s: '', c: 'y', d: null },
  { f: -2.25, i: 2147483647, u: 4294967295, b: null, s: 'знак — ünïcode 🚀', c: null, d: Date.UTC(1969, 11, 31, 23, 59, 59, 999) },
  { f: NaN, i: null, u: null, b: true, s: null, c: 'x', d: Date.UTC(2100, 5, 15, 12) },
]

// explicit dtypes (fromRows/fromColumns would fold the short string column into a category)
function frame(): DataFrame {
  const dtypes = { f: 'f64', i: 'i32', u: 'u32', b: 'bool', s: 'utf8', c: 'category', d: 'datetime' } as const
  return DataFrame.fromArrowLike({
    schema: Object.entries(dtypes).map(([name, dtype]) => ({ name, dtype, nullable: true })),
    columns: Object.entries(dtypes).map(([name, dtype]) => ({
      name,
      dtype,
      data: rows.map((r) => (r as Record<string, number | string | boolean | null>)[name] ?? null),
    })),
  })
}

const cell = (v: unknown): unknown => {
  if (v === null || v === undefined) return null
  if (typeof v === 'bigint') return Number(v)
  if (v instanceof Date) return v.getTime()
  return v
}

describe('toArrowIpc → apache-arrow', () => {
  it('stream format: schema, values, nulls and dictionary survive', () => {
    const df = frame()
    const bytes = df.toArrowIpc()
    const table = tableFromIPC(bytes)
    expect(table.numRows).toBe(4)
    expect(table.schema.fields.map((f) => f.name)).toEqual(['f', 'i', 'u', 'b', 's', 'c', 'd'])
    expect(table.schema.fields.map((f) => f.type.toString())).toEqual([
      'Float64',
      'Int32',
      'Uint32',
      'Bool',
      'Utf8',
      'Dictionary<Int32, Utf8>',
      'Timestamp<MILLISECOND>',
    ])
    const got = table.toArray().map((r) => Object.fromEntries(Object.entries(r.toJSON()).map(([k, v]) => [k, cell(v)])))
    const want = df.toArray()
    for (let i = 0; i < 4; i++) {
      for (const k of Object.keys(want[i]!)) {
        const w = (want[i] as Record<string, unknown>)[k]
        const g = got[i]![k]
        if (typeof w === 'number' && Number.isNaN(w)) expect(Number.isNaN(g as number)).toBe(true)
        else expect(g).toEqual(w)
      }
    }
  })

  it('file format (Feather v2) with several record batches reads back identically', () => {
    const n = 10_007
    const df = DataFrame.fromColumns({
      x: Float64Array.from({ length: n }, (_, i) => i / 3),
      k: Int32Array.from({ length: n }, (_, i) => (i * 7919) % 1000),
      t: Array.from({ length: n }, (_, i) => `t${i % 13}`),
    })
    const bytes = df.toArrowIpc({ format: 'file', batchRows: 4000 })
    expect(new TextDecoder().decode(bytes.subarray(0, 6))).toBe('ARROW1')
    expect(new TextDecoder().decode(bytes.subarray(bytes.length - 6))).toBe('ARROW1')
    const table = tableFromIPC(bytes)
    expect(table.numRows).toBe(n)
    expect(table.batches.length).toBe(3)
    const x = table.getChild('x')!.toArray() as Float64Array
    const k = table.getChild('k')!.toArray() as Int32Array
    const t = table.getChild('t')!
    for (const i of [0, 1, 3999, 4000, 8000, n - 1]) {
      expect(x[i]).toBe(i / 3)
      expect(k[i]).toBe((i * 7919) % 1000)
      expect(t.get(i)).toBe(`t${i % 13}`)
    }
    // and columna reads its own file format back, batches concatenated
    const back = DataFrame.fromArrowIpc(bytes)
    expect(back.shape).toEqual([n, 3])
    expect(back.dtypes).toEqual(df.dtypes)
    expect(back.getColumn('t').toArray().slice(0, 15)).toEqual(df.getColumn('t').toArray().slice(0, 15))
  })

  it('empty frame and all-null column are valid IPC', () => {
    const empty = DataFrame.fromColumns({ a: new Float64Array(0), b: [] as string[] })
    expect(tableFromIPC(empty.toArrowIpc()).numRows).toBe(0)
    const nulls = DataFrame.fromRows([{ a: null }, { a: null }])
    const t = tableFromIPC(nulls.toArrowIpc())
    expect(t.numRows).toBe(2)
    expect(t.getChild('a')!.nullCount).toBe(2)
  })
})

describe('apache-arrow → fromArrowIpc', () => {
  it('reads Arrow types columna itself never writes, mapped without silent loss', () => {
    const n = 5
    const table = new Table({
      i8: vectorFromArray([1, -2, 3, null, 127], new Int8()),
      u16: vectorFromArray([0, 65535, 1, 2, null], new Uint16()),
      i64: vectorFromArray([0n, 9007199254740991n, -5n, null, 42n], new Int64()),
      f16: vectorFromArray([0.5, -1.5, 1024, null, 0.25], new Float16()),
      f32: vectorFromArray([0.5, -1.5, 3.25, null, 0], new Float32()),
      big: vectorFromArray(['x', 'yy', null, '', 'zzz'], new LargeUtf8()),
      // apache-arrow's nanosecond builder takes epoch milliseconds and scales ×1e6 itself
      ns: vectorFromArray([1_700_000_000_123, 0, -1, null, 5], new TimestampNanosecond()),
      // DateDay builder takes epoch milliseconds too (floor to days)
      day: vectorFromArray([0, 86_400_000, 19723 * 86_400_000, null, -86_400_000], new DateDay()),
      nothing: vectorFromArray([null, null, null, null, null], new Null()),
      dict: vectorFromArray(['a', 'b', 'a', null, 'c'], new Dictionary(new Utf8(), new Int8())),
    })
    expect(table.numRows).toBe(n)
    const df = DataFrame.fromArrowIpc(tableToIPC(table, 'stream'))
    expect(df.shape).toEqual([n, 10])
    expect(df.dtypes).toEqual({
      i8: 'i32',
      u16: 'i32',
      i64: 'f64',
      f16: 'f32',
      f32: 'f32',
      big: 'utf8',
      ns: 'datetime',
      day: 'datetime',
      nothing: 'f64',
      dict: 'category',
    })
    expect(df.getColumn('i8').toArray()).toEqual([1, -2, 3, null, 127])
    expect(df.getColumn('u16').toArray()).toEqual([0, 65535, 1, 2, null])
    expect(df.getColumn('i64').toArray()).toEqual([0, 9007199254740991, -5, null, 42])
    expect(df.getColumn('f16').toArray()).toEqual([0.5, -1.5, 1024, null, 0.25])
    expect(df.getColumn('big').toArray()).toEqual(['x', 'yy', null, '', 'zzz'])
    expect(df.getColumn('ns').toArray()).toEqual([1_700_000_000_123, 0, -1, null, 5])
    expect(df.getColumn('day').toArray()).toEqual([0, 86_400_000, 19723 * 86_400_000, null, -86_400_000])
    expect(df.getColumn('nothing').toArray()).toEqual([null, null, null, null, null])
    expect(df.getColumn('dict').toArray()).toEqual(['a', 'b', 'a', null, 'c'])
  })

  it('reads a multi-batch stream written by RecordBatchStreamWriter and the file format', () => {
    const a = new Table({ v: vectorFromArray([1, 2, 3], new Int32()), s: vectorFromArray(['a', 'b', 'c'], new Utf8()) })
    const b = new Table({ v: vectorFromArray([4, 5], new Int32()), s: vectorFromArray(['d', null], new Utf8()) })
    const both = a.concat(b)
    expect(both.batches.length).toBe(2)
    const stream = tableToIPC(both, 'stream')
    const df = DataFrame.fromArrowIpc(stream)
    expect(df.getColumn('v').toArray()).toEqual([1, 2, 3, 4, 5])
    expect(df.getColumn('s').toArray()).toEqual(['a', 'b', 'c', 'd', null])
    const file = tableToIPC(both, 'file')
    expect(DataFrame.fromArrowIpc(file).toArray()).toEqual(df.toArray())
    const viaWriter = RecordBatchStreamWriter.writeAll(both).toUint8Array(true)
    expect(DataFrame.fromArrowIpc(viaWriter).shape).toEqual([5, 2])
  })

  it('accepts an ArrayBuffer and a Uint8Array view with a byte offset', () => {
    const bytes = frame().toArrowIpc()
    const padded = new Uint8Array(bytes.length + 16)
    padded.set(bytes, 16)
    const view = new Uint8Array(padded.buffer, 16, bytes.length)
    expect(DataFrame.fromArrowIpc(view).toArray()).toEqual(DataFrame.fromArrowIpc(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)).toArray())
  })

  it('refuses nested and other unmapped types by column name', () => {
    const list = makeVector(
      makeData({
        type: new List(new Field('item', new Int32())),
        length: 2,
        nullCount: 0,
        valueOffsets: Int32Array.from([0, 2, 3]),
        child: makeData({ type: new Int32(), data: Int32Array.from([1, 2, 3]) }),
      }),
    )
    const table = new Table({ ok: vectorFromArray([1, 2], new Int32()), nested: list })
    expect(() => DataFrame.fromArrowIpc(tableToIPC(table))).toThrow(/column "nested" \(List\)/)
    expect(() => DataFrame.fromArrowIpc(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))).toThrow(/fromArrowIpc/)
  })
})

describe('round trip through columna only', () => {
  it('toArrowIpc ∘ fromArrowIpc is the identity on schema and values', () => {
    const df = frame()
    const back = DataFrame.fromArrowIpc(df.toArrowIpc())
    expect(back.dtypes).toEqual(df.dtypes)
    const a = df.toArray()
    const b = back.toArray()
    expect(b.length).toBe(a.length)
    for (let i = 0; i < a.length; i++) {
      for (const k of Object.keys(a[i]!)) {
        const w = (a[i] as Record<string, unknown>)[k]
        const g = (b[i] as Record<string, unknown>)[k]
        if (typeof w === 'number' && Number.isNaN(w)) expect(Number.isNaN(g as number)).toBe(true)
        else expect(g).toEqual(w)
      }
    }
  })

  it('unused schema / type surface: Schema and Bool imports are exercised by the reference writer', () => {
    const schema = new Schema([new Field('flag', new Bool(), true), new Field('n', new Uint32(), false)])
    const t = new Table(schema, [])
    expect(DataFrame.fromArrowIpc(tableToIPC(t)).dtypes).toEqual({ flag: 'bool', n: 'u32' })
    expect(TimestampMillisecond.name).toBeTruthy()
  })
})

describe('Polars (Rust Arrow) fixtures — independent of apache-arrow JS', () => {
  // written by tests/fixtures/make_polars_ipc_fixture.py (Polars 1.44.1): Utf8View strings, Categorical and
  // Enum dictionaries, Datetime(us), Date32, Int8, UInt64, nullable booleans; stream and file format
  const want = [
    { f: 1.5, i8: 1, u64: 0, s: 'alpha', c: 'x', e: 'lo', d: Date.UTC(2024, 0, 1), day: Date.UTC(2024, 0, 1), b: true },
    { f: null, i8: -2, u64: 9007199254740991, s: '', c: 'y', e: 'hi', d: null, day: null, b: null },
    { f: -2.25, i8: null, u64: null, s: 'a very long string that does not fit in twelve bytes 🚀', c: null, e: 'lo', d: -1, day: 86_400_000, b: false },
    { f: 3, i8: 4, u64: 7, s: null, c: 'x', e: null, d: Date.UTC(2100, 5, 15, 12), day: -86_400_000, b: true },
  ]
  for (const name of ['polars-ipc.arrows', 'polars-ipc.arrow']) {
    it(`reads ${name}`, async () => {
      const { readFileSync } = await import('node:fs')
      const { fileURLToPath } = await import('node:url')
      const bytes = readFileSync(fileURLToPath(new URL(`./fixtures/${name}`, import.meta.url)))
      const df = DataFrame.fromArrowIpc(bytes)
      expect(df.dtypes).toEqual({ f: 'f64', i8: 'i32', u64: 'f64', s: 'utf8', c: 'category', e: 'category', d: 'datetime', day: 'datetime', b: 'bool' })
      expect(df.toArray()).toEqual(want)
    })
  }
})

describe('seeded random frames through apache-arrow and back', () => {
  function mulberry32(seed: number) {
    let a = seed >>> 0
    return () => {
      a = (a + 0x6d2b79f5) >>> 0
      let t = a
      t = Math.imul(t ^ (t >>> 15), t | 1)
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296
    }
  }
  const randomRows = (seed: number, n: number) => {
    const rnd = mulberry32(seed)
    const groups = ['a', 'b', 'c', '', 'ünï', 'x"y']
    return Array.from({ length: n }, (_, id) => ({
      id,
      k: Math.floor(rnd() * 7),
      g: groups[Math.floor(rnd() * groups.length)]!,
      s: rnd() < 0.3 ? 'dup' : `s${Math.floor(rnd() * 5000)}`,
      f: rnd() < 0.15 ? null : Math.round((rnd() - 0.5) * 1e6) / 1000,
      i: rnd() < 0.1 ? null : Math.floor((rnd() - 0.5) * 2 ** 32),
      b: rnd() < 0.5,
      'we ird,name': rnd(),
    }))
  }
  for (const seed of [1, 2, 3]) {
    it(`seed ${seed}: our writer → apache-arrow → its file writer → our reader is the identity`, () => {
      const rows = randomRows(seed, 1500)
      const df = DataFrame.fromRows(rows)
      const back = DataFrame.fromArrowIpc(tableToIPC(tableFromIPC(df.toArrowIpc({ batchRows: 400 })), 'file'))
      expect(back.toArray()).toEqual(rows)
      expect(back.dtypes).toEqual(df.dtypes)
    })
  }
})

describe('Int64 / UInt64 boundary values (apache-arrow writer → fromArrowIpc)', () => {
  const B = { p31: 2n ** 31n, p32: 2n ** 32n, safeMax: 2n ** 53n - 1n, p53: 2n ** 53n, p53p1: 2n ** 53n + 1n, i64max: 2n ** 63n - 1n, i64min: -(2n ** 63n), u64max: 2n ** 64n - 1n }
  const ipc = (name: string, values: Array<bigint | null>, type: Int64 | Uint64) => tableToIPC(new Table({ [name]: vectorFromArray(values, type) }))

  it('values within ±(2^53 − 1) — including 2^31, 2^32 and the extremes — read as an exact f64 column', () => {
    const df = DataFrame.fromArrowIpc(ipc('a', [B.p31, B.p32, B.safeMax, -B.safeMax, null], new Int64()))
    expect(df.dtypes.a).toBe('f64')
    expect(df.getColumn('a').toArray()).toEqual([2 ** 31, 2 ** 32, Number.MAX_SAFE_INTEGER, -Number.MAX_SAFE_INTEGER, null])
    const u = DataFrame.fromArrowIpc(ipc('u', [0n, B.p32, B.safeMax], new Uint64()))
    expect(u.getColumn('u').toArray()).toEqual([0, 2 ** 32, Number.MAX_SAFE_INTEGER])
  })

  const unsafe: Array<[string, bigint, Int64 | Uint64]> = [
    ['2^53', B.p53, new Int64()],
    ['2^53 + 1', B.p53p1, new Int64()],
    ['INT64_MAX', B.i64max, new Int64()],
    ['INT64_MIN', B.i64min, new Int64()],
    ['UINT64_MAX', B.u64max, new Uint64()],
  ]
  for (const [label, v, type] of unsafe) {
    it(`${label}: PrecisionLossError by default (column, row, exact value); 'string' exact; 'number' explicit nearest double`, () => {
      const bytes = ipc('id', [1n, v, null], type)
      expect(() => DataFrame.fromArrowIpc(bytes)).toThrow(PrecisionLossError)
      try {
        DataFrame.fromArrowIpc(bytes)
      } catch (e) {
        expect(e).toMatchObject({ column: 'id', row: 1, value: v.toString() })
      }
      expect(DataFrame.fromArrowIpc(bytes, { int64: 'string' }).getColumn('id').toArray()).toEqual(['1', v.toString(), null])
      expect(DataFrame.fromArrowIpc(bytes, { int64: 'number' }).getColumn('id').toArray()).toEqual([1, Number(v), null])
    })
  }

  it('the row index of the error counts across record batches', () => {
    const a = new Table({ id: vectorFromArray([1n, 2n], new Int64()) })
    const b = new Table({ id: vectorFromArray([3n, B.p53p1], new Int64()) })
    try {
      DataFrame.fromArrowIpc(tableToIPC(a.concat(b)))
      expect.unreachable()
    } catch (e) {
      expect(e).toMatchObject({ row: 3 })
    }
  })

  it('nanosecond timestamps beyond 2^53 ns convert to ms without rounding the integer part first', () => {
    // 2023-11-14T22:13:20.123456789Z in ns = 1_700_000_000_123_456_789 (> 2^53)
    const ns = 1_700_000_000_123_456_789n
    const t = makeVector(makeData({ type: new TimestampNanosecond(), data: BigInt64Array.from([ns, -1n]) }))
    const df = DataFrame.fromArrowIpc(tableToIPC(new Table({ t })))
    const [ms, neg] = df.getColumn('t').toArray() as number[]
    expect(Math.abs(ms! - 1_700_000_000_123.456789)).toBeLessThan(1e-3) // exact to the double's resolution (~0.2 µs)
    expect(Math.floor(ms!)).toBe(1_700_000_000_123)
    expect(neg).toBe(-1e-6)
  })
})
