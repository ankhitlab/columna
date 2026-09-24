/**
 * 64-bit integers never come back silently rounded. Every source that can carry them — Arrow IPC (the
 * apache-arrow cases live in packages/arrow-interop), Parquet INT64, JSON integer literals, driver BigInts,
 * BigInt row values — goes through one Int64Policy: exact f64 when every value is within ±(2^53 − 1),
 * otherwise PrecisionLossError (default), exact decimal strings ('string') or the nearest double ('number').
 */
import { describe, expect, it } from 'vitest'
import { parquetWriteBuffer } from 'hyparquet-writer'
import { DataFrame, PrecisionLossError } from '../src/index.js'
import type { SqlClient } from '../src/io/sql/types.js'

const B = {
  p31: 2n ** 31n,
  p32: 2n ** 32n,
  safeMax: 2n ** 53n - 1n,
  p53: 2n ** 53n,
  p53p1: 2n ** 53n + 1n,
  i64max: 2n ** 63n - 1n,
  i64min: -(2n ** 63n),
  u64max: 2n ** 64n - 1n,
}

describe('BigInt rows (fromRows)', () => {
  it('safe BigInts become an exact numeric column (not text)', () => {
    const df = DataFrame.fromRows([{ a: B.p31 }, { a: B.p32 }, { a: B.safeMax }, { a: -B.safeMax }, { a: null }])
    expect(df.dtypes.a).toBe('f64')
    expect(df.getColumn('a').toArray()).toEqual([2 ** 31, 2 ** 32, Number.MAX_SAFE_INTEGER, -Number.MAX_SAFE_INTEGER, null])
  })
  for (const [name, v] of Object.entries({ p53: B.p53, p53p1: B.p53p1, i64max: B.i64max, i64min: B.i64min, u64max: B.u64max })) {
    it(`${name} = ${v}: PrecisionLossError by default, exact with 'string', nearest double with 'number'`, () => {
      const rows = [{ id: 1n }, { id: v }]
      let err: unknown
      try {
        DataFrame.fromRows(rows)
      } catch (e) {
        err = e
      }
      expect(err).toBeInstanceOf(PrecisionLossError)
      expect((err as PrecisionLossError).column).toBe('id')
      expect((err as PrecisionLossError).row).toBe(1)
      expect((err as PrecisionLossError).value).toBe(v.toString())
      const exact = DataFrame.fromRows(rows, { int64: 'string' })
      expect(exact.getColumn('id').toArray()).toEqual(['1', v.toString()])
      const lossy = DataFrame.fromRows(rows, { int64: 'number' })
      expect(lossy.getColumn('id').toArray()).toEqual([1, Number(v)])
    })
  }
})

describe('JSON integer literals', () => {
  it('JSON.parse would round 2^53 + 1; readers refuse by default and keep it exact with int64: "string"', () => {
    const text = '[{"id": 9007199254740993, "x": 1.5}, {"id": 12, "x": 2}]'
    expect(JSON.parse(text)[0].id).toBe(9007199254740992) // what the platform does
    expect(() => DataFrame.fromJSON(text)).toThrow(PrecisionLossError)
    const exact = DataFrame.fromJSON(text, { int64: 'string' })
    expect(exact.getColumn('id').toArray()).toEqual(['9007199254740993', '12'])
    expect(exact.getColumn('x').toArray()).toEqual([1.5, 2])
  })

  it('INT64 extremes, negative values, NDJSON; literals inside strings and floats are untouched', () => {
    const lines = [
      `{"a": ${B.i64max}, "s": "12345678901234567890", "f": 12345678901234567890.5}`,
      `{"a": ${B.i64min}, "s": "x", "f": 1e30}`,
    ].join('\n')
    const df = DataFrame.fromJSON(lines, { lines: true, int64: 'string' })
    expect(df.getColumn('a').toArray()).toEqual([B.i64max.toString(), B.i64min.toString()])
    expect(df.getColumn('s').toArray()).toEqual(['12345678901234567890', 'x'])
    expect(df.getColumn('f').toArray()).toEqual([12345678901234567890.5, 1e30])
  })

  it('texts with only safe integers (≤ 2^53 − 1) take the fast path and read as numbers', () => {
    const df = DataFrame.fromJSON(`[{"a": ${Number.MAX_SAFE_INTEGER}}, {"a": -${Number.MAX_SAFE_INTEGER}}]`)
    expect(df.dtypes.a).toBe('f64')
    expect(df.getColumn('a').toArray()).toEqual([Number.MAX_SAFE_INTEGER, -Number.MAX_SAFE_INTEGER])
  })
})

describe('Parquet INT64', () => {
  const bytes = new Uint8Array(
    parquetWriteBuffer({
      columnData: [
        { name: 'small', data: [1n, B.safeMax, null], type: 'INT64' },
        { name: 'id', data: [1n, B.p53p1, null], type: 'INT64' },
      ],
    }),
  )
  it('a column of safe values is exact f64; a column with 2^53 + 1 fails by default, naming it', async () => {
    const ok = await DataFrame.readParquet(bytes, { columns: ['small'] })
    expect(ok.getColumn('small').toArray()).toEqual([1, Number.MAX_SAFE_INTEGER, null])
    await expect(DataFrame.readParquet(bytes)).rejects.toThrow(/readParquet: column "id" row 1 holds 9007199254740993/)
  })
  it("int64: 'string' keeps every value of the column exact (previously a per-value number/string mix)", async () => {
    const df = await DataFrame.readParquet(bytes, { int64: 'string' })
    expect(df.getColumn('id').toArray()).toEqual(['1', '9007199254740993', null])
    expect(df.getColumn('small').toArray()).toEqual(['1', '9007199254740991', null])
  })
})

describe('SQL driver BigInts', () => {
  const client: SqlClient = { async query() { return [{ id: 1n, n: 2 }, { id: B.i64max, n: 3 }] } }
  it('refused by default, exact with int64: "string"', async () => {
    await expect(DataFrame.readSql('SELECT 1', client)).rejects.toThrow(PrecisionLossError)
    const df = await DataFrame.readSql('SELECT 1', client, { int64: 'string' })
    expect(df.getColumn('id').toArray()).toEqual(['1', B.i64max.toString()])
    expect(df.getColumn('n').toArray()).toEqual([2, 3])
  })
})
