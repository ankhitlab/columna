import { describe, expect, it } from 'vitest'
import { DataFrame } from '../src/dataframe.js'
import { normalizeParquetRow } from '../src/io/parquet.js'
import { readSqlRows } from '../src/io/sql/index.js'
import { flattenObject, messageToRow } from '../src/io/kafka/index.js'
import { setRowField } from '@columna/arrow'

/** A row whose own "__proto__" key holds an object — the shape JSON.parse and drivers produce. */
const hostile = () => JSON.parse('{"__proto__": {"polluted": true}, "a": 1}') as Record<string, unknown>

function assertOrdinaryField(row: Record<string, unknown>, value: unknown) {
  expect(Object.getPrototypeOf(row)).toBe(Object.prototype) // prototype untouched
  expect(Object.hasOwn(row, '__proto__')).toBe(true) // field kept as an own property
  expect(Object.getOwnPropertyDescriptor(row, '__proto__')!.value).toEqual(value)
  expect(Object.keys(row)).toContain('__proto__')
  expect(({} as { polluted?: unknown }).polluted).toBeUndefined() // never global
}

describe('every row constructor treats "__proto__" as a column name', () => {
  it('setRowField is the single primitive: object values stay own properties', () => {
    const row: Record<string, unknown> = {}
    setRowField(row, '__proto__', { polluted: true })
    setRowField(row, 'a', 1)
    assertOrdinaryField(row, { polluted: true })
    // and the plain assignment it replaces really does rewrite the prototype
    const naive: Record<string, unknown> = {}
    naive['__proto__'] = { polluted: true }
    expect(Object.hasOwn(naive, '__proto__')).toBe(false)
    expect((naive as { polluted?: unknown }).polluted).toBe(true)
  })

  it('SQL: driver rows (duck-typed client) and the adapter normalizer', async () => {
    const rows = await readSqlRows('select 1', { query: async () => [hostile()] })
    assertOrdinaryField(rows[0]!, { polluted: true })
    const df = await DataFrame.readSql('select 1', { query: async () => [hostile()] })
    expect(df.columns).toEqual(['__proto__', 'a'])
  })

  it('Parquet: normalizeParquetRow', () => {
    const row = normalizeParquetRow({ ...hostile(), big: 12n, huge: 2n ** 60n })
    assertOrdinaryField(row, { polluted: true })
    // BigInts are kept as-is here; DataFrame.fromRows resolves them per column (tests/int64-precision.test.ts)
    expect(row.big).toBe(12n)
    expect(row.huge).toBe(2n ** 60n)
  })

  it('Kafka: flattened payload keys and header names', () => {
    const flat = flattenObject(hostile(), { separator: '.', maxDepth: 8, arrays: false })
    expect(Object.getPrototypeOf(flat)).toBe(Object.prototype)
    expect(Object.hasOwn(flat, '__proto__.polluted')).toBe(true)
    const nested = flattenObject(JSON.parse('{"__proto__": 5, "x": {"__proto__": 6}}'), { separator: '.', maxDepth: 8, arrays: false })
    assertOrdinaryField(nested, 5)
    expect(nested['x.__proto__']).toBe(6)
    const row = messageToRow({
      topic: 't',
      partition: 0,
      offset: '1',
      timestamp: 1,
      key: null,
      value: Buffer.from('{"__proto__": {"polluted": true}, "a": 1}'),
      headers: JSON.parse('{"__proto__":"h"}') as never, // an object literal would set the prototype instead
    } as never, { flatten: false as never, includeMeta: ['headers'] })
    expect(Object.getPrototypeOf(row)).toBe(Object.prototype)
    expect(JSON.parse(String(row._kafka_headers))).toEqual(JSON.parse('{"__proto__":"h"}'))
  })

  it('CSV dtype coercion and JSON orient=values keep the same guarantee', () => {
    const csv = DataFrame.fromCSV('__proto__,b\n7,2\n', { dtypes: JSON.parse('{"__proto__":"utf8"}') } as never).toArray()
    assertOrdinaryField(csv[0]!, '7')
    const json = DataFrame.fromJSON('[[1, 2]]', { orient: 'values' } as never).toArray()
    expect(json).toEqual([{ column_0: 1, column_1: 2 }])
  })
})
