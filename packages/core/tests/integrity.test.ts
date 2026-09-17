import { describe, expect, it } from 'vitest'
import { DataFrame } from '../src/dataframe.js'
import { tableToCsv } from '../src/io/write.js'

describe('data integrity', () => {
  it('fromRows infers f64 when a late row is fractional', () => {
    const rows = Array.from({ length: 300 }, (_, i) => ({ v: i < 256 ? i : 3.7 }))
    const df = DataFrame.fromRows(rows)
    expect(df.getColumn('v').dtype).toBe('f64')
    expect(df.getColumn('v').toArray()[256]).toBe(3.7)
  })

  it('fromRows uses f64 for integers outside Int32', () => {
    const df = DataFrame.fromRows([{ v: 3_000_000_000 }])
    expect(df.getColumn('v').dtype).toBe('f64')
    expect(df.getColumn('v').toArray()[0]).toBe(3_000_000_000)
  })

  it('cast to i32 rejects overflow and non-integers', async () => {
    const { col } = await import('../src/expr.js')
    await expect(
      DataFrame.fromRows([{ v: 3_000_000_000 }]).withColumn('v', col('v').cast('i32')).collect(),
    ).rejects.toThrow(/i32/)
    await expect(DataFrame.fromRows([{ v: 1.5 }]).withColumn('v', col('v').cast('i32')).collect()).rejects.toThrow(
      /i32/,
    )
  })

  it('CSV write preserves fractional values even with dense int cache', () => {
    const rows = Array.from({ length: 5000 }, (_, i) => ({ v: i === 4999 ? 100.5 : i % 200 }))
    const df = DataFrame.fromRows(rows)
    const csv = tableToCsv(df.table)
    const last = csv.trim().split('\n').at(-1)
    expect(last).toBe('100.5')
  })

  it('CSV writer: values after the 4096-row sample bypass the dense integer cache (Float64Array via fromColumns)', () => {
    // The sample sees only 0 / 1 and builds a 2-entry string cache; the late values must not index into it.
    const n = 4096 + 6
    const v = new Float64Array(n)
    for (let i = 0; i < 4096; i++) v[i] = i % 2
    v[4096] = 0.5
    v[4097] = -0.25
    v[4098] = 1e21
    v[4099] = NaN
    v[4100] = -1
    v[4101] = 5
    const csv = tableToCsv(DataFrame.fromColumns({ v }).table)
    const lines = csv.split('\n')
    expect(lines[0]).toBe('v')
    expect(lines[4096]).toBe('1') // last sampled row: cache hit
    expect(lines.slice(4097)).toEqual(['0.5', '-0.25', '1e+21', 'NaN', '-1', '5'])
    // same guarantee through the public API and with a null bitmap in the column
    const withNull = new Float64Array(4098)
    for (let i = 0; i < 4096; i++) withNull[i] = i % 3
    withNull[4096] = 2.75
    withNull[4097] = 0
    const df = DataFrame.fromColumns({ w: Array.from(withNull, (x, i) => (i === 4097 ? null : x)) })
    const tail = df.toCsv().split('\n').slice(-2)
    expect(tail).toEqual(['2.75', ''])
  })

  it('CSV escapeFormulas: text cells that a spreadsheet would execute are neutralised, numbers untouched', () => {
    const df = DataFrame.fromRows([
      { name: '=HYPERLINK("http://evil","click")', note: '+1', tag: '-cmd', at: '@SUM(A1)', tab: '\tx', cr: '\rx', n: -5, ok: 'plain' },
      { name: 'Ada', note: 'a=b', tag: ' -x', at: '', tab: 'x\ty', cr: 'y', n: 2, ok: '=' },
    ])
    // default: data written as-is (no silent alteration)
    expect(df.toCsv().split('\n')[1]).toBe('"=HYPERLINK(""http://evil"",""click"")",+1,-cmd,@SUM(A1),\tx,"\rx",-5,plain')
    const safe = df.toCsv({ escapeFormulas: true }).split('\n')
    expect(safe[1]).toBe('"\'=HYPERLINK(""http://evil"",""click"")","\'+1","\'-cmd","\'@SUM(A1)","\'\tx","\'\rx",-5,plain')
    // only a *leading* trigger matters; inner characters and numbers are left alone
    expect(safe[2]).toBe('Ada,a=b, -x,"",x\ty,y,2,"\'="') // an empty string is written as "" (a null would be an empty field)
    // header names are cells too
    const h = DataFrame.fromRows([{ '=cmd': 1 }]).toCsv({ escapeFormulas: true }).split('\n')[0]
    expect(h).toBe('"\'=cmd"')
    // category (dictionary) columns take the same path
    const cat = DataFrame.fromColumns({ c: ['=1', '=1', 'b'] })
    expect(cat.toCsv({ escapeFormulas: true }).split('\n').slice(1)).toEqual(['"\'=1"', '"\'=1"', 'b'])
    expect(cat.toBlob('csv', { escapeFormulas: true }).size).toBeGreaterThan(cat.toBlob('csv').size)
  })

  it('writeParquet refuses the false Apache Parquet claim', async () => {
    const df = DataFrame.fromRows([{ a: 1 }])
    await expect(df.writeParquet()).rejects.toThrow(/does not write Apache Parquet/)
  })

  it('writeParquetLike round-trips with readParquetLike payload shape', async () => {
    const df = DataFrame.fromRows([{ a: 1, b: 'x' }])
    const bytes = await df.writeParquetLike()
    const text = new TextDecoder().decode(bytes)
    const json = JSON.parse(text) as { format: string; numRows: number }
    expect(json.format).toBe('columna-parquet-like-v1')
    expect(json.numRows).toBe(1)
  })

  it('fromRows: a value past the sample window widens the column instead of wrapping / truncating', () => {
    // Int32 overflow after 256 small integers (reported repro: used to give -2147483648)
    const big = Array.from({ length: 256 }, () => ({ x: 1 }))
    big.push({ x: 2147483648 })
    const bigDf = DataFrame.fromRows(big)
    expect(bigDf.getColumn('x').dtype).toBe('f64')
    expect(bigDf.toArray().at(-1)).toEqual({ x: 2147483648 })

    // late fractional → f64, not 1
    const frac = Array.from({ length: 256 }, () => ({ x: 1 }))
    frac.push({ x: 1.5 })
    expect(DataFrame.fromRows(frac).toArray().at(-1)).toEqual({ x: 1.5 })

    // late string → utf8 column, numbers keep their text form (not 0)
    const str = Array.from({ length: 256 }, () => ({ x: 1 }))
    str.push({ x: 'n/a' })
    const strDf = DataFrame.fromRows(str)
    expect(strDf.getColumn('x').dtype).toBe('category')
    expect(strDf.toArray().at(-1)).toEqual({ x: 'n/a' })
    expect(strDf.toArray()[0]).toEqual({ x: '1' })

    // late boolean among integers widens to a numeric column (true → 1), late Date among numbers → utf8
    const mixed = Array.from({ length: 300 }, (_, i) => ({ b: i < 299 ? i : true }))
    expect(DataFrame.fromRows(mixed).toArray().at(-1)).toEqual({ b: 1 })
  })

  it('fromRows: schema is the union of all rows, missing keys read as null', () => {
    const df = DataFrame.fromRows([{ a: 1 }, { a: 2, b: 3 }, { c: 'x' }])
    expect(df.columns).toEqual(['a', 'b', 'c'])
    expect(df.toArray()).toEqual([
      { a: 1, b: null, c: null },
      { a: 2, b: 3, c: null },
      { a: null, b: null, c: 'x' },
    ])
    expect(df.getColumn('b').dtype).toBe('i32')
  })

  it('CSV / JSON readers inherit both guarantees (they go through fromRows)', () => {
    const lines = ['x']
    for (let i = 0; i < 300; i++) lines.push('1')
    lines.push('2147483648', '2.5')
    const csv = DataFrame.fromCSV(lines.join('\n'))
    expect(csv.getColumn('x').dtype).toBe('f64')
    expect(csv.toArray().slice(-2).map((r) => r.x)).toEqual([2147483648, 2.5])

    const json = DataFrame.fromJSON('[{"a":1},{"a":2,"b":"late"}]')
    expect(json.columns).toEqual(['a', 'b'])
    expect(json.toArray()[0]).toEqual({ a: 1, b: null })
  })
})
