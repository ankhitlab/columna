import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { mkdirSync, rmSync, writeFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { DataFrame } from '../src/dataframe.js'
import { CsvRecordSplitter, parseCsvToTable, streamCsvFileToTable } from '../src/io/csv-columnar.js'

const DIR = join(tmpdir(), `columna-csvcol-${Date.now()}`)
beforeAll(() => mkdirSync(DIR, { recursive: true }))
afterAll(() => rmSync(DIR, { recursive: true, force: true }))

const tricky = [
  'id,name,score,flag,note',
  '1,"Smith, J",1.5,true,"multi',
  'line"',
  '2,Ada,2,false,',
  '3,"He said ""hi""",3.25,TRUE,NA',
  '# comment',
  '',
  '4,Bob,4,,x',
  '5,Eve,2147483648,false,y',
  '6,Zoe,abc,true,z',
].join('\r\n')

describe('columnar CSV path equals the row-object path', () => {
  it('same table for quoted newlines, escaped quotes, comments, blanks, nulls, widening', () => {
    const opts = { comment: '#' }
    const a = DataFrame.fromCSV(tricky, opts)
    const b = DataFrame.fromCSVRows(tricky, opts)
    expect(a.columns).toEqual(b.columns)
    expect(a.toArray()).toEqual(b.toArray())
    expect(a.getColumn('score').dtype).toBe('category') // "abc" widened the numeric column to text
    expect(a.getColumn('id').dtype).toBe('i32')
    expect(a.toArray()[0]).toMatchObject({ name: 'Smith, J', note: 'multi\r\nline' })
    expect(a.toArray()[2]).toMatchObject({ name: 'He said "hi"', note: null })
  })

  it('every reader option behaves the same on both paths', () => {
    const text = 'x;y;z\n 1; 2;a\n3;4;b\n5;6;c\n7;8;d\n'
    const variants: Array<Record<string, unknown>> = [
      { separator: ';' },
      { separator: ';', skipInitialSpace: true },
      { separator: ';', nRows: 2 },
      { separator: ';', usecols: ['z', 'x'] },
      { separator: ';', usecols: [2] },
      { separator: ';', header: false },
      { separator: ';', names: ['a', 'b', 'c'] },
      { separator: ';', skipRows: 1, header: false },
      { separator: ';', header: 1 },
      { separator: ';', dtypes: { x: 'utf8', y: 'f64' } },
      { separator: ';', dtypes: { x: 'bool' } },
      { separator: ';', nullValues: ['a'] },
      { separator: ';', trueValues: ['a'], falseValues: ['b'] },
      { separator: ';', decimal: ',', thousands: '.' },
    ]
    for (const v of variants) {
      const a = DataFrame.fromCSV(text, v as never)
      const b = DataFrame.fromCSVRows(text, v as never)
      expect(a.columns, JSON.stringify(v)).toEqual(b.columns)
      expect(a.toArray(), JSON.stringify(v)).toEqual(b.toArray())
    }
  })

  it('record splitter is chunk-boundary safe: CRLF and quoted newlines split anywhere', () => {
    const text = 'a,b\r\n"x\r\ny",1\r\n"p,q",2\n3,"z"\r\n'
    const expected: string[] = []
    new CsvRecordSplitter().push(text, (r) => expected.push(r))
    for (let cut = 1; cut < text.length; cut++) {
      const got: string[] = []
      const s = new CsvRecordSplitter()
      s.push(text.slice(0, cut), (r) => got.push(r))
      s.push(text.slice(cut), (r) => got.push(r))
      s.end((r) => got.push(r))
      const ref: string[] = [...expected]
      new CsvRecordSplitter().end((r) => ref.push(r))
      expect(got, `cut at ${cut}`).toEqual(['a,b', '"x\r\ny",1', '"p,q",2', '3,"z"'])
    }
    // three-way split through a quoted CRLF
    const got: string[] = []
    const s = new CsvRecordSplitter()
    for (const part of ['a,b\r\n"x\r', '\ny",1\r', '\n"p,q",2']) s.push(part, (r) => got.push(r))
    s.end((r) => got.push(r))
    expect(got).toEqual(['a,b', '"x\r\ny",1', '"p,q",2'])
  })

  it('streams a file in chunks: same result as in-memory, nRows stops early, maxBytes is enforced on bytes read', async () => {
    const n = 200_000
    const lines = ['id,v,s']
    for (let i = 0; i < n; i++) lines.push(`${i},${(i % 7) / 4},"s${i % 5}"`)
    lines.push(`${n},2147483648,end`)
    const path = join(DIR, 'big.csv')
    writeFileSync(path, lines.join('\n'))
    const size = statSync(path).size
    expect(size).toBeGreaterThan(2 * (1 << 20)) // several 1 MB chunks
    const streamed = await DataFrame.readCsv(path)
    const inMemory = new DataFrame(parseCsvToTable(lines.join('\n')))
    expect(streamed.shape).toEqual([n + 1, 3])
    expect(streamed.getColumn('v').dtype).toBe('f64') // 2^31 arrived in the last chunk → widened, not wrapped
    expect(streamed.toArray().at(-1)).toEqual({ id: n, v: 2147483648, s: 'end' })
    expect(streamed.getColumn('id').toArray()).toEqual(inMemory.getColumn('id').toArray())
    expect(streamed.getColumn('s').toArray()).toEqual(inMemory.getColumn('s').toArray())

    const head = await DataFrame.readCsv(path, { nRows: 10 })
    expect(head.shape).toEqual([10, 3])
    const t = await streamCsvFileToTable(path, { nRows: 3, maxBytes: 1 << 20 })
    expect(t.numRows).toBe(3) // stopped inside the first chunk, before the cap could matter
    await expect(DataFrame.readCsv(path, { maxBytes: 1000 })).rejects.toThrow(/maxBytes/)
    await expect(DataFrame.readCsv(path, { allowedDirs: [join(DIR, 'elsewhere')] })).rejects.toThrow(/allowedDirs/)
  })

  it('fused unquoted path matches row path on spaced text, nulls, bools, and wide ints', () => {
    const text = [
      'a,b,c,d,e',
      '1,  hi  ,true,NA,2147483648',
      '2,x,FALSE,null,3.25',
      '3,,True,,',
    ].join('\n')
    const a = DataFrame.fromCSV(text)
    const b = DataFrame.fromCSVRows(text)
    expect(a.columns).toEqual(b.columns)
    expect(a.toArray()).toEqual(b.toArray())
    expect(a.getColumn('e').dtype).toBe('f64')
    expect(a.toArray()[0]).toEqual({ a: 1, b: '  hi  ', c: true, d: null, e: 2147483648 })
  })

  it('native CSV parse matches JS fused path when addon is loaded', async () => {
    const { tryParseCsvNative } = await import('../src/io/csv-parallel.js')
    const n = 5_000
    const lines = ['id,age,salary,city']
    for (let i = 0; i < n; i++) lines.push(`${i},${20 + (i % 40)},${(1000 + i * 0.25).toFixed(2)},${['A', 'B', 'C'][i % 3]}`)
    const path = join(DIR, 'native.csv')
    writeFileSync(path, lines.join('\n'))
    const js = await DataFrame.readCsv(path)
    const native = await tryParseCsvNative(path, {})
    if (!native) return // addon optional in CI without rebuild
    const ndf = new DataFrame(native)
    expect(ndf.shape).toEqual(js.shape)
    expect(ndf.columns).toEqual(js.columns)
    expect(ndf.getColumn('salary').toArray().slice(0, 20)).toEqual(js.getColumn('salary').toArray().slice(0, 20))
    expect(ndf.getColumn('city').toArray().slice(0, 20)).toEqual(js.getColumn('city').toArray().slice(0, 20))
  })

  it('native CSV write round-trips unquoted tables when addon is loaded', async () => {
    const n = 20_000
    const df = DataFrame.fromColumns({
      i: Int32Array.from({ length: n }, (_, k) => k),
      f: Float64Array.from({ length: n }, (_, k) => k + 0.25),
      city: Array.from({ length: n }, (_, k) => ['Berlin', 'Paris', 'Rome'][k % 3]!),
    })
    const path = join(DIR, 'native-out.csv')
    const ret = await df.writeCsv(path)
    expect(ret).toBe('')
    const back = await DataFrame.readCsv(path)
    expect(back.shape).toEqual(df.shape)
    expect(back.getColumn('i').toArray().slice(0, 10)).toEqual(df.getColumn('i').toArray().slice(0, 10))
    expect(back.getColumn('city').toArray().slice(0, 10)).toEqual(df.getColumn('city').toArray().slice(0, 10))
    // Floats: allow tiny formatter differences between JS String() and native ryu.
    const a = back.getColumn('f').toArray() as number[]
    const b = df.getColumn('f').toArray() as number[]
    for (let i = 0; i < 50; i++) expect(Math.abs(a[i]! - b[i]!)).toBeLessThan(1e-9)
  })
})

describe('CSV writer streams to disk in bounded chunks', () => {
  it('writeCsv(path) produces byte-identical output to toCsv() across several chunks, and resolves to ""', async () => {
    const n = 50_000 // > 3 × CSV_CHUNK_ROWS
    const df = DataFrame.fromColumns({
      i: Int32Array.from({ length: n }, (_, k) => k),
      f: Float64Array.from({ length: n }, (_, k) => (k % 9 === 0 ? k + 0.5 : k % 100)),
      s: Array.from({ length: n }, (_, k) => (k % 3 ? `s${k % 7}` : 'quote"me')),
    })
    const path = join(DIR, 'out.csv')
    const ret = await df.writeCsv(path)
    expect(ret).toBe('')
    const { readFileSync } = await import('node:fs')
    const onDisk = readFileSync(path, 'utf8')
    expect(onDisk).toBe(df.toCsv())
    expect(onDisk.split('\n').length).toBe(n + 1)
    const back = await DataFrame.readCsv(path)
    expect(back.shape).toEqual([n, 3])
    expect(back.getColumn('f').toArray().slice(0, 10)).toEqual(df.getColumn('f').toArray().slice(0, 10))
    expect(back.toArray()[3]).toEqual({ i: 3, f: 3, s: 'quote"me' })
  })
})
