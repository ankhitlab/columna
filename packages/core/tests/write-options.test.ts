import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { DataFrame } from '../src/dataframe.js'

const dirs: string[] = []
afterEach(() => {
  while (dirs.length) {
    const d = dirs.pop()
    if (d) rmSync(d, { recursive: true, force: true })
  }
})

function tmp(): string {
  const d = mkdtempSync(join(tmpdir(), 'columna-write-'))
  dirs.push(d)
  return d
}

describe('CsvWriteOptions', () => {
  const df = DataFrame.fromRows([
    { a: 1.5, b: 'x,y', c: null as number | null },
    { a: 2, b: 'z', c: 3 },
  ])

  it('defaults match prior comma / header / empty-null behaviour', () => {
    expect(df.toCsv()).toBe('a,b,c\n1.5,"x,y",\n2,z,3')
  })

  it('separator, header, naRep, decimal, columns, lineTerminator', () => {
    const csv = df.toCsv({
      separator: ';',
      header: false,
      naRep: 'NA',
      decimal: ',',
      columns: ['a', 'c'],
      lineTerminator: '\r\n',
    })
    expect(csv).toBe('1,5;NA\r\n2;3')
  })

  it('delimiter alias and quoteChar', () => {
    const csv = DataFrame.fromRows([{ s: 'a|b' }]).toCsv({ delimiter: '|', quoteChar: "'" })
    expect(csv).toBe("s\n'a|b'")
  })

  it('round-trips with fromCSV when options match', () => {
    const text = df.toCsv({ separator: ';', naRep: 'NA' })
    const back = DataFrame.fromCSV(text, { separator: ';', nullValues: ['NA'] })
    expect(back.toArray()).toEqual([
      { a: 1.5, b: 'x,y', c: null },
      { a: 2, b: 'z', c: 3 },
    ])
  })

  it('writeCsv encoding utf16le', async () => {
    const path = join(tmp(), 'out.csv')
    await df.writeCsv(path, { encoding: 'utf16le', columns: ['a'] })
    const buf = readFileSync(path)
    expect(buf.toString('utf16le')).toBe(df.toCsv({ columns: ['a'] }))
  })

  it('rejects empty separator', () => {
    expect(() => df.toCsv({ separator: '' })).toThrow(/separator/)
  })
})

describe('JsonWriteOptions', () => {
  const df = DataFrame.fromRows([
    { a: 1, b: 'x' },
    { a: 2, b: 'y' },
  ])

  it('records / columns / values / lines', () => {
    expect(JSON.parse(df.toJson())).toEqual([
      { a: 1, b: 'x' },
      { a: 2, b: 'y' },
    ])
    expect(JSON.parse(df.toJson({ orient: 'columns' }))).toEqual({ a: [1, 2], b: ['x', 'y'] })
    expect(JSON.parse(df.toJson({ orient: 'values' }))).toEqual([
      [1, 'x'],
      [2, 'y'],
    ])
    expect(df.toJson({ lines: true })).toBe('{"a":1,"b":"x"}\n{"a":2,"b":"y"}\n')
  })

  it('round-trips lines with fromJSON', () => {
    const text = df.toJson({ lines: true })
    expect(DataFrame.fromJSON(text, { lines: true }).toArray()).toEqual(df.toArray())
  })

  it('writeJson to path', async () => {
    const path = join(tmp(), 'out.json')
    const ret = await df.writeJson(path, { columns: ['a'] })
    expect(ret).toBe('')
    expect(JSON.parse(readFileSync(path, 'utf8'))).toEqual([{ a: 1 }, { a: 2 }])
  })
})

describe('ParquetWriteOptions', () => {
  const df = DataFrame.fromRows([
    { a: 1, b: 'x' },
    { a: 2, b: 'y' },
  ])

  it('round-trips with codec and columns', async () => {
    const bytes = await df.writeParquet(undefined, { codec: 'UNCOMPRESSED', columns: ['a'] })
    const back = await DataFrame.readParquet(bytes)
    expect(back.columns).toEqual(['a'])
    expect(back.toArray()).toEqual([{ a: 1 }, { a: 2 }])
  })

  it('GZIP codec round-trips', async () => {
    const bytes = await df.writeParquet(undefined, { codec: 'GZIP' })
    const back = await DataFrame.readParquet(bytes)
    expect(back.toArray()).toEqual(df.toArray())
  })

  it('parquet-like columns', async () => {
    const bytes = await df.writeParquetLike(undefined, { columns: ['b'] })
    const payload = JSON.parse(new TextDecoder().decode(bytes)) as {
      schema: Array<{ name: string }>
      columns: Array<{ name: string }>
    }
    expect(payload.schema.map((f) => f.name)).toEqual(['b'])
    expect(payload.columns.map((c) => c.name)).toEqual(['b'])
  })
})

describe('ExcelWriteOptions', () => {
  it('writeExcel → readExcel round-trip', async () => {
    const df = DataFrame.fromRows([
      { name: 'Ada', n: 1 },
      { name: 'Bob', n: 2 },
    ])
    const path = join(tmp(), 'out.xlsx')
    await df.writeExcel(path, { sheet: 'people', columns: ['name', 'n'] })
    const back = await DataFrame.readExcel(path, { sheet: 'people' })
    expect(back.toArray()).toEqual(df.toArray())
  })
})

describe('present / Arrow / toBlob options', () => {
  const df = DataFrame.fromRows([
    { a: 1, b: null as string | null },
    { a: 2, b: 'x' },
  ])

  it('toMarkdown / toHTML accept object options', () => {
    expect(df.toMarkdown({ columns: ['b'], naRep: 'NA', maxRows: 10 })).toContain('| NA |')
    expect(df.toHTML({ columns: ['a'], maxRows: 1 })).toContain('<td>1</td>')
    expect(df.toHTML({ columns: ['a'], maxRows: 1 })).not.toContain('<td>2</td>')
  })

  it('toArrowIpc columns', () => {
    const bytes = df.toArrowIpc({ columns: ['a'] })
    const back = DataFrame.fromArrowIpc(bytes)
    expect(back.columns).toEqual(['a'])
  })

  it('toBlob csv / json / parquet-like', async () => {
    expect(await new Response(df.toBlob('csv', { separator: ';' })).text()).toContain('a;b')
    expect(JSON.parse(await new Response(df.toBlob('json')).text())).toHaveLength(2)
    const like = df.toBlob('parquet-like', { columns: ['a'] })
    expect(like.type).toContain('json')
    const payload = JSON.parse(await new Response(like).text()) as { columns: Array<{ name: string }> }
    expect(payload.columns.map((c) => c.name)).toEqual(['a'])
  })

  it('writeParquet feeds a parquet Blob round-trip', async () => {
    const bytes = await df.writeParquet(undefined, { codec: 'SNAPPY' })
    const blob = new Blob([bytes], { type: 'application/vnd.apache.parquet' })
    const back = await DataFrame.readParquet(new Uint8Array(await blob.arrayBuffer()))
    expect(back.shape[0]).toBe(2)
  })
})
