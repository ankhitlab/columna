import { mkdirSync, writeFileSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { describe, expect, it, beforeAll, afterAll } from 'vitest'
import * as XLSX from 'xlsx'
import { DataFrame } from '../src/dataframe.js'

const DIR = join(tmpdir(), `columna-io-${Date.now()}`)

beforeAll(() => {
  mkdirSync(DIR, { recursive: true })

  writeFileSync(
    join(DIR, 'sample.csv'),
    ['city,age,active', 'Berlin,30,true', '# comment', 'Paris,41,false', '', 'Berlin,22,TRUE'].join('\n'),
    'utf8',
  )

  writeFileSync(
    join(DIR, 'sample.json'),
    JSON.stringify([
      { city: 'Berlin', age: 30 },
      { city: 'Paris', age: 41 },
    ]),
    'utf8',
  )

  writeFileSync(
    join(DIR, 'sample.jsonl'),
    ['{"city":"Berlin","age":30}', '{"city":"Paris","age":41}', '{"city":"Rome","age":28}'].join('\n'),
    'utf8',
  )

  const wb = XLSX.utils.book_new()
  const ws = XLSX.utils.aoa_to_sheet([
    ['city', 'age'],
    ['Berlin', 30],
    ['Paris', 41],
  ])
  XLSX.utils.book_append_sheet(wb, ws, 'people')
  // xlsx 0.20 ESM build has no fs binding by default (XLSX.set_fs); write the bytes ourselves
  writeFileSync(join(DIR, 'sample.xlsx'), XLSX.write(wb, { type: 'buffer', bookType: 'xlsx' }) as Buffer)
})

afterAll(() => {
  rmSync(DIR, { recursive: true, force: true })
})

describe('DataFrame IO readers', () => {
  it('fromCSV supports rich options', () => {
    const df = DataFrame.fromCSV('name;score\nAda;1,5\nBob;2,0\n', {
      separator: ';',
      decimal: ',',
      nRows: 1,
    })
    expect(df.shape[0]).toBe(1)
    expect(df.toArray()[0]).toEqual({ name: 'Ada', score: 1.5 })
  })

  it('readCsv from path with comment/skip/usecols', async () => {
    const df = await DataFrame.readCsv(join(DIR, 'sample.csv'), {
      comment: '#',
      skipBlankLines: true,
      usecols: ['city', 'age'],
      trueValues: ['true', 'TRUE'],
    })
    expect(df.columns).toEqual(['city', 'age'])
    expect(df.shape[0]).toBe(3)
    expect(df.toArray().map((r) => r.city)).toEqual(['Berlin', 'Paris', 'Berlin'])
  })

  it('readJson from path', async () => {
    const df = await DataFrame.readJson(join(DIR, 'sample.json'))
    expect(df.shape).toEqual([2, 2])
  })

  it('readJson NDJSON with nRows', async () => {
    const df = await DataFrame.readJson(join(DIR, 'sample.jsonl'), { lines: true, nRows: 2 })
    expect(df.shape[0]).toBe(2)
    expect(df.toArray()[1]).toEqual({ city: 'Paris', age: 41 })
  })

  it('readJson orient=columns', () => {
    const df = DataFrame.fromJSON(
      {
        city: ['Berlin', 'Paris'],
        age: [30, 41],
      },
      { orient: 'columns' },
    )
    expect(df.toArray()).toEqual([
      { city: 'Berlin', age: 30 },
      { city: 'Paris', age: 41 },
    ])
  })

  it('readExcel from path', async () => {
    const df = await DataFrame.readExcel(join(DIR, 'sample.xlsx'), { sheet: 'people' })
    expect(df.shape).toEqual([2, 2])
    expect(df.toArray()[0]).toMatchObject({ city: 'Berlin', age: 30 })
  })

  it('readParquet from path with column projection', async () => {
    const file = join(process.cwd(), 'packages/core/tests/fixtures/sample.parquet')
    const df = await DataFrame.readParquet(file, { columns: ['city', 'age'] })
    expect(df.shape[0]).toBe(2)
    expect(df.columns.sort()).toEqual(['age', 'city'])
    expect(df.toArray()[0]).toMatchObject({ city: 'Berlin', age: 30 })
  })

  it('readCsv rejects missing path', async () => {
    await expect(DataFrame.readCsv(join(DIR, 'missing.csv'))).rejects.toThrow(/not found/i)
  })
})
