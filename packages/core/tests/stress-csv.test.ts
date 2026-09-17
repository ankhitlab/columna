import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { mkdirSync, rmSync, writeFileSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { DataFrame } from '../src/dataframe.js'
import { parseCsvToTable, streamCsvFileToTable } from '../src/io/csv-columnar.js'
import {
  STRESS_TIMEOUT,
  adversarialRows,
  checksumNumeric,
  rowMultisetKey,
  stressN,
  unquotedSafeRows,
} from './stress-helpers.js'

const DIR = join(tmpdir(), `columna-stress-csv-${Date.now()}`)
beforeAll(() => mkdirSync(DIR, { recursive: true }))
afterAll(() => rmSync(DIR, { recursive: true, force: true }))

describe('stress: CSV round-trip and native parity', () => {
  it(
    'quoted adversarial CSV write→read preserves multiset checksums',
    async () => {
      const n = stressN(20_000)
      for (const seed of [3, 11]) {
        const rows = adversarialRows(seed, n)
        const df = DataFrame.fromRows(rows)
        const path = join(DIR, `adv-${seed}.csv`)
        await df.writeCsv(path)
        const back = await DataFrame.readCsv({ path })
        expect(back.shape[0]).toBe(n)
        // Column set may reorder slightly only if names collide — ours do not.
        expect(back.columns.sort()).toEqual([...df.columns].sort())
        const cols = ['id', 'k', 'city', 'seg', 'salary', 'age', 'x', 'y', 'flag', 'note']
        expect(rowMultisetKey(back.toArray(), cols)).toBe(rowMultisetKey(df.toArray(), cols))
      }
    },
    STRESS_TIMEOUT,
  )

  it(
    'unquoted-safe write→read matches; native parse matches JS when addon loaded',
    async () => {
      const n = stressN(40_000)
      const rows = unquotedSafeRows(9, n)
      const df = DataFrame.fromRows(rows)
      const path = join(DIR, 'safe.csv')
      await df.writeCsv(path)
      const back = await DataFrame.readCsv({ path })
      expect(back.shape).toEqual(df.shape)
      expect(checksumNumeric(back.toArray(), 'id')).toBe(checksumNumeric(rows, 'id'))
      expect(checksumNumeric(back.toArray(), 'salary')).toBeCloseTo(checksumNumeric(rows, 'salary'), 4)

      const { tryParseCsvNative } = await import('../src/io/csv-parallel.js')
      const native = await tryParseCsvNative(path, {})
      if (!native) return
      const ndf = new DataFrame(native)
      expect(ndf.shape).toEqual(df.shape)
      expect(checksumNumeric(ndf.toArray(), 'salary')).toBeCloseTo(checksumNumeric(rows, 'salary'), 4)
      expect(ndf.getColumn('city').toArray().slice(0, 50)).toEqual(back.getColumn('city').toArray().slice(0, 50))
    },
    STRESS_TIMEOUT,
  )

  it(
    'streamCsvFileToTable with nRows matches head of full parse; maxBytes stops early',
    async () => {
      const n = stressN(30_000)
      const rows = unquotedSafeRows(5, n)
      const path = join(DIR, 'stream.csv')
      writeFileSync(path, ['id,age,salary,x,y,city,seg', ...rows.map((r) => Object.values(r).join(','))].join('\n'))

      const headN = Math.min(5_000, n)
      const limited = await streamCsvFileToTable(path, { nRows: headN })
      expect(limited.numRows).toBe(headN)
      const full = parseCsvToTable(readFileSync(path, 'utf8'))
      expect(full.numRows).toBe(n)
      const limDf = new DataFrame(limited)
      const fullHead = await new DataFrame(full).head(headN).collect()
      expect(limDf.toArray()).toEqual(fullHead.toArray())

      await expect(streamCsvFileToTable(path, { maxBytes: 200 })).rejects.toThrow()
    },
    STRESS_TIMEOUT,
  )

  it(
    'ops then CSV round-trip keep filter checksum',
    async () => {
      const n = stressN(25_000)
      const df = DataFrame.fromRows(unquotedSafeRows(13, n))
      const filtered = await df.filter((c) => c.age.gt(30)).collect()
      const path = join(DIR, 'filtered-out.csv')
      await filtered.writeCsv(path)
      const back = await DataFrame.readCsv({ path })
      expect(back.shape[0]).toBe(filtered.shape[0])
      expect(checksumNumeric(back.toArray(), 'salary')).toBeCloseTo(checksumNumeric(filtered.toArray(), 'salary'), 4)
    },
    STRESS_TIMEOUT,
  )
})
