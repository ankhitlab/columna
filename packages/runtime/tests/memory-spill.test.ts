import { mkdtempSync, readdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { estimateTableBytes, tableFromColumns } from '@columna/arrow'
import { DataFrame, clearPersistCache } from '@columna/core'
import { Runtime, clearMemoryPolicy, setMemoryPolicy } from '../src/index.js'
import { resetExecMemoryStats } from '../src/memory.js'
import { ensureSpillSupport, spillRead, spillUnlink, spillWrite } from '../src/spill.js'

// direct spill calls (outside Runtime.execute) must initialise Node support first on Node < 20.16
await ensureSpillSupport()

afterEach(() => {
  clearMemoryPolicy()
  clearPersistCache()
})

describe('estimateTableBytes', () => {
  it('scales with rows and columns', () => {
    const small = DataFrame.fromRows([
      { a: 1, b: 2 },
      { a: 3, b: 4 },
    ]).table
    const big = DataFrame.fromRows(Array.from({ length: 1000 }, (_, i) => ({ a: i, b: i * 2 }))).table
    expect(estimateTableBytes(big)).toBeGreaterThan(estimateTableBytes(small) * 10)
  })
})

describe('spill read/write', () => {
  it('round-trips numeric and utf8 columns', () => {
    const table = tableFromColumns([
      {
        field: { name: 'x', dtype: 'f64', nullable: true },
        data: new Float64Array([1.5, 2.5, 0]),
        nullBitmap: (() => {
          const b = new Uint8Array(1)
          b[0] = 0b011 // rows 0,1 valid
          return b
        })(),
      },
      {
        field: { name: 's', dtype: 'utf8', nullable: false },
        data: ['hello', 'world', 'nullish'],
      },
    ])
    const dir = mkdtempSync(join(tmpdir(), 'columna-spill-test-'))
    const path = join(dir, 't.cspill')
    try {
      spillWrite(table, path)
      const back = spillRead(path)
      expect(back.numRows).toBe(3)
      expect(back.schema.map((f) => f.name)).toEqual(['x', 's'])
      expect([...(back.columns[0]!.data as Float64Array)].slice(0, 2)).toEqual([1.5, 2.5])
      expect(back.columns[1]!.data).toEqual(['hello', 'world', 'nullish'])
    } finally {
      spillUnlink(path)
      rmSync(dir, { recursive: true, force: true })
    }
  })
})

describe('MemoryPolicy spill sort/unique/join', () => {
  it('sort under tiny budget matches in-memory sort and reports spilledBytes', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'columna-spill-sort-'))
    const rows = Array.from({ length: 500 }, (_, i) => ({ k: (500 - i) % 97, v: i }))
    try {
      setMemoryPolicy({ maxBytes: 8_000, spill: true, spillDir: dir })
      resetExecMemoryStats()
      const rt = new Runtime({ memory: { maxBytes: 8_000, spill: true, spillDir: dir } })
      const df = DataFrame.fromRows(rows)
      const { table, report } = await rt.executeWithReport(df.lazy().sort('k').plan)
      clearMemoryPolicy()
      const expected = await DataFrame.fromRows(rows).sort('k').collect()
      expect([...(table.columns[0]!.data as Float64Array)]).toEqual([
        ...(expected.table.columns[0]!.data as Float64Array),
      ])
      expect(table.numRows).toBe(500)
      expect(report.spilledBytes ?? 0).toBeGreaterThan(0)
      expect(readdirSync(dir).filter((f) => f.endsWith('.cspill'))).toEqual([])
    } finally {
      clearMemoryPolicy()
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('unique under tiny budget matches full unique', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'columna-spill-uniq-'))
    const rows = Array.from({ length: 400 }, (_, i) => ({ k: i % 40, v: i }))
    try {
      const rt = new Runtime({ memory: { maxBytes: 6_000, spill: true, spillDir: dir } })
      const df = DataFrame.fromRows(rows)
      const { table, report } = await rt.executeWithReport(df.lazy().unique(['k']).plan)
      clearMemoryPolicy()
      const expected = await DataFrame.fromRows(rows).unique(['k']).collect()
      expect(table.numRows).toBe(expected.table.numRows)
      expect(report.spilledBytes ?? 0).toBeGreaterThan(0)
    } finally {
      clearMemoryPolicy()
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('inner join under tiny budget matches', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'columna-spill-join-'))
    const left = DataFrame.fromRows(Array.from({ length: 200 }, (_, i) => ({ id: i % 50, a: i })))
    const right = DataFrame.fromRows(Array.from({ length: 80 }, (_, i) => ({ id: i % 40, b: i * 10 })))
    try {
      const rt = new Runtime({ memory: { maxBytes: 4_000, spill: true, spillDir: dir } })
      const plan = left.lazy().join(right, { on: 'id', how: 'inner' }).plan
      const { table, report } = await rt.executeWithReport(plan)
      clearMemoryPolicy()
      const expected = await left.lazy().join(right, { on: 'id', how: 'inner' }).collect()
      expect(table.numRows).toBe(expected.table.numRows)
      expect(report.spilledBytes ?? 0).toBeGreaterThan(0)
    } finally {
      clearMemoryPolicy()
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('chunked groupBy under tiny budget matches full groupBy', async () => {
    const rows = Array.from({ length: 800 }, (_, i) => ({ g: i % 20, x: i }))
    const df = DataFrame.fromRows(rows)
    const rt = new Runtime({ memory: { maxBytes: 5_000, spill: true } })
    const { table } = await rt.executeWithReport(
      df.lazy().groupBy('g').agg({ x: 'sum' }).plan,
    )
    clearMemoryPolicy()
    const expected = await DataFrame.fromRows(rows).groupBy('g').agg({ x: 'sum' }).collect()
    expect(table.numRows).toBe(expected.table.numRows)
    const got = new Map<number, number>()
    const exp = new Map<number, number>()
    const gCol = table.columns[0]!.data as Float64Array | Int32Array
    const sCol = table.columns[1]!.data as Float64Array
    for (let i = 0; i < table.numRows; i++) got.set(Number(gCol[i]), sCol[i]!)
    const eg = expected.table.columns[0]!.data as Float64Array | Int32Array
    const es = expected.table.columns[1]!.data as Float64Array
    for (let i = 0; i < expected.table.numRows; i++) exp.set(Number(eg[i]), es[i]!)
    expect([...got.entries()].sort((a, b) => a[0] - b[0])).toEqual(
      [...exp.entries()].sort((a, b) => a[0] - b[0]),
    )
  })
})
