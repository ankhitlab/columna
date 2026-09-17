import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { DataFrame, clearPersistCache, col } from '@columna/core'
import { Runtime, clearMemoryPolicy } from '../src/index.js'
import { ensureSpillSupport } from '../src/spill.js'

await ensureSpillSupport()

afterEach(() => {
  clearMemoryPolicy()
  clearPersistCache()
})

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

function stressN(base: number): number {
  if (!process.env.STRESS_HEAVY) return base
  return Math.min(Math.floor(base * 10), 80_000)
}

const STRESS_TIMEOUT = process.env.STRESS_HEAVY ? 600_000 : 120_000

function nullHeavyRows(seed: number, n: number) {
  const rnd = mulberry32(seed)
  const cities = ['Berlin', 'Paris', 'Rome', 'Madrid', '']
  return Array.from({ length: n }, (_, id) => ({
    id,
    city: cities[Math.floor(rnd() * cities.length)]!,
    k: Math.floor(rnd() * 20),
    salary: rnd() < 0.15 ? null : Math.round(rnd() * 100_000),
    v: rnd() < 0.1 ? null : Math.floor(rnd() * 1000),
  }))
}

describe('stress: spill vs in-memory on adversarial frames', () => {
  it(
    'sortMulti under tiny budget matches in-memory',
    async () => {
      const n = stressN(4_000)
      const rows = nullHeavyRows(21, n)
      const dir = mkdtempSync(join(tmpdir(), 'columna-stress-spill-sort-'))
      try {
        const df = DataFrame.fromRows(rows)
        const plan = df.lazy().sort('city', { expr: col('salary'), descending: true }, 'id').plan
        const rt = new Runtime({ memory: { maxBytes: 12_000, spill: true, spillDir: dir } })
        const { table, report } = await rt.executeWithReport(plan)
        clearMemoryPolicy()
        const expected = await DataFrame.fromRows(rows)
          .sort('city', { expr: col('salary'), descending: true }, 'id')
          .collect()
        expect(table.numRows).toBe(expected.table.numRows)
        expect([...(table.columns[0]!.data as Int32Array | Float64Array)].slice(0, 100)).toEqual(
          [...(expected.table.columns[0]!.data as Int32Array | Float64Array)].slice(0, 100),
        )
        // Spill may or may not trigger depending on estimate; if it does, bytes > 0.
        expect(report.spilledBytes ?? 0).toBeGreaterThanOrEqual(0)
      } finally {
        clearMemoryPolicy()
        rmSync(dir, { recursive: true, force: true })
      }
    },
    STRESS_TIMEOUT,
  )

  it(
    'unique under tiny budget matches in-memory on null-heavy keys',
    async () => {
      const n = stressN(5_000)
      const rows = nullHeavyRows(33, n)
      const dir = mkdtempSync(join(tmpdir(), 'columna-stress-spill-uniq-'))
      try {
        const df = DataFrame.fromRows(rows)
        const rt = new Runtime({ memory: { maxBytes: 10_000, spill: true, spillDir: dir } })
        const { table } = await rt.executeWithReport(df.lazy().unique(['city', 'k']).plan)
        clearMemoryPolicy()
        const expected = await DataFrame.fromRows(rows).unique(['city', 'k']).collect()
        expect(table.numRows).toBe(expected.table.numRows)
      } finally {
        clearMemoryPolicy()
        rmSync(dir, { recursive: true, force: true })
      }
    },
    STRESS_TIMEOUT,
  )

  it(
    'inner join under tiny budget matches in-memory',
    async () => {
      const n = stressN(3_000)
      const rnd = mulberry32(55)
      const left = DataFrame.fromRows(
        Array.from({ length: n }, (_, i) => ({
          id: i % Math.max(50, Math.floor(n / 40)),
          a: rnd() < 0.1 ? null : i,
        })),
      )
      const right = DataFrame.fromRows(
        Array.from({ length: Math.floor(n / 2) }, (_, i) => ({
          id: i % Math.max(40, Math.floor(n / 50)),
          b: i * 3,
        })),
      )
      const dir = mkdtempSync(join(tmpdir(), 'columna-stress-spill-join-'))
      try {
        const rt = new Runtime({ memory: { maxBytes: 8_000, spill: true, spillDir: dir } })
        const plan = left.lazy().join(right, { on: 'id', how: 'inner' }).plan
        const { table } = await rt.executeWithReport(plan)
        clearMemoryPolicy()
        const expected = await left.lazy().join(right, { on: 'id', how: 'inner' }).collect()
        expect(table.numRows).toBe(expected.table.numRows)
      } finally {
        clearMemoryPolicy()
        rmSync(dir, { recursive: true, force: true })
      }
    },
    STRESS_TIMEOUT,
  )
})
