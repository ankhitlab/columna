import { describe, expect, it, vi, afterEach } from 'vitest'
import { mkdtempSync, rmSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { WebGpuBackend } from '@columna/webgpu'
import { DataFrame, col, executeCpu } from 'columna'
import { getColumn, getRowField } from '@columna/arrow'

const csvStreamProbe = vi.hoisted(() => ({
  active: false,
  maxErrorListeners: 0,
}))

vi.mock('node:fs', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:fs')>()
  const realCreateWriteStream = actual.createWriteStream
  return {
    ...actual,
    createWriteStream(
      ...args: Parameters<typeof realCreateWriteStream>
    ): ReturnType<typeof realCreateWriteStream> {
      const stream = realCreateWriteStream(...args)
      if (!csvStreamProbe.active) return stream
      const bump = () => {
        csvStreamProbe.maxErrorListeners = Math.max(
          csvStreamProbe.maxErrorListeners,
          stream.listenerCount('error'),
        )
      }
      const origWrite = stream.write.bind(stream)
      stream.write = ((chunk: unknown, encoding?: unknown, cb?: unknown) => {
        bump()
        return (origWrite as (a: unknown, b?: unknown, c?: unknown) => boolean)(chunk, encoding, cb)
      }) as typeof stream.write
      stream.on('drain', bump)
      return stream
    },
  }
})

describe('GPU map exact mode', () => {
  it('f32 arith with lossyF32 false matches CPU values and f64 dtype', async () => {
    const x = new Float32Array([16_777_216])
    const df = DataFrame.fromColumns({ x })
    const cpu = await executeCpu(df.withColumn('y', col('x').add(1)).plan)
    expect(getColumn(cpu, 'y').field.dtype).toBe('f64')
    expect(Number((getColumn(cpu, 'y').data as Float64Array)[0])).toBe(16_777_217)

    const gpu = new WebGpuBackend(executeCpu, null, { lossyF32: false })
    const out = await gpu.execute(df.withColumn('y', col('x').add(1)).plan)
    expect(getColumn(out, 'y').field.dtype).toBe('f64')
    expect(Number((getColumn(out, 'y').data as Float64Array)[0])).toBe(16_777_217)
  })
})

describe('fromRows own-property reads', () => {
  it('getRowField ignores prototype constructor', () => {
    const plain: Record<string, unknown> = { a: 1 }
    expect(getRowField(plain, 'constructor')).toBeUndefined()
    expect(plain['constructor']).toBe(Object.prototype.constructor)
  })

  it('missing constructor own field becomes null in fromRows', () => {
    const plain: Record<string, unknown> = { a: 1 }
    const framed = DataFrame.fromRows([
      { a: 1, constructor: 'fn' },
      plain as { a: number; constructor?: string },
    ])
    const rows = framed.toArray()
    expect(rows[0]).toEqual({ a: 1, constructor: 'fn' })
    expect(rows[1]).toEqual({ a: 1, constructor: null })
  })
})

describe('datetime toArray is epoch ms', () => {
  it('returns number not Date', () => {
    const ms = Date.UTC(2024, 0, 15)
    const df = DataFrame.fromRows([{ ts: new Date(ms) }])
    const v = df.toArray()[0]!.ts
    expect(typeof v).toBe('number')
    expect(v).toBe(ms)
  })
})

describe('writeCsvText stream error listeners', () => {
  afterEach(() => {
    csvStreamProbe.active = false
    csvStreamProbe.maxErrorListeners = 0
  })

  it('registers a single persistent error listener, not one per chunk', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'columna-csv-'))
    const path = join(dir, 'out.csv')
    csvStreamProbe.active = true
    csvStreamProbe.maxErrorListeners = 0

    try {
      const rows = Array.from({ length: 80_000 }, (_, i) => ({ v: i }))
      await DataFrame.fromRows(rows).writeCsv(path)
      expect(csvStreamProbe.maxErrorListeners).toBeLessThanOrEqual(2)
      expect(readFileSync(path, 'utf8').split('\n').length).toBeGreaterThan(1000)
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })
})
