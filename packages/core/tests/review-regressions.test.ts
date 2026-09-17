/**
 * Place at packages/core/tests/review-regressions.test.ts.
 * Review target: 0fa6d841229bbbdc71c7caccc2dac53ddc57281e.
 *
 * These tests specify the desired behavior, not the buggy behavior. Several are
 * expected to fail on the reviewed commit. They were prepared from the source
 * review but NOT executed against the full repository in the review environment.
 * GPU arithmetic is modeled with Math.fround; no GPU or external network is used.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { DataFrame, col } from '@columna/core'
import { executeCpu } from '@columna/runtime'
import { WebGpuBackend } from '@columna/webgpu'
import { getIoPolicy, loadText, setIoPolicy } from '../src/io/source.js'

const originalPolicy = getIoPolicy()

afterEach(() => {
  setIoPolicy(originalPolicy)
  vi.restoreAllMocks()
})

describe('review: preservation of values and row identity', () => {
  it('keeps a late integer larger than Int32 without wrapping (previous fix)', () => {
    const rows = Array.from({ length: 256 }, () => ({ x: 1 }))
    rows.push({ x: 2_147_483_648 })
    expect(DataFrame.fromRows(rows).toArray().at(-1)).toEqual({ x: 2_147_483_648 })
  })

  it('keeps a fractional value after the CSV integer-cache sample (previous fix)', () => {
    const x = Float64Array.from({ length: 4097 }, (_, i) => i === 4096 ? 0.5 : i % 2)
    expect(DataFrame.fromColumns({ x }).toCsv().split('\n').at(-1)).toBe('0.5')
  })

  it('unique must not drop a fractional value between integer endpoints', async () => {
    const df = DataFrame.fromColumns({ x: new Float64Array([0, 0.5, 1]) })
    const out = await df.unique(['x']).collect()
    expect(out.toArray()).toEqual([{ x: 0 }, { x: 0.5 }, { x: 1 }])
  })

  it('unique(keep=none) removes all three copies of null', async () => {
    const df = DataFrame.fromColumns({ x: [null, null, null] })
    expect((await df.unique(['x'], 'none').collect()).toArray()).toEqual([])
  })

  it('lowercase followed by unique compares values, not obsolete category codes', async () => {
    const out = await DataFrame.fromColumns({ s: ['A', 'a'] })
      .withColumn('s', col('s').str.toLowerCase())
      .unique(['s'])
      .collect()
    expect(out.toArray()).toEqual([{ s: 'a' }])
  })

  it('lowercase followed by valueCounts combines categories with equal labels', async () => {
    const out = await DataFrame.fromColumns({ s: ['A', 'a'] })
      .withColumn('s', col('s').str.toLowerCase())
      .valueCounts('s')
      .collect()
    expect(out.toArray()).toEqual([{ s: 'a', count: 2 }])
  })

  it('a missing constructor column is null, not inherited Object.prototype.constructor', () => {
    const df = DataFrame.fromRows<Record<string, unknown>>([{ constructor: 1 }, {}])
    expect(df.toArray()).toEqual([{ constructor: 1 }, { constructor: null }])
  })
})

describe('review: exact GPU mode', () => {
  it('an f32 source does not authorize a lossy arithmetic result', async () => {
    const query = DataFrame.fromColumns({ x: new Float32Array([16_777_216]) })
      .withColumn('y', col('x').add(1))
    const cpu = executeCpu(query.plan)
    expect(new DataFrame(cpu).toArray()[0]?.y).toBe(16_777_217)

    // This fake device is never touched: only the public arithmetic-kernel method
    // is replaced. The real backend dispatch and CPU fallback remain under test.
    const gpu = new WebGpuBackend(executeCpu, {} as never, { lossyF32: false })
    vi.spyOn(gpu, 'gpuMapTimed').mockImplementation(async (values, op, literal) => {
      if (op !== 0) throw new Error('This test models only the addition kernel')
      const roundedLiteral = Math.fround(literal)
      return {
        values: Float32Array.from(values, value => Math.fround(value + roundedLiteral)),
        transferMs: 0,
        computeMs: 0,
      }
    })
    const actual = await gpu.execute(query.plan)
    expect(new DataFrame(actual).toArray()).toEqual(new DataFrame(cpu).toArray())
    expect(actual.schema).toEqual(cpu.schema)
  })
})

describe('review: network policy rejects loopback without making a request', () => {
  it('denyPrivateHosts covers IPv4-mapped IPv6 addresses', async () => {
    setIoPolicy({})
    const fakeFetch = vi.fn(async () => new Response('x\n1', { status: 200 }))
    await expect(loadText(
      { url: 'http://[::ffff:127.0.0.1]/data.csv' },
      { denyPrivateHosts: true, fetch: fakeFetch },
    )).rejects.toThrow(/IO policy/)
    expect(fakeFetch).not.toHaveBeenCalled()
  })
})
