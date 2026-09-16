import { describe, expect, it } from 'vitest'
import { DataFrame, col, getDefaultRuntime, executeCpu } from 'columna'
import { WasmBackend, getWasmKernels, writeParquetLike, readParquetLike, wasmStringLength } from '@columna/wasm'
import { WebGpuBackend, detectWebGPU } from '@columna/webgpu'

describe('backends', () => {
  it('registers wasm and webgpu on default runtime', () => {
    const rt = getDefaultRuntime()
    rt.register(new WasmBackend(executeCpu))
    rt.register(new WebGpuBackend(executeCpu))
    const plan = DataFrame.fromRows([{ a: 1 }, { a: 2 }]).filter(col('a').gt(1)).plan
    const backend = rt.chooseBackend(plan)
    expect(['cpu', 'wasm', 'webgpu']).toContain(backend.name)
  })

  it('wasm kernels filter/sort', () => {
    const k = getWasmKernels()
    const values = new Float64Array([1, 5, 3, 8])
    const mask = k.filterMask(values, 2, 4)
    expect(Array.from(mask)).toEqual([0, 1, 0, 1])
    const idx = k.sortIndices(values, true)
    expect(Array.from(idx)).toEqual([3, 1, 2, 0])
  })

  it('wasm engine path matches cpu', async () => {
    const df = DataFrame.fromRows([
      { age: 10 },
      { age: 20 },
      { age: 30 },
    ])
    const cpu = await df.engine('cpu').filter(col('age').gt(15)).collect()
    const wasm = await df.engine('wasm').filter(col('age').gt(15)).collect()
    expect(wasm.toArray()).toEqual(cpu.toArray())
  })

  it('parquet-like roundtrip', () => {
    const df = DataFrame.fromRows([
      { a: 1, b: 'x' },
      { a: 2, b: 'y' },
    ])
    const bytes = writeParquetLike(df.table)
    const back = readParquetLike(bytes)
    expect(back.numRows).toBe(2)
    expect(back.schema.map((f) => f.name)).toEqual(['a', 'b'])
  })

  it('string length helper', () => {
    const { data } = wasmStringLength(['ab', 'cdef'])
    expect(Array.from(data)).toEqual([2, 4])
  })

  it('detectWebGPU returns null or device', async () => {
    const device = await detectWebGPU()
    expect(device === null || typeof device === 'object').toBe(true)
  })

  it('webgpu without device falls back for AND filter', async () => {
    const gpu = new WebGpuBackend(executeCpu, null)
    const df = DataFrame.fromRows([
      { age: 40, salary: 50_000, city: 'Berlin' },
      { age: 20, salary: 50_000, city: 'Paris' },
    ])
    const out = await gpu.execute(df.filter(col('age').gt(30).and(col('salary').gt(45_000))).plan)
    expect(out.numRows).toBe(1)
  })

  it('explain includes engine', () => {
    const text = DataFrame.fromRows([{ a: 1 }]).filter(col('a').eq(1)).explain()
    expect(text).toContain('Engine:')
    expect(text).toContain('Filter')
  })
})
