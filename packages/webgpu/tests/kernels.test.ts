import { describe, expect, it } from 'vitest'
import {
  flattenAnd,
  matchNumericAndFilter,
  matchNumericMap,
  maskToIndices,
  planHasGpuKernel,
  columnToF32,
  WebGpuBackend,
} from '@columna/webgpu'
import { DataFrame, col, executeCpu } from 'columna'
import type { Column } from '@columna/arrow'

describe('webgpu predicate matching', () => {
  it('flattens AND trees', () => {
    const pred = col('age').gt(30).and(col('salary').gt(45_000)).node
    const parts = flattenAnd(pred)
    expect(parts).toHaveLength(2)
  })

  it('matches numeric AND filter used in benches', () => {
    const pred = col('age').gt(30).and(col('salary').gt(45_000)).node
    const cmps = matchNumericAndFilter(pred)
    expect(cmps).toEqual([
      { column: 'age', op: 'gt', literal: 30 },
      { column: 'salary', op: 'gt', literal: 45_000 },
    ])
  })

  it('rejects non-numeric filter leaves', () => {
    const pred = col('city').eq('Berlin').and(col('age').gt(10)).node
    expect(matchNumericAndFilter(pred)).toBeNull()
  })

  it('matches numeric map expressions', () => {
    expect(matchNumericMap(col('salary').mul(1.1).node)).toEqual({
      column: 'salary',
      op: 'mul',
      literal: 1.1,
    })
  })

  it('detects GPU kernels inside pipelines', () => {
    const plan = DataFrame.fromRows([{ age: 1, salary: 2, city: 'x' }])
      .filter(col('age').gt(0).and(col('salary').gt(0)))
      .groupBy('city')
      .agg({ salary: 'mean' }).plan
    expect(planHasGpuKernel(plan)).toBe(true)
  })

  it('maskToIndices compacts correctly', () => {
    const mask = new Uint32Array([1, 0, 1, 1, 0])
    expect(Array.from(maskToIndices(mask))).toEqual([0, 2, 3])
  })

  it('columnToF32 reuses Float32Array', () => {
    const data = new Float32Array([1, 2, 3])
    const col: Column = { field: { name: 'a', dtype: 'f32', nullable: false }, data }
    expect(columnToF32(col, 3)).toBe(data)
  })
})

describe('webgpu backend without device', () => {
  it('falls back to CPU and matches results', async () => {
    const gpu = new WebGpuBackend(executeCpu, null)
    expect(await gpu.waitReady()).toBe(false)
    expect(gpu.supports({ type: 'scan', table: DataFrame.fromRows([{ a: 1 }]).table })).toBe(false)

    const df = DataFrame.fromRows([
      { age: 10, salary: 50_000 },
      { age: 40, salary: 50_000 },
      { age: 40, salary: 10_000 },
    ])
    const plan = df.filter(col('age').gt(30).and(col('salary').gt(45_000))).plan
    const out = await gpu.execute(plan)
    expect(out.numRows).toBe(1)
  })
})
