import { describe, expect, it, vi } from 'vitest'
import { DataFrame, col } from '@columna/core'

describe('analyst ergonomics', () => {
  it('DataFrame.head/tail are sync materialized peeks', () => {
    const df = DataFrame.fromRows([
      { id: 1 },
      { id: 2 },
      { id: 3 },
      { id: 4 },
      { id: 5 },
    ])
    const head = df.head(2)
    expect(head).toBeInstanceOf(DataFrame)
    expect(head.toArray()).toEqual([{ id: 1 }, { id: 2 }])
    expect(df.tail(2).toArray()).toEqual([{ id: 4 }, { id: 5 }])
    // chaining without collect
    expect(df.head(3).tail(1).toArray()).toEqual([{ id: 3 }])
  })

  it('LazyFrame.head stays lazy', async () => {
    const lf = DataFrame.fromRows([{ id: 1 }, { id: 2 }, { id: 3 }]).lazy().head(2)
    expect(typeof (lf as { collect?: unknown }).collect).toBe('function')
    expect((await lf.collect()).toArray()).toEqual([{ id: 1 }, { id: 2 }])
  })

  it('show/print logs markdown and returns this', () => {
    const df = DataFrame.fromRows([
      { a: 1, b: 'x' },
      { a: 2, b: 'y' },
    ])
    const spy = vi.spyOn(console, 'log').mockImplementation(() => {})
    expect(df.show(1)).toBe(df)
    expect(spy.mock.calls[0]?.[0]).toContain('| a | b |')
    expect(df.print(1)).toBe(df)
    spy.mockRestore()
  })

  it('fillNull accepts a per-column map', async () => {
    const df = DataFrame.fromRows([
      { a: 1, b: null as number | null, c: 'x' },
      { a: null as number | null, b: 2, c: null as string | null },
    ])
    const out = await df.fillNull({ a: 0, b: -1, c: '?' }).collect()
    expect(out.toArray()).toEqual([
      { a: 1, b: -1, c: 'x' },
      { a: 0, b: 2, c: '?' },
    ])
  })

  it('ffill / bfill leave edge nulls', async () => {
    const df = DataFrame.fromRows([
      { v: null as number | null },
      { v: 1 },
      { v: null as number | null },
      { v: null as number | null },
      { v: 4 },
      { v: null as number | null },
    ])
    const fwd = await df.ffill(['v']).collect()
    expect(fwd.toArray().map((r) => r.v)).toEqual([null, 1, 1, 1, 4, 4])
    const back = await df.bfill(['v']).collect()
    expect(back.toArray().map((r) => r.v)).toEqual([1, 1, 4, 4, 4, null])
  })

  it('GroupBy.count / mean skip non-numeric columns', async () => {
    const df = DataFrame.fromRows([
      { g: 'a', x: 1, y: 10, label: 'p', flag: true },
      { g: 'a', x: 3, y: 30, label: 'q', flag: false },
      { g: 'b', x: 5, y: 50, label: 'r', flag: true },
    ])
    const counts = await df.groupBy('g').count().sort('g').collect()
    expect(counts.toArray()).toEqual([
      { g: 'a', count: 2 },
      { g: 'b', count: 1 },
    ])
    const means = await df.groupBy('g').mean().sort('g').collect()
    expect(means.columns.sort()).toEqual(['g', 'x', 'y'])
    expect(means.toArray()).toEqual([
      { g: 'a', x: 2, y: 20 },
      { g: 'b', x: 5, y: 50 },
    ])
  })

  it('join suffixes and rightJoin', async () => {
    const left = DataFrame.fromRows([
      { id: 1, val: 10 },
      { id: 2, val: 20 },
    ])
    const right = DataFrame.fromRows([
      { id: 1, val: 100 },
      { id: 3, val: 300 },
    ])
    const defaultSuffix = await left.innerJoin(right, 'id').collect()
    expect(defaultSuffix.columns).toEqual(['id', 'val', 'val_right'])
    expect(defaultSuffix.toArray()).toEqual([{ id: 1, val: 10, val_right: 100 }])

    const custom = await left.join(right, { on: 'id', how: 'inner', lSuffix: '_l', rSuffix: '_r' }).collect()
    expect(custom.columns).toEqual(['id', 'val_l', 'val_r'])
    expect(custom.toArray()).toEqual([{ id: 1, val_l: 10, val_r: 100 }])

    const rj = await left.rightJoin(right, 'id').collect()
    expect(rj.shape[0]).toBe(2)
    expect(rj.columns).toContain('val_right')
    const onlyRight = rj.toArray().find((r) => r.id === 3)
    expect(onlyRight).toBeTruthy()
    expect(onlyRight?.val ?? null).toBeNull()
    expect(onlyRight?.val_right).toBe(300)
  })

  it('sort nullsFirst puts nulls at the start', async () => {
    const df = DataFrame.fromRows([
      { v: 2 },
      { v: null as number | null },
      { v: 1 },
    ])
    const last = await df.sort('v').collect()
    expect(last.toArray().map((r) => r.v)).toEqual([1, 2, null])
    const first = await df.sort(col('v').asc({ nullsLast: false })).collect()
    expect(first.toArray().map((r) => r.v)).toEqual([null, 1, 2])
  })

  it('DataFrame.describe is sync; LazyFrame.describe stays lazy', async () => {
    const df = DataFrame.fromRows([{ x: 1 }, { x: 2 }, { x: 3 }])
    const desc = df.describe()
    expect(desc).toBeInstanceOf(DataFrame)
    expect(desc.columns).toContain('stat')
    expect(desc.columns).toContain('x')
    const lazyDesc = await df.lazy().describe().collect()
    expect(lazyDesc.shape[0]).toBe(desc.shape[0])
  })

  it('GroupBy std and multi-agg expansion', async () => {
    const df = DataFrame.fromRows([
      { g: 'a', x: 1 },
      { g: 'a', x: 3 },
      { g: 'b', x: 5 },
    ])
    const multi = await df.groupBy('g').agg({ x: ['sum', 'mean'] }).sort('g').collect()
    expect(multi.columns.sort()).toEqual(['g', 'x_mean', 'x_sum'])
    expect(multi.toArray()).toEqual([
      { g: 'a', x_sum: 4, x_mean: 2 },
      { g: 'b', x_sum: 5, x_mean: 5 },
    ])
    const std = await df.groupBy('g').std().sort('g').collect()
    expect(std.columns).toContain('x')
    expect(Number(std.toArray()[0]?.x)).toBeCloseTo(Math.SQRT2, 10)
  })

  it('where / notNull / between / exclude / selectNumeric', async () => {
    const df = DataFrame.fromRows([
      { age: 17, city: 'A', score: 1 },
      { age: null as number | null, city: 'B', score: 2 },
      { age: 30, city: 'C', score: 3 },
    ])
    expect((await df.where((c) => c.age.gt(18)).collect()).shape[0]).toBe(1)
    expect((await df.notNull('age').collect()).shape[0]).toBe(2)
    expect((await df.between('age', 18, 40).collect()).toArray()).toEqual([{ age: 30, city: 'C', score: 3 }])
    expect((await df.exclude('city').collect()).columns.sort()).toEqual(['age', 'score'])
    expect((await df.selectNumeric().collect()).columns.sort()).toEqual(['age', 'score'])
  })

  it('outerJoin, validate, and assign/rename(fn)', async () => {
    const left = DataFrame.fromRows([
      { id: 1, a: 10 },
      { id: 2, a: 20 },
    ])
    const right = DataFrame.fromRows([
      { id: 1, b: 100 },
      { id: 3, b: 300 },
    ])
    const outer = await left.outerJoin(right, 'id').sort('id').collect()
    expect(outer.shape[0]).toBe(3)
    await expect(left.join(right, { on: 'id', validate: '1:1' }).collect()).resolves.toBeTruthy()
    const dupLeft = DataFrame.fromRows([
      { id: 1, a: 1 },
      { id: 1, a: 2 },
    ])
    await expect(dupLeft.join(right, { on: 'id', validate: '1:1' }).collect()).rejects.toThrow(/left keys/)

    const assigned = await left.assign({ z: col('a').mul(2) }).collect()
    expect(assigned.toArray()[0]).toMatchObject({ id: 1, a: 10, z: 20 })
    const renamed = await left.rename((n) => n.toUpperCase()).collect()
    expect(renamed.columns.sort()).toEqual(['A', 'ID'])
  })

  it('Series fillNull / valueCounts and frame nunique / col', async () => {
    const df = DataFrame.fromRows([
      { x: 1, tag: 'a' },
      { x: null as number | null, tag: 'a' },
      { x: 3, tag: 'b' },
    ])
    expect(df.col('x').fillNull(0).toArray()).toEqual([1, 0, 3])
    const vc = df.col('tag').valueCounts()
    expect(vc).toBeInstanceOf(DataFrame)
    expect(vc.shape[0]).toBe(2)
    const nu = df.nunique()
    expect(nu.toArray()).toEqual([
      { column: 'x', nunique: 3 },
      { column: 'tag', nunique: 2 },
    ])
  })

  it('typed filter callback still works alongside show', async () => {
    const df = DataFrame.fromRows([
      { age: 17, name: 'a' },
      { age: 22, name: 'b' },
    ])
    const spy = vi.spyOn(console, 'log').mockImplementation(() => {})
    df.show(1)
    spy.mockRestore()
    const out = await df.filter((c) => c.age.gt(18)).collect()
    expect(out.toArray()).toEqual([{ age: 22, name: 'b' }])
    void col
  })
})
