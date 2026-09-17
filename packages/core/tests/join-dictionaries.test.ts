import { describe, expect, it } from 'vitest'
import { DataFrame, col } from '../src/dataframe.js'

/**
 * Category columns are dictionary-encoded per frame. Two frames built separately encode the same strings
 * under different codes; a join must match strings, never codes. (Regression: the fast join compared codes
 * and silently produced Berlin → ES, Paris → IT when the right dictionary was in a different order.)
 */
describe('joins on category keys with different dictionaries', () => {
  const left = DataFrame.fromRows([
    { id: 1, city: 'Berlin' },
    { id: 2, city: 'Paris' },
    { id: 3, city: 'Rome' },
    { id: 4, city: 'Madrid' },
    { id: 5, city: 'Wien' },
  ])
  const right = DataFrame.fromRows([
    { city: 'Madrid', region: 'ES' },
    { city: 'Rome', region: 'IT' },
    { city: 'Berlin', region: 'DE' },
    { city: 'Lisboa', region: 'PT' },
  ])
  const expectedInner = [
    { id: 1, city: 'Berlin', region: 'DE' },
    { id: 3, city: 'Rome', region: 'IT' },
    { id: 4, city: 'Madrid', region: 'ES' },
  ]

  it('inner / left / semi / anti match by string value', async () => {
    expect((await left.join(right, { on: 'city' }).collect()).toArray()).toEqual(expectedInner)
    expect((await left.join(right, { on: 'city', how: 'left' }).collect()).toArray()).toEqual([
      { id: 1, city: 'Berlin', region: 'DE' },
      { id: 2, city: 'Paris', region: null },
      { id: 3, city: 'Rome', region: 'IT' },
      { id: 4, city: 'Madrid', region: 'ES' },
      { id: 5, city: 'Wien', region: null },
    ])
    expect((await left.semiJoin(right, 'city').collect()).toArray().map((r) => r.id)).toEqual([1, 3, 4])
    expect((await left.antiJoin(right, 'city').collect()).toArray().map((r) => r.id)).toEqual([2, 5])
  })

  it('category ⨝ utf8 and utf8 ⨝ category (mixed encodings) give the same answer', async () => {
    const rightUtf8 = DataFrame.fromColumns({ city: right.getColumn('city').toArray().map(String), region: right.getColumn('region').toArray().map(String) })
    // force utf8 by making the cardinality rule fail: cast through withColumn
    const r2 = await rightUtf8.withColumn('city', col('city').cast('utf8')).collect()
    expect(r2.getColumn('city').dtype).toBe('utf8')
    expect((await left.join(r2, { on: 'city' }).collect()).toArray()).toEqual(expectedInner)
    const l2 = await left.withColumn('city', col('city').cast('utf8')).collect()
    expect((await l2.join(right, { on: 'city' }).collect()).toArray()).toEqual(expectedInner)
  })

  it('CSV-loaded frames (independent dictionaries) join correctly at scale', async () => {
    const cities = ['Berlin', 'Paris', 'Rome', 'Madrid', 'Wien']
    const n = 20_000
    const csv = 'id,city\n' + Array.from({ length: n }, (_, i) => `${i},${cities[i % 5]}`).join('\n')
    const big = DataFrame.fromCSV(csv)
    const joined = await big.join(right, { on: 'city' }).collect()
    expect(joined.shape[0]).toBe((n / 5) * 3)
    const regions = new Map<string, string>()
    for (const r of joined.toArray()) regions.set(String(r.city), String(r.region))
    expect([...regions.entries()].sort()).toEqual([['Berlin', 'DE'], ['Madrid', 'ES'], ['Rome', 'IT']])
  })
})

describe('transpose decodes category cells', () => {
  it('header column and values show strings, not dictionary codes', async () => {
    const df = DataFrame.fromRows([{ k: 'a', city: 'Berlin', n: 1 }, { k: 'b', city: 'Paris', n: 2 }])
    expect((await df.transpose('k').collect()).toArray()).toEqual([
      { column: 'city', a: 'Berlin', b: 'Paris' },
      { column: 'n', a: '1', b: '2' },
    ])
  })
})
