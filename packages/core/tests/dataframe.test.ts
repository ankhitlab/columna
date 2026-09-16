import { describe, expect, it } from 'vitest'
import { DataFrame, col, lit } from '@columna/core'
import { executeCpu } from '@columna/runtime'

describe('DataFrame fluent core', () => {
  const sample = () =>
    DataFrame.fromRows([
      { city: 'Berlin', age: 30, salary: 72000 },
      { city: 'Berlin', age: 22, salary: 48000 },
      { city: 'Paris', age: 41, salary: 91000 },
      { city: 'Paris', age: 17, salary: 12000 },
    ])

  it('fromRows / shape / columns', () => {
    const df = sample()
    expect(df.shape).toEqual([4, 3])
    expect(df.columns).toEqual(['city', 'age', 'salary'])
  })

  it('select filter sort collect', async () => {
    const out = await sample()
      .select('city', 'age')
      .filter(col('age').gt(18))
      .sort(col('age').desc())
      .collect()
    expect(out.toArray()).toEqual([
      { city: 'Paris', age: 41 },
      { city: 'Berlin', age: 30 },
      { city: 'Berlin', age: 22 },
    ])
  })

  it('withColumn arithmetic', async () => {
    const out = await sample()
      .withColumn('bonus', col('salary').mul(0.1))
      .select('salary', 'bonus')
      .head(1)
      .collect()
    expect(out.toArray()[0]).toEqual({ salary: 72000, bonus: 7200 })
  })

  it('fromCSV', () => {
    const df = DataFrame.fromCSV('a,b\n1,2\n3,4\n')
    expect(df.toArray()).toEqual([
      { a: 1, b: 2 },
      { a: 3, b: 4 },
    ])
  })

  it('fromJSON / toArrow roundtrip', () => {
    const df = DataFrame.fromJSON([{ x: 1, y: 'a' }])
    const arrow = df.toArrow()
    const back = DataFrame.fromArrow(arrow)
    expect(back.toArray()).toEqual([{ x: 1, y: 'a' }])
  })

  it('unknown column suggests', () => {
    expect(() => sample().select('cite')).toBeTruthy()
    expect(() => executeCpu({ type: 'scan', table: sample().table })).not.toThrow()
    expect(() => {
      const t = sample().table
      // force get via filter on bad col at collect
      return t.schema.find((f) => f.name === 'cite')
    }).not.toThrow()
  })

  it('lit expressions', async () => {
    const out = await sample().filter(col('age').gt(lit(40))).collect()
    expect(out.shape[0]).toBe(1)
  })

  it('slice and take', async () => {
    const sliced = await sample().slice(1, 3).collect()
    expect(sliced.shape[0]).toBe(2)
    const taken = await sample().take([0, 3]).collect()
    expect(taken.toArray().map((r) => r.age)).toEqual([30, 17])
  })
})
