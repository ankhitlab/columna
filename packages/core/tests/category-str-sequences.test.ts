import { describe, expect, it } from 'vitest'
import { DataFrame, col } from '../src/dataframe.js'

describe('category str transform sequences', () => {
  it('toLowerCase then valueCounts merges collapsed dictionary levels', async () => {
    const result = await DataFrame.fromColumns({
      s: ['A', 'a'],
    })
      .withColumn('s', col('s').str.toLowerCase())
      .valueCounts('s')
      .collect()
    expect(result.toArray()).toEqual([{ s: 'a', count: 2 }])
  })

  it('toLowerCase then unique keeps a single row', async () => {
    const result = await DataFrame.fromColumns({
      s: ['A', 'a'],
    })
      .withColumn('s', col('s').str.toLowerCase())
      .unique(['s'])
      .collect()
    expect(result.toArray()).toEqual([{ s: 'a' }])
  })

  it('trim then valueCounts merges padded levels', async () => {
    const result = await DataFrame.fromColumns({
      s: [' x', 'x '],
    })
      .withColumn('s', col('s').str.trim())
      .valueCounts('s')
      .collect()
    expect(result.toArray()).toEqual([{ s: 'x', count: 2 }])
  })

  it('replace then unique merges collapsed levels', async () => {
    const result = await DataFrame.fromColumns({
      s: ['ab', 'aX'],
    })
      .withColumn('s', col('s').str.replace('X', 'b'))
      .unique(['s'])
      .collect()
    expect(result.toArray()).toEqual([{ s: 'ab' }])
  })
})
