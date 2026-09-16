import { describe, expect, it } from 'vitest'
import { DataFrame, col, lit, when } from '../src/dataframe.js'

describe('str / dt namespaces', () => {
  it('string ops', async () => {
    const df = DataFrame.fromRows([
      { name: 'Ada Lovelace' },
      { name: ' Alan ' },
    ])
    const out = await df
      .withColumns(
        col('name').str.trim().alias('t'),
        col('name').str.toLowerCase().alias('lo'),
        col('name').str.contains('Ada').alias('has'),
        col('name').str.len().alias('n'),
      )
      .collect()
    const rows = out.toArray()
    expect(rows[0]).toMatchObject({ t: 'Ada Lovelace', lo: 'ada lovelace', has: true, n: 12 })
    expect(rows[1]).toMatchObject({ t: 'Alan', has: false })
  })
  it('concat and padding', async () => {
    const df = DataFrame.fromRows([
      { first: 'Ada', last: 'Lovelace', id: 7 },
      { first: 'Alan', last: null, id: 42 },
    ])
    const out = await df
      .withColumns(
        col('first').str.concat(col('last'), ' ').alias('full'),
        col('first').str.concat('!').alias('bang'),
        col('id').cast('utf8').str.padStart(4, '0').alias('code'),
        col('first').str.padEnd(6, '.').alias('padded'),
      )
      .collect()
    const rows = out.toArray()
    expect(rows[0]).toMatchObject({ full: 'Ada Lovelace', bang: 'Ada!', code: '0007', padded: 'Ada...' })
    expect(rows[1]).toMatchObject({ full: null, bang: 'Alan!', code: '0042', padded: 'Alan..' })
  })

  it('datetime ops on epoch ms', async () => {
    const df = DataFrame.fromRows([{ ts: Date.UTC(2024, 0, 15, 12, 30, 0) }])
    const out = await df
      .select(
        col('ts').dt.year().alias('y'),
        col('ts').dt.month().alias('m'),
        col('ts').dt.day().alias('d'),
        col('ts').dt.hour().alias('h'),
      )
      .collect()
    expect(out.toArray()[0]).toEqual({ y: 2024, m: 1, d: 15, h: 12 })
  })
})

describe('when / isIn / aggs', () => {
  it('when then otherwise', async () => {
    const df = DataFrame.fromRows([{ x: 1 }, { x: 5 }, { x: 10 }])
    const out = await df
      .withColumn(
        'band',
        when(col('x').lt(3))
          .then('low')
          .when(col('x').lt(8))
          .then('mid')
          .otherwise('high'),
      )
      .collect()
    expect(out.toArray().map((r) => r.band)).toEqual(['low', 'mid', 'high'])
  })

  it('isIn and isBetween', async () => {
    const df = DataFrame.fromRows([{ x: 1 }, { x: 2 }, { x: 3 }])
    const out = await df
      .filter(col('x').isIn([1, 3]).and(col('x').isBetween(1, 3)))
      .collect()
    expect(out.toArray()).toEqual([{ x: 1 }, { x: 3 }])
  })

  it('std var median quantile groupBy', async () => {
    const df = DataFrame.fromRows([
      { g: 'a', v: 1 },
      { g: 'a', v: 3 },
      { g: 'a', v: 5 },
    ])
    const out = await df
      .groupBy('g')
      .agg({
        s: col('v').std(),
        med: col('v').median(),
        q: col('v').quantile(0.5),
      })
      .collect()
    const row = out.toArray()[0]!
    expect(row.med).toBe(3)
    expect(row.q).toBe(3)
    expect(Number(row.s)).toBeCloseTo(Math.sqrt(4), 5)
  })
})

describe('frame ergonomics', () => {
  it('tail sample explode withColumns', async () => {
    const df = DataFrame.fromRows([
      { i: 1, tags: '["a","b"]' },
      { i: 2, tags: '["c"]' },
      { i: 3, tags: '[]' },
    ])
    const tailed = await df.tail(2).collect()
    expect(tailed.shape[0]).toBe(2)

    const sampled = await df.sample({ n: 2, seed: 7 }).collect()
    expect(sampled.shape[0]).toBe(2)

    const exploded = await df.explode('tags').collect()
    expect(exploded.shape[0]).toBeGreaterThanOrEqual(3)

    const wc = await df.withColumns({ j: col('i').add(1) }).collect()
    expect(wc.toArray()[0]).toMatchObject({ i: 1, j: 2 })
  })

  it('cross semi anti joins', async () => {
    const left = DataFrame.fromRows([
      { id: 1, n: 'a' },
      { id: 2, n: 'b' },
    ])
    const right = DataFrame.fromRows([{ id: 1 }, { id: 3 }])
    const cross = await left.crossJoin(right.select(col('id').alias('rid'))).collect()
    expect(cross.shape[0]).toBe(4)

    const semi = await left.semiJoin(right, 'id').collect()
    expect(semi.toArray()).toEqual([{ id: 1, n: 'a' }])

    const anti = await left.antiJoin(right, 'id').collect()
    expect(anti.toArray()).toEqual([{ id: 2, n: 'b' }])
  })
})

describe('write / present / series / wave2', () => {
  it('toCsv markdown html profile', () => {
    const df = DataFrame.fromRows([
      { a: 1, b: 'x' },
      { a: 2, b: 'y' },
    ])
    expect(df.toCsv()).toContain('a,b')
    expect(df.toMarkdown()).toContain('| a | b |')
    expect(df.toHTML()).toContain('<table>')
    expect(df.profile().rows).toBe(2)
    expect(df.getColumn('a').sum()).toBe(3)
  })

  it('shift diff pctChange mapElements unnest', async () => {
    const df = DataFrame.fromRows([{ v: 10 }, { v: 20 }, { v: 30 }])
    const out = await df
      .withColumns(
        col('v').shift(1).alias('s'),
        col('v').diff(1).alias('d'),
        col('v').pctChange(1).alias('p'),
        col('v').mapElements((x) => Number(x) * 2).alias('m'),
      )
      .collect()
    const rows = out.toArray()
    expect(rows[1]).toMatchObject({ v: 20, s: 10, d: 10, p: 1, m: 40 })

    const nested = DataFrame.fromRows([{ payload: '{"user":{"id":1},"n":2}' }])
    const flat = await nested.unnest('payload').collect()
    expect(flat.toArray()[0]).toMatchObject({ 'user.id': 1, n: 2 })
  })

  it('joinAsof interpolate expanding pipe', async () => {
    const left = DataFrame.fromRows([
      { t: 1, x: 1 },
      { t: 3, x: 2 },
      { t: 5, x: 3 },
    ])
    const right = DataFrame.fromRows([
      { t: 2, y: 10 },
      { t: 4, y: 20 },
    ])
    const asof = await left.joinAsof(right, { leftOn: 't', strategy: 'backward' }).collect()
    expect(asof.shape[0]).toBe(3)

    const interp = await DataFrame.fromRows([{ a: 1 }, { a: null }, { a: 3 }])
      .interpolate(['a'])
      .collect()
    expect(interp.toArray()[1]?.a).toBe(2)

    const exp = await left.expanding('c', 'x', 'sum').collect()
    expect(exp.toArray().map((r) => r.c)).toEqual([1, 3, 6])

    const piped = left.pipe((d) => d.head(1))
    expect((await piped.collect()).shape[0]).toBe(1)
  })
})

void lit
