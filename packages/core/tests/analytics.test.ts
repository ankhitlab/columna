import { describe, expect, it } from 'vitest'
import { DataFrame, LazyFrame, col, lit, when } from '@columna/core'

describe('analytics', () => {
  const sample = () =>
    DataFrame.fromRows([
      { city: 'Berlin', age: 30, salary: 72000 },
      { city: 'Berlin', age: 22, salary: 48000 },
      { city: 'Paris', age: 41, salary: 91000 },
      { city: 'Paris', age: 35, salary: 80000 },
    ])

  it('groupBy agg mean count', async () => {
    const out = await sample()
      .groupBy('city')
      .agg({ salary: 'mean', age: 'count' })
      .sort('city')
      .collect()
    const rows = out.toArray().sort((a, b) => String(a.city).localeCompare(String(b.city)))
    expect(rows).toEqual([
      { city: 'Berlin', salary: 60000, age: 2 },
      { city: 'Paris', salary: 85500, age: 2 },
    ])
  })

  it('inner join', async () => {
    const left = DataFrame.fromRows([
      { id: 1, name: 'A' },
      { id: 2, name: 'B' },
    ])
    const right = DataFrame.fromRows([
      { id: 1, score: 10 },
      { id: 3, score: 30 },
    ])
    const out = await left.innerJoin(right, 'id').collect()
    expect(out.toArray()).toEqual([{ id: 1, name: 'A', score: 10 }])
  })

  it('left join keeps left', async () => {
    const left = DataFrame.fromRows([
      { id: 1, name: 'A' },
      { id: 2, name: 'B' },
    ])
    const right = DataFrame.fromRows([{ id: 1, score: 10 }])
    const out = await left.leftJoin(right, 'id').sort('id').collect()
    expect(out.shape[0]).toBe(2)
    expect(out.toArray().find((r) => r.id === 2)?.score ?? null).toBeNull()
  })

  it('fillNull dropNull', async () => {
    const df = DataFrame.fromRows([
      { a: 1, b: null },
      { a: null, b: 2 },
    ])
    const filled = await df.fillNull(0).collect()
    expect(filled.toArray()).toEqual([
      { a: 1, b: 0 },
      { a: 0, b: 2 },
    ])
    const dropped = await df.dropNull(['a']).collect()
    expect(dropped.toArray()).toEqual([{ a: 1, b: null }])
  })

  it('melt and pivot', async () => {
    const df = DataFrame.fromRows([
      { id: 1, x: 10, y: 20 },
      { id: 2, x: 30, y: 40 },
    ])
    const melted = await df.melt({ idVars: ['id'], valueVars: ['x', 'y'] }).collect()
    expect(melted.shape[0]).toBe(4)
    const pivoted = await melted
      .pivot({ index: 'id', columns: 'variable', values: 'value', agg: 'sum' })
      .sort('id')
      .collect()
    const rows = pivoted.toArray()
    expect(rows[0]).toMatchObject({ id: 1, x: 10, y: 20 })
  })

  it('concat vertical', async () => {
    const a = DataFrame.fromRows([{ v: 1 }])
    const b = DataFrame.fromRows([{ v: 2 }])
    const out = await LazyFrame.concat([a, b]).collect()
    expect(out.toArray()).toEqual([{ v: 1 }, { v: 2 }])
  })

  it('describe valueCounts unique', async () => {
    const df = sample()
    const desc = await df.describe().collect()
    expect(desc.columns).toContain('stat')
    expect(desc.columns).toContain('age')
    const vc = await df.valueCounts('city').collect()
    expect(vc.shape[0]).toBe(2)
    const uniq = await df.unique(['city']).collect()
    expect(uniq.shape[0]).toBe(2)
  })

  it('multi-key groupBy on categories', async () => {
    const df = DataFrame.fromColumns({
      city: { codes: new Uint32Array([0, 0, 1, 1]), dictionary: ['Berlin', 'Paris'] },
      category: { codes: new Uint32Array([0, 1, 0, 1]), dictionary: ['a', 'b'] },
      salary: new Float64Array([10, 30, 40, 60]),
      id: new Int32Array([1, 2, 3, 4]),
    })
    const out = await df
      .groupBy('city', 'category')
      .agg({ salary: 'mean', id: 'count' })
      .sort('city', 'category')
      .collect()
    const rows = out.toArray()
    expect(rows).toEqual([
      { city: 'Berlin', category: 'a', salary: 10, id: 1 },
      { city: 'Berlin', category: 'b', salary: 30, id: 1 },
      { city: 'Paris', category: 'a', salary: 40, id: 1 },
      { city: 'Paris', category: 'b', salary: 60, id: 1 },
    ])
  })

  it('unique keep first on numeric key', async () => {
    const df = DataFrame.fromColumns({
      user_id: new Int32Array([1, 2, 1, 3, 2]),
      v: new Int32Array([10, 20, 11, 30, 21]),
    })
    const uniq = await df.unique(['user_id']).collect()
    expect(uniq.toArray()).toEqual([
      { user_id: 1, v: 10 },
      { user_id: 2, v: 20 },
      { user_id: 3, v: 30 },
    ])
  })

  it('describe stats on typed columns', async () => {
    const df = DataFrame.fromColumns({
      x: new Float64Array([1, 2, 3, 4]),
    })
    const desc = await df.describe().collect()
    const rows = desc.toArray()
    const byStat = Object.fromEntries(rows.map((r) => [r.stat, r.x]))
    expect(byStat.count).toBe(4)
    expect(byStat.mean).toBe(2.5)
    expect(byStat.min).toBe(1)
    expect(byStat.max).toBe(4)
    expect(byStat['50%']).toBe(2.5)
  })

  it('unique keep last and none with null key', async () => {
    const df = DataFrame.fromRows([
      { user_id: 1, v: 10 },
      { user_id: 2, v: 20 },
      { user_id: 1, v: 11 },
      { user_id: null, v: 99 },
      { user_id: 2, v: 21 },
      { user_id: null, v: 98 },
      { user_id: 3, v: 30 },
    ])

    const last = await df.unique(['user_id'], 'last').collect()
    expect(last.toArray()).toEqual([
      { user_id: 1, v: 11 },
      { user_id: 2, v: 21 },
      { user_id: null, v: 98 },
      { user_id: 3, v: 30 },
    ])

    const none = await df.unique(['user_id'], 'none').collect()
    expect(none.toArray()).toEqual([{ user_id: 3, v: 30 }])
  })

  it('unique preserves fractional f64 between integer bounds', async () => {
    const result = await DataFrame.fromColumns({
      x: new Float64Array([0, 0.5, 1]),
    })
      .unique(['x'])
      .collect()
    expect(result.toArray()).toEqual([{ x: 0 }, { x: 0.5 }, { x: 1 }])
  })

  it('unique keep none drops all-null key column', async () => {
    const result = await DataFrame.fromRows([{ x: null }, { x: null }, { x: null }])
      .unique(['x'], 'none')
      .collect()
    expect(result.toArray()).toEqual([])
  })

  it('describe quantiles for i32, fractional f64, and integral f64', async () => {
    const df = DataFrame.fromColumns({
      a: new Float64Array([1.5, 2.25, 3.75, 4.5, 10.125, -2.5, 0.5]),
      b: new Int32Array([5, 3, 9, 1, 7, 2, 8]),
      c: new Float64Array([10, 20, 30, 40, 50, 60, 70]),
    })
    const desc = await df.describe().collect()
    const byStat = (col: string) =>
      Object.fromEntries(desc.toArray().map((r) => [String(r.stat), Number(r[col])]))

    const a = byStat('a')
    expect(a.count).toBe(7)
    expect(a.mean).toBeCloseTo(2.875, 12)
    expect(a.std).toBeCloseTo(3.932370998096017, 12)
    expect(a.min).toBe(-2.5)
    expect(a['25%']).toBe(1)
    expect(a['50%']).toBe(2.25)
    expect(a['75%']).toBe(4.125)
    expect(a.max).toBe(10.125)

    const b = byStat('b')
    expect(b.count).toBe(7)
    expect(b.mean).toBe(5)
    expect(b.std).toBeCloseTo(3.109126351029605, 12)
    expect(b.min).toBe(1)
    expect(b['25%']).toBe(2.5)
    expect(b['50%']).toBe(5)
    expect(b['75%']).toBe(7.5)
    expect(b.max).toBe(9)

    const c = byStat('c')
    expect(c.count).toBe(7)
    expect(c.mean).toBe(40)
    expect(c.std).toBeCloseTo(21.602468994692867, 12)
    expect(c.min).toBe(10)
    expect(c['25%']).toBe(25)
    expect(c['50%']).toBe(40)
    expect(c['75%']).toBe(55)
    expect(c.max).toBe(70)
  })

  it('rolling sum min max count nulls and window 1', async () => {
    const dense = DataFrame.fromColumns({ v: new Float64Array([1, 2, 3, 4]) })
    const sum = await dense.rolling('s', 'v', 2, 'sum').collect()
    expect(sum.toArray().map((r) => r.s)).toEqual([1, 3, 5, 7])
    const min = await dense.rolling('n', 'v', 2, 'min').collect()
    expect(min.toArray().map((r) => r.n)).toEqual([1, 1, 2, 3])
    const max = await dense.rolling('x', 'v', 2, 'max').collect()
    expect(max.toArray().map((r) => r.x)).toEqual([1, 2, 3, 4])
    const count = await dense.rolling('c', 'v', 2, 'count').collect()
    expect(count.toArray().map((r) => r.c)).toEqual([1, 2, 2, 2])

    const w1 = await dense.rolling('m', 'v', 1, 'mean').collect()
    expect(w1.toArray().map((r) => r.m)).toEqual([1, 2, 3, 4])

    const withNulls = DataFrame.fromRows([{ v: 1 }, { v: null }, { v: 3 }])
    const rolled = await withNulls.rolling('m', 'v', 2, 'mean').collect()
    expect(rolled.toArray().map((r) => r.m)).toEqual([1, 1, 3])

    const allNull = DataFrame.fromRows([{ v: null }, { v: null }])
    const empty = await allNull.rolling('m', 'v', 2, 'mean').collect()
    expect(empty.toArray().map((r) => r.m)).toEqual([null, null])
  })

  it('melt emits category variable and pivots back', async () => {
    const df = DataFrame.fromColumns({
      id: new Int32Array([1, 2]),
      age: new Int32Array([10, 30]),
      salary: new Float64Array([20, 40]),
    })
    const melted = await df.melt({ idVars: ['id'], valueVars: ['age', 'salary'] }).collect()
    expect(melted.dtypes.variable).toBe('category')
    expect(melted.shape[0]).toBe(4)

    const pivoted = await melted
      .pivot({ index: 'id', columns: 'variable', values: 'value', agg: 'sum' })
      .sort('id')
      .collect()
    expect(pivoted.toArray()).toEqual([
      { id: 1, age: 10, salary: 20 },
      { id: 2, age: 30, salary: 40 },
    ])
  })

  it('fused filter then groupBy matches materialize then groupBy', async () => {
    const df = DataFrame.fromColumns({
      city: { codes: new Uint32Array([0, 0, 1, 1, 0, 1]), dictionary: ['Berlin', 'Paris'] },
      age: new Int32Array([20, 40, 30, 50, 35, 22]),
      salary: new Float64Array([40_000, 70_000, 45_000, 90_000, 55_000, 80_000]),
      id: new Int32Array([1, 2, 3, 4, 5, 6]),
    })
    const predicate = col('age').gt(25).and(col('salary').gt(50_000))

    const fused = await df
      .filter(predicate)
      .groupBy('city')
      .agg({ salary: 'mean', id: 'count' })
      .sort('city')
      .collect()

    const stepwise = await (
      await df.filter(predicate).collect()
    )
      .groupBy('city')
      .agg({ salary: 'mean', id: 'count' })
      .sort('city')
      .collect()

    expect(fused.toArray()).toEqual(stepwise.toArray())
    expect(fused.toArray()).toEqual([
      { city: 'Berlin', salary: 62_500, id: 2 },
      { city: 'Paris', salary: 90_000, id: 1 },
    ])
  })

  it('fused filter then unique matches materialize then unique', async () => {
    const df = DataFrame.fromRows([
      { k: 'a', v: 1 },
      { k: 'a', v: 2 },
      { k: 'b', v: 3 },
      { k: 'a', v: 4 },
      { k: 'c', v: 5 },
      { k: 'b', v: 6 },
    ])
    const predicate = col('v').gt(2)

    const fused = await df.filter(predicate).unique(['k']).collect()
    const stepwise = await (await df.filter(predicate).collect()).unique(['k']).collect()

    expect(fused.toArray()).toEqual(stepwise.toArray())
    expect(fused.toArray()).toEqual([
      { k: 'b', v: 3 },
      { k: 'a', v: 4 },
      { k: 'c', v: 5 },
    ])
  })

  it('fused filter then sort then limit matches stepwise', async () => {
    const df = DataFrame.fromRows([
      { id: 1, score: 10 },
      { id: 2, score: 50 },
      { id: 3, score: 30 },
      { id: 4, score: 80 },
      { id: 5, score: 20 },
      { id: 6, score: 70 },
    ])
    const predicate = col('score').gt(25)

    const fused = await df.filter(predicate).sort(col('score').desc()).limit(2).collect()
    const stepwise = await (await df.filter(predicate).collect()).sort(col('score').desc()).limit(2).collect()

    expect(fused.toArray()).toEqual(stepwise.toArray())
    expect(fused.toArray()).toEqual([
      { id: 4, score: 80 },
      { id: 6, score: 70 },
    ])
  })

  it('project through join keeps only selected columns and matches stepwise', async () => {
    const left = DataFrame.fromRows([
      { id: 1, a: 10, x: 100 },
      { id: 2, a: 20, x: 200 },
      { id: 3, a: 30, x: 300 },
    ])
    const right = DataFrame.fromRows([
      { id: 1, b: 11, y: 111 },
      { id: 2, b: 22, y: 222 },
      { id: 4, b: 44, y: 444 },
    ])

    const fused = await left.join(right, { on: 'id' }).select('id', 'a', 'b').collect()
    const stepwise = await (await left.join(right, { on: 'id' }).collect()).select('id', 'a', 'b').collect()

    expect(fused.toArray()).toEqual(stepwise.toArray())
    expect(fused.columns).toEqual(['id', 'a', 'b'])
    expect(fused.toArray()).toEqual([
      { id: 1, a: 10, b: 11 },
      { id: 2, a: 20, b: 22 },
    ])
  })

  it('window and rolling', async () => {
    const df = DataFrame.fromRows([
      { g: 'a', v: 1 },
      { g: 'a', v: 2 },
      { g: 'a', v: 3 },
      { g: 'b', v: 10 },
    ])
    const w = await df
      .withWindow('rn', 'rowNumber', { partitionBy: ['g'], orderBy: ['v'] })
      .withWindow('cs', 'cumsum', { expr: col('v'), partitionBy: ['g'], orderBy: ['v'] })
      .collect()
    const aRows = w.toArray().filter((r) => r.g === 'a')
    expect(aRows.map((r) => r.rn)).toEqual([1, 2, 3])
    expect(aRows.map((r) => r.cs)).toEqual([1, 3, 6])

    const rolled = await DataFrame.fromRows([{ v: 1 }, { v: 2 }, { v: 3 }, { v: 4 }])
      .rolling('m', 'v', 2, 'mean')
      .collect()
    expect(rolled.toArray().map((r) => r.m)).toEqual([1, 1.5, 2.5, 3.5])
  })

  it('aggregates broadcast in scalar context (withColumn / select / filter / sort)', async () => {
    const df = DataFrame.fromRows([
      { g: 'a', x: 2, s: 'p' },
      { g: 'a', x: 4, s: 'q' },
      { g: 'b', x: 6, s: 'p' },
      { g: 'b', x: null, s: 'r' },
    ])
    // x: [2, 4, 6, null] → mean 4, sum 12, std 2, count 3

    const z = await df.withColumn('z', col('x').sub(col('x').mean()).div(col('x').std())).collect()
    expect(z.toArray().map((r) => r.z)).toEqual([-1, 0, 1, null])

    const share = await df.select('g', col('x').div(col('x').sum()).alias('share')).collect()
    expect(share.toArray().map((r) => r.share)).toEqual([2 / 12, 4 / 12, 6 / 12, null])

    const above = await df.filter(col('x').gt(col('x').mean())).collect()
    expect(above.toArray().map((r) => r.x)).toEqual([6])

    // Sort by distance from the median (median = 4)
    const byDist = await df.dropNull(['x']).sort(col('x').sub(col('x').median()).abs()).collect()
    expect(byDist.toArray().map((r) => r.x)).toEqual([4, 2, 6])

    // Nested aggregate: mean of deviations is 0; count/nunique on strings; quantile
    const nested = await df
      .withColumns(
        col('x').sub(col('x').mean()).mean().alias('dev'),
        col('s').nunique().alias('ns'),
        col('s').count().alias('cnt'),
        col('x').quantile(0.25).alias('q1'),
        col('x').max().sub(col('x').min()).alias('range'),
      )
      .collect()
    const row = nested.toArray()[0]!
    expect(row.dev).toBe(0)
    expect(row.ns).toBe(3)
    expect(row.cnt).toBe(4)
    expect(row.q1).toBe(3)
    expect(row.range).toBe(4)

    // Several aggregates of one expression share a single pass (memo): IQR with two quantiles, min/max/mean/std mix
    const multi = await df
      .select(
        col('x').quantile(0.75).sub(col('x').quantile(0.25)).alias('iqr'),
        col('x').max().sub(col('x').min()).div(col('x').std()).add(col('x').mean()).alias('mix'),
        col('x').sub(col('x').mean()).div(col('x').std()).alias('z'),
      )
      .collect()
    const m = multi.toArray()
    expect(m[0]!.iqr).toBe(2) // Q3 5, Q1 3
    expect(m[0]!.mix).toBeCloseTo((6 - 2) / 2 + 4, 12)
    expect(m.map((r) => r.z)).toEqual([-1, 0, 1, null])

    // Inside groupBy the same expression keeps per-group semantics
    const grouped = await df.groupBy('g').agg({ m: col('x').mean() }).sort('g').collect()
    expect(grouped.toArray().map((r) => r.m)).toEqual([3, 6])
  })

  it('math functions: sqrt/log/exp/pow/round/floor/ceil/sign', async () => {
    const df = DataFrame.fromRows([{ x: 4 }, { x: 1 }, { x: null }, { x: -2.5 }, { x: 1.005 }])
    const out = await df
      .withColumns(
        col('x').sqrt().alias('sq'),
        col('x').log().alias('ln'),
        col('x').log(10).alias('lg'),
        col('x').log10().alias('lg10'),
        col('x').log2().alias('lg2'),
        col('x').exp().alias('ex'),
        col('x').pow(2).alias('p2'),
        lit(2).pow(col('x')).alias('twoX'),
        col('x').pow(col('x')).alias('xx'),
        col('x').round().alias('r0'),
        col('x').round(2).alias('r2'),
        col('x').floor().alias('fl'),
        col('x').ceil().alias('ce'),
        col('x').sign().alias('sg'),
      )
      .collect()
    const rows = out.toArray()
    const c = (k: string) => rows.map((r) => r[k])

    expect(c('sq')).toEqual([2, 1, null, NaN, Math.sqrt(1.005)])
    expect(c('ln')).toEqual([Math.log(4), 0, null, NaN, Math.log(1.005)])
    expect(c('lg')[0]).toBeCloseTo(Math.log10(4), 12)
    expect(c('lg10')[0]).toBeCloseTo(Math.log10(4), 12)
    expect(c('lg2')).toEqual([2, 0, null, NaN, Math.log2(1.005)])
    expect(c('ex')[1]).toBe(Math.E)
    expect(c('p2')).toEqual([16, 1, null, 6.25, 1.005 ** 2])
    expect(c('twoX')).toEqual([16, 2, null, 2 ** -2.5, 2 ** 1.005])
    expect(c('xx')).toEqual([256, 1, null, NaN, 1.005 ** 1.005])
    // Half away from zero: -2.5 → -3 (Math.round would give -2); 1.005 → 1.01 despite binary drift
    expect(c('r0')).toEqual([4, 1, null, -3, 1])
    expect(c('r2')).toEqual([4, 1, null, -2.5, 1.01])
    expect(c('fl')).toEqual([4, 1, null, -3, 1])
    expect(c('ce')).toEqual([4, 1, null, -2, 2])
    expect(c('sg')).toEqual([1, 1, null, -1, 1])

    // Same ops through the scalar evaluator (when() forces the generic path) agree with the typed path
    const generic = await df
      .withColumns(
        when(col('x').isNull()).then(null).otherwise(col('x').round(2)).alias('r2'),
        when(col('x').isNull()).then(null).otherwise(col('x').pow(2)).alias('p2'),
      )
      .collect()
    expect(generic.toArray().map((r) => r.r2)).toEqual([4, 1, null, -2.5, 1.01])
    expect(generic.toArray().map((r) => r.p2)).toEqual([16, 1, null, 6.25, 1.005 ** 2])

    // Composes with broadcast aggregates: geometric mean = exp(mean(log x))
    const gm = await DataFrame.fromRows([{ v: 1 }, { v: 10 }, { v: 100 }])
      .select(col('v').log().mean().exp().round(6).alias('gm'))
      .collect()
    expect(gm.toArray()[0]!.gm).toBe(10)
  })

  it('corr / cov matrices match pandas (pearson, spearman, pairwise nulls)', async () => {
    // pandas: df = pd.DataFrame({x:[1,2,3,4,5], y:[2,4,5,4,5], z:[5,3,2,2,1]})
    // df.corr()  → x/y 0.774597, x/z -0.938315, y/z -0.942168
    // df.cov()   → var x 2.5, var y 1.5, var z 2.3, cov xy 1.5, cov xz -2.25, cov yz -1.75
    const df = DataFrame.fromRows([
      { x: 1, y: 2, z: 5, name: 'a' },
      { x: 2, y: 4, z: 3, name: 'b' },
      { x: 3, y: 5, z: 2, name: 'c' },
      { x: 4, y: 4, z: 2, name: 'd' },
      { x: 5, y: 5, z: 1, name: 'e' },
    ])
    const cell = (d: DataFrame, row: string, colName: string) =>
      d.toArray().find((r) => r.column === row)![colName] as number

    const corr = await df.corr().collect()
    expect(corr.columns).toEqual(['column', 'x', 'y', 'z']) // string column excluded
    expect(corr.toArray().map((r) => r.column)).toEqual(['x', 'y', 'z'])
    expect(cell(corr, 'x', 'x')).toBe(1)
    expect(cell(corr, 'x', 'y')).toBeCloseTo(0.7745966692414834, 12)
    expect(cell(corr, 'y', 'x')).toBeCloseTo(0.7745966692414834, 12)
    expect(cell(corr, 'x', 'z')).toBeCloseTo(-0.9383148632568363, 12)
    expect(cell(corr, 'y', 'z')).toBeCloseTo(-0.9421683286017897, 12)

    const cov = await df.cov().collect()
    expect(cell(cov, 'x', 'x')).toBeCloseTo(2.5, 12)
    expect(cell(cov, 'y', 'y')).toBeCloseTo(1.5, 12)
    expect(cell(cov, 'z', 'z')).toBeCloseTo(2.3, 12)
    expect(cell(cov, 'x', 'y')).toBeCloseTo(1.5, 12)
    expect(cell(cov, 'x', 'z')).toBeCloseTo(-2.25, 12)
    expect(cell(cov, 'y', 'z')).toBeCloseTo(-1.75, 12)

    // Spearman with ties: ranks y = [1, 2.5, 4.5, 2.5, 4.5], z = [5, 4, 2.5, 2.5, 1]
    // pandas df.corr(method='spearman') → x/y 0.737865, x/z -0.974679, y/z -0.865181
    const sp = await df.corr({ method: 'spearman' }).collect()
    expect(cell(sp, 'x', 'y')).toBeCloseTo(0.7378647873726218, 12)
    expect(cell(sp, 'x', 'z')).toBeCloseTo(-0.9746794344808963, 12)
    expect(cell(sp, 'y', 'z')).toBeCloseTo(-0.8651809126974003, 12)

    // Explicit subset keeps the given order
    const sub = await df.corr({ columns: ['z', 'x'] }).collect()
    expect(sub.columns).toEqual(['column', 'z', 'x'])
    expect(cell(sub, 'z', 'x')).toBeCloseTo(-0.9383148632568363, 12)
    await expect(df.corr({ columns: ['x', 'name'] }).collect()).rejects.toThrow(/not numeric/)

    // Pairwise-complete rows: null in y drops that row only for pairs involving y.
    // pandas: a=[1,2,3,4], b=[2,None,6,9], c=[1,1,1,1] → corr a/b 0.99419163 (over 3 rows), var b 12.333…, c → NaN
    const nulls = DataFrame.fromRows([
      { a: 1, b: 2, c: 1 },
      { a: 2, b: null, c: 1 },
      { a: 3, b: 6, c: 1 },
      { a: 4, b: 9, c: 1 },
    ])
    const nc = await nulls.corr().collect()
    expect(cell(nc, 'a', 'b')).toBeCloseTo(0.9941916256019201, 12)
    expect(cell(nc, 'a', 'a')).toBe(1)
    expect(cell(nc, 'c', 'c')).toBeNaN()
    expect(cell(nc, 'a', 'c')).toBeNaN()
    const nv = await nulls.cov().collect()
    expect(cell(nv, 'b', 'b')).toBeCloseTo(12.333333333333334, 12)
    expect(cell(nv, 'a', 'a')).toBeCloseTo(1.6666666666666667, 12) // all 4 rows
    expect(cell(nv, 'a', 'b')).toBeCloseTo(5.333333333333333, 12) // 3 complete rows
    expect(cell(nv, 'c', 'c')).toBe(0)
  })

  it('over(partitionBy): per-group aggregates broadcast to rows', async () => {
    // pandas: df.groupby('g')['x'].transform('mean') etc.
    const df = DataFrame.fromRows([
      { g: 'a', h: 1, x: 2, s: 'p' },
      { g: 'a', h: 1, x: 4, s: 'q' },
      { g: 'a', h: 2, x: 9, s: 'p' },
      { g: 'b', h: 1, x: 6, s: 'r' },
      { g: 'b', h: 1, x: null, s: 'r' },
      { g: 'c', h: 2, x: null, s: 's' },
    ])
    const out = await df
      .withColumns(
        col('x').mean().over('g').alias('gm'),
        col('x').sub(col('x').mean()).over('g').alias('dev'),
        col('x').div(col('x').sum().over('g')).alias('share'),
        col('x').std().over('g').alias('gsd'),
        col('x').count().over('g').alias('gn'),
        col('x').max().over(['g', 'h']).alias('ghmax'),
        col('x').median().over('g').alias('gmed'),
        col('s').nunique().over('g').alias('gsu'),
        col('s').first().over('g').alias('gs'),
        col('x').mean().over('g').sub(col('x').mean()).alias('gm_minus_mean'), // over + global broadcast together
      )
      .collect()
    const r = out.toArray()
    expect(out.columns).toEqual(['g', 'h', 'x', 's', 'gm', 'dev', 'share', 'gsd', 'gn', 'ghmax', 'gmed', 'gsu', 'gs', 'gm_minus_mean'])
    expect(r.map((v) => v.gm)).toEqual([5, 5, 5, 6, 6, null])
    expect(r.map((v) => v.dev)).toEqual([-3, -1, 4, 0, null, null])
    expect(r.map((v) => v.share)).toEqual([2 / 15, 4 / 15, 9 / 15, 1, null, null])
    expect((r[0]!.gsd as number)).toBeCloseTo(Math.sqrt(13), 12) // sample std of [2,4,9]
    expect(r[3]!.gsd).toBe(0) // single value → 0
    expect(r[5]!.gsd).toBeNull() // no values
    expect(r.map((v) => v.gn)).toEqual([3, 3, 3, 1, 1, 0])
    expect(r.map((v) => v.ghmax)).toEqual([4, 4, 9, 6, 6, null])
    expect(r.map((v) => v.gmed)).toEqual([4, 4, 4, 6, 6, null])
    expect(r.map((v) => v.gsu)).toEqual([2, 2, 2, 1, 1, 1])
    expect(r.map((v) => v.gs)).toEqual(['p', 'p', 'p', 'r', 'r', 's'])
    // global mean of x = (2+4+9+6)/4 = 5.25
    expect(r.map((v) => v.gm_minus_mean)).toEqual([-0.25, -0.25, -0.25, 0.75, 0.75, null])

    // filter / sort with over: rows above their group's mean; sort by deviation from group mean
    const above = await df.filter(col('x').gt(col('x').mean().over('g'))).collect()
    expect(above.toArray().map((v) => v.x)).toEqual([9])
    expect(above.columns).toEqual(['g', 'h', 'x', 's']) // no temp columns leak
    const sorted = await df.dropNull(['x']).sort(col('x').sub(col('x').mean()).over('g')).collect()
    expect(sorted.toArray().map((v) => v.x)).toEqual([2, 4, 6, 9])
    expect(sorted.columns).toEqual(['g', 'h', 'x', 's'])

    // numeric partition key (dense ints) and fused arithmetic around the over column
    const z = await df
      .dropNull(['x'])
      .withColumn('zg', col('x').sub(col('x').mean().over('h')).div(col('x').std().over('h')))
      .collect()
    // h=1: [2,4,6] mean 4 std 2 → [-1,0,1]; h=2: [9] std 0 → NaN
    const zg = z.toArray().map((v) => v.zg)
    expect(zg.slice(0, 2)).toEqual([-1, 0])
    expect(zg[2]).toBeNaN()
    expect(zg[3]).toBe(1)

    await expect(df.groupBy('g').agg({ m: col('x').mean().over('h') }).collect()).rejects.toThrow(/over()/)
  })

  it('over(partitionBy, { orderBy }): running aggregates within each group', async () => {
    // pandas: df.sort_values('t').groupby('g').x.cumsum() etc. — rows come back in the ORIGINAL order
    const df = DataFrame.fromRows([
      { id: 0, g: 'a', t: 3, x: 5 },
      { id: 1, g: 'a', t: 1, x: 2 },
      { id: 2, g: 'b', t: 2, x: 10 },
      { id: 3, g: 'a', t: 2, x: null },
      { id: 4, g: 'b', t: 1, x: 4 },
      { id: 5, g: 'a', t: 4, x: 1 },
    ])
    const out = await df
      .withColumns(
        col('x').sum().over('g', { orderBy: 't' }).alias('csum'),
        col('x').count().over('g', { orderBy: 't' }).alias('cnt'),
        col('x').mean().over('g', { orderBy: 't' }).alias('cmean'),
        col('x').max().over('g', { orderBy: 't' }).alias('cmax'),
        col('x').min().over('g', { orderBy: 't', descending: true }).alias('cmin_desc'),
        col('x').first().over('g', { orderBy: 't' }).alias('first'),
        col('x').last().over('g', { orderBy: 't' }).alias('last'),
        col('x').std().over('g', { orderBy: 't' }).alias('cstd'),
        col('x').sum().over([], { orderBy: 't' }).alias('gsum'), // no partition → running total in t order
        col('x').div(col('x').sum().over('g', { orderBy: 't' })).alias('share_so_far'),
      )
      .collect()
    const r = out.toArray()
    expect(out.columns.slice(0, 4)).toEqual(['id', 'g', 't', 'x'])
    // group a in t order: t=1 x=2 (id1), t=2 null (id3), t=3 x=5 (id0), t=4 x=1 (id5); group b: t=1 4 (id4), t=2 10 (id2)
    expect(r.map((v) => v.csum)).toEqual([7, 2, 14, 2, 4, 8])
    expect(r.map((v) => v.cnt)).toEqual([2, 1, 2, 1, 1, 3])
    expect(r.map((v) => v.cmean)).toEqual([3.5, 2, 7, 2, 4, 8 / 3])
    expect(r.map((v) => v.cmax)).toEqual([5, 2, 10, 2, 4, 5])
    // descending t: a: t=4 x=1 → t=3 x=5 → t=2 null → t=1 x=2; running min = 1,1,1,1; b: t=2 10 → t=1 4 → 10, 4
    expect(r.map((v) => v.cmin_desc)).toEqual([1, 1, 10, 1, 4, 1])
    expect(r.map((v) => v.first)).toEqual([2, 2, 4, 2, 4, 2])
    expect(r.map((v) => v.last)).toEqual([5, 2, 10, 2, 4, 1])
    expect(r[1]!.cstd).toBe(0)
    expect(r[0]!.cstd as number).toBeCloseTo(Math.sqrt(4.5), 12) // sd of [2, 5]
    expect(r[5]!.cstd as number).toBeCloseTo(Math.sqrt(((2 - 8 / 3) ** 2 + (5 - 8 / 3) ** 2 + (1 - 8 / 3) ** 2) / 2), 12)
    // global running total in t order: t=1: 2 (id1), 4 (id4) → 6 ; t=2: 10 (id2) → 16, null (id3) → 16 ; t=3: 5 → 21 ; t=4: 1 → 22
    expect(r.map((v) => v.gsum)).toEqual([21, 2, 16, 16, 6, 22])
    expect(r.map((v) => v.share_so_far)).toEqual([5 / 7, 1, 10 / 14, null, 1, 1 / 8])

    // a null before any valid value in its partition → null (count → 0)
    const lead = await DataFrame.fromRows([
      { g: 1, t: 1, x: null },
      { g: 1, t: 2, x: 3 },
    ])
      .withColumns(col('x').sum().over('g', { orderBy: 't' }).alias('s'), col('x').count().over('g', { orderBy: 't' }).alias('c'))
      .collect()
    expect(lead.toArray().map((v) => [v.s, v.c])).toEqual([
      [null, 0],
      [3, 1],
    ])
    // string orderBy works (comparator path); running median is rejected
    const byStr = await df.withColumn('s', col('x').sum().over('g', { orderBy: 'g' })).collect()
    expect(byStr.toArray().map((v) => v.s)).toEqual([5, 7, 10, 7, 14, 8])
    await expect(df.withColumn('m', col('x').median().over('g', { orderBy: 't' })).collect()).rejects.toThrow(/running median/)
  })

  it('rank handles ties (average by default; min/max/dense/ordinal)', async () => {
    // pandas: pd.Series([10, 20, 20, 30, 20, None]).rank() → [1, 3, 3, 5, 3, NaN]
    const df = DataFrame.fromRows([
      { id: 0, v: 10 },
      { id: 1, v: 20 },
      { id: 2, v: 20 },
      { id: 3, v: 30 },
      { id: 4, v: 20 },
      { id: 5, v: null },
    ])
    const byId = (d: DataFrame, key: string) =>
      d.toArray().sort((a, b) => (a.id as number) - (b.id as number)).map((r) => r[key])

    // Key from orderBy (partition pre-sorted) — ties are consecutive.
    const avg = await df.withWindow('r', 'rank', { orderBy: ['v'] }).collect()
    expect(byId(avg, 'r')).toEqual([1, 3, 3, 5, 3, null])

    // Key from expr without orderBy — ties are not consecutive in the input.
    const viaExpr = await df.withWindow('r', 'rank', { expr: col('v') }).collect()
    expect(byId(viaExpr, 'r')).toEqual([1, 3, 3, 5, 3, null])

    const min = await df.withWindow('r', 'rank', { orderBy: ['v'], method: 'min' }).collect()
    expect(byId(min, 'r')).toEqual([1, 2, 2, 5, 2, null])
    const max = await df.withWindow('r', 'rank', { orderBy: ['v'], method: 'max' }).collect()
    expect(byId(max, 'r')).toEqual([1, 4, 4, 5, 4, null])
    const dense = await df.withWindow('r', 'rank', { orderBy: ['v'], method: 'dense' }).collect()
    expect(byId(dense, 'r')).toEqual([1, 2, 2, 3, 2, null])
    const ordinal = await df.withWindow('r', 'rank', { expr: col('v'), method: 'ordinal' }).collect()
    expect(byId(ordinal, 'r')).toEqual([1, 2, 3, 5, 4, null])

    // Descending orderBy: rank 1 = largest; ties still averaged.
    const desc = await df.withWindow('r', 'rank', { orderBy: [col('v').desc()] }).collect()
    expect(byId(desc, 'r')).toEqual([5, 3, 3, 1, 3, null])

    // Ties averaged within each partition independently.
    const grouped = await DataFrame.fromRows([
      { g: 'a', v: 5 },
      { g: 'a', v: 5 },
      { g: 'b', v: 5 },
      { g: 'b', v: 7 },
    ])
      .withWindow('r', 'rank', { partitionBy: ['g'], orderBy: ['v'] })
      .collect()
    expect(grouped.toArray().map((r) => `${r.g}:${r.r}`).sort()).toEqual(['a:1.5', 'a:1.5', 'b:1', 'b:2'])
  })

  it('pass-through ops share column buffers (immutable contract)', async () => {
    const df = DataFrame.fromColumns({
      a: new Int32Array([1, 2, 3]),
      b: new Float64Array([10, 20, 30]),
    })
    const aData = df.getColumn('a').column.data

    const selected = await df.select('a', 'b').collect()
    expect(selected.getColumn('a').column.data).toBe(aData)

    const rolled = await df.rolling('m', 'b', 2, 'mean').collect()
    expect(rolled.getColumn('a').column.data).toBe(aData)
    expect(rolled.getColumn('b').column.data).toBe(df.getColumn('b').column.data)

    const aliased = await df.withColumn('a_copy', col('a')).collect()
    expect(aliased.getColumn('a_copy').column.data).toBe(aData)
    expect(aliased.getColumn('a').column.data).toBe(aData)

    // Documented contract: buffers are shared, so in-place mutation is visible on the source.
    ;(aliased.getColumn('a_copy').column.data as Int32Array)[0] = 99
    expect((df.getColumn('a').column.data as Int32Array)[0]).toBe(99)
  })

  it('withColumn arith allocates; comparisons yield bool; nulls propagate', async () => {
    const df = DataFrame.fromRows([
      { age: 20, salary: 40_000 },
      { age: 40, salary: null },
      { age: 50, salary: 90_000 },
    ])
    const src = df.getColumn('age').column.data

    const bonus = await df.withColumn('bonus', col('salary').mul(2)).collect()
    expect(bonus.getColumn('bonus').column.data).not.toBe(df.getColumn('salary').column.data)
    expect(bonus.toArray().map((r) => r.bonus)).toEqual([80_000, null, 180_000])

    const flag = await df.withColumn('ok', col('age').gt(30).and(col('salary').gt(50_000))).collect()
    expect(flag.dtypes.ok).toBe('bool')
    expect(flag.toArray().map((r) => r.ok)).toEqual([false, false, true])

    const plus = await df.withColumn('age1', col('age').add(1)).collect()
    expect(plus.getColumn('age1').column.data).not.toBe(src)
    expect(plus.toArray().map((r) => r.age1)).toEqual([21, 41, 51])
  })

  it('multi-key groupBy skips null keys; high-card falls back correctly', async () => {
    const { tableFromColumns, setValid } = await import('@columna/arrow')
    const cityNull = new Uint8Array(1)
    const catNull = new Uint8Array(1)
    // rows: 0 Berlin/a, 1 null/a, 2 Berlin/b, 3 Paris/null, 4 Paris/b
    for (const i of [0, 2, 3, 4]) setValid(cityNull, i, true)
    for (const i of [0, 1, 2, 4]) setValid(catNull, i, true)

    const withNull = new DataFrame(
      tableFromColumns([
        {
          field: { name: 'city', dtype: 'category', nullable: true },
          data: new Uint32Array([0, 0, 0, 1, 1]),
          dictionary: ['Berlin', 'Paris'],
          nullBitmap: cityNull,
        },
        {
          field: { name: 'category', dtype: 'category', nullable: true },
          data: new Uint32Array([0, 0, 1, 0, 1]),
          dictionary: ['a', 'b'],
          nullBitmap: catNull,
        },
        {
          field: { name: 'salary', dtype: 'f64', nullable: false },
          data: new Float64Array([10, 99, 30, 40, 60]),
        },
      ]),
    )
    const dense = await withNull
      .groupBy('city', 'category')
      .agg({ salary: 'mean' })
      .sort('city', 'category')
      .collect()
    // Dense category path drops null keys (pandas-like dropna).
    expect(dense.toArray()).toEqual([
      { city: 'Berlin', category: 'a', salary: 10 },
      { city: 'Berlin', category: 'b', salary: 30 },
      { city: 'Paris', category: 'b', salary: 60 },
    ])

    // Force hash/slow multi-key path: product of dict sizes > 65_536
    const dictA = Array.from({ length: 300 }, (_, i) => `a${i}`)
    const dictB = Array.from({ length: 300 }, (_, i) => `b${i}`)
    const n = 1_000
    const codesA = new Uint32Array(n)
    const codesB = new Uint32Array(n)
    const salary = new Float64Array(n)
    const id = new Int32Array(n)
    for (let i = 0; i < n; i++) {
      codesA[i] = i % 300
      codesB[i] = (i * 7) % 300
      salary[i] = i
      id[i] = i
    }
    const wide = DataFrame.fromColumns({
      ka: { codes: codesA, dictionary: dictA },
      kb: { codes: codesB, dictionary: dictB },
      salary,
      id,
    })
    const out = await wide.groupBy('ka', 'kb').agg({ salary: 'sum', id: 'count' }).collect()
    const rows = out.toArray()
    const g00 = rows.find((r) => r.ka === 'a0' && r.kb === 'b0')
    expect(g00).toBeTruthy()
    const expectedIdx: number[] = []
    for (let i = 0; i < n; i++) if (i % 300 === 0 && (i * 7) % 300 === 0) expectedIdx.push(i)
    expect(Number(g00!.salary)).toBe(expectedIdx.reduce((s, i) => s + i, 0))
    expect(Number(g00!.id)).toBe(expectedIdx.length)
  })

  it('unique multi-category keep last', async () => {
    const df = DataFrame.fromColumns({
      city: { codes: new Uint32Array([0, 0, 1, 0]), dictionary: ['Berlin', 'Paris'] },
      category: { codes: new Uint32Array([0, 1, 0, 0]), dictionary: ['a', 'b'] },
      v: new Int32Array([10, 20, 30, 11]),
    })
    const uniq = await df.unique(['city', 'category'], 'last').collect()
    expect(uniq.toArray()).toEqual([
      { city: 'Berlin', category: 'a', v: 11 },
      { city: 'Berlin', category: 'b', v: 20 },
      { city: 'Paris', category: 'a', v: 30 },
    ])
  })

  it('filter then select matches pushdown and stepwise', async () => {
    const df = DataFrame.fromColumns({
      id: new Int32Array([1, 2, 3, 4]),
      age: new Int32Array([20, 40, 30, 50]),
      salary: new Float64Array([10, 20, 30, 40]),
      city: { codes: new Uint32Array([0, 1, 0, 1]), dictionary: ['Berlin', 'Paris'] },
    })
    const pred = col('age').gt(25)

    const pushed = await df.filter(pred).select('id', 'city', 'salary').collect()
    const stepwise = await (await df.filter(pred).collect()).select('id', 'city', 'salary').collect()

    expect(pushed.toArray()).toEqual(stepwise.toArray())
    expect(pushed.toArray()).toEqual([
      { id: 2, city: 'Paris', salary: 20 },
      { id: 3, city: 'Berlin', salary: 30 },
      { id: 4, city: 'Paris', salary: 40 },
    ])
    expect(pushed.columns).toEqual(['id', 'city', 'salary'])
  })

  it('join dense and sparse numeric keys agree', async () => {
    const denseLeft = DataFrame.fromColumns({
      user_id: new Int32Array([0, 1, 2, 3]),
      v: new Int32Array([10, 20, 30, 40]),
    })
    const denseRight = DataFrame.fromColumns({
      user_id: new Int32Array([1, 2, 9]),
      score: new Int32Array([100, 200, 900]),
    })
    const dense = await denseLeft.innerJoin(denseRight, 'user_id').sort('user_id').collect()
    expect(dense.toArray()).toEqual([
      { user_id: 1, v: 20, score: 100 },
      { user_id: 2, v: 30, score: 200 },
    ])

    // Large key span forces hash probe instead of dense Int32 table
    const sparseLeft = DataFrame.fromColumns({
      user_id: new Int32Array([0, 10_000_000, 20_000_000, 30_000_000]),
      v: new Int32Array([10, 20, 30, 40]),
    })
    const sparseRight = DataFrame.fromColumns({
      user_id: new Int32Array([10_000_000, 20_000_000, 99_000_000]),
      score: new Int32Array([100, 200, 900]),
    })
    const sparse = await sparseLeft.innerJoin(sparseRight, 'user_id').sort('user_id').collect()
    expect(sparse.toArray()).toEqual([
      { user_id: 10_000_000, v: 20, score: 100 },
      { user_id: 20_000_000, v: 30, score: 200 },
    ])
  })

  it('sort head top-k matches full sort then limit', async () => {
    const df = DataFrame.fromColumns({
      id: new Int32Array([1, 2, 3, 4, 5]),
      v: new Float64Array([50, 10, 40, 20, 30]),
    })
    const topk = await df.sort(col('v').desc()).head(3).collect()
    const full = await df.sort(col('v').desc()).collect()
    const limited = await full.head(3).collect()
    expect(topk.toArray()).toEqual(limited.toArray())
    expect(topk.toArray().map((r) => r.v)).toEqual([50, 40, 30])
  })

  it('valueCounts on category with normalize', async () => {
    const df = DataFrame.fromColumns({
      city: { codes: new Uint32Array([0, 1, 0, 0, 1]), dictionary: ['Berlin', 'Paris'] },
    })
    const counts = await df.valueCounts('city').collect()
    expect(counts.toArray()).toEqual([
      { city: 'Berlin', count: 3 },
      { city: 'Paris', count: 2 },
    ])

    const props = await df.valueCounts('city', true).collect()
    expect(props.columns).toContain('proportion')
    const rows = props.toArray()
    expect(rows).toEqual([
      { city: 'Berlin', proportion: 0.6 },
      { city: 'Paris', proportion: 0.4 },
    ])
  })
})

describe('Series reductions', () => {
  it('sum/mean/min/max stream over large columns (no spread → no stack overflow) and skip nulls', () => {
    const n = 500_000
    const data = new Float64Array(n)
    for (let i = 0; i < n; i++) data[i] = i
    const s = DataFrame.fromColumns({ v: data }).getColumn('v')
    expect(s.max()).toBe(n - 1)
    expect(s.min()).toBe(0)
    expect(s.sum()).toBe((n * (n - 1)) / 2)
    expect(s.mean()).toBe((n - 1) / 2)

    const withNulls = DataFrame.fromRows([{ v: 3 }, { v: null }, { v: -1 }]).getColumn('v')
    expect(withNulls.max()).toBe(3)
    expect(withNulls.min()).toBe(-1)
    expect(withNulls.mean()).toBe(1)
    expect(DataFrame.fromRows([{ v: null }]).getColumn('v').max()).toBeNull()
    expect(DataFrame.fromRows([{ s: 'a' }]).getColumn('s').max()).toBeNull()
  })
})
