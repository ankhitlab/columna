import assert from 'node:assert/strict'
import { it } from 'vitest'
import { DataFrame, col, lit, when, createSession } from '../src/index.js'
import { parseCsvToRows } from '../src/io/csv.js'

it('F01: preserves category labels through shift and nested aliases', async () => {
  const df = DataFrame.fromRows([{ x: 'a' }, { x: 'b' }, { x: null }])
  assert.deepEqual(await df.withColumn('previous', col('x').shift()).toArray(), [
    { x: 'a', previous: null }, { x: 'b', previous: 'a' }, { x: null, previous: 'b' },
  ])
  const result = await df.select(col('x').shift().alias('p')).collect()
  assert.deepEqual(result.toArray(), [{ p: null }, { p: 'a' }, { p: 'b' }])
  assert.equal(result.col('p').dtype, 'utf8')
})

it('F01: boolean shift remains boolean, including its null bitmap', async () => {
  const out = await DataFrame.fromRows([{ x: true }, { x: false }, { x: true }])
    .withColumn('p', col('x').shift()).collect()
  assert.deepEqual(out.col('p').toArray(), [null, true, false])
  assert.equal(out.col('p').dtype, 'bool')
})

it('F01: a boolean literal does not materialize as a numeric scalar', async () => {
  const result = await DataFrame.fromRows([{ x: 1 }]).withColumn('flag', lit(true)).collect()
  assert.deepEqual(result.toArray(), [{ x: 1, flag: true }])
  assert.equal(result.col('flag').dtype, 'bool')
})

it('F01: static type survives an empty table and an all-null shift', async () => {
  const empty = DataFrame.fromColumns({ x: { codes: new Uint32Array(), dictionary: ['a'] } })
  const a = await empty.withColumn('p', col('x').shift()).collect()
  assert.equal(a.col('p').dtype, 'utf8')
  assert.deepEqual(a.toArray(), [])
  const b = await DataFrame.fromRows([{ x: 'a' }]).withColumn('p', col('x').shift(5)).collect()
  assert.equal(b.col('p').dtype, 'utf8')
  assert.deepEqual(b.col('p').toArray(), [null])
})

it('F01: conditional output does not depend on its first observed value', async () => {
  const df = DataFrame.fromRows([{ x: 0 }, { x: 1 }])
  const text = await df.withColumn('y', when(col('x').eq(0)).then(null).otherwise('ok')).collect()
  const flag = await df.withColumn('y', when(col('x').eq(0)).then(null).otherwise(true)).collect()
  assert.deepEqual(text.col('y').toArray(), [null, 'ok'])
  assert.deepEqual(flag.col('y').toArray(), [null, true])
  assert.equal(flag.col('y').dtype, 'bool')
})

it('F01: UDF results are evaluated once per row, even after a leading null', async () => {
  let calls = 0
  const df = DataFrame.fromRows([{ x: 0 }, { x: 1 }, { x: 2 }])
  const out = await df.withColumn('y', col('x').mapElements((v) => {
    calls++
    return v === 0 ? null : `call-${calls}`
  })).collect()
  assert.equal(calls, 3)
  assert.deepEqual(out.col('y').toArray(), [null, 'call-2', 'call-3'])
})

it('F01: nested UDFs keep conditional short-circuiting and run only once', async () => {
  let calls = 0
  const udf = col('x').mapElements(() => { calls++; return 'yes' })
  const out = await DataFrame.fromRows([{ x: 0 }, { x: 1 }, { x: 2 }])
    .withColumn('y', when(col('x').eq(0)).then(null).otherwise(udf)).collect()
  assert.deepEqual(out.col('y').toArray(), [null, 'yes', 'yes'])
  assert.equal(calls, 2)
})

it('F01: mixed UDF primitive families fail rather than corrupting data', async () => {
  const df = DataFrame.fromRows([{ x: 0 }, { x: 1 }])
  await assert.rejects(() => df.withColumn('y', col('x').mapElements((v) => v === 0 ? 1 : 'a')).collect(),
    /mixed result types.*explicit cast/)
  const cast = await df.withColumn('y', col('x').mapElements((v) => v === 0 ? 1 : 'a').cast('utf8')).collect()
  assert.deepEqual(cast.col('y').toArray(), ['1', 'a'])
})

it('F01: homogeneous numeric UDFs widen integer-looking values without truncation', async () => {
  const out = await DataFrame.fromRows([{ x: 0 }, { x: 1 }, { x: 2 }])
    .withColumn('y', col('x').mapElements((v) => v === 0 ? 1 : v === 1 ? 1.5 : 2 ** 40)).collect()
  assert.deepEqual(out.col('y').toArray(), [1, 1.5, 2 ** 40])
})

it('F01: a shifted datetime preserves its dtype and millisecond values', async () => {
  const out = await DataFrame.fromRows([{ x: new Date(0) }, { x: new Date(1000) }])
    .withColumn('p', col('x').shift()).collect()
  assert.equal(out.col('p').dtype, 'datetime')
  assert.deepEqual(out.col('p').toArray(), [null, 0])
})

for (const how of ['inner', 'left', 'right', 'outer', 'semi', 'anti'] as const) {
  it(`F03: rename/join/project agrees with a materialization barrier (${how})`, async () => {
    const left = DataFrame.fromRows([{ id: 1, x: 10 }, { id: 2, x: 11 }]).rename({ x: 'y' })
    const right = DataFrame.fromRows([{ id: 1, x: 20 }, { id: 3, x: 30 }])
    const joined = left.join(right, { on: 'id', how })
    const names = how === 'semi' || how === 'anti' ? ['id', 'y'] : ['id', 'x', 'y']
    const expected = await (await joined.collect()).select(...names.map(col)).toArray()
    assert.deepEqual(await joined.select(...names.map(col)).toArray(), expected)
  })
}

it('F03: added and dropped columns are classified by the child output', async () => {
  const left = DataFrame.fromRows([{ id: 1, x: 10 }]).drop('x').withColumn('y', lit(7))
  const right = DataFrame.fromRows([{ id: 1, x: 20 }])
  const joined = left.join(right, { on: 'id' })
  assert.deepEqual(await joined.select('x', 'y').toArray(), [{ x: 20, y: 7 }])
})

it('F03: unknown nested-join provenance conservatively skips pruning', async () => {
  const left = DataFrame.fromRows([{ id: 1, a: 10 }])
    .join(DataFrame.fromRows([{ id: 1, b: 20 }]), { on: 'id' })
  const query = left.join(DataFrame.fromRows([{ id: 1, c: 30 }]), { on: 'id' })
  assert.deepEqual(await query.select('b', 'c').toArray(), [{ b: 20, c: 30 }])
})

it('F03: pruning does not remove collisions and change suffix allocation', async () => {
  const left = DataFrame.fromRows([{ id: 1, x: 10 }])
  const right = DataFrame.fromRows([{ id: 1, x: 20 }])
  const query = left.join(right, { on: 'id', lSuffix: '_l', rSuffix: '_r' })
  const expected = await (await query.collect()).select(col('x_l'), col('x_r')).toArray()
  assert.deepEqual(await query.select(col('x_l'), col('x_r')).toArray(), expected)
  // There is no unsuffixed x. Pruning must not make that invalid request start succeeding.
  await assert.rejects(() => query.select(col('x')).collect(), /Unknown column/)
})

it('F04: computation still uses runtime-generated names; alias is explicit', async () => {
  const df = DataFrame.fromRows([{ x: 1 }])
  assert.deepEqual(await df.select((c) => [c.x.add(1)]).toArray(), [{ expr_0: 2 }])
  assert.deepEqual(await df.select((c) => [c.x.add(1).alias('x')]).toArray(), [{ x: 2 }])
})

it('F05: Date and bigint values match the normalized constructor types', () => {
  assert.deepEqual(DataFrame.fromRows([{ time: new Date(0), id: 42n }]).toArray(), [{ time: 0, id: 42 }])
  assert.deepEqual(DataFrame.fromRows([{ id: 9007199254740993n }], { int64: 'string' }).toArray(), [{ id: '9007199254740993' }])
  assert.throws(() => DataFrame.fromRows([{ id: 9007199254740993n }]), /precision|exact|64|53/i)
})

it('F05: Session forwards the bigint policy and retains its runtime', () => {
  const session = createSession({ backends: [] })
  try {
    const df = session.fromRows([{ id: 9007199254740993n }], { int64: 'string' })
    assert.deepEqual(df.toArray(), [{ id: '9007199254740993' }])
    assert.equal(df.getRuntime(), session.runtime)
  } finally { session.close() }
})

for (const separator of ['||', '::', '<>', '|||', '💠']) {
  it(`F07: quoted, escaped and multiline records agree across CSV paths (${separator})`, async () => {
    const csv = `x${separator}y\n"a${separator}b"${separator}2\n"a""b"${separator}3\n"line\nbreak"${separator}4\nb${separator}5`
    const expected = [{ x: `a${separator}b`, y: 2 }, { x: 'a"b', y: 3 }, { x: 'line\nbreak', y: 4 }, { x: 'b', y: 5 }]
    assert.deepEqual(DataFrame.fromCSV(csv, { separator }).toArray(), expected)
    assert.deepEqual(parseCsvToRows(csv, { separator }), expected)
    assert.deepEqual((await DataFrame.readCsv(csv, { separator, content: true })).toArray(), expected)
  })
}

it('F07: empty separator throws instead of entering a non-advancing loop', () => {
  assert.throws(() => DataFrame.fromCSV('x\n1', { separator: '' }), /separator must not be empty/)
})

it('F07: a quoted empty string stays distinct from a bare empty field', () => {
  const result = DataFrame.fromCSV('x||y\n""||1\n||2', { separator: '||' }).toArray()
  assert.deepEqual(result, [{ x: '', y: 1 }, { x: null, y: 2 }])
})

it('F08: null, the string null and duplicates have distinct identities', () => {
  const s = DataFrame.fromRows([{ x: null }, { x: 'null' }, { x: 'a' }, { x: null }]).col('x')
  assert.deepEqual(s.unique(), [null, 'null', 'a'])
  assert.equal(s.nunique(), 3)
})

it('F08: NaN is deduplicated and bool/null identities are preserved', () => {
  assert.deepEqual(DataFrame.fromColumns({ x: new Float64Array([NaN, NaN, 0, -0, 1]) }).col('x').unique(), [NaN, 0, 1])
  assert.deepEqual(DataFrame.fromRows([{ x: true }, { x: null }, { x: false }, { x: true }]).col('x').unique(), [true, null, false])
})

for (const n of [-2, -0, 0, 1, 1.9, 2, 5]) {
  it(`F09: eager/lazy/series head and tail share count semantics (${n})`, async () => {
    const df = DataFrame.fromRows([{ x: 1 }, { x: null }, { x: 3 }])
    const count = Math.max(0, Math.trunc(n))
    const head = df.toArray().slice(0, count)
    const tail = count === 0 ? [] : df.toArray().slice(-count)
    assert.deepEqual(df.head(n).toArray(), head)
    assert.deepEqual(await df.lazy().head(n).toArray(), head)
    assert.deepEqual(df.col('x').head(n), head.map((row) => row.x))
    assert.deepEqual(df.tail(n).toArray(), tail)
    assert.deepEqual(await df.lazy().tail(n).toArray(), tail)
    assert.deepEqual(df.col('x').tail(n), tail.map((row) => row.x))
  })
}

for (const n of [NaN, Infinity, -Infinity]) {
  it(`F09: all peek methods reject non-finite counts (${n})`, () => {
    const df = DataFrame.fromRows([{ x: 1 }])
    for (const value of [df, df.lazy(), df.col('x')]) {
      assert.throws(() => value.head(n), /count must be finite/)
      assert.throws(() => value.tail(n), /count must be finite/)
    }
  })
}

it('F09: Series peeks decode only a range and never call full toArray', () => {
  const s = DataFrame.fromRows([{ x: 'a' }, { x: null }, { x: 'b' }]).col('x')
  s.toArray = () => { throw new Error('full materialization is forbidden in a peek') }
  assert.deepEqual(s.head(2), ['a', null])
  assert.deepEqual(s.tail(2), [null, 'b'])
  assert.deepEqual(s.tail(0), [])
})

it('F10: unequal key arity fails before execution', () => {
  const a = DataFrame.fromRows([{ a: 1, b: 2 }])
  const b = DataFrame.fromRows([{ c: 1, d: 2 }])
  assert.throws(() => a.join(b, { leftOn: ['a', 'b'], rightOn: ['c'] }), /leftOn has 2 keys, rightOn has 1/)
})

it('F10: empty key arrays are rejected but an explicit cross join still works', async () => {
  const a = DataFrame.fromRows([{ a: 1 }, { a: 2 }])
  const b = DataFrame.fromRows([{ b: 3 }, { b: 4 }])
  assert.throws(() => a.join(b, { on: [] }), /keys must not be empty/)
  assert.equal((await a.join(b, { how: 'cross' }).toArray()).length, 4)
})

it('F10: missing known-schema keys fail early with side and available columns', () => {
  const df = DataFrame.fromRows([{ id: 1 }])
  assert.throws(() => df.join(df, { leftOn: 'missing', rightOn: 'id' }), /unknown left key.*missing.*available columns: id/)
  assert.throws(() => df.join(df, { leftOn: 'id', rightOn: 'missing' }), /unknown right key.*missing/)
})

it('F10: valid composite keys are copied rather than aliased to caller arrays', async () => {
  const a = DataFrame.fromRows([{ a: 1, b: 2 }])
  const b = DataFrame.fromRows([{ c: 1, d: 2 }])
  const leftOn = ['a', 'b']
  const rightOn = ['c', 'd']
  const query = a.join(b, { leftOn, rightOn })
  leftOn.length = 0
  rightOn.push('missing')
  assert.equal((await query.collect()).shape[0], 1)
})

it('F03: unaliased aggregate projections use their real generated names', async () => {
  const left = DataFrame.fromRows([{ id: 1, x: 10 }]).select('id', col('x').sum())
  const right = DataFrame.fromRows([{ id: 1, sum_x: 20 }])
  const joined = left.join(right, { on: 'id' })
  const expected = await (await joined.collect()).select(col('sum_x')).toArray()
  assert.deepEqual(expected, [{ sum_x: 10 }])
  assert.deepEqual(await joined.select(col('sum_x')).toArray(), expected)
})

it('F01: preserve existing numeric min/max aggregate semantics', async () => {
  const text = await DataFrame.fromRows([{ x: '10' }, { x: '2' }])
    .withColumn('minimum', col('x').min()).collect()
  assert.deepEqual(text.col('minimum').toArray(), [2, 2])
  const flags = await DataFrame.fromRows([{ x: true }, { x: false }])
    .withColumn('maximum', col('x').max()).collect()
  assert.deepEqual(flags.col('maximum').toArray(), [1, 1])
})
