import assert from 'node:assert/strict'
import { it } from 'vitest'
import { DataFrame, createSession, type FromRowsOptions } from '../src/index.js'

// Calls have no runtime assertions: TypeScript checks assignability and @ts-expect-error guards.
function accepts<T>(_value: T): void {}

it('F04 types: only col and a final alias promise a known output name', async () => {
  const df = DataFrame.fromRows([{ x: 1 }])
  const unnamed = (await df.select((c) => [c.x.add(1)]).collect()).toArray()[0]!
  // @ts-expect-error computed output is not named x without an alias
  accepts<number>(unnamed.x)
  const named = (await df.select((c) => [c.x.add(1).alias('x')]).collect()).toArray()[0]!
  accepts<number>(named.x)
  assert.equal(named.x, 2)
  const plain = (await df.select((c) => [c.x]).collect()).toArray()[0]!
  accepts<number>(plain.x)
  const renamedThenComputed = (await df.select((c) => [c.x.alias('old').mul(2)]).collect()).toArray()[0]!
  // @ts-expect-error alias is not propagated through a later transformation at runtime
  accepts<number>(renamedThenComputed.old)
  assert.deepEqual(renamedThenComputed, { expr_0: 2 })
})

it('F05 types: normalized dates and bigint policies match the runtime', () => {
  const row = DataFrame.fromRows([{ time: new Date(0), id: 42n }]).toArray()[0]!
  accepts<number>(row.time)
  accepts<number>(row.id)
  // @ts-expect-error Date input is normalized to epoch milliseconds
  accepts<Date>(row.time)
  // @ts-expect-error safe bigint input is normalized to number
  accepts<bigint>(row.id)
  assert.deepEqual(row, { time: 0, id: 42 })
  const text = DataFrame.fromRows([{ id: 42n }], { int64: 'string' }).toArray()[0]!
  accepts<string>(text.id)
  // @ts-expect-error string policy never promises a bigint output
  accepts<bigint>(text.id)
  const explicit = DataFrame.fromRows<{ id: bigint }>([{ id: 42n }], { int64: 'string' }).toArray()[0]!
  accepts<string>(explicit.id)
  assert.equal(explicit.id, '42')
  const lossy = DataFrame.fromRows([{ id: 42n }], { int64: 'number' }).toArray()[0]!
  accepts<number>(lossy.id)
  const options: FromRowsOptions = { int64: 'string' }
  const dynamic = DataFrame.fromRows([{ id: 42n }], options).toArray()[0]!
  accepts<number | string>(dynamic.id)
})

it('F05 types: missing and undefined cells become null, not undefined', () => {
  const row = DataFrame.fromRows([{ a: 1 }, { b: 2 }]).toArray()[0]!
  accepts<number | null>(row.a)
  accepts<number | null>(row.b)
  // @ts-expect-error not every input row had column b
  accepts<number>(row.b)
  assert.deepEqual(row, { a: 1, b: null })
  const optional = DataFrame.fromRows<{ value?: number }>([{ value: 1 }, {}]).toArray()[1]!
  accepts<number | null>(optional.value)
  assert.equal(optional.value, null)
})

it('F05 types: sessions use the same normalization and retain typed policies', () => {
  const session = createSession({ backends: [] })
  try {
    const number = session.fromRows([{ id: 42n, time: new Date(0) }]).toArray()[0]!
    accepts<number>(number.id)
    accepts<number>(number.time)
    const text = session.fromRows<{ id: bigint }>([{ id: 42n }], { int64: 'string' }).toArray()[0]!
    accepts<string>(text.id)
    assert.equal(text.id, '42')
  } finally { session.close() }
})
