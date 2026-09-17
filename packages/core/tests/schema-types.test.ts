import { describe, expect, expectTypeOf, it } from 'vitest'
import { DataFrame, LazyFrame, col, cols, lit, when, type Row } from '../src/index.js'
import type { Expr } from '../src/expr.js'

/** Schema carried by a frame (compared instead of the class, whose generic methods confuse structural equality). */
type SchemaOf<F> = F extends LazyFrame<infer S> ? S : F extends DataFrame<infer S> ? S : never
type ValueOf<E> = E extends Expr<infer T, any> ? T : never

/**
 * Compile-time schema tracking. These assertions are checked by `tsc` (packages/core/tsconfig.tests.json)
 * and the `@ts-expect-error` lines fail the build if the API stops rejecting them; the runtime expectations
 * confirm the typed calls execute like the untyped ones.
 */
describe('schema-typed frames', () => {
  const rows = [
    { city: 'Berlin', age: 30, salary: 72000, active: true },
    { city: 'Paris', age: 41, salary: 91000, active: false },
    { city: 'Berlin', age: 22, salary: 48000, active: true },
  ]

  it('fromRows infers the row type; select / drop / rename / withColumn track columns', async () => {
    const df = DataFrame.fromRows(rows)
    expectTypeOf<SchemaOf<typeof df>>().toEqualTypeOf<{ city: string; age: number; salary: number; active: boolean }>()

    const picked = df.select('city', 'age')
    expectTypeOf<SchemaOf<typeof picked>>().toEqualTypeOf<{ city: string; age: number }>()
    // @ts-expect-error — no such column
    df.select('cty')

    const dropped = df.drop('active')
    expectTypeOf<SchemaOf<typeof dropped>>().toEqualTypeOf<{ city: string; age: number; salary: number }>()
    // @ts-expect-error — cannot drop a column that does not exist
    df.drop('nope')

    const renamed = df.rename({ salary: 'pay' })
    expectTypeOf<SchemaOf<typeof renamed>>().toEqualTypeOf<{ city: string; age: number; active: boolean; pay: number }>()
    // @ts-expect-error — renaming a column that does not exist
    df.rename({ wages: 'pay' })

    const withZ = df.withColumn('z', (c) => c.salary.div(1000))
    expectTypeOf<SchemaOf<typeof withZ>>().toEqualTypeOf<{ city: string; age: number; salary: number; active: boolean; z: number }>()
    const out = (await withZ.collect()).toArray()
    expectTypeOf(out).toEqualTypeOf<Array<{ city: string; age: number; salary: number; active: boolean; z: number }>>()
    expect(out[0]).toEqual({ city: 'Berlin', age: 30, salary: 72000, active: true, z: 72 })

    // the compiler knows the value type of the new column
    const label = df.withColumn('label', (c) => c.city.str.toUpperCase())
    expectTypeOf<SchemaOf<typeof label>>().toEqualTypeOf<{ city: string; age: number; salary: number; active: boolean; label: string }>()
    expect((await label.collect()).toArray()[1]!.label).toBe('PARIS')
  })

  it('typed column refs: names and value types are checked in filter / sort / withColumns', async () => {
    const df = DataFrame.fromRows(rows)
    const adults = df.filter((c) => c.age.gt(18).and(c.active))
    expectTypeOf<SchemaOf<typeof adults>>().toEqualTypeOf<{ city: string; age: number; salary: number; active: boolean }>()
    expect((await adults.collect()).shape[0]).toBe(2)
    // @ts-expect-error — unknown column in the callback
    df.filter((c) => c.wages.gt(0))
    // @ts-expect-error — a number is not a predicate
    df.filter((c) => c.salary.add(1))
    // @ts-expect-error — neither is a string column
    df.filter((c) => c.city)

    const sorted = await df.sort((c) => c.salary.desc()).collect()
    expect(sorted.toArray().map((r) => r.city)).toEqual(['Paris', 'Berlin', 'Berlin'])
    // @ts-expect-error — sort key must be a column of the schema
    df.sort('salaray')

    const extended = df.withColumns((c) => [c.salary.mul(12).alias('yearly'), c.age.gte(40).alias('senior')])
    expectTypeOf<SchemaOf<typeof extended>>().toEqualTypeOf<{ city: string; age: number; salary: number; active: boolean; yearly: number; senior: boolean }>()
    const ext = (await extended.collect()).toArray()
    expect(ext[1]).toMatchObject({ yearly: 91000 * 12, senior: true })

    // untyped col() stays usable and stays loose: it is Expr<any>, so it passes where a boolean is required
    expect((await df.filter(col('age').gt(30)).collect()).shape[0]).toBe(1)
    // …but a typed non-boolean result is still rejected
    // @ts-expect-error — Expr<number> is not a predicate
    df.filter(col('age').add(1))
  })

  it('groupBy / agg result types: keys keep their type, aggregates are numbers unless the expression says otherwise', async () => {
    const df = DataFrame.fromRows(rows)
    const grouped = df.groupBy('city').agg((c) => ({ n: c.age.count(), pay: c.salary.mean(), any: c.active.max(), first: c.city.first() }))
    expectTypeOf<SchemaOf<typeof grouped>>().toEqualTypeOf<{ city: string; n: number; pay: number; any: boolean; first: string }>()
    const g = (await grouped.sort('city').collect()).toArray()
    expect(g[0]).toMatchObject({ city: 'Berlin', n: 2, pay: 60000 })
    // @ts-expect-error — grouping by a column that does not exist
    df.groupBy('country')
    const byName = df.groupBy('city').agg({ salary: 'mean' })
    expectTypeOf<SchemaOf<typeof byName>>().toEqualTypeOf<{ city: string; salary: number }>()
  })

  it('join result types combine both schemas; left joins make the right side nullable', async () => {
    const df = DataFrame.fromRows(rows)
    const regions = DataFrame.fromRows([{ city: 'Berlin', region: 'DE' }])
    const inner = df.join(regions, { on: 'city' })
    expectTypeOf<SchemaOf<typeof inner>>().toEqualTypeOf<{ city: string; age: number; salary: number; active: boolean; region: string }>()
    const left = df.leftJoin(regions, 'city')
    expectTypeOf((await left.collect()).toArray()[0]!.region).toEqualTypeOf<string | null>()
    expect((await left.collect()).toArray().map((r) => r.region)).toEqual(['DE', null, 'DE'])
    const semi = df.semiJoin(regions, 'city')
    expectTypeOf<SchemaOf<typeof semi>>().toEqualTypeOf<{ city: string; age: number; salary: number; active: boolean }>()
  })

  it('fromColumns infers types from buffers and arrays; getColumn is typed; untyped code keeps compiling', async () => {
    const df = DataFrame.fromColumns({ x: new Float64Array([1, 2]), n: new Int32Array([3, 4]), s: ['a', 'b'], b: new Uint8Array([1, 0]) })
    expectTypeOf<SchemaOf<typeof df>>().toEqualTypeOf<{ x: number; n: number; s: string; b: boolean }>()
    expectTypeOf(df.getColumn('s').toArray()).toEqualTypeOf<string[]>()
    expect(df.getColumn('s').toArray()).toEqual(['a', 'b'])
    // @ts-expect-error — no such column (and a runtime error too)
    expect(() => df.getColumn('y')).toThrow(/Unknown column/)

    // schema assertion on readers: the caller states the shape, the compiler cannot verify it
    const csv = DataFrame.fromCSV<{ a: number; b: number }>('a,b\n1,2\n')
    expectTypeOf(csv.toArray()).toEqualTypeOf<Array<{ a: number; b: number }>>()
    const loose = DataFrame.fromCSV('a,b\n1,2\n')
    expectTypeOf<SchemaOf<typeof loose>>().toEqualTypeOf<Row>()
    expectTypeOf(loose.toArray()).toEqualTypeOf<Row[]>()
    // untyped frames accept any string column name, as before
    expect((await loose.select('a').collect()).columns).toEqual(['a'])

    // cols<S>() gives typed refs outside callbacks; when/then/otherwise types the union of its branches
    const c = cols<{ age: number; city: string }>()
    const band = when(c.age.gt(30)).then('senior').otherwise(lit(0))
    expectTypeOf<ValueOf<typeof band>>().toEqualTypeOf<'senior' | 0>() // literal branch types
  })
})
