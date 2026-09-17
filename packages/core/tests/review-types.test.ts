/**
 * Compile-time regressions from the external recheck (commit 0fa6d84).
 * Checked by `tsc -p packages/core/tsconfig.tests.json` (and vitest for discovery).
 */
import { expectTypeOf, it } from 'vitest'
import { DataFrame } from '@columna/core'

it('shift introduces a nullable result even for a non-null input', () => {
  const plan = DataFrame.fromRows([{ x: 1 }]).withColumn('previous', (c) => c.x.shift())
  type Collected = Awaited<ReturnType<typeof plan.collect>>
  type Output = ReturnType<Collected['toArray']>[number]
  expectTypeOf<Output['previous']>().toEqualTypeOf<number | null>()
})

/**
 * Recheck asked for `Date`, but the public contract is epoch milliseconds (`number`):
 * `fromColumns` / `fromRows` store datetime as ms, and `toArray()` returns the same
 * (see `InferColumns` / `p1-correctness`). Returning `Date` would diverge from the schema type.
 */
it('fromColumns Date arrays are returned as epoch ms by DataFrame.toArray', () => {
  const frame = DataFrame.fromColumns({ timestamp: [new Date(0)] })
  type Output = ReturnType<typeof frame.toArray>[number]
  expectTypeOf<Output['timestamp']>().toEqualTypeOf<number>()
})
