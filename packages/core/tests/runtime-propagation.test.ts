/**
 * Runtime propagation for operations that take several frames (concat, join, cross join, as-of join): the
 * result runs on the inputs' runtime, never silently on the process default, and frames of two different
 * sessions are not combined without saying which runtime executes (RuntimeMismatchError).
 */
import { describe, expect, it } from 'vitest'
import { DataFrame, LazyFrame, RuntimeMismatchError, col, createSession, getDefaultRuntime } from '../src/index.js'

const rowsA = [
  { id: 1, g: 'x', t: 10 },
  { id: 2, g: 'y', t: 20 },
]
const rowsB = [
  { id: 3, g: 'x', t: 15 },
  { id: 4, g: 'z', t: 30 },
]
const dims = [
  { g: 'x', w: 1 },
  { g: 'y', w: 2 },
]

// every multi-input operator, from both LazyFrame and DataFrame receivers
const ops: Record<string, (a: DataFrame, b: DataFrame) => LazyFrame<any>> = {
  'LazyFrame.concat vertical': (a, b) => LazyFrame.concat([a.lazy(), b.lazy()]),
  'LazyFrame.concat of DataFrames': (a, b) => LazyFrame.concat([a, b]),
  'LazyFrame.concat horizontal': (a, b) => LazyFrame.concat([a.select('id'), b.select('t').rename({ t: 't2' })], 'horizontal'),
  'join inner': (a, b) => a.lazy().join(b.lazy(), { on: 'g' }),
  'join left (DataFrame receiver)': (a, b) => a.join(b, { on: 'g', how: 'left' }),
  'join outer': (a, b) => a.lazy().outerJoin(b, 'g'),
  'semi join': (a, b) => a.semiJoin(b, 'g'),
  'anti join': (a, b) => a.lazy().antiJoin(b.lazy(), 'g'),
  'cross join': (a, b) => a.crossJoin(b),
  'as-of join': (a, b) => a.lazy().sort('t').joinAsof(b.lazy().sort('t'), { leftOn: 't' }),
}

describe('multi-input operators keep the session runtime', () => {
  for (const [name, op] of Object.entries(ops)) {
    it(`${name}: both inputs on one session → the result is on that session`, async () => {
      const s = createSession()
      const out = op(s.fromRows(rowsA), s.fromRows(rowsB))
      expect(out.getRuntime()).toBe(s.runtime)
      // and stays there after execution and through derived plans
      const frame = await out.collect()
      expect(frame.getRuntime()).toBe(s.runtime)
      expect(frame.lazy().filter(col('id').gt(0)).getRuntime()).toBe(s.runtime)
    })

    it(`${name}: a session frame with an unbound (default-runtime) frame → the session, in either order`, () => {
      const s = createSession()
      expect(op(s.fromRows(rowsA), DataFrame.fromRows(rowsB)).getRuntime()).toBe(s.runtime)
      expect(op(DataFrame.fromRows(rowsA), s.fromRows(rowsB)).getRuntime()).toBe(s.runtime)
    })

    it(`${name}: frames of two different sessions → RuntimeMismatchError unless { runtime } is given`, () => {
      const s1 = createSession()
      const s2 = createSession()
      expect(() => op(s1.fromRows(rowsA), s2.fromRows(rowsB))).toThrow(RuntimeMismatchError)
    })
  }

  it('an explicit { runtime } chooses the runtime of the result', () => {
    const s1 = createSession()
    const s2 = createSession()
    const a = s1.fromRows(rowsA)
    const b = s2.fromRows(rowsB)
    expect(LazyFrame.concat([a, b], 'vertical', { runtime: s2.runtime }).getRuntime()).toBe(s2.runtime)
    expect(a.lazy().join(b, { on: 'g', runtime: s1.runtime }).getRuntime()).toBe(s1.runtime)
    expect(a.lazy().joinAsof(b, { leftOn: 't', runtime: s2.runtime }).getRuntime()).toBe(s2.runtime)
  })

  it('unbound inputs only: the result stays on the process default (and on the receiver’s engine fork)', () => {
    const a = DataFrame.fromRows(rowsA)
    const b = DataFrame.fromRows(rowsB)
    expect(LazyFrame.concat([a, b]).getRuntime()).toBe(getDefaultRuntime())
    const forked = a.lazy().engine('cpu', { strict: true })
    const joined = forked.join(b, { on: 'g' })
    expect(joined.getRuntime()).toBe(forked.getRuntime())
  })

  it('engine() forks of one session share its lineage: combining them is not a mismatch; the receiver’s fork wins', () => {
    const s = createSession()
    const a = s.fromRows(rowsA).lazy().engine('cpu', { strict: true })
    const b = s.fromRows(rowsB).lazy().engine('cpu')
    const joined = a.join(b, { on: 'g' })
    expect(joined.getRuntime()).toBe(a.getRuntime())
    expect(joined.getRuntime().root).toBe(s.runtime)
    expect(LazyFrame.concat([a, b]).getRuntime().root).toBe(s.runtime)
  })

  it('the result persists into the session cache, not the process cache', async () => {
    const s = createSession()
    const plan = LazyFrame.concat([s.fromRows(rowsA), s.fromRows(rowsB)]).join(DataFrame.fromRows(dims), { on: 'g' })
    await plan.persist().collect()
    expect(s.persistCache.stats().entries).toBe(1)
    expect(getDefaultRuntime().persist.stats().entries).toBe(0)
  })

  it('Series and sync helpers derived from a session frame stay on the session', () => {
    const s = createSession()
    const df = s.fromRows(rowsA)
    expect(df.getColumn('g').valueCounts().getRuntime()).toBe(s.runtime)
    expect(df.nunique().getRuntime()).toBe(s.runtime)
    expect(df.head(1).getRuntime()).toBe(s.runtime)
  })
})
