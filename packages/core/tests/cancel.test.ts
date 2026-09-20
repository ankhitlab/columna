/**
 * `collect({ signal, timeoutMs })`: cooperative cancellation and deadlines.
 *  - a pre-aborted signal rejects before any work; an abort from a timer lands between operators;
 *  - the deadline is checked without any timer having to fire;
 *  - the cooperative path yields to the event loop (other tasks run while a plan executes);
 *  - results and kernels are identical to the plain path — fused units stay fused;
 *  - an abort is never turned into an engine fallback.
 */
import { describe, expect, it } from 'vitest'
import { DataFrame, ExecutionAbortedError, LazyFrame, col } from '../src/index.js'

const N = 400_000
const base = DataFrame.fromColumns({
  id: Int32Array.from({ length: N }, (_, i) => i),
  x: Float64Array.from({ length: N }, (_, i) => ((i * 7919) % 10_007) / 7),
  g: Array.from({ length: N }, (_, i) => `g${i % 97}`),
})
const dims = DataFrame.fromRows(Array.from({ length: 97 }, (_, i) => ({ g: `g${i}`, w: i * 0.5 })))

/** ~20 operators so there are many boundaries to observe; a few hundred ms of CPU. */
const heavy = () => {
  let lf = base.lazy()
  for (let k = 0; k < 6; k++) {
    lf = lf
      .withColumn(`y${k}`, col('x').mul(k + 1).add(col('id')))
      .filter(col(`y${k}`).gt(k))
      .sort(col(`y${k}`).desc())
  }
  return lf.join(dims.lazy(), { on: 'g' }).groupBy('g').agg({ n: col('id').count(), s: col('x').sum(), w: col('w').mean() }).sort('g')
}

describe('collect({ signal })', () => {
  it('a signal aborted beforehand rejects with ExecutionAbortedError before any operator runs', async () => {
    const ac = new AbortController()
    ac.abort(new Error('user left the page'))
    const err = await heavy().collect({ signal: ac.signal }).catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ExecutionAbortedError)
    const e = err as ExecutionAbortedError
    expect(e.reason).toBe('signal')
    expect(e.node).toBe('start')
    expect(e.message).toMatch(/user left the page/)
  })

  it('an abort fired from a timer is observed between operators — the plan stops early', async () => {
    const t0 = performance.now()
    const full = await heavy().collect()
    const fullMs = performance.now() - t0
    const ac = new AbortController()
    const t1 = performance.now()
    setTimeout(() => ac.abort(), 5)
    const err = await heavy().collect({ signal: ac.signal }).catch((e: unknown) => e)
    const abortedMs = performance.now() - t1
    expect(err).toBeInstanceOf(ExecutionAbortedError)
    expect((err as ExecutionAbortedError).reason).toBe('signal')
    expect((err as ExecutionAbortedError).node).not.toBe('start')
    // stopped well before the whole plan would have finished
    expect(abortedMs).toBeLessThan(fullMs * 0.8)
    expect(full.shape[0]).toBe(97)
  })

  it('the cooperative path yields: an interval keeps ticking while the plan runs', async () => {
    let ticks = 0
    const timer = setInterval(() => ticks++, 1)
    try {
      await heavy().collect({ signal: new AbortController().signal })
    } finally {
      clearInterval(timer)
    }
    expect(ticks).toBeGreaterThan(3)
  })
})

describe('collect({ timeoutMs })', () => {
  it('a deadline is enforced without any timer firing (checked synchronously between operators)', async () => {
    const err = await heavy().collect({ timeoutMs: 1 }).catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ExecutionAbortedError)
    const e = err as ExecutionAbortedError
    expect(e.reason).toBe('timeout')
    expect(e.elapsedMs).toBeGreaterThanOrEqual(1)
    expect(e.message).toMatch(/timeoutMs/)
  })

  it('rejects a non-positive timeout up front', async () => {
    await expect(heavy().collect({ timeoutMs: 0 })).rejects.toThrow(RangeError)
  })
})

describe('cooperative execution ≡ plain execution', () => {
  const plans = () => ({
    'project → filter (fused gather)': base.lazy().filter(col('x').gt(500)).select('id', 'x'),
    'project → join (pruned)': base.lazy().join(dims.lazy(), { on: 'g' }).select('id', 'w'),
    'groupBy over sort': base.lazy().sort('x').groupBy('g').agg({ n: col('id').count(), m: col('x').max() }).sort('g'),
    'concat + unique': LazyFrame.concat([base.lazy().limit(1000), base.lazy().limit(1000)]).unique(['g']).sort('g'),
    heavy: heavy(),
  })
  for (const [name, plan] of Object.entries(plans())) {
    // Heavy plan + coverage instrumentation exceeds Vitest's 5s default on CI (node 22).
    it(
      `${name}: same rows, same kernels, and one trace event per execution unit`,
      async () => {
        const plain = await plan.collectWithReport()
        const coop = await plan.collectWithReport({ timeoutMs: 120_000 })
        expect(coop.frame.toArray()).toEqual(plain.frame.toArray())
        expect(coop.frame.dtypes).toEqual(plain.frame.dtypes)
        // the root unit ran on the same node type with the same kernel as the plain path (fusions survive the yields)
        const rootPlain = plain.report.events.at(-1)!
        const rootCoop = coop.report.events.at(-1)!
        expect({ node: rootCoop.node, kernel: rootCoop.kernel }).toEqual({
          node: rootPlain.node,
          kernel: rootPlain.kernel,
        })
        expect(coop.report.events.length).toBeGreaterThanOrEqual(plain.report.events.length)
        expect(coop.report.backendsUsed).toEqual(['cpu'])
      },
      30_000,
    )
  }
})

describe('no fallback on abort', () => {
  it('a strict engine request that is aborted rejects with ExecutionAbortedError, not EngineStrictError', async () => {
    const ac = new AbortController()
    ac.abort()
    const err = await base.lazy().filter(col('x').gt(1)).engine('cpu', { strict: true }).collect({ signal: ac.signal }).catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ExecutionAbortedError)
  })
})
