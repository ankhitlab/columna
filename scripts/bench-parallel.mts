/**
 * Quick check: worker pool dual-gt vs sync (SAB-backed columns = zero-copy workers).
 * Run: pnpm exec tsx scripts/bench-parallel.mts
 */
import { dualGtIndices } from '../packages/runtime/src/fast.ts'
import { parallelDualGtIndices } from '../packages/runtime/src/parallel.ts'

function median(xs: number[]): number {
  const a = [...xs].sort((x, y) => x - y)
  return a[Math.floor(a.length / 2)]!
}

async function time(fn: () => Promise<unknown> | unknown, runs = 3): Promise<number> {
  await fn()
  const samples: number[] = []
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now()
    await fn()
    samples.push(performance.now() - t0)
  }
  return median(samples)
}

function makeShared(n: number): { a: Int32Array; b: Float64Array } {
  const a = new Int32Array(new SharedArrayBuffer(n * 4))
  const b = new Float64Array(new SharedArrayBuffer(n * 8))
  for (let i = 0; i < n; i++) {
    a[i] = 18 + (i % 50)
    b[i] = 40_000 + (i % 1000) * 10
  }
  return { a, b }
}

async function main() {
  for (const n of [1_000_000, 5_000_000, 20_000_000]) {
    const { a, b } = makeShared(n)
    const la = 30
    const lb = 45_000

    const syncMs = await time(() => dualGtIndices(a, b, la, lb))
    // Warm pool once
    await parallelDualGtIndices(a, b, la, lb, { minRows: 1 })
    const poolMs = await time(() => parallelDualGtIndices(a, b, la, lb, { minRows: 1 }))
    const autoMs = await time(() => parallelDualGtIndices(a, b, la, lb))

    const s = dualGtIndices(a, b, la, lb)
    const p = await parallelDualGtIndices(a, b, la, lb, { minRows: 1 })
    const same = s.length === p.length && s[0] === p[0] && s[s.length - 1] === p[p.length - 1]

    console.log(
      JSON.stringify({
        n,
        syncMs: +syncMs.toFixed(2),
        poolMs: +poolMs.toFixed(2),
        autoMs: +autoMs.toFixed(2),
        speedup: +(syncMs / poolMs).toFixed(2),
        matches: same,
        hits: s.length,
      }),
    )
  }
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
