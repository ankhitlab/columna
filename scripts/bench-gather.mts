/**
 * Gather microbench: sync vs worker pool (SAB sources).
 */
import { runParallelGather } from '../packages/runtime/src/parallel-node.ts'

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

async function main() {
  for (const n of [500_000, 2_000_000, 5_000_000]) {
    const idx = new Uint32Array(new SharedArrayBuffer(n * 4))
    for (let i = 0; i < n; i++) idx[i] = (i * 3) % (n * 2 === 0 ? n : n) // messy
    // denser: take every other from 2n source
    const srcN = n * 2
    for (let i = 0; i < n; i++) idx[i] = i * 2

    const cols = Array.from({ length: 6 }, () => {
      const src = new Float64Array(new SharedArrayBuffer(srcN * 8))
      for (let i = 0; i < srcN; i++) src[i] = i
      const out = new Float64Array(new SharedArrayBuffer(n * 8))
      return { src, out }
    })

    const syncMs = await time(() => {
      for (const c of cols) {
        for (let i = 0; i < n; i++) c.out[i] = c.src[idx[i]!]!
      }
    })

    await runParallelGather(idx, cols) // warm
    const poolMs = await time(() => runParallelGather(idx, cols))

    console.log(
      JSON.stringify({
        n,
        cols: cols.length,
        syncMs: +syncMs.toFixed(2),
        poolMs: +poolMs.toFixed(2),
        speedup: +(syncMs / poolMs).toFixed(2),
      }),
    )
  }
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
