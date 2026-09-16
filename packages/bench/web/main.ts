/**
 * Browser WebGPU micro-bench. Open via `pnpm --filter @columna/bench webgpu`.
 * Measures filter / map / reduce with columna:webgpu vs columna:cpu.
 */
import { DataFrame, col, init, isWebGpuAvailable } from 'columna'

const el = (id: string) => document.getElementById(id)!

function makeFact(n: number) {
  const age = new Int32Array(n)
  const salary = new Float32Array(n)
  const city: string[] = new Array(n)
  for (let i = 0; i < n; i++) {
    age[i] = 18 + (i % 50)
    salary[i] = 40_000 + (i % 1000) * 10
    city[i] = `c${i % 10}`
  }
  return DataFrame.fromColumns({ age, salary, city })
}

async function timed(fn: () => Promise<unknown>, runs: number): Promise<number> {
  await fn()
  const samples: number[] = []
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now()
    await fn()
    samples.push(performance.now() - t0)
  }
  samples.sort((a, b) => a - b)
  return samples[Math.floor(samples.length / 2)]!
}

async function run() {
  const log = el('log')
  const append = (line: string) => {
    log.textContent += line + '\n'
  }
  log.textContent = ''
  append('init…')
  await init({ rust: false })
  const gpuOk = await isWebGpuAvailable()
  append(`WebGPU available: ${gpuOk}`)
  if (!gpuOk) {
    append('No GPU device — open in Chrome/Edge with WebGPU enabled.')
    return
  }

  const sizes = [1_000_000, 10_000_000]
  const runs = 3
  const rows: Array<Record<string, unknown>> = []

  for (const n of sizes) {
    append(`\n=== n=${n.toLocaleString()} ===`)
    const fact = makeFact(n)
    for (const engine of ['cpu', 'webgpu'] as const) {
      const base = () => fact.engine(engine)
      const filterMs = await timed(
        () => base().filter(col('age').gt(30).and(col('salary').gt(45_000))).collect(),
        runs,
      )
      const mapMs = await timed(
        () => base().withColumn('bonus', col('salary').mul(1.1)).collect(),
        runs,
      )
      const pipeMs = await timed(
        () =>
          base()
            .filter(col('age').gt(25).and(col('salary').gt(42_000)))
            .groupBy('city')
            .agg({ salary: 'mean' })
            .collect(),
        runs,
      )
      append(`  columna:${engine}  filter ${filterMs.toFixed(2)} ms  map ${mapMs.toFixed(2)} ms  pipeline ${pipeMs.toFixed(2)} ms`)
      rows.push({ library: `columna:${engine}`, n, op: 'filter', ms: filterMs })
      rows.push({ library: `columna:${engine}`, n, op: 'map', ms: mapMs })
      rows.push({ library: `columna:${engine}`, n, op: 'pipeline', ms: pipeMs })
    }
  }

  el('json').textContent = JSON.stringify({ generatedAt: new Date().toISOString(), results: rows }, null, 2)
  append('\nDone.')
}

el('run').addEventListener('click', () => {
  run().catch((err) => {
    el('log').textContent += String(err) + '\n'
  })
})
