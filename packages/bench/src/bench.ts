import { DataFrame, col, init } from 'columna'

function now() {
  return performance.now()
}

async function run() {
  await init()
  const n = Number(process.env.BENCH_N ?? 100_000)
  const rows = Array.from({ length: n }, (_, i) => ({
    city: i % 2 === 0 ? 'Berlin' : 'Paris',
    age: 18 + (i % 50),
    salary: 40_000 + (i % 1000) * 10,
  }))

  const t0 = now()
  const df = DataFrame.fromRows(rows)
  const t1 = now()

  const outCpu = await df.engine('cpu').filter(col('age').gt(30)).groupBy('city').agg({ salary: 'mean' }).collect()
  const t2 = now()

  const outWasm = await df.engine('wasm').filter(col('age').gt(30)).sort(col('salary').desc()).head(100).collect()
  const t3 = now()

  console.log(
    JSON.stringify(
      {
        rows: n,
        buildMs: +(t1 - t0).toFixed(2),
        cpuFilterGroupByMs: +(t2 - t1).toFixed(2),
        wasmFilterSortMs: +(t3 - t2).toFixed(2),
        cpuResult: outCpu.toArray(),
        wasmHeadRows: outWasm.shape[0],
      },
      null,
      2,
    ),
  )
}

run().catch((err) => {
  console.error(err)
  process.exit(1)
})
