/**
 * Bench + memory profile for the statistics wave: rank (ties), broadcast aggregates, Expr math,
 * corr / cov. Also writes a 100k-row CSV sample so python/stats_compare.py can cross-check the
 * numbers against pandas.
 *
 *   node --expose-gc --import tsx packages/bench/src/stats-bench.ts
 *   BENCH_SIZES=100000,1000000 node --expose-gc --import tsx packages/bench/src/stats-bench.ts
 */
import { mkdirSync, writeFileSync } from 'node:fs'
import { performance } from 'node:perf_hooks'
import { DataFrame, col, type LazyFrame } from 'columna'

const SIZES = (process.env.BENCH_SIZES ?? '100000,1000000,10000000').split(',').map(Number)
const REPEAT = Number(process.env.BENCH_REPEAT ?? 3)
const OUT_DIR = new URL('../results/', import.meta.url)

// Deterministic data so pandas sees the same sample (mulberry32)
function rng(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

type Cols = { x: Float64Array; y: Array<number | null>; z: Float64Array; w: Int32Array; v: Float64Array }

/** x ~ approx normal, y = 0.8x + noise with 1% nulls, z uniform, w int 0..100 (heavy ties), v = x² + noise */
function makeData(n: number, seed = 42): Cols {
  const r = rng(seed)
  const x = new Float64Array(n)
  const y: Array<number | null> = new Array(n)
  const z = new Float64Array(n)
  const w = new Int32Array(n)
  const v = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const g = (r() + r() + r() + r() - 2) * 1.7320508 // ~N(0,1)
    x[i] = 50 + 10 * g
    y[i] = i % 100 === 7 ? null : 0.8 * x[i]! + 5 * (r() - 0.5)
    z[i] = r() * 100
    w[i] = Math.floor(r() * 101)
    v[i] = (x[i]! - 50) ** 2 + 20 * (r() - 0.5)
  }
  return { x, y, z, w, v }
}

function writeCsv(c: Cols, n: number, path: URL): void {
  const lines = ['x,y,z,w,v']
  for (let i = 0; i < n; i++) {
    lines.push(`${c.x[i]},${c.y[i] ?? ''},${c.z[i]},${c.w[i]},${c.v[i]}`)
  }
  writeFileSync(path, lines.join('\n') + '\n')
}

type Mem = { heapUsed: number; external: number; arrayBuffers: number; rss: number }
const mem = (): Mem => {
  const m = process.memoryUsage()
  return { heapUsed: m.heapUsed, external: m.external, arrayBuffers: m.arrayBuffers, rss: m.rss }
}
const mb = (b: number) => (b / 1048576).toFixed(1).padStart(8) + ' MB'
const gc = () => {
  if (typeof globalThis.gc === 'function') globalThis.gc()
}

type Row = {
  n: number
  op: string
  msMedian: number
  msMin: number
  rowsPerSec: number
  retainedMB: number
  peakRssDeltaMB: number
  check?: string
}
const results: Row[] = []

async function bench(n: number, op: string, run: () => Promise<DataFrame> | DataFrame, check?: (d: DataFrame) => string) {
  await run() // warm-up (JIT)
  const times: number[] = []
  let retained = 0
  let peakRss = 0
  let last: DataFrame | null = null
  for (let i = 0; i < REPEAT; i++) {
    gc()
    const before = mem()
    let rssMax = before.rss
    const sampler = setInterval(() => (rssMax = Math.max(rssMax, process.memoryUsage().rss)), 5)
    const t0 = performance.now()
    last = await run()
    const dt = performance.now() - t0
    clearInterval(sampler)
    rssMax = Math.max(rssMax, process.memoryUsage().rss)
    times.push(dt)
    gc() // retained = what the result (still referenced) actually keeps alive
    const after = mem()
    retained = after.heapUsed + after.arrayBuffers - before.heapUsed - before.arrayBuffers
    peakRss = Math.max(peakRss, rssMax - before.rss)
  }
  times.sort((a, b) => a - b)
  const msMedian = times[Math.floor(times.length / 2)]!
  const row: Row = {
    n,
    op,
    msMedian,
    msMin: times[0]!,
    rowsPerSec: n / (msMedian / 1000),
    retainedMB: retained / 1048576,
    peakRssDeltaMB: peakRss / 1048576,
    check: check && last ? check(last) : undefined,
  }
  results.push(row)
  console.log(
    `  ${op.padEnd(34)} ${msMedian.toFixed(1).padStart(9)} ms  ${(row.rowsPerSec / 1e6).toFixed(1).padStart(6)} M rows/s` +
      `  retained ${mb(retained)}  peakΔrss ${mb(peakRss)}` +
      (row.check ? `  ${row.check}` : ''),
  )
}

// Plain-JS references to quantify library overhead on the same data.
function naiveZscore(x: Float64Array): Float64Array {
  let s = 0
  for (let i = 0; i < x.length; i++) s += x[i]!
  const m = s / x.length
  let m2 = 0
  for (let i = 0; i < x.length; i++) m2 += (x[i]! - m) ** 2
  const sd = Math.sqrt(m2 / (x.length - 1))
  const out = new Float64Array(x.length)
  for (let i = 0; i < x.length; i++) out[i] = (x[i]! - m) / sd
  return out
}
function naivePearson(a: Float64Array, b: Float64Array): number {
  let n = 0, mx = 0, my = 0, m2x = 0, m2y = 0, cxy = 0
  for (let i = 0; i < a.length; i++) {
    n++
    const dx = a[i]! - mx
    mx += dx / n
    const dy = b[i]! - my
    my += dy / n
    cxy += dx * (b[i]! - my)
    m2x += dx * (a[i]! - mx)
    m2y += dy * (b[i]! - my)
  }
  return cxy / Math.sqrt(m2x * m2y)
}

const cell = (d: DataFrame, row: string, c: string) => Number(d.toArray().find((r) => r.column === row)![c])
const f = (v: number, d = 6) => v.toFixed(d)

async function main() {
  mkdirSync(OUT_DIR, { recursive: true })
  console.log(`node ${process.version} · gc ${typeof globalThis.gc === 'function' ? 'exposed' : 'NOT exposed (run with --expose-gc)'}`)

  for (const n of SIZES) {
    console.log(`\n=== n = ${n.toLocaleString()} ===`)
    gc()
    const m0 = mem()
    const t0 = performance.now()
    const data = makeData(n)
    if (n === 100_000) writeCsv(data, n, new URL('stats-sample.csv', OUT_DIR))
    const df = DataFrame.fromColumns({ x: data.x, y: data.y, z: data.z, w: data.w, v: data.v })
    gc()
    const m1 = mem()
    console.log(
      `  build DataFrame (5 cols)              ${(performance.now() - t0).toFixed(0).padStart(9)} ms` +
        `  frame ≈ ${mb(m1.heapUsed + m1.arrayBuffers - m0.heapUsed - m0.arrayBuffers)}  dtypes ${JSON.stringify(df.dtypes)}`,
    )
    const collect = (lf: LazyFrame) => lf.collect()

    // --- rank -------------------------------------------------------------------------------
    await bench(n, 'rank average (orderBy w, ties)', () => collect(df.withWindow('r', 'rank', { orderBy: ['w'] })), (d) => {
      const r = d.getColumn('r').toArray() as number[]
      let s = 0
      for (const v of r) s += v
      return `Σrank=${s} expect=${(n * (n + 1)) / 2}`
    })
    await bench(n, 'rank average (expr x, no ties)', () => collect(df.withWindow('r', 'rank', { expr: col('x') })))
    await bench(n, 'rank dense (orderBy w)', () => collect(df.withWindow('r', 'rank', { orderBy: ['w'], method: 'dense' })), (d) =>
      `max=${d.getColumn('r').max()} (101 levels)`,
    )
    await bench(n, 'rowNumber (orderBy w) [baseline]', () => collect(df.withWindow('r', 'rowNumber', { orderBy: ['w'] })))

    // --- broadcast aggregates ---------------------------------------------------------------
    await bench(n, 'zscore withColumn (x−mean)/std', () => collect(df.withColumn('z1', col('x').sub(col('x').mean()).div(col('x').std()))), (d) => {
      const z = d.getColumn('z1')
      return `mean≈${f(z.mean()!, 3)}`
    })
    await bench(n, 'zscore naive JS loop [ref]', () => {
      const out = naiveZscore(data.x)
      return DataFrame.fromColumns({ z: out })
    })
    await bench(n, 'share x / sum(x)', () => collect(df.withColumn('s', col('x').div(col('x').sum()))))
    await bench(n, 'filter x > median(x)', () => collect(df.filter(col('x').gt(col('x').median()))), (d) => `rows=${d.shape[0]} (~50%)`)
    await bench(n, 'select range = max−min', () => collect(df.select(col('x').max().sub(col('x').min()).alias('range'))))
    await bench(n, 'nunique(w) broadcast', () => collect(df.select(col('w').nunique().alias('u'))), (d) => `u=${d.toArray()[0]!.u}`)

    // --- Expr math ---------------------------------------------------------------------------
    await bench(n, 'math: log, sqrt, pow2, round2 (4 cols)', () =>
      collect(df.withColumns(col('x').log().alias('l'), col('x').sqrt().alias('s'), col('x').pow(2).alias('p'), col('x').round(2).alias('r'))),
    )
    await bench(n, 'math: exp(mean(log x)) geo-mean', () => collect(df.select(col('x').log().mean().exp().alias('gm'))), (d) => `gm=${f(Number(d.toArray()[0]!.gm), 4)}`)
    await bench(n, 'box-cox λ=0.5 (pow, sub, div)', () => collect(df.withColumn('bc', col('x').pow(0.5).sub(1).div(0.5))))
    await bench(n, 'mapElements(Math.log) [old way]', () => collect(df.withColumn('l', col('x').mapElements((v) => Math.log(v as number)))))

    // --- corr / cov ----------------------------------------------------------------------------
    await bench(n, 'corr pearson 5×5 (1% nulls in y)', () => collect(df.corr()), (d) => `xy=${f(cell(d, 'x', 'y'))} xv=${f(cell(d, 'x', 'v'))}`)
    await bench(n, 'corr pearson naive 1 pair [ref]', () => {
      const c = naivePearson(data.x, data.v)
      return DataFrame.fromRows([{ c }])
    }, (d) => `xv=${f(Number(d.toArray()[0]!.c))}`)
    await bench(n, 'corr pearson 2 cols no nulls', () => collect(df.corr({ columns: ['x', 'v'] })))
    await bench(n, 'cov 5×5', () => collect(df.cov()), (d) => `var x=${f(cell(d, 'x', 'x'), 3)}`)
    if (n <= 1_000_000 || process.env.BENCH_SPEARMAN_10M) {
      await bench(n, 'corr spearman 5×5', () => collect(df.corr({ method: 'spearman' })), (d) => `xy=${f(cell(d, 'x', 'y'))} xv=${f(cell(d, 'x', 'v'))}`)
    }
    await bench(n, 'describe() [baseline]', () => collect(df.describe()))
  }

  writeFileSync(new URL('stats-bench-latest.json', OUT_DIR), JSON.stringify({ node: process.version, date: new Date().toISOString(), results }, null, 2))
  console.log(`\nresults → packages/bench/results/stats-bench-latest.json`)
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
