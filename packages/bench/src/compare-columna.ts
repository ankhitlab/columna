/**
 * Columna side of pandas/polars-style comparative bench.
 *
 * Includes classic ops + API-parity wave (str/dt/when/isIn, sample/tail/explode,
 * join variants, timeseries, write/profile).
 *
 * Optional: BENCH_OPS=filter,str_contains,groupby_stats
 */
import { writeFileSync } from 'node:fs'
import { performance } from 'node:perf_hooks'
import { DataFrame, col, when, init, isWebGpuAvailable, isRustKernelsLoaded, isNativeKernelsLoaded } from 'columna'

const CITIES = [
  'Berlin',
  'Paris',
  'London',
  'Madrid',
  'Rome',
  'Vienna',
  'Warsaw',
  'Prague',
  'Lisbon',
  'Dublin',
]

const CATEGORIES = Array.from({ length: 50 }, (_, i) => `c${i}`)
const TAG_VARIANTS = ['["a"]', '["a","b"]', '["x","y","z"]', '[]']
const PAYLOAD_VARIANTS = [
  '{"user":{"id":1},"n":10}',
  '{"user":{"id":2},"n":20}',
  '{"user":{"id":3},"n":30}',
]

export const ALL_OPS = [
  // classic
  'filter',
  'groupby_agg',
  'groupby_multi',
  'sort',
  'join',
  'pipeline',
  'select',
  'with_column',
  'unique',
  'value_counts',
  'rolling',
  'describe',
  'melt',
  // API parity / new
  'with_columns',
  'str_contains',
  'str_lower',
  'dt_parts',
  'when_then',
  'is_in',
  'is_between',
  'groupby_stats',
  'tail',
  'sample',
  'explode',
  'semi_join',
  'anti_join',
  'cross_join',
  'join_asof',
  'shift_diff',
  'pct_change',
  'expanding',
  'interpolate',
  'unnest',
  'transpose',
  'map_elements',
  'to_csv',
  'profile',
] as const

type OpName = (typeof ALL_OPS)[number]

interface Timing {
  library: string
  n: number
  op: string
  ms: number
  extra?: Record<string, unknown>
}

function makeFact(n: number, seed = 42) {
  const t0 = performance.now()
  const userMod = Math.max(Math.floor(n / 10), 1)
  const id = new Int32Array(n)
  const user_id = new Int32Array(n)
  const cityCodes = new Uint32Array(n)
  const categoryCodes = new Uint32Array(n)
  const age = new Int32Array(n)
  const salary = new Float64Array(n)
  const ts = new Float64Array(n)
  const tags: string[] = new Array(n)
  const payload: string[] = new Array(n)
  const baseTs = Date.UTC(2020, 0, 1)
  const nCities = CITIES.length
  for (let i = 0; i < n; i++) {
    id[i] = i
    user_id[i] = (i * 7 + seed) % userMod
    cityCodes[i] = i % nCities
    categoryCodes[i] = i % 50
    age[i] = 18 + (i % 50)
    salary[i] = 40_000 + (i % 1000) * 10 + (i % 17)
    ts[i] = baseTs + i * 60_000
    tags[i] = TAG_VARIANTS[i % TAG_VARIANTS.length]!
    payload[i] = PAYLOAD_VARIANTS[i % PAYLOAD_VARIANTS.length]!
  }
  const df = DataFrame.fromColumns({
    id,
    user_id,
    city: { codes: cityCodes, dictionary: CITIES },
    category: { codes: categoryCodes, dictionary: CATEGORIES },
    age,
    salary,
    ts,
    tags,
    payload,
  })
  console.log(`    build fact n=${n.toLocaleString()} in ${(performance.now() - t0).toFixed(0)} ms`)
  return df
}

function makeDim(nFact: number) {
  const n = Math.max(Math.floor(nFact / 10), 1)
  const user_id = new Int32Array(n)
  const segmentCodes = new Uint32Array(n)
  const score = new Int32Array(n)
  const segments = Array.from({ length: 20 }, (_, i) => `s${i}`)
  for (let i = 0; i < n; i++) {
    user_id[i] = i
    segmentCodes[i] = i % 20
    score[i] = (i * 13) % 100
  }
  return DataFrame.fromColumns({
    user_id,
    segment: { codes: segmentCodes, dictionary: segments },
    score,
  })
}

function makeAsofDim(nFact: number) {
  const n = Math.max(Math.floor(nFact / 20), 1)
  const baseTs = Date.UTC(2020, 0, 1)
  const ts = new Float64Array(n)
  const rate = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    ts[i] = baseTs + i * 120_000
    rate[i] = 1 + (i % 17) * 0.01
  }
  return DataFrame.fromColumns({ ts, rate })
}

function makeCrossDim() {
  const k = 5
  const kid = new Int32Array(k)
  const label: string[] = []
  for (let i = 0; i < k; i++) {
    kid[i] = i
    label.push(`L${i}`)
  }
  return DataFrame.fromColumns({ kid, label })
}

async function timed<T>(fn: () => Promise<T> | T, runs: number): Promise<{ ms: number; result: T }> {
  let result = await fn()
  const samples: number[] = []
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now()
    result = await fn()
    samples.push(performance.now() - t0)
  }
  samples.sort((a, b) => a - b)
  const mid = samples[Math.floor(samples.length / 2)]!
  return { ms: mid, result }
}

function runsFor(n: number, base: number): number {
  if (n >= 50_000_000) return Math.min(base, 2)
  if (n >= 10_000_000) return Math.min(base, 3)
  return base
}

function parseOps(raw: string | undefined): Set<OpName> {
  if (!raw || !raw.trim()) return new Set(ALL_OPS)
  const chosen = raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean) as OpName[]
  const allowed = new Set<string>(ALL_OPS)
  for (const op of chosen) {
    if (!allowed.has(op)) throw new Error(`Unknown BENCH_OPS entry "${op}". Allowed: ${ALL_OPS.join(',')}`)
  }
  return new Set(chosen)
}

function want(ops: Set<OpName>, name: OpName, n: number): boolean {
  if (!ops.has(name)) return false
  if ((name === 'describe' || name === 'melt') && n > 2_000_000) return false
  if (name === 'rolling' && n > 10_000_000) return false
  // Heavy reshape / materialize
  if (
    (name === 'explode' ||
      name === 'unnest' ||
      name === 'transpose' ||
      name === 'to_csv' ||
      name === 'cross_join' ||
      name === 'map_elements') &&
    n > 500_000
  ) {
    return false
  }
  if ((name === 'interpolate' || name === 'profile') && n > 2_000_000) return false
  return true
}

type Engine = 'cpu' | 'wasm' | 'auto' | 'webgpu'

async function benchColumna(n: number, runs: number, engine: Engine, ops: Set<OpName>): Promise<Timing[]> {
  const out: Timing[] = []
  const effectiveRuns = runsFor(n, runs)

  if (engine === 'webgpu') {
    const ok = await isWebGpuAvailable()
    if (!ok) {
      out.push({
        library: 'columna:webgpu',
        n,
        op: 'unavailable',
        ms: 0,
        extra: { reason: 'WebGPU device not available in this runtime (use browser bench)' },
      })
      return out
    }
  }

  const build = await timed(() => makeFact(n), 1)
  const fact = build.result
  const dim = makeDim(n)
  const asofDim = makeAsofDim(n)
  const crossDim = makeCrossDim()
  out.push({ library: `columna:${engine}`, n, op: 'build', ms: build.ms, extra: { runs: effectiveRuns } })

  const base = () => (engine === 'auto' ? fact : fact.engine(engine))

  if (want(ops, 'filter', n)) {
    const { ms } = await timed(
      () => base().filter(col('age').gt(30).and(col('salary').gt(45_000))).collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'filter', ms })
  }

  if (want(ops, 'groupby_agg', n)) {
    const { ms } = await timed(
      () => base().groupBy('city').agg({ salary: 'mean', age: 'count' }).collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'groupby_agg', ms })
  }

  if (want(ops, 'groupby_multi', n)) {
    const { ms } = await timed(
      () => base().groupBy('city', 'category').agg({ salary: 'mean', id: 'count' }).collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'groupby_multi', ms })
  }

  if (want(ops, 'sort', n)) {
    const { ms } = await timed(() => base().sort(col('salary').desc()).head(1000).collect(), effectiveRuns)
    out.push({ library: `columna:${engine}`, n, op: 'sort', ms })
  }

  if (want(ops, 'join', n)) {
    const { ms, result } = await timed(() => base().innerJoin(dim, 'user_id').collect(), effectiveRuns)
    out.push({ library: `columna:${engine}`, n, op: 'join', ms, extra: { rows: result.shape[0] } })
  }

  if (want(ops, 'pipeline', n)) {
    const { ms } = await timed(
      () =>
        base()
          .filter(col('age').gt(25).and(col('salary').gt(42_000)))
          .groupBy('city')
          .agg({ salary: 'mean', id: 'count' })
          .sort(col('salary').desc())
          .collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'pipeline', ms })
  }

  if (want(ops, 'select', n)) {
    const { ms } = await timed(() => base().select('id', 'city', 'salary').collect(), effectiveRuns)
    out.push({ library: `columna:${engine}`, n, op: 'select', ms })
  }

  if (want(ops, 'with_column', n)) {
    const { ms } = await timed(
      () => base().withColumn('bonus', col('salary').mul(1.1)).collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'with_column', ms })
  }

  if (want(ops, 'unique', n)) {
    const { ms, result } = await timed(() => base().unique(['user_id']).collect(), effectiveRuns)
    out.push({ library: `columna:${engine}`, n, op: 'unique', ms, extra: { rows: result.shape[0] } })
  }

  if (want(ops, 'value_counts', n)) {
    const { ms } = await timed(() => base().valueCounts('city').collect(), effectiveRuns)
    out.push({ library: `columna:${engine}`, n, op: 'value_counts', ms })
  }

  if (want(ops, 'rolling', n)) {
    const { ms } = await timed(() => base().rolling('m', 'salary', 32, 'mean').collect(), effectiveRuns)
    out.push({ library: `columna:${engine}`, n, op: 'rolling', ms })
  }

  if (want(ops, 'describe', n)) {
    const { ms } = await timed(() => base().describe().collect(), effectiveRuns)
    out.push({ library: `columna:${engine}`, n, op: 'describe', ms })
  }

  if (want(ops, 'melt', n)) {
    const { ms, result } = await timed(
      () =>
        base()
          .melt({ idVars: ['id', 'city'], valueVars: ['age', 'salary'] })
          .collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'melt', ms, extra: { rows: result.shape[0] } })
  }

  // —— new / parity ——
  if (want(ops, 'with_columns', n)) {
    const { ms } = await timed(
      () =>
        base()
          .withColumns(
            col('salary').mul(1.1).alias('bonus'),
            col('age').add(1).alias('age1'),
          )
          .collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'with_columns', ms })
  }

  if (want(ops, 'str_contains', n)) {
    const { ms } = await timed(
      () => base().filter(col('city').str.contains('a')).collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'str_contains', ms })
  }

  if (want(ops, 'str_lower', n)) {
    const { ms } = await timed(
      () => base().withColumn('city_lo', col('city').str.toLowerCase()).collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'str_lower', ms })
  }

  if (want(ops, 'dt_parts', n)) {
    const { ms } = await timed(
      () =>
        base()
          .withColumns(col('ts').dt.year().alias('y'), col('ts').dt.month().alias('m'))
          .collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'dt_parts', ms })
  }

  if (want(ops, 'when_then', n)) {
    const { ms } = await timed(
      () =>
        base()
          .withColumn(
            'band',
            when(col('age').lt(30))
              .then('young')
              .when(col('age').lt(50))
              .then('mid')
              .otherwise('senior'),
          )
          .collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'when_then', ms })
  }

  if (want(ops, 'is_in', n)) {
    const { ms } = await timed(
      () => base().filter(col('age').isIn([25, 30, 35, 40, 45])).collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'is_in', ms })
  }

  if (want(ops, 'is_between', n)) {
    const { ms } = await timed(
      () => base().filter(col('salary').isBetween(42_000, 48_000)).collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'is_between', ms })
  }

  if (want(ops, 'groupby_stats', n)) {
    const { ms } = await timed(
      () =>
        base()
          .groupBy('city')
          .agg({
            std: col('salary').std(),
            med: col('salary').median(),
            q75: col('salary').quantile(0.75),
          })
          .collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'groupby_stats', ms })
  }

  if (want(ops, 'tail', n)) {
    const { ms } = await timed(() => base().tail(1000).collect(), effectiveRuns)
    out.push({ library: `columna:${engine}`, n, op: 'tail', ms })
  }

  if (want(ops, 'sample', n)) {
    const { ms } = await timed(() => base().sample({ n: Math.min(10_000, n), seed: 7 }).collect(), effectiveRuns)
    out.push({ library: `columna:${engine}`, n, op: 'sample', ms })
  }

  if (want(ops, 'explode', n)) {
    const { ms, result } = await timed(() => base().explode('tags').collect(), effectiveRuns)
    out.push({ library: `columna:${engine}`, n, op: 'explode', ms, extra: { rows: result.shape[0] } })
  }

  if (want(ops, 'semi_join', n)) {
    const { ms, result } = await timed(() => base().semiJoin(dim, 'user_id').collect(), effectiveRuns)
    out.push({ library: `columna:${engine}`, n, op: 'semi_join', ms, extra: { rows: result.shape[0] } })
  }

  if (want(ops, 'anti_join', n)) {
    const { ms, result } = await timed(() => base().antiJoin(dim, 'user_id').collect(), effectiveRuns)
    out.push({ library: `columna:${engine}`, n, op: 'anti_join', ms, extra: { rows: result.shape[0] } })
  }

  if (want(ops, 'cross_join', n)) {
    const { ms, result } = await timed(() => base().crossJoin(crossDim).collect(), effectiveRuns)
    out.push({ library: `columna:${engine}`, n, op: 'cross_join', ms, extra: { rows: result.shape[0] } })
  }

  if (want(ops, 'join_asof', n)) {
    const { ms, result } = await timed(
      () => base().joinAsof(asofDim, { leftOn: 'ts', strategy: 'backward' }).collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'join_asof', ms, extra: { rows: result.shape[0] } })
  }

  if (want(ops, 'shift_diff', n)) {
    const { ms } = await timed(
      () =>
        base()
          .withColumns(col('salary').shift(1).alias('s'), col('salary').diff(1).alias('d'))
          .collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'shift_diff', ms })
  }

  if (want(ops, 'pct_change', n)) {
    const { ms } = await timed(
      () => base().withColumn('pct', col('salary').pctChange(1)).collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'pct_change', ms })
  }

  if (want(ops, 'expanding', n)) {
    const { ms } = await timed(() => base().expanding('csum', 'salary', 'sum').collect(), effectiveRuns)
    out.push({ library: `columna:${engine}`, n, op: 'expanding', ms })
  }

  if (want(ops, 'interpolate', n)) {
    const m = Math.min(n, 200_000)
    const rows = Array.from({ length: m }, (_, i) => ({ a: i % 17 === 0 ? null : i }))
    const sparse = DataFrame.fromRows(rows)
    const { ms } = await timed(
      () => sparse.engine(engine === 'auto' ? 'cpu' : engine).interpolate(['a']).collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'interpolate', ms })
  }

  if (want(ops, 'unnest', n)) {
    const { ms } = await timed(() => base().unnest('payload').collect(), effectiveRuns)
    out.push({ library: `columna:${engine}`, n, op: 'unnest', ms })
  }

  if (want(ops, 'transpose', n)) {
    const small = await base().head(20).collect()
    const { ms } = await timed(() => small.transpose('id').collect(), effectiveRuns)
    out.push({ library: `columna:${engine}`, n, op: 'transpose', ms })
  }

  if (want(ops, 'map_elements', n)) {
    const { ms } = await timed(
      () => base().withColumn('x2', col('age').mapElements((v) => Number(v) * 2)).collect(),
      effectiveRuns,
    )
    out.push({ library: `columna:${engine}`, n, op: 'map_elements', ms })
  }

  if (want(ops, 'to_csv', n)) {
    const df = await base().collect()
    const { ms, result } = await timed(() => df.toCsv(), effectiveRuns)
    out.push({ library: `columna:${engine}`, n, op: 'to_csv', ms, extra: { chars: result.length } })
  }

  if (want(ops, 'profile', n)) {
    const df = await base().collect()
    const { ms } = await timed(() => df.profile(), effectiveRuns)
    out.push({ library: `columna:${engine}`, n, op: 'profile', ms })
  }

  return out
}

async function main() {
  await init({ rust: true, native: true })
  console.log(`[columna] rust kernels: ${isRustKernelsLoaded() ? 'loaded' : 'ts-fallback'}`)
  console.log(`[columna] native rayon: ${isNativeKernelsLoaded() ? 'loaded' : 'unavailable'}`)
  const sizes = (process.env.BENCH_SIZES ?? '1000000')
    .split(',')
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isFinite(n) && n > 0)
  const runs = Number(process.env.BENCH_RUNS ?? 3)
  const engines = (process.env.BENCH_ENGINES ?? 'cpu,wasm').split(',').map((s) => s.trim()) as Engine[]
  const ops = parseOps(process.env.BENCH_OPS)
  const outPath = process.env.BENCH_OUT ?? ''

  const results: Timing[] = []
  for (const n of sizes) {
    console.log(`[columna] n=${n.toLocaleString()} (runs=${runsFor(n, runs)}) ops=${[...ops].sort().join(',')}`)
    for (const engine of engines) {
      const rows = await benchColumna(n, runs, engine, ops)
      for (const row of rows) {
        results.push(row)
        console.log(`  ${row.library.padEnd(14)} ${row.op.padEnd(14)} ${row.ms.toFixed(2).padStart(10)} ms`)
      }
    }
  }

  const payload = {
    engine: 'columna',
    runs,
    ops: [...ops].sort(),
    results,
  }
  const text = JSON.stringify(payload, null, 2)
  if (outPath) writeFileSync(outPath, text)
  else console.log(text)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
