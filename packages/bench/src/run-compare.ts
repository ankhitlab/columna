/**
 * Orchestrate pandas/polars (Python) + columna (Node) comparative bench.
 * Writes merged JSON to packages/bench/results/compare-latest.json
 *
 * Defaults target large-data compares (1M + 100M). Override with BENCH_SIZES.
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const resultsDir = join(root, 'results')
mkdirSync(resultsDir, { recursive: true })

const sizes = process.env.BENCH_SIZES ?? '1000000'
const runs = process.env.BENCH_RUNS ?? '3'
const ops = process.env.BENCH_OPS ?? ''
const pyOut = join(resultsDir, 'python.json')
const jsOut = join(resultsDir, 'columna.json')
const mergedOut = join(resultsDir, 'compare-latest.json')

// 100M categorical fact ≈ several GB; give Node room.
const nodeOptions = [process.env.NODE_OPTIONS, '--max-old-space-size=32768'].filter(Boolean).join(' ')

function run(cmd: string, args: string[], env: Record<string, string> = {}) {
  const quoted = args.map((a) => (/\s/.test(a) ? `"${a}"` : a))
  console.log(`\n> ${cmd} ${quoted.join(' ')}`)
  const res = spawnSync(cmd, quoted, {
    cwd: join(root, '../..'),
    env: { ...process.env, ...env },
    stdio: 'inherit',
    shell: true,
  })
  if (res.status !== 0) {
    throw new Error(`Command failed (${res.status}): ${cmd} ${quoted.join(' ')}`)
  }
}

console.log(`Comparative bench sizes=${sizes} runs=${runs}${ops ? ` ops=${ops}` : ' (all ops)'}`)

// Python: pandas + polars
const pyArgs = ['-3', 'packages/bench/python/compare.py', '--sizes', sizes, '--runs', runs, '--out', pyOut]
if (ops) pyArgs.push('--ops', ops)
run('py', pyArgs)

// Node: columna
run('pnpm', ['--filter', '@columna/bench', 'exec', 'tsx', 'src/compare-columna.ts'], {
  BENCH_SIZES: sizes,
  BENCH_RUNS: runs,
  BENCH_OUT: jsOut,
  BENCH_ENGINES: process.env.BENCH_ENGINES ?? 'cpu,wasm',
  BENCH_OPS: ops,
  NODE_OPTIONS: nodeOptions,
})

const py = JSON.parse(readFileSync(pyOut, 'utf8')) as {
  pandas: string
  polars: string
  runs: number
  results: Array<{ library: string; n: number; op: string; ms: number; extra?: unknown }>
}
const js = JSON.parse(readFileSync(jsOut, 'utf8')) as {
  runs: number
  results: Array<{ library: string; n: number; op: string; ms: number; extra?: unknown }>
}

const results = [...py.results, ...js.results]
const byKey = new Map<string, Record<string, number>>()
for (const r of results) {
  if (r.op === 'build' || r.op === 'unavailable') continue
  const key = `${r.n}|${r.op}`
  const row = byKey.get(key) ?? { n: r.n }
  row[r.library] = r.ms
  byKey.set(key, row)
}

const table = [...byKey.entries()]
  .map(([key, row]) => {
    const [, op] = key.split('|')
    return { op, ...row }
  })
  .sort((a, b) => Number(a.n) - Number(b.n) || String(a.op).localeCompare(String(b.op)))

const merged = {
  generatedAt: new Date().toISOString(),
  sizes: sizes.split(',').map(Number),
  runs: Number(runs),
  versions: {
    pandas: py.pandas,
    polars: py.polars,
    columna: '0.1.0',
  },
  methodology:
    'Median of N timed runs after 1 warmup. Schema: id,user_id,city,category,age,salary,ts,tags,payload. Classic + parity ops (str/dt/when/is_in/groupby_stats/sample/tail/explode/joins/timeseries/write/profile). Heavy ops skipped above size thresholds (explode/unnest/cross/to_csv/map_elements >500k; describe/melt/interpolate/profile >2M; rolling >10M).',
  ops: ops ? ops.split(',').map((s) => s.trim()).filter(Boolean) : undefined,
  results,
  table,
}

writeFileSync(mergedOut, JSON.stringify(merged, null, 2), 'utf8')
console.log(`\nWrote ${mergedOut}`)
console.log('\nSummary (ms, median):')
console.table(table)
