/**
 * End-to-end benchmark: CSV file → filter + groupBy → CSV file, measured the way a deployment decision
 * needs it — cold (fresh process, module load and JIT included) and warm (same process, second run),
 * peak RSS sampled during the run, and the backend that *actually* executed each node from the execution
 * report, not the engine that was requested.
 *
 *   pnpm --filter @columna/bench run e2e                 # 2M rows × 8 columns (~150 MB CSV)
 *   E2E_ROWS=10000000 pnpm --filter @columna/bench run e2e
 *   E2E_ENGINE=wasm pnpm --filter @columna/bench run e2e  # requested engine (report shows what ran)
 *
 * The child process protocol: this file re-executes itself with E2E_CHILD=1 for the cold measurement.
 */
import { spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, statSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const ROWS = Number(process.env.E2E_ROWS ?? 2_000_000)
const ENGINE = (process.env.E2E_ENGINE ?? 'cpu') as 'cpu' | 'wasm' | 'webgpu' | 'auto'
const OUT = new URL('../results/', import.meta.url)
const INPUT = fileURLToPath(new URL(`e2e-input-${ROWS}.csv`, OUT))
const OUTPUT = fileURLToPath(new URL(`e2e-output-${ROWS}.csv`, OUT))

function ensureInput(): void {
  if (existsSync(INPUT)) return
  mkdirSync(OUT, { recursive: true })
  // 8 columns: 2 int, 4 float, 2 low-cardinality strings — the schema is part of every number below
  const parts: string[] = ['id,age,salary,x,y,z,city,segment']
  let seed = 7
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296)
  const cities = ['Berlin', 'Paris', 'Rome', 'Madrid', 'Wien']
  for (let i = 0; i < ROWS; i++) {
    parts.push(`${i},${18 + Math.floor(rnd() * 60)},${(20_000 + rnd() * 120_000).toFixed(2)},${(rnd() * 100).toFixed(4)},${(rnd() * 100).toFixed(4)},${(rnd() * 100).toFixed(4)},${cities[i % 5]},${i % 7 ? 'B2B' : 'B2C'}`)
    if (parts.length >= 100_000) {
      writeFileSync(INPUT, parts.join('\n') + '\n', { flag: 'a' })
      parts.length = 0
    }
  }
  if (parts.length) writeFileSync(INPUT, parts.join('\n') + '\n', { flag: 'a' })
}

type Phase = { name: string; ms: number }
type Run = { phases: Phase[]; totalMs: number; peakRssMb: number; backendsUsed: string[]; kernels: string[]; rowsOut: number }

async function runOnce(label: string): Promise<Run> {
  const { DataFrame, col, init } = await import('columna')
  await init()
  let peak = process.memoryUsage().rss
  const sampler = setInterval(() => (peak = Math.max(peak, process.memoryUsage().rss)), 20)
  const phases: Phase[] = []
  const t0 = performance.now()
  let t = t0
  const lap = (name: string) => {
    const now = performance.now()
    phases.push({ name, ms: now - t })
    t = now
  }
  const df = await DataFrame.readCsv({ path: INPUT })
  lap('readCsv (stream → columns)')
  const { frame, report } = await df
    .lazy()
    .filter(col('age').gt(30).and(col('salary').gt(45_000)))
    .groupBy('city')
    .agg({ n: col('id').count(), salary: col('salary').mean(), x: col('x').sum() })
    .engine(ENGINE)
    .collectWithReport()
  lap(`filter+groupBy (${report.backendsUsed.join('+') || 'cpu'})`)
  await frame.writeCsv(OUTPUT)
  lap('writeCsv (stream)')
  clearInterval(sampler)
  peak = Math.max(peak, process.memoryUsage().rss)
  const totalMs = performance.now() - t0
  void label
  return {
    phases,
    totalMs,
    peakRssMb: peak / 1e6,
    backendsUsed: report.backendsUsed,
    kernels: report.events.map((e) => `${e.node}:${e.backend}${e.kernel ? `[${e.kernel}]` : ''}${e.reason ? ` (${e.reason})` : ''}`),
    rowsOut: frame.shape[0],
  }
}

async function main(): Promise<void> {
  ensureInput()
  if (process.env.E2E_CHILD) {
    const r = await runOnce('cold')
    process.stdout.write(JSON.stringify(r))
    return
  }
  const inputMb = statSync(INPUT).size / 1e6
  const t0 = performance.now()
  const child = spawnSync(process.execPath, ['--import', 'tsx', fileURLToPath(import.meta.url)], {
    env: { ...process.env, E2E_CHILD: '1' },
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
  })
  const wallCold = performance.now() - t0
  if (child.status !== 0) throw new Error(child.stderr)
  const cold = JSON.parse(child.stdout) as Run
  const warm1 = await runOnce('warm')
  const warm2 = await runOnce('warm')
  const fmt = (r: Run) => `${r.totalMs.toFixed(0)} ms (${r.phases.map((p) => `${p.name} ${p.ms.toFixed(0)}`).join(' · ')}) · peak RSS ${r.peakRssMb.toFixed(0)} MB`
  const baseRss = process.memoryUsage().rss / 1e6
  const lines = [
    `# e2e: ${ROWS.toLocaleString()} rows × 8 columns, CSV ${inputMb.toFixed(0)} MB → filter + groupBy → CSV · engine requested: ${ENGINE} · node ${process.version}`,
    `cold (fresh process incl. startup): wall ${wallCold.toFixed(0)} ms, in-process ${fmt(cold)}`,
    `warm #1: ${fmt(warm1)}`,
    `warm #2: ${fmt(warm2)}`,
    `backends that executed nodes: ${warm2.backendsUsed.join(', ')}`,
    ...warm2.kernels.map((k) => `  ${k}`),
    `rows out: ${warm2.rowsOut}; process RSS after runs ${baseRss.toFixed(0)} MB`,
    `numeric buffers alone: ${((ROWS * (2 * 4 + 4 * 8)) / 1e6).toFixed(0)} MB (2×i32 + 4×f64) + 2 category columns ${((ROWS * 2 * 4) / 1e6).toFixed(0)} MB`,
  ]
  console.log(lines.join('\n'))
  mkdirSync(OUT, { recursive: true })
  writeFileSync(new URL(`e2e-${ROWS}-${ENGINE}.json`, OUT), JSON.stringify({ rows: ROWS, engine: ENGINE, inputMb, wallColdMs: wallCold, cold, warm1, warm2 }, null, 2))
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
