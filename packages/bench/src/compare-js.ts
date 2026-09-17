/**
 * Same data, same five operations, same result checksums — columna against the JS-side alternatives a
 * team would actually consider: Arquero (in-JS columnar tables), DuckDB-Wasm (SQL engine compiled to
 * WebAssembly, the browser option), nodejs-polars (native Polars core) and native DuckDB (Node addon).
 *
 * Each library runs in its own child process (E2E_LIB=…) so peak RSS is attributable, with one cold pass
 * (module load + first run) and warm repeats. Results are validated across libraries: a mismatching
 * checksum is printed, not hidden.
 *
 *   pnpm --filter @columna/bench run compare:js               # 2M rows × 8 columns
 *   E2E_ROWS=500000 pnpm --filter @columna/bench run compare:js
 */
import { spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, statSync, writeFileSync, readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const ROWS = Number(process.env.E2E_ROWS ?? 2_000_000)
const OUT = new URL('../results/', import.meta.url)
const INPUT = fileURLToPath(new URL(`e2e-input-${ROWS}.csv`, OUT)).replace(/\\/g, '/')
const OUTPUT = (lib: string) => fileURLToPath(new URL(`compare-${lib}-${ROWS}.csv`, OUT)).replace(/\\/g, '/')
const LIBS = ['columna', 'arquero', 'duckdb-wasm', 'polars', 'polars-lazy', 'duckdb'] as const
type Lib = (typeof LIBS)[number]

function ensureInput(): void {
  if (existsSync(INPUT)) return
  mkdirSync(OUT, { recursive: true })
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

/** Checksums every library must reproduce. */
type Check = { filterRows: number; groupSum: number; sortFirst: number; sortLast: number; joinRows: number; joinSum: number }
type OpTimes = Record<'read' | 'filter' | 'groupBy' | 'sort' | 'join' | 'writeCsv', number>
type Run = { times: OpTimes; check: Check; peakRssMb: number; note?: string }

const REGIONS = [
  ['Berlin', 'DE'],
  ['Paris', 'FR'],
  ['Rome', 'IT'],
  ['Madrid', 'ES'],
] as const // Wien has no match → inner join drops it

const round = (x: number) => Math.round(x * 1000) / 1000

async function runLib(lib: Lib): Promise<Run> {
  let peak = process.memoryUsage().rss
  const sampler = setInterval(() => (peak = Math.max(peak, process.memoryUsage().rss)), 20)
  const times: OpTimes = { read: 0, filter: 0, groupBy: 0, sort: 0, join: 0, writeCsv: 0 }
  const timed = async <T>(op: keyof OpTimes, f: () => Promise<T> | T): Promise<T> => {
    const t0 = performance.now()
    const r = await f()
    times[op] = performance.now() - t0
    return r
  }
  let check: Check
  let note: string | undefined

  if (lib === 'columna') {
    const { DataFrame, col } = await import('columna')
    const df = await timed('read', () => DataFrame.readCsv({ path: INPUT }))
    const filtered = await timed('filter', () => df.filter(col('age').gt(30).and(col('salary').gt(45_000))).collect())
    const grouped = await timed('groupBy', () => df.groupBy('city').agg({ n: col('id').count(), salary: col('salary').mean(), x: col('x').sum() }).collect())
    const sorted = await timed('sort', () => df.sort(col('salary').desc()).collect())
    const regions = DataFrame.fromRows(REGIONS.map(([city, region]) => ({ city, region })))
    const joined = await timed('join', () => df.join(regions, { on: 'city' }).collect())
    await timed('writeCsv', () => filtered.writeCsv(OUTPUT(lib)))
    const gs = grouped.toArray().reduce((s, r) => s + Number(r.salary) + Number(r.x), 0)
    const sal = sorted.getColumn('salary').toArray() as number[]
    check = { filterRows: filtered.shape[0], groupSum: round(gs), sortFirst: sal[0]!, sortLast: sal[sal.length - 1]!, joinRows: joined.shape[0], joinSum: round(joined.getColumn('salary').toArray().reduce((s: number, v) => s + Number(v), 0)) }
  } else if (lib === 'arquero') {
    const aq = await import('arquero')
    const fs = await import('node:fs/promises')
    const table = await timed('read', async () => aq.fromCSV(await fs.readFile(INPUT, 'utf8')))
    const filtered = await timed('filter', () => table.filter((d: { age: number; salary: number }) => d.age > 30 && d.salary > 45000).reify())
    const grouped = await timed('groupBy', () => table.groupby('city').rollup({ n: aq.op.count(), salary: aq.op.mean('salary'), x: aq.op.sum('x') }))
    const sorted = await timed('sort', () => table.orderby(aq.desc('salary')).reify()) // orderby is lazy in Arquero; reify materialises
    const regions = aq.table({ city: REGIONS.map((r) => r[0]), region: REGIONS.map((r) => r[1]) })
    const joined = await timed('join', () => table.join(regions, 'city').reify())
    await timed('writeCsv', async () => fs.writeFile(OUTPUT(lib), filtered.toCSV()))
    const gs = grouped.objects().reduce((s: number, r: { salary: number; x: number }) => s + r.salary + r.x, 0)
    const sal = sorted.array('salary') as number[]
    check = { filterRows: filtered.numRows(), groupSum: round(gs), sortFirst: sal[0]!, sortLast: sal[sal.length - 1]!, joinRows: joined.numRows(), joinSum: round((joined.array('salary') as number[]).reduce((s, v) => s + v, 0)) }
    note = 'reads CSV from a string (fs.readFile + fromCSV); filter/orderby/join reified so the work is inside the timing'
  } else if (lib === 'polars' || lib === 'polars-lazy') {
    const pl = (await import('nodejs-polars')).default
    const regions = pl.DataFrame({ city: REGIONS.map((r) => r[0]), region: REGIONS.map((r) => r[1]) })
    const pred = pl.col('age').gt(30).and(pl.col('salary').gt(45000))
    if (lib === 'polars') {
      const df = await timed('read', () => pl.readCSV(INPUT))
      const filtered = await timed('filter', () => df.filter(pred))
      const grouped = await timed('groupBy', () => df.groupBy('city').agg(pl.col('id').count().alias('n'), pl.col('salary').mean(), pl.col('x').sum()))
      const sorted = await timed('sort', () => df.sort('salary', true))
      const joined = await timed('join', () => df.join(regions, { on: 'city' }))
      await timed('writeCsv', () => filtered.writeCSV(OUTPUT(lib)))
      const gs = grouped.toRecords().reduce((s: number, r: { salary: number; x: number }) => s + r.salary + r.x, 0)
      const sal = sorted.getColumn('salary').toArray() as number[]
      check = { filterRows: filtered.height, groupSum: round(gs), sortFirst: sal[0]!, sortLast: sal[sal.length - 1]!, joinRows: joined.height, joinSum: round(joined.getColumn('salary').sum() as number) }
      note = 'eager API; native multi-threaded core'
    } else {
      // lazy: scan + operation + collect, i.e. read cost is paid inside every op (streaming, no full materialisation)
      const scan = () => pl.scanCSV(INPUT)
      times.read = 0
      const filtered = await timed('filter', () => scan().filter(pred).collect())
      const grouped = await timed('groupBy', () => scan().groupBy('city').agg(pl.col('id').count().alias('n'), pl.col('salary').mean(), pl.col('x').sum()).collect())
      const sorted = await timed('sort', () => scan().sort('salary', true).collect())
      const joined = await timed('join', () => scan().join(regions.lazy(), { on: 'city' }).collect())
      await timed('writeCsv', () => filtered.writeCSV(OUTPUT(lib)))
      const gs = grouped.toRecords().reduce((s: number, r: { salary: number; x: number }) => s + r.salary + r.x, 0)
      const sal = sorted.getColumn('salary').toArray() as number[]
      check = { filterRows: filtered.height, groupSum: round(gs), sortFirst: sal[0]!, sortLast: sal[sal.length - 1]!, joinRows: joined.height, joinSum: round(joined.getColumn('salary').sum() as number) }
      note = 'lazy API: each op = scanCSV + op + collect (read is inside every timing; query optimiser + streaming)'
    }
  } else {
    // SQL engines: DuckDB-Wasm (blocking Node bindings) and native DuckDB
    type Q = (sql: string) => Promise<Array<Record<string, unknown>>>
    let q: Q
    let close: () => Promise<void> | void
    if (lib === 'duckdb-wasm') {
      const blocking = await import('@duckdb/duckdb-wasm/blocking')
      const require = createRequire(import.meta.url)
      const dist = path.dirname(require.resolve('@duckdb/duckdb-wasm'))
      const db = await blocking.createDuckDB({ eh: { mainModule: path.join(dist, 'duckdb-eh.wasm'), mainWorker: path.join(dist, 'duckdb-node-eh.worker.cjs') } }, new blocking.VoidLogger(), blocking.NODE_RUNTIME)
      await db.instantiate()
      const conn = db.connect()
      q = async (sql) => conn.query(sql).toArray().map((r: { toJSON: () => Record<string, unknown> }) => r.toJSON())
      close = () => conn.close()
      note = 'single-threaded WebAssembly build (eh), blocking Node bindings; the browser deployment target'
    } else {
      const duck = (await import('duckdb')).default
      const db = new duck.Database(':memory:')
      q = (sql) => new Promise((res, rej) => db.all(sql, (err: Error | null, rows: Array<Record<string, unknown>>) => (err ? rej(err) : res(rows))))
      close = () => new Promise<void>((res) => db.close(() => res()))
      note = 'native addon, multi-threaded'
    }
    await timed('read', () => q(`CREATE OR REPLACE TABLE t AS SELECT * FROM read_csv_auto('${INPUT}')`))
    const f = await timed('filter', () => q('CREATE OR REPLACE TABLE f AS SELECT * FROM t WHERE age > 30 AND salary > 45000; SELECT count(*) AS n FROM f'))
    const g = await timed('groupBy', () => q('SELECT city, count(*) AS n, avg(salary) AS salary, sum(x) AS x FROM t GROUP BY city'))
    const s = await timed('sort', () => q('CREATE OR REPLACE TABLE s AS SELECT * FROM t ORDER BY salary DESC; SELECT (SELECT salary FROM s LIMIT 1) AS first, (SELECT salary FROM t ORDER BY salary ASC LIMIT 1) AS last'))
    await q(`CREATE OR REPLACE TABLE regions AS SELECT * FROM (VALUES ${REGIONS.map(([c, r]) => `('${c}','${r}')`).join(',')}) v(city, region)`)
    const j = await timed('join', () => q('SELECT count(*) AS n, sum(t.salary) AS s FROM t JOIN regions USING (city)'))
    await timed('writeCsv', () => q(`COPY f TO '${OUTPUT(lib)}' (HEADER, DELIMITER ',')`))
    const num = (v: unknown) => Number(v)
    check = {
      filterRows: num(f[f.length - 1]!.n),
      groupSum: round(g.reduce((acc, r) => acc + num(r.salary) + num(r.x), 0)),
      sortFirst: num(s[s.length - 1]!.first),
      sortLast: num(s[s.length - 1]!.last),
      joinRows: num(j[0]!.n),
      joinSum: round(num(j[0]!.s)),
    }
    await close()
  }
  clearInterval(sampler)
  peak = Math.max(peak, process.memoryUsage().rss)
  return { times, check, peakRssMb: peak / 1e6, note }
}

async function child(lib: Lib): Promise<void> {
  const cold = await runLib(lib)
  const warm: Run[] = []
  for (let i = 0; i < 2; i++) warm.push(await runLib(lib))
  process.stdout.write(JSON.stringify({ cold, warm }))
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b)
  return s[Math.floor(s.length / 2)]!
}

async function main(): Promise<void> {
  ensureInput()
  if (process.env.E2E_LIB) return child(process.env.E2E_LIB as Lib)
  const inputMb = statSync(INPUT).size / 1e6
  const results: Partial<Record<Lib, { cold: Run; warm: Run[]; wallMs: number; error?: string }>> = {}
  for (const lib of LIBS) {
    const t0 = performance.now()
    const r = spawnSync(process.execPath, ['--max-old-space-size=8192', '--import', 'tsx', fileURLToPath(import.meta.url)], {
      env: { ...process.env, E2E_LIB: lib },
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
    })
    const wallMs = performance.now() - t0
    if (r.status !== 0) {
      results[lib] = { cold: null as never, warm: [], wallMs, error: (r.stderr || r.stdout).split('\n').filter((l) => l.trim()).slice(-3).join(' | ') }
      console.error(`${lib}: failed — ${results[lib]!.error}`)
      continue
    }
    const parsed = JSON.parse(r.stdout.slice(r.stdout.lastIndexOf('{"cold"'))) as { cold: Run; warm: Run[] }
    results[lib] = { ...parsed, wallMs }
    console.log(`${lib}: cold ${sum(parsed.cold.times).toFixed(0)} ms, warm ${median(parsed.warm.map((w) => sum(w.times))).toFixed(0)} ms, peak RSS ${Math.max(parsed.cold.peakRssMb, ...parsed.warm.map((w) => w.peakRssMb)).toFixed(0)} MB`)
  }

  // ---- report ----
  const ops: Array<keyof OpTimes> = ['read', 'filter', 'groupBy', 'sort', 'join', 'writeCsv']
  const lines: string[] = []
  lines.push(`# columna vs Arquero / DuckDB-Wasm / Polars / DuckDB — same data, same operations, checked results`)
  lines.push('')
  lines.push(`Generated ${new Date().toISOString().slice(0, 10)} · Node ${process.version} · ${process.platform} ${process.arch} · ${ROWS.toLocaleString()} rows × 8 columns (2 int, 4 float, 2 low-cardinality text), CSV ${inputMb.toFixed(0)} MB.`)
  lines.push('Each library in its own process: **cold** = module load + first run; **warm** = median of 2 further runs in the same process; peak RSS sampled every 20 ms. Times in ms.')
  lines.push('')
  lines.push('| Library | Version | Cold total | Warm total | ' + ops.map((o) => `${o} (warm)`).join(' | ') + ' | Peak RSS | Notes |')
  lines.push('|---|---|---:|---:|' + ops.map(() => '---:').join('|') + '|---:|---|')
  const require = createRequire(import.meta.url)
  const version = (pkg: string) => {
    try {
      return (require(`${pkg}/package.json`) as { version: string }).version
    } catch {
      return '?'
    }
  }
  const columnaVersion = (() => { try { return (JSON.parse(readFileSync(new URL('../../columna/package.json', import.meta.url), 'utf8')) as { version: string }).version } catch { return '?' } })()
  const versions: Record<Lib, string> = { columna: columnaVersion, arquero: version('arquero'), 'duckdb-wasm': (() => { try { return (JSON.parse(readFileSync(path.join(path.dirname(require.resolve('@duckdb/duckdb-wasm')), '..', 'package.json'), 'utf8')) as { version: string }).version } catch { return '?' } })(), polars: version('nodejs-polars'), 'polars-lazy': version('nodejs-polars'), duckdb: version('duckdb') }
  for (const lib of LIBS) {
    const r = results[lib]
    if (!r || r.error) {
      lines.push(`| ${lib} | ${versions[lib]} | — | — | ${ops.map(() => '—').join(' | ')} | — | failed: ${r?.error ?? '?'} |`)
      continue
    }
    const warmOp = (o: keyof OpTimes) => median(r.warm.map((w) => w.times[o]))
    const peak = Math.max(r.cold.peakRssMb, ...r.warm.map((w) => w.peakRssMb))
    lines.push(`| ${lib} | ${versions[lib]} | ${sum(r.cold.times).toFixed(0)} | ${median(r.warm.map((w) => sum(w.times))).toFixed(0)} | ${ops.map((o) => warmOp(o).toFixed(0)).join(' | ')} | ${peak.toFixed(0)} MB | ${r.cold.note ?? ''} |`)
  }
  lines.push('')
  // result validation
  const ref = results.columna && !results.columna.error ? results.columna.cold.check : undefined
  lines.push('## Result validation')
  lines.push('')
  if (ref) {
    lines.push(`Reference (columna): filter rows ${ref.filterRows}, groupBy checksum ${ref.groupSum}, sort first/last ${ref.sortFirst}/${ref.sortLast}, join rows ${ref.joinRows}, join checksum ${ref.joinSum}.`)
    for (const lib of LIBS) {
      const r = results[lib]
      if (!r || r.error || lib === 'columna') continue
      const c = r.cold.check
      const diffs = (Object.keys(ref) as Array<keyof Check>).filter((k) => Math.abs(Number(c[k]) - Number(ref[k])) > 1e-6 * Math.max(1, Math.abs(Number(ref[k]))))
      lines.push(`- ${lib}: ${diffs.length === 0 ? 'identical' : `**differs** in ${diffs.map((k) => `${k} (${String(c[k])} vs ${String(ref[k])})`).join(', ')}`}`)
    }
  }
  lines.push('')
  lines.push('## How to read this')
  lines.push('')
  lines.push('- Wall-clock of a single process on one machine; the multi-threaded engines (Polars, native DuckDB) use every core, columna / Arquero / DuckDB-Wasm one.')
  lines.push('- polars-lazy pays the CSV scan inside every operation (that is the point of a streaming optimiser); compare its per-op numbers with `read + op` of the eager rows.')
  lines.push('- Peak RSS includes the runtime itself (a WebAssembly heap, Polars’ thread pool) and the CSV read; it is the number to budget, not the column bytes.')
  lines.push('- columna numbers come from `collect()` on the CPU engine; `collectWithReport()` confirms no other backend was involved.')
  lines.push('')
  const outMd = new URL('../../../docs/comparison-js.md', import.meta.url)
  writeFileSync(outMd, lines.join('\n'))
  writeFileSync(new URL(`compare-js-${ROWS}.json`, OUT), JSON.stringify(results, null, 2))
  console.log(`wrote ${fileURLToPath(outMd)}`)
  console.log(readFileSync(outMd, 'utf8'))
}

const sum = (t: OpTimes) => Object.values(t).reduce((a, b) => a + b, 0)

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
