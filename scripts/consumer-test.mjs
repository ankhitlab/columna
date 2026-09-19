/**
 * Install the *packed* packages into a clean consumer project and use them the way a user would:
 * ESM import, CommonJS require, subpath exports (`columna/core`, `columna/advanced`), the TypeScript
 * declarations, and a Node script that reads a CSV, runs a plan and a statistical test.
 *
 *   node scripts/consumer-test.mjs        (after `pnpm build`)
 *
 * This catches what workspace tests cannot: a missing `files` entry, a broken `exports` map, a sibling
 * module the runtime loads by relative path that did not make it into the tarball, or a Node-only import
 * that leaks into the bundle.
 */
import { spawnSync } from 'node:child_process'
import { mkdtempSync, mkdirSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const work = mkdtempSync(path.join(tmpdir(), 'columna-consumer-'))
const packDir = path.join(work, 'tarballs')
const app = path.join(work, 'app')
mkdirSync(packDir)
mkdirSync(app)
// with shell: true on Windows the plain names resolve through PATH (npm.cmd / pnpm.cmd shims)
const npm = 'npm'
const pnpm = 'pnpm'
const run = (cmd, args, cwd) => {
  // shell only for the npm / pnpm shims; node / tsc are spawned directly (paths with spaces)
  const viaShell = process.platform === 'win32' && (cmd === npm || cmd === pnpm)
  const r = spawnSync(cmd, args, { cwd, encoding: 'utf8', shell: viaShell })
  if (r.status !== 0) {
    console.error(r.stdout)
    console.error(r.stderr)
    throw new Error(`${cmd} ${args.join(' ')} failed in ${cwd}`)
  }
  return r.stdout
}

// 1. pack every published package (pnpm rewrites workspace:* to real versions)
const published = ['arrow', 'runtime', 'wasm', 'webgpu', 'core', 'advanced', 'columna']
for (const p of published) run(pnpm, ['pack', '--pack-destination', packDir], path.join(root, 'packages', p))
const tarballs = readdirSync(packDir).map((f) => path.join(packDir, f))
console.log(`packed ${tarballs.length} tarballs`)

// 2. clean consumer: npm (not pnpm) so the install path is the one most users take
writeFileSync(path.join(app, 'package.json'), JSON.stringify({ name: 'consumer', private: true, version: '0.0.0' }, null, 2))
run(npm, ['install', '--no-audit', '--no-fund', '--loglevel=error', ...tarballs], app)
const installed = JSON.parse(run(npm, ['ls', '--json', '--depth=0'], app))
console.log('installed:', Object.keys(installed.dependencies ?? {}).join(', '))

// 3. ESM consumer
writeFileSync(path.join(app, 'esm.mjs'), `
import { DataFrame, col, init, setIoPolicy, formatExecutionReport, ExecutionAbortedError, fromArrowIpc, createSession } from 'columna'
import { ttest1, quantile } from 'columna/advanced'
import { DataFrame as CoreDataFrame } from 'columna/core'
import { writeFileSync } from 'node:fs'
await init()
setIoPolicy({ allowedDirs: [process.cwd()], maxBytes: 10e6 })
writeFileSync('data.csv', 'id,x,g\\n' + Array.from({ length: 5000 }, (_, i) => i + ',' + (i % 97) / 7 + ',' + 'abc'[i % 3]).join('\\n') + '\\n')
const df = await DataFrame.readCsv({ path: 'data.csv' })
const { frame, report } = await df.filter((c) => c.x.gt(3)).groupBy('g').agg((c) => ({ n: c.id.count(), m: c.x.mean() })).collectWithReport()
if (frame.shape[0] !== 3 || report.backendsUsed.length === 0) throw new Error('unexpected result ' + formatExecutionReport(report))
const t = ttest1(df.getColumn('x').toArray(), { mu: 6 })
if (!(t.pValue >= 0 && t.pValue <= 1)) throw new Error('ttest1')
if (quantile([1, 2, 3, 4], 0.25) !== 1.25) throw new Error('quantile')
if (typeof CoreDataFrame !== 'function') throw new Error('columna/core')
await frame.writeCsv('out.csv')
// Arrow IPC round trip through the packed package, and a pre-aborted collect
const ipc = frame.toArrowIpc()
if (DataFrame.fromArrowIpc(ipc).shape[0] !== 3 || fromArrowIpc(ipc).numRows !== 3) throw new Error('arrow ipc')
const ac = new AbortController(); ac.abort()
const aborted = await df.filter((c) => c.x.gt(1)).collect({ signal: ac.signal }).catch((e) => e)
if (!(aborted instanceof ExecutionAbortedError)) throw new Error('abort: ' + aborted)
// a session: own runtime + cache + IO floor
const tenant = createSession({ io: { allowedDirs: [process.cwd()] }, persist: {} })
const td = await tenant.readCsv({ path: 'data.csv' })
const { report: r1 } = await td.lazy().groupBy('g').agg({ n: col('id').count() }).persist().collectWithReport()
const { report: r2 } = await td.lazy().groupBy('g').agg({ n: col('id').count() }).collectWithReport()
if (r1.cacheHit || !r2.cacheHit || tenant.persistCache.stats().entries !== 1) throw new Error('session persist')
tenant.close()
console.log('esm ok', frame.shape, report.backendsUsed.join('+'))
`)
console.log(run(process.execPath, ['esm.mjs'], app).trim())

// 4. CommonJS consumer
writeFileSync(path.join(app, 'cjs.cjs'), `
const { DataFrame, col } = require('columna')
const { normal } = require('columna/advanced')
;(async () => {
  const f = await DataFrame.fromRows([{ a: 1 }, { a: 2 }]).filter(col('a').gt(1)).collect()
  if (f.shape[0] !== 1) throw new Error('cjs filter')
  if (Math.abs(normal().cdf(1.96) - 0.9750021048517795) > 1e-12) throw new Error('cjs advanced')
  // CJS + spill: createRequire must not see empty import.meta.url
  await DataFrame.fromColumns({ x: new Float64Array([3, 2, 1]) })
    .sort('x')
    .collect({ memory: { maxBytes: 1 } })
  console.log('cjs ok')
})().catch((err) => {
  console.error(err)
  process.exit(1)
})
`)
console.log(run(process.execPath, ['cjs.cjs'], app).trim())

// 5. TypeScript declarations resolve through the exports map (typed schema included)
writeFileSync(path.join(app, 'types.ts'), `
import { DataFrame } from 'columna'
import { quantile } from 'columna/advanced'
const df = DataFrame.fromRows([{ city: 'x', n: 1 }])
const rows: Array<{ city: string; n: number }> = df.toArray()
// @ts-expect-error unknown column must not compile
df.select('nope')
const q: number = quantile([1, 2], 0.5)
export { rows, q }
`)
writeFileSync(path.join(app, 'tsconfig.json'), JSON.stringify({ compilerOptions: { strict: true, module: 'NodeNext', moduleResolution: 'NodeNext', target: 'ES2022', noEmit: true, skipLibCheck: true }, files: ['types.ts'] }))
// run the repo's TypeScript compiler directly through node (no .cmd shim, no shell quoting issues)
const tscJs = path.join(root, 'node_modules', 'typescript', 'lib', 'tsc.js')
run(process.execPath, [tscJs, '-p', 'tsconfig.json'], app)
console.log('types ok')

rmSync(work, { recursive: true, force: true })
console.log('consumer install test passed')
