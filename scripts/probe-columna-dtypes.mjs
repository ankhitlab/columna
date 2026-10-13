/**
 * Local probe for CSV text-literal / dtypes.utf8 behaviour.
 * After the fix: leading zeros, quoted zeros, and +phones keep their literals under utf8.
 *
 *   node scripts/probe-columna-dtypes.mjs
 */
import { pathToFileURL } from 'node:url'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')

async function loadDataFrame() {
  for (const rel of ['packages/columna/dist/index.js', 'packages/core/dist/index.js']) {
    try {
      const mod = await import(pathToFileURL(resolve(root, rel)).href)
      if (mod.DataFrame) return mod.DataFrame
    } catch {
      /* try next */
    }
  }
  throw new Error('Build packages/columna (or packages/core) first: pnpm --filter columna build')
}

const DataFrame = await loadDataFrame()

const cases = [
  { name: 'leading zero + utf8', csv: 'c\n0070\n', opts: { dtypes: { c: 'utf8' } }, want: '0070' },
  { name: 'quoted leading zero + utf8', csv: 'c\n"0070"\n', opts: { dtypes: { c: 'utf8' } }, want: '0070' },
  { name: 'phone + utf8', csv: 'c\n+79001234567\n', opts: { dtypes: { c: 'utf8' } }, want: '+79001234567' },
  { name: 'leading zero infer', csv: 'c\n0070\n', opts: {}, want: '0070' },
  { name: 'plain int infer', csv: 'c\n70\n', opts: {}, want: 70 },
  { name: 'date dotted infer', csv: 'c\n05.02.2026\n', opts: {}, want: '05.02.2026' },
]

let failed = 0
for (const c of cases) {
  const got = DataFrame.fromCSV(c.csv, c.opts).toArray()[0]?.c
  const ok = Object.is(got, c.want)
  console.log(`${ok ? 'OK' : 'FAIL'}  ${c.name}: got=${JSON.stringify(got)} want=${JSON.stringify(c.want)}`)
  if (!ok) failed++
}

if (failed) {
  console.error(`\n${failed} case(s) failed`)
  process.exit(1)
}
console.log('\nall probes passed')
