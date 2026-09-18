/**
 * Print vitest's json-summary as a Markdown table (CI appends it to the job summary).
 *   node scripts/coverage-summary.mjs coverage/coverage-summary.json
 */
import { readFileSync } from 'node:fs'
const file = process.argv[2] ?? 'coverage/coverage-summary.json'
const { total } = JSON.parse(readFileSync(file, 'utf8'))
const row = (k) => `| ${k} | ${total[k].pct.toFixed(2)}% | ${total[k].covered} / ${total[k].total} |`
console.log('### Test coverage (v8, published packages\' src/)')
console.log('')
console.log('| | covered | of |')
console.log('|---|---:|---:|')
for (const k of ['lines', 'statements', 'branches', 'functions']) console.log(row(k))
console.log('')
console.log('Full HTML report: the `coverage` artifact of this run.')

// --badge <file>: shields.io "endpoint" JSON (https://img.shields.io/endpoint?url=…) for the README badge
const badgeAt = process.argv.indexOf('--badge')
if (badgeAt > 0 && process.argv[badgeAt + 1]) {
  const { writeFileSync, mkdirSync } = await import('node:fs')
  const { dirname } = await import('node:path')
  const pct = total.lines.pct
  const color = pct >= 90 ? 'brightgreen' : pct >= 80 ? 'green' : pct >= 70 ? 'yellowgreen' : pct >= 60 ? 'yellow' : 'orange'
  const out = process.argv[badgeAt + 1]
  mkdirSync(dirname(out), { recursive: true })
  writeFileSync(out, JSON.stringify({ schemaVersion: 1, label: 'coverage', message: `${pct.toFixed(1)}% lines`, color }))
  console.error(`badge → ${out}`)
}
