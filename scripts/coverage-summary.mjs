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
