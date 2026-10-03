/**
 * Compare two columna bench JSON dumps (before vs after).
 * Usage: node --import tsx scripts/diff-bench.ts before.json after.json
 */
import { readFileSync } from 'node:fs'

type Row = { library: string; n: number; op: string; ms: number }
type Payload = { results: Row[] }

function load(path: string): Map<string, number> {
  const data = JSON.parse(readFileSync(path, 'utf8')) as Payload
  const m = new Map<string, number>()
  for (const r of data.results) {
    if (r.op === 'build' || r.op === 'unavailable') continue
    m.set(`${r.library}|${r.n}|${r.op}`, r.ms)
  }
  return m
}

const beforePath = process.argv[2]!
const afterPath = process.argv[3]!
const before = load(beforePath)
const after = load(afterPath)

const keys = [...new Set([...before.keys(), ...after.keys()])].sort()

type Diff = {
  key: string
  before: number
  after: number
  deltaMs: number
  pct: number
}

const diffs: Diff[] = []
for (const key of keys) {
  const b = before.get(key)
  const a = after.get(key)
  if (b == null || a == null) continue
  const deltaMs = a - b
  const pct = b === 0 ? 0 : (deltaMs / b) * 100
  diffs.push({ key, before: b, after: a, deltaMs, pct })
}

const meaningful = diffs.filter((d) => d.before >= 2 || d.after >= 2)
const regressed = meaningful.filter((d) => d.pct > 15 && d.deltaMs > 1).sort((a, b) => b.pct - a.pct)
const improved = meaningful.filter((d) => d.pct < -15 && d.deltaMs < -1).sort((a, b) => a.pct - b.pct)

const sumBefore = meaningful.reduce((s, d) => s + d.before, 0)
const sumAfter = meaningful.reduce((s, d) => s + d.after, 0)

console.log(`Compared ${meaningful.length} ops (excluding <2ms noise)`)
console.log(`Total ms before=${sumBefore.toFixed(1)} after=${sumAfter.toFixed(1)} delta=${(sumAfter - sumBefore).toFixed(1)} (${(((sumAfter - sumBefore) / sumBefore) * 100).toFixed(1)}%)`)
console.log('')

console.log('Top regressions (>+15% and >+1ms):')
if (regressed.length === 0) console.log('  (none)')
for (const d of regressed.slice(0, 15)) {
  console.log(
    `  ${d.key.padEnd(42)} ${d.before.toFixed(2).padStart(8)} → ${d.after.toFixed(2).padStart(8)}  ${d.pct >= 0 ? '+' : ''}${d.pct.toFixed(1)}%`,
  )
}

console.log('')
console.log('Top improvements (<-15% and <-1ms):')
if (improved.length === 0) console.log('  (none)')
for (const d of improved.slice(0, 15)) {
  console.log(
    `  ${d.key.padEnd(42)} ${d.before.toFixed(2).padStart(8)} → ${d.after.toFixed(2).padStart(8)}  ${d.pct.toFixed(1)}%`,
  )
}

console.log('')
console.log('All cpu ops @ 1M:')
for (const d of diffs.filter((x) => x.key.includes('columna:cpu|1000000|'))) {
  const mark = d.pct > 15 && d.deltaMs > 1 ? '  SLOWER' : d.pct < -15 && d.deltaMs < -1 ? '  FASTER' : ''
  console.log(
    `  ${d.key.split('|').pop()!.padEnd(16)} ${d.before.toFixed(2).padStart(8)} → ${d.after.toFixed(2).padStart(8)}  ${d.pct >= 0 ? '+' : ''}${d.pct.toFixed(1)}%${mark}`,
  )
}
