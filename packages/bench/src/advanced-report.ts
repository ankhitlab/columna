/**
 * Merge results/advanced-columna.json and results/advanced-python.json into docs/advanced-benchmarks.md.
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs'

const dir = new URL('../results/', import.meta.url)
const tsPath = new URL('advanced-columna.json', dir)
const pyPath = new URL('advanced-python.json', dir)
const outPath = new URL('../../../docs/advanced-benchmarks.md', import.meta.url)

interface TsRow { id: string; group: string; label: string; n: number; msMedian: number; msMin: number; repeat: number }
interface PyRow { id: string; lib: string; msMedian: number; msMin: number }

const ts = JSON.parse(readFileSync(tsPath, 'utf8')) as { generatedAt: string; node: string; scale: number; repeat: number; rows: TsRow[] }
const py = existsSync(pyPath) ? (JSON.parse(readFileSync(pyPath, 'utf8')) as { numpy: string; pandas: string; polars: string | null; scipy: string; rows: PyRow[] }) : null

const pyBy = new Map<string, PyRow[]>()
for (const r of py?.rows ?? []) {
  const list = pyBy.get(r.id) ?? []
  list.push(r)
  pyBy.set(r.id, list)
}

const fmt = (ms: number) => (ms < 0.1 ? ms.toFixed(3) + ' ms' : ms < 10 ? ms.toFixed(2) + ' ms' : ms < 1000 ? ms.toFixed(1) + ' ms' : (ms / 1000).toFixed(2) + ' s')
const ratio = (a: number, b: number) => {
  const r = b / a
  return r >= 1 ? `${r.toFixed(r >= 10 ? 0 : 1)}× faster` : `${(1 / r).toFixed(1 / r >= 10 ? 0 : 1)}× slower`
}

const groups = new Map<string, TsRow[]>()
for (const r of ts.rows) {
  const list = groups.get(r.group) ?? []
  list.push(r)
  groups.set(r.group, list)
}

let faster = 0
let slower = 0
let compared = 0
const lines: string[] = []
lines.push('# Benchmarks: `columna/advanced` vs pandas / polars / scipy / numpy')
lines.push('')
lines.push(`Generated ${ts.generatedAt.slice(0, 10)} · Node ${ts.node} · scale ${ts.scale} · median of ${ts.repeat} runs after a warm-up (heavy cases fewer).`)
if (py) lines.push(`Python: numpy ${py.numpy}, scipy ${py.scipy}, pandas ${py.pandas}${py.polars ? `, polars ${py.polars}` : ''}.`)
lines.push('')
lines.push('How to read: every row is one `columna/advanced` function on synthetic data of the stated size. The Python column times the closest')
lines.push('equivalent on data of the same size and distribution — **not** the same numbers (numerical agreement is covered by the vitest fixtures).')
lines.push('Where Python has no equivalent (statsmodels / sklearn / lifelines are not installed, or the tool is Minitab-only) the cell is "—".')
lines.push('Comparisons are throughput only: columna often computes more per call (full Minitab table: SE, CI, diagnostics, unusual observations),')
lines.push('and the scipy.optimize rows use a generic BFGS on the same likelihood rather than a purpose-built estimator.')
lines.push('')
lines.push('Regenerate: `pnpm run bench:advanced` (TS → Python → this file). Quick pass: `BENCH_SCALE=0.1 pnpm run bench:advanced`.')
lines.push('')
for (const [group, rows] of groups) {
  lines.push(`## ${group}`)
  lines.push('')
  lines.push('| Function (size) | columna | Python equivalent | Python | columna vs Python |')
  lines.push('|---|---:|---|---:|---|')
  for (const r of rows) {
    const pys = pyBy.get(r.id) ?? []
    if (!pys.length) {
      lines.push(`| ${r.label} | ${fmt(r.msMedian)} | — | — | — |`)
      continue
    }
    pys.forEach((p, i) => {
      compared++
      if (p.msMedian >= r.msMedian) faster++
      else slower++
      lines.push(`| ${i === 0 ? r.label : ''} | ${i === 0 ? fmt(r.msMedian) : ''} | ${p.lib} | ${fmt(p.msMedian)} | ${ratio(r.msMedian, p.msMedian)} |`)
    })
  }
  lines.push('')
}
lines.push('## Summary')
lines.push('')
lines.push(`- ${ts.rows.length} functions timed; ${compared} comparisons against Python libraries: columna faster or equal in ${faster}, slower in ${slower}.`)
const worst = ts.rows.filter((r) => (pyBy.get(r.id) ?? []).length).map((r) => ({ r, p: Math.min(...(pyBy.get(r.id) ?? []).map((p) => p.msMedian)) })).sort((a, b) => a.p / a.r.msMedian - b.p / b.r.msMedian).slice(0, 8)
lines.push(`- Largest gaps vs the fastest Python library: ${worst.map((w) => `${w.r.id} (${(w.r.msMedian / w.p).toFixed(1)}× slower)`).join(', ')}.`)
const best = ts.rows.filter((r) => (pyBy.get(r.id) ?? []).length).map((r) => ({ r, p: Math.min(...(pyBy.get(r.id) ?? []).map((p) => p.msMedian)) })).sort((a, b) => b.p / b.r.msMedian - a.p / a.r.msMedian).slice(0, 8)
lines.push(`- Largest wins: ${best.map((w) => `${w.r.id} (${(w.p / w.r.msMedian).toFixed(1)}× faster)`).join(', ')}.`)
lines.push('')
lines.push('Notes: all timings are single-threaded JavaScript (V8 JIT, no WebAssembly / GPU) versus compiled C / Fortran / Rust kernels on the Python side.')
lines.push('')
writeFileSync(outPath, lines.join('\n'))
console.log(`wrote ${outPath.pathname}: ${ts.rows.length} rows, ${compared} comparisons (faster ${faster}, slower ${slower})`)
