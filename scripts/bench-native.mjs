import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
const require = createRequire(import.meta.url)
const { filterAnd2I32F64 } = require(join(dirname(fileURLToPath(import.meta.url)), '../packages/native/index.js'))

function median(xs) {
  const a = [...xs].sort((x, y) => x - y)
  return a[Math.floor(a.length / 2)]
}

function jsDual(a, b, la, lb) {
  const n = a.length
  const tmp = new Uint32Array(n)
  let j = 0
  for (let i = 0; i < n; i++) {
    tmp[j] = i
    j += (a[i] > la ? 1 : 0) & (b[i] > lb ? 1 : 0)
  }
  return tmp.subarray(0, j)
}

function time(fn, runs = 5) {
  fn()
  const s = []
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now()
    fn()
    s.push(performance.now() - t0)
  }
  return median(s)
}

for (const n of [1_000_000, 5_000_000, 20_000_000]) {
  const a = new Int32Array(n)
  const b = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    a[i] = 18 + (i % 50)
    b[i] = 40_000 + (i % 1000) * 10
  }
  const jsMs = time(() => jsDual(a, b, 30, 45_000))
  const rustMs = time(() => filterAnd2I32F64(a, b, 2, 30, 2, 45_000))
  console.log(JSON.stringify({ n, jsMs: +jsMs.toFixed(2), rustMs: +rustMs.toFixed(2), speedup: +(jsMs / rustMs).toFixed(2) }))
}
