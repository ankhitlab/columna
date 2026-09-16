/**
 * Throughput + heap delta for SciPy parity Waves 1–3.
 *
 *   pnpm --filter @columna/bench exec -- node --expose-gc --max-old-space-size=8192 --import tsx src/scipy-parity-bench.ts
 */
import { performance } from 'node:perf_hooks'
import * as A from '../../../packages/advanced/src/index.ts'

const REPEAT = Number(process.env.BENCH_REPEAT ?? 7)
const WARMUP = 2

type Row = {
  id: string
  group: string
  label: string
  n: number
  msMedian: number
  msMin: number
  heapDeltaMB: number
  heapUsedMB: number
}

const g = A.random(20260915)
const normal = (n: number, mu = 0, sd = 1) => Array.from(g.normal(n, mu, sd))

function median(xs: number[]) {
  const s = xs.slice().sort((a, b) => a - b)
  return s[Math.floor(s.length / 2)]!
}

function memMB() {
  return process.memoryUsage().heapUsed / (1024 * 1024)
}

function bench(id: string, group: string, label: string, n: number, setup: () => () => unknown, repeat = REPEAT): Row {
  if (global.gc) global.gc()
  const before = memMB()
  const fn = setup()
  for (let i = 0; i < WARMUP; i++) fn()
  if (global.gc) global.gc()
  const heap0 = memMB()
  const times: number[] = []
  let last: unknown
  for (let i = 0; i < repeat; i++) {
    const t0 = performance.now()
    last = fn()
    times.push(performance.now() - t0)
  }
  void last
  const heap1 = memMB()
  return {
    id,
    group,
    label,
    n,
    msMedian: median(times),
    msMin: Math.min(...times),
    heapDeltaMB: heap1 - heap0,
    heapUsedMB: memMB() - before + (heap1 - heap0),
  }
}

const rows: Row[] = []

// ---- Wave 1 ----
{
  const n = 50_000
  const a = normal(n)
  const b = a.map((v, i) => 0.6 * v + normal(1)[0]! + (i % 7) * 0.01)
  const z = normal(n)
  const nk = 2_000
  rows.push(bench('corrTest.kendall', 'W1', `corrTest kendall n=${nk}`, nk, () => () => A.corrTest(a.slice(0, nk), b.slice(0, nk), { method: 'kendall' }), 5))
  rows.push(bench('partialCorr', 'W1', `partialCorr n=${n}`, n, () => () => A.partialCorr(a, b, [z])))
  rows.push(bench('jarqueBera', 'W1', `jarqueBera n=${n}`, n, () => () => A.jarqueBera(a)))
  rows.push(bench('dagostinoK2', 'W1', `dagostinoK2 n=${n}`, n, () => () => A.dagostinoK2(a)))
  rows.push(bench('cramerVonMises', 'W1', `cramerVonMises n=${n}`, n, () => () => A.cramerVonMises(a)))
  const g1 = normal(10_000, 0, 1)
  const g2 = normal(10_000, 0, 2)
  rows.push(bench('fligner', 'W1', 'fligner 2×10k', 20_000, () => () => A.fligner([g1, g2])))
  rows.push(bench('ansariBradley', 'W1', 'ansariBradley 10k+10k', 20_000, () => () => A.ansariBradley(g1, g2)))
  rows.push(bench('brunnerMunzel', 'W1', 'brunnerMunzel 10k+10k', 20_000, () => () => A.brunnerMunzel(g1, g2)))
  rows.push(bench('mcnemar', 'W1', 'mcnemar 2×2', 4, () => () => A.mcnemar([[40, 12], [5, 80]])))
  rows.push(bench('cohensD', 'W1', 'cohensD 10k+10k', 20_000, () => () => A.cohensD(g1, g2)))
  rows.push(bench('gaussianKde', 'W1', 'gaussianKde n=5k grid=200', 5000, () => {
    const x = normal(5000)
    return () => A.gaussianKde(x).grid(200)
  }))
  rows.push(bench('bootstrap', 'W1', 'bootstrap n=2k nBoot=999', 2000, () => {
    const x = normal(2000)
    return () => A.bootstrap(x, (s) => {
      let m = 0
      for (let i = 0; i < s.length; i++) m += s[i]!
      return m / s.length
    }, { nBoot: 999, seed: 1 })
  }, 3))
  rows.push(bench('permutationTest', 'W1', 'permutation 1k+1k nPerm=999', 2000, () => () => A.permutationTest(g1.slice(0, 1000), g2.slice(0, 1000), { nPerm: 999, seed: 2 }), 3))
  rows.push(bench('negativeBinomial.cdf', 'W1', 'nbinom cdf ×50k', 50_000, () => {
    const d = A.negativeBinomial(5, 0.4)
    const k = Array.from({ length: 50_000 }, (_, i) => i % 40)
    return () => { let s = 0; for (const v of k) s += d.cdf(v); return s }
  }))
  rows.push(bench('hypergeometric.pmf', 'W1', 'hypergeom pmf ×50k', 50_000, () => {
    const d = A.hypergeometric(50, 20, 10)
    return () => { let s = 0; for (let i = 0; i < 50_000; i++) s += d.pmf(i % 11); return s }
  }))
  rows.push(bench('gumbel.ppf', 'W1', 'gumbel.ppf ×100k', 100_000, () => {
    const d = A.gumbel()
    const u = Array.from(g.uniform(100_000))
    return () => d.map('ppf', u)
  }))
  rows.push(bench('invgauss.cdf', 'W1', 'invgauss.cdf ×20k', 20_000, () => {
    const d = A.invgauss(1, 2)
    const x = Array.from({ length: 20_000 }, (_, i) => 0.1 + (i % 100) * 0.05)
    return () => { let s = 0; for (const v of x) s += d.cdf(v); return s }
  }, 3))
}

// ---- Wave 2 ----
{
  const n = 5_000
  const y = normal(n, 10, 3)
  const row = Array.from({ length: n }, (_, i) => (i % 3 === 0 ? 'A' : i % 3 === 1 ? 'B' : 'C'))
  const col = Array.from({ length: n }, (_, i) => String(i % 4))
  const cov = normal(n)
  rows.push(bench('anovaTwoWay', 'W2', `anovaTwoWay n=${n}`, n, () => () => A.anovaTwoWay(y, row, col)))
  rows.push(bench('ancova', 'W2', `ancova n=${n}`, n, () => () => A.ancova(y, row, cov)))
  const series = (() => {
    let x = 0
    return Array.from({ length: 5_000 }, () => {
      x = 0.7 * x + g.normal(1)[0]!
      return x
    })
  })()
  rows.push(bench('adfTest', 'W2', 'adfTest n=5k', 5000, () => () => A.adfTest(series, { lags: 5 })))
  rows.push(bench('kpssTest', 'W2', 'kpssTest n=5k', 5000, () => () => A.kpssTest(series)))
  const ts = Array.from({ length: 480 }, (_, i) => Math.sin((2 * Math.PI * i) / 12) + 0.4 * Math.sin((2 * Math.PI * i) / 5) + i * 0.01 + g.normal(1)[0]! * 0.1)
  rows.push(bench('stl.multi', 'W2', 'stl periods=[12,5] n=480', 480, () => () => A.stl(ts, { periods: [12, 5] }), 5))
  const yar = Array.from({ length: 200 }, (_, i) => 2 + 0.3 * i + Math.sin(i / 3) + g.normal(1)[0]! * 0.2)
  const xreg = Array.from({ length: 200 }, (_, i) => Math.sin(i / 5))
  rows.push(bench('arima.ml.transfer', 'W2', 'arima ML+transfer n=200', 200, () => () => A.arima(yar, { p: 1, d: 0, q: 0, method: 'ML', transfer: [{ x: xreg }] }), 3))
  {
    const yy: number[] = []
    const X: number[][] = []
    const group: string[] = []
    const slope: number[] = []
    for (let j = 0; j < 12; j++) {
      for (let i = 0; i < 15; i++) {
        const z = (i - 7) / 7
        yy.push(i + j > 10 ? 1 : 0)
        X.push([z])
        group.push(`g${j}`)
        slope.push(z)
      }
    }
    rows.push(bench('glmm.agq.rho', 'W2', 'glmm AGQ+slope n=180 g=12', 180, () => () => A.glmm(yy, { family: 'binomial', fixed: X, group, slope, method: 'agq', nAGQ: 5 }), 3))
  }
  rows.push(bench('eppsSingleton', 'W2', 'eppsSingleton 2k+2k', 4000, () => () => A.eppsSingleton(normal(2000), normal(2000, 0.5))))
}

// ---- Wave 3 ----
{
  const a = normal(10_000)
  const b = normal(10_000, 0.3)
  rows.push(bench('ksTwoSample', 'W3', 'ksTwoSample 10k+10k', 20_000, () => () => A.ksTwoSample(a, b)))
  rows.push(bench('andersonKSample', 'W3', 'andersonKSample 3×3k', 9000, () => () => A.andersonKSample([normal(3000), normal(3000, 0.2), normal(3000, 0.4)])))
  rows.push(bench('energyDistance', 'W3', 'energyDistance 2k+2k', 4000, () => () => A.energyDistance(normal(2000), normal(2000, 1)), 3))
  const x = Array.from({ length: 5_000 }, (_, i) => i / 100)
  const y = x.map((v) => 2 * v + g.normal(1)[0]! * 0.3)
  rows.push(bench('lowess', 'W3', 'lowess n=5k', 5000, () => () => A.lowess(x, y, { frac: 0.3 }), 3))
  rows.push(bench('isotonic', 'W3', 'isotonic n=50k', 50_000, () => () => A.isotonicRegression(normal(50_000))))
  const X = Array.from({ length: 5_000 }, () => Array.from(g.normal(8)))
  const yy = X.map((r) => 1 + r.reduce((s, v, j) => s + (j + 1) * 0.2 * v, 0) + g.normal(1)[0]! * 0.5)
  rows.push(bench('ridge', 'W3', 'ridge n=5k p=8', 5000, () => () => A.ridge(yy, X, { alpha: 1 })))
  rows.push(bench('lasso', 'W3', 'lasso n=5k p=8', 5000, () => () => A.lasso(yy, X, { alpha: 0.1 }), 3))
  rows.push(bench('quantileRegression', 'W3', 'QR n=2k p=5', 2000, () => {
    const Xp = X.slice(0, 2000).map((r) => r.slice(0, 5))
    const yp = yy.slice(0, 2000)
    return () => A.quantileRegression(yp, Xp, { tau: 0.5 })
  }, 3))
  const pts = Array.from({ length: 2_000 }, () => [g.normal(1)[0]! * 3 + (Math.random() > 0.5 ? 8 : 0), g.normal(1)[0]!])
  rows.push(bench('dbscan', 'W3', 'dbscan n=2k', 2000, () => () => A.dbscan(pts, { eps: 0.8, minSamples: 5 }), 3))
  rows.push(bench('pdist', 'W3', 'pdist n=800', 800, () => () => A.pdist(pts.slice(0, 800)), 3))
  {
    const M = A.matrix(200, 20)
    for (let i = 0; i < 200 * 20; i++) M.data[i] = Math.abs(g.normal(1)[0]!)
    const bb = Float64Array.from({ length: 200 }, () => Math.abs(g.normal(1)[0]!))
    rows.push(bench('nnls', 'W3', 'nnls 200×20', 200, () => () => A.nnls(M, bb), 5))
  }
  const sig = Array.from({ length: 8_192 }, (_, i) => Math.sin((2 * Math.PI * i) / 32) + 0.2 * g.normal(1)[0]!)
  rows.push(bench('welchPsd', 'W3', 'welchPsd n=8192', 8192, () => () => A.welchPsd(sig, { nperseg: 256 }), 3))
  rows.push(bench('savitzkyGolay', 'W3', 'savgol n=50k', 50_000, () => () => A.savitzkyGolay(normal(50_000), { windowLength: 21, polyOrder: 3 })))
  rows.push(bench('brentq', 'W3', 'brentq ×10k', 10_000, () => () => {
    let s = 0
    for (let i = 0; i < 10_000; i++) s += A.brentq((x) => x * x - (1 + (i % 7) * 0.1), 0, 5)
    return s
  }))
  rows.push(bench('trapz', 'W3', 'trapz n=1e6', 1_000_000, () => {
    const yy = normal(1_000_000)
    return () => A.trapz(yy)
  }))
}

rows.sort((a, b) => b.msMedian - a.msMedian)

const fmt = (r: Row) =>
  `${r.msMedian.toFixed(r.msMedian >= 10 ? 1 : 3).padStart(8)} ms  Δheap ${r.heapDeltaMB.toFixed(1).padStart(6)} MB  ${r.group}  ${r.label}`

console.log(`SciPy parity bench  repeats=${REPEAT}  node=${process.version}`)
console.log(`heapUsed start≈${(process.memoryUsage().heapUsed / 1024 / 1024).toFixed(1)} MB\n`)
console.log('— by median time (slowest first) —')
for (const r of rows) console.log(fmt(r))

const byGroup = new Map<string, Row[]>()
for (const r of rows) {
  const g0 = byGroup.get(r.group) ?? []
  g0.push(r)
  byGroup.set(r.group, g0)
}
console.log('\n— group totals (sum of medians) —')
for (const [group, list] of byGroup) {
  const sum = list.reduce((s, r) => s + r.msMedian, 0)
  const heap = Math.max(...list.map((r) => r.heapDeltaMB))
  console.log(`${group}: ${sum.toFixed(1)} ms across ${list.length} cases; peak Δheap among cases ${heap.toFixed(1)} MB`)
}

const heavy = rows.filter((r) => r.msMedian >= 50)
const light = rows.filter((r) => r.msMedian < 5)
console.log(`\n≥50 ms: ${heavy.length}  |  <5 ms: ${light.length}  |  total cases: ${rows.length}`)
console.log(JSON.stringify({ cases: rows }, null, 2))
