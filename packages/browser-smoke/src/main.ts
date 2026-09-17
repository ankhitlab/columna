/**
 * Runs in a real browser against the built `columna` bundle. Every check writes to `window.__smoke`;
 * run.mjs (Playwright) reads it and fails the process on any `ok: false`.
 *
 * Checks: import has no side effects (no GPU before init), CSV → filter → groupBy → join → CSV round trip,
 * typed exactness (a late 2^31 widens), and — when the browser exposes WebGPU — the strict GPU engine
 * returning exactly the CPU rows plus an execution report naming the gpu kernel.
 */
import { DataFrame, col, init, isWebGpuAvailable, formatExecutionReport, EngineStrictError } from 'columna'
import { quantile, ttest1 } from 'columna/advanced'

type Check = { name: string; ok: boolean; detail?: string; ms?: number }
const checks: Check[] = []
const log = document.getElementById('log')!
const report = (c: Check) => {
  checks.push(c)
  log.textContent += `${c.ok ? 'ok ' : 'FAIL'} ${c.name}${c.ms !== undefined ? ` ${c.ms.toFixed(1)} ms` : ''}${c.detail ? ` — ${c.detail}` : ''}\n`
}
async function check(name: string, fn: () => Promise<string | void> | string | void): Promise<void> {
  const t0 = performance.now()
  try {
    const detail = await fn()
    report({ name, ok: true, detail: detail ?? undefined, ms: performance.now() - t0 })
  } catch (e) {
    report({ name, ok: false, detail: e instanceof Error ? e.message : String(e), ms: performance.now() - t0 })
  }
}
const assert = (cond: unknown, msg: string) => {
  if (!cond) throw new Error(msg)
}

declare global {
  interface Window {
    __smoke?: { done: boolean; checks: Check[]; webgpu: boolean; userAgent: string }
  }
}

async function main(): Promise<void> {
  const hasGpu = typeof navigator !== 'undefined' && 'gpu' in navigator && Boolean((navigator as { gpu?: unknown }).gpu)

  await check('import is side-effect free: engine("webgpu", strict) refuses before init()', async () => {
    const df = DataFrame.fromRows(Array.from({ length: 20_000 }, (_, i) => ({ x: i })))
    let threw = false
    try {
      await df.filter(col('x').gt(5)).engine('webgpu', { strict: true }).collect()
    } catch (e) {
      threw = e instanceof EngineStrictError
    }
    assert(threw, 'expected EngineStrictError before init()')
  })

  const gpuReady = await init()
    .then(() => (hasGpu ? isWebGpuAvailable() : false))
    .catch(() => false)

  const n = 200_000
  const csv = ['id,age,salary,city', ...Array.from({ length: n }, (_, i) => `${i},${18 + (i % 60)},${(20_000 + ((i * 7919) % 100_000) / 2).toFixed(2)},${['Berlin', 'Paris', 'Rome', 'Madrid'][i % 4]}`)].join('\n')
  let df = DataFrame.fromRows([{ id: 0, age: 0, salary: 0, city: '' }])

  await check(`CSV parse ${n.toLocaleString()} rows`, () => {
    df = DataFrame.fromCSV<{ id: number; age: number; salary: number; city: string }>(csv)
    assert(df.shape[0] === n && df.shape[1] === 4, `shape ${df.shape}`)
    assert(df.dtypes.id === 'i32' && df.dtypes.salary === 'f64' && df.dtypes.city === 'category', JSON.stringify(df.dtypes))
  })

  let filteredRows = 0
  await check('filter + groupBy + join on the CPU engine', async () => {
    const f = await df.filter((c) => c.age.gt(30).and(c.salary.gt(45_000))).collect()
    filteredRows = f.shape[0]
    let expected = 0
    for (let i = 0; i < n; i++) if (18 + (i % 60) > 30 && Number((20_000 + ((i * 7919) % 100_000) / 2).toFixed(2)) > 45_000) expected++
    assert(filteredRows === expected, `filter rows ${filteredRows} ≠ ${expected}`)
    const g = (await df.groupBy('city').agg((c) => ({ n: c.id.count(), pay: c.salary.mean() })).sort('city').collect()).toArray()
    assert(g.length === 4 && g.reduce((s, r) => s + r.n, 0) === n, 'groupBy counts')
    const regions = DataFrame.fromRows([{ city: 'Berlin', region: 'DE' }, { city: 'Rome', region: 'IT' }])
    const j = await df.join(regions, { on: 'city' }).collect()
    assert(j.shape[0] === n / 2, `join rows ${j.shape[0]}`)
    const back = DataFrame.fromCSV(f.toCsv())
    assert(back.shape[0] === filteredRows && back.dtypes.city === 'category', 'CSV round trip')
    return `${filteredRows} rows pass the filter`
  })

  await check('typed exactness: a late 2^31 widens i32 → f64 instead of wrapping', () => {
    const rows = Array.from({ length: 300 }, () => ({ v: 1 }))
    rows.push({ v: 2147483648 })
    const d = DataFrame.fromRows(rows)
    assert(d.dtypes.v === 'f64' && d.toArray().at(-1)!.v === 2147483648, JSON.stringify(d.dtypes))
  })

  await check('advanced: quantile (Minitab) and one-sample t in the browser bundle', () => {
    const q = quantile([1, 2, 3, 4], 0.25)
    assert(q === 1.25, `quantile ${q}`)
    const t = ttest1([5.1, 4.9, 5.3, 5.0, 5.2], { mu: 5 })
    assert(Number.isFinite(t.pValue) && t.pValue > 0 && t.pValue < 1, 'ttest1 p')
  })

  if (gpuReady) {
    await check('WebGPU strict filter returns exactly the CPU rows; report names the gpu kernel', async () => {
      const plan = df.filter((c) => c.age.gt(30).and(c.salary.gt(45_000)))
      const cpu = (await plan.engine('cpu').collect()).getColumn('id').toArray()
      const { frame, report: rep } = await plan.engine('webgpu', { strict: true }).collectWithReport()
      const gpu = frame.getColumn('id').toArray()
      assert(gpu.length === cpu.length, `rows gpu ${gpu.length} cpu ${cpu.length}`)
      for (let i = 0; i < cpu.length; i++) if (gpu[i] !== cpu[i]) throw new Error(`row ${i}: gpu ${gpu[i]} cpu ${cpu[i]}`)
      assert(rep.backendsUsed.includes('webgpu'), formatExecutionReport(rep))
      const ev = rep.events.find((e) => e.backend === 'webgpu')
      return `${cpu.length} rows · ${ev?.kernel ?? '?'} · transfer ${ev?.transferMs?.toFixed(1)} ms compute ${ev?.computeMs?.toFixed(1)} ms`
    })
    await check('WebGPU declines f64 columns (exactness) and reports why', async () => {
      const d = DataFrame.fromColumns({ v: Float64Array.from({ length: 50_000 }, (_, i) => i + 0.5) })
      const { report: rep } = await d.filter(col('v').gt(100)).engine('webgpu').collectWithReport()
      assert(!rep.backendsUsed.includes('webgpu'), 'f64 must not run on the float32 kernel')
      assert(rep.events.some((e) => /f64/.test(e.reason ?? '')), formatExecutionReport(rep))
    })
  } else {
    report({ name: 'WebGPU', ok: true, detail: hasGpu ? 'navigator.gpu present but no adapter (headless without GPU) — GPU checks skipped' : 'not available in this browser — GPU checks skipped' })
  }

  window.__smoke = { done: true, checks, webgpu: gpuReady, userAgent: navigator.userAgent }
}

main().catch((e) => {
  report({ name: 'main', ok: false, detail: e instanceof Error ? e.stack ?? e.message : String(e) })
  window.__smoke = { done: true, checks, webgpu: false, userAgent: navigator.userAgent }
})
