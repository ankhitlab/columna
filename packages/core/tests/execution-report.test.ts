import { describe, expect, it } from 'vitest'
import { DataFrame, col, EngineStrictError, formatExecutionReport } from 'columna'
import { Runtime, executeCpu } from '@columna/runtime'
import { WasmBackend, isRustKernelsLoaded } from '@columna/wasm'
import { WebGpuBackend } from '@columna/webgpu'

const rows = Array.from({ length: 2000 }, (_, i) => ({ age: i % 80, salary: (i * 37) % 100_000, city: i % 3 ? 'Berlin' : 'Paris' }))

function runtimeWith(...backends: Array<WasmBackend | WebGpuBackend>): Runtime {
  const rt = new Runtime({ wasmMinRows: 1, webgpuMinRows: 1 })
  for (const b of backends) rt.register(b)
  return rt
}

describe('execution report and strict engine mode', () => {
  it('reports the backend that actually ran each node, not the one that was requested', async () => {
    const rt = runtimeWith(new WasmBackend(executeCpu))
    const df = DataFrame.fromRows(rows)
    const lf = df.lazy().filter(col('age').gt(30).and(col('salary').gt(45_000)))
    const plan = new (lf.constructor as new (p: unknown, r: Runtime) => typeof lf)(lf.plan, rt.withEngine('wasm'))
    const { frame, report } = await plan.collectWithReport()
    expect(frame.shape[0]).toBeGreaterThan(0)
    expect(report.requested).toBe('wasm')
    expect(report.dispatched).toBe('wasm')
    // no Rust kernels in this test process (pkg not built) → every node ran on the CPU and the report says why
    if (!isRustKernelsLoaded()) {
      expect(report.backendsUsed).toEqual(['cpu'])
      expect(report.events[0]!.reason).toMatch(/Rust kernels not loaded/)
    }
    const text = formatExecutionReport(report)
    expect(text).toMatch(/^requested: wasm · dispatched: wasm · used: cpu/)
    expect(text).toMatch(/filter: cpu .* ms rows=.* Rust kernels not loaded/)
    // explain() no longer pretends to know what will run
    expect(plan.explain()).toMatch(/planned/)
  })

  it('strict wasm rejects when Rust did no work; non-strict silently succeeds', async () => {
    const rt = runtimeWith(new WasmBackend(executeCpu))
    const df = DataFrame.fromRows(rows)
    const base = df.lazy().filter(col('age').gt(30))
    const mk = (strict: boolean) => new (base.constructor as new (p: unknown, r: Runtime) => typeof base)(base.plan, rt.withEngine('wasm', { strict }))
    await expect(mk(false).collect()).resolves.toBeInstanceOf(DataFrame)
    if (!isRustKernelsLoaded()) {
      const err = await mk(true).collect().catch((e: unknown) => e)
      expect(err).toBeInstanceOf(EngineStrictError)
      expect((err as EngineStrictError).engine).toBe('wasm')
      expect((err as EngineStrictError).reasons.join()).toMatch(/Rust kernels not loaded/)
      expect(String(err)).toMatch(/engine "wasm" \(strict\)/)
    }
  })

  it('webgpu without a device: fallback is recorded with its reason; strict rejects instead', async () => {
    const gpu = new WebGpuBackend(executeCpu, null)
    const rt = runtimeWith(gpu)
    const df = DataFrame.fromRows(rows)
    const base = df.lazy().filter(col('age').gt(30))
    const mk = (strict: boolean) => new (base.constructor as new (p: unknown, r: Runtime) => typeof base)(base.plan, rt.withEngine('webgpu', { strict }))
    // supports() is false without a device → the runtime dispatches to CPU (non-strict) …
    const { report } = await mk(false).collectWithReport()
    expect(report.requested).toBe('webgpu')
    expect(report.dispatched).toBe('cpu')
    expect(report.backendsUsed).toEqual(['cpu'])
    // … and refuses in strict mode
    await expect(mk(true).collect()).rejects.toThrow(EngineStrictError)
    await expect(mk(true).collect()).rejects.toThrow(/does not support this plan/)
  })

  it('cpu engine names its kernel path and never trips strict mode', async () => {
    const df = DataFrame.fromRows(rows)
    const { report } = await df.lazy().filter(col('age').gt(30).and(col('salary').gt(45_000))).engine('cpu', { strict: true }).collectWithReport()
    expect(report.backendsUsed).toEqual(['cpu'])
    expect(report.events[0]!.kernel).toMatch(/dualFilter/)
    expect(report.events[0]!.ms).toBeGreaterThanOrEqual(0)
    expect(report.totalMs).toBeGreaterThanOrEqual(report.events[0]!.ms!)
    expect(report.fallbacks).toEqual([])
  })
})
