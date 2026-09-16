import {
  getColumn,
  isNumeric,
  isValid,
  tableFromColumns,
  takeColumn,
  type Column,
  type TableView,
} from '@columna/arrow'
import type { Backend, ExprNode, PlanNode } from '@columna/runtime'

export const FILTER_SHADER = /* wgsl */ `
struct Params {
  n: u32,
  op: u32,
  literal: f32,
  mode: u32, // 0 = write, 1 = AND into existing mask
}
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> mask: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let v = values[i];
  var ok = false;
  switch (params.op) {
    case 0u: { ok = v == params.literal; }
    case 1u: { ok = v != params.literal; }
    case 2u: { ok = v > params.literal; }
    case 3u: { ok = v >= params.literal; }
    case 4u: { ok = v < params.literal; }
    case 5u: { ok = v <= params.literal; }
    default: { ok = false; }
  }
  let bit = select(0u, 1u, ok);
  if (params.mode == 0u) {
    mask[i] = bit;
  } else {
    mask[i] = mask[i] & bit;
  }
}
`

export const MAP_SHADER = /* wgsl */ `
struct Params {
  n: u32,
  op: u32, // 0 add, 1 sub, 2 mul, 3 div
  literal: f32,
  _pad: u32,
}
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> outValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let v = values[i];
  var r = v;
  switch (params.op) {
    case 0u: { r = v + params.literal; }
    case 1u: { r = v - params.literal; }
    case 2u: { r = v * params.literal; }
    case 3u: { r = v / params.literal; }
    default: { r = v; }
  }
  outValues[i] = r;
}
`

export const REDUCE_SHADER = /* wgsl */ `
struct Params { n: u32, mode: u32, _p1: u32, _p2: u32, }
// mode: 0 sum, 1 min, 2 max
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> outVals: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let i = gid.x;
  var v: f32;
  if (params.mode == 0u) {
    v = 0.0;
    if (i < params.n) { v = values[i]; }
  } else if (params.mode == 1u) {
    v = 3.402823e+38;
    if (i < params.n) { v = values[i]; }
  } else {
    v = -3.402823e+38;
    if (i < params.n) { v = values[i]; }
  }
  scratch[lid.x] = v;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lid.x < stride) {
      let a = scratch[lid.x];
      let b = scratch[lid.x + stride];
      if (params.mode == 0u) {
        scratch[lid.x] = a + b;
      } else if (params.mode == 1u) {
        scratch[lid.x] = min(a, b);
      } else {
        scratch[lid.x] = max(a, b);
      }
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    outVals[wid.x] = scratch[0];
  }
}
`

const CMP_OPS: Record<string, number> = { eq: 0, neq: 1, gt: 2, gte: 3, lt: 4, lte: 5 }
const ARITH_OPS: Record<string, number> = { add: 0, sub: 1, mul: 2, div: 3 }

export type NumericCmp = { column: string; op: keyof typeof CMP_OPS; literal: number }

/** Flatten `a AND b AND …` into leaf binary nodes; null if non-AND structure. */
export function flattenAnd(expr: ExprNode): ExprNode[] | null {
  if (expr.type === 'binary' && expr.op === 'and') {
    const left = flattenAnd(expr.left)
    const right = flattenAnd(expr.right)
    if (!left || !right) return null
    return [...left, ...right]
  }
  if (expr.type === 'binary') return [expr]
  return null
}

/** Match filter predicates of form col OP lit (AND-combined). */
export function matchNumericAndFilter(predicate: ExprNode): NumericCmp[] | null {
  const parts = flattenAnd(predicate)
  if (!parts || parts.length === 0) return null
  const out: NumericCmp[] = []
  for (const p of parts) {
    if (p.type !== 'binary' || p.left.type !== 'col' || p.right.type !== 'lit') return null
    if (typeof p.right.value !== 'number') return null
    if (!(p.op in CMP_OPS)) return null
    out.push({ column: p.left.name, op: p.op as NumericCmp['op'], literal: p.right.value })
  }
  return out
}

export function matchNumericMap(
  expr: ExprNode,
): { column: string; op: keyof typeof ARITH_OPS; literal: number } | null {
  if (expr.type === 'alias') return matchNumericMap(expr.expr)
  if (expr.type !== 'binary' || !(expr.op in ARITH_OPS)) return null
  if (expr.left.type === 'col' && expr.right.type === 'lit' && typeof expr.right.value === 'number') {
    return { column: expr.left.name, op: expr.op as keyof typeof ARITH_OPS, literal: expr.right.value }
  }
  if (expr.right.type === 'col' && expr.left.type === 'lit' && typeof expr.left.value === 'number' && expr.op === 'add') {
    return { column: expr.right.name, op: 'add', literal: expr.left.value }
  }
  if (expr.right.type === 'col' && expr.left.type === 'lit' && typeof expr.left.value === 'number' && expr.op === 'mul') {
    return { column: expr.right.name, op: 'mul', literal: expr.left.value }
  }
  return null
}

/** True if this plan node (or a filter subtree) can use GPU kernels. */
export function planHasGpuKernel(plan: PlanNode): boolean {
  switch (plan.type) {
    case 'scan':
      return false
    case 'filter':
      return matchNumericAndFilter(plan.predicate) !== null
    case 'withColumn':
      return matchNumericMap(plan.expr) !== null
    case 'project':
    case 'sort':
    case 'limit':
    case 'drop':
    case 'rename':
    case 'groupBy':
    case 'fillNull':
    case 'dropNull':
    case 'melt':
    case 'pivot':
    case 'window':
    case 'rolling':
    case 'slice':
    case 'take':
    case 'unique':
    case 'valueCounts':
    case 'describe':
    case 'corr':
    case 'withColumns':
    case 'expanding':
    case 'sample':
    case 'explode':
    case 'unnest':
    case 'transpose':
    case 'interpolate':
      return planHasGpuKernel(plan.input)
    case 'join':
    case 'asofJoin':
      return planHasGpuKernel(plan.left) || planHasGpuKernel(plan.right)
    case 'concat':
      return plan.frames.some(planHasGpuKernel)
  }
}

export function maskToIndices(mask: Uint32Array): Uint32Array {
  let count = 0
  for (let i = 0; i < mask.length; i++) if (mask[i]) count++
  const out = new Uint32Array(count)
  let j = 0
  for (let i = 0; i < mask.length; i++) if (mask[i]) out[j++] = i
  return out
}

export function columnToF32(col: Column, numRows: number): Float32Array {
  if (col.data instanceof Float32Array) return col.data
  const out = new Float32Array(numRows)
  const data = col.data as Float64Array | Int32Array | Uint32Array | Uint8Array | string[]
  const bitmap = col.nullBitmap
  if (Array.isArray(data)) {
    out.fill(Number.NaN)
    return out
  }
  for (let i = 0; i < numRows; i++) {
    out[i] = bitmap && !isValid(bitmap, i) ? Number.NaN : Number(data[i])
  }
  return out
}

export class GpuBufferManager {
  private cache = new WeakMap<object, GPUBuffer>()

  constructor(private device: GPUDevice) {}

  /** Upload once per TypedArray identity; subsequent calls reuse the GPU buffer. */
  uploadF32(data: Float32Array, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST): GPUBuffer {
    const hit = this.cache.get(data)
    if (hit) return hit
    const buffer = this.device.createBuffer({
      size: Math.max(4, data.byteLength),
      usage,
      mappedAtCreation: true,
    })
    new Float32Array(buffer.getMappedRange()).set(data)
    buffer.unmap()
    this.cache.set(data, buffer)
    return buffer
  }

  createStorage(data: Float32Array | Uint32Array, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: Math.max(4, data.byteLength),
      usage,
      mappedAtCreation: true,
    })
    if (data instanceof Float32Array) new Float32Array(buffer.getMappedRange()).set(data)
    else new Uint32Array(buffer.getMappedRange()).set(data)
    buffer.unmap()
    return buffer
  }

  createEmptyStorage(byteLength: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST): GPUBuffer {
    return this.device.createBuffer({
      size: Math.max(4, byteLength),
      usage,
    })
  }

  createUniform(values: Uint32Array | Float32Array): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: Math.max(16, values.byteLength),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    })
    if (values instanceof Float32Array) new Float32Array(buffer.getMappedRange()).set(values)
    else new Uint32Array(buffer.getMappedRange()).set(values)
    buffer.unmap()
    return buffer
  }

  async readU32(buffer: GPUBuffer, length: number): Promise<Uint32Array> {
    const size = length * 4
    const staging = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    const encoder = this.device.createCommandEncoder()
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, size)
    this.device.queue.submit([encoder.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const data = new Uint32Array(staging.getMappedRange().slice(0))
    staging.unmap()
    staging.destroy()
    return data
  }

  async readF32(buffer: GPUBuffer, length: number): Promise<Float32Array> {
    const size = length * 4
    const staging = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    const encoder = this.device.createCommandEncoder()
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, size)
    this.device.queue.submit([encoder.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const data = new Float32Array(staging.getMappedRange().slice(0))
    staging.unmap()
    staging.destroy()
    return data
  }
}

export async function detectWebGPU(): Promise<GPUDevice | null> {
  const nav = globalThis as typeof globalThis & { navigator?: Navigator }
  if (!nav.navigator?.gpu) return null
  try {
    const adapter = await nav.navigator.gpu.requestAdapter()
    if (!adapter) return null
    return await adapter.requestDevice()
  } catch {
    return null
  }
}

function hasNavigatorGpu(): boolean {
  const nav = globalThis as typeof globalThis & { navigator?: Navigator }
  return Boolean(nav.navigator?.gpu)
}

export class WebGpuBackend implements Backend {
  readonly name = 'webgpu' as const
  readonly capabilities = { name: 'webgpu' as const, gpuFriendlyOnly: false, minRows: 10_000 }
  private device: GPUDevice | null = null
  private settled = false
  private fallback: (plan: PlanNode) => TableView | Promise<TableView>
  private ready: Promise<void>
  private mgr: GpuBufferManager | null = null
  private pipelines = new Map<string, GPUComputePipeline>()

  constructor(fallback: (plan: PlanNode) => TableView | Promise<TableView>, device?: GPUDevice | null) {
    this.fallback = fallback
    if (device !== undefined) {
      this.device = device
      this.settled = true
      if (device) this.mgr = new GpuBufferManager(device)
      this.ready = Promise.resolve()
    } else {
      this.ready = detectWebGPU().then((d) => {
        this.device = d
        this.settled = true
        if (d) this.mgr = new GpuBufferManager(d)
      })
    }
  }

  /** Await adapter/device init. */
  async waitReady(): Promise<boolean> {
    await this.ready
    return this.device !== null
  }

  supports(_plan: PlanNode): boolean {
    if (this.settled && !this.device) return false
    if (!this.settled && !hasNavigatorGpu()) return false
    // Hybrid executor: GPU filter/map subtrees, CPU for the rest.
    return true
  }

  async execute(plan: PlanNode): Promise<TableView> {
    await this.ready
    if (!this.device || !this.mgr) return this.fallback(plan)

    try {
      return await this.executeHybrid(plan)
    } catch {
      return this.fallback(plan)
    }
  }

  /** GPU for filter/map nodes; CPU for the rest, with inputs already materialised. */
  private async executeHybrid(plan: PlanNode): Promise<TableView> {
    if (plan.type === 'scan') return plan.table

    if (plan.type === 'filter') {
      const cmps = matchNumericAndFilter(plan.predicate)
      if (cmps) {
        const input = await this.executeHybrid(plan.input)
        if (this.columnsNumeric(input, cmps.map((c) => c.column))) {
          return this.gpuFilterTable(input, cmps)
        }
      }
      const input = await this.executeHybrid(plan.input)
      return this.fallback({ type: 'filter', input: { type: 'scan', table: input }, predicate: plan.predicate })
    }

    if (plan.type === 'withColumn') {
      const mapped = matchNumericMap(plan.expr)
      if (mapped) {
        const input = await this.executeHybrid(plan.input)
        const col = getColumn(input, mapped.column)
        if (isNumeric(col.field.dtype) || col.field.dtype === 'category') {
          const values = await this.gpuMap(columnToF32(col, input.numRows), ARITH_OPS[mapped.op]!, mapped.literal)
          const newCol: Column = {
            field: { name: plan.name, dtype: 'f32', nullable: col.field.nullable },
            data: values,
            nullBitmap: col.nullBitmap ? new Uint8Array(col.nullBitmap) : undefined,
          }
          const others = input.columns.filter((c) => c.field.name !== plan.name)
          return tableFromColumns([...others, newCol])
        }
      }
      const input = await this.executeHybrid(plan.input)
      return this.fallback({ type: 'withColumn', input: { type: 'scan', table: input }, name: plan.name, expr: plan.expr })
    }

    if (plan.type === 'join') {
      const left = await this.executeHybrid(plan.left)
      const right = await this.executeHybrid(plan.right)
      return this.fallback({
        ...plan,
        left: { type: 'scan', table: left },
        right: { type: 'scan', table: right },
      })
    }

    if (plan.type === 'concat') {
      const frames = await Promise.all(plan.frames.map((f) => this.executeHybrid(f)))
      return this.fallback({
        type: 'concat',
        how: plan.how,
        frames: frames.map((table) => ({ type: 'scan' as const, table })),
      })
    }

    if ('input' in plan) {
      const input = await this.executeHybrid(plan.input)
      return this.fallback({ ...(plan as PlanNode & { input: PlanNode }), input: { type: 'scan', table: input } } as PlanNode)
    }

    return this.fallback(plan)
  }

  private columnsNumeric(table: TableView, names: string[]): boolean {
    for (const name of names) {
      const col = getColumn(table, name)
      if (!isNumeric(col.field.dtype) && col.field.dtype !== 'category' && col.field.dtype !== 'datetime') {
        return false
      }
    }
    return true
  }

  private async gpuFilterTable(table: TableView, cmps: NumericCmp[]): Promise<TableView> {
    const n = table.numRows
    if (n === 0 || cmps.length === 0) return table
    const mask = await this.gpuFilterAnd(
      cmps.map((c) => ({
        values: columnToF32(getColumn(table, c.column), n),
        op: CMP_OPS[c.op]!,
        literal: c.literal,
      })),
    )
    const indices = maskToIndices(mask)
    return tableFromColumns(table.columns.map((c) => takeColumn(c, indices)))
  }

  private getPipeline(key: string, code: string): GPUComputePipeline {
    const hit = this.pipelines.get(key)
    if (hit) return hit
    const device = this.device!
    const module = device.createShaderModule({ code })
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    })
    this.pipelines.set(key, pipeline)
    return pipeline
  }

  /** Multi-predicate AND filter; columns uploaded with residency cache when possible. */
  async gpuFilterAnd(
    predicates: Array<{ values: Float32Array; op: number; literal: number }>,
  ): Promise<Uint32Array> {
    const device = this.device!
    const mgr = this.mgr!
    const n = predicates[0]!.values.length
    const maskBuf = mgr.createEmptyStorage(n * 4)
    const pipeline = this.getPipeline('filter', FILTER_SHADER)

    const encoder = device.createCommandEncoder()
    for (let p = 0; p < predicates.length; p++) {
      const pred = predicates[p]!
      const valueBuf = mgr.uploadF32(pred.values)
      const params = new ArrayBuffer(16)
      const u32 = new Uint32Array(params)
      const f32 = new Float32Array(params)
      u32[0] = n
      u32[1] = pred.op
      f32[2] = pred.literal
      u32[3] = p === 0 ? 0 : 1
      const paramBuf = mgr.createUniform(new Uint32Array(params))
      const pass = encoder.beginComputePass()
      pass.setPipeline(pipeline)
      pass.setBindGroup(
        0,
        device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: valueBuf } },
            { binding: 1, resource: { buffer: maskBuf } },
            { binding: 2, resource: { buffer: paramBuf } },
          ],
        }),
      )
      pass.dispatchWorkgroups(Math.ceil(n / 256) || 1)
      pass.end()
    }
    device.queue.submit([encoder.finish()])
    return mgr.readU32(maskBuf, n)
  }

  async gpuMap(values: Float32Array, op: number, literal: number): Promise<Float32Array> {
    const device = this.device!
    const mgr = this.mgr!
    const n = values.length
    const valueBuf = mgr.uploadF32(values)
    const outBuf = mgr.createEmptyStorage(n * 4)
    const params = new ArrayBuffer(16)
    const u32 = new Uint32Array(params)
    const f32 = new Float32Array(params)
    u32[0] = n
    u32[1] = op
    f32[2] = literal
    const paramBuf = mgr.createUniform(new Uint32Array(params))
    const pipeline = this.getPipeline('map', MAP_SHADER)
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(
      0,
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: valueBuf } },
          { binding: 1, resource: { buffer: outBuf } },
          { binding: 2, resource: { buffer: paramBuf } },
        ],
      }),
    )
    pass.dispatchWorkgroups(Math.ceil(n / 256) || 1)
    pass.end()
    device.queue.submit([encoder.finish()])
    return mgr.readF32(outBuf, n)
  }

  async gpuReduce(values: Float32Array, mode: 'sum' | 'min' | 'max' = 'sum'): Promise<number> {
    await this.ready
    if (!this.device || !this.mgr) {
      if (mode === 'sum') return values.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0)
      if (mode === 'min') return values.reduce((a, b) => Math.min(a, b), Infinity)
      return values.reduce((a, b) => Math.max(a, b), -Infinity)
    }
    const device = this.device
    const mgr = this.mgr
    const n = values.length
    const workgroups = Math.ceil(n / 256) || 1
    const valueBuf = mgr.uploadF32(values)
    const outBuf = mgr.createEmptyStorage(workgroups * 4)
    const modeId = mode === 'sum' ? 0 : mode === 'min' ? 1 : 2
    const params = new Uint32Array([n, modeId, 0, 0])
    const paramBuf = mgr.createUniform(params)
    const pipeline = this.getPipeline('reduce', REDUCE_SHADER)
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(
      0,
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: valueBuf } },
          { binding: 1, resource: { buffer: outBuf } },
          { binding: 2, resource: { buffer: paramBuf } },
        ],
      }),
    )
    pass.dispatchWorkgroups(workgroups)
    pass.end()
    device.queue.submit([encoder.finish()])
    const partial = await mgr.readF32(outBuf, workgroups)
    if (mode === 'sum') return partial.reduce((a, b) => a + b, 0)
    if (mode === 'min') return partial.reduce((a, b) => Math.min(a, b), Infinity)
    return partial.reduce((a, b) => Math.max(a, b), -Infinity)
  }

  /** @deprecated use gpuReduce(..., 'sum') */
  async gpuReduceSum(values: Float32Array): Promise<number> {
    return this.gpuReduce(values, 'sum')
  }
}
