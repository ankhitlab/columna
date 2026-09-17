import {
  getColumn,
  isNumeric,
  isValid,
  tableFromColumns,
  takeColumn,
  type Column,
  type TableView,
} from '@columna/arrow'
import type { Backend, ExecContext, ExprNode, PlanNode } from '@columna/runtime'

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now())

/**
 * Filter kernel. Values are bound as raw 32-bit words and compared in their own type (f32 / i32 / u32 via
 * bitcast) so integer columns keep every bit — no float32 rounding of 16 777 217 into 16 777 216. Null rows
 * come from the validity bitmap (bit i of valid[i >> 5]) and never match, exactly like the CPU path; f32 NaN
 * is detected by bit pattern (shader float comparisons need not honour IEEE NaN) and matches only "!=".
 */
export const FILTER_SHADER = /* wgsl */ `
struct Params {
  n: u32,
  op: u32,       // 0 eq, 1 neq, 2 gt, 3 gte, 4 lt, 5 lte
  literal: u32,  // raw bits of the literal in the column's kind
  mode: u32,     // 0 = write, 1 = AND into existing mask
  kind: u32,     // 0 = f32, 1 = i32, 2 = u32
  hasValid: u32, // 1 = validity bitmap bound
  _p0: u32,
  _p1: u32,
}
@group(0) @binding(0) var<storage, read> values: array<u32>;
@group(0) @binding(1) var<storage, read_write> mask: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> valid: array<u32>;

fn cmp_f32(v: f32, l: f32, op: u32) -> bool {
  switch (op) {
    case 0u: { return v == l; }
    case 1u: { return v != l; }
    case 2u: { return v > l; }
    case 3u: { return v >= l; }
    case 4u: { return v < l; }
    case 5u: { return v <= l; }
    default: { return false; }
  }
}
fn cmp_i32(v: i32, l: i32, op: u32) -> bool {
  switch (op) {
    case 0u: { return v == l; }
    case 1u: { return v != l; }
    case 2u: { return v > l; }
    case 3u: { return v >= l; }
    case 4u: { return v < l; }
    case 5u: { return v <= l; }
    default: { return false; }
  }
}
fn cmp_u32(v: u32, l: u32, op: u32) -> bool {
  switch (op) {
    case 0u: { return v == l; }
    case 1u: { return v != l; }
    case 2u: { return v > l; }
    case 3u: { return v >= l; }
    case 4u: { return v < l; }
    case 5u: { return v <= l; }
    default: { return false; }
  }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  var ok = false;
  let isValid = params.hasValid == 0u || ((valid[i >> 5u] >> (i & 31u)) & 1u) == 1u;
  if (isValid) {
    let raw = values[i];
    if (params.kind == 0u) {
      if ((raw & 0x7fffffffu) > 0x7f800000u) {
        ok = params.op == 1u; // NaN: only "!=" holds
      } else {
        ok = cmp_f32(bitcast<f32>(raw), bitcast<f32>(params.literal), params.op);
      }
    } else if (params.kind == 1u) {
      ok = cmp_i32(bitcast<i32>(raw), bitcast<i32>(params.literal), params.op);
    } else {
      ok = cmp_u32(raw, params.literal, params.op);
    }
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
    case 'ffill':
    case 'bfill':
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

/** Column kind on the GPU: how the raw 32-bit words of a filter column are interpreted. */
export type GpuKind = 0 | 1 | 2 // f32 | i32 | u32

export type GpuFilterPredicate = {
  /** Raw 32-bit words (zero-copy view of the column buffer where the dtype already is 32-bit). */
  words: Uint32Array
  /** Identity used for GPU residency caching (the column's own typed array when zero-copy). */
  cacheKey: object
  kind: GpuKind
  op: number
  /** Literal as raw bits of `kind`. */
  literalBits: number
  /** Validity bitmap repacked into u32 words, or null when the column has no nulls. */
  valid: Uint32Array | null
  validKey: object | null
}

const F32_SCRATCH = new Float32Array(1)
const U32_SCRATCH = new Uint32Array(F32_SCRATCH.buffer)
const I32_SCRATCH = new Int32Array(1)
const I32_U32_SCRATCH = new Uint32Array(I32_SCRATCH.buffer)

export function f32Bits(v: number): number {
  F32_SCRATCH[0] = v
  return U32_SCRATCH[0]!
}
export function i32Bits(v: number): number {
  I32_SCRATCH[0] = v
  return I32_U32_SCRATCH[0]!
}

/** Validity bitmap (bit i of byte i >> 3) as u32 words: byte k lands in word k >> 2 — same bit indexing. */
export function bitmapToWords(bitmap: Uint8Array, numRows: number): Uint32Array {
  const words = new Uint32Array(Math.max(1, Math.ceil(numRows / 32)))
  new Uint8Array(words.buffer).set(bitmap.subarray(0, Math.min(bitmap.length, words.byteLength)))
  return words
}

/**
 * Build the typed GPU inputs for `col OP literal`, or null when the GPU cannot reproduce the CPU result
 * exactly: f64 / datetime columns (float32 would merge 16 777 216 and 16 777 217, or neighbouring epoch
 * milliseconds), bool / category columns (the CPU compares them as booleans / strings, not numbers) and literals not representable in the
 * column's type (`x > 2.5` on i32, `x > 0.1` on f32). With `lossy` those cases run in float32 anyway.
 */
export function prepareGpuFilterPredicate(
  col: Column,
  numRows: number,
  op: number,
  literal: number,
  lossy = false,
): GpuFilterPredicate | null {
  const dtype = col.field.dtype
  const data = col.data
  let words: Uint32Array
  let cacheKey: object
  let kind: GpuKind
  let literalBits: number
  if (data instanceof Float32Array && dtype === 'f32') {
    if (Math.fround(literal) !== literal && !lossy) return null
    words = new Uint32Array(data.buffer, data.byteOffset, data.length)
    cacheKey = data
    kind = 0
    literalBits = f32Bits(literal)
  } else if (data instanceof Int32Array && dtype === 'i32') {
    if (!Number.isInteger(literal) || literal < -2147483648 || literal > 2147483647) {
      if (!lossy) return null
      return lossyF32Predicate(col, numRows, op, literal)
    }
    words = new Uint32Array(data.buffer, data.byteOffset, data.length)
    cacheKey = data
    kind = 1
    literalBits = i32Bits(literal)
  } else if (data instanceof Uint32Array && dtype === 'u32') {
    if (!Number.isInteger(literal) || literal < 0 || literal > 4294967295) {
      if (!lossy) return null
      return lossyF32Predicate(col, numRows, op, literal)
    }
    words = data
    cacheKey = data
    kind = 2
    literalBits = literal >>> 0
  } else if (lossy && (dtype === 'f64' || dtype === 'datetime' || isNumeric(dtype))) {
    return lossyF32Predicate(col, numRows, op, literal)
  } else {
    return null
  }
  const bitmap = col.nullBitmap
  return {
    words,
    cacheKey,
    kind,
    op,
    literalBits,
    valid: bitmap ? bitmapToWords(bitmap, numRows) : null,
    validKey: bitmap ?? null,
  }
}

/** Opt-in approximate path: values and literal rounded to float32 (the pre-typed behaviour). */
function lossyF32Predicate(col: Column, numRows: number, op: number, literal: number): GpuFilterPredicate {
  const f32 = columnToF32(col, numRows)
  const bitmap = col.nullBitmap
  return {
    words: new Uint32Array(f32.buffer, f32.byteOffset, f32.length),
    cacheKey: f32 === col.data ? col.data : f32,
    kind: 0,
    op,
    literalBits: f32Bits(Math.fround(literal)),
    valid: bitmap ? bitmapToWords(bitmap, numRows) : null,
    validKey: bitmap ?? null,
  }
}

/**
 * Convert a numeric column to float32 for the map / reduce kernels. Null rows become NaN — including for
 * columns already stored as Float32Array, which used to be returned as-is and let a null read as its
 * physical 0. Lossy for f64 / datetime / integers beyond 2^24 by construction; callers gate on dtype.
 */
export function columnToF32(col: Column, numRows: number): Float32Array {
  if (col.data instanceof Float32Array && !col.nullBitmap) return col.data
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

  /** Upload raw 32-bit words; `key` (default: the array itself) identifies the buffer for residency reuse. */
  uploadWords(data: Uint32Array, key: object = data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST): GPUBuffer {
    const hit = this.cache.get(key)
    if (hit) return hit
    const buffer = this.device.createBuffer({
      size: Math.max(4, data.byteLength),
      usage,
      mappedAtCreation: true,
    })
    new Uint32Array(buffer.getMappedRange()).set(data)
    buffer.unmap()
    this.cache.set(key, buffer)
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

/** Whether this environment exposes WebGPU at all (no adapter request is made). */
export function hasNavigatorGpu(): boolean {
  const nav = globalThis as typeof globalThis & { navigator?: Navigator }
  return Boolean(nav.navigator?.gpu)
}

export class WebGpuBackend implements Backend {
  readonly name = 'webgpu' as const
  readonly capabilities = { name: 'webgpu' as const, gpuFriendlyOnly: false, minRows: 10_000 }
  private device: GPUDevice | null = null
  private settled = false
  private fallback: (plan: PlanNode) => TableView | Promise<TableView>
  /** Detection promise; created on the first `waitReady()` / `execute()`, never at construction (no import-time side effects). */
  private ready: Promise<void> | null = null
  private mgr: GpuBufferManager | null = null
  private pipelines = new Map<string, GPUComputePipeline>()
  /**
   * `lossyF32: true` re-enables the approximate float32 path for f64 / datetime columns and for literals
   * the column type cannot represent exactly. Off by default: the GPU must return the same rows as the CPU.
   */
  readonly options: { lossyF32: boolean }

  constructor(
    fallback: (plan: PlanNode) => TableView | Promise<TableView>,
    device?: GPUDevice | null,
    options: { lossyF32?: boolean } = {},
  ) {
    this.fallback = fallback
    this.options = { lossyF32: options.lossyF32 ?? false }
    if (device !== undefined) {
      this.device = device
      this.settled = true
      if (device) this.mgr = new GpuBufferManager(device)
      this.ready = Promise.resolve()
    }
  }

  /**
   * Request the adapter / device (once). Nothing talks to `navigator.gpu` before this is called — importing
   * the library or constructing the backend has no side effects. `init()` in `columna` calls it.
   */
  async waitReady(): Promise<boolean> {
    if (!this.ready) {
      this.ready = detectWebGPU().then((d) => {
        this.device = d
        this.settled = true
        if (d) this.mgr = new GpuBufferManager(d)
      })
    }
    await this.ready
    return this.device !== null
  }

  /** True once detection has run (successfully or not). */
  get initialised(): boolean {
    return this.settled
  }

  supports(_plan: PlanNode): boolean {
    // Not initialised → not a candidate: the runtime never starts GPU detection implicitly.
    if (!this.settled || !this.device) return false
    // Hybrid executor: GPU filter/map subtrees, CPU for the rest.
    return true
  }

  async execute(plan: PlanNode, ctx?: ExecContext): Promise<TableView> {
    await this.waitReady()
    if (!this.device || !this.mgr) {
      if (ctx?.strict) throw new Error('no WebGPU device (navigator.gpu missing or adapter request failed)')
      const out = await this.fallback(plan)
      ctx?.trace({ node: plan.type, backend: 'cpu', reason: 'no WebGPU device (navigator.gpu missing or adapter request failed)', rows: out.numRows })
      return out
    }

    try {
      return await this.executeHybrid(plan, ctx)
    } catch (err) {
      if (ctx?.strict) throw err
      const out = await this.fallback(plan)
      ctx?.trace({ node: plan.type, backend: 'cpu', reason: `GPU execution threw: ${err instanceof Error ? err.message : String(err)}`, rows: out.numRows })
      return out
    }
  }

  /** Delegate one node to the CPU engine and say why in the report. */
  private async cpuNode(plan: PlanNode, reason: string, ctx?: ExecContext): Promise<TableView> {
    const t0 = now()
    const out = await this.fallback(plan)
    ctx?.trace({ node: plan.type, backend: 'cpu', reason, ms: now() - t0, rows: out.numRows })
    return out
  }

  /** GPU for filter/map nodes; CPU for the rest, with inputs already materialised. */
  private async executeHybrid(plan: PlanNode, ctx?: ExecContext): Promise<TableView> {
    if (plan.type === 'scan') return plan.table

    if (plan.type === 'filter') {
      const cmps = matchNumericAndFilter(plan.predicate)
      if (cmps) {
        const input = await this.executeHybrid(plan.input, ctx)
        const prepared = this.prepareFilter(input, cmps)
        if (prepared.preds) return this.gpuFilterTable(input, prepared.preds, ctx)
        return this.cpuNode({ type: 'filter', input: { type: 'scan', table: input }, predicate: plan.predicate }, prepared.reason, ctx)
      }
      const input = await this.executeHybrid(plan.input, ctx)
      return this.cpuNode({ type: 'filter', input: { type: 'scan', table: input }, predicate: plan.predicate }, 'predicate is not an AND of "col OP number" comparisons', ctx)
    }

    if (plan.type === 'withColumn') {
      const mapped = matchNumericMap(plan.expr)
      if (mapped) {
        const input = await this.executeHybrid(plan.input, ctx)
        const col = getColumn(input, mapped.column)
        // GPU map kernel is float32-only. CPU arith always materializes f64, so even an f32 input
        // would diverge in value and dtype unless the caller opts into lossyF32.
        if (this.options.lossyF32 && isNumeric(col.field.dtype)) {
          const t0 = now()
          const f32 = columnToF32(col, input.numRows)
          const tConv = now()
          const { values, transferMs, computeMs } = await this.gpuMapTimed(f32, ARITH_OPS[mapped.op]!, mapped.literal)
          const newCol: Column = {
            field: { name: plan.name, dtype: 'f32', nullable: col.field.nullable },
            data: values,
            nullBitmap: col.nullBitmap ? new Uint8Array(col.nullBitmap) : undefined,
          }
          const others = input.columns.filter((c) => c.field.name !== plan.name)
          const out = tableFromColumns([...others, newCol])
          ctx?.trace({ node: 'withColumn', backend: 'webgpu', kernel: 'gpu:map', ms: now() - t0, transferMs: transferMs + (tConv - t0), computeMs, rows: out.numRows })
          return out
        }
        const input2 = input
        return this.cpuNode(
          { type: 'withColumn', input: { type: 'scan', table: input2 }, name: plan.name, expr: plan.expr },
          `column "${mapped.column}" is ${col.field.dtype}: GPU map is float32-lossy (init({ gpuLossyF32: true }) to allow)`,
          ctx,
        )
      }
      const input = await this.executeHybrid(plan.input, ctx)
      return this.cpuNode({ type: 'withColumn', input: { type: 'scan', table: input }, name: plan.name, expr: plan.expr }, 'expression is not "col ARITH number"', ctx)
    }

    if (plan.type === 'join') {
      const left = await this.executeHybrid(plan.left, ctx)
      const right = await this.executeHybrid(plan.right, ctx)
      return this.cpuNode({ ...plan, left: { type: 'scan', table: left }, right: { type: 'scan', table: right } }, 'no GPU kernel for this node type', ctx)
    }

    if (plan.type === 'concat') {
      const frames = await Promise.all(plan.frames.map((f) => this.executeHybrid(f, ctx)))
      return this.cpuNode({ type: 'concat', how: plan.how, frames: frames.map((table) => ({ type: 'scan' as const, table })) }, 'no GPU kernel for this node type', ctx)
    }

    if ('input' in plan) {
      const input = await this.executeHybrid(plan.input, ctx)
      return this.cpuNode({ ...(plan as PlanNode & { input: PlanNode }), input: { type: 'scan', table: input } } as PlanNode, 'no GPU kernel for this node type', ctx)
    }

    return this.cpuNode(plan, 'no GPU kernel for this node type', ctx)
  }

  /** Typed predicates for every comparison, or the reason the node must run on the CPU. */
  private prepareFilter(table: TableView, cmps: NumericCmp[]): { preds: GpuFilterPredicate[] | null; reason: string } {
    const out: GpuFilterPredicate[] = []
    for (const c of cmps) {
      const col = getColumn(table, c.column)
      const p = prepareGpuFilterPredicate(col, table.numRows, CMP_OPS[c.op]!, c.literal, this.options.lossyF32)
      if (!p) {
        const dt = col.field.dtype
        const reason =
          dt === 'f64' || dt === 'datetime'
            ? `column "${c.column}" is ${dt}: float32 would merge neighbouring values (init({ gpuLossyF32: true }) to allow)`
            : dt === 'bool' || dt === 'category'
              ? `column "${c.column}" is ${dt}: compared as ${dt === 'bool' ? 'booleans' : 'strings'} on the CPU, not numbers`
              : `literal ${c.literal} is not exactly representable in ${dt} (init({ gpuLossyF32: true }) to allow)`
        return { preds: null, reason }
      }
      out.push(p)
    }
    return { preds: out, reason: '' }
  }

  /**
   * GPU filter as a whole query, not just the shader: upload (transfer), compute, mask readback (transfer),
   * then mask → indices and the row gather on the CPU. All four are timed and reported.
   */
  private async gpuFilterTable(table: TableView, preds: GpuFilterPredicate[], ctx?: ExecContext): Promise<TableView> {
    const n = table.numRows
    if (n === 0 || preds.length === 0) return table
    const t0 = now()
    const { mask, transferMs, computeMs } = await this.gpuFilterAndTimed(preds)
    const tGather = now()
    const indices = maskToIndices(mask)
    const out = tableFromColumns(table.columns.map((c) => takeColumn(c, indices)))
    ctx?.trace({
      node: 'filter',
      backend: 'webgpu',
      kernel: `gpu:filter(${preds.length} predicate${preds.length > 1 ? 's' : ''}) + cpu:gather ${(now() - tGather).toFixed(1)} ms`,
      ms: now() - t0,
      transferMs,
      computeMs,
      rows: out.numRows,
    })
    return out
  }

  private readonly noValidKey = {}

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

  /** Multi-predicate AND filter over typed predicates (see `prepareGpuFilterPredicate`); columns stay resident. */
  async gpuFilterAnd(predicates: GpuFilterPredicate[]): Promise<Uint32Array> {
    return (await this.gpuFilterAndTimed(predicates)).mask
  }

  /** Same as `gpuFilterAnd` with host↔device transfer and submit→readback timings split out. */
  async gpuFilterAndTimed(predicates: GpuFilterPredicate[]): Promise<{ mask: Uint32Array; transferMs: number; computeMs: number }> {
    const device = this.device!
    const mgr = this.mgr!
    const n = predicates[0]!.words.length
    const tUp = now()
    const maskBuf = mgr.createEmptyStorage(n * 4)
    const pipeline = this.getPipeline('filter', FILTER_SHADER)
    const noValid = mgr.uploadWords(new Uint32Array(1), this.noValidKey)

    const encoder = device.createCommandEncoder()
    for (let p = 0; p < predicates.length; p++) {
      const pred = predicates[p]!
      const valueBuf = mgr.uploadWords(pred.words, pred.cacheKey)
      const validBuf = pred.valid ? mgr.uploadWords(pred.valid, pred.validKey ?? pred.valid) : noValid
      const params = new Uint32Array([n, pred.op, pred.literalBits, p === 0 ? 0 : 1, pred.kind, pred.valid ? 1 : 0, 0, 0])
      const paramBuf = mgr.createUniform(params)
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
            { binding: 3, resource: { buffer: validBuf } },
          ],
        }),
      )
      pass.dispatchWorkgroups(Math.ceil(n / 256) || 1)
      pass.end()
    }
    const tSubmit = now()
    device.queue.submit([encoder.finish()])
    await device.queue.onSubmittedWorkDone()
    const tDone = now()
    const mask = await mgr.readU32(maskBuf, n)
    return { mask, transferMs: tSubmit - tUp + (now() - tDone), computeMs: tDone - tSubmit }
  }

  async gpuMap(values: Float32Array, op: number, literal: number): Promise<Float32Array> {
    return (await this.gpuMapTimed(values, op, literal)).values
  }

  async gpuMapTimed(values: Float32Array, op: number, literal: number): Promise<{ values: Float32Array; transferMs: number; computeMs: number }> {
    const device = this.device!
    const mgr = this.mgr!
    const n = values.length
    const tUp = now()
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
    const tSubmit = now()
    device.queue.submit([encoder.finish()])
    await device.queue.onSubmittedWorkDone()
    const tDone = now()
    const out = await mgr.readF32(outBuf, n)
    return { values: out, transferMs: tSubmit - tUp + (now() - tDone), computeMs: tDone - tSubmit }
  }

  async gpuReduce(values: Float32Array, mode: 'sum' | 'min' | 'max' = 'sum'): Promise<number> {
    await this.waitReady()
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
