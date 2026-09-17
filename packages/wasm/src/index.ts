/**
 * Portable WASM-oriented backend.
 *
 * Hybrid executor: uses the same fast CPU plan kernels (filter/groupby/join fusion),
 * and optionally swaps in Rust wasm-pack kernels when `../pkg` is built.
 */
import {
  allocateData,
  getColumn,
  getValue,
  isValid,
  setValid,
  setValue,
  tableFromColumns,
  takeColumn,
  type TableView,
} from '@columna/arrow'
import type { Backend, ExecContext, PlanNode } from '@columna/runtime'

export interface WasmKernels {
  filterMask(values: Float64Array, op: number, literal: number, nullBitmap?: Uint8Array): Uint8Array
  filterMaskAnd2?(
    a: Float64Array,
    b: Float64Array,
    opA: number,
    litA: number,
    opB: number,
    litB: number,
  ): Uint32Array
  filterAnd2I32F64?(
    a: Int32Array,
    b: Float64Array,
    opA: number,
    litA: number,
    opB: number,
    litB: number,
  ): Uint32Array
  filterAnd2I32I32?(
    a: Int32Array,
    b: Int32Array,
    opA: number,
    litA: number,
    opB: number,
    litB: number,
  ): Uint32Array
  compactIndices(mask: Uint8Array): Uint32Array
  sortIndices(values: Float64Array, descending: boolean, nullBitmap?: Uint8Array): Uint32Array
  hashGroupBy(
    keyHashes: Uint32Array,
    values: Float64Array,
    aggOp: number,
    nullBitmap?: Uint8Array,
  ): { keys: Uint32Array; values: Float64Array }
}

function cmpOp(op: number, v: number, lit: number): boolean {
  switch (op) {
    case 0:
      return v === lit
    case 1:
      return v !== lit
    case 2:
      return v > lit
    case 3:
      return v >= lit
    case 4:
      return v < lit
    case 5:
      return v <= lit
    default:
      return false
  }
}

/** Pure-TS stand-in for the Rust kernels (same ABI). */
export const tsKernels: WasmKernels = {
  filterMask(values, op, literal, nullBitmap) {
    const mask = new Uint8Array(values.length)
    for (let i = 0; i < values.length; i++) {
      if (nullBitmap && !isValid(nullBitmap, i)) continue
      if (cmpOp(op, values[i]!, literal)) mask[i] = 1
    }
    return mask
  },
  filterMaskAnd2(a, b, opA, litA, opB, litB) {
    const n = a.length
    let count = 0
    for (let i = 0; i < n; i++) if (cmpOp(opA, a[i]!, litA) && cmpOp(opB, b[i]!, litB)) count++
    const out = new Uint32Array(count)
    let j = 0
    for (let i = 0; i < n; i++) if (cmpOp(opA, a[i]!, litA) && cmpOp(opB, b[i]!, litB)) out[j++] = i
    return out
  },
  compactIndices(mask) {
    let count = 0
    for (let i = 0; i < mask.length; i++) if (mask[i]) count++
    const out = new Uint32Array(count)
    let j = 0
    for (let i = 0; i < mask.length; i++) if (mask[i]) out[j++] = i
    return out
  },
  sortIndices(values, descending, nullBitmap) {
    const idx = new Uint32Array(values.length)
    for (let i = 0; i < values.length; i++) idx[i] = i
    idx.sort((a, b) => {
      const aNull = nullBitmap ? !isValid(nullBitmap, a) : false
      const bNull = nullBitmap ? !isValid(nullBitmap, b) : false
      if (aNull && bNull) return 0
      if (aNull) return 1
      if (bNull) return -1
      const cmp = values[a]! < values[b]! ? -1 : values[a]! > values[b]! ? 1 : 0
      return descending ? -cmp : cmp
    })
    return idx
  },
  hashGroupBy(keyHashes, values, aggOp, nullBitmap) {
    const map = new Map<number, { count: number; sum: number; min: number; max: number; first: number; last: number }>()
    for (let i = 0; i < keyHashes.length; i++) {
      if (nullBitmap && !isValid(nullBitmap, i)) continue
      const k = keyHashes[i]!
      const v = values[i]!
      let g = map.get(k)
      if (!g) {
        g = { count: 0, sum: 0, min: v, max: v, first: v, last: v }
        map.set(k, g)
      }
      g.count++
      g.sum += v
      g.min = Math.min(g.min, v)
      g.max = Math.max(g.max, v)
      g.last = v
    }
    const keys = Uint32Array.from(map.keys())
    const out = new Float64Array(keys.length)
    for (let i = 0; i < keys.length; i++) {
      const g = map.get(keys[i]!)!
      switch (aggOp) {
        case 0:
          out[i] = g.sum
          break
        case 1:
          out[i] = g.sum / g.count
          break
        case 2:
          out[i] = g.min
          break
        case 3:
          out[i] = g.max
          break
        case 4:
          out[i] = g.count
          break
        default:
          out[i] = g.first
      }
    }
    return { keys, values: out }
  },
}

let activeKernels: WasmKernels = tsKernels
let rustLoaded = false

export function setWasmKernels(kernels: WasmKernels): void {
  activeKernels = kernels
}

export function getWasmKernels(): WasmKernels {
  return activeKernels
}

export function isRustKernelsLoaded(): boolean {
  return rustLoaded
}

function asF64(col: ReturnType<typeof getColumn>, numRows: number): Float64Array {
  if (col.data instanceof Float64Array) return col.data
  const out = new Float64Array(numRows)
  for (let i = 0; i < numRows; i++) out[i] = Number(getValue(col.data, i))
  return out
}

void asF64 // kept for potential single-col rust paths


/** Rows below which the JS kernels beat the Rust dual filter (copy-in cost is not amortised). */
export const WASM_RUST_FILTER_MIN_ROWS = 5_000_000

/**
 * Hybrid: Rust typed dual-filter when available; else CPU planner (fusion / dense join).
 * Every decision is traced into `ctx` so a report shows whether Rust did anything at all.
 */
function executeWithKernels(plan: PlanNode, fallback: (p: PlanNode) => TableView, ctx?: ExecContext): TableView {
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now()
  const delegate = (reason: string): TableView => {
    const out = fallback(plan)
    ctx?.trace({ node: plan.type, backend: 'cpu', reason, ms: (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0, rows: out.numRows })
    return out
  }
  if (plan.type !== 'filter') return delegate('wasm has kernels for filter nodes only')
  if (!rustLoaded) return delegate('Rust kernels not loaded (tryLoadRustKernels / pkg missing) — TypeScript kernels are the CPU engine')
  if (
    plan.type === 'filter' &&
    plan.predicate.type === 'binary' &&
    plan.predicate.op === 'and' &&
    rustLoaded
  ) {
    const left = plan.predicate.left
    const right = plan.predicate.right
    if (
      left.type === 'binary' &&
      right.type === 'binary' &&
      left.left.type === 'col' &&
      right.left.type === 'col' &&
      left.right.type === 'lit' &&
      right.right.type === 'lit' &&
      typeof left.right.value === 'number' &&
      typeof right.right.value === 'number'
    ) {
      const opMap: Record<string, number> = { eq: 0, neq: 1, gt: 2, gte: 3, lt: 4, lte: 5 }
      const opA = opMap[left.op]
      const opB = opMap[right.op]
      if (opA !== undefined && opB !== undefined) {
        const input = fallback(plan.input)
        // Rust dual-filter wins when n is large enough to amortize copy_to; else JS kernels are faster.
        if (input.numRows < WASM_RUST_FILTER_MIN_ROWS) return delegate(`${input.numRows} rows < WASM_RUST_FILTER_MIN_ROWS (${WASM_RUST_FILTER_MIN_ROWS}); JS kernels are faster below that`)
        const ca = getColumn(input, left.left.name)
        const cb = getColumn(input, right.left.name)
        let idx: Uint32Array | null = null
        let kernel = ''
        if (
          ca.data instanceof Int32Array &&
          cb.data instanceof Float64Array &&
          activeKernels.filterAnd2I32F64
        ) {
          idx = activeKernels.filterAnd2I32F64(ca.data, cb.data, opA, left.right.value, opB, right.right.value)
          kernel = 'rust:filterAnd2I32F64'
        } else if (
          ca.data instanceof Int32Array &&
          cb.data instanceof Int32Array &&
          activeKernels.filterAnd2I32I32
        ) {
          idx = activeKernels.filterAnd2I32I32(ca.data, cb.data, opA, left.right.value, opB, right.right.value)
          kernel = 'rust:filterAnd2I32I32'
        } else if (
          ca.data instanceof Float64Array &&
          cb.data instanceof Float64Array &&
          activeKernels.filterMaskAnd2
        ) {
          idx = activeKernels.filterMaskAnd2(ca.data, cb.data, opA, left.right.value, opB, right.right.value)
          kernel = 'rust:filterMaskAnd2'
        }
        if (idx) {
          const out = tableFromColumns(input.columns.map((c) => takeColumn(c, idx!)))
          ctx?.trace({ node: 'filter', backend: 'wasm', kernel, ms: (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0, rows: out.numRows })
          return out
        }
        return delegate(`no Rust kernel for column types ${ca.field.dtype} × ${cb.field.dtype} (i32×f64, i32×i32, f64×f64 only)`)
      }
      return delegate('comparison operator not supported by the Rust dual filter')
    }
    return delegate('predicate is not "col OP number AND col OP number"')
  }
  return delegate('predicate is not an AND of two numeric comparisons')
}

export class WasmBackend implements Backend {
  readonly name = 'wasm' as const
  readonly capabilities = { name: 'wasm' as const, minRows: 1000 }
  private fallbackExecute: (plan: PlanNode) => TableView

  constructor(fallbackExecute: (plan: PlanNode) => TableView) {
    this.fallbackExecute = fallbackExecute
  }

  supports(plan: PlanNode): boolean {
    void plan
    return true
  }

  execute(plan: PlanNode, ctx?: ExecContext): TableView {
    return executeWithKernels(plan, this.fallbackExecute, ctx)
  }
}

export async function tryLoadRustKernels(): Promise<boolean> {
  try {
    const url = new URL('../pkg/columna_wasm.js', import.meta.url)
    const mod = await import(url.href)
    const api = mod.default ?? mod
    if (api && typeof api.filter_mask === 'function') {
      setWasmKernels({
        filterMask: (values, op, literal, nullBitmap) =>
          api.filter_mask(values, op, literal, nullBitmap ?? new Uint8Array(0)),
        filterMaskAnd2:
          typeof api.filter_mask_and2 === 'function'
            ? (a, b, opA, litA, opB, litB) => api.filter_mask_and2(a, b, opA, litA, opB, litB)
            : undefined,
        filterAnd2I32F64:
          typeof api.filter_and2_i32_f64 === 'function'
            ? (a, b, opA, litA, opB, litB) => api.filter_and2_i32_f64(a, b, opA, litA, opB, litB)
            : undefined,
        filterAnd2I32I32:
          typeof api.filter_and2_i32_i32 === 'function'
            ? (a, b, opA, litA, opB, litB) => api.filter_and2_i32_i32(a, b, opA, litA, opB, litB)
            : undefined,
        compactIndices: (mask) => api.compact_indices(mask),
        sortIndices: (values, descending, nullBitmap) =>
          api.sort_indices(values, descending, nullBitmap ?? new Uint8Array(0)),
        hashGroupBy: (keyHashes, values, aggOp, nullBitmap) => {
          const result = api.hash_group_by(keyHashes, values, aggOp, nullBitmap ?? new Uint8Array(0))
          return { keys: result.keys, values: result.values }
        },
      })
      rustLoaded = true
      return true
    }
  } catch {
    // no rust artifact
  }
  rustLoaded = false
  return false
}

/** Simple string helpers used by Phase 5 string ops (WASM-oriented). */
export function wasmStringContains(values: string[], needle: string, nullBitmap?: Uint8Array): Uint8Array {
  const mask = new Uint8Array(values.length)
  for (let i = 0; i < values.length; i++) {
    if (nullBitmap && !isValid(nullBitmap, i)) continue
    if (values[i]!.includes(needle)) mask[i] = 1
  }
  return mask
}

export function wasmStringLength(values: string[], nullBitmap?: Uint8Array): { data: Int32Array; nullBitmap?: Uint8Array } {
  const data = new Int32Array(values.length)
  let anyNull = false
  const outNull = new Uint8Array(Math.ceil(values.length / 8) || 1)
  for (let i = 0; i < values.length; i++) {
    if (nullBitmap && !isValid(nullBitmap, i)) {
      anyNull = true
      continue
    }
    setValid(outNull, i, true)
    data[i] = values[i]!.length
  }
  return { data, nullBitmap: anyNull ? outNull : undefined }
}

export function writeParquetLike(table: TableView): Uint8Array {
  const payload = {
    format: 'columna-parquet-like-v1',
    numRows: table.numRows,
    schema: table.schema,
    columns: table.columns.map((c) => {
      const values: Array<number | string | boolean | null> = []
      for (let i = 0; i < table.numRows; i++) {
        if (!isValid(c.nullBitmap, i)) values.push(null)
        else values.push(getValue(c.data, i) as number | string | boolean)
      }
      return { name: c.field.name, dtype: c.field.dtype, values, dictionary: c.dictionary }
    }),
  }
  return new TextEncoder().encode(JSON.stringify(payload))
}

export function readParquetLike(bytes: Uint8Array): TableView {
  const payload = JSON.parse(new TextDecoder().decode(bytes)) as {
    numRows: number
    schema: TableView['schema']
    columns: Array<{
      name: string
      dtype: TableView['schema'][number]['dtype']
      values: Array<number | string | boolean | null>
      dictionary?: string[]
    }>
  }
  const columns = payload.columns.map((c, idx) => {
    const field = payload.schema[idx]!
    if (field.dtype === 'utf8') {
      const data = c.values.map((v) => (v == null ? '' : String(v)))
      let anyNull = false
      const nullBitmap = new Uint8Array(Math.ceil(c.values.length / 8) || 1)
      for (let i = 0; i < c.values.length; i++) {
        if (c.values[i] == null) anyNull = true
        else setValid(nullBitmap, i, true)
      }
      return { field: { ...field }, data, nullBitmap: anyNull ? nullBitmap : undefined }
    }
    const data = allocateData(field.dtype, c.values.length)
    let anyNull = false
    const nullBitmap = new Uint8Array(Math.ceil(c.values.length / 8) || 1)
    for (let i = 0; i < c.values.length; i++) {
      const v = c.values[i]
      if (v == null) {
        anyNull = true
        continue
      }
      setValid(nullBitmap, i, true)
      setValue(data, i, v, field.dtype)
    }
    return {
      field: { ...field },
      data,
      nullBitmap: anyNull ? nullBitmap : undefined,
      dictionary: c.dictionary,
    }
  })
  return tableFromColumns(columns)
}
