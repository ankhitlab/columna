import { describe, expect, it } from 'vitest'
import { bitmapToWords, maskToIndices, prepareGpuFilterPredicate, columnToF32, type GpuFilterPredicate } from '@columna/webgpu'
import { DataFrame, col, executeCpu } from 'columna'
import { getColumn, type Column } from '@columna/arrow'

/**
 * JavaScript transcription of FILTER_SHADER: same raw-word inputs, same bitcast interpretation, same NaN
 * rule and validity test. It cannot prove what a GPU driver does, but it pins the contract the shader
 * implements against the CPU engine on the exact inputs the backend would upload.
 */
function emulateFilter(preds: GpuFilterPredicate[]): Uint32Array {
  const n = preds[0]!.words.length
  const mask = new Uint32Array(n)
  const f32 = new Float32Array(1)
  const f32u = new Uint32Array(f32.buffer)
  const i32 = new Int32Array(1)
  const i32u = new Uint32Array(i32.buffer)
  const cmp = (v: number, l: number, op: number) =>
    op === 0 ? v === l : op === 1 ? v !== l : op === 2 ? v > l : op === 3 ? v >= l : op === 4 ? v < l : op === 5 ? v <= l : false
  preds.forEach((p, pi) => {
    for (let i = 0; i < n; i++) {
      let ok = false
      const valid = !p.valid || ((p.valid[i >> 5]! >>> (i & 31)) & 1) === 1
      if (valid) {
        const raw = p.words[i]!
        if (p.kind === 0) {
          if ((raw & 0x7fffffff) > 0x7f800000) ok = p.op === 1
          else {
            f32u[0] = raw
            const v = f32[0]!
            f32u[0] = p.literalBits
            ok = cmp(v, f32[0]!, p.op)
          }
        } else if (p.kind === 1) {
          i32u[0] = raw
          const v = i32[0]!
          i32u[0] = p.literalBits
          ok = cmp(v, i32[0]!, p.op)
        } else ok = cmp(raw >>> 0, p.literalBits >>> 0, p.op)
      }
      const bit = ok ? 1 : 0
      mask[i] = pi === 0 ? bit : mask[i]! & bit
    }
  })
  return mask
}

const OPS = { eq: 0, neq: 1, gt: 2, gte: 3, lt: 4, lte: 5 } as const
type Op = keyof typeof OPS

async function cpuRows(df: DataFrame, column: string, op: Op, literal: number): Promise<number[]> {
  const e = col(column)
  const expr = op === 'eq' ? e.eq(literal) : op === 'neq' ? e.neq(literal) : op === 'gt' ? e.gt(literal) : op === 'gte' ? e.gte(literal) : op === 'lt' ? e.lt(literal) : e.lte(literal)
  const out = await executeCpu(df.withColumn('__i', col('__i')).filter(expr).plan)
  return Array.from(getColumn(out, '__i').data as Int32Array)
}

function gpuRows(df: DataFrame, column: string, op: Op, literal: number): number[] | null {
  const c = getColumn(df.table, column)
  const p = prepareGpuFilterPredicate(c, df.table.numRows, OPS[op], literal)
  if (!p) return null
  return Array.from(maskToIndices(emulateFilter([p])))
}

describe('GPU filter exactness: same rows as the CPU engine', () => {
  const idx = (n: number) => Int32Array.from({ length: n }, (_, i) => i)

  it('i32 values around 2^24 are compared bit-exactly (16 777 216 ≠ 16 777 217)', async () => {
    const x = new Int32Array([16_777_216, 16_777_217, 16_777_215, -16_777_217, 2_147_483_647, -2_147_483_648])
    const df = DataFrame.fromColumns({ x, __i: idx(x.length) })
    for (const op of Object.keys(OPS) as Op[]) {
      for (const lit of [16_777_216, 16_777_217, 2_147_483_647, -2_147_483_648, 0]) {
        expect(gpuRows(df, 'x', op, lit), `${op} ${lit}`).toEqual(await cpuRows(df, 'x', op, lit))
      }
    }
    expect(gpuRows(df, 'x', 'eq', 16_777_216)).toEqual([0])
  })

  it('f64 and datetime columns are not sent to the float32 kernel (CPU fallback) unless lossy is requested', () => {
    const f64: Column = { field: { name: 'v', dtype: 'f64', nullable: false }, data: new Float64Array([16_777_216, 16_777_217]) }
    const dt: Column = { field: { name: 't', dtype: 'datetime', nullable: false }, data: new Float64Array([1_700_000_000_000, 1_700_000_000_001]) }
    const cat: Column = { field: { name: 'c', dtype: 'category', nullable: false }, data: new Uint32Array([0, 1]), dictionary: ['a', 'b'] }
    const bool: Column = { field: { name: 'b', dtype: 'bool', nullable: false }, data: new Uint8Array([0, 1]) }
    expect(prepareGpuFilterPredicate(f64, 2, 0, 16_777_216)).toBeNull()
    expect(prepareGpuFilterPredicate(dt, 2, 0, 1_700_000_000_001)).toBeNull()
    expect(prepareGpuFilterPredicate(cat, 2, 0, 1)).toBeNull()
    expect(prepareGpuFilterPredicate(bool, 2, 0, 1)).toBeNull() // CPU: false === 0 is false — booleans are not numbers
    // explicit opt-in reproduces the old float32 behaviour, and only then
    const lossy = prepareGpuFilterPredicate(f64, 2, 0, 16_777_216, true)!
    expect(lossy.kind).toBe(0)
    expect(Array.from(maskToIndices(emulateFilter([lossy])))).toEqual([0, 1])
  })

  it('literals the column type cannot represent fall back to the CPU', () => {
    const i32: Column = { field: { name: 'x', dtype: 'i32', nullable: false }, data: new Int32Array([1, 2, 3]) }
    const u32: Column = { field: { name: 'x', dtype: 'u32', nullable: false }, data: new Uint32Array([1, 2, 3]) }
    const f32: Column = { field: { name: 'x', dtype: 'f32', nullable: false }, data: new Float32Array([0.1, 0.2]) }
    expect(prepareGpuFilterPredicate(i32, 3, 2, 2.5)).toBeNull()
    expect(prepareGpuFilterPredicate(i32, 3, 2, 3_000_000_000)).toBeNull()
    expect(prepareGpuFilterPredicate(u32, 3, 2, -1)).toBeNull()
    expect(prepareGpuFilterPredicate(f32, 2, 2, 0.1)).toBeNull() // 0.1 is not an f32
    expect(prepareGpuFilterPredicate(f32, 2, 2, 0.5)).not.toBeNull()
    expect(prepareGpuFilterPredicate(i32, 3, 2, 2)).not.toBeNull()
  })

  it('null rows never match, including physical zeros inside a Float32Array column', async () => {
    const f32 = new Float32Array([0, 1, 0, 2.5, 0])
    const bitmap = new Uint8Array([0b10110]) // rows 1, 2, 4 valid; rows 0 and 3 null (row 2 is a real 0)
    const column: Column = { field: { name: 'v', dtype: 'f32', nullable: true }, data: f32, nullBitmap: bitmap }
    const df = new DataFrame({ schema: [column.field, { name: '__i', dtype: 'i32', nullable: false }], numRows: 5, columns: [column, { field: { name: '__i', dtype: 'i32', nullable: false }, data: idx(5) }] })
    for (const op of Object.keys(OPS) as Op[]) {
      for (const lit of [0, 1, 2.5]) {
        expect(gpuRows(df, 'v', op, lit), `${op} ${lit}`).toEqual(await cpuRows(df, 'v', op, lit))
      }
    }
    expect(gpuRows(df, 'v', 'eq', 0)).toEqual([2, 4]) // rows 2 and 4 hold real zeros; row 0 is null
    // the map / reduce conversion also masks nulls now instead of returning the raw Float32Array
    const asF32 = columnToF32(column, 5)
    expect(asF32).not.toBe(f32)
    expect(Number.isNaN(asF32[0])).toBe(true)
    expect(asF32[2]).toBe(0)
    expect(bitmapToWords(bitmap, 5)[0]).toBe(0b10110)
  })

  it('random i32 / f32 / u32 data with nulls: every operator agrees with the CPU', async () => {
    let seed = 12345
    const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296)
    const n = 2000
    const i32 = Int32Array.from({ length: n }, () => Math.floor((rnd() - 0.5) * 2 ** 32))
    const f32 = Float32Array.from({ length: n }, () => (rnd() - 0.5) * 1000)
    const u32 = Uint32Array.from({ length: n }, () => Math.floor(rnd() * 2 ** 32))
    const bool = Uint8Array.from({ length: n }, () => (rnd() < 0.5 ? 1 : 0))
    const bitmap = new Uint8Array(Math.ceil(n / 8))
    for (let i = 0; i < n; i++) if (rnd() < 0.8) bitmap[i >> 3]! |= 1 << (i & 7)
    const mk = (name: string, dtype: 'i32' | 'f32' | 'u32' | 'bool', data: Column['data']): Column => ({ field: { name, dtype, nullable: true }, data, nullBitmap: bitmap })
    const cols = [mk('i', 'i32', i32), mk('f', 'f32', f32), mk('u', 'u32', u32), mk('b', 'bool', bool), { field: { name: '__i', dtype: 'i32' as const, nullable: false }, data: idx(n) }]
    const df = new DataFrame({ schema: cols.map((c) => c.field), numRows: n, columns: cols })
    const lits: Record<string, number[]> = {
      i: [0, i32[7]!, 16_777_217, -2_147_483_648],
      f: [0, Math.fround(f32[3]!), 0.5, -250],
      u: [0, u32[5]!, 4_294_967_295, 2 ** 31],
    }
    let checked = 0
    for (const c of ['i', 'f', 'u']) {
      for (const op of Object.keys(OPS) as Op[]) {
        for (const lit of lits[c]!) {
          const gpu = gpuRows(df, c, op, lit)
          expect(gpu, `${c} ${op} ${lit}`).not.toBeNull()
          expect(gpu, `${c} ${op} ${lit}`).toEqual(await cpuRows(df, c, op, lit))
          checked++
        }
      }
    }
    expect(checked).toBe(72)
  })
})
