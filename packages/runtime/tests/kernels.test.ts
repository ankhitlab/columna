import { describe, expect, it } from 'vitest'
import { isValid, tableFromColumns } from '@columna/arrow'
import type { ExprNode } from '../src/types.js'
import { __setFusedCompile } from '../src/fused.js'
import { roundHalfAway } from '../src/math.js'
import {
  argsortNumeric,
  countUniqueNumeric,
  tryFastExprColumn,
  tryFastFilter,
  quantileSelect,
  dualGtIndices,
  projectColumnNames,
  tryFastJoin,
  tryFastSort,
} from '../src/fast.js'
import { PARALLEL_MIN_ROWS, parallelDualGtIndices } from '../src/parallel.js'

describe('filter kernels', () => {
  it('dualGtIndices matches naive scan', () => {
    const a = new Int32Array([10, 20, 30, 40, 50])
    const b = new Float64Array([1, 2, 3, 4, 5])
    const idx = dualGtIndices(a, b, 25, 2.5)
    // a>25 && b>2.5 → indices 2 (30,3), 3 (40,4), 4 (50,5)
    expect([...idx]).toEqual([2, 3, 4])
  })

  it('dualGtIndices returns empty when nothing matches', () => {
    const a = new Int32Array([1, 2, 3])
    const b = new Int32Array([1, 2, 3])
    expect(dualGtIndices(a, b, 100, 100).length).toBe(0)
  })

  it('parallelDualGtIndices falls back to sync below PARALLEL_MIN_ROWS', async () => {
    expect(PARALLEL_MIN_ROWS).toBeGreaterThan(1000)
    const n = 10_000
    const a = new Int32Array(n)
    const b = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      a[i] = i
      b[i] = i * 0.5
    }
    const sync = dualGtIndices(a, b, n / 2, n / 4)
    const parallel = await parallelDualGtIndices(a, b, n / 2, n / 4)
    expect(parallel.length).toBe(sync.length)
    expect([...parallel]).toEqual([...sync])
  })

  it('parallelDualGtIndices with lowered minRows matches sync', async () => {
    const n = 4_000
    const a = new Int32Array(n)
    const b = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      a[i] = i
      b[i] = i
    }
    const sync = dualGtIndices(a, b, 1000, 1000)
    const parallel = await parallelDualGtIndices(a, b, 1000, 1000, { minRows: 1_000 })
    expect([...parallel]).toEqual([...sync])
  })
})

describe('tryFastSort / tryFastJoin', () => {
  it('tryFastSort top-k matches full sort truncated', () => {
    const table = tableFromColumns([
      { field: { name: 'id', dtype: 'i32', nullable: false }, data: new Int32Array([1, 2, 3, 4, 5]) },
      { field: { name: 'v', dtype: 'f64', nullable: false }, data: new Float64Array([50, 10, 40, 20, 30]) },
    ])
    const by = [{ expr: { type: 'col' as const, name: 'v' }, descending: true }]
    const top = tryFastSort(table, by, 3)
    const full = tryFastSort(table, by)
    expect(top).not.toBeNull()
    expect(full).not.toBeNull()
    expect(top!.numRows).toBe(3)
    expect([...(top!.columns[1]!.data as Float64Array)]).toEqual([50, 40, 30])
    expect([...(full!.columns[1]!.data as Float64Array).subarray(0, 3)]).toEqual([50, 40, 30])
  })

  it('tryFastJoin dense and sparse probes', () => {
    const leftDense = tableFromColumns([
      { field: { name: 'user_id', dtype: 'i32', nullable: false }, data: new Int32Array([0, 1, 2]) },
      { field: { name: 'v', dtype: 'i32', nullable: false }, data: new Int32Array([10, 20, 30]) },
    ])
    const rightDense = tableFromColumns([
      { field: { name: 'user_id', dtype: 'i32', nullable: false }, data: new Int32Array([1, 2]) },
      { field: { name: 'score', dtype: 'i32', nullable: false }, data: new Int32Array([100, 200]) },
    ])
    const dense = tryFastJoin(leftDense, rightDense, ['user_id'], ['user_id'], 'inner')
    expect(dense).not.toBeNull()
    expect(dense!.numRows).toBe(2)
    expect([...(dense!.columns.find((c) => c.field.name === 'score')!.data as Int32Array)]).toEqual([100, 200])

    const leftSparse = tableFromColumns([
      {
        field: { name: 'user_id', dtype: 'i32', nullable: false },
        data: new Int32Array([0, 10_000_000, 20_000_000]),
      },
      { field: { name: 'v', dtype: 'i32', nullable: false }, data: new Int32Array([10, 20, 30]) },
    ])
    const rightSparse = tableFromColumns([
      {
        field: { name: 'user_id', dtype: 'i32', nullable: false },
        data: new Int32Array([10_000_000, 20_000_000]),
      },
      { field: { name: 'score', dtype: 'i32', nullable: false }, data: new Int32Array([100, 200]) },
    ])
    const sparse = tryFastJoin(leftSparse, rightSparse, ['user_id'], ['user_id'], 'inner')
    expect(sparse).not.toBeNull()
    expect(sparse!.numRows).toBe(2)
    expect([...(sparse!.columns.find((c) => c.field.name === 'v')!.data as Int32Array)]).toEqual([20, 30])
  })
})

describe('projectColumnNames', () => {
  it('extracts plain columns and rejects complex exprs', () => {
    expect(projectColumnNames(['a', 'b'])).toEqual(['a', 'b'])
    expect(projectColumnNames([{ type: 'col', name: 'x' }, { type: 'alias', expr: { type: 'col', name: 'y' }, name: 'yy' }])).toEqual([
      'x',
      'y',
    ])
    expect(projectColumnNames([{ type: 'binary', op: 'add', left: { type: 'col', name: 'a' }, right: { type: 'lit', value: 1 } }])).toBeNull()
  })
})

describe('argsortNumeric (radix)', () => {
  const naive = (v: number[], desc: boolean) =>
    v
      .map((x, i) => i)
      .sort((a, b) => {
        const av = v[a]!
        const bv = v[b]!
        if (av === bv) return a - b // stable
        return desc ? (av < bv ? 1 : -1) : av < bv ? -1 : 1
      })

  it('matches a stable comparator sort on mixed signs, duplicates, ints and fractions', () => {
    let seed = 7
    const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296)
    const v: number[] = []
    for (let i = 0; i < 20000; i++) {
      const r = rnd()
      v.push(r < 0.3 ? Math.floor(rnd() * 50) - 25 : r < 0.6 ? (rnd() - 0.5) * 1e6 : r < 0.8 ? -0 + 0 : (rnd() - 0.5) * 1e-6)
    }
    const data = new Float64Array(v)
    expect(Array.from(argsortNumeric(data, data.length, undefined, false))).toEqual(naive(v, false))
    expect(Array.from(argsortNumeric(data, data.length, undefined, true))).toEqual(naive(v, true))
    const i32 = new Int32Array(v.map((x) => Math.trunc(x)))
    expect(Array.from(argsortNumeric(i32, i32.length, undefined, false))).toEqual(naive(Array.from(i32), false))
  })

  it('puts nulls last in input order; infinities and NaN ordered after finite values', () => {
    const data = new Float64Array([3, -Infinity, 1, NaN, 2, Infinity, 0])
    const bm = new Uint8Array([0b01011101]) // rows 1 and 5 are null (bits 1,5 clear)
    const order = Array.from(argsortNumeric(data, data.length, bm, false))
    expect(order).toEqual([6, 2, 4, 0, 3, 1, 5])
    expect(Array.from(argsortNumeric(data, data.length, bm, true))).toEqual([3, 0, 4, 2, 6, 1, 5])
  })
})

describe('quantileSelect', () => {
  it('matches sort-based type-7 quantiles', () => {
    let seed = 3
    const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296)
    for (const len of [1, 2, 3, 7, 100, 5001]) {
      const v = Float64Array.from({ length: len }, () => (rnd() < 0.3 ? Math.floor(rnd() * 10) : (rnd() - 0.5) * 100))
      const sorted = Array.from(v).sort((a, b) => a - b)
      for (const q of [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]) {
        const pos = (len - 1) * q
        const lo = Math.floor(pos)
        const hi = Math.ceil(pos)
        const expected = lo === hi ? sorted[lo]! : sorted[lo]! * (1 - (pos - lo)) + sorted[hi]! * (pos - lo)
        expect(quantileSelect(v.slice(), len, q)).toBeCloseTo(expected, 12)
      }
    }
  })
})

describe('countUniqueNumeric', () => {
  it('dense-int and hashed-float paths agree with Set semantics', () => {
    const ints = new Float64Array([5, 3, 5, -2, 3, 1e6, -2, 0])
    expect(countUniqueNumeric(ints, ints.length)).toBe(5)
    const floats = new Float64Array([0.5, 0.25, 0.5, 1e300, -0, 0, NaN, NaN, 0.25])
    // Set treats -0 === 0 and NaN as one value → {0.5, 0.25, 1e300, 0, NaN} = 5
    expect(countUniqueNumeric(floats, floats.length)).toBe(5)
    const wide = new Int32Array([1, 2_000_000_000, 1, -2_000_000_000]) // span > 16M → hash path
    expect(countUniqueNumeric(wide, wide.length)).toBe(3)
    expect(countUniqueNumeric(new Float64Array(0), 0)).toBe(0)
    let seed = 11
    const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296)
    const big = Float64Array.from({ length: 50_000 }, () => Math.floor(rnd() * 1000) + (rnd() < 0.5 ? 0.5 : 0))
    expect(countUniqueNumeric(big, big.length)).toBe(new Set(big).size)
  })
})

describe('fused arithmetic kernel', () => {
  const rows = 1000
  const build = () => {
    const a = new Float64Array(rows)
    const b = new Int32Array(rows)
    const bm = new Uint8Array(Math.ceil(rows / 8)).fill(0xff)
    for (let i = 0; i < rows; i++) {
      a[i] = (i % 97) * 0.37 - 12.5
      b[i] = (i * 7) % 23
      if (i % 10 === 3) bm[i >> 3] &= ~(1 << (i & 7)) // b null on every 10th row
    }
    return tableFromColumns([
      { field: { name: 'a', dtype: 'f64', nullable: false }, data: a },
      { field: { name: 'b', dtype: 'i32', nullable: true }, data: b, nullBitmap: bm },
    ])
  }
  const col = (name: string): ExprNode => ({ type: 'col', name })
  const lit = (value: number): ExprNode => ({ type: 'lit', value })
  const bin = (op: 'add' | 'sub' | 'mul' | 'div' | 'mod' | 'pow' | 'gt' | 'lte', left: ExprNode, right: ExprNode): ExprNode => ({
    type: 'binary',
    op,
    left,
    right,
  })
  const un = (op: 'abs' | 'neg' | 'sqrt' | 'log' | 'exp' | 'round' | 'floor', expr: ExprNode, extra: object = {}): ExprNode =>
    ({ type: 'unary', op, expr, ...extra }) as ExprNode

  // ((a − 3) / 2.5) ** 2 + |b| · log_10(b + 1) − round(a, 1)
  const chain = bin(
    'sub',
    bin('add', bin('pow', bin('div', bin('sub', col('a'), lit(3)), lit(2.5)), lit(2)), bin('mul', un('abs', col('b')), un('log', bin('add', col('b'), lit(1)), { base: 10 }))),
    un('round', col('a'), { decimals: 1 }),
  )
  const reference = (a: number, b: number) => ((a - 3) / 2.5) ** 2 + Math.abs(b) * (Math.log(b + 1) * (1 / Math.log(10))) - roundHalfAway(a, 1)

  it('fuses a chain into one pass with the same values and null mask as the per-node path', () => {
    const table = build()
    const out = tryFastExprColumn(table, chain, 'y')!
    expect(out).not.toBeNull()
    const data = out.data as Float64Array
    const a = table.columns[0]!.data as Float64Array
    const b = table.columns[1]!.data as Int32Array
    for (let i = 0; i < rows; i++) {
      const isNull = i % 10 === 3
      expect(isValid(out.nullBitmap, i)).toBe(!isNull)
      if (!isNull) expect(data[i]).toBeCloseTo(reference(a[i]!, b[i]!), 10)
    }
    expect(out.field.dtype).toBe('f64')
    expect(out.field.nullable).toBe(true)
  })

  it('closure fallback (no eval) matches the compiled kernel bit for bit', () => {
    const table = build()
    const compiled = tryFastExprColumn(table, chain, 'y')!.data as Float64Array
    __setFusedCompile(false)
    try {
      const closured = tryFastExprColumn(table, chain, 'y')!.data as Float64Array
      expect(Array.from(closured)).toEqual(Array.from(compiled))
    } finally {
      __setFusedCompile(null)
    }
  })

  it('comparison at the root yields a bool mask usable by filter', () => {
    const table = build()
    // (a − 3) / 2.5 > 5  → bool
    const pred = bin('gt', bin('div', bin('sub', col('a'), lit(3)), lit(2.5)), lit(5))
    const mask = tryFastExprColumn(table, pred, 'm')!
    expect(mask.field.dtype).toBe('bool')
    const a = table.columns[0]!.data as Float64Array
    const m = mask.data as Uint8Array
    for (let i = 0; i < rows; i++) expect(m[i]).toBe((a[i]! - 3) / 2.5 > 5 ? 1 : 0)
    // and through the filter kernel
    const filtered = tryFastFilter(table, pred)!
    let expected = 0
    for (let i = 0; i < rows; i++) if ((a[i]! - 3) / 2.5 > 5) expected++
    expect(filtered.numRows).toBe(expected)
  })

  it('is transparent for non-fusable trees (string column inside) — result still correct via per-node path', () => {
    const table = tableFromColumns([
      { field: { name: 'a', dtype: 'f64', nullable: false }, data: new Float64Array([1, 2, 3]) },
      { field: { name: 's', dtype: 'utf8', nullable: false }, data: ['x', 'y', 'z'] },
    ])
    const expr = bin('add', bin('mul', col('a'), lit(2)), { type: 'str', op: 'len', expr: col('s') } as ExprNode)
    const out = tryFastExprColumn(table, expr, 'y')
    // str len is not part of evalVec's numeric subset → null here; the CPU scalar path takes over.
    // What matters: fusion must not throw or return a wrong partial result.
    expect(out === null || Array.from(out.data as Float64Array).join() === '3,5,7').toBe(true)
  })
})
