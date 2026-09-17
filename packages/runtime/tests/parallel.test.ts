import { describe, expect, it, afterAll } from 'vitest'
import {
  parallelFilter,
  parallelSort,
  parallelUnique,
  parallelGroupBy,
  closeParallelPool,
  PARALLEL_FILTER_MIN_ROWS,
  PARALLEL_SORT_MIN_ROWS,
  PARALLEL_UNIQUE_MIN_ROWS,
  PARALLEL_GROUPBY_MIN_ROWS,
} from '../src/parallel.js'

const isNode = typeof process !== 'undefined' && Boolean(process.versions?.node)

function makeF64(n: number, seed = 1): Float64Array {
  const a = new Float64Array(n)
  let s = seed
  for (let i = 0; i < n; i++) {
    // simple LCG
    s = (s * 1103515245 + 12345) & 0x7fffffff
    a[i] = s % 1000
  }
  return a
}

function syncFilterRef(cols: Float64Array[], ops: ('eq' | 'neq' | 'gt' | 'gte' | 'lt' | 'lte')[], lits: number[], n: number): Uint32Array {
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    let hit = true
    for (let c = 0; c < cols.length; c++) {
      const v = cols[c]![i]!
      const op = ops[c]!
      const lit = lits[c]!
      let ok = false
      switch (op) {
        case 'eq': ok = v === lit; break
        case 'neq': ok = v !== lit; break
        case 'gt': ok = v > lit; break
        case 'gte': ok = v >= lit; break
        case 'lt': ok = v < lit; break
        case 'lte': ok = v <= lit; break
      }
      if (!ok) { hit = false; break }
    }
    if (hit) out.push(i)
  }
  return new Uint32Array(out)
}

function syncUniqueRef(cols: Float64Array[], n: number): Uint32Array {
  const seen = new Set<string>()
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    const key = cols.map((c) => c[i]!).join('\0')
    if (!seen.has(key)) { seen.add(key); out.push(i) }
  }
  return new Uint32Array(out)
}

function toSharedF64(src: Float64Array): SharedArrayBuffer {
  const sab = new SharedArrayBuffer(src.length * 8)
  new Float64Array(sab).set(src)
  return sab
}

afterAll(async () => {
  await closeParallelPool()
})

describe.runIf(isNode)('parallel worker dispatch (Node worker_threads)', () => {
  it('parallelFilter matches sync on small input forced via minRows', async () => {
    const n = 1000
    const a = makeF64(n, 7)
    const b = makeF64(n, 13)
    const ops = ['gt', 'lt'] as const
    const lits = [100, 500]
    const expected = syncFilterRef([a, b], ops as never, lits, n)
    const got = await parallelFilter([a, b], ops as never, lits, n, { minRows: 0 })
    expect(Array.from(got)).toEqual(Array.from(expected))
  })

  it('parallelSort matches sync (single key, ascending)', async () => {
    const n = 2000
    const a = makeF64(n, 21)
    const buf = toSharedF64(a)
    const idx = await parallelSort(
      [{ buffer: buf, kind: 'Float64Array', length: n, descending: false, nullsLast: true }],
      n,
      { minRows: 0 },
    )
    // Verify sorted order
    for (let i = 1; i < n; i++) expect(a[idx[i]!]).toBeGreaterThanOrEqual(a[idx[i - 1]!])
    // Verify it's a permutation
    const sorted = Array.from(idx).sort((x, y) => x - y)
    for (let i = 0; i < n; i++) expect(sorted[i]).toBe(i)
  })

  it('parallelSort matches sync (multi-key, mixed directions)', async () => {
    const n = 1500
    const a = makeF64(n, 5)
    const b = makeF64(n, 9)
    const idx = await parallelSort(
      [
        { buffer: toSharedF64(a), kind: 'Float64Array', length: n, descending: false, nullsLast: true },
        { buffer: toSharedF64(b), kind: 'Float64Array', length: n, descending: true, nullsLast: true },
      ],
      n,
      { minRows: 0 },
    )
    // lexicographic check
    for (let i = 1; i < n; i++) {
      const pa = a[idx[i - 1]!]!, pb = a[idx[i]!]!
      expect(pa <= pb).toBe(true)
      if (pa === pb) expect(b[idx[i - 1]!]).toBeGreaterThanOrEqual(b[idx[i]!])
    }
  })

  it('parallelUnique matches sync (multi-column, first-seen)', async () => {
    const n = 1200
    const a = makeF64(n, 31)
    const b = makeF64(n, 41)
    const expected = syncUniqueRef([a, b], n)
    const got = await parallelUnique([a, b], n, { minRows: 0 })
    expect(got).not.toBeNull()
    expect(Array.from(got!)).toEqual(Array.from(expected))
  })

  it('parallelGroupBy returns aggregated sums matching sync', async () => {
    const n = 3000
    const keys = makeF64(n, 3) // values 0..999
    const vals = makeF64(n, 99)
    const res = await parallelGroupBy(
      [keys],
      [vals],
      [{ op: 'sum', colIdx: 0 }],
      n,
      { minRows: 0 },
    )
    expect(res).not.toBeNull()
    // Build sync reference: Map<key, sum>
    const ref = new Map<number, number>()
    for (let i = 0; i < n; i++) {
      const k = keys[i]!
      ref.set(k, (ref.get(k) ?? 0) + vals[i]!)
    }
    // Parse returned data: rows of [keyCount, key, count, sum, min, max, m2]
    const data = (res as { data: Float64Array }).data
    const got = new Map<number, number>()
    let pos = 0
    while (pos < data.length) {
      const parts = data[pos++]!
      const key = data[pos++]!
      pos++ // count
      const sum = data[pos++]!
      pos++ // min
      pos++ // max
      pos++ // m2
      got.set(key, sum)
      void parts
    }
    expect(got.size).toBe(ref.size)
    for (const [k, v] of ref) {
      expect(got.get(k)).toBeCloseTo(v, 6)
    }
  })

  it('pool is reused across calls', async () => {
    const n = 500
    const a = makeF64(n, 1)
    const ops = ['gt'] as const
    const lits = [100]
    const first = await parallelFilter([a], ops as never, lits, n, { minRows: 0 })
    const second = await parallelFilter([a], ops as never, lits, n, { minRows: 0 })
    expect(Array.from(second)).toEqual(Array.from(first))
  })

  it('falls back to sync below threshold (returns same-shape result)', async () => {
    const n = 50
    const a = makeF64(n, 2)
    const ops = ['gt'] as const
    const lits = [100]
    // With default minRows (10M), small input must still produce a correct sync result.
    const got = await parallelFilter([a], ops as never, lits, n)
    const expected = syncFilterRef([a], ops as never, lits, n)
    expect(Array.from(got)).toEqual(Array.from(expected))
  })
})

describe('parallel thresholds exported', () => {
  it('exposes filter/sort/unique/groupby thresholds', () => {
    expect(PARALLEL_FILTER_MIN_ROWS).toBeGreaterThan(0)
    expect(PARALLEL_SORT_MIN_ROWS).toBeGreaterThan(0)
    expect(PARALLEL_UNIQUE_MIN_ROWS).toBeGreaterThan(0)
    expect(PARALLEL_GROUPBY_MIN_ROWS).toBeGreaterThan(0)
  })
})
