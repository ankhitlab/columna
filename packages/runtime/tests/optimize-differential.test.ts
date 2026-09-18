/**
 * Differential property test: optimizer rewrites must not change query results.
 *
 * executeCpu() always runs optimizePlan first, so this uses executeCpuUnoptimized
 * on both the raw plan and optimizePlan(plan).
 *
 * Scope: filter / project / rename / sort / withColumn / alias. Joins and
 * filter→sort→limit fuse are covered by dedicated regressions (random join+prune
 * and fused top-k still have separate open issues).
 */
import { describe, expect, it } from 'vitest'
import { DataFrame, col, type LazyFrame } from '@columna/core'
import type { TableView } from '@columna/arrow'
import { executeCpuUnoptimized, optimizePlan } from '../src/index.js'
import type { PlanNode } from '../src/types.js'

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function pick<T>(rng: () => number, xs: readonly T[]): T {
  return xs[Math.floor(rng() * xs.length)]!
}

function jsonReplacer(_k: string, v: unknown): unknown {
  if (typeof v === 'number') {
    if (Object.is(v, -0)) return { __num: '-0' }
    if (Number.isNaN(v)) return { __num: 'NaN' }
    if (v === Infinity) return { __num: 'Infinity' }
    if (v === -Infinity) return { __num: '-Infinity' }
  }
  return v
}

function fingerprint(table: TableView): string {
  const df = new DataFrame(table)
  return JSON.stringify(
    {
      columns: df.columns,
      dtypes: df.dtypes,
      rows: df.toArray(),
    },
    jsonReplacer,
  )
}

function makeFrame(rng: () => number, n: number): DataFrame {
  const labels = ['p', 'q', 'r', 'z', 'a'] as const
  return DataFrame.fromRows(
    Array.from({ length: n }, (_, id) => {
      const a = Math.floor(rng() * 5)
      const lo = Math.floor(rng() * 3)
      return {
        id,
        a,
        b: Math.floor(rng() * 7),
        x: rng() < 0.12 ? null : Math.round((rng() - 0.5) * 2000) / 10,
        lo,
        hi: lo + 1 + Math.floor(rng() * 4),
        s: labels[Math.floor(rng() * labels.length)]!,
      }
    }),
  )
}

type Query = LazyFrame<any>

type State = {
  query: Query
  cols: Set<string>
}

function has(cols: Set<string>, ...names: string[]): boolean {
  return names.every((n) => cols.has(n))
}

function applyRandomOp(state: State, rng: () => number): State {
  const { query: q, cols } = state
  const op = Math.floor(rng() * 7)

  switch (op) {
    case 0:
      if (!has(cols, 'a')) return state
      return { ...state, query: q.filter(col('a').gt(pick(rng, [0, 1, 2]))) }

    case 1:
      if (!has(cols, 'x')) return state
      return { ...state, query: q.filter(col('x').gt(0)) }

    case 2:
      if (!has(cols, 'a', 'lo', 'hi')) return state
      return { ...state, query: q.filter(col('a').isBetween(col('lo'), col('hi'))) }

    case 3: {
      if (!has(cols, 'id', 'a', 'b')) return state
      const extras = ['x', 's', 'lo', 'hi', 'y'].filter((c) => cols.has(c))
      const next = ['id', 'a', 'b', ...extras.slice(0, Math.floor(rng() * (extras.length + 1)))]
      return { ...state, query: q.select(...next), cols: new Set(next) }
    }

    case 4: {
      if (!has(cols, 'a', 'b')) return state
      return {
        ...state,
        query: q.rename({ a: 'b', b: 'a' }),
        cols: new Set([...cols].map((c) => (c === 'a' ? 'b' : c === 'b' ? 'a' : c))),
      }
    }

    case 5: {
      const key = cols.has('id') ? 'id' : cols.has('a') ? 'a' : [...cols][0]
      if (!key) return state
      return { ...state, query: q.sort(key) }
    }

    case 6: {
      if (!has(cols, 'x') || cols.has('y')) return state
      return {
        ...state,
        query: q.withColumn('y', col('x').fillNull(0).add(1)),
        cols: new Set([...cols, 'y']),
      }
    }

    default:
      return state
  }
}

function randomPlan(seed: number): PlanNode {
  const rng = mulberry32(seed)
  let state: State = {
    query: makeFrame(rng, 4 + Math.floor(rng() * 12)).lazy(),
    cols: new Set(['id', 'a', 'b', 'x', 'lo', 'hi', 's']),
  }

  if (rng() < 0.55) {
    state = {
      ...state,
      query: state.query.filter(col('a').gte(0)).filter(col('b').gte(0)),
    }
  }

  const depth = 2 + Math.floor(rng() * 5)
  for (let i = 0; i < depth; i++) state = applyRandomOp(state, rng)

  if (rng() < 0.25 && has(state.cols, 'a', 'b', 'id')) {
    state = {
      ...state,
      query: state.query.select(col('a').alias('b'), col('b').alias('a'), 'id'),
      cols: new Set(['b', 'a', 'id']),
    }
  }

  const key = state.cols.has('id') ? 'id' : state.cols.has('a') ? 'a' : [...state.cols][0]
  if (key) state = { ...state, query: state.query.sort(key) }

  return state.query.plan
}

describe('optimizePlan differential property', () => {
  it('execute(unoptimized) matches execute(optimizePlan) on 2000 seeded random plans', () => {
    const failures: Array<{ seed: number; detail: string }> = []

    for (let seed = 0; seed < 2000; seed++) {
      const plan = randomPlan(seed ^ 0x9e3779b9)
      try {
        const rawFp = fingerprint(executeCpuUnoptimized(plan))
        const optFp = fingerprint(executeCpuUnoptimized(optimizePlan(plan)))
        if (rawFp !== optFp) failures.push({ seed, detail: `mismatch\nraw=${rawFp}\nopt=${optFp}` })
      } catch (err) {
        failures.push({
          seed,
          detail: `throw:${err instanceof Error ? err.message : String(err)}`,
        })
      }
      if (failures.length >= 5) break
    }

    expect(failures, JSON.stringify(failures, null, 2)).toEqual([])
  })
})

describe('optimizer join-swap + duplicate build keys', () => {
  it('preserves many-match rows when build side is swapped onto a non-unique key', () => {
    const left = DataFrame.fromRows([
      { id: 6, a: 3, b: 6, x: 1.0, lo: 0, hi: 5, s: 'p' },
      { id: 7, a: 3, b: 3, x: 2.0, lo: 0, hi: 5, s: 'q' },
      { id: 11, a: 2, b: 5, x: 3.0, lo: 0, hi: 5, s: 'r' },
      { id: 99, a: 3, b: 1, x: -1.0, lo: 0, hi: 5, s: 'z' },
    ])
    const right = DataFrame.fromRows([
      { rid: 0, score: 1, tag: 'p' },
      { rid: 1, score: 1, tag: 'p' },
      { rid: 2, score: 1, tag: 'p' },
      { rid: 3, score: 1, tag: 'p' },
    ])

    const plan = left
      .filter(col('a').gte(0))
      .filter(col('b').gte(0))
      .join(right, { leftOn: 'a', rightOn: 'rid' })
      .sort('id')
      .filter(col('a').isBetween(col('lo'), col('hi')))
      .filter(col('a').gt(1))
      .filter(col('x').gt(0))
      .select(col('a').alias('b'), col('b').alias('a'), 'id')
      .sort('id').plan

    const raw = new DataFrame(executeCpuUnoptimized(plan)).toArray()
    const opt = new DataFrame(executeCpuUnoptimized(optimizePlan(plan))).toArray()
    expect(opt).toEqual(raw)
    expect(raw).toEqual([
      { b: 3, a: 6, id: 6 },
      { b: 3, a: 3, id: 7 },
      { b: 2, a: 5, id: 11 },
    ])
  })
})
