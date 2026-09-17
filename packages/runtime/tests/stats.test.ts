import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import {
  approxNdv,
  estimateFilterSelectivity,
  estimatePlanRows,
  sampleIndices,
} from '../src/stats.js'
import { getColumn } from '@columna/arrow'
import type { PlanNode } from '../src/types.js'
import { explainPlan } from '../src/types.js'
import { optimizePlan } from '../src/optimize.js'

describe('stats helpers', () => {
  it('sampleIndices returns at most maxSample unique indices', () => {
    const idx = sampleIndices(100_000, 64, 1)
    expect(idx.length).toBe(64)
    expect(new Set(idx).size).toBe(64)
  })

  it('filter selectivity ≈ 0.5 for half-positive column', () => {
    const rows = Array.from({ length: 2000 }, (_, i) => ({ x: i - 1000 }))
    const table = DataFrame.fromRows(rows).table
    const pred = { type: 'binary' as const, op: 'gt' as const, left: { type: 'col' as const, name: 'x' }, right: { type: 'lit' as const, value: 0 } }
    const sel = estimateFilterSelectivity(table, pred)
    expect(sel).toBeGreaterThan(0.4)
    expect(sel).toBeLessThan(0.6)
  })

  it('approxNdv is high for unique key', () => {
    const table = DataFrame.fromRows(Array.from({ length: 500 }, (_, i) => ({ id: i }))).table
    const col = getColumn(table, 'id')
    const ndv = approxNdv(col, table.numRows)
    expect(ndv).toBeGreaterThan(400)
  })

  it('inner join estimate shrinks when key is unique on one side', () => {
    const left: PlanNode = {
      type: 'scan',
      table: DataFrame.fromRows(Array.from({ length: 100 }, (_, i) => ({ id: i, a: i }))).table,
    }
    const rightUnique: PlanNode = {
      type: 'scan',
      table: DataFrame.fromRows(Array.from({ length: 100 }, (_, i) => ({ id: i, b: i }))).table,
    }
    const rightDup: PlanNode = {
      type: 'scan',
      table: DataFrame.fromRows(Array.from({ length: 100 }, (_, i) => ({ id: i % 5, b: i }))).table,
    }
    const joinUnique: PlanNode = {
      type: 'join',
      left,
      right: rightUnique,
      leftOn: ['id'],
      rightOn: ['id'],
      how: 'inner',
    }
    const joinDup: PlanNode = {
      type: 'join',
      left,
      right: rightDup,
      leftOn: ['id'],
      rightOn: ['id'],
      how: 'inner',
    }
    expect(estimatePlanRows(joinUnique)).toBeLessThanOrEqual(estimatePlanRows(joinDup))
  })

  it('explainPlan shows rows≈ on filter/join', () => {
    const plan: PlanNode = {
      type: 'filter',
      predicate: {
        type: 'binary',
        op: 'gt',
        left: { type: 'col', name: 'x' },
        right: { type: 'lit', value: 0 },
      },
      input: {
        type: 'scan',
        table: DataFrame.fromRows([
          { x: -1 },
          { x: 1 },
          { x: 2 },
        ]).table,
      },
    }
    const text = explainPlan(optimizePlan(plan))
    expect(text).toMatch(/Filter rows≈\d+/)
  })

  it('limit estimate respects n', () => {
    const plan: PlanNode = {
      type: 'limit',
      n: 3,
      input: {
        type: 'scan',
        table: DataFrame.fromRows(Array.from({ length: 50 }, (_, i) => ({ i }))).table,
      },
    }
    expect(estimatePlanRows(plan)).toBe(3)
  })
})

describe('NDV-aware filter + join build choice', () => {
  it('selective filter makes small side preferred as build', () => {
    // Big table filtered to few rows vs medium unfiltered — after push, build should be the tiny side
    const big = DataFrame.fromRows(
      Array.from({ length: 200 }, (_, i) => ({ id: i % 20, flag: i < 5 ? 1 : 0, a: i })),
    )
    const medium = DataFrame.fromRows(
      Array.from({ length: 40 }, (_, i) => ({ id: i % 20, b: i })),
    )
    const plan: PlanNode = {
      type: 'join',
      left: {
        type: 'filter',
        predicate: {
          type: 'binary',
          op: 'eq',
          left: { type: 'col', name: 'flag' },
          right: { type: 'lit', value: 1 },
        },
        input: { type: 'scan', table: big.table },
      },
      right: { type: 'scan', table: medium.table },
      leftOn: ['id'],
      rightOn: ['id'],
      how: 'inner',
    }
    const out = optimizePlan(plan)
    const findJoin = (p: PlanNode): Extract<PlanNode, { type: 'join' }> | null => {
      if (p.type === 'join') return p
      if ('input' in p && p.input) return findJoin(p.input as PlanNode)
      return null
    }
    const j = findJoin(out)!
    expect(estimatePlanRows(j.right)).toBeLessThanOrEqual(estimatePlanRows(j.left))
  })
})
