import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import { pushdownProjections } from '../src/pushdown.js'
import { createFilterView, isFilterView, materializeView, ensureMaterialized } from '../src/views.js'
import type { PlanNode } from '../src/types.js'

describe('select shallow columns', () => {
  it('reuses typed-array buffers for plain column select', async () => {
    const df = DataFrame.fromRows([
      { a: 1, b: 2, c: 3 },
      { a: 4, b: 5, c: 6 },
    ])
    const src = df.table.columns[0]!.data as Float64Array | Int32Array
    const out = await df.select('a').collect()
    const dst = out.table.columns[0]!.data as Float64Array | Int32Array
    expect(dst.buffer).toBe(src.buffer)
  })
})

describe('filter views', () => {
  it('shares base buffers until materialize', () => {
    const base = DataFrame.fromRows([
      { a: 1 },
      { a: 2 },
      { a: 3 },
      { a: 4 },
    ]).table
    const view = createFilterView(base, new Uint32Array([0, 2]))
    expect(isFilterView(view)).toBe(true)
    expect(view.numRows).toBe(2)
    expect(view.columns[0]!.data).toBe(base.columns[0]!.data)
    const dense = materializeView(view)
    expect(isFilterView(dense)).toBe(false)
    expect(dense.numRows).toBe(2)
    expect([...(dense.columns[0]!.data as Float64Array | Int32Array)]).toEqual([1, 3])
  })

  it('filter+collect returns dense rows matching predicate', async () => {
    const df = DataFrame.fromRows([
      { a: 1, b: 10 },
      { a: 2, b: 20 },
      { a: 3, b: 30 },
    ])
    const out = await df.filter((c) => c.a.gt(1)).collect()
    expect(out.table.numRows).toBe(2)
    expect(out.table.columns[0]!.data.length).toBe(2)
  })
})

describe('projection pushdown', () => {
  it('pushes project below sort', () => {
    const scan: PlanNode = {
      type: 'scan',
      table: DataFrame.fromRows([{ a: 1, b: 2, c: 3 }]).table,
    }
    const plan: PlanNode = {
      type: 'project',
      columns: ['a'],
      input: {
        type: 'sort',
        input: scan,
        by: [{ expr: { type: 'col', name: 'a' }, descending: false }],
      },
    }
    const out = pushdownProjections(plan)
    expect(out.type).toBe('project')
    if (out.type === 'project') {
      expect(out.columns).toEqual(['a'])
      expect(out.input.type).toBe('sort')
      if (out.input.type === 'sort') {
        expect(out.input.input.type).toBe('project')
      }
    }
  })

  it('prunes join sides to projection + keys', () => {
    const left: PlanNode = {
      type: 'scan',
      table: DataFrame.fromRows([{ id: 1, a: 10, x: 100 }]).table,
    }
    const right: PlanNode = {
      type: 'scan',
      table: DataFrame.fromRows([{ id: 1, b: 11, y: 111 }]).table,
    }
    const plan: PlanNode = {
      type: 'project',
      columns: ['id', 'a', 'b'],
      input: {
        type: 'join',
        left,
        right,
        leftOn: ['id'],
        rightOn: ['id'],
        how: 'inner',
      },
    }
    const out = pushdownProjections(plan)
    expect(out.type).toBe('project')
    if (out.type === 'project' && out.input.type === 'join') {
      expect(out.input.left.type).toBe('project')
      expect(out.input.right.type).toBe('project')
      if (out.input.left.type === 'project') {
        expect(out.input.left.columns).toEqual(expect.arrayContaining(['id', 'a']))
        expect(out.input.left.columns).not.toContain('x')
      }
      if (out.input.right.type === 'project') {
        expect(out.input.right.columns).toEqual(expect.arrayContaining(['id', 'b']))
        expect(out.input.right.columns).not.toContain('y')
      }
    }
  })
})
