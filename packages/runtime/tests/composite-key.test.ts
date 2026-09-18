import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import { encodeCompositeKey } from '../src/composite_key.js'

describe('encodeCompositeKey', () => {
  it('distinguishes null from the literal empty-marker string', () => {
    expect(encodeCompositeKey([null, 'x'])).not.toBe(encodeCompositeKey(['∅', 'x']))
  })

  it('distinguishes embedded delimiters across part boundaries', () => {
    expect(encodeCompositeKey(['a\0b', 'c'])).not.toBe(encodeCompositeKey(['a', 'b\0c']))
  })

  it('join does not conflate null with literal ∅', async () => {
    const left = DataFrame.fromRows([
      { k1: null, k2: 'x', v: 1 },
      { k1: '∅', k2: 'x', v: 2 },
    ])
    const right = DataFrame.fromRows([
      { k1: null, k2: 'x', r: 10 },
      { k1: '∅', k2: 'x', r: 20 },
    ])
    const out = await left.join(right, { on: ['k1', 'k2'] }).collect()
    expect(out.toArray()).toEqual([
      { k1: null, k2: 'x', v: 1, r: 10 },
      { k1: '∅', k2: 'x', v: 2, r: 20 },
    ])
  })

  it('unique does not conflate delimiter-bearing parts', async () => {
    const out = await DataFrame.fromRows([
      { a: 'a\0b', b: 'c', v: 1 },
      { a: 'a', b: 'b\0c', v: 2 },
    ])
      .unique(['a', 'b'])
      .collect()

    expect(out.shape[0]).toBe(2)
  })
})
