/**
 * simpson / trapz: scipy.integrate references (tests/refs/quadrature_ref.py) plus polynomial exactness —
 * the property that separates Simpson from the trapezoid and that the previous implementation (which
 * silently became `trapz` whenever `x` was given) failed.
 */
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { simpson, trapz } from '../src/index.js'

const ref = JSON.parse(readFileSync(new URL('./fixtures/quadrature-scipy.json', import.meta.url), 'utf8')) as {
  cases: Array<{ name: string; y: number[]; x: number[] | null; dx: number | null; simpson: number; trapz: number }>
}

const close = (got: number, want: number, rel = 1e-12) =>
  expect(Math.abs(got - want)).toBeLessThanOrEqual(rel * Math.max(1, Math.abs(want)))

describe('simpson / trapz vs scipy.integrate', () => {
  for (const c of ref.cases) {
    it(c.name, () => {
      const xArg = c.x ?? c.dx ?? undefined
      close(simpson(c.y, xArg), c.simpson)
      close(trapz(c.y, xArg), c.trapz)
    })
  }
})

describe('polynomial exactness', () => {
  const poly = (k: number) => (t: number) => t ** k
  const exact = (k: number, a: number, b: number) => (b ** (k + 1) - a ** (k + 1)) / (k + 1)
  const grids: Record<string, (n: number) => number[]> = {
    uniform: (n) => Array.from({ length: n }, (_, i) => -1 + (3 * i) / (n - 1)),
    // deliberately very uneven: widths vary by a factor of ~20
    uneven: (n) => Array.from({ length: n }, (_, i) => -1 + 3 * ((i / (n - 1)) ** 2 * 0.7 + (i / (n - 1)) * 0.3)),
  }
  for (const [gridName, grid] of Object.entries(grids)) {
    // equal widths: the parabola rule is also exact for cubics (symmetry); unequal widths: up to quadratics
    const maxDeg = gridName === 'uniform' ? 3 : 2
    it(`odd sample count (${gridName}): exact up to degree ${maxDeg}`, () => {
      for (const n of [3, 5, 9, 21]) {
        const x = grid(n)
        for (let k = 0; k <= maxDeg; k++) close(simpson(x.map(poly(k)), x), exact(k, x[0]!, x[n - 1]!), 1e-12)
      }
    })
    it(`even sample count (${gridName}): ∫1, ∫x, ∫x² exact (Cartwright last interval)`, () => {
      for (const n of [4, 6, 10, 20]) {
        const x = grid(n)
        for (const k of [0, 1, 2]) close(simpson(x.map(poly(k)), x), exact(k, x[0]!, x[n - 1]!), 1e-12)
      }
    })
  }

  it('uniform spacing via dx: exact for cubics, and the trapezoid is not', () => {
    const y = [0, 1, 8, 27, 64] // x³ at x = 0..4
    expect(simpson(y)).toBeCloseTo(64, 12)
    expect(simpson(y.map((v) => v), 0.5)).toBeCloseTo(32, 12) // same samples read as x = 0, .5, …, 2 scaled
    expect(Math.abs(trapz(y) - 64)).toBeGreaterThan(1)
  })

  it('convergence order: halving h divides the error by ~16 (fourth order) for a smooth non-polynomial integrand', () => {
    const err = (n: number) => {
      const x = Array.from({ length: n }, (_, i) => (2 * i) / (n - 1))
      return Math.abs(simpson(x.map(Math.exp), x) - (Math.exp(2) - 1)) // ∫₀² eᵗ dt
    }
    const ratio = err(9) / err(17)
    expect(ratio).toBeGreaterThan(12)
    expect(ratio).toBeLessThan(20)
  })

  it('decreasing x integrates with the sign of the direction', () => {
    const x = [0, 0.4, 1, 1.9, 3]
    const y = x.map((t) => t * t)
    close(simpson([...y].reverse(), [...x].reverse()), -simpson(y, x))
  })
})

describe('input validation', () => {
  it('rejects mismatched lengths, repeated / non-monotonic x and a zero dx', () => {
    expect(() => simpson([1, 2, 3], [0, 1])).toThrow(/x has 2 samples, y has 3/)
    expect(() => simpson([1, 2, 3], [0, 1, 1])).toThrow(/strictly monotonic/)
    expect(() => simpson([1, 2, 3, 4], [0, 2, 1, 3])).toThrow(/strictly monotonic/)
    expect(() => simpson([1, 2, 3], [0, NaN, 1])).toThrow(/strictly monotonic/)
    expect(() => simpson([1, 2, 3], 0)).toThrow(/dx/)
    expect(() => trapz([1, 2, 3], [0, 1])).toThrow(/x has 2 samples/)
  })
  it('degenerate sizes: 0 or 1 samples integrate to 0; 2 samples are the trapezoid', () => {
    expect(simpson([])).toBe(0)
    expect(simpson([5])).toBe(0)
    expect(simpson([1, 3], [0, 2])).toBe(4)
  })
})
