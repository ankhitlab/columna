import { describe, expect, it } from 'vitest'
import { hclust, individualDistributionID, johnsonFit, multiVari, parseFormula, plackettBurman, random, trendAnalysis } from '@columna/advanced'

describe('robustness: hangs, exponential expansions, non-finite input, large arrays', () => {
  it('plackettBurman rejects non-array input instead of looping forever on NaN', () => {
    expect(() => plackettBurman(3 as never)).toThrow(RangeError)
    expect(() => plackettBurman([])).toThrow(RangeError)
    expect(plackettBurman(['A', 'B', 'C']).info.runs).toBe(4)
    expect(plackettBurman(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']).info.runs).toBe(12)
  })

  it('parseFormula expands a*b*c in one pass and caps the number of crossed factors', () => {
    const t0 = performance.now()
    const f = parseFormula('y ~ ' + 'abcdefghijkl'.split('').join('*'))
    expect(f.terms.length).toBe(2 ** 12 - 1)
    expect(performance.now() - t0).toBeLessThan(2000)
    expect(f.terms.slice(0, 12).every((t) => t.vars.length === 1)).toBe(true)
    expect(f.terms[12]!.vars).toEqual(['a', 'b'])
    expect(() => parseFormula('y ~ ' + 'abcdefghijklm'.split('').join('*'))).toThrow(RangeError)
  })

  it('hclust rejects NaN / ragged rows with a clear error', () => {
    expect(() => hclust([[1], [NaN], [3]])).toThrow(/non-finite/)
    expect(() => hclust([[1, 2], [3], [4, 5]])).toThrow(/expected 2/)
  })

  it('min / max over 300k values do not overflow the argument stack', () => {
    const rng = random(5)
    const x = rng.normal(300_000, 50, 10)
    expect(() => trendAnalysis(x, { model: 's-curve' })).not.toThrow()
    expect(() => johnsonFit(x)).not.toThrow()
    expect(() => individualDistributionID(x)).not.toThrow()
    const y = rng.normal(150_000)
    const f = Array.from({ length: 150_000 }, (_, i) => (i % 2 ? 'a' : 'b'))
    expect(() => multiVari(y, [f])).not.toThrow()
  })
})
