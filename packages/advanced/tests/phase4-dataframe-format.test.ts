import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import {
  ADVANCED_INSTALLED,
  adfTest,
  ancova,
  anova,
  formatReport,
  kpssTest,
  ksTwoSample,
  lasso,
  ols,
  propTest1,
  propTest2,
  ridge,
  ttest1,
} from '@columna/advanced'

void ADVANCED_INSTALLED

const close = (got: number, want: number, digits = 10) => expect(got).toBeCloseTo(want, digits)

describe('Phase 4 DataFrame wrappers', () => {
  it('propTest: one-sample (summed columns) and two-sample by group', () => {
    const one = DataFrame.fromRows([
      { hits: 3, n: 10 },
      { hits: 2, n: 10 },
    ])
    const r1 = one.propTest('hits', { trials: 'n', p0: 0.25 })
    const ref1 = propTest1(5, 20, { p0: 0.25 })
    expect(r1.test).toBe('1 proportion')
    close(r1.pValue, ref1.pValue)

    const two = DataFrame.fromRows([
      { g: 'A', hits: 1, n: 10 },
      { g: 'A', hits: 0, n: 10 },
      { g: 'B', hits: 4, n: 10 },
      { g: 'B', hits: 3, n: 10 },
    ])
    const r2 = two.propTest('hits', { trials: 'n', by: 'g' })
    const ref2 = propTest2(1, 20, 7, 20)
    close(r2.pValue, ref2.pValue)

    const binary = DataFrame.fromRows([
      { g: 'x', y: 1 },
      { g: 'x', y: 0 },
      { g: 'y', y: 1 },
      { g: 'y', y: 1 },
    ])
    const rb = binary.propTest('y', { by: 'g' })
    close(rb.pValue, propTest2(1, 2, 2, 2).pValue)
  })

  it('adfTest / kpssTest / ancova / ksTwoSample / ridge / lasso match array APIs', async () => {
    const y = [1.2, 1.5, 1.1, 1.4, 1.3, 1.6, 1.2, 1.5, 1.4, 1.3, 1.2, 1.1]
    const group = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b']
    const cov = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    const df = DataFrame.fromRows(y.map((v, i) => ({ v, group: group[i], cov: cov[i], x1: i * 0.1, x2: i * 0.2 })))

    close(df.adfTest('v', { lags: 1 }).statistic, adfTest(y, { lags: 1 }).statistic)
    close(df.kpssTest('v').statistic, kpssTest(y).statistic)
    close((await df.lazy().adfTest('v', { lags: 1 })).pValue, adfTest(y, { lags: 1 }).pValue)

    close(df.ancova('v', 'group', 'cov').residualDf, ancova(y, group, cov).residualDf)

    const ks = df.ksTwoSample('v', 'group')
    const ga = y.filter((_, i) => group[i] === 'a')
    const gb = y.filter((_, i) => group[i] === 'b')
    close(ks.statistic, ksTwoSample(ga, gb).statistic)

    const yy = [1, 2, 3, 4, 5]
    const X = [
      [1, 0],
      [1, 1],
      [1, 2],
      [1, 3],
      [1, 4],
    ]
    const reg = DataFrame.fromRows(yy.map((v, i) => ({ y: v, x1: X[i]![1]! })))
    close(reg.ridge('y', ['x1'], { alpha: 1 }).intercept, ridge(yy, X.map((r) => [r[1]!]), { alpha: 1 }).intercept, 6)
    close(reg.lasso('y', ['x1'], { alpha: 0.01 }).coef[0]!, lasso(yy, X.map((r) => [r[1]!]), { alpha: 0.01 }).coef[0]!, 5)
  })
})

describe('formatReport markdown goldens', () => {
  const x = [9.5, 10.2, 8.8, 11.1, 10.0, 9.7, 10.5, 10.1, 9.9, 10.3]
  const t = ttest1(x, { mu: 10 })
  const a = anova({ a: [1, 2, 3], b: [4, 5, 6], c: [2, 3, 4] })
  const r = ols([1, 2, 3, 4, 5], [[0, 1, 2, 3, 4]], { names: ['x1'] })

  it('t-test report contains title, stats, and sample table', () => {
    const md = formatReport(t, 'markdown')
    expect(md).toMatch(/^## one-sample t/m)
    expect(md).toContain('**p-value:**')
    expect(md).toContain('| n | Mean | SD |')
    expect(md).toContain(`| ${t.n} |`)
  })

  it('ANOVA report contains F, groups table', () => {
    const md = formatReport(a, 'markdown')
    expect(md).toContain('## one-way ANOVA')
    expect(md).toContain('**F(2, 6):**')
    expect(md).toContain('| Group | n | Mean | SD |')
    expect(md).toContain('| a |')
  })

  it('OLS report contains coefficients and ANOVA blocks', () => {
    const md = formatReport(r, 'markdown')
    expect(md).toContain('## Regression')
    expect(md).toContain('### Coefficients')
    expect(md).toContain('| Term | Coef | SE | t | p |')
    expect(md).toContain('Constant')
    expect(md).toContain('x1')
    expect(md).toContain('### ANOVA')
  })

  it('HTML escapes and wraps tables', () => {
    const html = formatReport(t, 'html')
    expect(html).toContain('<h2>')
    expect(html).toContain('<table>')
    expect(html).not.toContain('<script>')
  })
})
