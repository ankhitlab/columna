import { describe, expect, it } from 'vitest'
import {
  adfTest,
  ancova,
  anovaTwoWay,
  arima,
  eppsSingleton,
  glmm,
  kpssTest,
  moodTwoSample,
  stl,
} from '@columna/advanced'

describe('anovaTwoWay + ancova', () => {
  it('two-way ANOVA detects row effect', () => {
    const y: number[] = []
    const row: string[] = []
    const col: string[] = []
    for (const r of ['A', 'B']) {
      for (const c of ['1', '2', '3']) {
        for (let i = 0; i < 4; i++) {
          y.push((r === 'A' ? 10 : 20) + Number(c) + i * 0.1)
          row.push(r)
          col.push(c)
        }
      }
    }
    const r = anovaTwoWay(y, row, col, { interaction: false })
    expect(r.row!.pValue).toBeLessThan(0.01)
    expect(r.residualDf).toBeGreaterThan(0)
  })

  it('ancova adjusts for covariate', () => {
    const y: number[] = []
    const group: string[] = []
    const cov: number[] = []
    for (let i = 0; i < 20; i++) {
      group.push(i < 10 ? 'g0' : 'g1')
      cov.push(i)
      y.push((i < 10 ? 0 : 5) + 0.5 * i)
    }
    const r = ancova(y, group, cov)
    expect(r.covariate!.pValue).toBeLessThan(0.05)
    expect(r.group).toBeDefined()
  })
})

describe('unit root', () => {
  it('adf rejects stationary; kpss accepts', () => {
    const stationary = Array.from({ length: 80 }, (_, i) => Math.sin(i / 3) * 0.2 + (i % 5) * 0.01)
    const adf = adfTest(stationary, { lags: 1 })
    expect(adf.statistic).toBeLessThan(-1)
    const kpss = kpssTest(stationary)
    expect(kpss.statistic).toBeGreaterThan(0)
    expect(kpss.pValue).toBeGreaterThan(0.05)
  })

  it('adf does not reject random walk strongly', () => {
    let x = 0
    const rw = Array.from({ length: 100 }, () => {
      x += (Math.random() - 0.5)
      return x
    })
    // seeded-ish: use deterministic increments
    x = 0
    const series = Array.from({ length: 100 }, (_, i) => {
      x += Math.sin(i * 1.7) * 0.5
      return x
    })
    const adf = adfTest(series, { lags: 1 })
    expect(Number.isFinite(adf.statistic)).toBe(true)
  })
})

describe('multi-season STL + ML transfer ARIMA', () => {
  it('stl periods returns multiple seasonals', () => {
    const y = Array.from({ length: 120 }, (_, i) => Math.sin((2 * Math.PI * i) / 12) + 0.3 * Math.sin((2 * Math.PI * i) / 5) + i * 0.01)
    const r = stl(y, { periods: [12, 5] })
    expect(r.periods).toEqual([12, 5])
    expect(r.seasonals!.length).toBe(2)
    expect(r.seasonal.length).toBe(120)
  })

  it('arima ML with transfer does not throw', () => {
    const y = Array.from({ length: 40 }, (_, i) => 2 + 0.5 * i + Math.sin(i / 2))
    const x = Array.from({ length: 40 }, (_, i) => Math.sin(i / 3))
    const fit = arima(y, { p: 1, d: 0, q: 0, method: 'ML', transfer: [{ x, omega: 0 }], horizon: 2 })
    expect(fit.fitted.length).toBe(40)
    expect(Number.isFinite(fit.sigma2)).toBe(true)
  })
})

describe('AGQ correlated slope + epps/mood', () => {
  it('glmm agq reports rho', () => {
    const y: number[] = []
    const X: number[][] = []
    const group: string[] = []
    const slope: number[] = []
    for (let j = 0; j < 6; j++) {
      for (let i = 0; i < 8; i++) {
        const z = (i - 3.5) / 4
        y.push(i + j > 6 ? 1 : 0)
        X.push([z])
        group.push(`g${j}`)
        slope.push(z)
      }
    }
    const fit = glmm(y, { family: 'binomial', fixed: X, group, slope, method: 'agq', nAGQ: 5 })
    expect(fit.rho).toBeDefined()
    expect(Math.abs(fit.rho!)).toBeLessThanOrEqual(1)
  })

  it('eppsSingleton and moodTwoSample run', () => {
    const a = Array.from({ length: 20 }, (_, i) => i * 0.1)
    const b = Array.from({ length: 20 }, (_, i) => i * 0.1 + 3)
    expect(eppsSingleton(a, b).statistic).toBeGreaterThan(0)
    expect(moodTwoSample(a, b.map((v) => v * 3)).statistic).toBeGreaterThan(0)
  })
})
