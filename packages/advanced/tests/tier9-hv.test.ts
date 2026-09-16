import { describe, expect, it } from 'vitest'
import {
  arima,
  coxPH,
  glmm,
  mixedModel,
  plotSeries,
  renderPlotSeries,
  taguchi,
} from '@columna/advanced'

function rng(seed: number) {
  let a = seed >>> 0
  const u = () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296
  }
  const normal = () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u())
  const exponential = (mean: number) => -mean * Math.log(u())
  return { u, normal, exponential }
}

describe('renderPlotSeries SVG', () => {
  it('renders static SVG over plotSeries layers', () => {
    const y = Array.from({ length: 30 }, (_, i) => 10 + Math.sin(i / 3))
    const fit = arima(y, { p: 1, d: 0, q: 0, horizon: 5 })
    const series = plotSeries(fit)
    const svg = renderPlotSeries(series, { title: 'ARIMA' })
    expect(svg.startsWith('<svg')).toBe(true)
    expect(svg).toContain('<polyline')
    expect(svg).toContain('</svg>')
  })
})

describe('AGQ GLMM + correlated RI+RS', () => {
  it('estimates rho for RI+RS', () => {
    const g = rng(11)
    const y: number[] = []
    const X: number[][] = []
    const group: string[] = []
    const slope: number[] = []
    for (let j = 0; j < 12; j++) {
      const u = g.normal()
      const s = 0.5 * u + 0.3 * g.normal()
      for (let i = 0; i < 8; i++) {
        const z = (i - 3.5) / 3
        y.push(1 + 0.4 * z + u + s * z + 0.4 * g.normal())
        X.push([z])
        group.push(`g${j}`)
        slope.push(z)
      }
    }
    const fit = mixedModel(y, { fixed: X, group, slope })
    expect(fit.sigmaSlope!).toBeGreaterThan(0)
    expect(Number.isFinite(fit.rho!)).toBe(true)
    expect(Math.abs(fit.rho!)).toBeLessThanOrEqual(1)
  })

  it('AGQ GLMM returns method agq', () => {
    const g = rng(7)
    const y: number[] = []
    const X: number[][] = []
    const group: string[] = []
    for (let j = 0; j < 8; j++) {
      const u = 0.5 * g.normal()
      for (let i = 0; i < 10; i++) {
        const z = g.u()
        const p = 1 / (1 + Math.exp(-(0.1 + z + u)))
        y.push(g.u() < p ? 1 : 0)
        X.push([z])
        group.push(`g${j}`)
      }
    }
    const fit = glmm(y, { family: 'binomial', fixed: X, group, method: 'agq', nAGQ: 5 })
    expect(fit.method).toBe('agq')
    expect(fit.nAGQ).toBe(5)
    expect(Number.isFinite(fit.logLik)).toBe(true)
  })
})

describe('piecewise TV frailty bands', () => {
  it('coxPH frailty bands yields frailtyByBand', () => {
    const g = rng(3)
    const time: number[] = []
    const X: number[][] = []
    const group: string[] = []
    const bands: string[] = []
    for (let c = 0; c < 6; c++) {
      for (let i = 0; i < 8; i++) {
        time.push(g.exponential(20))
        X.push([g.u() > 0.5 ? 1 : 0])
        group.push(`c${c}`)
        bands.push(i < 4 ? 'early' : 'late')
      }
    }
    const fit = coxPH(time, X, { frailty: { group, bands } })
    expect(fit.timeVaryingFrailty).toBe(true)
    expect(fit.frailtyByBand!.length).toBeGreaterThan(0)
    expect(fit.frailtyVar!).toBeGreaterThan(0)
  })
})

describe('transfer delta + ML with xreg', () => {
  it('transfer with delta produces one filtered regressor', () => {
    const g = rng(4)
    const x: number[] = []
    const y: number[] = []
    let v = 0
    for (let t = 0; t < 70; t++) {
      x.push(g.normal())
      const u = t >= 1 ? x[t - 1]! : 0
      v = u + 0.4 * v
      y.push(0.8 * v + 0.4 * g.normal())
    }
    const fit = arima(y, {
      p: 0,
      d: 0,
      q: 1,
      method: 'CSS-ML',
      transfer: [{ x, delay: 1, omega: 0, delta: 1 }],
    })
    expect(fit.xregCoef!.length).toBe(1)
    expect(Number.isFinite(fit.xregCoef![0]!)).toBe(true)
  })

  it('ML with xreg includes regression mean', () => {
    const g = rng(8)
    const x: number[][] = []
    const y: number[] = []
    for (let t = 0; t < 50; t++) {
      const z = g.normal()
      x.push([z])
      y.push(2 + 1.5 * z + 0.5 * g.normal())
    }
    const fit = arima(y, { p: 1, d: 0, q: 0, method: 'ML', xreg: x })
    expect(fit.method).toBe('ML')
    expect(fit.xregCoef!.length).toBe(1)
    expect(Number.isFinite(fit.logLik)).toBe(true)
  })
})

describe('Taguchi L108+ OAs', () => {
  it('generates L108/L121/L128/L243', () => {
    expect(taguchi('L108').matrix.length).toBe(108)
    expect(taguchi('L121').matrix.length).toBe(121)
    expect(taguchi('L128').matrix.length).toBe(128)
    expect(taguchi('L243').matrix.length).toBe(243)
    expect(taguchi('L121', ['A', 'B']).factors).toEqual(['A', 'B'])
  })
})
