import { describe, expect, it } from 'vitest'
import {
  arima,
  coxPH,
  glmm,
  renderPlotSeries,
  taguchi,
  transferIdentify,
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

describe('interactive SVG lite', () => {
  it('adds role attrs, tooltips, legend when interactive', () => {
    const y = Array.from({ length: 24 }, (_, i) => 10 + Math.sin(i / 3))
    const fit = arima(y, { p: 1, d: 0, q: 0, horizon: 4 })
    const svg = renderPlotSeries(fit, { interactive: true, title: 'demo' })
    expect(svg).toContain('data-role=')
    expect(svg).toContain('data-name=')
    expect(svg).toContain('<title>')
    expect(svg).toContain('class="legend"')
    expect(svg).toContain('.plot-layer')
  })
})

describe('AGQ GLMM with random slope', () => {
  it('estimates sigmaSlope under AGQ', () => {
    const g = rng(12)
    const y: number[] = []
    const X: number[][] = []
    const group: string[] = []
    const slope: number[] = []
    for (let j = 0; j < 8; j++) {
      const u = 0.5 * g.normal()
      const s = 0.4 * g.normal()
      for (let i = 0; i < 10; i++) {
        const z = (i - 4.5) / 4
        const p = 1 / (1 + Math.exp(-(0.1 + z + u + s * z)))
        y.push(g.u() < p ? 1 : 0)
        X.push([z])
        group.push(`g${j}`)
        slope.push(z)
      }
    }
    const fit = glmm(y, { family: 'binomial', fixed: X, group, slope, method: 'agq', nAGQ: 5 })
    expect(fit.method).toBe('agq')
    expect(fit.sigmaSlope!).toBeGreaterThan(0)
    expect(fit.rho).toBeDefined()
    expect(Math.abs(fit.rho!)).toBeLessThanOrEqual(1)
    expect(fit.ranef.some((r) => r.slope != null)).toBe(true)
  })
})

describe('AR(1) frailty bands', () => {
  it('returns frailtyRho for process ar1', () => {
    const g = rng(3)
    const time: number[] = []
    const X: number[][] = []
    const group: string[] = []
    const bands: number[] = []
    for (let c = 0; c < 6; c++) {
      for (let b = 0; b < 3; b++) {
        for (let i = 0; i < 4; i++) {
          time.push(g.exponential(18 + b))
          X.push([g.u() > 0.5 ? 1 : 0])
          group.push(`c${c}`)
          bands.push(b)
        }
      }
    }
    const fit = coxPH(time, X, { frailty: { group, bands, process: 'ar1' } })
    expect(fit.timeVaryingFrailty).toBe(true)
    expect(Number.isFinite(fit.frailtyRho!)).toBe(true)
    expect(Math.abs(fit.frailtyRho!)).toBeLessThanOrEqual(1)
    expect(fit.frailtyByBand!.length).toBeGreaterThan(0)
  })
})

describe('transferIdentify', () => {
  it('suggests delay from prewhitened CCF', () => {
    const g = rng(5)
    const x: number[] = []
    const y: number[] = []
    for (let t = 0; t < 80; t++) {
      x.push(0.5 * (x[t - 1] ?? 0) + g.normal())
      const lag = t >= 2 ? x[t - 2]! : 0
      y.push(0.9 * lag + 0.4 * g.normal())
    }
    const id = transferIdentify(y, x, { maxDelay: 8 })
    expect(id.delay).toBeGreaterThanOrEqual(0)
    expect(id.omega).toBeGreaterThanOrEqual(0)
    expect(id.omega).toBeLessThanOrEqual(2)
    expect(id.delta).toBeLessThanOrEqual(2)
    expect(id.ccf.lag.length).toBeGreaterThan(0)
    const fit = arima(y, {
      p: 0,
      d: 0,
      q: 1,
      method: 'CSS-ML',
      transfer: [{ x, delay: id.delay, omega: id.omega, delta: id.delta }],
    })
    expect(fit.xregCoef!.length).toBeGreaterThan(0)
  })
})

describe('Taguchi rare OAs', () => {
  it('generates L20/L24/L28/L40/L44/L48', () => {
    expect(taguchi('L20').matrix.length).toBe(20)
    expect(taguchi('L24').matrix.length).toBe(24)
    expect(taguchi('L28').matrix.length).toBe(28)
    expect(taguchi('L40').matrix.length).toBe(40)
    expect(taguchi('L44').matrix.length).toBe(44)
    expect(taguchi('L48').matrix.length).toBe(48)
  })
})
