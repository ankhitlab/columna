import { describe, expect, it } from 'vitest'
import {
  arima,
  capabilitySixpack,
  coxPH,
  fineGray,
  glmm,
  mixedModel,
  nestedAnova,
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

describe('capabilitySixpack', () => {
  it('returns histogram, QQ, SPC, and capability panels', () => {
    const x = Array.from({ length: 50 }, (_, i) => 10 + 0.3 * Math.sin(i / 4) + (i % 7) * 0.05)
    const r = capabilitySixpack(x, { lsl: 9, usl: 12 })
    expect(r.panels.histogram.length).toBeGreaterThan(0)
    expect(r.panels.normalPlot.some((s) => s.name === 'qq')).toBe(true)
    expect(r.panels.controlChart.some((s) => s.role === 'data')).toBe(true)
    expect(r.capability.Cpk).toBeDefined()
    expect(r.series.length).toBeGreaterThan(5)
  })
})

describe('crossed LMM + Laplace GLMM', () => {
  it('crossed group2 yields sigmaRandom2', () => {
    const g = rng(3)
    const y: number[] = []
    const X: number[][] = []
    const a: string[] = []
    const b: string[] = []
    for (let i = 0; i < 8; i++) {
      const uA = 1.2 * g.normal()
      for (let j = 0; j < 6; j++) {
        const uB = 0.8 * g.normal()
        for (let k = 0; k < 4; k++) {
          const z = g.normal()
          y.push(1 + 0.4 * z + uA + uB + 0.5 * g.normal())
          X.push([z])
          a.push(`A${i}`)
          b.push(`B${j}`)
        }
      }
    }
    const fit = mixedModel(y, { fixed: X, group: a, group2: b })
    expect(fit.sigmaRandom).toBeGreaterThan(0)
    expect(fit.sigmaRandom2!).toBeGreaterThan(0)
    expect(fit.ranef2!.length).toBe(6)
  })

  it('laplace GLMM returns method laplace', () => {
    const g = rng(5)
    const y: number[] = []
    const X: number[][] = []
    const group: string[] = []
    for (let j = 0; j < 10; j++) {
      const u = 0.6 * g.normal()
      for (let i = 0; i < 12; i++) {
        const z = g.u()
        const p = 1 / (1 + Math.exp(-(0.2 + z + u)))
        y.push(g.u() < p ? 1 : 0)
        X.push([z])
        group.push(`g${j}`)
      }
    }
    const fit = glmm(y, { family: 'binomial', fixed: X, group, method: 'laplace' })
    expect(fit.method).toBe('laplace')
    expect(Number.isFinite(fit.logLik)).toBe(true)
  })
})

describe('Fine–Gray + TV frailty flag', () => {
  it('fineGray fits and returns CIF', () => {
    const g = rng(9)
    const time: number[] = []
    const X: number[][] = []
    const eventType: number[] = []
    for (let i = 0; i < 80; i++) {
      const x = i < 40 ? 0 : 1
      time.push(g.exponential(x ? 25 : 18))
      X.push([x])
      eventType.push(g.u() < 0.5 ? 1 : 2)
    }
    const fit = fineGray(time, X, { eventType, cause: 1 })
    expect(fit.nEvents).toBeGreaterThan(5)
    expect(fit.cif.length).toBeGreaterThan(0)
    expect(Number.isFinite(fit.coefficients[0]!)).toBe(true)
  })

  it('coxPH marks timeVaryingFrailty when start+frailty', () => {
    const g = rng(2)
    const time: number[] = []
    const start: number[] = []
    const X: number[][] = []
    const group: string[] = []
    for (let c = 0; c < 8; c++) {
      for (let i = 0; i < 6; i++) {
        const t = g.exponential(20)
        time.push(t)
        start.push(0)
        X.push([g.u() > 0.5 ? 1 : 0])
        group.push(`c${c}`)
      }
    }
    const fit = coxPH(time, X, { start, frailty: { group } })
    expect(fit.timeVaryingFrailty).toBe(true)
    expect(fit.frailtyVar!).toBeGreaterThan(0)
  })
})

describe('seasonal ML + transfer ARIMA', () => {
  it('seasonal ML fits without throwing', () => {
    const g = rng(1)
    const y: number[] = []
    for (let t = 0; t < 48; t++) y.push(10 + 2 * Math.sin((2 * Math.PI * t) / 12) + 0.5 * g.normal())
    const fit = arima(y, { p: 1, d: 0, q: 0, method: 'ML', seasonal: { P: 1, D: 0, Q: 0, period: 12 } })
    expect(fit.method).toBe('ML')
    expect(Number.isFinite(fit.logLik)).toBe(true)
  })

  it('transfer function expands regressors', () => {
    const g = rng(4)
    const x: number[] = []
    const y: number[] = []
    for (let t = 0; t < 60; t++) {
      x.push(g.normal())
      const lag = t >= 1 ? x[t - 1]! : 0
      y.push(0.5 * lag + 0.3 * g.normal())
    }
    const fit = arima(y, { p: 0, d: 0, q: 1, method: 'CSS-ML', transfer: [{ x, delay: 0, omega: 1 }] })
    expect(fit.xregCoef!.length).toBeGreaterThanOrEqual(2)
  })
})

describe('Taguchi L25/L81 + nestedAnova', () => {
  it('generates L25 and L81', () => {
    expect(taguchi('L25').matrix.length).toBe(25)
    expect(taguchi('L81').matrix.length).toBe(81)
  })

  it('nestedAnova returns variance components', () => {
    const g = rng(6)
    const y: number[] = []
    const A: string[] = []
    const B: string[] = []
    for (let a = 0; a < 4; a++) {
      const uA = 2 * g.normal()
      for (let b = 0; b < 3; b++) {
        const uB = g.normal()
        for (let i = 0; i < 5; i++) {
          y.push(uA + uB + 0.5 * g.normal())
          A.push(`A${a}`)
          B.push(`B${b}`)
        }
      }
    }
    const r = nestedAnova(y, [A, B])
    expect(r.components.length).toBe(2)
    expect(r.residual.df).toBeGreaterThan(0)
    expect(r.totalVar).toBeGreaterThan(0)
  })
})
