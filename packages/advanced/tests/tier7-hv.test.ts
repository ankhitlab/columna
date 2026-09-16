import { describe, expect, it } from 'vitest'
import {
  arima,
  analyzeMixture,
  controlChart,
  coxPH,
  ets,
  ewma,
  gamesHowell,
  glmm,
  manova,
  mixedModel,
  mixtureDesign,
  plotSeries,
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

describe('plotSeries', () => {
  it('exports SPC and ETS/ARIMA series with roles', () => {
    const x = Array.from({ length: 30 }, (_, i) => 10 + 0.1 * Math.sin(i / 3))
    const cc = controlChart(x, { type: 'i-mr' })
    const ps = plotSeries(cc)
    expect(ps.series.some((s) => s.role === 'data')).toBe(true)
    expect(ps.series.some((s) => s.role === 'ucl')).toBe(true)

    const ew = ewma(x)
    expect(plotSeries(ew).series.find((s) => s.name === 'ewma')!.y.length).toBe(30)

    const fit = ets(x, { method: 'ses', horizon: 3 })
    const etsPlot = plotSeries(fit)
    expect(etsPlot.series.some((s) => s.role === 'forecast')).toBe(true)
  })
})

describe('nested LMM + NB GLMM', () => {
  it('nested RI yields positive outer and nested sigma', () => {
    const g = rng(3)
    const y: number[] = []
    const X: number[][] = []
    const outer: string[] = []
    const inner: string[] = []
    for (let o = 0; o < 8; o++) {
      const uO = 1.2 * g.normal()
      for (let n = 0; n < 4; n++) {
        const uN = 0.7 * g.normal()
        for (let i = 0; i < 5; i++) {
          const z = g.normal()
          y.push(1 + 0.5 * z + uO + uN + 0.4 * g.normal())
          X.push([z])
          outer.push(`O${o}`)
          inner.push(`I${n}`)
        }
      }
    }
    const fit = mixedModel(y, { fixed: X, group: outer, groupNested: inner })
    expect(fit.sigmaRandom).toBeGreaterThan(0)
    expect(fit.sigmaNested!).toBeGreaterThan(0)
    expect(fit.nestedRanef!.length).toBeGreaterThan(1)
  })

  it('negbin GLMM estimates theta', () => {
    const g = rng(7)
    const y: number[] = []
    const X: number[][] = []
    const group: string[] = []
    for (let j = 0; j < 12; j++) {
      const u = 0.5 * g.normal()
      for (let i = 0; i < 10; i++) {
        const z = g.u()
        const mu = Math.exp(0.5 + 0.8 * z + u)
        // crude NB-ish: poisson with extra noise
        let k = 0
        const lam = mu * (1 + 0.5 * g.u())
        while (g.u() > Math.exp(-lam) && k < 40) k++
        y.push(k)
        X.push([z])
        group.push(`g${j}`)
      }
    }
    const fit = glmm(y, { family: 'negbin', fixed: X, group })
    expect(fit.family).toBe('negbin')
    expect(fit.theta!).toBeGreaterThan(0)
    expect(Number.isFinite(fit.fixed[1]!.coefficient)).toBe(true)
  })
})

describe('Cox cluster + cause-specific', () => {
  it('cluster SE is finite and robust flag set', () => {
    const g = rng(4)
    const time: number[] = []
    const X: number[][] = []
    const cluster: string[] = []
    for (let c = 0; c < 15; c++) {
      for (let i = 0; i < 6; i++) {
        const x = g.u() > 0.5 ? 1 : 0
        time.push(g.exponential(x ? 25 : 15))
        X.push([x])
        cluster.push(`c${c}`)
      }
    }
    const fit = coxPH(time, X, { cluster })
    expect(fit.robust).toBe(true)
    expect(fit.se[0]!).toBeGreaterThan(0)
    expect(fit.seModel![0]!).toBeGreaterThan(0)
  })

  it('cause-specific reduces events vs all-cause', () => {
    const g = rng(8)
    const time: number[] = []
    const X: number[][] = []
    const eventType: number[] = []
    for (let i = 0; i < 80; i++) {
      const x = i < 40 ? 0 : 1
      time.push(g.exponential(20))
      X.push([x])
      eventType.push(g.u() < 0.55 ? 1 : 2)
    }
    const all = coxPH(time, X)
    const cs = coxPH(time, X, { eventType, cause: 1 })
    expect(cs.causeSpecific).toBe(true)
    expect(cs.nEvents).toBeLessThan(all.nEvents)
  })
})

describe('ARIMA ML + mixture process', () => {
  it('Kalman ML fits AR(1)', () => {
    const g = rng(2)
    const y: number[] = [0]
    for (let i = 1; i < 80; i++) y.push(0.6 * y[i - 1]! + g.normal())
    const fit = arima(y, { p: 1, d: 0, q: 0, method: 'ML', horizon: 2 })
    expect(fit.method).toBe('ML')
    expect(fit.ar[0]!).toBeGreaterThan(0.2)
    expect(fit.ar[0]!).toBeLessThan(0.95)
    expect(Number.isFinite(fit.logLik)).toBe(true)
  })

  it('mixture × process design and analysis', () => {
    const d = mixtureDesign(['A', 'B', 'C'], { type: 'lattice', degree: 2, process: ['Temp'] })
    expect(d.factors).toContain('Temp')
    expect(d.matrix[0]!.length).toBe(4)
    const y = d.matrix.map((r) => 10 * r[0]! + 8 * r[1]! + 6 * r[2]! + 2 * r[3]! + r[0]! * r[3]!)
    const fit = analyzeMixture(d, y, { model: 'linear' })
    expect(fit.nProcess).toBe(1)
    expect(fit.names.some((n) => n.includes('Z'))).toBe(true)
    expect(fit.r2).toBeGreaterThan(0.9)
  })
})

describe('Games-Howell + MANOVA + Taguchi OAs', () => {
  it('gamesHowell returns pairwise comparisons', () => {
    const r = gamesHowell({
      a: [10, 11, 9, 10.5, 12],
      b: [14, 15, 13, 16, 14.5, 15.2],
      c: [10.2, 9.8, 11, 10],
    })
    expect(r.test).toBe('Games-Howell')
    expect(r.comparisons.length).toBe(3)
  })

  it('one-way MANOVA separates groups', () => {
    const Y: number[][] = []
    const group: string[] = []
    for (let g = 0; g < 3; g++) {
      for (let i = 0; i < 20; i++) {
        Y.push([g * 2 + 0.1 * i, g * 1.5 + 0.05 * i])
        group.push(`G${g}`)
      }
    }
    const m = manova(Y, group)
    expect(m.pillai.statistic).toBeGreaterThan(0)
    expect(m.wilks.statistic).toBeLessThan(1)
    expect(m.pillai.pValue).toBeLessThan(0.05)
  })

  it('Taguchi L50/L54/L64 generate designs', () => {
    expect(taguchi('L50').matrix.length).toBe(50)
    expect(taguchi('L54').matrix.length).toBe(54)
    expect(taguchi('L64').matrix.length).toBe(64)
  })
})
