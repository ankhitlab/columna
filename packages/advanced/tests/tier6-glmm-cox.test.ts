import { describe, expect, it } from 'vitest'
import { coxPH, glmm, mixedModel } from '@columna/advanced'

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

describe('Cox extensions', () => {
  it('strata + counting-process agree with baseline Cox', () => {
    const g = rng(5)
    const time: number[] = []
    const X: number[][] = []
    const censor: number[] = []
    for (let i = 0; i < 60; i++) {
      const x = i < 30 ? 0 : 1
      const t = g.exponential(x === 0 ? 20 : 40)
      time.push(t)
      X.push([x])
      censor.push(0)
    }
    const base = coxPH(time, X, { censor })
    const strata = time.map((_, i) => (i % 2 === 0 ? 'a' : 'b'))
    const strat = coxPH(time, X, { censor, strata })
    expect(strat.stratified).toBe(true)
    expect(Number.isFinite(strat.coefficients[0]!)).toBe(true)

    // counting process: start=0, stop=time → same as standard
    const start = time.map(() => 0)
    const cp = coxPH(time, X, { censor, start })
    expect(Math.abs(cp.coefficients[0]! - base.coefficients[0]!)).toBeLessThan(0.15)
  })

  it('shared frailty yields positive frailty variance', () => {
    const g = rng(9)
    const time: number[] = []
    const X: number[][] = []
    const group: string[] = []
    for (let c = 0; c < 12; c++) {
      const frailty = Math.exp(0.8 * g.normal())
      for (let i = 0; i < 8; i++) {
        const x = g.u() > 0.5 ? 1 : 0
        time.push(g.exponential(30 / frailty) * (x === 1 ? 1.4 : 1))
        X.push([x])
        group.push(`c${c}`)
      }
    }
    const fit = coxPH(time, X, { frailty: { group } })
    expect(fit.frailtyVar!).toBeGreaterThan(0)
    expect(fit.frailty!.length).toBe(12)
  })
})

describe('LMM random slope', () => {
  it('recovers slope BLUPs correlated with truth', () => {
    const g = rng(11)
    const y: number[] = []
    const X: number[][] = []
    const group: string[] = []
    const slope: number[] = []
    const trueS: number[] = []
    for (let j = 0; j < 18; j++) {
      const u0 = 1.5 * g.normal()
      const u1 = 0.8 * g.normal()
      trueS.push(u1)
      for (let i = 0; i < 8; i++) {
        const z = g.normal()
        y.push(0.5 + 0.3 * z + u0 + u1 * z + 0.5 * g.normal())
        X.push([z])
        slope.push(z)
        group.push(`g${j}`)
      }
    }
    const fit = mixedModel(y, { fixed: X, group, slope })
    expect(fit.sigmaSlope!).toBeGreaterThan(0.1)
    let num = 0
    let d1 = 0
    let d2 = 0
    const mB = fit.ranef.reduce((s, r) => s + (r.slope ?? 0), 0) / 18
    const mT = trueS.reduce((a, b) => a + b, 0) / 18
    for (let j = 0; j < 18; j++) {
      const a = (fit.ranef[j]!.slope ?? 0) - mB
      const b = trueS[j]! - mT
      num += a * b
      d1 += a * a
      d2 += b * b
    }
    expect(num / Math.sqrt(d1 * d2)).toBeGreaterThan(0.4)
  })
})

describe('GLMM PQL', () => {
  it('binomial recovers positive group variance and coef sign', () => {
    const g = rng(3)
    const y: number[] = []
    const X: number[][] = []
    const group: string[] = []
    for (let j = 0; j < 15; j++) {
      const u = 1.2 * g.normal()
      for (let i = 0; i < 10; i++) {
        const x = g.u() > 0.5 ? 1 : 0
        const eta = -0.5 + 1.2 * x + u
        const p = 1 / (1 + Math.exp(-eta))
        y.push(g.u() < p ? 1 : 0)
        X.push([x])
        group.push(`g${j}`)
      }
    }
    const fit = glmm(y, { family: 'binomial', fixed: X, group })
    expect(fit.sigmaRandom).toBeGreaterThan(0.1)
    expect(fit.fixed.find((f) => f.name === 'X1')!.coefficient).toBeGreaterThan(0.3)
  })

  it('poisson recovers positive group RE', () => {
    const g = rng(4)
    const y: number[] = []
    const X: number[][] = []
    const group: string[] = []
    for (let j = 0; j < 12; j++) {
      const u = 0.6 * g.normal()
      for (let i = 0; i < 8; i++) {
        const x = g.normal()
        const mu = Math.exp(0.2 + 0.4 * x + u)
        // Poisson via exponential waiting (thin): use rounded mean + noise
        y.push(Math.max(0, Math.round(mu + g.normal() * Math.sqrt(mu))))
        X.push([x])
        group.push(`g${j}`)
      }
    }
    const fit = glmm(y, { family: 'poisson', fixed: X, group })
    expect(fit.sigmaRandom).toBeGreaterThan(0.05)
    expect(Number.isFinite(fit.logLik)).toBe(true)
  })
})
