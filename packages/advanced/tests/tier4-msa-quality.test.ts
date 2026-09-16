import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import {
  acceptanceSampling,
  attributeAgreement,
  gageLinearity,
  gageRR,
  gageType1,
  individualDistributionID,
  multiVari,
  pareto,
  runChart,
  symmetryTest,
} from '@columna/advanced'

const ref = JSON.parse(readFileSync(new URL('./fixtures/tier4-scipy.json', import.meta.url), 'utf8'))
const close = (got: number, want: number, digits = 10) => expect(got).toBeCloseTo(want, digits)

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
  return { u, normal }
}

describe('4.8 Gage R&R / Linearity / Type 1', () => {
  it('crossed ANOVA recovers large part variance and moderate GRR on the fixture', () => {
    const r = gageRR({ part: ref.gage.part, operator: ref.gage.operator, measurement: ref.gage.measurement })
    expect(r.design).toBe('crossed')
    expect(r.nParts).toBe(10)
    expect(r.nOperators).toBe(3)
    expect(r.sigmaPart).toBeGreaterThan(r.sigmaGageRR * 0.5)
    expect(r.components.find((c) => c.name === 'Total Variation')!.pctContribution).toBeCloseTo(100, 8)
    // %Contribution of GRR should be well below part-to-part for this synthetic design
    const grr = r.components.find((c) => c.name === 'Total Gage R&R')!
    const part = r.components.find((c) => c.name === 'Part-to-Part')!
    expect(part.pctContribution).toBeGreaterThan(grr.pctContribution)
  })

  it('recovers planted σ_repeat / σ_reprod ratios (seeded)', () => {
    const g = rng(55)
    const parts: string[] = []
    const ops: string[] = []
    const ys: number[] = []
    for (let p = 0; p < 12; p++) {
      const pe = 3 * g.normal()
      for (let o = 0; o < 3; o++) {
        const oe = 0.5 * g.normal()
        for (let r = 0; r < 3; r++) {
          parts.push(`P${p}`)
          ops.push(`O${o}`)
          ys.push(pe + oe + 0.3 * g.normal())
        }
      }
    }
    const fit = gageRR({ part: parts, operator: ops, measurement: ys })
    expect(fit.sigmaRepeat).toBeGreaterThan(0.15)
    expect(fit.sigmaRepeat).toBeLessThan(0.55)
    expect(fit.sigmaPart).toBeGreaterThan(2)
    expect(fit.ndc).toBeGreaterThanOrEqual(3)
  })

  it('linearity / Type 1 basics', () => {
    const refVals = [2, 4, 6, 8, 10, 2, 4, 6, 8, 10]
    const meas = refVals.map((r, i) => r + 0.1 + 0.02 * r + (i % 2 === 0 ? 0.05 : -0.05))
    const lin = gageLinearity(refVals, meas)
    expect(lin.slope).toBeGreaterThan(0)
    expect(lin.n).toBe(10)
    const t1 = gageType1(Array.from({ length: 20 }, (_, i) => 10 + 0.1 * Math.sin(i)), { reference: 10, tolerance: 2 })
    expect(t1.Cg).toBeGreaterThan(0)
    expect(Number.isFinite(t1.Cgk)).toBe(true)
  })
})

describe('4.9 attribute agreement', () => {
  it('Cohen κ matches the confusion-matrix formula', () => {
    const r = attributeAgreement(ref.cohen.a, { method: 'cohen', other: ref.cohen.b })
    close(r.kappa, ref.cohen.kappa, 10)
    expect(r.pValue).toBeGreaterThanOrEqual(0)
    expect(r.pValue).toBeLessThanOrEqual(1)
  })

  it('Fleiss κ matches the reference', () => {
    const r = attributeAgreement(ref.fleiss.ratings, { method: 'fleiss' })
    close(r.kappa, ref.fleiss.kappa, 8)
  })
})

describe('4.10 acceptance sampling', () => {
  it('OC / AOQ match binomial Pa', () => {
    const r = acceptanceSampling({ type: 'attributes', n: ref.acceptance.n, c: ref.acceptance.c }, { p: ref.acceptance.curve.map((c: { p: number }) => c.p) })
    for (let i = 0; i < r.curve.length; i++) {
      close(r.curve[i]!.Pa, ref.acceptance.curve[i].Pa, 10)
      close(r.curve[i]!.AOQ, ref.acceptance.curve[i].AOQ, 10)
    }
    expect(r.AQL).toBeDefined()
    expect(r.LTPD).toBeDefined()
    expect(r.AQL!).toBeLessThan(r.LTPD!)
  })

  it('binomial simulation of acceptance matches Pa (seeded)', () => {
    const g = rng(77)
    const n = 40
    const c = 1
    const p = 0.05
    const plan = acceptanceSampling({ type: 'attributes', n, c }, { p: [p] })
    let accept = 0
    const reps = 4000
    for (let i = 0; i < reps; i++) {
      let d = 0
      for (let j = 0; j < n; j++) if (g.u() < p) d++
      if (d <= c) accept++
    }
    expect(Math.abs(accept / reps - plan.curve[0]!.Pa)).toBeLessThan(0.02)
  })
})

describe('4.11 Pareto / run chart / multi-vari / symmetry / IDI', () => {
  it('Pareto cumulative reaches 100% in fixture order', () => {
    const r = pareto(ref.pareto.categories)
    expect(r.items[0]!.category).toBe('A')
    expect(r.items[0]!.count).toBe(40)
    close(r.items[r.items.length - 1]!.cumulativePct, 100, 8)
  })

  it('run chart reports four tests; random data rarely rejects clustering', () => {
    const g = rng(88)
    const x = Array.from({ length: 50 }, () => g.normal())
    const r = runChart(x)
    expect(r.nRuns).toBeGreaterThan(1)
    expect(r.clustering.pValue).toBeGreaterThan(0.01)
  })

  it('multi-vari and symmetry / IDI basics', () => {
    const y = [1, 2, 3, 4, 5, 6, 7, 8]
    const f1 = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
    const f2 = ['1', '1', '2', '2', '1', '1', '2', '2']
    const mv = multiVari(y, [f1, f2])
    expect(mv.cells.length).toBe(4)
    const sym = symmetryTest(Array.from({ length: 40 }, (_, i) => (i < 20 ? -1 : 1) * Math.sqrt(i + 1)))
    expect(sym.pValue).toBeGreaterThanOrEqual(0)
    const idi = individualDistributionID(ref.idiNormal)
    // Normal data: normal or Box–Cox(λ≈1) should win
    expect(['normal', 'box-cox']).toContain(idi.best.distribution)
    if (idi.best.distribution === 'box-cox') expect(Math.abs(idi.best.params.lambda! - 1)).toBeLessThan(0.4)
    expect(idi.fits.some((f) => f.distribution === 'normal')).toBe(true)
    expect(idi.fits.length).toBeGreaterThanOrEqual(8)
  })

  it('df helpers install', () => {
    const df = DataFrame.fromColumns({ cat: ref.pareto.categories, x: ref.idiNormal.concat(Array(45).fill(0)).slice(0, ref.pareto.categories.length) })
    expect(df.pareto('cat').items[0]!.category).toBe('A')
  })
})
