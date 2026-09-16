import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DataFrame } from '@columna/core'
import {
  aliasStructure,
  boxplotStats,
  causeAndEffect,
  cumulativePeriodogram,
  descriptiveStats,
  dist,
  dotplot,
  ecdf,
  gChart,
  generalizedVarianceChart,
  graphicalSummary,
  interactionPlot,
  intervalPlot,
  mainEffectsPlot,
  manova,
  manovaModel,
  mewma,
  periodogram,
  poissonGof,
  random,
  stabilityStudy,
  t2Chart,
  tChart,
} from '@columna/advanced'

const ref = JSON.parse(readFileSync(new URL('./fixtures/tier8-scipy.json', import.meta.url), 'utf8'))
const close = (got: number, want: number, digits = 10) => expect(got).toBeCloseTo(want, digits)
const closeArr = (got: ArrayLike<number>, want: number[], digits = 10) => {
  expect(got.length).toBe(want.length)
  for (let i = 0; i < want.length; i++) close(got[i]!, want[i]!, digits)
}

describe('descriptive statistics and graphical summary', () => {
  it('Display Descriptive Statistics matches numpy / scipy (type-6 quartiles, adjusted skew / kurtosis, trimmed mean, MSSD)', () => {
    const c = ref.descriptive
    const d = descriptiveStats(c.x) as ReturnType<typeof descriptiveStats> & { mean: number }
    const s = d as Exclude<typeof d, unknown[]>
    close(s.mean, c.mean)
    close(s.sd, c.sd)
    close(s.seMean, c.seMean)
    close(s.q1, c.q1)
    close(s.median, c.median)
    close(s.q3, c.q3)
    close(s.skewness, c.skew, 10)
    close(s.kurtosis, c.kurt, 10)
    close(s.trimmedMean, c.trimmed, 10)
    close(s.mssd, c.mssd, 10)
    close(s.coefVar, c.coefVar, 10)
    expect(s.mode).toContain(c.mode)
    expect(s.nMode).toBe(2)
    expect(s.n).toBe(40)
    const byGroup = descriptiveStats({ a: c.x, b: c.x }, { by: c.x.map((_: number, i: number) => (i % 2 ? 'odd' : 'even')) })
    expect(byGroup.length).toBe(4)
    expect(byGroup.map((r) => r.name)).toEqual(['a[even]', 'a[odd]', 'b[even]', 'b[odd]'])
    expect(byGroup[0]!.n).toBe(20)
    expect(() => descriptiveStats([null, undefined])).toThrow(RangeError)
  })
  it('Graphical Summary: CIs for mean / sd match closed forms, median CI from the sign test, histogram counts add to n', () => {
    const c = ref.descriptive
    const g = graphicalSummary(c.x)
    closeArr(g.ci.mean, c.ci_mean, 10)
    closeArr(g.ci.sd, c.ci_sd, 10)
    expect(g.ci.median[0]).toBeLessThan(g.stats.median)
    expect(g.ci.median[1]).toBeGreaterThan(g.stats.median)
    expect(g.normality.test).toBe('Anderson-Darling')
    expect(g.histogram.reduce((s, b) => s + b.count, 0)).toBe(40)
    expect(g.boxplot.n).toBe(40)
  })
  it('boxplot statistics, ECDF, dotplot and interval plot', () => {
    const c = ref.boxplot
    const b = boxplotStats(c.x) as Exclude<ReturnType<typeof boxplotStats>, unknown[]>
    close(b.q1, c.q1)
    close(b.q3, c.q3)
    close(b.median, c.median)
    close(b.whiskerLow, c.wlo)
    close(b.whiskerHigh, c.whi)
    expect(b.outliers).toEqual(c.outliers)
    const grouped = boxplotStats(c.x, { by: c.x.map((v: number) => (v > 6 ? 'hi' : 'lo')) }) as Array<{ name?: string; n: number }>
    expect(grouped.map((g) => g.name)).toEqual(['hi', 'lo'])
    expect(grouped[0]!.n + grouped[1]!.n).toBe(c.x.length)
    const e = ecdf([3, 1, 2])
    expect(e.x).toEqual([1, 2, 3])
    closeArr(e.f, [1 / 3, 2 / 3, 1], 12)
    const d = dotplot([1, 1.01, 2, 5], { binWidth: 0.5 })
    expect(d.bins.reduce((s, b2) => s + b2.count, 0)).toBe(4)
    expect(d.bins[0]!.count).toBe(2)
    const iv = intervalPlot([1, 2, 3, 4, 10, 11, 12, 13], ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'])
    expect(iv.map((r) => r.level)).toEqual(['a', 'b'])
    close(iv[0]!.mean, 2.5, 12)
    // pooled sd = √(((2.5·... ) )) : both groups have ss = 5 → s_p = √(10/6), se = s_p/2
    close(iv[0]!.se, Math.sqrt(10 / 6) / 2, 12)
    close(iv[0]!.ci[0], 2.5 - dist.t(6).ppf(0.975) * iv[0]!.se, 12)
    const un = intervalPlot([1, 2, 3, 4, 10, 11, 12, 13], ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'], { pooled: false })
    close(un[0]!.se, Math.sqrt(5 / 3) / 2, 12)
  })
  it('main effects / interaction plot data and cause-and-effect layout', () => {
    const y = [1, 2, 3, 4, 5, 6, 7, 8]
    const a = ['p', 'p', 'p', 'p', 'q', 'q', 'q', 'q']
    const b = ['u', 'v', 'u', 'v', 'u', 'v', 'u', 'v']
    const me = mainEffectsPlot(y, { a, b })
    close(me.grandMean, 4.5, 12)
    expect(me.factors[0]!.levels.map((l) => l.mean)).toEqual([2.5, 6.5])
    expect(me.factors[1]!.levels.map((l) => l.mean)).toEqual([4, 5])
    const ip = interactionPlot(y, a, b, ['a', 'b'])
    expect(ip.means).toEqual([[2, 3], [6, 7]])
    expect(ip.counts).toEqual([[2, 2], [2, 2]])
    const fish = causeAndEffect({ effect: 'Defects', categories: { Man: ['Training'], Machine: ['Wear', { label: 'Setup', children: [{ label: 'Torque' }] }], Method: [], Material: ['Supplier'] } })
    expect(fish.categories.length).toBe(4)
    expect(fish.categories.filter((c) => c.side === 'top').length).toBe(2)
    expect(fish.svg).toContain('<svg')
    expect(fish.svg).toContain('Torque')
    expect(() => causeAndEffect({ effect: 'x', categories: {} })).toThrow(RangeError)
  })
  it('Monte-Carlo (seeded): adjusted skewness / kurtosis are unbiased under normality, sd CI covers ≈ 95 %', () => {
    const g = random(8101)
    const reps = 2000
    let sumSkew = 0
    let sumKurt = 0
    let cover = 0
    for (let i = 0; i < reps; i++) {
      const x = g.normal(25, 3, 2)
      const d = descriptiveStats(x) as Exclude<ReturnType<typeof descriptiveStats>, unknown[]>
      sumSkew += d.skewness
      sumKurt += d.kurtosis
      const gs = graphicalSummary(x)
      if (gs.ci.sd[0] <= 2 && 2 <= gs.ci.sd[1]) cover++
    }
    expect(Math.abs(sumSkew / reps)).toBeLessThan(0.03)
    expect(Math.abs(sumKurt / reps)).toBeLessThan(0.06)
    expect(cover / reps).toBeGreaterThan(0.93)
    expect(cover / reps).toBeLessThan(0.97)
  })
})

describe('Poisson goodness-of-fit', () => {
  it('matches the pooled χ² computed in numpy', () => {
    const c = ref.poissonGof
    const r = poissonGof(c.counts)
    close(r.mean, c.mean, 12)
    close(r.statistic, c.stat, 8)
    expect(r.df).toBe(c.df)
    expect(r.categories.length).toBe(c.categories)
    close(r.pValue, c.p, 8)
    const ft = poissonGof({ values: [0, 1, 2, 3, 4], frequencies: [30, 40, 20, 8, 2] })
    close(ft.mean, (40 + 40 + 24 + 8) / 100, 12)
    expect(() => poissonGof([1.5, 2])).toThrow(RangeError)
  })
  it('Monte-Carlo (seeded): size ≈ α for Poisson counts, high power against a negative binomial', () => {
    const g = random(8102)
    const reps = 1500
    let rej = 0
    let pow = 0
    for (let i = 0; i < reps; i++) {
      if (poissonGof(Array.from(g.poisson(150, 3))).pValue < 0.05) rej++
      // overdispersed: Poisson with gamma-distributed mean (NB with mean 3, shape 1.5)
      const lam = g.gamma(150, 1.5, 2)
      const nb = Array.from(lam, (l) => g.poisson(1, l)[0]!)
      if (poissonGof(nb).pValue < 0.05) pow++
    }
    expect(rej / reps).toBeGreaterThan(0.025)
    expect(rej / reps).toBeLessThan(0.08)
    expect(pow / reps).toBeGreaterThan(0.9)
  })
})

describe('spectral analysis', () => {
  it('periodogram matches scipy.signal.periodogram (one-sided density) and finds the dominant period', () => {
    const c = ref.periodogram
    const r = periodogram(c.x)
    closeArr(r.frequency, c.freq, 12)
    closeArr(r.spectrum, c.spectrum, 7)
    close(r.dominant[0]!.frequency, c.domFreq, 12)
    expect(Math.abs(r.dominant[0]!.period - 12)).toBeLessThan(0.5) // nearest Fourier period 200/17
    const sm = periodogram(c.x, { spans: [3, 3] })
    expect(sm.power.length).toBe(r.power.length)
    // smoothing preserves the total power approximately
    const tot = (p: number[]) => p.reduce((s, v) => s + v, 0)
    expect(Math.abs(tot(sm.power) - tot(r.power)) / tot(r.power)).toBeLessThan(0.05)
    expect(() => periodogram([1, 2, 3])).toThrow(RangeError)
    expect(() => periodogram(c.x, { spans: [4] })).toThrow(RangeError)
  })
  it('cumulative periodogram: Bartlett statistic matches, white noise is not rejected', () => {
    const c = ref.cumper
    const r = cumulativePeriodogram(c.x)
    close(r.statistic, c.D, 10)
    close(r.critical[0.05], c.crit05, 10)
    expect(r.whiteNoise).toBe(r.statistic < r.critical[0.05])
  })
  it('Monte-Carlo (seeded): white-noise test size ≈ α; AR(1) with φ = 0.6 is rejected', () => {
    const g = random(8103)
    const reps = 800
    let rej = 0
    let pow = 0
    for (let i = 0; i < reps; i++) {
      if (cumulativePeriodogram(g.normal(120)).pValue < 0.05) rej++
      const e = g.normal(120)
      const y: number[] = [e[0]!]
      for (let t = 1; t < 120; t++) y.push(0.6 * y[t - 1]! + e[t]!)
      if (cumulativePeriodogram(y).pValue < 0.05) pow++
    }
    expect(rej / reps).toBeGreaterThan(0.02)
    expect(rej / reps).toBeLessThan(0.08)
    expect(pow / reps).toBeGreaterThan(0.95)
  })
})

describe('alias structure', () => {
  it('defining relation, resolution and alias chains for 2^(5−1) and 2^(6−2)', () => {
    const a5 = aliasStructure(5, ['E=ABCD'])
    expect(a5.definingRelation).toEqual(ref.alias.k5.defining)
    expect(a5.resolution).toBe(5)
    expect(a5.aliases.A).toEqual(ref.alias.k5.A)
    expect(a5.runs).toBe(16)
    const a6 = aliasStructure(['A', 'B', 'C', 'D', 'E', 'F'], ['ABCE', 'BCDF'])
    expect(a6.definingRelation).toEqual(ref.alias.k6.defining)
    expect(a6.resolution).toBe(4)
    expect(a6.aliases.AB).toEqual(ref.alias.k6.AB)
    expect(a6.chains.find((c) => c.startsWith('AB '))).toBe('AB + CE')
    expect(a6.chains.find((c) => c.startsWith('A '))).toBe('A + BCE + DEF')
    expect(() => aliasStructure(3, [])).toThrow(RangeError)
    expect(() => aliasStructure(3, ['C=AZ'])).toThrow(RangeError)
  })
})

describe('stability study', () => {
  it('model selection p-values and per-batch shelf life match the numpy reference', () => {
    const c = ref.stability
    const r = stabilityStudy(c.y, c.time, c.batch, { lsl: 95 })
    close(r.selection[0]!.pValue, c.pInt, 8)
    expect(r.selection[0]!.removed).toBe(c.pInt > 0.25)
    close(r.selection[1]!.pValue, c.pBatch, 8)
    expect(r.model).toBe(c.model)
    for (const s of r.shelfLife) close(s.shelfLife, c.shelf[s.batch], 6)
    close(r.overall, Math.min(...(Object.values(c.shelf) as number[])), 6)
    expect(r.shelfLife.every((s) => s.limit === 'lsl')).toBe(true)
    const pr = r.predict(12, 'B1')
    expect(pr.lower).toBeLessThan(pr.fit)
    expect(() => stabilityStudy(c.y, c.time, c.batch, {})).toThrow(RangeError)
    expect(() => r.predict(1, 'nope')).toThrow(RangeError)
  })
  it('behaviour: identical batches pool to a common line; a steeper slope shortens the shelf life; usl works', () => {
    const g = random(8104)
    const time = [0, 3, 6, 9, 12, 18, 24, 0, 3, 6, 9, 12, 18, 24]
    const batch = time.map((_, i) => (i < 7 ? 'x' : 'y'))
    const y = time.map((t) => 100 - 0.2 * t + 0.05 * g.normal(1)[0]!)
    const r = stabilityStudy(y, time, batch, { lsl: 90 })
    expect(r.shelfLife.length).toBeGreaterThan(0)
    expect(r.overall).toBeGreaterThan(30)
    const steep = stabilityStudy(time.map((t) => 100 - 0.5 * t + 0.05 * g.normal(1)[0]!), time, batch, { lsl: 90 })
    expect(steep.overall).toBeLessThan(r.overall)
    const up = stabilityStudy(time.map((t) => 100 + 0.3 * t + 0.05 * g.normal(1)[0]!), time, batch, { usl: 105 })
    expect(up.shelfLife[0]!.limit).toBe('usl')
    expect(up.overall).toBeGreaterThan(10)
    expect(up.overall).toBeLessThan(20)
  })
})

describe('rare-event and multivariate control charts', () => {
  it('G chart limits from geometric quantiles / 3σ; T chart from the Weibull fit', () => {
    const c = ref.gchart
    const gc = gChart(c.g)
    close(gc.parameters.p!, c.p, 12)
    expect(gc.ucl).toBe(c.ucl)
    expect(gc.lcl).toBe(c.lcl)
    close(gChart(c.g, { limits: 'sigma' }).ucl, c.sigmaUcl, 10)
    expect(gc.points.length).toBe(c.g.length)
    expect(() => gChart([1.5, 2])).toThrow(RangeError)
    const t = ref.tchart
    const tc = tChart(t.t)
    close(tc.parameters.shape!, t.shape, 4)
    close(tc.parameters.scale!, t.scale, 3)
    close(tc.center / t.center, 1, 4)
    close(tc.ucl / t.ucl, 1, 3)
    close(tc.lcl / t.lcl, 1, 3)
    const ex = tChart(t.t, { distribution: 'exponential' })
    expect(ex.parameters.shape).toBe(1)
  })
  it('Hotelling T² (individuals and subgroups) and the generalized variance chart match numpy / scipy limits', () => {
    const c = ref.t2
    const r = t2Chart(c.X)
    closeArr(r.t2, c.t2, 9)
    close(r.ucl, c.ucl1, 9)
    close(t2Chart(c.X, { phase: 2 }).ucl, c.ucl2, 9)
    expect(r.p).toBe(2)
    expect(r.k).toBe(40)
    const s = ref.t2sub
    const rs = t2Chart(s.X, { subgroup: s.sg })
    closeArr(rs.t2, s.t2, 8)
    close(rs.ucl, s.ucl1, 8)
    expect(rs.m).toBe(4)
    const gv = generalizedVarianceChart(s.X, s.sg)
    closeArr(gv.values, s.dets, 9)
    close(gv.center, s.detPooled, 9)
    expect(gv.ucl).toBeGreaterThan(gv.center)
    expect(() => t2Chart([[1, 2], [2, 3]])).toThrow(RangeError)
  })
  it('MEWMA: T² sequence, h calibrated to ARL₀ ≈ 200, shift detected', () => {
    const g = random(8105)
    const rows = Array.from({ length: 60 }, () => [g.normal(1)[0]!, g.normal(1)[0]!])
    const r = mewma(rows, { lambda: 0.1, phase1: 40 })
    expect(r.h).toBeGreaterThan(5)
    expect(r.h).toBeLessThan(15)
    expect(r.arl0).toBe(200)
    expect(r.t2.length).toBe(60)
    const fixed = mewma(rows, { lambda: 0.1, h: 8.5 })
    expect(fixed.arl0).toBeUndefined()
    const shifted = rows.map((row, i) => (i >= 40 ? [row[0]! + 1.5, row[1]! + 1.5] : row))
    const s = mewma(shifted, { lambda: 0.1, h: 8.5, mean: [0, 0], covariance: { rows: 2, cols: 2, data: Float64Array.from([1, 0, 0, 1]) } })
    expect(s.outOfControl.some((i) => i >= 40)).toBe(true)
  })
  it('Monte-Carlo (seeded): T² phase-I false-alarm rate ≈ α; G chart false alarms ≈ 0.27 % with known p', () => {
    const g = random(8106)
    let flagged = 0
    let total = 0
    for (let i = 0; i < 300; i++) {
      const rows = Array.from({ length: 30 }, () => [g.normal(1)[0]!, g.normal(1)[0]!, g.normal(1)[0]!])
      const r = t2Chart(rows, { alpha: 0.05 })
      flagged += r.outOfControl.length
      total += 30
    }
    expect(flagged / total).toBeGreaterThan(0.035)
    expect(flagged / total).toBeLessThan(0.065)
    let gFlags = 0
    let gTotal = 0
    for (let i = 0; i < 200; i++) {
      const counts = Array.from({ length: 100 }, () => {
        let k = 0
        while (g.next() >= 0.05) k++
        return k
      })
      gFlags += gChart(counts, { p: 0.05 }).outOfControl.length
      gTotal += 100
    }
    expect(gFlags / gTotal).toBeLessThan(0.006)
  })
})

describe('general MANOVA', () => {
  it('Type III term tests match scipy.linalg.eigh(H, E); one-way model equals manova()', () => {
    const c = ref.manovaModel
    const r = manovaModel({ y1: c.y1, y2: c.y2, a: c.a, b: c.b }, ['y1', 'y2'], 'a*b')
    for (const t of r.terms) {
      const w = c.terms[t.term]
      close(t.pillai.statistic, w.pillai, 9)
      close(t.wilks.statistic, w.wilks, 9)
      close(t.hotelling.statistic, w.hotelling, 9)
      close(t.roy.statistic, w.roy, 9)
      expect(t.dfHypothesis).toBe(w.dfH)
    }
    expect(r.dfError).toBe(24)
    expect(r.univariate.length).toBe(6)
    const one = manovaModel({ y1: c.y1, y2: c.y2, a: c.a }, ['y1', 'y2'], 'a')
    const ow = manova(c.y1.map((v: number, i: number) => [v, c.y2[i]]), c.a)
    close(one.terms[0]!.wilks.statistic, ow.wilks.statistic, 10)
    close(one.terms[0]!.pillai.pValue, ow.pillai.pValue, 10)
    expect(() => manovaModel({ y1: c.y1, a: c.a }, [], 'a')).toThrow(RangeError)
  })
  it('Monte-Carlo (seeded): Pillai test for a null factor keeps its size in a two-factor model', () => {
    const g = random(8107)
    const reps = 800
    let rej = 0
    for (let i = 0; i < reps; i++) {
      const a = Array.from({ length: 36 }, (_, j) => ['p', 'q', 'r'][j % 3]!)
      const b = Array.from({ length: 36 }, (_, j) => (j % 2 ? 'u' : 'v'))
      const y1 = a.map((v) => (v === 'p' ? 1 : 0) + g.normal(1)[0]!)
      const y2 = a.map(() => g.normal(1)[0]!)
      const r = manovaModel({ y1, y2, a, b }, ['y1', 'y2'], 'a + b')
      if (r.terms[1]!.pillai.pValue < 0.05) rej++
    }
    expect(rej / reps).toBeGreaterThan(0.03)
    expect(rej / reps).toBeLessThan(0.075)
  })
})

describe('DataFrame methods (gap-closing A)', () => {
  it('descriptiveStats / graphicalSummary / poissonGof / boxplotStats / intervalPlot / plots / stability / charts / periodogram / manovaModel', async () => {
    const c = ref.descriptive
    const df = DataFrame.fromColumns({ x: c.x, g: c.x.map((_: number, i: number) => (i % 2 ? 'a' : 'b')) })
    close(df.descriptiveStats('x')[0]!.mean, c.mean, 12)
    expect(df.descriptiveStats(['x'], { by: 'g' }).length).toBe(2)
    close(df.graphicalSummary('x').ci.mean[0], c.ci_mean[0], 10)
    expect((df.boxplotStats('x', 'g') as unknown[]).length).toBe(2)
    expect(df.intervalPlot('x', 'g').length).toBe(2)
    expect(df.mainEffectsPlot('x', ['g']).factors[0]!.levels.length).toBe(2)
    expect(df.interactionPlot('x', 'g', 'g').means.length).toBe(2)
    const pg = DataFrame.fromColumns({ k: ref.poissonGof.counts })
    close(pg.poissonGof('k').statistic, ref.poissonGof.stat, 8)
    const st = ref.stability
    const sd = DataFrame.fromColumns({ y: st.y, t: st.time, b: st.batch })
    expect(sd.stabilityStudy('y', 't', 'b', { lsl: 95 }).model).toBe(st.model)
    expect(DataFrame.fromColumns({ g: ref.gchart.g }).gChart('g').ucl).toBe(ref.gchart.ucl)
    close(DataFrame.fromColumns({ t: ref.tchart.t }).tChart('t').parameters.shape!, ref.tchart.shape, 4)
    const tx = DataFrame.fromColumns({ a: ref.t2.X.map((r: number[]) => r[0]), b: ref.t2.X.map((r: number[]) => r[1]) })
    close(tx.t2Chart(['a', 'b']).ucl, ref.t2.ucl1, 9)
    expect(tx.mewma(['a', 'b'], { h: 9 }).t2.length).toBe(40)
    const px = DataFrame.fromColumns({ s: ref.periodogram.x })
    close(px.periodogram('s').dominant[0]!.frequency, ref.periodogram.domFreq, 12)
    const m = ref.manovaModel
    const md = DataFrame.fromColumns({ y1: m.y1, y2: m.y2, a: m.a, b: m.b })
    close(md.manovaModel(['y1', 'y2'], 'a*b').terms[0]!.wilks.statistic, m.terms.a.wilks, 9)
    close((await md.lazy().manovaModel(['y1', 'y2'], 'a')).terms[0]!.wilks.statistic, manova(m.y1.map((v: number, i: number) => [v, m.y2[i]]), m.a).wilks.statistic, 10)
  })
})
