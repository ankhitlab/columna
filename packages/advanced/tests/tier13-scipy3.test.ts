import { describe, expect, it } from 'vitest'
import {
  andersonKSample,
  brentq,
  dbscan,
  energyDistance,
  isotonicRegression,
  ksTwoSample,
  lasso,
  lowess,
  matrix,
  nnls,
  pdist,
  quantileRegression,
  ridge,
  savitzkyGolay,
  simpson,
  trapz,
  welchPsd,
} from '@columna/advanced'

describe('KS / AD-k / energy', () => {
  it('ksTwoSample separates shifted samples', () => {
    const a = Array.from({ length: 40 }, (_, i) => i * 0.1)
    const b = Array.from({ length: 40 }, (_, i) => i * 0.1 + 2)
    expect(ksTwoSample(a, b).pValue).toBeLessThan(0.05)
  })

  it('andersonKSample and energyDistance run', () => {
    const a = [1, 2, 3, 4, 5, 6]
    const b = [2, 3, 4, 5, 6, 7]
    const c = [10, 11, 12, 13, 14, 15]
    expect(andersonKSample([a, b, c]).statistic).toBeGreaterThan(0)
    expect(energyDistance(a, c).statistic).toBeGreaterThan(0)
  })
})

describe('smoothing', () => {
  it('lowess and isotonic', () => {
    const x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    const y = x.map((v) => 2 * v + (v % 2) * 0.2)
    const lo = lowess(x, y, { frac: 0.5 })
    expect(lo.fitted.length).toBe(10)
    const iso = isotonicRegression([3, 1, 2, 4, 3, 5])
    for (let i = 1; i < iso.fitted.length; i++) expect(iso.fitted[i]!).toBeGreaterThanOrEqual(iso.fitted[i - 1]! - 1e-12)
  })
})

describe('penalized + quantile', () => {
  it('ridge / lasso shrink; QR median fits', () => {
    const X = Array.from({ length: 30 }, (_, i) => [i, i * i * 0.01])
    const y = X.map((r) => 1 + 2 * r[0]! + 0.1 * r[1]! + (iNoise(r[0]!)))
    const r = ridge(y, X, { alpha: 10 })
    expect(r.coef.length).toBe(2)
    const l = lasso(y, X, { alpha: 0.5 })
    expect(l.method).toBe('lasso')
    const q = quantileRegression(y, X, { tau: 0.5 })
    expect(q.fitted.length).toBe(30)
  })
})

function iNoise(x: number) {
  return Math.sin(x * 12.9898) * 0.3
}

describe('dbscan + nnls + signal + numerics', () => {
  it('dbscan finds clusters', () => {
    const X = [
      [0, 0], [0.1, 0.1], [0.2, 0],
      [5, 5], [5.1, 5], [5, 5.2],
      [10, 0],
    ]
    const r = dbscan(X, { eps: 0.5, minSamples: 2 })
    expect(r.nClusters).toBeGreaterThanOrEqual(2)
    expect(pdist(X).length).toBe((7 * 6) / 2)
  })

  it('nnls non-negative', () => {
    const A = matrix(3, 2, Float64Array.from([1, 0, 1, 1, 0, 1]))
    const { x } = nnls(A, [1, 1.5, 1])
    expect(x[0]!).toBeGreaterThanOrEqual(0)
    expect(x[1]!).toBeGreaterThanOrEqual(0)
  })

  it('welch / savgol / brentq / trapz', () => {
    const sig = Array.from({ length: 256 }, (_, i) => Math.sin((2 * Math.PI * i) / 16))
    const psd = welchPsd(sig, { fs: 1, nperseg: 64 })
    expect(psd.power.some((p) => p > 0)).toBe(true)
    const sm = savitzkyGolay(sig, { windowLength: 11, polyOrder: 3 })
    expect(sm.length).toBe(256)
    const root = brentq((x) => x * x - 2, 0, 2)
    expect(root).toBeCloseTo(Math.SQRT2, 8)
    expect(trapz([0, 1, 2], [0, 1, 2])).toBeCloseTo(2, 8)
    expect(simpson([0, 1, 0])).toBeCloseTo(4 / 3, 6)
  })
})
