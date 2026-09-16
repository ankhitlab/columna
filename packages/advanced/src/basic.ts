/**
 * Tier 2 of the Minitab parity plan — the rest of Stat › Basic Statistics:
 * 1-Sample Z, 1 / 2 Proportions (exact and normal), 1- / 2-Sample Poisson Rate, 1 Variance
 * (χ² and Bonett), Correlation with p-value and CI, Outlier tests (Grubbs, Dixon).
 */
import { betainc, binomial, chi2 as chi2Dist, hypergeomPmf, normal, poisson, t as tDist } from './dist.js'
import { cleanNumbers, type Alternative } from './tests.js'

const STD = normal()

function checkAlt(a: Alternative | undefined): Alternative {
  const alt = a ?? 'two-sided'
  if (alt !== 'two-sided' && alt !== 'less' && alt !== 'greater') throw new RangeError(`alternative must be 'two-sided' | 'less' | 'greater', got ${String(alt)}`)
  return alt
}
function checkConf(c: number | undefined): number {
  const conf = c ?? 0.95
  if (!(conf > 0 && conf < 1)) throw new RangeError(`confidence must be in (0, 1), got ${conf}`)
  return conf
}
function zPValue(z: number, alternative: Alternative): number {
  if (alternative === 'less') return STD.cdf(z)
  if (alternative === 'greater') return STD.sf(z)
  return Math.min(1, 2 * STD.sf(Math.abs(z)))
}
function zInterval(estimate: number, se: number, confidence: number, alternative: Alternative): [number, number] {
  if (alternative === 'less') return [-Infinity, estimate + STD.ppf(confidence) * se]
  if (alternative === 'greater') return [estimate - STD.ppf(confidence) * se, Infinity]
  const h = STD.ppf(0.5 + confidence / 2) * se
  return [estimate - h, estimate + h]
}
function moments(v: Float64Array) {
  const n = v.length
  let s = 0
  for (let i = 0; i < n; i++) s += v[i]!
  const mean = s / n
  let m2 = 0
  for (let i = 0; i < n; i++) m2 += (v[i]! - mean) ** 2
  const variance = n > 1 ? m2 / (n - 1) : NaN
  return { n, mean, variance, sd: Math.sqrt(variance) }
}

// ---- 1-Sample Z ---------------------------------------------------------------------------------------

export interface ZTestResult {
  test: 'one-sample z'
  estimate: number
  se: number
  statistic: number
  pValue: number
  alternative: Alternative
  ci: [number, number]
  confidence: number
  n: number
  sigma: number
}

/** 1-Sample Z: H0 mean = mu with known population σ (Minitab 1-Sample Z). */
export function ztest1(
  x: ArrayLike<number | null | undefined>,
  options: { sigma: number; mu?: number; alternative?: Alternative; confidence?: number },
): ZTestResult {
  if (!(options.sigma > 0)) throw new RangeError(`ztest1: sigma must be > 0 (got ${options.sigma})`)
  const alternative = checkAlt(options.alternative)
  const confidence = checkConf(options.confidence)
  const v = cleanNumbers(x)
  if (v.length < 1) throw new RangeError('ztest1 needs at least 1 observation')
  const m = moments(v)
  const se = options.sigma / Math.sqrt(m.n)
  const statistic = (m.mean - (options.mu ?? 0)) / se
  return { test: 'one-sample z', estimate: m.mean, se, statistic, pValue: zPValue(statistic, alternative), alternative, ci: zInterval(m.mean, se, confidence, alternative), confidence, n: m.n, sigma: options.sigma }
}

// ---- proportions -------------------------------------------------------------------------------------

export interface PropTestResult {
  test: '1 proportion' | '2 proportions'
  method: 'exact' | 'normal' | 'fisher'
  /** p̂ for one sample, p̂₁ − p̂₂ for two. */
  estimate: number
  statistic?: number
  pValue: number
  alternative: Alternative
  ci: [number, number]
  confidence: number
  samples: Array<{ events: number; trials: number; proportion: number }>
}

/** Clopper–Pearson (exact) CI for a binomial proportion; one-sided variants for the alternatives. */
function clopperPearson(x: number, n: number, confidence: number, alternative: Alternative): [number, number] {
  const lower = (a: number) => (x === 0 ? 0 : betaQuantile(a, x, n - x + 1))
  const upper = (a: number) => (x === n ? 1 : betaQuantile(1 - a, x + 1, n - x))
  if (alternative === 'less') return [0, upper(1 - confidence)]
  if (alternative === 'greater') return [lower(1 - confidence), 1]
  const a = (1 - confidence) / 2
  return [lower(a), upper(a)]
}
/** Beta(a, b) quantile by bisection on the regularized incomplete beta. */
function betaQuantile(p: number, a: number, b: number): number {
  let lo = 0
  let hi = 1
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi)
    if (betainc(a, b, mid) < p) lo = mid
    else hi = mid
    if (hi - lo < 1e-15) break
  }
  return 0.5 * (lo + hi)
}

/**
 * 1 Proportion (Minitab): exact binomial test and Clopper–Pearson interval by default, or the
 * normal approximation (`method: 'normal'`, Wald interval as Minitab prints for that option).
 */
export function propTest1(
  events: number,
  trials: number,
  options: { p0?: number; alternative?: Alternative; confidence?: number; method?: 'exact' | 'normal' } = {},
): PropTestResult {
  if (!(Number.isInteger(events) && Number.isInteger(trials) && trials > 0 && events >= 0 && events <= trials)) {
    throw new RangeError(`propTest1: events must be an integer in [0, trials] (got ${events} of ${trials})`)
  }
  const p0 = options.p0 ?? 0.5
  if (!(p0 > 0 && p0 < 1)) throw new RangeError(`propTest1: p0 must be in (0, 1), got ${p0}`)
  const alternative = checkAlt(options.alternative)
  const confidence = checkConf(options.confidence)
  const method = options.method ?? 'exact'
  const phat = events / trials
  let pValue: number
  let statistic: number | undefined
  let ci: [number, number]
  if (method === 'exact') {
    const d = binomial(trials, p0)
    if (alternative === 'less') pValue = d.cdf(events)
    else if (alternative === 'greater') pValue = d.sf(events - 1)
    else {
      // sum of probabilities of outcomes at most as likely as the observed one (scipy binomtest)
      const pObs = d.pmf(events)
      let s = 0
      for (let k = 0; k <= trials; k++) {
        const pk = d.pmf(k)
        if (pk <= pObs * (1 + 1e-7)) s += pk
      }
      pValue = Math.min(1, s)
    }
    ci = clopperPearson(events, trials, confidence, alternative)
  } else {
    const se0 = Math.sqrt((p0 * (1 - p0)) / trials)
    statistic = (phat - p0) / se0
    pValue = zPValue(statistic, alternative)
    ci = zInterval(phat, Math.sqrt((phat * (1 - phat)) / trials), confidence, alternative)
    ci = [Math.max(0, ci[0]), Math.min(1, ci[1])]
  }
  return { test: '1 proportion', method, estimate: phat, statistic, pValue, alternative, ci, confidence, samples: [{ events, trials, proportion: phat }] }
}

/**
 * 2 Proportions (Minitab): pooled-variance z test for H0 p₁ = p₂ with an unpooled CI for p₁ − p₂;
 * `method: 'fisher'` gives Fisher's exact test instead (Minitab's option for small samples).
 */
export function propTest2(
  events1: number,
  trials1: number,
  events2: number,
  trials2: number,
  options: { alternative?: Alternative; confidence?: number; method?: 'normal' | 'fisher'; pooled?: boolean } = {},
): PropTestResult {
  for (const [e, n] of [[events1, trials1], [events2, trials2]] as const) {
    if (!(Number.isInteger(e) && Number.isInteger(n) && n > 0 && e >= 0 && e <= n)) throw new RangeError(`propTest2: events must be integers in [0, trials] (got ${e} of ${n})`)
  }
  const alternative = checkAlt(options.alternative)
  const confidence = checkConf(options.confidence)
  const p1 = events1 / trials1
  const p2 = events2 / trials2
  const diff = p1 - p2
  const seUnpooled = Math.sqrt((p1 * (1 - p1)) / trials1 + (p2 * (1 - p2)) / trials2)
  let ci = zInterval(diff, seUnpooled, confidence, alternative)
  ci = [Math.max(-1, ci[0]), Math.min(1, ci[1])]
  const samples = [
    { events: events1, trials: trials1, proportion: p1 },
    { events: events2, trials: trials2, proportion: p2 },
  ]
  if (options.method === 'fisher') {
    // Fisher exact on [[e1, n1−e1], [e2, n2−e2]]: hypergeometric on e1 given margins
    const N = trials1 + trials2
    const K = events1 + events2
    const lo = Math.max(0, K - trials2)
    const hi = Math.min(K, trials1)
    const pObs = hypergeomPmf(events1, N, K, trials1)
    let pValue = 0
    if (alternative === 'greater') for (let k = events1; k <= hi; k++) pValue += hypergeomPmf(k, N, K, trials1)
    else if (alternative === 'less') for (let k = lo; k <= events1; k++) pValue += hypergeomPmf(k, N, K, trials1)
    else for (let k = lo; k <= hi; k++) {
      const pk = hypergeomPmf(k, N, K, trials1)
      if (pk <= pObs * (1 + 1e-7)) pValue += pk
    }
    return { test: '2 proportions', method: 'fisher', estimate: diff, pValue: Math.min(1, pValue), alternative, ci, confidence, samples }
  }
  const pooled = options.pooled ?? true
  const pbar = (events1 + events2) / (trials1 + trials2)
  const se0 = pooled ? Math.sqrt(pbar * (1 - pbar) * (1 / trials1 + 1 / trials2)) : seUnpooled
  const statistic = diff / se0
  return { test: '2 proportions', method: 'normal', estimate: diff, statistic, pValue: zPValue(statistic, alternative), alternative, ci, confidence, samples }
}

// ---- Poisson rates ---------------------------------------------------------------------------------------

export interface RateTestResult {
  test: '1-sample Poisson rate' | '2-sample Poisson rate'
  method: 'exact' | 'normal'
  /** Rate λ̂ (one sample) or the difference λ̂₁ − λ̂₂ (two samples). */
  estimate: number
  statistic?: number
  pValue: number
  alternative: Alternative
  ci: [number, number]
  confidence: number
  samples: Array<{ events: number; exposure: number; rate: number }>
}

/** Exact (Garwood) CI for a Poisson mean given `events`. */
function poissonExactCI(events: number, confidence: number, alternative: Alternative): [number, number] {
  const lower = (a: number) => (events === 0 ? 0 : chi2Dist(2 * events).ppf(a) / 2)
  const upper = (a: number) => chi2Dist(2 * events + 2).ppf(1 - a) / 2
  if (alternative === 'less') return [0, upper(1 - confidence)]
  if (alternative === 'greater') return [lower(1 - confidence), Infinity]
  const a = (1 - confidence) / 2
  return [lower(a), upper(a)]
}

/** 1-Sample Poisson Rate (Minitab): exact test of H0 λ = λ₀ over `exposure` units, or the normal approximation. */
export function poissonRateTest1(
  events: number,
  exposure: number,
  options: { lambda0?: number; alternative?: Alternative; confidence?: number; method?: 'exact' | 'normal' } = {},
): RateTestResult {
  if (!(Number.isInteger(events) && events >= 0) || !(exposure > 0)) throw new RangeError(`poissonRateTest1: events must be a non-negative integer and exposure > 0`)
  const lambda0 = options.lambda0 ?? 1
  if (!(lambda0 > 0)) throw new RangeError(`poissonRateTest1: lambda0 must be > 0 (got ${lambda0})`)
  const alternative = checkAlt(options.alternative)
  const confidence = checkConf(options.confidence)
  const method = options.method ?? 'exact'
  const rate = events / exposure
  let pValue: number
  let statistic: number | undefined
  let ci: [number, number]
  if (method === 'exact') {
    const d = poisson(lambda0 * exposure)
    if (alternative === 'less') pValue = d.cdf(events)
    else if (alternative === 'greater') pValue = d.sf(events - 1)
    else {
      const pObs = d.pmf(events)
      let s = 0
      const kmax = Math.max(events, Math.ceil(lambda0 * exposure + 40 * Math.sqrt(lambda0 * exposure) + 50))
      for (let k = 0; k <= kmax; k++) {
        const pk = d.pmf(k)
        if (pk <= pObs * (1 + 1e-7)) s += pk
      }
      pValue = Math.min(1, s)
    }
    const c = poissonExactCI(events, confidence, alternative)
    ci = [c[0] / exposure, c[1] / exposure]
  } else {
    statistic = (rate - lambda0) / Math.sqrt(lambda0 / exposure)
    pValue = zPValue(statistic, alternative)
    ci = zInterval(rate, Math.sqrt(rate / exposure), confidence, alternative)
    ci = [Math.max(0, ci[0]), ci[1]]
  }
  return { test: '1-sample Poisson rate', method, estimate: rate, statistic, pValue, alternative, ci, confidence, samples: [{ events, exposure, rate }] }
}

/**
 * 2-Sample Poisson Rate (Minitab): exact conditional test — given the total, X₁ ~ Binomial(X₁+X₂, t₁/(t₁+t₂))
 * under H0 — or the normal approximation with pooled rate; CI for λ₁ − λ₂ is normal (unpooled).
 */
export function poissonRateTest2(
  events1: number,
  exposure1: number,
  events2: number,
  exposure2: number,
  options: { alternative?: Alternative; confidence?: number; method?: 'exact' | 'normal' } = {},
): RateTestResult {
  for (const [e, x] of [[events1, exposure1], [events2, exposure2]] as const) {
    if (!(Number.isInteger(e) && e >= 0) || !(x > 0)) throw new RangeError('poissonRateTest2: events must be non-negative integers and exposures > 0')
  }
  const alternative = checkAlt(options.alternative)
  const confidence = checkConf(options.confidence)
  const method = options.method ?? 'exact'
  const r1 = events1 / exposure1
  const r2 = events2 / exposure2
  const diff = r1 - r2
  const ci = zInterval(diff, Math.sqrt(r1 / exposure1 + r2 / exposure2), confidence, alternative)
  const samples = [
    { events: events1, exposure: exposure1, rate: r1 },
    { events: events2, exposure: exposure2, rate: r2 },
  ]
  let pValue: number
  let statistic: number | undefined
  if (method === 'exact') {
    const total = events1 + events2
    const p0 = exposure1 / (exposure1 + exposure2)
    if (total === 0) pValue = 1
    else {
      const d = binomial(total, p0)
      if (alternative === 'less') pValue = d.cdf(events1)
      else if (alternative === 'greater') pValue = d.sf(events1 - 1)
      else {
        const pObs = d.pmf(events1)
        let s = 0
        for (let k = 0; k <= total; k++) {
          const pk = d.pmf(k)
          if (pk <= pObs * (1 + 1e-7)) s += pk
        }
        pValue = Math.min(1, s)
      }
    }
  } else {
    const pooled = (events1 + events2) / (exposure1 + exposure2)
    statistic = diff / Math.sqrt(pooled / exposure1 + pooled / exposure2)
    pValue = zPValue(statistic, alternative)
  }
  return { test: '2-sample Poisson rate', method, estimate: diff, statistic, pValue, alternative, ci, confidence, samples }
}

// ---- 1 Variance ---------------------------------------------------------------------------------------

export interface VarTest1Result {
  test: '1 variance'
  method: 'chi-square' | 'bonett'
  /** Sample variance s². */
  estimate: number
  sd: number
  statistic?: number
  df?: number
  pValue: number
  alternative: Alternative
  /** CI for the variance σ² (take √ for σ). */
  ci: [number, number]
  confidence: number
  n: number
  kurtosis?: number
}

/**
 * 1 Variance (Minitab): χ² test of H0 σ² = σ₀² (normal data) or Bonett's kurtosis-adjusted method
 * (default in Minitab for non-normal data; Bonett 2006): CI exp[ln(c·s²) ± z·c·se₀],
 * se₀² = (γ̂ − (n−3)/n)/(n−1), c = n/(n − z), γ̂ the kurtosis about a trimmed mean;
 * p-value = smallest α whose interval excludes σ₀².
 */
export function varTest1(
  x: ArrayLike<number | null | undefined>,
  options: { sigma0: number; alternative?: Alternative; confidence?: number; method?: 'chi-square' | 'bonett' },
): VarTest1Result {
  if (!(options.sigma0 > 0)) throw new RangeError(`varTest1: sigma0 must be > 0 (got ${options.sigma0})`)
  const alternative = checkAlt(options.alternative)
  const confidence = checkConf(options.confidence)
  const v = cleanNumbers(x)
  const m = moments(v)
  if (m.n < 2) throw new RangeError(`varTest1 needs at least 2 observations, got ${m.n}`)
  const method = options.method ?? 'chi-square'
  const s2 = m.variance
  const var0 = options.sigma0 ** 2
  if (method === 'chi-square') {
    const df = m.n - 1
    const statistic = (df * s2) / var0
    const d = chi2Dist(df)
    const pValue = alternative === 'less' ? d.cdf(statistic) : alternative === 'greater' ? d.sf(statistic) : Math.min(1, 2 * Math.min(d.cdf(statistic), d.sf(statistic)))
    let ci: [number, number]
    if (alternative === 'less') ci = [0, (df * s2) / d.ppf(1 - confidence)]
    else if (alternative === 'greater') ci = [(df * s2) / d.ppf(confidence), Infinity]
    else ci = [(df * s2) / d.ppf(0.5 + confidence / 2), (df * s2) / d.ppf(0.5 - confidence / 2)]
    return { test: '1 variance', method, estimate: s2, sd: m.sd, statistic, df, pValue, alternative, ci, confidence, n: m.n }
  }
  // Bonett one-sample
  const n = m.n
  const sorted = v.slice().sort()
  const trim = n > 4 ? Math.floor(n / (2 * Math.sqrt(n - 4))) : 0
  let tm = 0
  for (let i = trim; i < n - trim; i++) tm += sorted[i]!
  tm /= n - 2 * trim
  let s4 = 0
  let s2c = 0
  for (let i = 0; i < n; i++) {
    const d2 = (v[i]! - tm) ** 2
    s2c += d2
    s4 += d2 * d2
  }
  const kurt = (n * s4) / (s2c * s2c)
  const se0 = Math.sqrt((kurt - (n - 3) / n) / (n - 1))
  const bounds = (alpha: number, sided: 1 | 2): [number, number] => {
    const z = STD.ppf(1 - alpha / sided)
    const c = n / (n - z)
    const center = Math.log(c * s2)
    return [Math.exp(center - z * c * se0), Math.exp(center + z * c * se0)]
  }
  let ci: [number, number]
  if (alternative === 'two-sided') ci = bounds(1 - confidence, 2)
  else if (alternative === 'less') ci = [0, bounds(1 - confidence, 1)[1]]
  else ci = [bounds(1 - confidence, 1)[0], Infinity]
  const excludes = (alpha: number) => {
    if (alternative === 'two-sided') {
      const [lo, hi] = bounds(alpha, 2)
      return lo > var0 || hi < var0
    }
    const [lo, hi] = bounds(alpha, 1)
    return alternative === 'less' ? hi < var0 : lo > var0
  }
  let pValue = 1
  if (excludes(1 - 1e-12)) {
    let lo = 1e-16
    let hi = 1
    for (let i = 0; i < 100; i++) {
      const mid = 0.5 * (lo + hi)
      if (excludes(mid)) hi = mid
      else lo = mid
      if (hi - lo < 1e-12 * hi) break
    }
    pValue = hi
  }
  return { test: '1 variance', method: 'bonett', estimate: s2, sd: m.sd, pValue, alternative, ci, confidence, n, kurtosis: kurt }
}

// ---- correlation with inference -------------------------------------------------------------------------

export interface CorrTestResult {
  test: 'Pearson correlation' | 'Spearman correlation' | 'Kendall correlation'
  estimate: number
  statistic: number
  df: number
  pValue: number
  alternative: Alternative
  /** Fisher-z interval (Pearson) or Bonett–Wright interval (Spearman); Kendall uses z-approx CI. */
  ci: [number, number]
  confidence: number
  n: number
}

function averageRanks(v: Float64Array): Float64Array {
  const n = v.length
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => v[a]! - v[b]!)
  const r = new Float64Array(n)
  for (let i = 0; i < n; ) {
    let j = i
    while (j + 1 < n && v[order[j + 1]!] === v[order[i]!]) j++
    const rank = (i + 1 + j + 1) / 2
    for (let k = i; k <= j; k++) r[order[k]!] = rank
    i = j + 1
  }
  return r
}

/** Correlation with a t-test of H0 ρ = 0 and a confidence interval (Minitab Correlation; scipy pearsonr / spearmanr / kendalltau). */
export function corrTest(
  a: ArrayLike<number | null | undefined>,
  b: ArrayLike<number | null | undefined>,
  options: { method?: 'pearson' | 'spearman' | 'kendall'; alternative?: Alternative; confidence?: number } = {},
): CorrTestResult {
  if (a.length !== b.length) throw new RangeError(`corrTest needs equal-length samples, got ${a.length} and ${b.length}`)
  const alternative = checkAlt(options.alternative)
  const confidence = checkConf(options.confidence)
  const method = options.method ?? 'pearson'
  // pairwise-complete
  const xs: number[] = []
  const ys: number[] = []
  for (let i = 0; i < a.length; i++) {
    const u = a[i]
    const w = b[i]
    if (typeof u === 'number' && typeof w === 'number' && Number.isFinite(u) && Number.isFinite(w)) {
      xs.push(u)
      ys.push(w)
    }
  }
  const n = xs.length
  if (n < 3) throw new RangeError(`corrTest needs at least 3 complete pairs, got ${n}`)

  if (method === 'kendall') {
    let concordant = 0
    let discordant = 0
    let tx = 0
    let ty = 0
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = xs[i]! - xs[j]!
        const dy = ys[i]! - ys[j]!
        const sx = Math.sign(dx)
        const sy = Math.sign(dy)
        if (sx === 0 && sy === 0) {
          /* double tie ignored in tau-b numerator */
        } else if (sx === 0) tx++
        else if (sy === 0) ty++
        else if (sx * sy > 0) concordant++
        else discordant++
      }
    }
    const n0 = (n * (n - 1)) / 2
    const tau = (concordant - discordant) / Math.sqrt((n0 - tx) * (n0 - ty))
    const se = Math.sqrt((2 * (2 * n + 5)) / (9 * n * (n - 1)))
    const statistic = tau / se
    const pValue = zPValue(statistic, alternative)
    const zi = zInterval(tau, se, confidence, alternative)
    const ci: [number, number] = [Math.max(-1, zi[0]), Math.min(1, zi[1])]
    return {
      test: 'Kendall correlation',
      estimate: tau,
      statistic,
      df: n - 2,
      pValue,
      alternative,
      ci,
      confidence,
      n,
    }
  }

  const raw = [Float64Array.from(xs), Float64Array.from(ys)] as const
  const x: Float64Array = method === 'spearman' ? averageRanks(raw[0]) : raw[0]
  const y: Float64Array = method === 'spearman' ? averageRanks(raw[1]) : raw[1]
  const mx = moments(x)
  const my = moments(y)
  let sxy = 0
  for (let i = 0; i < n; i++) sxy += (x[i]! - mx.mean) * (y[i]! - my.mean)
  const r = sxy / Math.sqrt(mx.variance * my.variance * (n - 1) * (n - 1))
  const df = n - 2
  const statistic = r * Math.sqrt(df / Math.max(1e-300, 1 - r * r))
  const d = tDist(df)
  const pValue = alternative === 'less' ? d.cdf(statistic) : alternative === 'greater' ? d.sf(statistic) : Math.min(1, 2 * d.sf(Math.abs(statistic)))
  const z = Math.atanh(Math.max(-1 + 1e-15, Math.min(1 - 1e-15, r)))
  const sez = method === 'pearson' ? 1 / Math.sqrt(n - 3) : Math.sqrt((1 + (r * r) / 2) / (n - 3))
  const zi = zInterval(z, sez, confidence, alternative)
  const ci: [number, number] = [Math.tanh(zi[0]), Math.tanh(zi[1])]
  return { test: method === 'pearson' ? 'Pearson correlation' : 'Spearman correlation', estimate: r, statistic, df, pValue, alternative, ci, confidence, n }
}

export interface PartialCorrResult {
  test: 'partial Pearson' | 'partial Spearman'
  estimate: number
  statistic: number
  df: number
  pValue: number
  n: number
  nControls: number
}

/**
 * Partial correlation of x and y controlling for z columns (Pearson residual or Spearman-on-ranks).
 */
export function partialCorr(
  x: ArrayLike<number | null | undefined>,
  y: ArrayLike<number | null | undefined>,
  z: ArrayLike<ArrayLike<number | null | undefined>>,
  options: { method?: 'pearson' | 'spearman'; alternative?: Alternative } = {},
): PartialCorrResult {
  const method = options.method ?? 'pearson'
  const alternative = checkAlt(options.alternative)
  const zCols = Array.from(z)
  const n0 = x.length
  const rows: number[][] = []
  for (let i = 0; i < n0; i++) {
    const xi = x[i]
    const yi = y[i]
    if (typeof xi !== 'number' || !Number.isFinite(xi) || typeof yi !== 'number' || !Number.isFinite(yi)) continue
    const row = [xi, yi]
    let ok = true
    for (const col of zCols) {
      const v = col[i]
      if (typeof v !== 'number' || !Number.isFinite(v)) {
        ok = false
        break
      }
      row.push(v)
    }
    if (ok) rows.push(row)
  }
  const n = rows.length
  const p = zCols.length
  if (n < p + 3) throw new RangeError(`partialCorr needs ≥ ${p + 3} complete rows, got ${n}`)
  let X = rows.map((r) => r[0]!)
  let Y = rows.map((r) => r[1]!)
  const Z = rows.map((r) => r.slice(2))
  if (method === 'spearman') {
    X = Array.from(averageRanks(Float64Array.from(X)))
    Y = Array.from(averageRanks(Float64Array.from(Y)))
    for (let j = 0; j < p; j++) {
      const col = Float64Array.from(Z.map((r) => r[j]!))
      const ranks = averageRanks(col)
      for (let i = 0; i < n; i++) Z[i]![j] = ranks[i]!
    }
  }
  const resid = (resp: number[], preds: number[][]) => {
    const cols = preds[0]?.length ?? 0
    const M = cols + 1
    // OLS with intercept via normal equations
    const XtX = Array.from({ length: M }, () => new Array(M).fill(0))
    const Xty = new Array(M).fill(0)
    for (let i = 0; i < n; i++) {
      const row = [1, ...preds[i]!]
      for (let a = 0; a < M; a++) {
        Xty[a]! += row[a]! * resp[i]!
        for (let b = 0; b < M; b++) XtX[a]![b]! += row[a]! * row[b]!
      }
    }
    // Gaussian elimination
    const A = XtX.map((r, i) => [...r, Xty[i]!])
    for (let col = 0; col < M; col++) {
      let piv = col
      for (let r = col + 1; r < M; r++) if (Math.abs(A[r]![col]!) > Math.abs(A[piv]![col]!)) piv = r
      ;[A[col], A[piv]] = [A[piv]!, A[col]!]
      const d0 = A[col]![col]!
      if (Math.abs(d0) < 1e-14) continue
      for (let j = col; j <= M; j++) A[col]![j]! /= d0
      for (let r = 0; r < M; r++) {
        if (r === col) continue
        const f = A[r]![col]!
        for (let j = col; j <= M; j++) A[r]![j]! -= f * A[col]![j]!
      }
    }
    const beta = A.map((r) => r[M]!)
    return resp.map((yi, i) => {
      let pred = beta[0]!
      for (let j = 0; j < cols; j++) pred += beta[j + 1]! * preds[i]![j]!
      return yi - pred
    })
  }
  const rx = resid(X, Z)
  const ry = resid(Y, Z)
  const mx = moments(Float64Array.from(rx))
  const my = moments(Float64Array.from(ry))
  let sxy = 0
  for (let i = 0; i < n; i++) sxy += (rx[i]! - mx.mean) * (ry[i]! - my.mean)
  const r = sxy / Math.sqrt(Math.max(1e-300, mx.variance * my.variance * (n - 1) * (n - 1)))
  const df = n - 2 - p
  const statistic = r * Math.sqrt(df / Math.max(1e-300, 1 - r * r))
  const d = tDist(Math.max(1, df))
  const pValue =
    alternative === 'less' ? d.cdf(statistic) : alternative === 'greater' ? d.sf(statistic) : Math.min(1, 2 * d.sf(Math.abs(statistic)))
  return {
    test: method === 'spearman' ? 'partial Spearman' : 'partial Pearson',
    estimate: r,
    statistic,
    df,
    pValue,
    n,
    nControls: p,
  }
}

// ---- outliers ---------------------------------------------------------------------------------------------

export interface OutlierResult {
  test: 'Grubbs' | 'Dixon'
  statistic: number
  critical: number
  pValue: number
  alpha: number
  alternative: 'two-sided' | 'min' | 'max'
  /** Index (in the cleaned sample) and value of the suspected outlier. */
  outlier: { index: number; value: number }
  significant: boolean
  n: number
}

/**
 * Grubbs' test for a single outlier (Minitab Outlier Test › Grubbs). G = max|xᵢ − x̄|/s; the critical
 * value comes from the t distribution, G_crit = (n − 1)/√n · √(t²/(n − 2 + t²)), t = t(1 − α/(2n); n − 2)
 * (α/n for the one-sided smallest/largest variants); the p-value inverts the same relation.
 */
export function grubbs(
  x: ArrayLike<number | null | undefined>,
  options: { alpha?: number; alternative?: 'two-sided' | 'min' | 'max' } = {},
): OutlierResult {
  const alpha = options.alpha ?? 0.05
  if (!(alpha > 0 && alpha < 1)) throw new RangeError(`alpha must be in (0, 1), got ${alpha}`)
  const alternative = options.alternative ?? 'two-sided'
  const v = cleanNumbers(x)
  const n = v.length
  if (n < 3) throw new RangeError(`grubbs needs at least 3 observations, got ${n}`)
  const m = moments(v)
  if (!(m.sd > 0)) throw new RangeError('grubbs: all observations are identical')
  let idx = 0
  let dev = -Infinity
  for (let i = 0; i < n; i++) {
    const d = alternative === 'min' ? m.mean - v[i]! : alternative === 'max' ? v[i]! - m.mean : Math.abs(v[i]! - m.mean)
    if (d > dev) {
      dev = d
      idx = i
    }
  }
  const G = dev / m.sd
  const sides = alternative === 'two-sided' ? 2 : 1
  const d = tDist(n - 2)
  const tc = d.ppf(1 - alpha / (sides * n))
  const critical = ((n - 1) / Math.sqrt(n)) * Math.sqrt((tc * tc) / (n - 2 + tc * tc))
  // invert G → t: t² = (n − 2) G² n / ((n − 1)² − n G²)
  const denom = (n - 1) ** 2 - n * G * G
  const pValue = denom <= 0 ? 0 : Math.min(1, sides * n * d.sf(Math.sqrt(((n - 2) * n * G * G) / denom)))
  return { test: 'Grubbs', statistic: G, critical, pValue, alpha, alternative, outlier: { index: idx, value: v[idx]! }, significant: G > critical, n }
}

/** Dixon's ratio for the extreme at one end of a sorted sample, using the Dean–Dixon rule for the sample size. */
function dixonRatio(sorted: Float64Array, end: 'min' | 'max'): number {
  const n = sorted.length
  const s = end === 'min' ? sorted : Float64Array.from(sorted, (_, i) => -sorted[n - 1 - i]!)
  // gap index a and range index b: r10 (n ≤ 7), r11 (8–10), r21 (11–13), r22 (n ≥ 14)
  const [a, b] = n <= 7 ? [1, 0] : n <= 10 ? [1, 1] : n <= 13 ? [2, 1] : [2, 2]
  const range = s[n - 1 - b]! - s[0]!
  return range === 0 ? 0 : (s[a]! - s[0]!) / range
}

/** Seeded uniform generator (mulberry32) for the Dixon reference distribution. */
function mulberry(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296
  }
}
const dixonCache = new Map<string, Float64Array>()
/** Null distribution of the Dixon ratio (max over both ends for two-sided) from 20 000 seeded normal samples. */
function dixonReference(n: number, alternative: 'two-sided' | 'min' | 'max'): Float64Array {
  const key = `${n}:${alternative}`
  const hit = dixonCache.get(key)
  if (hit) return hit
  const reps = 20000
  const u = mulberry(0x5eed + n)
  const out = new Float64Array(reps)
  const sample = new Float64Array(n)
  for (let r = 0; r < reps; r++) {
    for (let i = 0; i < n; i += 2) {
      const rad = Math.sqrt(-2 * Math.log(u()))
      const ang = 2 * Math.PI * u()
      sample[i] = rad * Math.cos(ang)
      if (i + 1 < n) sample[i + 1] = rad * Math.sin(ang)
    }
    sample.sort()
    out[r] = alternative === 'min' ? dixonRatio(sample, 'min') : alternative === 'max' ? dixonRatio(sample, 'max') : Math.max(dixonRatio(sample, 'min'), dixonRatio(sample, 'max'))
  }
  out.sort()
  dixonCache.set(key, out)
  return out
}

/**
 * Dixon's Q test for a single outlier (Minitab Outlier Test › Dixon), 3 ≤ n ≤ 30, with the Dean–Dixon
 * ratio r10 / r11 / r21 / r22 chosen by sample size as Minitab does. Critical values and p-values come
 * from a seeded 20 000-sample simulation of the null distribution (agrees with Rorabacher's 1991 tables
 * to ±0.01), cached per (n, alternative).
 */
export function dixon(
  x: ArrayLike<number | null | undefined>,
  options: { alpha?: number; alternative?: 'two-sided' | 'min' | 'max' } = {},
): OutlierResult {
  const alpha = options.alpha ?? 0.05
  if (!(alpha > 0 && alpha < 1)) throw new RangeError(`alpha must be in (0, 1), got ${alpha}`)
  const alternative = options.alternative ?? 'two-sided'
  const v = cleanNumbers(x)
  const n = v.length
  if (n < 3 || n > 30) throw new RangeError(`dixon needs 3 ≤ n ≤ 30 observations, got ${n}`)
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => v[a]! - v[b]!)
  const sorted = Float64Array.from(order, (i) => v[i]!)
  const rMin = dixonRatio(sorted, 'min')
  const rMax = dixonRatio(sorted, 'max')
  let statistic: number
  let idx: number
  if (alternative === 'min') {
    statistic = rMin
    idx = order[0]!
  } else if (alternative === 'max') {
    statistic = rMax
    idx = order[n - 1]!
  } else {
    statistic = Math.max(rMin, rMax)
    idx = rMax >= rMin ? order[n - 1]! : order[0]!
  }
  const ref = dixonReference(n, alternative)
  const critical = ref[Math.min(ref.length - 1, Math.floor((1 - alpha) * ref.length))]!
  // p = P(R ≥ observed) under H0
  let lo = 0
  let hi = ref.length
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (ref[mid]! < statistic) lo = mid + 1
    else hi = mid
  }
  const pValue = (ref.length - lo) / ref.length
  return { test: 'Dixon', statistic, critical, pValue, alpha, alternative, outlier: { index: idx, value: v[idx]! }, significant: statistic > critical, n }
}
