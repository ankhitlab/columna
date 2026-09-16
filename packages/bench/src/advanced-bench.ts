/**
 * Speed benchmark of every `columna/advanced` function (the Minitab-parity layer).
 *
 *   pnpm --filter @columna/bench run advanced            # → results/advanced-columna.json
 *   pnpm --filter @columna/bench run advanced:python     # → results/advanced-python.json (pandas / polars / scipy / numpy)
 *   pnpm --filter @columna/bench run advanced:report     # → docs/advanced-benchmarks.md
 *
 * Every case has an `id` shared with python/advanced_compare.py, which times the closest equivalent in
 * scipy / numpy / pandas / polars on data of the same size and distribution (values differ — this is a
 * throughput comparison; numerical agreement is covered by the vitest fixtures). Sizes are chosen so the
 * whole run takes a few minutes; BENCH_SCALE=0.1 shrinks the O(n) cases for a quick pass.
 */
import { mkdirSync, writeFileSync } from 'node:fs'
import { performance } from 'node:perf_hooks'
import * as A from 'columna/advanced'

const SCALE = Number(process.env.BENCH_SCALE ?? 1)
const REPEAT = Number(process.env.BENCH_REPEAT ?? 5)
const ONLY = process.env.BENCH_ONLY // substring filter on id
const OUT = new URL('../results/advanced-columna.json', import.meta.url)

const N = (base: number) => Math.max(50, Math.round(base * SCALE))

interface Case {
  id: string
  group: string
  /** Human label incl. size. */
  label: string
  n: number
  /** Build inputs once (excluded from timing); returns the timed closure. */
  setup: () => () => unknown
  /** Fewer repeats for heavy cases. */
  repeat?: number
  /** Which library python compares against (documentation only; python decides itself). */
  py?: string
}

interface Row {
  id: string
  group: string
  label: string
  n: number
  msMedian: number
  msMin: number
  repeat: number
}

const g = A.random(12345)
const normal = (n: number, mu = 0, sd = 1) => Array.from(g.normal(n, mu, sd))
const levels = (n: number, k: number, prefix = 'L') => Array.from({ length: n }, (_, i) => `${prefix}${i % k}`)
const rowsOf = (n: number, p: number) => Array.from({ length: n }, () => Array.from(g.normal(p)))
const linear = (X: number[][], beta: number[], noise = 1) => X.map((r) => r.reduce((s, v, j) => s + v * beta[j]!, 1) + noise * g.normal(1)[0]!)
const groups = (k: number, size: number, shift = 0.3) => Object.fromEntries(Array.from({ length: k }, (_, i) => [`g${i}`, normal(size, i * shift)]))

const cases: Case[] = []
const add = (c: Case) => cases.push(c)

// ---- distributions -------------------------------------------------------------------------------------------
{
  const n = N(1_000_000)
  add({ id: 'dist.normal.cdf', group: 'Distributions', label: `normal.cdf × ${n}`, n, py: 'scipy', setup: () => { const x = g.normal(n); const d = A.normal(); return () => d.map('cdf', x) } })
  add({ id: 'dist.normal.ppf', group: 'Distributions', label: `normal.ppf × ${n}`, n, py: 'scipy', setup: () => { const x = g.uniform(n); const d = A.normal(); return () => d.map('ppf', x) } })
  const m = N(100_000)
  add({ id: 'dist.t.cdf', group: 'Distributions', label: `t(10).cdf × ${m}`, n: m, py: 'scipy', setup: () => { const x = g.normal(m); const d = A.t(10); return () => d.map('cdf', x) } })
  add({ id: 'dist.t.ppf', group: 'Distributions', label: `t(10).ppf × ${m}`, n: m, py: 'scipy', setup: () => { const x = g.uniform(m); const d = A.t(10); return () => d.map('ppf', x) } })
  add({ id: 'dist.chi2.sf', group: 'Distributions', label: `chi2(5).sf × ${m}`, n: m, py: 'scipy', setup: () => { const x = g.chi2(m, 5); const d = A.chi2(5); return () => d.map('sf', x) } })
  add({ id: 'dist.f.ppf', group: 'Distributions', label: `f(3,20).ppf × ${m}`, n: m, py: 'scipy', setup: () => { const x = g.uniform(m); const d = A.f(3, 20); return () => d.map('ppf', x) } })
  add({ id: 'dist.gamma.cdf', group: 'Distributions', label: `gamma(2.5,3).cdf × ${m}`, n: m, py: 'scipy', setup: () => { const x = g.gamma(m, 2.5, 3); const d = A.gamma(2.5, 3); return () => d.map('cdf', x) } })
  add({ id: 'dist.beta.ppf', group: 'Distributions', label: `beta(2,5).ppf × ${N(10_000)}`, n: N(10_000), py: 'scipy', setup: () => { const x = g.uniform(N(10_000)); const d = A.beta(2, 5); return () => d.map('ppf', x) } })
  add({ id: 'dist.weibull.ppf', group: 'Distributions', label: `weibull(1.8,50).ppf × ${n}`, n, py: 'scipy', setup: () => { const x = g.uniform(n); const d = A.weibull(1.8, 50); return () => d.map('ppf', x) } })
  add({ id: 'dist.binomial.cdf', group: 'Distributions', label: `binomial(50,0.3).cdf × ${m}`, n: m, py: 'scipy', setup: () => { const k = g.binomial(m, 50, 0.3); const d = A.binomial(50, 0.3); return () => { let s = 0; for (let i = 0; i < m; i++) s += d.cdf(k[i]!); return s } } })
  add({ id: 'dist.poisson.pmf', group: 'Distributions', label: `poisson(4).pmf × ${m}`, n: m, py: 'scipy', setup: () => { const k = g.poisson(m, 4); const d = A.poisson(4); return () => { let s = 0; for (let i = 0; i < m; i++) s += d.pmf(k[i]!); return s } } })
  const q = N(10_000)
  add({ id: 'dist.nct.cdf', group: 'Distributions', label: `nctCdf × ${q}`, n: q, py: 'scipy', setup: () => { const x = g.normal(q, 1, 1); return () => { let s = 0; for (let i = 0; i < q; i++) s += A.nctCdf(x[i]!, 12, 0.8); return s } } })
  add({ id: 'dist.ptukey', group: 'Distributions', label: `ptukey × ${N(1000)}`, n: N(1000), py: 'scipy', setup: () => { const x = g.uniform(N(1000), 1, 5); return () => { let s = 0; for (let i = 0; i < x.length; i++) s += A.ptukey(x[i]!, 4, 20); return s } } })
  add({ id: 'dist.qtukey', group: 'Distributions', label: `qtukey × ${N(200)}`, n: N(200), py: 'scipy', setup: () => { const x = g.uniform(N(200), 0.5, 0.99); return () => { let s = 0; for (let i = 0; i < x.length; i++) s += A.qtukey(x[i]!, 4, 20); return s } } })
}

// ---- basic tests ----------------------------------------------------------------------------------------------
{
  const n = N(100_000)
  add({ id: 'ttest1', group: 'Basic statistics', label: `ttest1 n=${n}`, n, py: 'scipy', setup: () => { const x = normal(n, 50, 10); return () => A.ttest1(x, { mu: 50 }) } })
  add({ id: 'ttest2', group: 'Basic statistics', label: `ttest2 (Welch) n=${n}+${n}`, n, py: 'scipy', setup: () => { const a = normal(n); const b = normal(n, 0.1, 1.2); return () => A.ttest2(a, b) } })
  add({ id: 'ttestPaired', group: 'Basic statistics', label: `ttestPaired n=${n}`, n, py: 'scipy', setup: () => { const a = normal(n); const b = a.map((v) => v + 0.1 + 0.5 * g.normal(1)[0]!); return () => A.ttestPaired(a, b) } })
  add({ id: 'ztest1', group: 'Basic statistics', label: `ztest1 n=${n}`, n, py: 'numpy', setup: () => { const x = normal(n, 50, 10); return () => A.ztest1(x, { sigma: 10, mu: 50 }) } })
  add({ id: 'propTest1', group: 'Basic statistics', label: 'propTest1 (exact) 350/1000', n: 1000, py: 'scipy', setup: () => () => A.propTest1(350, 1000, { p0: 0.3 }) })
  add({ id: 'propTest2.fisher', group: 'Basic statistics', label: 'propTest2 Fisher 120/400 vs 90/380', n: 780, py: 'scipy', setup: () => () => A.propTest2(120, 400, 90, 380, { method: 'fisher' }) })
  add({ id: 'poissonRateTest1', group: 'Basic statistics', label: 'poissonRateTest1 (exact) 120 events', n: 120, py: 'scipy', setup: () => () => A.poissonRateTest1(120, 100, { lambda0: 1 }) })
  add({ id: 'varTest1', group: 'Basic statistics', label: `varTest1 (χ²) n=${n}`, n, py: 'scipy', setup: () => { const x = normal(n, 0, 2); return () => A.varTest1(x, { sigma0: 2 }) } })
  add({ id: 'varTest1.bonett', group: 'Basic statistics', label: `varTest1 (Bonett) n=${n}`, n, setup: () => { const x = normal(n, 0, 2); return () => A.varTest1(x, { sigma0: 2, method: 'bonett' }) } })
  add({ id: 'corrTest.pearson', group: 'Basic statistics', label: `corrTest Pearson n=${n}`, n, py: 'scipy', setup: () => { const a = normal(n); const b = a.map((v) => 0.5 * v + g.normal(1)[0]!); return () => A.corrTest(a, b) } })
  add({ id: 'corrTest.spearman', group: 'Basic statistics', label: `corrTest Spearman n=${n}`, n, py: 'scipy', setup: () => { const a = normal(n); const b = a.map((v) => 0.5 * v + g.normal(1)[0]!); return () => A.corrTest(a, b, { method: 'spearman' }) } })
  add({ id: 'grubbs', group: 'Basic statistics', label: `grubbs n=${n}`, n, py: 'numpy', setup: () => { const x = normal(n); return () => A.grubbs(x) } })
  add({ id: 'dixon', group: 'Basic statistics', label: 'dixon n=25 (cached null distribution)', n: 25, setup: () => { const x = normal(25); A.dixon(x); return () => A.dixon(x) } })
  add({ id: 'anova', group: 'ANOVA', label: `anova 5 × ${N(20_000)}`, n: N(100_000), py: 'scipy', setup: () => { const gr = groups(5, N(20_000)); return () => A.anova(gr) } })
  add({ id: 'levene', group: 'ANOVA', label: `levene 5 × ${N(20_000)}`, n: N(100_000), py: 'scipy', setup: () => { const gr = groups(5, N(20_000)); return () => A.levene(gr) } })
  add({ id: 'bartlett', group: 'ANOVA', label: `bartlett 5 × ${N(20_000)}`, n: N(100_000), py: 'scipy', setup: () => { const gr = groups(5, N(20_000)); return () => A.bartlett(gr) } })
  add({ id: 'bonett', group: 'ANOVA', label: `bonett 5 × ${N(20_000)}`, n: N(100_000), setup: () => { const gr = groups(5, N(20_000)); return () => A.bonett(gr) } })
  add({ id: 'chi2test', group: 'Tables', label: 'chi2test 20×20 table', n: 400, py: 'scipy', setup: () => { const t = Array.from({ length: 20 }, () => Array.from(g.integer(20, 5, 100))); return () => A.chi2test(t) } })
  add({ id: 'chi2gof', group: 'Tables', label: 'chi2gof 50 categories', n: 50, py: 'scipy', setup: () => { const o = Array.from(g.integer(50, 20, 100)); return () => A.chi2gof(o) } })
  add({ id: 'crosstab', group: 'Tables', label: `crosstab n=${N(1_000_000)}`, n: N(1_000_000), py: 'pandas', setup: () => { const a = levels(N(1_000_000), 10, 'a'); const b = levels(N(1_000_000), 8, 'b').map((v, i) => (i % 3 ? v : 'b0')); return () => A.crosstab(a, b) } })
}

// ---- normality ------------------------------------------------------------------------------------------------
{
  const n = N(100_000)
  add({ id: 'andersonDarling', group: 'Normality', label: `andersonDarling n=${n}`, n, py: 'scipy', setup: () => { const x = normal(n); return () => A.andersonDarling(x) } })
  add({ id: 'shapiroWilk', group: 'Normality', label: 'shapiroWilk n=5000', n: 5000, py: 'scipy', setup: () => { const x = normal(5000); return () => A.shapiroWilk(x) } })
  add({ id: 'ryanJoiner', group: 'Normality', label: `ryanJoiner n=${n}`, n, setup: () => { const x = normal(n); return () => A.ryanJoiner(x) } })
  add({ id: 'kolmogorovSmirnov', group: 'Normality', label: `kolmogorovSmirnov (Lilliefors) n=${n}`, n, py: 'scipy', setup: () => { const x = normal(n); return () => A.kolmogorovSmirnov(x) } })
  add({ id: 'individualDistributionID', group: 'Normality', label: `individualDistributionID n=${N(10_000)}`, n: N(10_000), py: 'scipy', setup: () => { const x = Array.from(g.weibull(N(10_000), 1.8, 12)); return () => A.individualDistributionID(x) } })
}

// ---- multiple comparisons ---------------------------------------------------------------------------------------
{
  const gr = () => groups(6, N(5000))
  add({ id: 'tukeyHSD', group: 'ANOVA', label: `tukeyHSD 6 × ${N(5000)}`, n: N(30_000), py: 'scipy', setup: () => { const x = gr(); return () => A.tukeyHSD(x) } })
  add({ id: 'fisherLSD', group: 'ANOVA', label: `fisherLSD 6 × ${N(5000)}`, n: N(30_000), setup: () => { const x = gr(); return () => A.fisherLSD(x) } })
  add({ id: 'dunnett', group: 'ANOVA', label: `dunnett 6 × ${N(5000)}`, n: N(30_000), py: 'scipy', setup: () => { const x = gr(); return () => A.dunnett(x, { control: 'g0' }) } })
  add({ id: 'hsuMCB', group: 'ANOVA', label: `hsuMCB 6 × ${N(5000)}`, n: N(30_000), setup: () => { const x = gr(); return () => A.hsuMCB(x) } })
  add({ id: 'gamesHowell', group: 'ANOVA', label: `gamesHowell 6 × ${N(5000)}`, n: N(30_000), setup: () => { const x = gr(); return () => A.gamesHowell(x) } })
  add({ id: 'equalVariances', group: 'ANOVA', label: `equalVariances (Levene) 6 × ${N(5000)}`, n: N(30_000), py: 'scipy', setup: () => { const x = gr(); return () => A.equalVariances(x) } })
}

// ---- nonparametrics ------------------------------------------------------------------------------------------------
{
  const n = N(20_000)
  add({ id: 'mannWhitney', group: 'Nonparametrics', label: `mannWhitney (asymptotic) ${n}+${n}`, n: 2 * n, py: 'scipy', setup: () => { const a = normal(n); const b = normal(n, 0.1); return () => A.mannWhitney(a, b) } })
  add({ id: 'mannWhitney.exact', group: 'Nonparametrics', label: 'mannWhitney (exact) 40+40', n: 80, py: 'scipy', setup: () => { const a = normal(40); const b = normal(40, 0.3); return () => A.mannWhitney(a, b, { method: 'exact' }) } })
  add({ id: 'kruskal', group: 'Nonparametrics', label: `kruskal 5 × ${N(20_000)}`, n: N(100_000), py: 'scipy', setup: () => { const x = groups(5, N(20_000)); return () => A.kruskal(x) } })
  add({ id: 'signTest', group: 'Nonparametrics', label: `signTest n=${N(100_000)}`, n: N(100_000), py: 'scipy', setup: () => { const x = normal(N(100_000)); return () => A.signTest(x) } })
  add({ id: 'wilcoxonSigned', group: 'Nonparametrics', label: `wilcoxonSigned (asymptotic) n=${n}`, n, py: 'scipy', setup: () => { const x = normal(n, 0.05); return () => A.wilcoxonSigned(x) } })
  add({ id: 'wilcoxonSigned.exact', group: 'Nonparametrics', label: 'wilcoxonSigned (exact) n=50', n: 50, py: 'scipy', setup: () => { const x = normal(50, 0.2); return () => A.wilcoxonSigned(x, { method: 'exact' }) } })
  add({ id: 'moodMedian', group: 'Nonparametrics', label: `moodMedian 5 × ${N(20_000)}`, n: N(100_000), py: 'scipy', setup: () => { const x = groups(5, N(20_000)); return () => A.moodMedian(x) } })
  add({ id: 'friedman', group: 'Nonparametrics', label: `friedman ${N(5000)} blocks × 5`, n: N(25_000), py: 'scipy', setup: () => { const t = Array.from({ length: N(5000) }, () => Array.from(g.normal(5))); return () => A.friedman(t) } })
  add({ id: 'runsTest', group: 'Nonparametrics', label: `runsTest n=${N(1_000_000)}`, n: N(1_000_000), py: 'numpy', setup: () => { const x = normal(N(1_000_000)); return () => A.runsTest(x) } })
}

// ---- equivalence & power -----------------------------------------------------------------------------------------------
{
  const n = N(100_000)
  add({ id: 'tost1', group: 'Equivalence / power', label: `tost1 n=${n}`, n, py: 'numpy', setup: () => { const x = normal(n, 0.1, 1); return () => A.tost1(x, { limits: [-0.5, 0.5] }) } })
  add({ id: 'tost2', group: 'Equivalence / power', label: `tost2 n=${n}+${n}`, n, py: 'numpy', setup: () => { const a = normal(n); const b = normal(n, 0.1); return () => A.tost2(a, b, { limits: [-0.5, 0.5] }) } })
  add({ id: 'power.t', group: 'Equivalence / power', label: 'power 2-sample t (solve n) × 50', n: 50, setup: () => () => { let s = 0; for (let i = 0; i < 50; i++) s += A.power({ test: '2-sample t', effect: 0.2 + i * 0.01, power: 0.8 }).n; return s } })
  add({ id: 'power.anova', group: 'Equivalence / power', label: 'power one-way ANOVA (solve effect) × 20', n: 20, setup: () => () => { let s = 0; for (let i = 0; i < 20; i++) s += A.power({ test: 'one-way anova', groups: 4, n: 10 + i, power: 0.8 }).effect; return s } })
}

// ---- descriptive / plot data -------------------------------------------------------------------------------------------
{
  const n = N(1_000_000)
  add({ id: 'descriptiveStats', group: 'Descriptive', label: `descriptiveStats n=${n}`, n, py: 'pandas', setup: () => { const x = normal(n, 50, 5); return () => A.descriptiveStats(x) } })
  add({ id: 'descriptiveStats.by', group: 'Descriptive', label: `descriptiveStats by 10 groups n=${n}`, n, py: 'pandas', setup: () => { const x = normal(n, 50, 5); const by = levels(n, 10); return () => A.descriptiveStats(x, { by }) } })
  add({ id: 'graphicalSummary', group: 'Descriptive', label: `graphicalSummary n=${N(100_000)}`, n: N(100_000), setup: () => { const x = normal(N(100_000)); return () => A.graphicalSummary(x) } })
  add({ id: 'poissonGof', group: 'Descriptive', label: `poissonGof n=${n}`, n, py: 'numpy', setup: () => { const x = Array.from(g.poisson(n, 3)); return () => A.poissonGof(x) } })
  add({ id: 'boxplotStats', group: 'Descriptive', label: `boxplotStats n=${n}`, n, py: 'numpy', setup: () => { const x = normal(n); return () => A.boxplotStats(x) } })
  add({ id: 'mainEffectsPlot', group: 'Descriptive', label: `mainEffectsPlot 3 factors n=${n}`, n, py: 'pandas', setup: () => { const y = normal(n); const f = { a: levels(n, 3, 'a'), b: levels(n, 4, 'b'), c: levels(n, 5, 'c') }; return () => A.mainEffectsPlot(y, f) } })
  add({ id: 'interactionPlot', group: 'Descriptive', label: `interactionPlot 4×5 n=${n}`, n, py: 'pandas', setup: () => { const y = normal(n); const a = levels(n, 4, 'a'); const b = levels(n, 5, 'b'); return () => A.interactionPlot(y, a, b) } })
  add({ id: 'intervalPlot', group: 'Descriptive', label: `intervalPlot 10 groups n=${n}`, n, py: 'pandas', setup: () => { const y = normal(n); const by = levels(n, 10); return () => A.intervalPlot(y, by) } })
  add({ id: 'ecdf', group: 'Descriptive', label: `ecdf n=${n}`, n, py: 'numpy', setup: () => { const x = normal(n); return () => A.ecdf(x) } })
  add({ id: 'dotplot', group: 'Descriptive', label: `dotplot n=${n}`, n, py: 'numpy', setup: () => { const x = normal(n); return () => A.dotplot(x) } })
  add({ id: 'causeAndEffect', group: 'Descriptive', label: 'causeAndEffect 6 categories × 5 causes', n: 30, setup: () => { const spec = { effect: 'Defects', categories: Object.fromEntries(['Man', 'Machine', 'Method', 'Material', 'Measurement', 'Environment'].map((c) => [c, ['a', 'b', 'c', 'd', 'e']])) }; return () => A.causeAndEffect(spec) } })
}

// ---- regression ---------------------------------------------------------------------------------------------------------
{
  const n = N(100_000)
  const p = 10
  add({ id: 'ols', group: 'Regression', label: `ols n=${n}, p=${p} (full diagnostics)`, n, py: 'numpy', setup: () => { const X = rowsOf(n, p); const y = linear(X, Array.from({ length: p }, (_, j) => 0.5 - 0.1 * j)); const cols = Array.from({ length: p }, (_, j) => X.map((r) => r[j]!)); return () => A.ols(y, cols) } })
  add({ id: 'fittedLine', group: 'Regression', label: `fittedLine (quadratic) n=${n}`, n, py: 'numpy', setup: () => { const x = normal(n); const y = x.map((v) => 1 + 2 * v - 0.5 * v * v + g.normal(1)[0]!); return () => A.fittedLine(x, y, { degree: 2 }) } })
  add({ id: 'stepwise', group: 'Regression', label: `stepwise n=${N(10_000)}, p=8`, n: N(10_000), setup: () => { const X = rowsOf(N(10_000), 8); const y = linear(X, [1, -1, 0.5, 0, 0, 0, 0, 0]); const cols = Array.from({ length: 8 }, (_, j) => X.map((r) => r[j]!)); return () => A.stepwise(y, cols) } })
  add({ id: 'bestSubsets', group: 'Regression', label: `bestSubsets n=${N(10_000)}, p=8 (255 fits)`, n: N(10_000), setup: () => { const X = rowsOf(N(10_000), 8); const y = linear(X, [1, -1, 0.5, 0, 0, 0, 0, 0]); const cols = Array.from({ length: 8 }, (_, j) => X.map((r) => r[j]!)); return () => A.bestSubsets(y, cols) } })
  add({ id: 'logit', group: 'Regression', label: `logit n=${n}, p=5`, n, py: 'scipy.optimize', setup: () => { const X = rowsOf(n, 5); const yb = X.map((r) => (g.next() < 1 / (1 + Math.exp(-(0.3 + r[0]! - 0.5 * r[1]!))) ? 1 : 0)); const cols = Array.from({ length: 5 }, (_, j) => X.map((r) => r[j]!)); return () => A.logit(yb, cols) } })
  add({ id: 'poissonRegression', group: 'Regression', label: `poissonRegression n=${n}, p=5`, n, py: 'scipy.optimize', setup: () => { const X = rowsOf(n, 5); const yc = X.map((r) => g.poisson(1, Math.exp(1 + 0.3 * r[0]! - 0.2 * r[1]!))[0]!); const cols = Array.from({ length: 5 }, (_, j) => X.map((r) => r[j]!)); return () => A.poissonRegression(yc, cols) } })
  add({ id: 'ologit', group: 'Regression', label: `ologit n=${N(10_000)}, p=3, 4 levels`, n: N(10_000), setup: () => { const X = rowsOf(N(10_000), 3); const yo = X.map((r) => { const z = r[0]! - 0.5 * r[1]! + g.normal(1)[0]!; return z < -1 ? 0 : z < 0 ? 1 : z < 1 ? 2 : 3 }); const cols = Array.from({ length: 3 }, (_, j) => X.map((r) => r[j]!)); return () => A.ologit(yo, cols) } })
  add({ id: 'mlogit', group: 'Regression', label: `mlogit n=${N(10_000)}, p=3, 3 classes`, n: N(10_000), setup: () => { const X = rowsOf(N(10_000), 3); const ym = X.map((r) => { const e1 = Math.exp(0.3 + r[0]!); const e2 = Math.exp(-0.4 - 0.8 * r[1]!); const u = g.next() * (1 + e1 + e2); return u < 1 ? 'a' : u < 1 + e1 ? 'b' : 'c' }); const cols = Array.from({ length: 3 }, (_, j) => X.map((r) => r[j]!)); return () => A.mlogit(ym, cols) } })
  add({ id: 'linearModel', group: 'Regression', label: `linearModel 'y ~ a*b + x' n=${n}`, n, py: 'numpy', setup: () => { const a = levels(n, 3, 'a'); const b = levels(n, 4, 'b'); const x = normal(n); const y = x.map((v, i) => v + (a[i] === 'a0' ? 1 : 0) + g.normal(1)[0]!); return () => A.linearModel({ y, a, b, x }, 'y ~ a*b + x') } })
  add({ id: 'nls', group: 'Regression', label: `nls (2 params, LM) n=${N(10_000)}`, n: N(10_000), py: 'scipy', setup: () => { const x = Array.from(g.uniform(N(10_000), 0, 800)); const y = x.map((v) => 238 * (1 - Math.exp(-5.5e-4 * v)) + 0.1 * g.normal(1)[0]!); return () => A.nls((t, [a, b]) => a! * (1 - Math.exp(-b! * (t as number))), x, y, { start: [500, 1e-4] }) } })
  add({ id: 'orthogonalRegression', group: 'Regression', label: `orthogonalRegression (jackknife SE) n=${N(5000)}`, n: N(5000), py: 'scipy', setup: () => { const x = normal(N(5000), 10, 3); const y = x.map((v) => 2 + 1.1 * v + g.normal(1)[0]!); return () => A.orthogonalRegression(x, y) } })
  add({ id: 'pls', group: 'Regression', label: `pls n=${N(2000)}, p=10, 3 comp (LOO CV)`, n: N(2000), setup: () => { const X = rowsOf(N(2000), 10); const y = linear(X, Array.from({ length: 10 }, (_, j) => 1 - 0.1 * j)); const cols = Array.from({ length: 10 }, (_, j) => X.map((r) => r[j]!)); return () => A.pls(y, cols, { components: 3 }) }, repeat: 3 })
  add({ id: 'stabilityStudy', group: 'Regression', label: 'stabilityStudy 5 batches × 8 times', n: 40, setup: () => { const time = Array.from({ length: 40 }, (_, i) => [0, 3, 6, 9, 12, 18, 24, 36][i % 8]!); const batch = time.map((_, i) => `B${Math.floor(i / 8)}`); const y = time.map((t, i) => 100 - 0.2 * t - 0.1 * Math.floor(i / 8) + 0.3 * g.normal(1)[0]!); return () => A.stabilityStudy(y, time, batch, { lsl: 90 }) } })
}

// ---- SPC / quality ----------------------------------------------------------------------------------------------------
{
  const n = N(1_000_000)
  add({ id: 'controlChart.imr', group: 'Control charts', label: `controlChart I-MR (Nelson 1–8) n=${n}`, n, py: 'numpy', setup: () => { const x = normal(n, 10, 1); return () => A.controlChart(x, { type: 'i-mr' }) } })
  add({ id: 'controlChart.xbar', group: 'Control charts', label: `controlChart X̄-R subgroups of 5 n=${n}`, n, py: 'numpy', setup: () => { const x = normal(n, 10, 1); return () => A.controlChart(x, { type: 'xbar-r', subgroup: 5 }) } })
  add({ id: 'controlChart.p', group: 'Control charts', label: `controlChart P n=${N(100_000)}`, n: N(100_000), py: 'numpy', setup: () => { const d = Array.from(g.binomial(N(100_000), 100, 0.05)); const sizes = new Array(N(100_000)).fill(100); return () => A.controlChart(d, { type: 'p', sizes }) } })
  add({ id: 'ewma', group: 'Control charts', label: `ewma n=${n}`, n, py: 'pandas', setup: () => { const x = normal(n); return () => A.ewma(x, { lambda: 0.2 }) } })
  add({ id: 'cusum', group: 'Control charts', label: `cusum n=${n}`, n, py: 'numpy', setup: () => { const x = normal(n); return () => A.cusum(x) } })
  add({ id: 'movingAverage', group: 'Control charts', label: `movingAverage span 5 n=${n}`, n, py: 'pandas', setup: () => { const x = normal(n); return () => A.movingAverage(x, { span: 5 }) } })
  add({ id: 'gChart', group: 'Control charts', label: `gChart n=${N(100_000)}`, n: N(100_000), py: 'scipy', setup: () => { const x = Array.from(g.integer(N(100_000), 0, 200)); return () => A.gChart(x) } })
  add({ id: 'tChart', group: 'Control charts', label: `tChart (Weibull) n=${N(100_000)}`, n: N(100_000), py: 'scipy', setup: () => { const x = Array.from(g.weibull(N(100_000), 1.5, 20)); return () => A.tChart(x) } })
  add({ id: 't2Chart', group: 'Control charts', label: `t2Chart p=5 n=${N(100_000)}`, n: N(100_000), py: 'numpy', setup: () => { const X = rowsOf(N(100_000), 5); return () => A.t2Chart(X) } })
  add({ id: 'mewma', group: 'Control charts', label: `mewma p=3 n=${N(100_000)} (h given)`, n: N(100_000), py: 'numpy', setup: () => { const X = rowsOf(N(100_000), 3); return () => A.mewma(X, { h: 10 }) } })
  add({ id: 'mewma.calibrate', group: 'Control charts', label: 'mewma h calibration (ARL₀ 200, p=3)', n: 200, setup: () => { const X = rowsOf(200, 3); return () => A.mewma(X, { lambda: 0.1 }) }, repeat: 3 })
  add({ id: 'generalizedVarianceChart', group: 'Control charts', label: `generalizedVarianceChart p=3, ${N(10_000)} subgroups × 5`, n: N(50_000), py: 'numpy', setup: () => { const X = rowsOf(N(50_000), 3); const sg = X.map((_, i) => Math.floor(i / 5)); return () => A.generalizedVarianceChart(X, sg) } })
  add({ id: 'capability', group: 'Capability', label: `capability n=${n}`, n, py: 'numpy', setup: () => { const x = normal(n, 10, 1); return () => A.capability(x, { lsl: 7, usl: 13, subgroup: 5 }) } })
  add({ id: 'capabilitySixpack', group: 'Capability', label: `capabilitySixpack n=${N(100_000)}`, n: N(100_000), setup: () => { const x = normal(N(100_000), 10, 1); return () => A.capabilitySixpack(x, { lsl: 7, usl: 13, subgroup: 5 }) } })
  add({ id: 'boxCoxLambda', group: 'Capability', label: `boxCoxLambda n=${N(100_000)}`, n: N(100_000), py: 'scipy', setup: () => { const x = Array.from(g.lognormal(N(100_000), 1, 0.4)); return () => A.boxCoxLambda(x) } })
  add({ id: 'johnsonFit', group: 'Capability', label: `johnsonFit n=${N(100_000)}`, n: N(100_000), setup: () => { const x = Array.from(g.lognormal(N(100_000), 1, 0.4)); return () => A.johnsonFit(x) } })
  add({ id: 'weibullFit', group: 'Capability', label: `weibullFit n=${N(100_000)}`, n: N(100_000), py: 'scipy', setup: () => { const x = Array.from(g.weibull(N(100_000), 1.8, 12)); return () => A.weibullFit(x) } })
  add({ id: 'toleranceInterval', group: 'Capability', label: `toleranceInterval n=${N(100_000)}`, n: N(100_000), setup: () => { const x = normal(N(100_000)); return () => A.toleranceInterval(x) } })
  add({ id: 'gageRR', group: 'Measurement systems', label: 'gageRR crossed 10 parts × 3 operators × 3', n: 90, setup: () => { const part = levels(90, 10, 'P'); const operator = Array.from({ length: 90 }, (_, i) => `O${Math.floor(i / 30)}`); const m = part.map((p, i) => Number(p.slice(1)) + (operator[i] === 'O1' ? 0.2 : 0) + 0.3 * g.normal(1)[0]!); return () => A.gageRR({ part, operator, measurement: m }) } })
  add({ id: 'gageLinearity', group: 'Measurement systems', label: 'gageLinearity 5 refs × 12', n: 60, setup: () => { const ref = Array.from({ length: 60 }, (_, i) => 2 + (i % 5) * 2); const m = ref.map((r) => r + 0.02 * r + 0.1 * g.normal(1)[0]!); return () => A.gageLinearity(ref, m, { processVariation: 10 }) } })
  add({ id: 'gageType1', group: 'Measurement systems', label: 'gageType1 n=50', n: 50, setup: () => { const m = normal(50, 10, 0.05); return () => A.gageType1(m, { reference: 10, tolerance: 1 }) } })
  add({ id: 'attributeAgreement', group: 'Measurement systems', label: 'attributeAgreement Fleiss κ 3 raters × 500', n: 1500, setup: () => { const truth = Array.from(g.integer(500, 0, 2)); const r = Array.from({ length: 3 }, () => truth.map((t) => (g.next() < 0.85 ? t : g.integer(1, 0, 2)[0]!))); return () => A.attributeAgreement(r, { method: 'fleiss' }) } })
  add({ id: 'acceptanceSampling', group: 'Measurement systems', label: 'acceptanceSampling n=125,c=3 OC curve', n: 125, py: 'scipy', setup: () => () => A.acceptanceSampling({ type: 'attributes', n: 125, c: 3 }) })
  add({ id: 'pareto', group: 'Quality tools', label: `pareto n=${n}, 30 categories`, n, py: 'pandas', setup: () => { const c = levels(n, 30, 'cat'); return () => A.pareto(c) } })
  add({ id: 'runChart', group: 'Quality tools', label: `runChart n=${N(100_000)}`, n: N(100_000), setup: () => { const x = normal(N(100_000)); return () => A.runChart(x) } })
  add({ id: 'multiVari', group: 'Quality tools', label: `multiVari 3 factors n=${N(100_000)}`, n: N(100_000), py: 'pandas', setup: () => { const m = normal(N(100_000)); const f = [levels(N(100_000), 3, 'a'), levels(N(100_000), 4, 'b'), levels(N(100_000), 2, 'c')]; return () => A.multiVari(m, f) } })
  add({ id: 'symmetryTest', group: 'Quality tools', label: `symmetryTest n=${N(100_000)}`, n: N(100_000), setup: () => { const x = normal(N(100_000)); return () => A.symmetryTest(x) } })
}

// ---- time series ---------------------------------------------------------------------------------------------------------
{
  const n = N(100_000)
  const series = (m: number) => Array.from({ length: m }, (_, i) => 10 + 0.01 * i + 2 * Math.sin((2 * Math.PI * i) / 12) + g.normal(1)[0]!)
  add({ id: 'trendAnalysis', group: 'Time series', label: `trendAnalysis quadratic n=${n}`, n, py: 'numpy', setup: () => { const y = series(n); return () => A.trendAnalysis(y, { model: 'quadratic', horizon: 12 }) } })
  add({ id: 'decompose', group: 'Time series', label: `decompose (12) n=${n}`, n, setup: () => { const y = series(n); return () => A.decompose(y, { seasonLength: 12 }) } })
  add({ id: 'stl', group: 'Time series', label: `stl (12) n=${N(10_000)}`, n: N(10_000), setup: () => { const y = series(N(10_000)); return () => A.stl(y, { seasonLength: 12 }) } })
  add({ id: 'ets', group: 'Time series', label: `ets winters-add n=${N(10_000)}`, n: N(10_000), setup: () => { const y = series(N(10_000)); return () => A.ets(y, { method: 'winters-add', seasonLength: 12, horizon: 12 }) } })
  add({ id: 'acf', group: 'Time series', label: `acf 40 lags n=${n}`, n, py: 'numpy', setup: () => { const y = series(n); return () => A.acf(y, { maxLag: 40 }) } })
  add({ id: 'pacf', group: 'Time series', label: `pacf 40 lags n=${n}`, n, setup: () => { const y = series(n); return () => A.pacf(y, { maxLag: 40 }) } })
  add({ id: 'ccf', group: 'Time series', label: `ccf ±20 lags n=${n}`, n, py: 'numpy', setup: () => { const x = series(n); const y = x.map((v, i) => (i > 2 ? x[i - 2]! : v) + g.normal(1)[0]!); return () => A.ccf(x, y, { maxLag: 20 }) } })
  add({ id: 'ljungBox', group: 'Time series', label: `ljungBox 20 lags n=${n}`, n, setup: () => { const y = normal(n); return () => A.ljungBox(y, { lags: 20 }) } })
  add({ id: 'arima', group: 'Time series', label: 'arima (1,1,1) CSS-ML n=2000', n: 2000, setup: () => { const y = series(2000); return () => A.arima(y, { p: 1, d: 1, q: 1, horizon: 12 }) }, repeat: 3 })
  add({ id: 'arima.seasonal', group: 'Time series', label: 'arima (1,1,1)(1,1,1)₁₂ n=600', n: 600, setup: () => { const y = series(600); return () => A.arima(y, { p: 1, d: 1, q: 1, seasonal: { P: 1, D: 1, Q: 1, period: 12 }, horizon: 12 }) }, repeat: 3 })
  add({ id: 'autoArima', group: 'Time series', label: 'autoArima n=500', n: 500, setup: () => { const y = series(500); return () => A.autoArima(y, { horizon: 12 }) }, repeat: 2 })
  add({ id: 'periodogram', group: 'Time series', label: `periodogram n=${N(20_000)}`, n: N(20_000), py: 'scipy', setup: () => { const y = series(N(20_000)); return () => A.periodogram(y) } })
  add({ id: 'cumulativePeriodogram', group: 'Time series', label: `cumulativePeriodogram n=${N(20_000)}`, n: N(20_000), setup: () => { const y = normal(N(20_000)); return () => A.cumulativePeriodogram(y) } })
}

// ---- DOE ---------------------------------------------------------------------------------------------------------------
{
  add({ id: 'fullFactorial', group: 'DOE', label: 'fullFactorial 12 factors (4096 runs)', n: 4096, setup: () => () => A.fullFactorial(Array.from({ length: 12 }, (_, i) => `F${i}`)) })
  add({ id: 'fractionalFactorial', group: 'DOE', label: 'fractionalFactorial 2^(8-3)', n: 32, setup: () => () => A.fractionalFactorial(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], ['ABCF', 'ABDG', 'ACDEH']) })
  add({ id: 'aliasStructure', group: 'DOE', label: 'aliasStructure 2^(10-4), order ≤ 3', n: 64, setup: () => () => A.aliasStructure(10, ['ABCG', 'BCDH', 'ACDI', 'ABDEJ'], { maxOrder: 3 }) })
  add({ id: 'plackettBurman', group: 'DOE', label: 'plackettBurman 23 factors', n: 24, setup: () => () => A.plackettBurman(Array.from({ length: 23 }, (_, i) => `F${i}`)) })
  add({ id: 'ccd', group: 'DOE', label: 'ccd 6 factors', n: 90, setup: () => () => A.ccd(['A', 'B', 'C', 'D', 'E', 'F']) })
  add({ id: 'boxBehnken', group: 'DOE', label: 'boxBehnken 7 factors', n: 62, setup: () => () => A.boxBehnken(['A', 'B', 'C', 'D', 'E', 'F', 'G']) })
  add({ id: 'taguchi', group: 'DOE', label: 'taguchi L243', n: 243, setup: () => () => A.taguchi('L243') })
  add({ id: 'definitiveScreening', group: 'DOE', label: 'definitiveScreening 10 factors', n: 21, setup: () => () => A.definitiveScreening(Array.from({ length: 10 }, (_, i) => `F${i}`)) })
  add({ id: 'mixtureDesign', group: 'DOE', label: 'mixtureDesign 4 components lattice degree 3', n: 20, setup: () => () => A.mixtureDesign(['A', 'B', 'C', 'D'], { degree: 3 }) })
  add({ id: 'analyzeEffects', group: 'DOE', label: 'analyzeEffects 2^7 with interactions', n: 128, setup: () => { const d = A.fullFactorial(['A', 'B', 'C', 'D', 'E', 'F', 'G']); const y = d.matrix.map((r) => 10 + 2 * r[0]! - r[1]! + 0.5 * r[0]! * r[1]! + g.normal(1)[0]!); return () => A.analyzeEffects(d, y, { interactions: true }) } })
  add({ id: 'analyzeDoe', group: 'DOE', label: 'analyzeDoe 2^6 × 2 replicates (linearModel)', n: 128, setup: () => { const base = A.fullFactorial(['A', 'B', 'C', 'D', 'E', 'F']); const d = { ...base, matrix: [...base.matrix, ...base.matrix] }; const y = d.matrix.map((r) => 10 + 2 * r[0]! - r[1]! + g.normal(1)[0]!); return () => A.analyzeDoe(d, y) } })
  add({ id: 'analyzeTaguchi', group: 'DOE', label: 'analyzeTaguchi L27 × 3 replicates', n: 81, setup: () => { const d = A.taguchi('L27'); const resp = d.matrix.map(() => Array.from(g.normal(3, 50, 2))); return () => A.analyzeTaguchi(d, resp, { snRatio: 'larger' }) } })
  add({ id: 'analyzeMixture', group: 'DOE', label: 'analyzeMixture 3 components quadratic', n: 20, setup: () => { const d = A.mixtureDesign(['A', 'B', 'C'], { degree: 3 }); const y = d.matrix.map((r) => 5 * r[0]! + 3 * r[1]! + 4 * r[2]! + 2 * r[0]! * r[1]! + 0.2 * g.normal(1)[0]!); return () => A.analyzeMixture(d, y, { model: 'quadratic' }) } })
  add({ id: 'responseOptimizer', group: 'DOE', label: 'responseOptimizer 2 responses, 3 factors', n: 3, setup: () => { const goals: A.OptimizerGoal[] = [{ predict: (x) => 10 - (x[0]! - 0.3) ** 2 - (x[1]! - 0.1) ** 2, goal: 'maximize', lower: 5, upper: 10 }, { predict: (x) => (x[2]! + 0.2) ** 2, goal: 'minimize', lower: 0, upper: 1 }]; return () => A.responseOptimizer(goals, { bounds: [[-1, 1], [-1, 1], [-1, 1]] }) } })
  add({ id: 'nestedAnova', group: 'DOE', label: `nestedAnova 3 levels n=${N(10_000)}`, n: N(10_000), setup: () => { const n = N(10_000); const a = levels(n, 5, 'a'); const b = a.map((v, i) => `${v}${i % 4}`); const y = normal(n); return () => A.nestedAnova(y, [a, b]) } })
}

// ---- reliability ------------------------------------------------------------------------------------------------------
{
  const n = N(10_000)
  const life = (m: number) => { const t = Array.from(g.weibull(m, 1.5, 100)); const c = t.map((v) => (v > 150 ? 1 : 0)); return { t: t.map((v) => Math.min(v, 150)), c } }
  add({ id: 'reliabilityFit', group: 'Reliability', label: `reliabilityFit Weibull censored n=${n}`, n, py: 'scipy', setup: () => { const { t, c } = life(n); return () => A.reliabilityFit(t, { censor: c }) } })
  add({ id: 'kaplanMeier', group: 'Reliability', label: `kaplanMeier n=${N(100_000)}`, n: N(100_000), py: 'numpy', setup: () => { const { t, c } = life(N(100_000)); return () => A.kaplanMeier(t, { censor: c }) } })
  add({ id: 'logRank', group: 'Reliability', label: `logRank 3 groups n=${N(100_000)}`, n: N(100_000), setup: () => { const { t, c } = life(N(100_000)); const gr = levels(N(100_000), 3); return () => A.logRank(t, gr, { censor: c }) } })
  add({ id: 'coxPH', group: 'Reliability', label: `coxPH n=${n}, p=3`, n, setup: () => { const { t, c } = life(n); const X = rowsOf(n, 3); return () => A.coxPH(t, X, { censor: c }) } })
  add({ id: 'fineGray', group: 'Reliability', label: `fineGray n=${N(3000)}, p=2`, n: N(3000), setup: () => { const m = N(3000); const { t, c } = life(m); const X = rowsOf(m, 2); const ev = c.map((v) => (v ? 0 : g.next() < 0.6 ? 1 : 2)); return () => A.fineGray(t, X, { eventType: ev, cause: 1 }) }, repeat: 3 })
  add({ id: 'lifeRegression', group: 'Reliability', label: `lifeRegression Weibull n=${n}, p=2`, n, py: 'scipy.optimize', setup: () => { const X = rowsOf(n, 2); const t = X.map((r) => Math.exp(4 + 0.5 * r[0]! - 0.3 * r[1]! + 0.5 * Math.log(-Math.log(g.next())))); const c = t.map((v) => (v > 200 ? 1 : 0)); const cols = [X.map((r) => r[0]!), X.map((r) => r[1]!)]; return () => A.lifeRegression(t.map((v) => Math.min(v, 200)), cols, { censor: c }) } })
  add({ id: 'altRegression', group: 'Reliability', label: `altRegression Arrhenius n=${n}`, n, setup: () => { const s = Array.from({ length: n }, (_, i) => [80, 100, 120][i % 3]!); const t = s.map((T) => Math.exp(2 + 5802 / (T + 273.15) + 0.4 * Math.log(-Math.log(g.next())))); return () => A.altRegression(t, s, { useStress: 50 }) } })
  add({ id: 'powerLawNHPP', group: 'Reliability', label: `powerLawNHPP n=${n} failures`, n, py: 'numpy', setup: () => { const t = Array.from(g.uniform(n, 0, 1000), (u) => 1000 * Math.sqrt(u / 1000)).sort((a, b) => a - b); return () => A.powerLawNHPP(t, { endTime: 1000 }) } })
  add({ id: 'probitAnalysis', group: 'Reliability', label: 'probitAnalysis 8 doses × 100', n: 800, py: 'scipy.optimize', setup: () => { const dose = [1, 2, 3, 4, 5, 6, 7, 8]; const tr = dose.map(() => 100); const ev = dose.map((d) => g.binomial(1, 100, A.normal().cdf(-3 + 0.8 * d))[0]!); return () => A.probitAnalysis(ev, tr, dose) } })
  add({ id: 'demonstrationTestPlan', group: 'Reliability', label: 'demonstrationTestPlan × 100', n: 100, setup: () => () => { let s = 0; for (let i = 0; i < 100; i++) s += A.demonstrationTestPlan({ reliability: 0.9, time: 1000, shape: 1.5, testTime: 1000 + i * 10 }).sampleSize; return s } })
  add({ id: 'estimationTestPlan', group: 'Reliability', label: 'estimationTestPlan × 20', n: 20, setup: () => () => { let s = 0; for (let i = 0; i < 20; i++) s += A.estimationTestPlan({ shape: 2, scale: 1000, ratio: 1.5 + i * 0.05, censorTime: 800 }).sampleSize; return s } })
  add({ id: 'probabilityPlot', group: 'Reliability', label: `probabilityPlot n=${n}`, n, setup: () => { const { t, c } = life(n); return () => A.probabilityPlot(t, { censor: c }) } })
}

// ---- multivariate ----------------------------------------------------------------------------------------------------
{
  const n = N(100_000)
  add({ id: 'pca', group: 'Multivariate', label: `pca n=${n}, p=10`, n, py: 'numpy', setup: () => { const X = rowsOf(n, 10); return () => A.pca(X) } })
  add({ id: 'factorAnalysis', group: 'Multivariate', label: `factorAnalysis ML + varimax n=${N(10_000)}, p=8, 2 factors`, n: N(10_000), setup: () => { const f1 = normal(N(10_000)); const f2 = normal(N(10_000)); const X = f1.map((a, i) => [a + 0.5 * g.normal(1)[0]!, 0.9 * a + 0.5 * g.normal(1)[0]!, 0.8 * a + 0.3 * f2[i]! + 0.5 * g.normal(1)[0]!, 0.7 * a + 0.6 * g.normal(1)[0]!, f2[i]! + 0.5 * g.normal(1)[0]!, 0.9 * f2[i]! + 0.5 * g.normal(1)[0]!, 0.8 * f2[i]! + 0.6 * g.normal(1)[0]!, 0.6 * f2[i]! + 0.7 * g.normal(1)[0]!]); return () => A.factorAnalysis(X, { nFactors: 2, method: 'ml' }) } })
  add({ id: 'promax', group: 'Multivariate', label: 'promax 20 × 3 loadings', n: 60, setup: () => { const L = rowsOf(20, 3).map((r) => r.map((v) => v * 0.3)); return () => A.promax(L) } })
  add({ id: 'kmeans', group: 'Multivariate', label: `kmeans k=4 n=${n}, p=5`, n, setup: () => { const X = rowsOf(n, 5).map((r, i) => r.map((v) => v + (i % 4) * 3)); return () => A.kmeans(X, { k: 4, seed: 1 }) }, repeat: 3 })
  add({ id: 'hclust', group: 'Multivariate', label: 'hclust average n=1500, p=5', n: 1500, py: 'scipy', setup: () => { const X = rowsOf(1500, 5); return () => A.hclust(X) }, repeat: 3 })
  add({ id: 'clusterVariables', group: 'Multivariate', label: `clusterVariables 30 variables n=${N(10_000)}`, n: N(10_000), py: 'scipy', setup: () => { const data = Object.fromEntries(Array.from({ length: 30 }, (_, j) => [`v${j}`, normal(N(10_000))])); return () => A.clusterVariables(data) } })
  add({ id: 'discriminant', group: 'Multivariate', label: `discriminant LDA 3 classes n=${N(10_000)}, p=5`, n: N(10_000), setup: () => { const X = rowsOf(N(10_000), 5); const y = X.map((_, i) => `c${i % 3}`); X.forEach((r, i) => (r[0] = r[0]! + (i % 3))); return () => A.discriminant(X, y) } })
  add({ id: 'correspondence', group: 'Multivariate', label: 'correspondence 30×30 table', n: 900, py: 'numpy', setup: () => { const t = Array.from({ length: 30 }, () => Array.from(g.integer(30, 1, 100))); return () => A.correspondence(t) } })
  add({ id: 'multipleCorrespondence', group: 'Multivariate', label: `multipleCorrespondence 4 variables n=${N(5000)}`, n: N(5000), py: 'numpy', setup: () => { const m = N(5000); const data = { a: levels(m, 3, 'a'), b: levels(m, 4, 'b').map((v, i) => (i % 7 ? v : 'b1')), c: levels(m, 5, 'c'), d: levels(m, 2, 'd') }; return () => A.multipleCorrespondence(data) }, repeat: 3 })
  add({ id: 'itemAnalysis', group: 'Multivariate', label: `itemAnalysis 20 items n=${N(10_000)}`, n: N(10_000), py: 'numpy', setup: () => { const lat = normal(N(10_000)); const data = Object.fromEntries(Array.from({ length: 20 }, (_, j) => [`i${j}`, lat.map((v) => v + (0.5 + 0.05 * j) * g.normal(1)[0]!)])); return () => A.itemAnalysis(data) } })
  add({ id: 'manova', group: 'Multivariate', label: `manova one-way 5 groups n=${N(10_000)}, p=4`, n: N(10_000), setup: () => { const X = rowsOf(N(10_000), 4); const gr = X.map((_, i) => `g${i % 5}`); return () => A.manova(X, gr) } })
  add({ id: 'manovaModel', group: 'Multivariate', label: `manovaModel 'a*b' n=${N(10_000)}, p=3`, n: N(10_000), py: 'numpy', setup: () => { const m = N(10_000); const data = { y1: normal(m), y2: normal(m), y3: normal(m), a: levels(m, 3, 'a'), b: levels(m, 4, 'b') }; return () => A.manovaModel(data, ['y1', 'y2', 'y3'], 'a*b') } })
  add({ id: 'mixedModel', group: 'Multivariate', label: `mixedModel RI 100 groups n=${N(10_000)}`, n: N(10_000), setup: () => { const m = N(10_000); const grp = levels(m, 100, 'g'); const x = normal(m); const eff = Array.from(g.normal(100, 0, 0.5)); const y = x.map((v, i) => 1 + 0.5 * v + eff[i % 100]! + g.normal(1)[0]!); return () => A.mixedModel(y, { fixed: x.map((v) => [v]), group: grp }) }, repeat: 3 })
  add({ id: 'glmm', group: 'Multivariate', label: `glmm binomial PQL 50 groups n=${N(5000)}`, n: N(5000), setup: () => { const m = N(5000); const grp = levels(m, 50, 'g'); const x = normal(m); const eff = Array.from(g.normal(50, 0, 0.5)); const y = x.map((v, i) => (g.next() < 1 / (1 + Math.exp(-(0.3 + v + eff[i % 50]!))) ? 1 : 0)); return () => A.glmm(y, { family: 'binomial', fixed: x.map((v) => [v]), group: grp }) }, repeat: 3 })
}

// ---- predictive -------------------------------------------------------------------------------------------------------
{
  const n = N(10_000)
  const mk = () => { const X = rowsOf(n, 5); const y = X.map((r) => (r[0]! > 0 ? 3 : 0) + r[1]! * r[2]! + 0.3 * g.normal(1)[0]!); return { X, y } }
  add({ id: 'cart', group: 'Predictive', label: `cart regression n=${n}, p=5`, n, setup: () => { const { X, y } = mk(); return () => A.cart(X, y, { maxDepth: 6 }) } })
  add({ id: 'randomForest', group: 'Predictive', label: `randomForest 30 trees n=${N(3000)}, p=5`, n: N(3000), setup: () => { const X = rowsOf(N(3000), 5); const y = X.map((r) => (r[0]! > 0 ? 3 : 0) + r[1]! * r[2]! + 0.3 * g.normal(1)[0]!); return () => A.randomForest(X, y, { nTrees: 30, maxDepth: 6 }) }, repeat: 2 })
  add({ id: 'treeNet', group: 'Predictive', label: `treeNet 50 trees n=${N(3000)}, p=5`, n: N(3000), setup: () => { const X = rowsOf(N(3000), 5); const y = X.map((r) => (r[0]! > 0 ? 3 : 0) + r[1]! * r[2]! + 0.3 * g.normal(1)[0]!); return () => A.treeNet(X, y, { nTrees: 50, maxDepth: 3 }) }, repeat: 2 })
  add({ id: 'mars', group: 'Predictive', label: `mars n=${N(3000)}, p=5`, n: N(3000), setup: () => { const X = rowsOf(N(3000), 5); const y = X.map((r) => Math.max(0, r[0]!) * 2 + r[1]! + 0.3 * g.normal(1)[0]!); return () => A.mars(X, y, { maxTerms: 10 }) }, repeat: 2 })
  add({ id: 'crossValidate.ols', group: 'Predictive', label: `crossValidate ols 5-fold n=${n}`, n, setup: () => { const { X, y } = mk(); return () => A.crossValidate(X, y, { model: 'ols' }) } })
  add({ id: 'autoModel', group: 'Predictive', label: `autoModel (ols, cart, rf 20) 3-fold n=${N(2000)}`, n: N(2000), setup: () => { const X = rowsOf(N(2000), 5); const y = X.map((r) => (r[0]! > 0 ? 3 : 0) + r[1]! + 0.3 * g.normal(1)[0]!); return () => A.autoModel(X, y, { folds: 3, models: ['ols', 'cart', 'random-forest'], nTrees: 20 }) }, repeat: 2 })
}

// ---- random data -------------------------------------------------------------------------------------------------------
{
  const n = N(1_000_000)
  add({ id: 'random.normal', group: 'Random data', label: `random.normal × ${n}`, n, py: 'numpy', setup: () => { const r = A.random(1); return () => r.normal(n) } })
  add({ id: 'random.gamma', group: 'Random data', label: `random.gamma × ${n}`, n, py: 'numpy', setup: () => { const r = A.random(1); return () => r.gamma(n, 2.5, 2) } })
  add({ id: 'random.poisson', group: 'Random data', label: `random.poisson × ${n}`, n, py: 'numpy', setup: () => { const r = A.random(1); return () => r.poisson(n, 4) } })
}

// ---- runner ---------------------------------------------------------------------------------------------------------
function timeIt(fn: () => unknown, repeat: number): { median: number; min: number } {
  fn() // warm-up (JIT)
  const times: number[] = []
  for (let i = 0; i < repeat; i++) {
    const t0 = performance.now()
    fn()
    times.push(performance.now() - t0)
  }
  times.sort((a, b) => a - b)
  return { median: times[Math.floor(times.length / 2)]!, min: times[0]! }
}

const rows: Row[] = []
const selected = ONLY ? cases.filter((c) => c.id.includes(ONLY)) : cases
console.log(`advanced bench: ${selected.length} cases, scale ${SCALE}, repeat ${REPEAT}`)
console.log(''.padEnd(58) + 'median ms'.padStart(12) + 'min ms'.padStart(12))
for (const c of selected) {
  let run: () => unknown
  try {
    run = c.setup()
  } catch (e) {
    console.log(`${c.id.padEnd(58)} setup failed: ${(e as Error).message}`)
    continue
  }
  try {
    const { median, min } = timeIt(run, c.repeat ?? REPEAT)
    rows.push({ id: c.id, group: c.group, label: c.label, n: c.n, msMedian: median, msMin: min, repeat: c.repeat ?? REPEAT })
    console.log(`${c.label.padEnd(58)}${median.toFixed(2).padStart(12)}${min.toFixed(2).padStart(12)}`)
  } catch (e) {
    console.log(`${c.id.padEnd(58)} failed: ${(e as Error).message}`)
  }
}
mkdirSync(new URL('../results/', import.meta.url), { recursive: true })
writeFileSync(OUT, JSON.stringify({ generatedAt: new Date().toISOString(), node: process.version, scale: SCALE, repeat: REPEAT, rows }, null, 1))
console.log(`\nwrote ${OUT.pathname}`)
