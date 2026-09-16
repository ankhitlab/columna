# columna

Fluent DataFrame library for TypeScript — pandas-like analytics with **WebGPU → WASM → CPU** backends. Aimed at TypeScript/JavaScript developers who want a typed, fluent tabular API in Node.js and modern browsers.

```ts
import { DataFrame, col } from 'columna'

const out = await DataFrame.fromRows([
  { city: 'Berlin', age: 30, salary: 72000 },
  { city: 'Berlin', age: 22, salary: 48000 },
  { city: 'Paris', age: 41, salary: 91000 },
])
  .select('city', 'age', 'salary')
  .filter(col('age').gt(18))
  .groupBy('city')
  .agg({ salary: 'mean', age: 'count' })
  .sort(col('salary').desc())
  .collect()
```

## Features

- Fluent, lazy-friendly DataFrame API with expression DSL (`col`, `when`, string/datetime helpers)
- Pluggable compute backends: WebGPU (when available), WASM, and CPU
- IO for CSV / JSON / Excel / Parquet; optional SQL and Kafka batch reads via peer drivers
- Separate advanced statistics module (`columna/advanced`) for tests, regression, SPC, DOE, and related workflows
- Dual ESM / CommonJS builds with TypeScript declarations
- Optional browser IDE: Columna Studio (workspace app)

## Packages

| Package | Role |
|---|---|
| `columna` | Public umbrella entry |
| `@columna/core` | Fluent API, Expr DSL, query plan |
| `@columna/arrow` | Columnar TableView / schema |
| `@columna/runtime` | Backend routing |
| `@columna/wasm` | Portable WASM kernels |
| `@columna/webgpu` | WebGPU compute path |
| `@columna/advanced` | **advanced**: Minitab-parity stats — tests, regression, SPC, DOE, time series, reliability, multivariate, predictive (`columna/advanced`) |
| `columna-studio` | Spyder-like web IDE for columna |

## Installation

```bash
npm install columna
# or: pnpm add columna
```

Optional peer drivers (install only what you need): `pg`, `mssql`, `mysql2`, `better-sqlite3`, `@clickhouse/client`, `kafkajs`.

Workspace packages under `@columna/*` are internal to this monorepo and are **not** published separately — they are bundled into `columna`.

## Quick Start

```ts
import { DataFrame, col, init } from 'columna'

await init() // optional: wait for WebGPU / WASM / native backends

const df = await DataFrame.fromRows([
  { city: 'Berlin', age: 30, salary: 72000 },
  { city: 'Paris', age: 41, salary: 91000 },
])
  .filter(col('age').gt(18))
  .groupBy('city')
  .agg({ salary: 'mean' })
  .collect()

console.log(df.toArray())
```

Advanced statistics (separate entry):

```ts
import { DataFrame } from 'columna'
import { dist, ttest1 } from 'columna/advanced'

dist.normal().ppf(0.975)
ttest1([1, 2, 3, 4], { mu: 2 })
```

## Requirements

- **Node.js** ≥ 18 (see root `package.json` `engines`)
- **TypeScript** 5.x recommended for consumers (declarations ship with the packages)
- **Browsers:** modern engines with ES2022; WebGPU path needs a supporting browser (e.g. Chrome/Edge)
- **Package manager for development:** pnpm 9 (`packageManager` field)
- Optional peers only when using SQL/Kafka helpers (listed under Installation)

## Benchmarks

Comparative **pandas vs polars vs columna** (filter / groupby / sort / join / pipeline):

```bash
py -3 -m pip install pandas polars
pnpm bench:compare
# default sizes: 1M + 100M (override with BENCH_SIZES)
```

Results land in `packages/bench/results/compare-latest.json`.

Statistics wave (rank / broadcast aggregates / Expr math / corr / cov) vs **pandas**, plus a value cross-check on the same CSV:

```bash
pnpm bench:stats
```

Results land in `packages/bench/results/stats-*.json`.

Every `columna/advanced` function (172 cases) vs **scipy / numpy / pandas / polars** on same-size synthetic data — report in [docs/advanced-benchmarks.md](docs/advanced-benchmarks.md):

```bash
pnpm bench:advanced                    # TS → Python → docs/advanced-benchmarks.md (≈ 5 min)
BENCH_SCALE=0.1 pnpm bench:advanced    # quick pass
```

Browser **WebGPU** micro-bench (Chrome/Edge; filter / map / pipeline vs CPU):

```bash
pnpm bench:webgpu
```

## Columna Studio

Spyder-style browser IDE: Monaco editor, REPL, Variable Explorer, DataFrame viewer, plots.

```bash
pnpm install
pnpm studio
```

Opens `http://localhost:5173`. Shortcuts: `Ctrl+Enter` run buffer/selection, `Shift+Enter` run line. Drop a `.csv` to load it as a DataFrame.

## Development

```bash
pnpm install
pnpm lint
pnpm typecheck
pnpm test
pnpm build
```

## IO

Read CSV / JSON / Excel / Parquet from a **path**, **URL**, string content, or bytes:

```ts
import { DataFrame } from 'columna'

const csv = await DataFrame.readCsv('./data/people.csv', {
  separator: ',',
  comment: '#',
  nullValues: ['NA', ''],
  usecols: ['city', 'age'],
  nRows: 1000,
})

const json = await DataFrame.readJson('https://example.com/data.jsonl', { lines: true })
const xlsx = await DataFrame.readExcel('./report.xlsx', { sheet: 'Sheet1', header: 0 })
const pq = await DataFrame.readParquet('./events.parquet', { columns: ['ts', 'value'] })
```

Sync helpers for in-memory strings remain: `fromCSV`, `fromJSON`.

## Databases

`DataFrame.readSql` / `readDatabase` (pandas `read_sql` / polars `read_database`):

| Dialect | URL example | Optional driver |
|---|---|---|
| PostgreSQL | `postgres://user:pass@host:5432/db` | `pg` |
| MS SQL Server | `mssql://user:pass@host:1433/db` | `mssql` |
| ClickHouse | `clickhouse://user:pass@host:8123/default` | `@clickhouse/client` |
| MySQL / MariaDB | `mysql://user:pass@host:3306/db` | `mysql2` |
| SQLite | `./file.db`, `:memory:`, `sqlite://…` | `better-sqlite3` |

```ts
import { DataFrame } from 'columna'

// URL
const df = await DataFrame.readSql('SELECT * FROM events WHERE day = $1', 'postgres://…/analytics', {
  params: ['2026-01-01'],
})

// Explicit config
const ch = await DataFrame.readSql(
  'SELECT city, count() AS n FROM hits GROUP BY city',
  { dialect: 'clickhouse', host: 'localhost', port: 8123, database: 'default' },
)

// Existing client / pool (duck-typed)
const df2 = await DataFrame.readSql('SELECT 1 AS x', {
  query: async (sql) => [{ x: 1 }],
})
```

Install only the drivers you need, e.g. `pnpm add pg` or `pnpm add @clickhouse/client`.

## Kafka

`DataFrame.readKafka` consumes a **bounded batch** (not an infinite stream) into a DataFrame. Nested JSON payloads are flattened to dotted columns by default (`user.address.city`). Arrays stay as JSON strings unless `flatten: { arrays: true }`.

Optional driver: `kafkajs` (`pnpm add kafkajs`).

```ts
import { DataFrame } from 'columna'

// Config
const df = await DataFrame.readKafka({
  brokers: ['localhost:9092'],
  topic: 'events',
  fromBeginning: true,
  nMessages: 1000,
  maxWaitMs: 10_000,
  // flatten: true by default
})

// URL
const df2 = await DataFrame.readKafka('kafka://localhost:9092/events?fromBeginning=true&nMessages=500')

// Nested JSON message { "user": { "profile": { "city": "NY" } }, "n": 1 }
// → columns: user.profile.city, n, _kafka_topic, _kafka_partition, …
```

Stop conditions: `nMessages` (default 1000) or `maxWaitMs` (default 10s), whichever comes first.

## Expressions & transforms

```ts
import { DataFrame, col, when } from 'columna'

const df = DataFrame.fromRows([{ name: 'Ada', x: 3, ts: Date.UTC(2024, 0, 15) }])

await df
  .withColumns(
    col('name').str.toLowerCase().alias('lo'),
    col('name').str.concat(col('band'), ' / ').alias('label'),   // string concat; also str.padStart / padEnd
    col('ts').dt.year().alias('year'),
    when(col('x').gt(2)).then('hi').otherwise('lo').alias('band'),
  )
  .filter(col('x').isIn([3]).and(col('x').isBetween(1, 5)))
  .collect()

await df.groupBy('band').agg({
  mean: col('x').mean(),
  std: col('x').std(),
  med: col('x').median(),
  q1: col('x').quantile(0.25),            // pandas / polars definition: position (n − 1)q (type 7)
  q1m: col('x').quantile(0.25, 'minitab'), // Minitab / SPSS: position q(n + 1), clamped (type 6) — Q1 of 1..4 is 1.25, not 1.75
})
await df.describe({ quantileMethod: 'minitab' }).collect() // quartiles the way Minitab's Display Descriptive Statistics reports them

// Arithmetic / math chains compile to ONE fused loop (no per-node temporaries): (x - mean) / std is a single pass.
// Element-wise math (typed fast path; null → null): sqrt, log(base?), log10, log2, exp, pow, round(decimals), floor, ceil, sign
await df
  .withColumns(
    col('x').log().alias('lnx'),
    col('x').pow(2).sqrt().alias('absx'),
    col('x').round(2).alias('x2'), // half away from zero, like Minitab / Excel
  )
  .collect()

// Correlation / covariance matrices (pandas df.corr() / df.cov(); pairwise-complete rows):
await df.corr().collect()                                  // Pearson, all numeric columns
await df.corr({ method: 'spearman', columns: ['x', 'y'] }).collect()
await df.cov().collect()                                   // sample covariance (n−1)

// Aggregates in a scalar context are broadcast over the whole frame (polars-style):
await df
  .withColumn('z', col('x').sub(col('x').mean()).div(col('x').std()))
  .withColumn('share', col('x').div(col('x').sum()))
  .filter(col('x').gt(col('x').median()))
  .collect()

await df.tail(5).sample({ n: 10, seed: 1 }).collect()
await left.semiJoin(right, 'id').crossJoin(dims).collect()

df.toCsv()
await df.writeParquet('./out.parquet')
df.toMarkdown()
df.profile()
```

## Advanced statistics (`columna/advanced`)

Everything statistical beyond the DataFrame engine lives in **`@columna/advanced`** and is imported separately, so the standard bundle stays a DataFrame library. Importing it also installs the column-level methods on `DataFrame` / `LazyFrame` (`df.ttest(…)`, `await lazy.anova(…)`). Roadmap and test protocol: [docs/advanced-roadmap.md](docs/advanced-roadmap.md). Minitab menu coverage matrix: [docs/minitab-coverage.md](docs/minitab-coverage.md).

```ts
import { DataFrame, col } from 'columna'
import { dist, ttest1, anova, tukeyHSD, power } from 'columna/advanced'   // functions + DataFrame methods
```

### Distributions

`dist` — normal, Student t, chi-square and F with `pdf`, `cdf`, `sf` (upper tail, no cancellation), `ppf`, `isf` and a vectorized `map`. Values agree with scipy to ~1e-13.

```ts
dist.normal().ppf(0.975)              // 1.959964
dist.t(10).sf(2.228) * 2              // two-sided p-value ≈ 0.05
dist.chi2(5).sf(11.07)                // 0.05001
dist.f(3, 10).cdf(3.708)              // 0.94999

// one-sample t-test in two steps: statistics via broadcast aggregates, p-value via dist
const s = (await df.select(col('x').mean().alias('m'), col('x').std().alias('sd'), col('x').count().alias('n')).collect()).toArray()[0]!
const tStat = (s.m - 50) / (s.sd / Math.sqrt(s.n))
const p = 2 * dist.t(s.n - 1).sf(Math.abs(tStat))

// theoretical quantiles for a normal probability plot
dist.normal().map('ppf', [0.1, 0.3, 0.5, 0.7, 0.9])
```

Discrete distributions `dist.binomial(n, p)` and `dist.poisson(λ)` have `pmf`, `cdf`, `sf`, `ppf`, `mean`, `variance`; `hypergeomPmf(k, N, K, n)` and the noncentral cdfs `nctCdf(x, df, δ)`, `ncfCdf(x, d1, d2, λ)`, `ncChi2Cdf(x, k, λ)` back the exact tests and power calculations.
Special functions are exported too: `lgamma`, `gammainc` / `gammaincc`, `betainc`, `erf` / `erfc`.

### Hypothesis tests

Ready-made tests on columns (Minitab Basic Statistics / ANOVA / Tables; numbers match scipy.stats):

```ts
df.ttest('x', { mu: 50 })                        // 1-Sample t → { statistic, df, pValue, ci, estimate, se, samples }
df.ttest('x', { by: 'line' })                     // 2-Sample t (Welch; equalVar: true pools) between the two levels of line
df.ttest('after', { paired: 'before' })           // Paired t on after − before
df.anova('yield', 'fertilizer')                   // One-Way ANOVA → { statistic, pValue, ssBetween, ssWithin, etaSquared, groups }
df.tukey('yield', 'fertilizer', { alpha: 0.05 })  // Tukey HSD pairwise comparisons → { comparisons[{ a, b, diff, se, q, pValue, ci, significant }], groups[{ …, letters }], qCritical }
df.fisher('yield', 'fertilizer')                  // Fisher LSD (individual error rate; familyAlpha reported)
df.dunnett('yield', 'fertilizer', { control: 'none', alternative: 'greater' })  // Dunnett vs control: adjusted p, simultaneous CIs
df.hsu('yield', 'fertilizer', { best: 'largest' })  // Hsu MCB: constrained CIs for each mean vs the best of the others, canBeBest / isBest flags
df.equalVariances('yield', 'fertilizer')          // Test for Equal Variances: Levene / Brown–Forsythe (default), 'bartlett', or 'bonett' (kurtosis-robust)
df.chi2test('region', 'outcome')                  // Chi-Square Test for Association on the crosstab → { statistic, df, pValue, expected, rows, cols }
df.mannWhitney('x', 'group')                      // Nonparametrics › Mann-Whitney: U, W, Hodges–Lehmann estimate + CI for η₁ − η₂, exact / asymptotic p
df.kruskal('x', 'group')                          // Kruskal-Wallis: tie-adjusted H, p, per-group medians / average ranks / z
df.normalityTest('x')                             // Anderson–Darling (Minitab default) → { statistic: A², adjusted, pValue, n, mean, sd }
df.normalityTest('x', 'ryan-joiner')              // Ryan–Joiner R with Minitab's critical values → pValue (+ pBound '> 0.100' / '< 0.010')
df.normalityTest('x', 'kolmogorov-smirnov')       // Kolmogorov–Smirnov (Lilliefors, parameters estimated; Dallal–Wilkinson p)
df.normalityTest('x', 'shapiro-wilk')             // Shapiro–Wilk W (Royston; matches R shapiro.test / scipy.shapiro)
df.ztest('x', { sigma: 2, mu: 50 })               // 1-Sample Z with known σ
df.varTest('x', { sigma0: 2 })                    // 1 Variance: χ² (default) or { method: 'bonett' } for non-normal data
df.corrTest('x', 'y')                             // Correlation with p-value and Fisher-z CI; { method: 'spearman' }
df.outlierTest('x')                               // Outlier Test: Grubbs (default) or { method: 'dixon' }, 3 ≤ n ≤ 30
df.signTest('x', { median: 50 })                  // 1-Sample Sign: exact p, Minitab's interpolated CI + achievable intervals
df.wilcoxon('x', { median: 50 })                  // 1-Sample Wilcoxon: exact / asymptotic p, Walsh-average estimate + CI
df.mood('x', 'group')                             // Mood's Median Test
df.friedman('y', 'treatment', 'block')            // Friedman (long format, one observation per block × treatment)
df.runsTest('x')                                  // Runs Test about the mean ({ k } for another cut point)
df.equivalence('y', { limits: [-0.5, 0.5], by: 'g', reference: 'ref' })  // TOST: one-sample, two-sample (by) or paired
await df.filter(col('ok')).lazy().ttest('x')      // LazyFrame versions are async
```

### Regression and models

```ts
df.regress('y', ['x1', 'x2'])                     // OLS → coefficients (SE, t, p, CI, VIF), s / r2 / r2adj / r2pred, anova, terms (seq / adj SS), diagnostics, unusual, predict()
df.regress('y', ['x1', 'x2']).predict([10, 5])    // → [{ fit, se, ci, pi }]
df.fittedLine('x', 'y', { degree: 2 })            // Fitted Line Plot: polynomial fit with .curve(x) and .equation
df.stepwise('y', ['x1', 'x2', 'x3'], { alphaIn: 0.15, alphaOut: 0.15 })  // or method: 'forward' | 'backward'; steps, selected, model
df.bestSubsets('y', ['x1', 'x2', 'x3'])           // R², R²adj, R²pred, Mallows Cp, S for the best subsets of each size
df.logistic('event', ['dose', 'age'])             // Binary Logistic: odds ratios with CI, deviance / Pearson / Hosmer–Lemeshow, G test, predict()
df.logistic('events', ['dose'], { trials: 'n' })  // events / trials form; link: 'probit' | 'cloglog'
df.glm('count', ['x'], { family: 'poisson', offset: 'logExposure' })  // Poisson Regression (rate ratios); families: binomial, poisson, gaussian, gamma
df.ologit('rating', ['x1', 'x2'])                 // Ordinal Logistic (proportional odds, Minitab's sign convention)
df.mlogit('choice', ['x'], { reference: 'A' })    // Nominal Logistic (one equation per non-reference level)
df.linearModel('y ~ a*b + x')                     // General Linear Model: factors (effects coding), covariates, interactions, x^2; Type III + sequential SS, means
df.nls((x, [a, b]) => a * (1 - Math.exp(-b * x)), 'x', 'y', { start: [500, 1e-4] })  // Nonlinear Regression (Levenberg–Marquardt)
df.orthogonalRegression('x', 'y', { errorVarianceRatio: 1 })  // Orthogonal (Deming) regression
df.pls('y', ['x1', 'x2', 'x3', 'x4'], { components: 3 })      // Partial Least Squares with leave-one-out predicted R²
```

Array forms: `ols(y, { x1, x2 })`, `fittedLine(x, y)`, `stepwise`, `bestSubsets`, `glm` / `logit` / `poissonRegression`, `ologit`, `mlogit`, `linearModel({ y, a, b, x }, 'y ~ a*b + x')`, `nls`, `orthogonalRegression`, `pls`. The small dense linear algebra behind them (`qr`, `lstsq`, `svd`, `cholesky`, `inverse`) is exported too.

### Quality tools and control charts (SPC)

```ts
spcConstants(5)                                   // A2, A3, B3, B4, D3, D4, d2, c4 (ASTM / Minitab)
df.controlChart('x')                              // I-MR by default; { type: 'xbar-r', subgroup: 5 } / 'p'|'np'|'c'|'u'|'laney-p'…
df.controlChart('defects', { type: 'p', sizes: n }) // attribute chart with variable n; Nelson rules 1–8
df.ewma('x', { lambda: 0.2, L: 3 })               // EWMA; cusum({ h, k }), movingAverage({ span })
df.capability('x', { lsl: 10, usl: 20, subgroup: 5, target: 15 })  // Cp, Cpk, Pp, Ppk, Cpm, PPM, Z.Bench
boxCoxLambda(x); johnsonFit(x); weibullFit(x)     // nonnormal capability transforms / fits
df.toleranceInterval('x', { coverage: 0.95, confidence: 0.95 })
df.gageRR({ part: 'part', operator: 'op', measurement: 'y' })  // crossed ANOVA Gage R&R; gageLinearity, gageType1
attributeAgreement(rater1, { method: 'cohen', other: rater2 }) // Fleiss / Kendall W / τ too
acceptanceSampling({ type: 'attributes', n: 50, c: 2 })        // OC, AOQ, ATI; variables k-method
df.pareto('defect'); df.runChart('x'); df.individualDistributionID('x')
```

### Time series, DOE, reliability, multivariate, predictive

```ts
df.trendAnalysis('y', { model: 'quadratic', horizon: 5 })
df.decompose('y', { seasonLength: 12 })
df.stl('y', { seasonLength: 12, robust: true })
df.ets('y', { method: 'winters-add', seasonLength: 12, horizon: 6 })
df.acf('y', { maxLag: 20 }); ljungBox(y, { lags: 10 })
df.arima('y', { p: 1, d: 1, q: 1, horizon: 5 })
df.autoArima('y', { seasonalPeriod: 12 })
fullFactorial(['A','B','C']); analyzeEffects(design, y)
ccd(['A','B']); boxBehnken(['A','B','C']); taguchi('L32'); definitiveScreening(['A','B','C','D','E','F'])
mixtureDesign(['A','B','C']); responseOptimizer(goals, { simplex: true, nComponents: 3 })
analyzeTaguchi(taguchi('L9', ['A','B','C','D']), responses, { snRatio: 'larger' })
df.reliabilityFit('time', { distribution: 'weibull', censor: 'c' })
df.kaplanMeier('time', { censor: 'c' }); df.logRank('time', 'group', { weight: 'wilcoxon' })
df.coxPH('time', ['x1'], { censor: 'c', strata: 'site', frailtyGroup: 'center' })
df.mixedModel('y', { fixed: ['x'], group: 'batch', slope: 'x' })
df.glmm('y', { family: 'binomial', fixed: ['x'], group: 'batch' })
warrantyPrediction(fit, { warranty: 1000, nUnits: 500 })
df.pca(['x1','x2','x3']); factorAnalysis(data, { method: 'ml' }); df.kmeans(['x1','x2'], { k: 3 })
df.cart('y', ['x1','x2']); df.randomForest('y', ['x1','x2'], { nTrees: 100 })
treeNet(X, y, { task: 'classification' }); mars(X, y, { maxTerms: 12 })
```

### Everything else in the Minitab menus

```ts
df.descriptiveStats(['x', 'y'], { by: 'g' })     // Display / Store Descriptive Statistics: full Minitab table per variable × level
df.graphicalSummary('x')                          // Graphical Summary: descriptives, A-D, CIs (mean / median / σ), histogram + boxplot data
df.poissonGof('defects')                          // Goodness-of-Fit Test for Poisson
df.boxplotStats('x', 'g'); df.intervalPlot('x', 'g'); df.mainEffectsPlot('y', ['a', 'b']); df.interactionPlot('y', 'a', 'b'); ecdf(x); dotplot(x)
causeAndEffect({ effect: 'Defects', categories: { Man: ['Training'], Machine: ['Wear'] } }).svg
df.stabilityStudy('assay', 'month', 'batch', { lsl: 95 })   // Stability Study: pooling at α = 0.25, shelf life per batch / overall
df.gChart('daysBetween'); df.tChart('hoursBetween')          // Rare event charts
df.t2Chart(['x1', 'x2'], { subgroup: 'lot' }); df.mewma(['x1', 'x2'], { lambda: 0.1 }); generalizedVarianceChart(rows, subgroup)
df.periodogram('y', { spans: [3, 3] }); cumulativePeriodogram(y)   // Spectral analysis + Bartlett white-noise test
aliasStructure(6, ['E=ABC', 'F=BCD'])             // defining relation, resolution, alias chains
df.lifeRegression('time', ['temp'], { censor: 'c', distribution: 'weibull' })   // Regression with Life Data (AFT)
df.altRegression('time', 'tempC', { relation: 'arrhenius', useStress: 50 })    // Accelerated Life Testing
demonstrationTestPlan({ reliability: 0.9, time: 1000, shape: 1.5, testTime: 1500 }); estimationTestPlan({ shape: 2, scale: 1000, ratio: 2, censorTime: 800 })
df.powerLawNHPP('failureTime', { endTime: 500, system: 'unit' })   // Repairable systems (Crow–AMSAA)
df.probitAnalysis('dead', 'n', 'dose', { logStress: true })        // Probit Analysis with Fieller limits
df.clusterVariables(['x1', 'x2', 'x3'], { nClusters: 2 }); df.multipleCorrespondence(['a', 'b', 'c']); df.itemAnalysis(['q1', 'q2', 'q3'])
promax(factorAnalysis(data, { nFactors: 2 }).rotatedLoadings!)
df.manovaModel(['y1', 'y2'], 'a*b + x')          // General MANOVA (Type III SSCP per term)
df.crossValidate('y', ['x1', 'x2'], { model: 'random-forest', folds: 5 }); df.autoModel('y', ['x1', 'x2'])
random(42).normal(100, 10, 2); random(42).weibull(50, 1.8, 100); patterned(1, 5, { repeat: 2 })
dist.weibull(1.8, 50).ppf(0.1); dist.gamma(2, 3).cdf(4); dist.beta(2, 5).mean; dist.lognormal(0, 0.5).sf(2)
```

Count-based tests take the counts directly, and `power` solves any one of effect / n / power for the nine Minitab power tools:

```ts
propTest1(7, 20, { p0: 0.5 })                     // 1 Proportion: exact binomial p + Clopper–Pearson CI ({ method: 'normal' } for Wald)
propTest2(12, 40, 6, 35, { method: 'fisher' })    // 2 Proportions: pooled z (default) or Fisher's exact
poissonRateTest1(12, 10, { lambda0: 1 })          // 1-Sample Poisson Rate: exact p, Garwood CI; poissonRateTest2 for two rates
power({ test: '2-sample t', effect: 0.5, power: 0.8 })          // → n = 64 per group
power({ test: '1-sample t', effect: 1, n: 10 })                 // → power 0.803
power({ test: 'one-way anova', groups: 4, n: 10, power: 0.8 })  // → maximum difference between means (σ = 1)
// tests: '1-sample z' | '1-sample t' | '2-sample t' | 'paired t' | '1 proportion' | '2 proportions' | 'one-way anova' | '1 variance' | '2 variances'
```

Options: `alternative: 'two-sided' | 'less' | 'greater'`, `confidence` (default 0.95), `correction: true` for Yates on 2×2.
The same tests work on plain arrays: `ttest1`, `ttest2`, `ttestPaired`, `anova({ a: [...], b: [...] })`, `chi2test([[30, 10], [20, 25]])`, `chi2gof(observed, expected?)`, `crosstab(a, b)`, `andersonDarling(x)`, `ryanJoiner(x)`, `kolmogorovSmirnov(x)`, `shapiroWilk(x)`, `tukeyHSD(groups)`, `fisherLSD(groups)`, `dunnett(groups, { control })`, `hsuMCB(groups, { best })`, `levene(groups)`, `bartlett(groups)`, `bonett(groups)`, `varTest2(a, b)`, `bonett2(a, b)` (2 Variances, Bonett's method with CI for the ratio), `mannWhitney(a, b, { alternative, method })`, `kruskal(groups)`, `ztest1(x, { sigma })`, `varTest1(x, { sigma0 })`, `corrTest(a, b)`, `grubbs(x)`, `dixon(x)`, `signTest(x)`, `wilcoxonSigned(x)`, `moodMedian(groups)`, `friedman(table)`, `runsTest(x)`, `tost1(x, { limits })`, `tost2(a, b, { limits })`, `tostPaired(a, b, { limits })`.
The studentized range distribution is exported as `ptukey(q, k, df)` / `qtukey(p, k, df)` (R names; matches scipy `studentized_range` to ~1e-7), Dunnett's as `pdunnett(c, lambdas, df)` / `qdunnett(p, lambdas, df)`.

## Engines

Prefer `.engine('webgpu' | 'wasm' | 'cpu')` or let runtime auto-pick. Use `.explain()` to inspect the plan.

WebGPU runs a **hybrid** plan: numeric AND-filters and col∋lit maps on the GPU (with buffer residency), other ops on CPU. Call `await init()` so the device is ready before `collect()`.

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and the issue templates under `.github/ISSUE_TEMPLATE/`. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md). Security reports: [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
