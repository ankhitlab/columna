# columna

Typed DataFrames plus a **Minitab-class statistics library** for TypeScript — in Node.js and the browser, with no native binary and no runtime dependencies in the core.

The DataFrame part competes with Arquero (same job; on 2M rows columna is 5–10× faster per operation and uses ~3× less memory), overlaps with DuckDB-Wasm (which is a real SQL engine and reads files faster) and is not a substitute for Polars or DuckDB native when a server can run one. What none of them have is the statistics layer: ~250 procedures — hypothesis tests, ANOVA, regression with full diagnostics, DOE, SPC, capability, reliability, time series, multivariate — each checked against scipy / numpy / NIST references. Measured comparison and an honest "when to use what": [docs/positioning.md](docs/positioning.md).

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
- Compute backends: CPU (typed kernels, optional native addon and worker parallelism in Node), with narrow WebGPU / WASM accelerations that report what they actually ran (see Engines)
- IO for CSV / JSON / Excel / Parquet; optional SQL and Kafka batch reads via peer drivers
- `columna/advanced`: Minitab-level statistics (~250 procedures) with fixture, property and Monte-Carlo tests
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

### Saying what a source is, and fencing it in

A bare string is classified by a heuristic (`http(s)://` → fetch, path-like → file, otherwise content). That is
fine for a script and wrong for a server: a string taken from a request could name any URL or file the process
can reach. Say what it is, and set boundaries — per call or once for the process:

```ts
import { DataFrame, io, setIoPolicy } from 'columna'

await DataFrame.readCsv({ text: body })                 // content, never a path — also io.text(body) / { mode: 'text' }
await DataFrame.readCsv({ path: file })                 // filesystem — io.path(file)
await DataFrame.readJson({ url: link }, {               // network — io.url(link)
  allowedHosts: ['data.example.com', '*.cdn.example.com'],
  denyPrivateHosts: true,                               // no loopback / RFC 1918 / link-local / cloud metadata
  maxBytes: 50e6,                                       // body is read incrementally and cut off past the cap
  timeoutMs: 10_000,
  signal: controller.signal,
})

// process-wide floor: per-call options can only narrow it, never widen it
setIoPolicy({ allowedDirs: ['/srv/data'], allowedHosts: ['data.example.com'], maxBytes: 100e6, timeoutMs: 30_000 })
```

`allowedDirs` compares real paths (symlinks cannot escape); `file://` URLs are filesystem reads under the same rule;
every redirect hop is checked against `allowedHosts` / `denyPrivateHosts`; URLs with embedded credentials and
non-http(s) protocols are refused. `denyPrivateHosts` is a name check, not DNS — for DNS-rebinding protection pass
your own `fetch` (it receives `{ redirect: 'manual', signal }`).

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
import { DataFrame, openSqlClient } from 'columna'

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

// Reuse one connection across reads: open it yourself, pass it in, close it when done.
// readSql never closes a client it was given. Each openSqlClient() is a dedicated connection / pool —
// two MS SQL databases can be open side by side (the driver's global pool is never used).
const client = await openSqlClient('mssql://user:pass@host/db_a')
const a = await DataFrame.readSql('SELECT * FROM orders', client)
const b = await DataFrame.readSql('SELECT * FROM customers', client)
await client.close?.()
```

A URL or config passed straight to `readSql` opens a connection for that call and closes it afterwards.
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

## Typed schemas

A frame carries its row type. `fromRows` / `fromColumns` infer it; every column-changing operation computes the
next one, so the compiler knows which columns exist, what `toArray()` returns, and rejects a misspelt name:

```ts
const df = DataFrame.fromRows([{ city: 'Berlin', age: 30, salary: 72000 }])
//    ^ DataFrame<{ city: string; age: number; salary: number }>

const out = await df
  .filter((c) => c.age.gt(18))                       // c: typed column refs — c.age is Expr<number, 'age'>
  .withColumn('k', (c) => c.salary.div(1000))        // schema gains k: number
  .rename({ city: 'town' })                          // schema: town, age, salary, k
  .select('town', 'k')                               // only known names compile
  .collect()
out.toArray()                                        // Array<{ town: string; k: number }>

df.select('cty')                 // error: '"cty"' is not assignable to '"city" | "age" | "salary"'
df.filter((c) => c.salary)       // error: Expr<number> is not a predicate (Expr<boolean> required)
df.groupBy('city').agg((c) => ({ n: c.age.count(), pay: c.salary.mean() }))
//                                                   // LazyFrame<{ city: string; n: number; pay: number }>
df.join(regions, { on: 'city' })                     // LazyFrame<L & R>; leftJoin makes R's columns | null
```

What is and is not checked:

- `col('x')` is untyped on purpose (`Expr<any>`): a string names a column the compiler knows nothing about. It
  still composes everywhere; only typed results are checked (`col('x').add(1)` is rejected as a predicate).
  `cols<S>()` gives typed refs outside callbacks.
- Readers (`readCsv` / `readJson` / …) return `DataFrame<Row>`. `readCsv<S>(…)` is an **assertion** by the caller —
  the file is not validated against `S`.
- Joins do not model suffixes for colliding non-key columns; `melt`, `pivot`, `transpose`, `describe`, `valueCounts`,
  `corr` return `LazyFrame<Row>` (their columns depend on data).
- Untyped code keeps compiling: every generic defaults to `Row` = `Record<string, unknown>`.

The type-level guarantees are themselves tested (`packages/core/tests/schema-types.test.ts` runs under `tsc` with
`@ts-expect-error` lines, as part of `pnpm typecheck`).

### Type system: what exists and what does not

`DType` is `f64 | f32 | i32 | u32 | bool | utf8 | category | datetime`. Not available: **int64 / uint64** (integers past
2⁵³ are kept as text by the CSV / Parquet / SQL readers rather than rounded; there is no BigInt column), **decimal**
(no fixed-point arithmetic — money in f64 rounds), **list / struct** (`explode` / `unnest` flatten JSON-shaped input on
the way in; nested values are not a column type), **time zones** (`datetime` is epoch milliseconds, UTC arithmetic only).
Pick a different tool for those, or model them as text.

### Buffer ownership

`fromColumns` **shares** typed arrays and pre-encoded category codes zero-copy: the frame reads your buffer, so writing
to it later changes the frame, and a `readonly TableView` does not freeze its contents. Pass `{ copy: true }` to detach.
Everything the library produces (operations, readers) is owned by the library and never aliases user memory. The GPU
buffer cache is keyed by array identity: a buffer mutated after its first upload is *not* re-uploaded — treat shared
buffers as immutable once handed over, or copy.

### Importing has no side effects

`import { DataFrame } from 'columna'` registers inert backend objects and nothing else — no WebGPU adapter request,
no worker, no native module load. Until `await init()` every plan runs on the CPU engine deterministically, and
`engine('webgpu')` reports "does not support this plan" in strict mode. `init()` is the single place infrastructure starts.

## Expressions & transforms

```ts
import { DataFrame, col, when } from 'columna'

const df = DataFrame.fromRows([{ name: 'Ada', x: 3, ts: Date.UTC(2024, 0, 15) }])
// fromRows (and every reader built on it: CSV, JSON, Excel, Parquet, SQL, Kafka) infers each dtype over the
// whole column - a late 1.5, 2^31 or 'n/a' widens the column to f64 / utf8 instead of being coerced - and the
// schema is the union of all rows' keys (missing -> null). cast('i32' | 'u32') throws on overflow / fractions.

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
df.toCsv({ escapeFormulas: true })  // neutralise =, +, -, @ cells for Excel-bound exports of untrusted text
await df.writeParquetLike('./out.columna.json')  // custom JSON format; Apache Parquet writer not yet available
df.toMarkdown()
df.profile()
```

> **Note:** `writeParquet()` previously wrote a JSON payload while `readParquet()` reads real Apache Parquet via hyparquet — that mismatch is now an explicit error. Use `writeParquetLike()` for the JSON format.

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

## Engines — what a backend name guarantees

`.engine('webgpu' | 'wasm' | 'cpu')` is a **request**, not a guarantee. Every backend is a hybrid over the same CPU
planner, and by default an engine that cannot run a node hands it to the CPU without a word. The honest picture:

| Engine | Executes on its own | Everything else |
|---|---|---|
| `cpu` | all nodes: typed JS kernels; optional native Rust addon (filter ≥ 1M rows, gather ≥ 250k, join / groupBy ≥ 500k, strings ≥ 10M) and worker-thread parallelism (dual filter ≥ 50M rows, gather ≥ 2M) — Node only, browsers stay single-threaded | — |
| `wasm` | one kernel family: `col OP n AND col OP n` filters on i32×f64 / i32×i32 / f64×f64 columns, **≥ 5M rows**, and only when the Rust `pkg` is built and loaded | CPU |
| `webgpu` | AND-filters over i32 / u32 / f32 columns and `col ARITH n` maps on f32 columns (exact; f64 / datetime need `gpuLossyF32`); rows ≥ 10 000 | CPU — including mask → indices and the row gather *after* every GPU filter |

Two tools make the difference visible:

```ts
// 1. the execution report: what ran, where, why not, how long (GPU: transfer vs compute vs CPU gather)
const { frame, report } = await df.lazy().filter(...).engine('wasm').collectWithReport()
console.log(formatExecutionReport(report))
// requested: wasm · dispatched: wasm · used: cpu · 41.2 ms
//   filter: cpu 40.9 ms rows=812345 — 2000000 rows < WASM_RUST_FILTER_MIN_ROWS (5000000); JS kernels are faster below that

// 2. strict mode: the engine must do the work or the call fails with EngineStrictError listing the reasons
await df.lazy().filter(...).engine('webgpu', { strict: true }).collect()
```

`.explain()` prints the plan and the backend it is *dispatched* to; only `collectWithReport()` knows what executed.
A benchmark that does not read the report may be timing JavaScript under a WASM label.

### Memory and scale

Rows are cheap to count and expensive to hold. Rule of thumb for peak memory: **the columns** (8 bytes per f64 / datetime
cell, 4 per i32 / u32 / f32 / category code, 1 per bool, plus a JS string per utf8 cell) **plus the largest intermediate**
of the operation (a sort or join materialises index arrays; a groupBy its accumulators). 100M rows × 8 f64 columns is
6.4 GB of buffers before any operation — state a schema, an operation and a peak RSS with any row count.

- **CSV in**: `readCsv({ path })` streams the file in 1 MB chunks straight into typed column builders — no row objects,
  no full-text copy; text columns are dictionary-encoded on the fly. 2M rows × 8 columns (107 MB CSV): 2.9 s, peak RSS
  477 MB vs 4.2 s / 990 MB for the previous row-object path (Node 24, single thread; `pnpm bench:e2e`).
  `nRows` stops the read early; `maxBytes` is enforced on bytes read.
- **CSV out**: `writeCsv(path)` streams 16 384-row chunks with back-pressure; only one chunk of cell strings exists at a
  time. `toCsv()` necessarily builds the whole string.
- **JSON / Excel / Parquet in**: still whole-file → row objects → columns (Parquet no longer copies the input buffer). Budget
  roughly 3–5× the file size in peak RSS for these.
- **SQL**: `nRows` slices the driver's result **after** it arrived — it bounds the DataFrame, not the query, the transfer
  or the driver's buffer. Put `LIMIT` in the SQL for that.
- **Browser**: WebGPU / WASM do not make the main thread asynchronous; a 10M-row groupBy blocks the UI for as long as it
  takes. Measure bundle size, device init (`await init()`) and main-thread blocking, not only kernel time.

`pnpm bench:e2e` measures the way a deployment decision needs: cold (fresh process) and warm runs, read → process → write,
peak RSS sampled during the run, and the backend that actually executed each node.

WebGPU runs a **hybrid** plan: numeric AND-filters and col∋lit maps on the GPU (with buffer residency), other ops on CPU. Call `await init()` so the device is ready before `collect()`.

GPU results are **bit-identical to the CPU**: the filter kernel compares `i32` / `u32` / `f32` columns in their own type (no float32 rounding of integers past 2^24) and honours the null bitmap; `f64` / `datetime` columns, bool / category columns and literals the column type cannot hold exactly (`x > 2.5` on `i32`, `x > 0.1` on `f32`) stay on the CPU. `init({ gpuLossyF32: true })` opts into the approximate float32 path for those cases.

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and the issue templates under `.github/ISSUE_TEMPLATE/`. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md). Security reports: [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
