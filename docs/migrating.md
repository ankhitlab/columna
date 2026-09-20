# Migrating to columna

Side-by-side vocabulary for the three APIs people usually come from. Every columna snippet is the fluent API on
`DataFrame` / `LazyFrame`; `lazy()` … `collect()` is where the optimizer and the engines run, the sync
`DataFrame` helpers are the same kernels without a plan. Full API: [docs/api](api/index.html).

## The one example

```ts
import { DataFrame, col } from 'columna'

const out = await DataFrame.fromRows(rows)
  .lazy()
  .filter((c) => c.age.gt(18))
  .groupBy('city')
  .agg((c) => ({ avgSalary: c.salary.mean(), n: c.salary.count() }))
  .sort(col('avgSalary').desc())
  .collect()
out.toArray()
```

## Arquero → columna

| Arquero | columna | Notes |
|---|---|---|
| `aq.from(rows)` / `aq.table({ a: [...] })` | `DataFrame.fromRows(rows)` / `DataFrame.fromColumns({ a: [...] })` | `fromColumns` takes TypedArrays zero-copy (`{ copy: true }` to own them). |
| `aq.fromCSV(text)` / `aq.loadCSV(url)` | `DataFrame.fromCSV(text)` / `await DataFrame.readCsv({ url })` | Sources are explicit: `{ text }`, `{ path }`, `{ url }`; policy via `setIoPolicy`. |
| `.filter(d => d.age > 18)` | `.filter((c) => c.age.gt(18))` or `.filter(col('age').gt(18))` | Expressions, not closures: they plan, push down and run on typed kernels. `mapElements(fn)` is the escape hatch. |
| `.derive({ y: d => d.a * 2 })` | `.withColumn('y', col('a').mul(2))` / `.withColumns((c) => [c.a.mul(2).alias('y')])` | |
| `.select('a', 'b')` / `.select(aq.not('c'))` | `.select('a', 'b')` / `.drop('c')` / `.exclude('c')` | |
| `.rename({ a: 'b' })` | `.rename({ a: 'b' })` | Type of the frame follows the rename. |
| `.groupby('g').rollup({ n: aq.op.count(), m: aq.op.mean('x') })` | `.groupBy('g').agg((c) => ({ n: c.x.count(), m: c.x.mean() }))` | Also `agg({ x: ['sum', 'mean'] })` → `x_sum`, `x_mean`; shortcuts `.groupBy('g').mean()`. |
| `.orderby('a', aq.desc('b'))` | `.sort('a', col('b').desc())` | `nullsLast` per key. |
| `.join(other, ['k1', 'k2'])` / `join_left` | `.join(other, { on: 'k' })` / `.join(other, { on, how: 'left' })` | `how`: inner, left, right, outer, semi, anti, cross; `suffix`, `validate`. |
| `.dedupe('a')` | `.unique(['a'])` | |
| `.pivot('key', 'value')` / `.fold(['a', 'b'])` | `.pivot(...)` / `.melt(...)` | |
| `.objects()` / `.array('a')` | `.toArray()` / `.getColumn('a').toArray()` | |
| `.toArrow()` | `.toArrowIpc()` | Real Arrow IPC bytes (Arquero's `toArrow` gives an `apache-arrow` Table: `tableFromIPC(df.toArrowIpc())`). |
| `aq.fromArrow(table)` | `DataFrame.fromArrowIpc(tableToIPC(table))` | |
| window: `aq.op.lag('x')`, `aq.op.cume_dist()` | `col('x').shift(1)`, `.over(...)`, `rolling` / `expanding` | |

What changes conceptually: transformations are **lazy** (`lazy()` … `collect()` is async) and the runtime picks
kernels (native / workers / GPU) and reports what ran (`collectWithReport`). Arquero's per-row closures become
expressions; anything an expression cannot say goes through `mapElements`.

## Polars (`nodejs-polars` / Python) → columna

| Polars | columna | Notes |
|---|---|---|
| `pl.DataFrame({...})` / `pl.read_csv` | `DataFrame.fromColumns({...})` / `DataFrame.readCsv` | dtypes: f64, f32, i32, u32, bool, utf8, category, datetime (ms). No i64 / decimal / list / struct. |
| `df.lazy()` … `.collect()` | same | `.explain()` prints the optimized plan; `collect({ signal, timeoutMs, memory })`. |
| `pl.col('a').gt(1)` | `col('a').gt(1)` | Same expression style; `when(...).then(...).otherwise(...)`, `.isIn`, `.isBetween`, `.cast`, `.fillNull`, `.str.*`, `.dt.*`. |
| `.group_by('g').agg(pl.col('x').sum().alias('s'))` | `.groupBy('g').agg({ s: col('x').sum() })` | |
| `.sort('a', descending=True)` | `.sort(col('a').desc())` | |
| `.join(other, on='k', how='left')` | `.join(other, { on: 'k', how: 'left' })` | `join_asof` → `joinAsof`. |
| `.with_columns(...)` | `.withColumns(...)` | |
| `.unique(subset=['a'])` | `.unique(['a'])` | |
| `.select(pl.col('^x.*$'))` | `.selectDtypes(...)` / `.select(...names)` | No regex selectors. |
| `.over('g')` | `.over('g')` | |
| `df.write_ipc(path)` / `pl.read_ipc_stream(bytes)` | `df.writeArrowIpc(path, { format: 'file' })` / `DataFrame.fromArrowIpc(bytes)` | Polars' Utf8View / Categorical / Enum / Datetime(us) are read; columna writes Utf8 + Dictionary + Timestamp(ms). |
| `df.write_parquet` / `pl.read_parquet` | `df.writeParquet` / `DataFrame.readParquet` | via hyparquet. |
| `pl.SQLContext` | — | No SQL; see DuckDB-Wasm exchange through Arrow IPC in the README. |
| streaming engine / out-of-core | Node: `collect({ memory: { maxBytes } })` spills sort / unique / join intermediates | Not a streaming query engine; see [positioning.md](positioning.md). |

What you gain: the same code in the **browser**, and `columna/advanced` (hypothesis tests, ANOVA, regression
diagnostics, DOE, SPC, capability, reliability, time series). What you give up: Polars' native multi-threaded
core, i64 / nested types and the streaming engine.

## pandas → columna

The cheat-sheet; the long form with worked examples is [pandas-to-columna.md](pandas-to-columna.md).

| pandas | columna | Notes |
|---|---|---|
| `pd.DataFrame(rows)` / `pd.read_csv` | `DataFrame.fromRows(rows)` / `DataFrame.readCsv` | No index: rows are positional; `set_index` has no equivalent — keep the key as a column. |
| `df[df.age > 18]` | `df.filter((c) => c.age.gt(18))` | |
| `df['y'] = df.a * 2` | `df.withColumn('y', col('a').mul(2))` | Frames are immutable; every step returns a new frame / plan. |
| `df.groupby('g').agg(n=('x', 'count'), m=('x', 'mean'))` | `df.groupBy('g').agg({ x: ['count', 'mean'] })` | |
| `df.sort_values(['a', 'b'], ascending=[True, False])` | `df.sort('a', col('b').desc())` | |
| `df.merge(other, on='k', how='left')` | `df.join(other, { on: 'k', how: 'left' })` | |
| `df.drop_duplicates(['a'])` | `df.unique(['a'])` | |
| `df.pivot_table` / `df.melt` | `df.pivot` / `df.melt` | |
| `df.describe()` | `df.describe()` | Quantiles: `quantileMethod: 'minitab'` for Minitab's type-6 positions. |
| `df['m'] = df.x.rolling(3).mean()` | `df.rolling('m', 'x', 3, 'mean')` | `expanding(name, column, agg)` likewise. |
| `df.fillna(0)` / `ffill()` | `df.fillNull(0)` / `df.ffill()` | |
| `df.astype({ a: 'int32' })` | `df.withColumn('a', col('a').cast('i32'))` | `cast` refuses overflow instead of wrapping. |
| `df.to_feather` / `pd.read_feather` | `df.writeArrowIpc(path, { format: 'file' })` / `DataFrame.readArrowIpc({ path })` | |
| `scipy.stats.ttest_ind(a, b)` | `ttest2(a, b)` from `columna/advanced` | Same conventions where scipy and Minitab agree; documented where they differ ([advanced-roadmap.md](advanced-roadmap.md)). |
| `statsmodels.OLS` | `ols(y, X)` / `df.regress('y', ['x1', 'x2'])` | Coefficients, SE, t, p, CI, R², adjusted R², F, residual diagnostics, VIF. |

## Things that have no equivalent — plan for them

- **No SQL.** DuckDB-Wasm next to columna, exchanging Arrow IPC, is the intended pairing.
- **No i64 / decimal / nested types.** Int64 columns arrive as f64 (exact to 2⁵³); lists and structs are refused.
- **No mutable frames, no index.** Positional rows; keys are columns.
- **Cancellation is per operator.** `collect({ signal })` lands between operators, not inside a kernel.
- **Browser has no disk spill.** Memory budgets spill only on Node.

## Upgrading from 0.2 to 0.3

`0.3.0` is a **minor** under the 0.x rules in [compatibility.md](compatibility.md): new Tier 1 surfaces plus
behaviour fixes that change results for some existing inputs. Install with `npm install columna@0.3.0`.

### New APIs worth adopting

- **Apache Arrow IPC:** prefer `toArrowIpc()` / `fromArrowIpc()` / `readArrowIpc` / `writeArrowIpc` for exchange
  with Polars, DuckDB, pyarrow and `apache-arrow`. `toArrow()` / `fromArrow()` still work but are deprecated
  (they speak columna's JSON `ArrowLike`, not Arrow).
- **Cancellation / deadlines:** `collect({ signal, timeoutMs })` — rejects with `ExecutionAbortedError`.
- **Sessions:** `createSession({ io, runtime, persist })` for per-tenant runtime, persist cache and IO policy floor;
  `new Runtime({ memory })` no longer sets the process-wide memory policy (use `setMemoryPolicy` or
  `collect({ memory })`).
- **Non-blocking spill** and SQL `nRows` pushdown under memory / read options (see [operations.md](operations.md)).

### Behaviour to re-check

- `groupBy().agg({ x: 'count' })` counts **non-null** values of `x` on every kernel (was row-count on some fast paths).
- `sort(...).head(k)` matches `sort(...).collect().slice(0, k)` for ties and NaN keys.
- Join projection of collision names (`score_right`) and `select(col('x').alias('y'))` after a join keep the
  expected schema.
- `unique` keeps multi-key null rows and does not merge `null` with `NaN`.
- CommonJS + `collect({ memory })` spill works (empty `import.meta` in the CJS bundle no longer breaks
  `createRequire`).

Full list: [CHANGELOG.md](../CHANGELOG.md) `[0.3.0]`.
