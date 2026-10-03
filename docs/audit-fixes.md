# Correctness fixes for the 36ef59d audit

Baseline: `36ef59de0bface0ca4969d5d32163510805a518f` (manifest 0.4.0).
Shipped in `columna@0.4.1`. Review the compatibility changes below when upgrading.

## Correctness changes

| Finding | Change |
| --- | --- |
| F01 | Infer expression output from the expression tree, not row zero. Unknown UDF results are evaluated once per row and cached for type inference. String/category labels, booleans and datetimes survive leading nulls. |
| F02 | Stop extrapolating the Anderson-Darling p-value polynomial beyond adjusted A-squared 13. Follow the statsmodels `normal_ad` convention and report zero for that very small tail. This is a numerical convention, not an assertion of exact probability zero. |
| F03 | Classify join projections using each child's output names. Preserve collision context and skip pruning for unknown provenance; an optimization must never invent a column or change a result. |
| F04 | An expression transformation no longer promises the old phantom column name. Runtime naming is unchanged. Put `alias()` last to obtain a statically named computed output. |
| F05 | `fromRows()` types describe normalized output cells: Date to milliseconds, bigint to number or string according to policy, missing/undefined to null. Session constructors use the same types and forward the policy. |
| F06 | Compute Shapiro-Wilk W on centered, scaled values, pairing symmetric observations and using compensated sums. The shared mean/sd reporting implementation is unchanged. |
| F07 | Match the complete CSV separator in quoted records too. Reject an empty separator instead of entering a non-advancing loop. |
| F08 | Series unique/nunique use primitive identity; null and the text "null" remain distinct. NaN uses Set/SameValueZero semantics. |
| F09 | DataFrame, LazyFrame and Series share head/tail count semantics. Series peeks decode only the requested range. |
| F10 | Ordinary joins require nonempty key lists with equal arity. Known-schema key names are checked before execution. Key arrays are copied into the plan. |

## Compatibility changes

An expression cannot losslessly store a mixture of number, string and boolean families in one physical column.
A UDF or conditional that actually returns multiple primitive families now throws a TypeError rather than
coercing data based on the first row. Cast the complete expression explicitly when conversion is intended:

```ts
import { DataFrame, col } from 'columna'

const result = await DataFrame.fromRows([{ x: 0 }, { x: 1 }])
  .withColumn('label', col('x').mapElements(value => value === 0 ? 1 : 'ok').cast('utf8'))
  .collect()

console.log(result.toArray()) // [{ x: 0, label: '1' }, { x: 1, label: 'ok' }]
```

Unknown UDF results require one temporary array. A UDF on an empty input has no observable return type and
keeps the existing fallback dtype f64. This patch does not add a return-dtype option or claim a complete
schema-inference implementation for every plan node.

Computed expression types no longer claim that the source or an earlier alias survives a later operation:

```ts
const result = await DataFrame.fromRows([{ x: 1 }])
  .select(c => [c.x.add(1).alias('x')])
  .collect()

result.toArray()[0]!.x.toFixed(2)
```

Unaliased computed expressions still run with the existing runtime-generated names. Only a column reference
or a final alias gives a statically named result. `fromRows()` input primitives are widened in output types;
unions of primitive input families are conservatively represented, not interpreted as heterogeneous storage.
For a generic bigint policy, the result is conservatively `number | string`. No numeric conversion policy
or Date/bigint runtime representation changes in F05.

Peek counts are finite numbers, truncated toward zero and clamped at zero. Negative counts return no rows,
`tail(0)` is empty everywhere, and NaN/Infinity/-Infinity throw RangeError. Previously these edge cases
were inconsistent between the three classes. Empty ordinary join-key lists now throw; use `crossJoin()`
or `{ how: 'cross' }` for a Cartesian product. Structural join checks cover the fluent API, not arbitrary
hand-authored `PlanNode` objects passed directly to low-level runtime APIs.

## Conservative optimization

Projection pruning is deliberately skipped when output names or collision provenance are uncertain.
This can make some collision-heavy joins process extra columns; correctness takes priority. This is not a
complete replacement of the optimizer with a unified schema/provenance model. Dependency splitting,
structured error codes across all modules and a complete public API redesign are outside this patch set.

## Regression checks

After installing the repository's dependencies:

```sh
pnpm build
pnpm typecheck
pnpm exec vitest run packages/core/tests/audit-fixes.test.ts packages/core/tests/audit-fixes-types.test.ts packages/advanced/tests/audit-normality.test.ts
pnpm test
```

The added tests cover the original examples, category/bool/datetime/empty inputs, UDF call counts,
join collisions and composite keys, CSV multiline/escaped records, peek boundaries, and type/runtime
contracts. Numeric fixtures were generated independently with SciPy 1.17.0 and statsmodels 0.14.6 using
NumPy `default_rng(20260930)`: 16 Shapiro-Wilk and 10 Anderson-Darling reference cases.

Approximation-domain reference:
https://www.statsmodels.org/v0.14.1/_modules/statsmodels/stats/_adnorm.html

Run the full repository CI (including packed-consumer/browser tests) before publishing. Source-level CPU
tests are not a substitute for tarball packaging, native/WASM artifact or hardware-WebGPU verification.
