/** Cross-check columna against pandas on the same CSV sample (results/stats-sample.csv + stats-pandas-reference.json). */
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { DataFrame, col } from 'columna'

const RES = new URL('../results/', import.meta.url)
const ref = JSON.parse(readFileSync(new URL('stats-pandas-reference.json', RES), 'utf8'))
const df = await DataFrame.readCsv(fileURLToPath(new URL('stats-sample.csv', RES)))
console.log(`rows ${df.shape[0]} (pandas ${ref.n})  dtypes ${JSON.stringify(df.dtypes)}`)

let worst = 0
const cmp = (name: string, a: number[] | number, b: number[] | number) => {
  const aa = Array.isArray(a) ? a : [a]
  const bb = Array.isArray(b) ? b : [b]
  let d = 0
  for (let i = 0; i < aa.length; i++) d = Math.max(d, Math.abs(aa[i]! - bb[i]!) / Math.max(1, Math.abs(bb[i]!)))
  worst = Math.max(worst, d)
  console.log(`  ${name.padEnd(28)} max rel diff ${d.toExponential(2)}  ${d < 1e-9 ? 'OK' : d < 1e-6 ? 'ok (fp)' : 'MISMATCH'}`)
}

const r = await df.withWindow('r', 'rank', { orderBy: ['w'] }).collect()
const rr = r.getColumn('r').toArray() as number[]
cmp('rank(w) sum', rr.reduce((a, b) => a + b, 0), ref.rank_w_sum)
// orderBy sorts the frame, so compare by original position: re-rank via expr keeps input order
const r2 = await df.withWindow('r', 'rank', { expr: col('w') }).collect()
cmp('rank(w) first 10 (expr)', (r2.getColumn('r').toArray() as number[]).slice(0, 10), ref.rank_w_first10)

const z = await df.withColumn('z1', col('x').sub(col('x').mean()).div(col('x').std())).collect()
cmp('zscore first 5', (z.getColumn('z1').toArray() as number[]).slice(0, 5), ref.zscore_first5)

const gm = await df.select(col('x').log().mean().exp().alias('gm')).collect()
cmp('geo-mean', Number(gm.toArray()[0]!.gm), ref.geo_mean_x)
const lg = await df.select(col('x').log().alias('l')).collect()
cmp('log first 3', (lg.getColumn('l').toArray() as number[]).slice(0, 3), ref.log_first3)
const rd = await df.select(col('x').round(2).alias('r')).collect()
cmp('round(2) first 5 [half-away vs pandas half-even]', (rd.getColumn('r').toArray() as number[]).slice(0, 5), ref.round2_first5)

const med = await df.filter(col('x').gt(col('x').median())).collect()
cmp('rows > median', med.shape[0], ref.rows_gt_median)

const names = ['x', 'y', 'z', 'w', 'v']
const flat = (d: DataFrame, table: Record<string, Record<string, number>>) => {
  const a: number[] = []
  const b: number[] = []
  for (const row of d.toArray()) for (const c of names) {
    a.push(Number(row[c]))
    b.push(table[c]![row.column as string]!)
  }
  return [a, b] as const
}
cmp('corr pearson 5×5', ...flat(await df.corr().collect(), ref.corr_pearson))
cmp('corr spearman 5×5', ...flat(await df.corr({ method: 'spearman' }).collect(), ref.corr_spearman))
cmp('cov 5×5', ...flat(await df.cov().collect(), ref.cov))
console.log(`\nworst relative difference vs pandas: ${worst.toExponential(2)}`)
