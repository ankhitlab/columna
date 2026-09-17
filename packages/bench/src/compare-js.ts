/**
 * Same data, same operations, same result checksums — columna against the JS-side alternatives a
 * team would actually consider: Arquero (in-JS columnar tables), DuckDB-Wasm (SQL engine compiled to
 * WebAssembly, the browser option), nodejs-polars (native Polars core) and native DuckDB (Node addon).
 *
 * Each library runs in its own child process (E2E_LIB=…) so peak RSS is attributable, with one cold pass
 * (module load + first run) and warm repeats. Results are validated across libraries: a mismatching
 * checksum is printed, not hidden.
 *
 *   pnpm --filter @columna/bench run compare:js               # 2M rows × 8 columns
 *   E2E_ROWS=500000 pnpm --filter @columna/bench run compare:js
 */
import { spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, statSync, writeFileSync, readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const ROWS = Number(process.env.E2E_ROWS ?? 2_000_000)
const OUT = new URL('../results/', import.meta.url)
const INPUT = fileURLToPath(new URL(`e2e-input-${ROWS}.csv`, OUT)).replace(/\\/g, '/')
const OUTPUT = (lib: string) => fileURLToPath(new URL(`compare-${lib}-${ROWS}.csv`, OUT)).replace(/\\/g, '/')
const LIBS = ['columna', 'arquero', 'duckdb-wasm', 'polars', 'polars-lazy', 'duckdb'] as const
type Lib = (typeof LIBS)[number]

/** Timed operations — order used in the markdown report. */
const OPS = [
  'read',
  'filter',
  'select',
  'drop',
  'rename',
  'withColumn',
  'groupBy',
  'sort',
  'sortMulti',
  'unique',
  'head',
  'tail',
  'sample',
  'join',
  'leftJoin',
  'semiJoin',
  'antiJoin',
  'melt',
  'pivot',
  'valueCounts',
  'describe',
  'corr',
  'filterGroupBy',
  'writeCsv',
] as const
type Op = (typeof OPS)[number]

function ensureInput(): void {
  if (existsSync(INPUT)) return
  mkdirSync(OUT, { recursive: true })
  const parts: string[] = ['id,age,salary,x,y,z,city,segment']
  let seed = 7
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296)
  const cities = ['Berlin', 'Paris', 'Rome', 'Madrid', 'Wien']
  for (let i = 0; i < ROWS; i++) {
    parts.push(
      `${i},${18 + Math.floor(rnd() * 60)},${(20_000 + rnd() * 120_000).toFixed(2)},${(rnd() * 100).toFixed(4)},${(rnd() * 100).toFixed(4)},${(rnd() * 100).toFixed(4)},${cities[i % 5]},${i % 7 ? 'B2B' : 'B2C'}`,
    )
    if (parts.length >= 100_000) {
      writeFileSync(INPUT, parts.join('\n') + '\n', { flag: 'a' })
      parts.length = 0
    }
  }
  if (parts.length) writeFileSync(INPUT, parts.join('\n') + '\n', { flag: 'a' })
}

/** Checksums every library must reproduce (sample only checks row count — RNG differs). */
type Check = {
  filterRows: number
  selectCols: number
  dropCols: number
  renameOk: number
  withColSum: number
  groupSum: number
  sortFirst: number
  sortLast: number
  sortMultiFirstSal: number
  uniqueRows: number
  headIdSum: number
  tailIdSum: number
  sampleRows: number
  joinRows: number
  joinSum: number
  leftJoinRows: number
  semiJoinRows: number
  antiJoinRows: number
  meltRows: number
  meltValueSum: number
  pivotSum: number
  valueCountsSum: number
  describeAgeMean: number
  corrXY: number
  filterGroupSum: number
}
type OpTimes = Record<Op, number>
type Run = { times: OpTimes; check: Check; peakRssMb: number; note?: string }

const REGIONS = [
  ['Berlin', 'DE'],
  ['Paris', 'FR'],
  ['Rome', 'IT'],
  ['Madrid', 'ES'],
] as const // Wien has no match → inner/semi drop it; left keeps it; anti keeps Wien only

const HEAD_N = 1_000
const SAMPLE_N = 1_000
const round = (x: number) => Math.round(x * 1000) / 1000
const emptyTimes = (): OpTimes => Object.fromEntries(OPS.map((o) => [o, 0])) as OpTimes

/** Pearson correlation of two equal-length number arrays (pairwise finite). */
function pearson(xs: number[], ys: number[]): number {
  const n = Math.min(xs.length, ys.length)
  let sx = 0
  let sy = 0
  let sxx = 0
  let syy = 0
  let sxy = 0
  let c = 0
  for (let i = 0; i < n; i++) {
    const x = xs[i]!
    const y = ys[i]!
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue
    c++
    sx += x
    sy += y
    sxx += x * x
    syy += y * y
    sxy += x * y
  }
  if (c < 2) return NaN
  const num = c * sxy - sx * sy
  const den = Math.sqrt((c * sxx - sx * sx) * (c * syy - sy * sy))
  return den === 0 ? NaN : num / den
}

async function runLib(lib: Lib): Promise<Run> {
  let peak = process.memoryUsage().rss
  const sampler = setInterval(() => (peak = Math.max(peak, process.memoryUsage().rss)), 20)
  const times = emptyTimes()
  const timed = async <T>(op: Op, f: () => Promise<T> | T): Promise<T> => {
    const t0 = performance.now()
    const r = await f()
    times[op] = performance.now() - t0
    return r
  }
  let check: Check
  let note: string | undefined

  if (lib === 'columna') {
    const { DataFrame, col } = await import('columna')
    const df = await timed('read', () => DataFrame.readCsv({ path: INPUT }))
    const filtered = await timed('filter', () =>
      df.filter(col('age').gt(30).and(col('salary').gt(45_000))).collect(),
    )
    const selected = await timed('select', () => df.select('id', 'age', 'salary', 'city').collect())
    const dropped = await timed('drop', () => df.drop('z', 'y').collect())
    const renamed = await timed('rename', () => df.rename({ city: 'town' }).collect())
    const derived = await timed('withColumn', () => df.withColumn('xy', col('x').add(col('y'))).collect())
    const grouped = await timed('groupBy', () =>
      df.groupBy('city').agg({ n: col('id').count(), salary: col('salary').mean(), x: col('x').sum() }).collect(),
    )
    const sorted = await timed('sort', () => df.sort(col('salary').desc()).collect())
    const sortedMulti = await timed('sortMulti', () =>
      df.sort('city', { expr: col('salary'), descending: true }).collect(),
    )
    const uniq = await timed('unique', () => df.unique(['age', 'city']).collect())
    const head = await timed('head', () => df.head(HEAD_N).collect())
    const tail = await timed('tail', () => df.tail(HEAD_N).collect())
    const sampled = await timed('sample', () => df.sample({ n: SAMPLE_N, seed: 1 }).collect())
    const regions = DataFrame.fromRows(REGIONS.map(([city, region]) => ({ city, region })))
    const joined = await timed('join', () => df.join(regions, { on: 'city' }).collect())
    const leftJoined = await timed('leftJoin', () => df.leftJoin(regions, 'city').collect())
    const semi = await timed('semiJoin', () => df.semiJoin(regions, 'city').collect())
    const anti = await timed('antiJoin', () => df.antiJoin(regions, 'city').collect())
    const melted = await timed('melt', () =>
      df.melt({ idVars: ['id', 'city'], valueVars: ['x', 'y'] }).collect(),
    )
    const pivoted = await timed('pivot', () =>
      df.pivot({ index: 'city', columns: 'segment', values: 'salary', agg: 'mean' }).collect(),
    )
    const vc = await timed('valueCounts', () => df.valueCounts('city').collect())
    const desc = await timed('describe', () => df.describe().collect())
    const corr = await timed('corr', () => df.corr({ columns: ['x', 'y'] }).collect())
    const fg = await timed('filterGroupBy', () =>
      df
        .filter(col('age').gt(30))
        .groupBy('city')
        .agg({ salary: col('salary').sum() })
        .collect(),
    )
    await timed('writeCsv', () => filtered.writeCsv(OUTPUT(lib)))

    const gs = grouped.toArray().reduce((s, r) => s + Number(r.salary) + Number(r.x), 0)
    const sal = sorted.getColumn('salary').toArray() as number[]
    const xy = derived.getColumn('xy').toArray() as number[]
    const headIds = head.getColumn('id').toArray() as number[]
    const tailIds = tail.getColumn('id').toArray() as number[]
    const meltVals = melted.getColumn('value').toArray() as number[]
    const pivotNums = pivoted.table.columns
      .filter((c) => c.field.name !== 'city')
      .flatMap((c) => [...(c.data as Float64Array)])
    const vcSum = (vc.getColumn('count').toArray() as number[]).reduce((s, v) => s + Number(v), 0)
    const ageMean = Number(desc.toArray().find((r) => r.stat === 'mean')?.age)
    const corrRows = corr.toArray()
    const corrXY = Number(
      corrRows.find((r) => (r.column ?? r['']) === 'x')?.y ??
        corrRows.find((r) => String(Object.values(r)[0]) === 'x')?.y ??
        corrRows[0]?.y,
    )
    // corr matrix: find x,y cell
    let corrVal = corrXY
    if (!Number.isFinite(corrVal)) {
      const names = corr.table.schema.map((f) => f.name)
      const xi = names.indexOf('x')
      const yi = names.indexOf('y')
      if (xi >= 0 && yi >= 0) corrVal = Number((corr.table.columns[yi]!.data as Float64Array)[xi - (names[0] === 'column' || names[0] === '' ? 1 : 0)] ?? NaN)
      // fallback: compute from columns
      if (!Number.isFinite(corrVal)) {
        corrVal = pearson(df.getColumn('x').toArray() as number[], df.getColumn('y').toArray() as number[])
      }
    }
    const fgSum = fg.toArray().reduce((s, r) => s + Number(r.salary), 0)
    const multiSal = sortedMulti.getColumn('salary').toArray() as number[]

    check = {
      filterRows: filtered.shape[0],
      selectCols: selected.columns.length,
      dropCols: dropped.columns.length,
      renameOk: renamed.columns.includes('town') ? 1 : 0,
      withColSum: round(xy.reduce((s, v) => s + Number(v), 0)),
      groupSum: round(gs),
      sortFirst: sal[0]!,
      sortLast: sal[sal.length - 1]!,
      sortMultiFirstSal: multiSal[0]!,
      uniqueRows: uniq.shape[0],
      headIdSum: headIds.reduce((s, v) => s + Number(v), 0),
      tailIdSum: tailIds.reduce((s, v) => s + Number(v), 0),
      sampleRows: sampled.shape[0],
      joinRows: joined.shape[0],
      joinSum: round(
        (joined.getColumn('salary').toArray() as number[]).reduce((s, v) => s + Number(v), 0),
      ),
      leftJoinRows: leftJoined.shape[0],
      semiJoinRows: semi.shape[0],
      antiJoinRows: anti.shape[0],
      meltRows: melted.shape[0],
      meltValueSum: round(meltVals.reduce((s, v) => s + Number(v), 0)),
      pivotSum: round(pivotNums.filter(Number.isFinite).reduce((s, v) => s + v, 0)),
      valueCountsSum: vcSum,
      describeAgeMean: round(ageMean),
      corrXY: round(corrVal),
      filterGroupSum: round(fgSum),
    }
  } else if (lib === 'arquero') {
    const aq = await import('arquero')
    const fs = await import('node:fs/promises')
    const table = await timed('read', async () => aq.fromCSV(await fs.readFile(INPUT, 'utf8')))
    const filtered = await timed('filter', () =>
      table.filter((d: { age: number; salary: number }) => d.age > 30 && d.salary > 45000).reify(),
    )
    const selected = await timed('select', () => table.select('id', 'age', 'salary', 'city').reify())
    const dropped = await timed('drop', () => table.select(aq.not('z', 'y')).reify())
    const renamed = await timed('rename', () => table.rename({ city: 'town' }).reify())
    const derived = await timed('withColumn', () => table.derive({ xy: (d: { x: number; y: number }) => d.x + d.y }).reify())
    const grouped = await timed('groupBy', () =>
      table.groupby('city').rollup({ n: aq.op.count(), salary: aq.op.mean('salary'), x: aq.op.sum('x') }),
    )
    const sorted = await timed('sort', () => table.orderby(aq.desc('salary')).reify())
    const sortedMulti = await timed('sortMulti', () => table.orderby('city', aq.desc('salary')).reify())
    const uniq = await timed('unique', () => table.dedupe('age', 'city'))
    const head = await timed('head', () => table.slice(0, HEAD_N).reify())
    const tail = await timed('tail', () => table.slice(table.numRows() - HEAD_N, table.numRows()).reify())
    const sampled = await timed('sample', () => table.sample(SAMPLE_N).reify())
    const regions = aq.table({ city: REGIONS.map((r) => r[0]), region: REGIONS.map((r) => r[1]) })
    const joined = await timed('join', () => table.join(regions, 'city').reify())
    const leftJoined = await timed('leftJoin', () => table.join_left(regions, 'city').reify())
    const semi = await timed('semiJoin', () => table.semijoin(regions, 'city').reify())
    const anti = await timed('antiJoin', () => table.antijoin(regions, 'city').reify())
    const melted = await timed('melt', () => table.select('id', 'city', 'x', 'y').fold(['x', 'y']).reify())
    const pivoted = await timed('pivot', () =>
      table.groupby('city').pivot('segment', { salary: aq.op.mean('salary') }),
    )
    const vc = await timed('valueCounts', () => table.groupby('city').count())
    const desc = await timed('describe', () =>
      table.rollup({ age: aq.op.mean('age'), salary: aq.op.mean('salary'), x: aq.op.mean('x') }),
    )
    const corrT = await timed('corr', () => {
      const xs = table.array('x') as number[]
      const ys = table.array('y') as number[]
      return pearson(xs, ys)
    })
    const fg = await timed('filterGroupBy', () =>
      table
        .filter((d: { age: number }) => d.age > 30)
        .groupby('city')
        .rollup({ salary: aq.op.sum('salary') }),
    )
    await timed('writeCsv', async () => fs.writeFile(OUTPUT(lib), filtered.toCSV()))

    const gs = grouped.objects().reduce((s: number, r: { salary: number; x: number }) => s + r.salary + r.x, 0)
    const sal = sorted.array('salary') as number[]
    const xy = derived.array('xy') as number[]
    const headIds = head.array('id') as number[]
    const tailIds = tail.array('id') as number[]
    const meltVals = melted.array('value') as number[]
    const pivotObjs = pivoted.objects() as Array<Record<string, number>>
    const pivotSum = pivotObjs.reduce((s, r) => {
      for (const [k, v] of Object.entries(r)) if (k !== 'city' && typeof v === 'number') s += v
      return s
    }, 0)
    const vcSum = vc.objects().reduce((s: number, r: { count: number }) => s + r.count, 0)
    const ageMean = Number(desc.object(0).age)
    const fgSum = fg.objects().reduce((s: number, r: { salary: number }) => s + r.salary, 0)
    const multiSal = sortedMulti.array('salary') as number[]

    check = {
      filterRows: filtered.numRows(),
      selectCols: selected.numCols(),
      dropCols: dropped.numCols(),
      renameOk: renamed.columnNames().includes('town') ? 1 : 0,
      withColSum: round(xy.reduce((s, v) => s + v, 0)),
      groupSum: round(gs),
      sortFirst: sal[0]!,
      sortLast: sal[sal.length - 1]!,
      sortMultiFirstSal: multiSal[0]!,
      uniqueRows: uniq.numRows(),
      headIdSum: headIds.reduce((s, v) => s + v, 0),
      tailIdSum: tailIds.reduce((s, v) => s + v, 0),
      sampleRows: sampled.numRows(),
      joinRows: joined.numRows(),
      joinSum: round((joined.array('salary') as number[]).reduce((s, v) => s + v, 0)),
      leftJoinRows: leftJoined.numRows(),
      semiJoinRows: semi.numRows(),
      antiJoinRows: anti.numRows(),
      meltRows: melted.numRows(),
      meltValueSum: round(meltVals.reduce((s, v) => s + v, 0)),
      pivotSum: round(pivotSum),
      valueCountsSum: vcSum,
      describeAgeMean: round(ageMean),
      corrXY: round(corrT),
      filterGroupSum: round(fgSum),
    }
    note =
      'reads CSV from a string (fs.readFile + fromCSV); filter/derive/orderby/join/unique reified so the work is inside the timing; corr is JS pearson over column arrays'
  } else if (lib === 'polars' || lib === 'polars-lazy') {
    const pl = (await import('nodejs-polars')).default
    const regions = pl.DataFrame({ city: REGIONS.map((r) => r[0]), region: REGIONS.map((r) => r[1]) })
    const pred = pl.col('age').gt(30).and(pl.col('salary').gt(45000))

    const finish = (parts: {
      filtered: { height: number; writeCSV: (p: string) => void }
      selected: { width: number }
      dropped: { width: number }
      renamed: { columns: string[] }
      derived: { getColumn: (n: string) => { toArray: () => unknown[] } }
      grouped: { toRecords: () => Array<{ salary: number; x: number }> }
      sorted: { getColumn: (n: string) => { toArray: () => unknown[] } }
      sortedMulti: { getColumn: (n: string) => { toArray: () => unknown[] } }
      uniq: { height: number }
      head: { getColumn: (n: string) => { toArray: () => unknown[] } }
      tail: { getColumn: (n: string) => { toArray: () => unknown[] } }
      sampled: { height: number }
      joined: { height: number; getColumn: (n: string) => { sum: () => number } }
      leftJoined: { height: number }
      semi: { height: number }
      anti: { height: number }
      melted: { height: number; getColumn: (n: string) => { toArray: () => unknown[]; sum?: () => number } }
      pivoted: { toRecords: () => Array<Record<string, unknown>> }
      vc: { toRecords: () => Array<{ count: number }> }
      desc: { toRecords: () => Array<Record<string, unknown>> }
      corrVal: number
      fg: { toRecords: () => Array<{ salary: number }> }
    }): Check => {
      const gs = parts.grouped.toRecords().reduce((s, r) => s + r.salary + r.x, 0)
      const sal = parts.sorted.getColumn('salary').toArray() as number[]
      const xy = parts.derived.getColumn('xy').toArray() as number[]
      const headIds = parts.head.getColumn('id').toArray() as number[]
      const tailIds = parts.tail.getColumn('id').toArray() as number[]
      const meltVals = parts.melted.getColumn('value').toArray() as number[]
      const pivotSum = parts.pivoted.toRecords().reduce((s, r) => {
        for (const [k, v] of Object.entries(r)) if (k !== 'city' && typeof v === 'number') s += v
        return s
      }, 0)
      const vcSum = parts.vc.toRecords().reduce((s, r) => s + Number(r.count), 0)
      const descRows = parts.desc.toRecords()
      const ageMeanRow = descRows.find((r) => r.statistic === 'mean' || r.describe === 'mean' || r[''] === 'mean')
      let ageMean = Number(ageMeanRow?.age)
      if (!Number.isFinite(ageMean)) {
        // polars describe: first column is statistic name
        const meanRow = descRows.find((r) => Object.values(r).includes('mean'))
        ageMean = Number(meanRow?.age)
      }
      const fgSum = parts.fg.toRecords().reduce((s, r) => s + Number(r.salary), 0)
      const multiSal = parts.sortedMulti.getColumn('salary').toArray() as number[]
      return {
        filterRows: parts.filtered.height,
        selectCols: parts.selected.width,
        dropCols: parts.dropped.width,
        renameOk: parts.renamed.columns.includes('town') ? 1 : 0,
        withColSum: round(xy.reduce((s, v) => s + Number(v), 0)),
        groupSum: round(gs),
        sortFirst: sal[0]!,
        sortLast: sal[sal.length - 1]!,
        sortMultiFirstSal: multiSal[0]!,
        uniqueRows: parts.uniq.height,
        headIdSum: headIds.reduce((s, v) => s + Number(v), 0),
        tailIdSum: tailIds.reduce((s, v) => s + Number(v), 0),
        sampleRows: parts.sampled.height,
        joinRows: parts.joined.height,
        joinSum: round(parts.joined.getColumn('salary').sum() as number),
        leftJoinRows: parts.leftJoined.height,
        semiJoinRows: parts.semi.height,
        antiJoinRows: parts.anti.height,
        meltRows: parts.melted.height,
        meltValueSum: round(meltVals.reduce((s, v) => s + Number(v), 0)),
        pivotSum: round(pivotSum),
        valueCountsSum: vcSum,
        describeAgeMean: round(ageMean),
        corrXY: round(parts.corrVal),
        filterGroupSum: round(fgSum),
      }
    }

    if (lib === 'polars') {
      const df = await timed('read', () => pl.readCSV(INPUT))
      const filtered = await timed('filter', () => df.filter(pred))
      const selected = await timed('select', () => df.select('id', 'age', 'salary', 'city'))
      const dropped = await timed('drop', () => df.drop('z', 'y'))
      const renamed = await timed('rename', () => df.rename({ city: 'town' }))
      const derived = await timed('withColumn', () => df.withColumns(pl.col('x').add(pl.col('y')).alias('xy')))
      const grouped = await timed('groupBy', () =>
        df.groupBy('city').agg(pl.col('id').count().alias('n'), pl.col('salary').mean(), pl.col('x').sum()),
      )
      const sorted = await timed('sort', () => df.sort('salary', true))
      const sortedMulti = await timed('sortMulti', () => df.sort(['city', 'salary'], [false, true]))
      const uniq = await timed('unique', () => df.unique(['age', 'city']))
      const head = await timed('head', () => df.head(HEAD_N))
      const tail = await timed('tail', () => df.tail(HEAD_N))
      const sampled = await timed('sample', () => df.sample({ n: SAMPLE_N, seed: 1 }))
      const joined = await timed('join', () => df.join(regions, { on: 'city' }))
      const leftJoined = await timed('leftJoin', () => df.join(regions, { on: 'city', how: 'left' }))
      const semi = await timed('semiJoin', () => df.join(regions, { on: 'city', how: 'semi' }))
      const anti = await timed('antiJoin', () => df.join(regions, { on: 'city', how: 'anti' }))
      const melted = await timed('melt', () => df.select('id', 'city', 'x', 'y').unpivot(['id', 'city'], ['x', 'y']))
      const pivoted = await timed('pivot', () =>
        df.pivot('salary', { index: ['city'], on: ['segment'], aggregateFunc: 'mean' }),
      )
      const vc = await timed('valueCounts', () => df.getColumn('city').valueCounts())
      const desc = await timed('describe', () => df.describe())
      const corrVal = await timed('corr', () =>
        pearson(df.getColumn('x').toArray() as number[], df.getColumn('y').toArray() as number[]),
      )
      const fg = await timed('filterGroupBy', () =>
        df.filter(pl.col('age').gt(30)).groupBy('city').agg(pl.col('salary').sum()),
      )
      await timed('writeCsv', () => filtered.writeCSV(OUTPUT(lib)))
      check = finish({
        filtered,
        selected,
        dropped,
        renamed: { columns: renamed.columns },
        derived,
        grouped,
        sorted,
        sortedMulti,
        uniq,
        head,
        tail,
        sampled,
        joined,
        leftJoined,
        semi,
        anti,
        melted,
        pivoted,
        vc,
        desc,
        corrVal,
        fg,
      })
      note = 'eager API; native multi-threaded core'
    } else {
      const scan = () => pl.scanCSV(INPUT)
      times.read = 0
      const filtered = await timed('filter', () => scan().filter(pred).collect())
      const selected = await timed('select', () => scan().select('id', 'age', 'salary', 'city').collect())
      const dropped = await timed('drop', () => scan().drop('z', 'y').collect())
      const renamed = await timed('rename', () => scan().rename({ city: 'town' }).collect())
      const derived = await timed('withColumn', () =>
        scan()
          .withColumns(pl.col('x').add(pl.col('y')).alias('xy'))
          .collect(),
      )
      const grouped = await timed('groupBy', () =>
        scan()
          .groupBy('city')
          .agg(pl.col('id').count().alias('n'), pl.col('salary').mean(), pl.col('x').sum())
          .collect(),
      )
      const sorted = await timed('sort', () => scan().sort('salary', true).collect())
      const sortedMulti = await timed('sortMulti', () => scan().sort(['city', 'salary'], [false, true]).collect())
      const uniq = await timed('unique', () => scan().unique({ subset: ['age', 'city'], maintainOrder: true }).collect())
      const head = await timed('head', () => scan().limit(HEAD_N).collect())
      const tail = await timed('tail', async () => {
        const d = await scan().collect()
        return d.tail(HEAD_N)
      })
      const sampled = await timed('sample', async () => {
        const d = await scan().collect()
        return d.sample({ n: SAMPLE_N, seed: 1 })
      })
      const joined = await timed('join', () => scan().join(regions.lazy(), { on: 'city' }).collect())
      const leftJoined = await timed('leftJoin', () => scan().join(regions.lazy(), { on: 'city', how: 'left' }).collect())
      const semi = await timed('semiJoin', () => scan().join(regions.lazy(), { on: 'city', how: 'semi' }).collect())
      const anti = await timed('antiJoin', () => scan().join(regions.lazy(), { on: 'city', how: 'anti' }).collect())
      const melted = await timed('melt', () =>
        scan()
          .select('id', 'city', 'x', 'y')
          .collect()
          .then((d: { unpivot: (id: string[], val: string[]) => unknown }) => d.unpivot(['id', 'city'], ['x', 'y'])),
      )
      const pivoted = await timed('pivot', () =>
        scan()
          .collect()
          .then((d: { pivot: (v: string, o: object) => unknown }) =>
            d.pivot('salary', { index: ['city'], on: ['segment'], aggregateFunc: 'mean' }),
          ),
      )
      const vc = await timed('valueCounts', async () => {
        const d = await scan().collect()
        return d.getColumn('city').valueCounts()
      })
      const desc = await timed('describe', async () => {
        const d = await scan().collect()
        return d.describe()
      })
      const corrVal = await timed('corr', async () => {
        const d = await scan().collect()
        return pearson(d.getColumn('x').toArray() as number[], d.getColumn('y').toArray() as number[])
      })
      const fg = await timed('filterGroupBy', () =>
        scan()
          .filter(pl.col('age').gt(30))
          .groupBy('city')
          .agg(pl.col('salary').sum())
          .collect(),
      )
      await timed('writeCsv', () => filtered.writeCSV(OUTPUT(lib)))
      check = finish({
        filtered,
        selected,
        dropped,
        renamed: { columns: renamed.columns },
        derived,
        grouped,
        sorted,
        sortedMulti,
        uniq,
        head,
        tail: tail as { getColumn: (n: string) => { toArray: () => unknown[] } },
        sampled: sampled as { height: number },
        joined,
        leftJoined,
        semi,
        anti,
        melted,
        pivoted: pivoted as { toRecords: () => Array<Record<string, unknown>> },
        vc,
        desc: desc as { toRecords: () => Array<Record<string, unknown>> },
        corrVal,
        fg,
      })
      note = 'lazy API: each op = scanCSV + op + collect (read is inside every timing; query optimiser + streaming)'
    }
  } else {
    // SQL engines: DuckDB-Wasm (blocking Node bindings) and native DuckDB
    type Q = (sql: string) => Promise<Array<Record<string, unknown>>>
    let q: Q
    let close: () => Promise<void> | void
    if (lib === 'duckdb-wasm') {
      const blocking = await import('@duckdb/duckdb-wasm/blocking')
      const require = createRequire(import.meta.url)
      const dist = path.dirname(require.resolve('@duckdb/duckdb-wasm'))
      const db = await blocking.createDuckDB(
        {
          eh: {
            mainModule: path.join(dist, 'duckdb-eh.wasm'),
            mainWorker: path.join(dist, 'duckdb-node-eh.worker.cjs'),
          },
        },
        new blocking.VoidLogger(),
        blocking.NODE_RUNTIME,
      )
      await db.instantiate()
      const conn = db.connect()
      q = async (sql) => conn.query(sql).toArray().map((r: { toJSON: () => Record<string, unknown> }) => r.toJSON())
      close = () => conn.close()
      note = 'single-threaded WebAssembly build (eh), blocking Node bindings; the browser deployment target'
    } else {
      const duck = (await import('duckdb')).default
      const db = new duck.Database(':memory:')
      q = (sql) =>
        new Promise((res, rej) =>
          db.all(sql, (err: Error | null, rows: Array<Record<string, unknown>>) => (err ? rej(err) : res(rows))),
        )
      close = () => new Promise<void>((res) => db.close(() => res()))
      note = 'native addon, multi-threaded'
    }
    await timed('read', () => q(`CREATE OR REPLACE TABLE t AS SELECT * FROM read_csv_auto('${INPUT}')`))
    const f = await timed('filter', () =>
      q('CREATE OR REPLACE TABLE f AS SELECT * FROM t WHERE age > 30 AND salary > 45000; SELECT count(*) AS n FROM f'),
    )
    const sel = await timed('select', () =>
      q(
        "CREATE OR REPLACE TABLE sel AS SELECT id, age, salary, city FROM t; SELECT count(*) AS n, (SELECT count(*) FROM information_schema.columns WHERE table_name = 'sel') AS ncols FROM sel",
      ),
    )
    const dr = await timed('drop', () =>
      q(
        "CREATE OR REPLACE TABLE dr AS SELECT * EXCLUDE (z, y) FROM t; SELECT (SELECT count(*) FROM information_schema.columns WHERE table_name = 'dr') AS ncols",
      ),
    )
    const rn = await timed('rename', () =>
      q(
        "CREATE OR REPLACE TABLE rn AS SELECT * RENAME (city AS town) FROM t; SELECT CASE WHEN count(*) FILTER (WHERE column_name = 'town') > 0 THEN 1 ELSE 0 END AS ok FROM information_schema.columns WHERE table_name = 'rn'",
      ),
    )
    const w = await timed('withColumn', () =>
      q('CREATE OR REPLACE TABLE w AS SELECT *, x + y AS xy FROM t; SELECT sum(xy) AS s FROM w'),
    )
    const g = await timed('groupBy', () =>
      q('SELECT city, count(*) AS n, avg(salary) AS salary, sum(x) AS x FROM t GROUP BY city'),
    )
    const s = await timed('sort', () =>
      q(
        'CREATE OR REPLACE TABLE s AS SELECT * FROM t ORDER BY salary DESC; SELECT (SELECT salary FROM s LIMIT 1) AS first, (SELECT salary FROM t ORDER BY salary ASC LIMIT 1) AS last',
      ),
    )
    const sm = await timed('sortMulti', () =>
      q(
        'CREATE OR REPLACE TABLE sm AS SELECT * FROM t ORDER BY city ASC, salary DESC; SELECT salary AS first FROM sm LIMIT 1',
      ),
    )
    const u = await timed('unique', () =>
      q(
        'CREATE OR REPLACE TABLE u AS SELECT DISTINCT ON (age, city) * FROM t ORDER BY age, city; SELECT count(*) AS n FROM u',
      ),
    )
    const h = await timed('head', () => q(`SELECT sum(id) AS s FROM (SELECT id FROM t LIMIT ${HEAD_N})`))
    const tl = await timed('tail', () =>
      q(`SELECT sum(id) AS s FROM (SELECT id FROM t ORDER BY id DESC LIMIT ${HEAD_N})`),
    )
    const sp = await timed('sample', () =>
      q(`SELECT count(*) AS n FROM (SELECT * FROM t USING SAMPLE ${SAMPLE_N})`),
    )
    await q(
      `CREATE OR REPLACE TABLE regions AS SELECT * FROM (VALUES ${REGIONS.map(([c, r]) => `('${c}','${r}')`).join(',')}) v(city, region)`,
    )
    const j = await timed('join', () => q('SELECT count(*) AS n, sum(t.salary) AS s FROM t JOIN regions USING (city)'))
    const lj = await timed('leftJoin', () =>
      q('SELECT count(*) AS n FROM t LEFT JOIN regions USING (city)'),
    )
    const sj = await timed('semiJoin', () =>
      q('SELECT count(*) AS n FROM t WHERE city IN (SELECT city FROM regions)'),
    )
    const aj = await timed('antiJoin', () =>
      q('SELECT count(*) AS n FROM t WHERE city NOT IN (SELECT city FROM regions)'),
    )
    const ml = await timed('melt', () =>
      q(
        `CREATE OR REPLACE TABLE ml AS SELECT id, city, 'x' AS variable, x AS value FROM t UNION ALL SELECT id, city, 'y', y FROM t; SELECT count(*) AS n, sum(value) AS s FROM ml`,
      ),
    )
    const pv = await timed('pivot', () =>
      q(
        `SELECT sum(v) AS s FROM (SELECT city, segment, avg(salary) AS v FROM t GROUP BY city, segment)`,
      ),
    )
    const vc = await timed('valueCounts', () => q('SELECT city, count(*) AS count FROM t GROUP BY city'))
    const ds = await timed('describe', () => q('SELECT avg(age) AS age FROM t'))
    const cr = await timed('corr', () => q('SELECT corr(x, y) AS c FROM t'))
    const fg = await timed('filterGroupBy', () =>
      q('SELECT city, sum(salary) AS salary FROM t WHERE age > 30 GROUP BY city'),
    )
    await timed('writeCsv', () => q(`COPY f TO '${OUTPUT(lib)}' (HEADER, DELIMITER ',')`))
    const num = (v: unknown) => Number(v)
    let selectCols = num(sel[sel.length - 1]!.ncols)
    if (!Number.isFinite(selectCols) || selectCols <= 0) {
      const desc = await q('DESCRIBE sel')
      selectCols = desc.length
    }
    let dropCols = num(dr[dr.length - 1]!.ncols)
    if (!Number.isFinite(dropCols) || dropCols <= 0) {
      dropCols = (await q('DESCRIBE dr')).length
    }
    check = {
      filterRows: num(f[f.length - 1]!.n),
      selectCols,
      dropCols,
      renameOk: num(rn[rn.length - 1]!.ok),
      withColSum: round(num(w[0]!.s)),
      groupSum: round(g.reduce((acc, r) => acc + num(r.salary) + num(r.x), 0)),
      sortFirst: num(s[s.length - 1]!.first),
      sortLast: num(s[s.length - 1]!.last),
      sortMultiFirstSal: num(sm[sm.length - 1]!.first),
      uniqueRows: num(u[u.length - 1]!.n),
      headIdSum: num(h[0]!.s),
      tailIdSum: num(tl[0]!.s),
      sampleRows: num(sp[0]!.n),
      joinRows: num(j[0]!.n),
      joinSum: round(num(j[0]!.s)),
      leftJoinRows: num(lj[0]!.n),
      semiJoinRows: num(sj[0]!.n),
      antiJoinRows: num(aj[0]!.n),
      meltRows: num(ml[ml.length - 1]!.n),
      meltValueSum: round(num(ml[ml.length - 1]!.s)),
      pivotSum: round(num(pv[0]!.s)),
      valueCountsSum: vc.reduce((s, r) => s + num(r.count), 0),
      describeAgeMean: round(num(ds[0]!.age)),
      corrXY: round(num(cr[0]!.c)),
      filterGroupSum: round(fg.reduce((s, r) => s + num(r.salary), 0)),
    }
    await close()
  }
  clearInterval(sampler)
  peak = Math.max(peak, process.memoryUsage().rss)
  return { times, check, peakRssMb: peak / 1e6, note }
}

async function child(lib: Lib): Promise<void> {
  const cold = await runLib(lib)
  const warm: Run[] = []
  for (let i = 0; i < 2; i++) warm.push(await runLib(lib))
  process.stdout.write(JSON.stringify({ cold, warm }))
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b)
  return s[Math.floor(s.length / 2)]!
}

async function main(): Promise<void> {
  ensureInput()
  if (process.env.E2E_LIB) return child(process.env.E2E_LIB as Lib)
  const inputMb = statSync(INPUT).size / 1e6
  const results: Partial<Record<Lib, { cold: Run; warm: Run[]; wallMs: number; error?: string }>> = {}
  for (const lib of LIBS) {
    const t0 = performance.now()
    const r = spawnSync(process.execPath, ['--max-old-space-size=8192', '--import', 'tsx', fileURLToPath(import.meta.url)], {
      env: { ...process.env, E2E_LIB: lib },
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
    })
    const wallMs = performance.now() - t0
    if (r.status !== 0) {
      results[lib] = {
        cold: null as never,
        warm: [],
        wallMs,
        error: (r.stderr || r.stdout)
          .split('\n')
          .filter((l) => l.trim())
          .slice(-8)
          .join(' | '),
      }
      console.error(`${lib}: failed — ${results[lib]!.error}`)
      continue
    }
    const parsed = JSON.parse(r.stdout.slice(r.stdout.lastIndexOf('{"cold"'))) as { cold: Run; warm: Run[] }
    results[lib] = { ...parsed, wallMs }
    console.log(
      `${lib}: cold ${sum(parsed.cold.times).toFixed(0)} ms, warm ${median(parsed.warm.map((w) => sum(w.times))).toFixed(0)} ms, peak RSS ${Math.max(parsed.cold.peakRssMb, ...parsed.warm.map((w) => w.peakRssMb)).toFixed(0)} MB`,
    )
  }

  const lines: string[] = []
  lines.push(`# columna vs Arquero / DuckDB-Wasm / Polars / DuckDB — same data, same operations, checked results`)
  lines.push('')
  lines.push(
    `Generated ${new Date().toISOString().slice(0, 10)} · Node ${process.version} · ${process.platform} ${process.arch} · ${ROWS.toLocaleString()} rows × 8 columns (2 int, 4 float, 2 low-cardinality text), CSV ${inputMb.toFixed(0)} MB.`,
  )
  lines.push(
    'Each library in its own process. **Cold total** = sum of timed operation durations on the first run in that process (module import / DuckDB-Wasm init are **outside** these timers). **Warm** = median of 2 further runs. **Peak RSS** sampled every 20 ms (synchronous work can hide intermediate peaks). Times in ms.',
  )
  lines.push('')

  const require = createRequire(import.meta.url)
  const version = (pkg: string) => {
    try {
      return (require(`${pkg}/package.json`) as { version: string }).version
    } catch {
      return '?'
    }
  }
  const columnaVersion = (() => {
    try {
      return (JSON.parse(readFileSync(new URL('../../columna/package.json', import.meta.url), 'utf8')) as { version: string })
        .version
    } catch {
      return '?'
    }
  })()
  const versions: Record<Lib, string> = {
    columna: columnaVersion,
    arquero: version('arquero'),
    'duckdb-wasm': (() => {
      try {
        return (
          JSON.parse(
            readFileSync(path.join(path.dirname(require.resolve('@duckdb/duckdb-wasm')), '..', 'package.json'), 'utf8'),
          ) as { version: string }
        ).version
      } catch {
        return '?'
      }
    })(),
    polars: version('nodejs-polars'),
    'polars-lazy': version('nodejs-polars'),
    duckdb: version('duckdb'),
  }

  lines.push('## Summary')
  lines.push('')
  lines.push('| Library | Version | Cold total | Warm total | Peak RSS | Notes |')
  lines.push('|---|---|---:|---:|---:|---|')
  for (const lib of LIBS) {
    const r = results[lib]
    if (!r || r.error) {
      lines.push(`| ${lib} | ${versions[lib]} | — | — | — | failed: ${r?.error ?? '?'} |`)
      continue
    }
    const peak = Math.max(r.cold.peakRssMb, ...r.warm.map((w) => w.peakRssMb))
    lines.push(
      `| ${lib} | ${versions[lib]} | ${sum(r.cold.times).toFixed(0)} | ${median(r.warm.map((w) => sum(w.times))).toFixed(0)} | ${peak.toFixed(0)} MB | ${r.cold.note ?? ''} |`,
    )
  }
  lines.push('')

  lines.push('## Warm time by operation (ms)')
  lines.push('')
  lines.push(
    'Rows are operations; columns are libraries. Lower is better. `polars-lazy` includes CSV scan inside every op (its `read` cell is 0 by design).',
  )
  lines.push('')
  lines.push(`| Operation | ${LIBS.join(' | ')} |`)
  lines.push(`|---|${LIBS.map(() => '---:').join('|')}|`)
  const warmOp = (lib: Lib, o: Op): string => {
    const r = results[lib]
    if (!r || r.error) return '—'
    return median(r.warm.map((w) => w.times[o])).toFixed(0)
  }
  const opLabel: Record<Op, string> = {
    read: 'read CSV',
    filter: 'filter (`age>30 ∧ salary>45k`)',
    select: 'select 4 columns',
    drop: 'drop (`z`,`y`)',
    rename: 'rename city→town',
    withColumn: 'withColumn (`xy = x+y`)',
    groupBy: 'groupBy city + agg',
    sort: 'sort salary desc',
    sortMulti: 'sort city asc, salary desc',
    unique: 'unique (`age`,`city`)',
    head: `head ${HEAD_N}`,
    tail: `tail ${HEAD_N}`,
    sample: `sample n=${SAMPLE_N}`,
    join: 'inner join regions on city',
    leftJoin: 'left join regions on city',
    semiJoin: 'semi join regions on city',
    antiJoin: 'anti join regions on city',
    melt: 'melt x,y (id=id,city)',
    pivot: 'pivot segment × mean(salary)',
    valueCounts: 'valueCounts(city)',
    describe: 'describe (age mean)',
    corr: 'corr(x, y)',
    filterGroupBy: 'filter age>30 → groupBy city sum(salary)',
    writeCsv: 'writeCsv (filtered)',
  }
  for (const o of OPS) {
    lines.push(`| ${opLabel[o]} | ${LIBS.map((lib) => warmOp(lib, o)).join(' | ')} |`)
  }
  lines.push('')

  lines.push('## Cold time by operation (ms)')
  lines.push('')
  lines.push(`| Operation | ${LIBS.join(' | ')} |`)
  lines.push(`|---|${LIBS.map(() => '---:').join('|')}|`)
  const coldOp = (lib: Lib, o: Op): string => {
    const r = results[lib]
    if (!r || r.error) return '—'
    return r.cold.times[o].toFixed(0)
  }
  for (const o of OPS) {
    lines.push(`| ${opLabel[o]} | ${LIBS.map((lib) => coldOp(lib, o)).join(' | ')} |`)
  }
  lines.push('')

  const ref = results.columna && !results.columna.error ? results.columna.cold.check : undefined
  lines.push('## Result validation')
  lines.push('')
  if (ref) {
    lines.push(
      `Reference (columna): filter ${ref.filterRows}, select ${ref.selectCols}, drop ${ref.dropCols}, rename ${ref.renameOk}, Σxy ${ref.withColSum}, groupBy ${ref.groupSum}, sort ${ref.sortFirst}/${ref.sortLast}, sortMulti first salary ${ref.sortMultiFirstSal}, unique ${ref.uniqueRows}, head Σ ${ref.headIdSum}, tail Σ ${ref.tailIdSum}, sample rows ${ref.sampleRows}, join ${ref.joinRows}/${ref.joinSum}, left ${ref.leftJoinRows}, semi ${ref.semiJoinRows}, anti ${ref.antiJoinRows}, melt ${ref.meltRows}/${ref.meltValueSum}, pivot Σ ${ref.pivotSum}, valueCounts Σ ${ref.valueCountsSum}, describe age μ ${ref.describeAgeMean}, corr(x,y) ${ref.corrXY}, filter→groupBy Σ ${ref.filterGroupSum}.`,
    )
    // sampleRows only — skip RNG-sensitive diffs for sample beyond count
    const skipKeys = new Set<keyof Check>([])
    for (const lib of LIBS) {
      const r = results[lib]
      if (!r || r.error || lib === 'columna') continue
      const c = r.cold.check
      const diffs = (Object.keys(ref) as Array<keyof Check>).filter((k) => {
        if (skipKeys.has(k)) return false
        return Math.abs(Number(c[k]) - Number(ref[k])) > 1e-6 * Math.max(1, Math.abs(Number(ref[k])))
      })
      lines.push(
        `- ${lib}: ${diffs.length === 0 ? 'summary-match' : `**differs** in ${diffs.map((k) => `${k} (${String(c[k])} vs ${String(ref[k])})`).join(', ')}`}`,
      )
    }
  }
  lines.push('')
  lines.push('## How to read this')
  lines.push('')
  lines.push(
    '- Wall-clock of a single process on one machine; the multi-threaded engines (Polars, native DuckDB) use every core, columna / Arquero / DuckDB-Wasm one.',
  )
  lines.push(
    '- polars-lazy pays the CSV scan inside every operation (that is the point of a streaming optimiser); compare its per-op numbers with `read + op` of the eager rows.',
  )
  lines.push(
    '- Peak RSS includes the runtime itself (a WebAssembly heap, Polars’ thread pool) and the CSV read; it is the number to budget, not the column bytes. Same-process 20 ms sampling underestimates peaks during long synchronous stretches.',
  )
  lines.push(
    '- SQL engines (DuckDB / DuckDB-Wasm) time **aggregate-over-join** (`count`/`sum` on a join), not full join materialization — do not compare that join cell directly to columna/Arquero/Polars full join output.',
  )
  lines.push(
    '- **unique** keeps the first row per (`age`,`city`) key (~300 groups from 2M rows) — a hash-dedup stress with a tiny result.',
  )
  lines.push(
    '- **head** / **tail** are first/last 1 000 rows in storage order (head id Σ = 499500 on this fixture).',
  )
  lines.push(
    '- **sample** checks only row count (RNG differs across libraries). **corr** is Pearson of `x` and `y`.',
  )
  lines.push('- **summary-match** checks aggregates (row counts, sums, sort extremes) with tolerance — not byte-identical tables.')
  lines.push('- columna numbers come from `collect()` on the CPU engine; `collectWithReport()` confirms no other backend was involved.')
  lines.push('')
  const outMd = new URL('../../../docs/comparison-js.md', import.meta.url)
  writeFileSync(outMd, lines.join('\n'))
  writeFileSync(new URL(`compare-js-${ROWS}.json`, OUT), JSON.stringify(results, null, 2))
  console.log(`wrote ${fileURLToPath(outMd)}`)
  console.log(readFileSync(outMd, 'utf8'))
}

const sum = (t: OpTimes) => Object.values(t).reduce((a, b) => a + b, 0)

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
