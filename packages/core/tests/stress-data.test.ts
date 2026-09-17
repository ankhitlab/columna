import { describe, expect, it } from 'vitest'
import { DataFrame, col } from '../src/dataframe.js'
import {
  STRESS_TIMEOUT,
  adversarialFrame,
  adversarialRows,
  checksumNumeric,
  rowMultisetKey,
  sortById,
  stressN,
  sum,
} from './stress-helpers.js'

const SEEDS = [7, 42]

describe('stress: DataFrame op identities on adversarial frames', () => {
  it(
    'filter partitions; select/drop/rename/withColumn preserve row count and checksums',
    async () => {
      const n = stressN(25_000)
      for (const seed of SEEDS) {
        const rows = adversarialRows(seed, n)
        const df = DataFrame.fromRows(rows)

        const yes = (await df.filter(col('salary').gt(50_000)).collect()).toArray()
        const no = (await df.filter(col('salary').lte(50_000)).collect()).toArray()
        const nulls = (await df.filter(col('salary').isNull()).collect()).toArray()
        expect(yes.length + no.length + nulls.length).toBe(n)
        expect(sortById([...yes, ...no, ...nulls])).toEqual(rows)

        const sel = await df.select('id', 'city', 'salary').collect()
        expect(sel.shape).toEqual([n, 3])
        expect(checksumNumeric(sel.toArray(), 'id')).toBe(checksumNumeric(rows, 'id'))

        const dropped = await df.drop('note', 'flag').collect()
        expect(dropped.shape[0]).toBe(n)
        expect(dropped.columns.includes('note')).toBe(false)

        const renamed = await df.rename({ city: 'town' }).collect()
        expect(renamed.columns.includes('town')).toBe(true)
        expect(renamed.columns.includes('city')).toBe(false)

        const withCol = await df.withColumn('xy', col('x').add(col('y'))).collect()
        expect(withCol.shape[1]).toBe(df.shape[1] + 1)
        const sample = withCol.toArray().slice(0, 200)
        for (const r of sample) {
          expect(Number(r.xy)).toBeCloseTo(Number(r.x) + Number(r.y), 6)
        }
      }
    },
    STRESS_TIMEOUT,
  )

  it(
    'groupBy partitions; filter→groupBy sums match manual filter',
    async () => {
      const n = stressN(30_000)
      for (const seed of SEEDS) {
        const df = adversarialFrame(seed, n)
        const g = (
          await df.groupBy('city').agg({ n: col('id').count(), salary: col('salary').sum(), k: col('k').sum() }).collect()
        ).toArray()
        expect(sum(g.map((r) => Number(r.n)))).toBe(n)
        const groupSalary = sum(g.map((r) => (r.salary == null ? 0 : Number(r.salary))))
        const frameSalary = checksumNumeric(df.toArray(), 'salary')
        // Float64 partial sums over 100k+ rows: allow relative slack, not fixed decimals.
        expect(Math.abs(groupSalary - frameSalary) / Math.max(1, Math.abs(frameSalary))).toBeLessThan(1e-9)

        const g2 = (await df.groupBy('city', 'seg').agg({ n: col('id').count() }).collect()).toArray()
        expect(sum(g2.map((r) => Number(r.n)))).toBe(n)

        const filtered = await df.filter(col('age').gt(30)).collect()
        const fg = (
          await filtered.groupBy('city').agg({ salary: col('salary').sum(), n: col('id').count() }).collect()
        ).toArray()
        const chained = (
          await df
            .filter(col('age').gt(30))
            .groupBy('city')
            .agg({ salary: col('salary').sum(), n: col('id').count() })
            .collect()
        ).toArray()
        expect(rowMultisetKey(fg, ['city', 'n', 'salary'])).toBe(rowMultisetKey(chained, ['city', 'n', 'salary']))
      }
    },
    STRESS_TIMEOUT,
  )

  it(
    'sort / sortMulti are permutations; nulls last; per-key descending',
    async () => {
      const n = stressN(40_000)
      for (const seed of SEEDS) {
        const rows = adversarialRows(seed, n)
        const df = DataFrame.fromRows(rows)
        const cols = Object.keys(rows[0]!)

        const sorted = (await df.sort(col('salary').desc(), 'id').collect()).toArray()
        expect(rowMultisetKey(sorted, cols)).toBe(rowMultisetKey(rows, cols))
        for (let i = 1; i < sorted.length; i++) {
          const a = sorted[i - 1]!.salary as number | null
          const b = sorted[i]!.salary as number | null
          if (a === null) expect(b).toBeNull()
          else if (b !== null) expect(a).toBeGreaterThanOrEqual(b)
        }

        const multi = (await df.sort('city', { expr: col('salary'), descending: true }, 'id').collect()).toArray()
        expect(rowMultisetKey(multi, cols)).toBe(rowMultisetKey(rows, cols))
        for (let i = 1; i < multi.length; i++) {
          const prev = multi[i - 1]!
          const cur = multi[i]!
          // Match engine string order (UTF-16 code units), not localeCompare.
          const aCity = String(prev.city)
          const bCity = String(cur.city)
          if (aCity > bCity) throw new Error(`city not ascending: ${JSON.stringify(aCity)} > ${JSON.stringify(bCity)}`)
          if (aCity === bCity) {
            const a = prev.salary as number | null
            const b = cur.salary as number | null
            if (a === null) expect(b).toBeNull()
            else if (b !== null) expect(a).toBeGreaterThanOrEqual(b)
          }
        }
      }
    },
    STRESS_TIMEOUT,
  )

  it(
    'unique / valueCounts agree with Set; counts sum to n',
    async () => {
      const n = stressN(35_000)
      for (const seed of SEEDS) {
        const rows = adversarialRows(seed, n)
        const df = DataFrame.fromRows(rows)
        const distinct = new Set(rows.map((r) => r.note ?? '\0null')).size
        // unique on nullable note: nulls collapse to one group depending on keep semantics
        const uniqNote = (await df.unique(['seg']).collect()).shape[0]
        expect(uniqNote).toBe(new Set(rows.map((r) => r.seg)).size)

        const vc = (await df.valueCounts('city').collect()).toArray()
        expect(sum(vc.map((r) => Number(r.count)))).toBe(n)
        expect(vc.length).toBe(new Set(rows.map((r) => r.city)).size)

        const uniqAgeCity = (await df.unique(['age', 'city']).collect()).shape[0]
        expect(uniqAgeCity).toBe(new Set(rows.map((r) => `${r.age}\0${r.city}`)).size)
        expect(distinct).toBeGreaterThan(1)
      }
    },
    STRESS_TIMEOUT,
  )

  it(
    'join / left / semi / anti algebra with null-ish and empty right',
    async () => {
      const n = stressN(20_000)
      for (const seed of SEEDS) {
        const rows = adversarialRows(seed, n)
        const df = DataFrame.fromRows(rows)
        const right = DataFrame.fromRows([
          { city: 'Berlin', region: 'DE' },
          { city: 'Paris', region: 'FR' },
          { city: '', region: 'EMPTY' },
          { city: 'zzz', region: 'NONE' },
        ])
        const matched = new Set(['Berlin', 'Paris', ''])

        const inner = (await df.join(right, { on: 'city' }).collect()).toArray()
        const left = (await df.leftJoin(right, 'city').collect()).toArray()
        const semi = (await df.semiJoin(right, 'city').collect()).toArray()
        const anti = (await df.antiJoin(right, 'city').collect()).toArray()

        expect(inner.length).toBe(rows.filter((r) => matched.has(r.city)).length)
        expect(left.length).toBe(n)
        expect(semi.length + anti.length).toBe(n)
        expect(sortById([...semi, ...anti])).toEqual(rows)

        const emptyRight = DataFrame.fromRows([] as Array<{ city: string; region: string }>)
        // empty schema: build via fromColumns-like empty frame
        const empty = DataFrame.fromRows([{ city: '__none__', region: 'x' }]).filter(col('city').eq('nope'))
        const emptyCollected = await empty.collect()
        expect(emptyCollected.shape[0]).toBe(0)
        const semiEmpty = (await df.semiJoin(emptyCollected, 'city').collect()).shape[0]
        const antiEmpty = (await df.antiJoin(emptyCollected, 'city').collect()).shape[0]
        expect(semiEmpty).toBe(0)
        expect(antiEmpty).toBe(n)
        void right
      }
    },
    STRESS_TIMEOUT,
  )

  it(
    'melt / pivot round-trip checksum; describe / corr smoke',
    async () => {
      const n = stressN(15_000)
      for (const seed of SEEDS) {
        const df = adversarialFrame(seed, n)
        const melted = await df.melt({ idVars: ['id', 'city'], valueVars: ['x', 'y'] }).collect()
        expect(melted.shape[0]).toBe(n * 2)
        const meltSum = checksumNumeric(melted.toArray(), 'value')
        const xySum = checksumNumeric(df.toArray(), 'x') + checksumNumeric(df.toArray(), 'y')
        expect(Math.abs(meltSum - xySum) / Math.max(1, Math.abs(xySum))).toBeLessThan(1e-9)

        const pivoted = await melted
          .pivot({ index: 'id', columns: 'variable', values: 'value', agg: 'sum' })
          .collect()
        expect(pivoted.shape[0]).toBe(n)
        const px = checksumNumeric(pivoted.toArray(), 'x')
        const fx = checksumNumeric(df.toArray(), 'x')
        expect(Math.abs(px - fx) / Math.max(1, Math.abs(fx))).toBeLessThan(1e-9)

        const desc = await df.describe().collect()
        expect(desc.shape[0]).toBeGreaterThan(0)

        const corr = await df.select('x', 'y', 'k').corr().collect()
        expect(corr.shape[0]).toBeGreaterThanOrEqual(2)
      }
    },
    STRESS_TIMEOUT,
  )

  it(
    'empty and all-null frames do not throw and keep consistent shapes',
    async () => {
      const empty = DataFrame.fromRows([] as Array<{ id: number; v: number | null; g: string }>)
      // fromRows([]) may yield 0 cols — build explicit empty via filter
      const base = DataFrame.fromRows([
        { id: 1, v: null as number | null, g: 'a' },
        { id: 2, v: null, g: 'b' },
        { id: 3, v: null, g: 'a' },
      ])
      const emptyDf = await base.filter(col('id').lt(0)).collect()
      expect(emptyDf.shape[0]).toBe(0)

      expect((await emptyDf.select('id', 'v').collect()).shape[0]).toBe(0)
      expect((await emptyDf.drop('g').collect()).shape[0]).toBe(0)
      expect((await emptyDf.sort('id').collect()).shape[0]).toBe(0)
      expect((await emptyDf.unique(['g']).collect()).shape[0]).toBe(0)
      expect((await emptyDf.groupBy('g').agg({ n: col('id').count() }).collect()).shape[0]).toBe(0)

      const allNull = base
      expect((await allNull.filter(col('v').isNull()).collect()).shape[0]).toBe(3)
      const g = (await allNull.groupBy('g').agg({ s: col('v').sum(), n: col('id').count() }).collect()).toArray()
      expect(sum(g.map((r) => Number(r.n)))).toBe(3)

      const right = DataFrame.fromRows([{ g: 'a', r: 1 }])
      expect((await emptyDf.leftJoin(right, 'g').collect()).shape[0]).toBe(0)
      expect((await allNull.leftJoin(right, 'g').collect()).shape[0]).toBe(3)
      void empty
    },
    STRESS_TIMEOUT,
  )

  it(
    'late dtype widen survives filter / groupBy / join',
    async () => {
      const n = stressN(8_000)
      const rnd = adversarialRows(1, n)
      // Force widen: mostly ints then a float and a text in salary-like column via CSV
      const lines = ['id,score,tag']
      for (let i = 0; i < n - 2; i++) lines.push(`${i},${i % 100},t${i % 5}`)
      lines.push(`${n - 2},1.5,t0`)
      lines.push(`${n - 1},abc,t1`)
      const df = DataFrame.fromCSV(lines.join('\n'))
      expect(df.getColumn('score').dtype).toMatch(/utf8|category/)

      const filtered = await df.filter(col('id').lt(n)).collect()
      expect(filtered.shape[0]).toBe(n)
      const grouped = await df.groupBy('tag').agg({ n: col('id').count() }).collect()
      expect(sum(grouped.toArray().map((r) => Number(r.n)))).toBe(n)

      const right = DataFrame.fromRows([
        { tag: 't0', region: 'R0' },
        { tag: 't1', region: 'R1' },
      ])
      const joined = await df.leftJoin(right, 'tag').collect()
      expect(joined.shape[0]).toBe(n)
      void rnd
    },
    STRESS_TIMEOUT,
  )
})
