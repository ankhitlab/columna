import { describe, expect, it } from 'vitest'
import { DataFrame, col, lit } from '@columna/core'

describe('robustness: malformed input, prototype keys, large boxed groups', () => {
  const df = DataFrame.fromColumns({ x: new Float64Array([1, 2, 3, 4, 5]), s: ['a', 'b', 'c', 'd', 'e'], g: ['u', 'u', 'v', 'v', 'v'] })

  it('boxed min / max do not spread the group values (stack overflow past ~120k rows)', async () => {
    const n = 300_000
    const big = DataFrame.fromColumns({ g: new Array<string>(n).fill('a'), x: Float64Array.from({ length: n }, (_, i) => (i * 7919) % n) })
    const out = (await big.groupBy('g').agg({ lo: col('x').add(lit(0)).min(), hi: col('x').add(lit(0)).max() }).collect()).toArray()
    expect(out).toEqual([{ g: 'a', lo: 0, hi: n - 1 }])
  })

  it('take rejects out-of-range and fractional indices instead of reading past the buffers', async () => {
    await expect(df.take([0, 5]).collect()).rejects.toThrow(RangeError)
    await expect(df.take([-1]).collect()).rejects.toThrow(RangeError)
    await expect(df.take([1.5]).collect()).rejects.toThrow(RangeError)
    expect((await df.take([4, 0]).collect()).toArray().map((r) => r.s)).toEqual(['e', 'a'])
  })

  it('rename: duplicate targets throw, unknown sources throw, prototype names are not looked up on Object.prototype', async () => {
    await expect(df.rename({ x: 's' }).collect()).rejects.toThrow(/duplicate/)
    await expect(df.rename({ zz: 'y' }).collect()).rejects.toThrow(/Unknown column/)
    const t = DataFrame.fromColumns({ constructor: [1, 2], toString: [3, 4] })
    const out = await t.rename({ constructor: 'c' }).collect()
    expect(out.columns).toEqual(['c', 'toString'])
    expect(out.toArray()).toEqual([{ c: 1, toString: 3 }, { c: 2, toString: 4 }])
  })

  it('a column named __proto__ survives toArray / fromCSV / fromJSON without touching prototypes', async () => {
    const rows = (await DataFrame.fromCSV('__proto__,b\n1,2\n').collect()).toArray()
    expect(Object.keys(rows[0]!)).toEqual(['__proto__', 'b'])
    expect(Object.getPrototypeOf(rows[0])).toBe(Object.prototype)
    expect(rows[0]!['__proto__']).toBe(1)
    const fromJson = DataFrame.fromJSON('[{"__proto__":{"polluted":1},"a":1}]').toArray()
    expect(({} as { polluted?: unknown }).polluted).toBeUndefined()
    expect(Object.getPrototypeOf(fromJson[0])).toBe(Object.prototype)
  })

  it('CSV: quoted newlines stay inside the field, duplicate headers are mangled, letter delimiters are escaped', async () => {
    const multi = DataFrame.fromCSV('a,b\n"x\ny",2\n"p\r\nq",3\n').toArray()
    expect(multi).toEqual([{ a: 'x\ny', b: 2 }, { a: 'p\r\nq', b: 3 }])
    const dup = DataFrame.fromCSV('a,a,b,a\n1,2,3,4\n').toArray()
    expect(dup).toEqual([{ a: 1, 'a.1': 2, b: 3, 'a.2': 4 }])
    const letter = DataFrame.fromCSV('1d2\n7d 8\n', { delimiter: 'd', hasHeader: false, skipInitialSpace: true }).toArray()
    expect(letter).toEqual([{ column_0: 1, column_1: 2 }, { column_0: 7, column_1: 8 }])
  })

  it('str.replace treats the replacement literally ($& is not expanded)', async () => {
    const out = (await DataFrame.fromColumns({ s: ['a-b'] }).select(col('s').str.replace('-', '$&$1').alias('r')).collect()).toArray()
    expect(out[0]!.r).toBe('a$&$1b')
  })

  it('rolling / cast validate their arguments', () => {
    expect(() => df.rolling('r', 'x', 0)).toThrow(RangeError)
    expect(() => df.rolling('r', 'x', 2.5)).toThrow(RangeError)
    expect(() => col('x').cast('nope' as never)).toThrow(RangeError)
  })

  it('toMarkdown escapes pipes and newlines inside cells', () => {
    const md = DataFrame.fromRows([{ 'a|b': 'x|y\nz' }]).toMarkdown()
    expect(md.split('\n')[0]).toBe('| a\\|b |')
    expect(md.split('\n')[2]).toBe('| x\\|y z |')
  })
})
