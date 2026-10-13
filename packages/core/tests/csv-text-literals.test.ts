import { describe, expect, it } from 'vitest'
import { DataFrame } from '../src/dataframe.js'

/** Matrix: (A) infer, (B) dtypes utf8 — text-like CSV cells must not round-trip through Number. */
describe('CSV text literals: leading zeros, phones, forced utf8', () => {
  function both(csv: string, col = 'c') {
    const infer = DataFrame.fromCSV(csv)
    const forced = DataFrame.fromCSV(csv, { dtypes: { [col]: 'utf8' } })
    const rows = DataFrame.fromCSVRows(csv, { dtypes: { [col]: 'utf8' } })
    return { infer, forced, rows }
  }

  it('1–2: leading zero / quoted leading zero stay literal with utf8; infer keeps text', () => {
    for (const body of ['0070', '"0070"', '0370']) {
      const csv = `c\n${body}\n`
      const { infer, forced, rows } = both(csv)
      const lit = body.replaceAll('"', '')
      expect(forced.toArray()[0]!.c).toBe(lit)
      expect(forced.dtypes.c).toBe('utf8')
      expect(rows.toArray()[0]!.c).toBe(lit)
      expect(infer.toArray()[0]!.c).toBe(lit)
      expect(['utf8', 'category']).toContain(infer.dtypes.c)
    }
  })

  it('3: mix of 0070 and 70 → text column; utf8 keeps both literals', () => {
    const csv = 'c\n0070\n70\n'
    const { infer, forced } = both(csv)
    expect(infer.toArray().map((r) => r.c)).toEqual(['0070', '70'])
    expect(forced.toArray().map((r) => r.c)).toEqual(['0070', '70'])
  })

  it('4–6: lone 0 / plain int / float still numeric on infer; utf8 keeps file literal', () => {
    expect(DataFrame.fromCSV('c\n0\n').toArray()[0]!.c).toBe(0)
    expect(DataFrame.fromCSV('c\n0\n').dtypes.c).toBe('i32')
    expect(DataFrame.fromCSV('c\n0\n', { dtypes: { c: 'utf8' } }).toArray()[0]!.c).toBe('0')

    expect(DataFrame.fromCSV('c\n70\n').toArray()[0]!.c).toBe(70)
    expect(DataFrame.fromCSV('c\n-42\n').toArray()[0]!.c).toBe(-42)
    expect(DataFrame.fromCSV('c\n70\n', { dtypes: { c: 'utf8' } }).toArray()[0]!.c).toBe('70')

    expect(DataFrame.fromCSV('c\n12.5\n').toArray()[0]!.c).toBe(12.5)
    expect(DataFrame.fromCSV('c\n12.5\n', { dtypes: { c: 'utf8' } }).toArray()[0]!.c).toBe('12.5')
  })

  it('7: decimal comma with utf8 keeps literal; infer parses number', () => {
    // comma is the field separator by default — use ';' so ',' can be the decimal mark
    const csv = 'c\n12,5\n'
    expect(DataFrame.fromCSV(csv, { separator: ';', decimal: ',' }).toArray()[0]!.c).toBe(12.5)
    expect(DataFrame.fromCSV(csv, { separator: ';', decimal: ',', dtypes: { c: 'utf8' } }).toArray()[0]!.c).toBe('12,5')
  })

  it('8: scientific notation infer numeric; utf8 keeps literal', () => {
    const csv = 'c\n4.85E+15\n'
    expect(DataFrame.fromCSV(csv).dtypes.c).toBe('f64')
    expect(DataFrame.fromCSV(csv, { dtypes: { c: 'utf8' } }).toArray()[0]!.c).toBe('4.85E+15')
  })

  it('9: unsafe integer ID stays exact string; safe long digit string may be i32/f64 on infer', () => {
    const unsafe = '9007199254740993' // > Number.MAX_SAFE_INTEGER
    expect(DataFrame.fromCSV(`c\n${unsafe}\n`).toArray()[0]!.c).toBe(unsafe)
    expect(DataFrame.fromCSV(`c\n${unsafe}\n`, { dtypes: { c: 'utf8' } }).toArray()[0]!.c).toBe(unsafe)

    const safeEnp = '4856130895000070' // within 2^53 — infer may keep as number
    expect(DataFrame.fromCSV(`c\n${safeEnp}\n`, { dtypes: { c: 'utf8' } }).toArray()[0]!.c).toBe(safeEnp)
  })

  it('10–12: spaced ID and dotted/slashed dates stay text', () => {
    for (const lit of ['4856 1308 9500 0070', '05.02.2026', '05/02/2026']) {
      const csv = `c\n${lit}\n`
      const { infer, forced } = both(csv)
      expect(infer.toArray()[0]!.c).toBe(lit)
      expect(forced.toArray()[0]!.c).toBe(lit)
    }
  })

  it('13: phone with leading + keeps + on infer and utf8', () => {
    const lit = '+79001234567'
    const csv = `c\n${lit}\n`
    const { infer, forced, rows } = both(csv)
    expect(infer.toArray()[0]!.c).toBe(lit)
    expect(forced.toArray()[0]!.c).toBe(lit)
    expect(rows.toArray()[0]!.c).toBe(lit)
  })

  it('14: bool infer; utf8 keeps file casing', () => {
    expect(DataFrame.fromCSV('c\ntrue\n').toArray()[0]!.c).toBe(true)
    expect(DataFrame.fromCSV('c\nFALSE\n', { dtypes: { c: 'utf8' } }).toArray()[0]!.c).toBe('FALSE')
    expect(DataFrame.fromCSV('c\ntrue\n', { dtypes: { c: 'utf8' } }).toArray()[0]!.c).toBe('true')
  })

  it('15: null tokens still null under utf8; quoted empty stays empty string', () => {
    expect(DataFrame.fromCSV('c\nNA\n', { dtypes: { c: 'utf8' } }).toArray()[0]!.c).toBeNull()
    expect(DataFrame.fromCSV('c\nnull\n', { dtypes: { c: 'utf8' } }).toArray()[0]!.c).toBeNull()
    // bare empty data line is skipBlankLines; empty field beside a neighbour is null
    expect(DataFrame.fromCSV('a,c\nx,\n', { dtypes: { c: 'utf8' } }).toArray()[0]!.c).toBeNull()
    expect(DataFrame.fromCSV('c\n""\n', { dtypes: { c: 'utf8' } }).toArray()[0]!.c).toBe('')
  })

  it('16: percent / hex-like codes stay text', () => {
    for (const lit of ['12%', '00a1']) {
      const csv = `c\n${lit}\n`
      expect(DataFrame.fromCSV(csv).toArray()[0]!.c).toBe(lit)
      expect(DataFrame.fromCSV(csv, { dtypes: { c: 'utf8' } }).toArray()[0]!.c).toBe(lit)
    }
  })

  it('forced category also keeps leading-zero literal', () => {
    const df = DataFrame.fromCSV('c\n0070\n', { dtypes: { c: 'category' } })
    expect(df.toArray()[0]!.c).toBe('0070')
    expect(df.dtypes.c).toBe('category')
  })

  it('dtypes.utf8 round-trip via toCsv / toArray', () => {
    const df = DataFrame.fromCSV('c\n0070\n+79001234567\n', { dtypes: { c: 'utf8' } })
    expect(df.toArray().map((r) => r.c)).toEqual(['0070', '+79001234567'])
    const back = DataFrame.fromCSV(df.toCsv(), { dtypes: { c: 'utf8' } })
    expect(back.toArray().map((r) => r.c)).toEqual(['0070', '+79001234567'])
  })

  it('columnar and row paths agree on cell values for the acceptance matrix', () => {
    const cases = [
      { csv: 'c\n0070\n', opts: { dtypes: { c: 'utf8' as const } } },
      { csv: 'c\n"0070"\n', opts: { dtypes: { c: 'utf8' as const } } },
      { csv: 'c\n+79001234567\n', opts: { dtypes: { c: 'utf8' as const } } },
      { csv: 'c\n0070\n', opts: {} },
      { csv: 'c\n70\n', opts: {} },
      { csv: 'c\n05.02.2026\n', opts: {} },
    ]
    for (const { csv, opts } of cases) {
      const a = DataFrame.fromCSV(csv, opts)
      const b = DataFrame.fromCSVRows(csv, opts)
      // Values must match; forced utf8 may be category on the row path (fromRows dict-encode).
      expect(a.toArray(), csv).toEqual(b.toArray())
    }
  })
})
