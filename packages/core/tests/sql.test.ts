import { describe, expect, it } from 'vitest'
import { DataFrame } from '../src/dataframe.js'
import { inferDialectFromUrl, normalizeDialect, readSqlRows } from '../src/io/sql/index.js'
import type { SqlClient } from '../src/io/sql/types.js'

describe('SQL dialect helpers', () => {
  it('normalizes aliases', () => {
    expect(normalizeDialect('postgresql')).toBe('postgres')
    expect(normalizeDialect('sqlserver')).toBe('mssql')
    expect(normalizeDialect('mariadb')).toBe('mysql')
    expect(normalizeDialect('ch')).toBe('clickhouse')
  })

  it('infers dialect from URLs and sqlite paths', () => {
    expect(inferDialectFromUrl('postgres://u:p@localhost:5432/db')).toBe('postgres')
    expect(inferDialectFromUrl('mssql://localhost/db')).toBe('mssql')
    expect(inferDialectFromUrl('clickhouse://localhost:8123/default')).toBe('clickhouse')
    expect(inferDialectFromUrl('mysql://root@127.0.0.1/app')).toBe('mysql')
    expect(inferDialectFromUrl('sqlite://:memory:')).toBe('sqlite')
    expect(inferDialectFromUrl('./data/app.sqlite')).toBe('sqlite')
    expect(inferDialectFromUrl(':memory:')).toBe('sqlite')
  })
})

describe('DataFrame.readSql', () => {
  it('reads via duck-typed SqlClient', async () => {
    const client: SqlClient = {
      async query(sql, params) {
        expect(sql).toContain('SELECT')
        expect(params).toEqual([1])
        return [
          { id: 1, name: 'Ada' },
          { id: 2n, name: 'Bob' },
        ]
      },
    }
    const df = await DataFrame.readSql('SELECT * FROM t WHERE id >= $1', client, { params: [1] })
    expect(df.shape).toEqual([2, 2])
    expect(df.toArray()).toEqual([
      { id: 1, name: 'Ada' },
      { id: 2, name: 'Bob' },
    ])
  })

  it('honours nRows', async () => {
    const client: SqlClient = {
      async query() {
        return [{ a: 1 }, { a: 2 }, { a: 3 }]
      },
    }
    const rows = await readSqlRows('select 1', client, { nRows: 2 })
    expect(rows).toHaveLength(2)
  })

  it('readDatabase is an alias', async () => {
    const client: SqlClient = {
      async query() {
        return [{ ok: true }]
      },
    }
    const df = await DataFrame.readDatabase('select 1', client)
    expect(df.toArray()[0]).toEqual({ ok: true })
  })

  it('errors clearly when optional driver is missing', async () => {
    let hasPg = false
    try {
      await import('pg')
      hasPg = true
    } catch {
      // optional peer not installed
    }
    if (hasPg) return
    await expect(
      DataFrame.readSql('select 1', { dialect: 'postgres', host: 'localhost', database: 'x' }),
    ).rejects.toThrow(/pnpm add pg/)
  })
})
