import { describe, expect, it } from 'vitest'
import { DataFrame } from '../src/dataframe.js'
import { inferDialectFromUrl, normalizeDialect, readSqlRows } from '../src/io/sql/index.js'
import type { SqlClient } from '../src/io/sql/types.js'
import { applyRowLimit, isSingleSelect } from '../src/io/sql/limit.js'

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

describe('nRows pushdown', () => {
  it('wraps a single SELECT / WITH in a derived table with LIMIT for postgres, mysql, sqlite, clickhouse', () => {
    for (const d of ['postgres', 'mysql', 'sqlite', 'clickhouse'] as const) {
      const r = applyRowLimit('SELECT a, b FROM t WHERE x = $1 ORDER BY a;  -- trailing\n', 10, d)
      expect(r.pushed).toBe(true)
      expect(r.sql).toBe('SELECT * FROM (\nSELECT a, b FROM t WHERE x = $1 ORDER BY a\n) AS __columna_q LIMIT 10')
    }
    const cte = applyRowLimit('WITH q AS (SELECT 1 AS a) SELECT * FROM q', 5, 'postgres')
    expect(cte.pushed).toBe(true)
    expect(cte.sql).toMatch(/^SELECT \* FROM \(\nWITH q AS/)
  })

  it('SQL Server gets SET ROWCOUNT (derived tables refuse ORDER BY / CTEs there)', () => {
    const r = applyRowLimit('SELECT TOP 100 PERCENT * FROM t ORDER BY id', 7, 'mssql')
    expect(r.pushed).toBe(true)
    expect(r.sql).toBe('SET ROWCOUNT 7;\nSELECT TOP 100 PERCENT * FROM t ORDER BY id;\nSET ROWCOUNT 0;')
  })

  it('leaves anything that is not one SELECT alone: several statements, EXEC, unknown dialect, bad nRows', () => {
    expect(applyRowLimit('SELECT 1; SELECT 2', 1, 'postgres').pushed).toBe(false)
    expect(applyRowLimit('EXEC dbo.report @year = 2024', 1, 'mssql').pushed).toBe(false)
    expect(applyRowLimit('INSERT INTO t SELECT * FROM s', 1, 'postgres').pushed).toBe(false)
    expect(applyRowLimit('SELECT 1', 1, undefined).pushed).toBe(false)
    expect(applyRowLimit('SELECT 1', 1.5, 'postgres').pushed).toBe(false)
    // a `;` inside a string literal or a comment is not a statement separator
    expect(isSingleSelect("SELECT ';' AS s, 'it''s' AS q FROM t -- ; not here\n/* ; nor here */")).toBe(true)
    expect(isSingleSelect('SELECT [a;b] FROM t')).toBe(true)
    expect(isSingleSelect('/* lead */ (SELECT 1)')).toBe(true)
    expect(isSingleSelect('SELECT 1; DROP TABLE t')).toBe(false)
  })

  it('readSqlRows pushes the limit into the query when the dialect is known and slices as a safety net', async () => {
    const seen: string[] = []
    const client: SqlClient = {
      async query(sql) {
        seen.push(sql)
        return Array.from({ length: 50 }, (_, i) => ({ i }))
      },
    }
    const rows = await readSqlRows('SELECT i FROM big', client, { nRows: 3, dialect: 'postgres' })
    expect(rows).toHaveLength(3)
    expect(seen[0]).toContain('AS __columna_q LIMIT 3')
    // duck-typed client without a dialect: untouched SQL, sliced result
    await readSqlRows('SELECT i FROM big', client, { nRows: 3 })
    expect(seen[1]).toBe('SELECT i FROM big')
    // explicit opt-out
    await readSqlRows('SELECT i FROM big', client, { nRows: 3, dialect: 'postgres', pushdown: false })
    expect(seen[2]).toBe('SELECT i FROM big')
  })
})

describe('nRows pushdown against a real SQLite engine (node:sqlite, Node ≥ 22.5)', () => {
  it('the rewritten statement executes and returns the first n rows in query order', async (ctx) => {
    let sqlite: { DatabaseSync: new (file: string) => { exec(s: string): void; prepare(s: string): { all(...p: unknown[]): Record<string, unknown>[] } } }
    // process.getBuiltinModule bypasses the test bundler, which does not know node:sqlite
    const get = (process as { getBuiltinModule?: (id: string) => unknown }).getBuiltinModule
    try {
      sqlite = get?.('node:sqlite') as typeof sqlite
    } catch {
      sqlite = undefined as never
    }
    if (!sqlite) {
      ctx.skip()
      return
    }
    const db = new sqlite.DatabaseSync(':memory:')
    db.exec("CREATE TABLE t(a INT, b TEXT); INSERT INTO t VALUES (3,'x'),(1,'y'),(2,'z'),(5,'w')")
    const client: SqlClient = { async query(sql, params) { return db.prepare(sql).all(...((params as unknown[]) ?? [])) } }
    const rows = await readSqlRows('SELECT a, b FROM t ORDER BY a DESC;', client, { nRows: 2, dialect: 'sqlite' })
    expect(rows).toEqual([{ a: 5, b: 'w' }, { a: 3, b: 'x' }])
    const cte = await readSqlRows("WITH big AS (SELECT * FROM t WHERE a > ?) SELECT b FROM big ORDER BY b -- tail", client, { nRows: 1, dialect: 'sqlite', params: [1] })
    expect(cte).toEqual([{ b: 'w' }])
  })
})
