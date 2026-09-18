import type { SqlDialect } from './types.js'

/**
 * Turn `nRows` into a database-side limit when it is safe to do so.
 *
 * Only a single `SELECT` / `WITH` statement is rewritten; anything else (several statements, DDL, EXEC,
 * comments-only text) is returned unchanged and the caller slices the buffered result instead. The rewrite
 * is a derived table — `SELECT * FROM (<sql>) AS __columna_q LIMIT n` — which is valid for PostgreSQL,
 * MySQL / MariaDB, SQLite and ClickHouse, including a leading CTE. SQL Server refuses `ORDER BY` inside a
 * derived table and a CTE inside one, so there the statement is prefixed with `SET ROWCOUNT n;`, which caps
 * the rows the following statement returns and is reset afterwards.
 */
export function applyRowLimit(sql: string, nRows: number, dialect: SqlDialect | undefined): { sql: string; pushed: boolean } {
  if (!dialect || !Number.isInteger(nRows) || nRows < 0) return { sql, pushed: false }
  const body = stripTrailingTerminators(sql)
  if (!isSingleSelect(body)) return { sql, pushed: false }
  switch (dialect) {
    case 'postgres':
    case 'mysql':
    case 'sqlite':
    case 'clickhouse':
      return { sql: `SELECT * FROM (\n${body}\n) AS __columna_q LIMIT ${nRows}`, pushed: true }
    case 'mssql':
      return { sql: `SET ROWCOUNT ${nRows};\n${body};\nSET ROWCOUNT 0;`, pushed: true }
    default:
      return { sql, pushed: false }
  }
}

/** Trailing whitespace, `;` and trailing comments are removed so the text can be embedded. */
function stripTrailingTerminators(sql: string): string {
  let s = sql
  for (;;) {
    const t = s.replace(/\s+$/, '')
    const noSemi = t.replace(/;+$/, '')
    const noLineComment = noSemi.replace(/--[^\n]*$/, '')
    const noBlockComment = noLineComment.replace(/\/\*[\s\S]*?\*\/\s*$/, '')
    if (noBlockComment === s) return s
    s = noBlockComment
  }
}

/** One statement (no `;` outside strings / comments) whose first keyword is SELECT or WITH. */
export function isSingleSelect(sql: string): boolean {
  let i = 0
  const n = sql.length
  let firstWord: string | null = null
  while (i < n) {
    const c = sql[i]!
    if (c === '-' && sql[i + 1] === '-') {
      const nl = sql.indexOf('\n', i)
      i = nl < 0 ? n : nl + 1
      continue
    }
    if (c === '/' && sql[i + 1] === '*') {
      const end = sql.indexOf('*/', i + 2)
      i = end < 0 ? n : end + 2
      continue
    }
    if (c === "'" || c === '"' || c === '`') {
      i = skipQuoted(sql, i, c)
      continue
    }
    if (c === '[') {
      const end = sql.indexOf(']', i + 1)
      i = end < 0 ? n : end + 1
      continue
    }
    if (c === ';') return false
    if (firstWord === null && /[A-Za-z(]/.test(c)) {
      if (c === '(') {
        // parenthesised select: skip leading parens
        i++
        continue
      }
      let j = i
      while (j < n && /[A-Za-z_]/.test(sql[j]!)) j++
      firstWord = sql.slice(i, j).toUpperCase()
      if (firstWord !== 'SELECT' && firstWord !== 'WITH') return false
      i = j
      continue
    }
    i++
  }
  return firstWord === 'SELECT' || firstWord === 'WITH'
}

function skipQuoted(sql: string, start: number, quote: string): number {
  let i = start + 1
  while (i < sql.length) {
    if (sql[i] === quote) {
      if (sql[i + 1] === quote) {
        i += 2
        continue
      }
      return i + 1
    }
    if (quote === "'" && sql[i] === '\\' && sql[i + 1] === "'") {
      i += 2
      continue
    }
    i++
  }
  return sql.length
}
