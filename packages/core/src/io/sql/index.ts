import {
  connectClickhouse,
  connectMssql,
  connectMysql,
  connectPostgres,
  connectSqlite,
} from './adapters.js'
import type {
  ReadSqlOptions,
  SqlClient,
  SqlConnection,
  SqlConnectionConfig,
  SqlDialect,
} from './types.js'
import { setRowField } from '@columna/arrow'
import { resolveDialect } from './util.js'
import { applyRowLimit } from './limit.js'
export { applyRowLimit, isSingleSelect } from './limit.js'

export type {
  ReadSqlOptions,
  SqlClient,
  SqlConnection,
  SqlConnectionConfig,
  SqlDialect,
  SqlParams,
} from './types.js'
export { inferDialectFromUrl, normalizeDialect } from './util.js'

function isSqlClient(value: unknown): value is SqlClient {
  return !!value && typeof value === 'object' && typeof (value as SqlClient).query === 'function'
}

function asConfig(connection: string | SqlConnectionConfig): SqlConnectionConfig {
  if (typeof connection === 'string') {
    if (/\.(db|sqlite|sqlite3)$/i.test(connection) || connection === ':memory:') {
      return { dialect: 'sqlite', filename: connection }
    }
    return { url: connection }
  }
  return connection
}

async function openClient(config: SqlConnectionConfig, dialect: SqlDialect): Promise<SqlClient> {
  switch (dialect) {
    case 'postgres':
      return connectPostgres(config)
    case 'mssql':
      return connectMssql(config)
    case 'clickhouse':
      return connectClickhouse(config)
    case 'mysql':
      return connectMysql(config)
    case 'sqlite':
      return connectSqlite(config)
    default: {
      const _exhaustive: never = dialect
      throw new Error(`Unsupported dialect: ${String(_exhaustive)}`)
    }
  }
}

/**
 * Open a dedicated driver client / pool for a URL or config. The caller owns it: pass it to `readSql` as
 * many times as needed (readSql never closes a client it was given) and call `close()` when done. This is the
 * supported way to share one connection across reads; with MS SQL Server it is also the only way to hold two
 * databases open at once (each client has its own `ConnectionPool`, nothing touches the driver's global pool).
 */
export async function openSqlClient(
  connection: string | SqlConnectionConfig,
  options: { dialect?: SqlDialect } = {},
): Promise<SqlClient> {
  const config = asConfig(connection)
  return openClient(config, resolveDialect(config, options.dialect))
}

/**
 * Execute SQL and return row objects.
 *
 * `connection` may be:
 * - URL string (`postgres://…`, `mssql://…`, `clickhouse://…`, `mysql://…`, `sqlite://…` / `.db` path)
 * - `{ dialect, host, … }` config
 * - an existing `{ query, close? }` client
 */
export async function readSqlRows(
  sql: string,
  connection: SqlConnection,
  options: ReadSqlOptions = {},
): Promise<Record<string, unknown>[]> {
  const limit = options.nRows
  const limited = (dialect: SqlDialect | undefined): string =>
    limit !== undefined && options.pushdown !== false ? applyRowLimit(sql, limit, dialect).sql : sql

  if (isSqlClient(connection)) {
    // a duck-typed client has no dialect of its own: pushdown only when the caller names one
    const rows = (await connection.query(limited(options.dialect), options.params)).map(normalizeSqlRow)
    return limit !== undefined ? rows.slice(0, limit) : rows
  }

  const config = asConfig(connection)
  const dialect = resolveDialect(config, options.dialect)
  const client = await openClient(config, dialect)
  try {
    const rows = (await client.query(limited(dialect), options.params)).map(normalizeSqlRow)
    return limit !== undefined ? rows.slice(0, limit) : rows
  } finally {
    if (!config.keepAlive && client.close) await client.close()
  }
}

function normalizeSqlRow(row: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(row)) {
    if (typeof v === 'bigint') {
      const n = Number(v)
      setRowField(out, k, Number.isSafeInteger(n) ? n : v.toString())
    } else if (v instanceof Date) {
      setRowField(out, k, v.getTime())
    } else {
      setRowField(out, k, v)
    }
  }
  return out
}
