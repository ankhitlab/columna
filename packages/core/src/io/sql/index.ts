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
import { resolveDialect } from './util.js'

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

  if (isSqlClient(connection)) {
    const rows = (await connection.query(sql, options.params)).map(normalizeSqlRow)
    return limit !== undefined ? rows.slice(0, limit) : rows
  }

  const config = asConfig(connection)
  const dialect = resolveDialect(config, options.dialect)
  const client = await openClient(config, dialect)
  try {
    const rows = (await client.query(sql, options.params)).map(normalizeSqlRow)
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
      out[k] = Number.isSafeInteger(n) ? n : v.toString()
    } else if (v instanceof Date) {
      out[k] = v.getTime()
    } else {
      out[k] = v
    }
  }
  return out
}
