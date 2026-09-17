/** Supported SQL dialects for DataFrame.readSql / readDatabase. */
export type SqlDialect = 'postgres' | 'mssql' | 'clickhouse' | 'mysql' | 'sqlite'

export type SqlParams = unknown[] | Record<string, unknown>

/**
 * Duck-typed client: any object with `query(sql, params?) → rows`.
 * Useful for custom pools, ORMs, or tests.
 */
export type SqlClient = {
  query: (sql: string, params?: SqlParams) => Promise<Record<string, unknown>[]>
  close?: () => void | Promise<void>
}

/** Structured connection settings (URL or discrete fields). */
export type SqlConnectionConfig = {
  dialect?: SqlDialect
  /** Connection URL, e.g. `postgres://user:pass@host:5432/db`. */
  url?: string
  host?: string
  /** Alias of `host` for MS SQL configs. */
  server?: string
  port?: number
  database?: string
  user?: string
  password?: string
  /**
   * SQLite file path (`:memory:` allowed).
   * Also accepted as `database` or bare string path ending in `.db` / `.sqlite`.
   */
  filename?: string
  /** MS SQL / tedious extras. */
  options?: {
    encrypt?: boolean
    trustServerCertificate?: boolean
    instanceName?: string
  }
  /** ClickHouse HTTP url override (default builds from host/port). */
  clickhouseUrl?: string
  /**
   * @deprecated Every `readSql` call with a URL / config opens a dedicated connection (or pool) and closes it
   * afterwards; with `keepAlive` that connection is left open but nothing returns a handle to it, so it can
   * only leak. To reuse a connection, open it once with `openSqlClient()` and pass the client to `readSql`,
   * which never closes a client it did not open.
   */
  keepAlive?: boolean
}

export type SqlConnection = string | SqlConnectionConfig | SqlClient

export type ReadSqlOptions = {
  /** Query parameters (positional array or named object, dialect-dependent). */
  params?: SqlParams
  /** Force dialect when it cannot be inferred from the URL. */
  dialect?: SqlDialect
  /**
   * Maximum rows in the returned DataFrame. The driver's full result set is still fetched and buffered — this is
   * a slice after the fact, never a LIMIT pushdown. To bound the query, transfer and driver memory, put LIMIT /
   * TOP / FETCH FIRST in the SQL itself.
   */
  nRows?: number
}
