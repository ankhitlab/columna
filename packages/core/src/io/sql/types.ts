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
  /** Keep the underlying driver connection open (caller must close). */
  keepAlive?: boolean
}

export type SqlConnection = string | SqlConnectionConfig | SqlClient

export type ReadSqlOptions = {
  /** Query parameters (positional array or named object, dialect-dependent). */
  params?: SqlParams
  /** Force dialect when it cannot be inferred from the URL. */
  dialect?: SqlDialect
  /** Max rows to materialize (applied after fetch when driver has no LIMIT pushdown). */
  nRows?: number
}
