import type { SqlConnectionConfig, SqlDialect } from './types.js'

const DIALECT_ALIASES: Record<string, SqlDialect> = {
  postgres: 'postgres',
  postgresql: 'postgres',
  pg: 'postgres',
  mssql: 'mssql',
  sqlserver: 'mssql',
  'sql-server': 'mssql',
  clickhouse: 'clickhouse',
  ch: 'clickhouse',
  mysql: 'mysql',
  mariadb: 'mysql',
  sqlite: 'sqlite',
  sqlite3: 'sqlite',
}

export function normalizeDialect(raw: string): SqlDialect {
  const key = raw.trim().toLowerCase()
  const d = DIALECT_ALIASES[key]
  if (!d) {
    throw new Error(
      `Unknown SQL dialect "${raw}". Supported: postgres, mssql, clickhouse, mysql, sqlite`,
    )
  }
  return d
}

/** Infer dialect from a connection URL or sqlite file path. */
export function inferDialectFromUrl(url: string): SqlDialect | null {
  const s = url.trim()
  const m = /^([a-z0-9+.-]+):/i.exec(s)
  if (m) {
    const scheme = m[1]!.toLowerCase().split('+')[0]!
    if (scheme === 'file') {
      if (/\.(db|sqlite|sqlite3)$/i.test(s) || s.includes('sqlite')) return 'sqlite'
      return null
    }
    try {
      return normalizeDialect(scheme)
    } catch {
      /* fall through */
    }
  }
  if (/\.(db|sqlite|sqlite3)$/i.test(s) || s === ':memory:') return 'sqlite'
  return null
}

export function resolveDialect(config: SqlConnectionConfig, override?: SqlDialect): SqlDialect {
  if (override) return override
  if (config.dialect) return normalizeDialect(config.dialect)
  if (config.url) {
    const inferred = inferDialectFromUrl(config.url)
    if (inferred) return inferred
  }
  if (config.filename || (config.database && /\.(db|sqlite|sqlite3)$/i.test(config.database))) {
    return 'sqlite'
  }
  throw new Error(
    'Cannot infer SQL dialect. Pass dialect: "postgres" | "mssql" | "clickhouse" | "mysql" | "sqlite", or use a typed URL (postgres://, mssql://, …).',
  )
}

export function missingDriverMessage(pkg: string, dialect: SqlDialect): string {
  return (
    `${dialect} support requires the optional "${pkg}" package. ` +
    `Install it with: pnpm add ${pkg}`
  )
}

export async function importDriver<T>(pkg: string, dialect: SqlDialect): Promise<T> {
  try {
    return (await import(pkg)) as T
  } catch {
    throw new Error(missingDriverMessage(pkg, dialect))
  }
}

export function parsePostgresUrl(url: string): {
  connectionString: string
} {
  return { connectionString: url }
}

export function hostPortFromUrl(url: string): { protocol: string; host: string; port?: number; pathname: string; username?: string; password?: string; searchParams: URLSearchParams } {
  const u = new URL(url)
  return {
    protocol: u.protocol.replace(/:$/, ''),
    host: u.hostname,
    port: u.port ? Number(u.port) : undefined,
    pathname: u.pathname.replace(/^\//, ''),
    username: u.username ? decodeURIComponent(u.username) : undefined,
    password: u.password ? decodeURIComponent(u.password) : undefined,
    searchParams: u.searchParams,
  }
}
