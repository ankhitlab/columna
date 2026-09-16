import type { SqlClient, SqlConnectionConfig, SqlParams } from './types.js'
import { hostPortFromUrl, importDriver, missingDriverMessage } from './util.js'

type PgModule = {
  default?: { Client: new (cfg: unknown) => PgClient }
  Client: new (cfg: unknown) => PgClient
}
type PgClient = {
  connect(): Promise<void>
  query: (sql: string, params?: unknown[]) => Promise<{ rows: Record<string, unknown>[] }>
  end(): Promise<void>
}

export async function connectPostgres(config: SqlConnectionConfig): Promise<SqlClient> {
  let mod: PgModule
  try {
    mod = await importDriver<PgModule>('pg', 'postgres')
  } catch (e) {
    throw e instanceof Error ? e : new Error(missingDriverMessage('pg', 'postgres'))
  }
  const Client = mod.Client ?? mod.default?.Client
  if (!Client) throw new Error(missingDriverMessage('pg', 'postgres'))

  const cfg = config.url
    ? { connectionString: config.url }
    : {
        host: config.host ?? 'localhost',
        port: config.port ?? 5432,
        database: config.database,
        user: config.user,
        password: config.password,
      }

  const client = new Client(cfg)
  await client.connect()
  return {
    async query(sql, params) {
      const values = Array.isArray(params) ? params : params ? Object.values(params) : undefined
      const res = await client.query(sql, values)
      return res.rows.map(normalizeRow)
    },
    async close() {
      await client.end()
    },
  }
}

export async function connectMssql(config: SqlConnectionConfig): Promise<SqlClient> {
  type MssqlMod = {
    connect: (cfg: unknown) => Promise<MssqlPool>
    close?: () => Promise<void>
  }
  type MssqlPool = {
    request: () => {
      input: (name: string, value: unknown) => unknown
      query: (sql: string) => Promise<{ recordset: Record<string, unknown>[] }>
    }
    close: () => Promise<void>
  }

  const sql = await importDriver<MssqlMod>('mssql', 'mssql')

  let cfg: Record<string, unknown>
  if (config.url) {
    const u = hostPortFromUrl(config.url)
    cfg = {
      server: u.host || 'localhost',
      port: u.port ?? 1433,
      database: u.pathname || config.database,
      user: u.username ?? config.user,
      password: u.password ?? config.password,
      options: {
        encrypt: config.options?.encrypt ?? u.searchParams.get('encrypt') !== 'false',
        trustServerCertificate:
          config.options?.trustServerCertificate ??
          u.searchParams.get('trustServerCertificate') === 'true',
        instanceName: config.options?.instanceName,
      },
    }
  } else {
    cfg = {
      server: config.server ?? config.host ?? 'localhost',
      port: config.port ?? 1433,
      database: config.database,
      user: config.user,
      password: config.password,
      options: {
        encrypt: config.options?.encrypt ?? true,
        trustServerCertificate: config.options?.trustServerCertificate ?? false,
        instanceName: config.options?.instanceName,
      },
    }
  }

  const pool = await sql.connect(cfg)
  return {
    async query(queryText, params) {
      const req = pool.request()
      if (params) {
        if (Array.isArray(params)) {
          params.forEach((v, i) => {
            ;(req as { input: (n: string, v: unknown) => void }).input(`p${i}`, v)
          })
          // rewrite $1 / ? to @p0 style if needed — keep caller using @name for mssql
        } else {
          for (const [k, v] of Object.entries(params)) {
            ;(req as { input: (n: string, v: unknown) => void }).input(k, v)
          }
        }
      }
      const res = await req.query(queryText)
      return res.recordset.map(normalizeRow)
    },
    async close() {
      await pool.close()
    },
  }
}

export async function connectClickhouse(config: SqlConnectionConfig): Promise<SqlClient> {
  type ChMod = {
    createClient: (cfg: unknown) => {
      query: (opts: {
        query: string
        query_params?: Record<string, unknown>
        format: string
      }) => Promise<{ json: <T>() => Promise<T> }>
      close: () => Promise<void>
    }
  }

  const { createClient } = await importDriver<ChMod>('@clickhouse/client', 'clickhouse')

  let url = config.clickhouseUrl ?? config.url
  let username = config.user
  let password = config.password
  let database = config.database

  if (url && /^clickhouse:|^ch:/i.test(url)) {
    const u = hostPortFromUrl(url.replace(/^ch:/i, 'clickhouse:'))
    url = `http://${u.host}:${u.port ?? 8123}`
    username = username ?? u.username
    password = password ?? u.password
    database = database ?? (u.pathname || undefined)
  } else if (!url) {
    url = `http://${config.host ?? 'localhost'}:${config.port ?? 8123}`
  }

  const client = createClient({
    url,
    username: username ?? 'default',
    password: password ?? '',
    database,
  })

  return {
    async query(sql, params) {
      const query_params =
        params && !Array.isArray(params) ? (params as Record<string, unknown>) : undefined
      // ClickHouse named params: {name:Type}; positional arrays not portable — join as literals discouraged
      if (Array.isArray(params) && params.length > 0) {
        throw new Error('ClickHouse adapter expects named params as an object, not a positional array')
      }
      const result = await client.query({
        query: sql,
        query_params,
        format: 'JSONEachRow',
      })
      const rows = await result.json<Record<string, unknown>[]>()
      return rows.map(normalizeRow)
    },
    async close() {
      await client.close()
    },
  }
}

export async function connectMysql(config: SqlConnectionConfig): Promise<SqlClient> {
  type MysqlMod = {
    createConnection: (cfg: unknown) => Promise<{
      execute: (sql: string, params?: unknown[]) => Promise<[Record<string, unknown>[], unknown]>
      end: () => Promise<void>
    }>
  }

  const mysql = await importDriver<MysqlMod>('mysql2/promise', 'mysql')

  const cfg = config.url
    ? config.url
    : {
        host: config.host ?? 'localhost',
        port: config.port ?? 3306,
        database: config.database,
        user: config.user,
        password: config.password,
      }

  const conn = await mysql.createConnection(cfg)
  return {
    async query(sql, params) {
      const values = Array.isArray(params) ? params : params ? Object.values(params) : []
      const [rows] = await conn.execute(sql, values)
      return (Array.isArray(rows) ? rows : []).map(normalizeRow)
    },
    async close() {
      await conn.end()
    },
  }
}

export async function connectSqlite(config: SqlConnectionConfig): Promise<SqlClient> {
  type BetterSqlite = {
    default: new (filename: string) => {
      prepare: (sql: string) => {
        all: (...params: unknown[]) => Record<string, unknown>[]
      }
      close: () => void
    }
  }

  const mod = await importDriver<BetterSqlite>('better-sqlite3', 'sqlite')
  const Database = mod.default
  const filename = config.filename ?? config.database ?? config.url?.replace(/^file:/, '') ?? ':memory:'
  const db = new Database(filename === 'sqlite://:memory:' || filename === 'sqlite::memory:' ? ':memory:' : filename.replace(/^sqlite:\/\//, ''))

  return {
    async query(sql, params) {
      const stmt = db.prepare(sql)
      const values = Array.isArray(params) ? params : params ? Object.values(params) : []
      return stmt.all(...values).map(normalizeRow)
    },
    close() {
      db.close()
    },
  }
}

function normalizeRow(row: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(row)) {
    if (typeof v === 'bigint') {
      const n = Number(v)
      out[k] = Number.isSafeInteger(n) ? n : v.toString()
    } else if (v instanceof Date) {
      out[k] = v.getTime()
    } else if (isNodeBuffer(v)) {
      out[k] = new Uint8Array(v as Uint8Array)
    } else {
      out[k] = v
    }
  }
  return out
}

function isNodeBuffer(v: unknown): boolean {
  return (
    typeof globalThis !== 'undefined' &&
    'Buffer' in globalThis &&
    typeof (globalThis as { Buffer?: { isBuffer?: (x: unknown) => boolean } }).Buffer?.isBuffer ===
      'function' &&
    Boolean((globalThis as { Buffer: { isBuffer: (x: unknown) => boolean } }).Buffer.isBuffer(v))
  )
}