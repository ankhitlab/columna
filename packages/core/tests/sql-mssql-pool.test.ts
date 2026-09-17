import { describe, expect, it, vi } from 'vitest'

/**
 * `mssql` is mocked: the driver's global `connect()` returns one process-wide pool regardless of config,
 * which is exactly what the adapter must never rely on. The mock records every pool it hands out.
 */
const pools: Array<{ cfg: Record<string, unknown>; closed: boolean }> = []
let globalConnectCalls = 0

vi.mock('mssql', () => {
  class ConnectionPool {
    closed = false
    constructor(public cfg: Record<string, unknown>) {
      pools.push(this)
    }
    async connect() {
      return this
    }
    request() {
      const inputs: Record<string, unknown> = {}
      return {
        input: (name: string, value: unknown) => {
          inputs[name] = value
        },
        query: async (sql: string) => ({ recordset: [{ sql, database: this.cfg.database, param_id: inputs.id ?? null }] }),
      }
    }
    async close() {
      this.closed = true
    }
  }
  const connect = async () => {
    globalConnectCalls++
    throw new Error('global pool must not be used')
  }
  return { ConnectionPool, connect, default: { ConnectionPool, connect } }
})

import { DataFrame, openSqlClient } from '../src/index.js'

describe('mssql adapter: one dedicated ConnectionPool per configuration', () => {
  it('two databases read through two pools with their own configs; the global pool is never touched', async () => {
    const a = await DataFrame.readSql('select 1', 'mssql://ua:pa@host-a:1433/db_a')
    const b = await DataFrame.readSql('select 2', { dialect: 'mssql', server: 'host-b', database: 'db_b', user: 'ub', password: 'pb' })
    expect(a.toArray()[0]).toMatchObject({ database: 'db_a' })
    expect(b.toArray()[0]).toMatchObject({ database: 'db_b' })
    expect(pools.length).toBe(2)
    expect(pools[0]!.cfg).toMatchObject({ server: 'host-a', database: 'db_a', user: 'ua' })
    expect(pools[1]!.cfg).toMatchObject({ server: 'host-b', database: 'db_b', user: 'ub' })
    expect(globalConnectCalls).toBe(0)
    // a per-call read closes its own pool only
    expect(pools.map((p) => p.closed)).toEqual([true, true])
  })

  it('openSqlClient hands out a pool the caller owns: reads do not close it, close() closes only that pool', async () => {
    const before = pools.length
    const client = await openSqlClient('mssql://u:p@host-c/db_c')
    const other = await openSqlClient('mssql://u:p@host-d/db_d')
    expect(pools.length).toBe(before + 2)
    const r1 = await DataFrame.readSql('select 1', client)
    const r2 = await DataFrame.readSql('select 2', client, { params: { id: 7 } })
    expect(r1.toArray()[0]).toMatchObject({ database: 'db_c' })
    expect(r2.toArray()[0]).toMatchObject({ database: 'db_c', param_id: 7 })
    expect(pools[before]!.closed).toBe(false)
    await client.close?.()
    expect(pools[before]!.closed).toBe(true)
    expect(pools[before + 1]!.closed).toBe(false)
    await other.close?.()
    expect(globalConnectCalls).toBe(0)
  })
})
