import type { TableView } from '@columna/arrow'
import { estimateTableBytes } from '@columna/arrow'
import type { PlanNode } from './types.js'
import { getMemoryPolicy } from './memory.js'

export type PersistLookup = { table: TableView } | null

type CacheEntry = {
  key: string
  table: TableView
  bytes: number
  pinned: boolean
}

const cache = new Map<string, CacheEntry>()
const pendingPersist = new Set<string>()
let cacheBytes = 0
const lru: string[] = []

/** Identity tokens for scan tables — never stringify column payloads. */
const tableIds = new WeakMap<object, number>()
let nextTableId = 1

function tableToken(table: TableView): { __table: number; rows: number; cols: string[] } {
  let id = tableIds.get(table)
  if (id === undefined) {
    id = nextTableId++
    tableIds.set(table, id)
  }
  return {
    __table: id,
    rows: table.numRows,
    cols: table.columns.map((c) => c.field.name),
  }
}

function isTableView(v: unknown): v is TableView {
  return (
    !!v &&
    typeof v === 'object' &&
    'numRows' in v &&
    'columns' in v &&
    Array.isArray((v as TableView).columns) &&
    'schema' in v
  )
}

/** Process-local identity for UDFs so distinct mapElements.fn values do not collide. */
let fnIdSeq = 0
const fnIds = new WeakMap<object, number>()

function functionToken(fn: object): { __fn: number } {
  let id = fnIds.get(fn)
  if (id === undefined) {
    id = ++fnIdSeq
    fnIds.set(fn, id)
  }
  return { __fn: id }
}

/** Stable structural hash of a plan tree (good enough for persist keys). */
export function hashPlan(plan: PlanNode): string {
  return JSON.stringify(plan, (_k, v) => {
    if (typeof v === 'bigint') return v.toString()
    if (typeof v === 'function') return functionToken(v)
    // JSON.stringify maps NaN / ±Infinity to null — keep them distinct from null literals.
    if (typeof v === 'number') {
      if (Number.isNaN(v)) return { __num: 'NaN' }
      if (v === Infinity) return { __num: 'Infinity' }
      if (v === -Infinity) return { __num: '-Infinity' }
    }
    if (isTableView(v)) return tableToken(v)
    // Typed arrays / buffers appear only inside tables; skip if any leak through.
    if (ArrayBuffer.isView(v)) return { __view: (v as ArrayBufferView).byteLength }
    return v
  })
}

export function markPlanPersist(plan: PlanNode): void {
  pendingPersist.add(hashPlan(plan))
}

export function unmarkPlanPersist(plan: PlanNode): void {
  const key = hashPlan(plan)
  pendingPersist.delete(key)
  dropPersistCacheKey(key)
}

export function lookupPersistCache(plan: PlanNode): PersistLookup {
  if (cache.size === 0) return null
  const key = hashPlan(plan)
  const hit = cache.get(key)
  if (!hit) return null
  const idx = lru.indexOf(key)
  if (idx >= 0) lru.splice(idx, 1)
  lru.push(key)
  return { table: hit.table }
}

export function maybeStorePersist(plan: PlanNode, table: TableView): void {
  if (pendingPersist.size === 0 && cache.size === 0) return
  const key = hashPlan(plan)
  if (!pendingPersist.has(key) && !cache.has(key)) return
  storePersistCache(plan, table, true)
}

export function storePersistCache(plan: PlanNode, table: TableView, pinned = true): void {
  const key = hashPlan(plan)
  pendingPersist.add(key)
  const bytes = estimateTableBytes(table)
  const existing = cache.get(key)
  if (existing) {
    cacheBytes -= existing.bytes
    const idx = lru.indexOf(key)
    if (idx >= 0) lru.splice(idx, 1)
  }
  cache.set(key, { key, table, bytes, pinned })
  lru.push(key)
  cacheBytes += bytes
  evictIfNeeded()
}

export function dropPersistCache(plan: PlanNode): boolean {
  return dropPersistCacheKey(hashPlan(plan))
}

function dropPersistCacheKey(key: string): boolean {
  pendingPersist.delete(key)
  const existing = cache.get(key)
  if (!existing) return false
  cache.delete(key)
  cacheBytes -= existing.bytes
  const idx = lru.indexOf(key)
  if (idx >= 0) lru.splice(idx, 1)
  return true
}

export function clearPersistCache(): void {
  cache.clear()
  lru.length = 0
  cacheBytes = 0
  pendingPersist.clear()
}

function evictIfNeeded(): void {
  const max = getMemoryPolicy().maxCacheBytes
  if (max == null || max <= 0) return
  while (cacheBytes > max && lru.length) {
    let victim = -1
    for (let i = 0; i < lru.length; i++) {
      const e = cache.get(lru[i]!)
      if (e && !e.pinned) {
        victim = i
        break
      }
    }
    if (victim < 0) victim = 0
    const key = lru.splice(victim, 1)[0]!
    const e = cache.get(key)
    if (e) {
      cacheBytes -= e.bytes
      cache.delete(key)
    }
  }
}

export function persistCacheStats(): { entries: number; bytes: number } {
  return { entries: cache.size, bytes: cacheBytes }
}
