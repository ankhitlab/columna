import type { TableView } from '@columna/arrow'
import { estimateTableBytes } from '@columna/arrow'
import type { PlanNode } from './types.js'
import { getMemoryPolicy } from './memory.js'
import { optimizePlan } from './optimize.js'

export type PersistLookup = { table: TableView } | null

type CacheEntry = {
  key: string
  table: TableView
  bytes: number
  pinned: boolean
}

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
      if (Object.is(v, -0)) return { __num: '-0' }
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

/** Canonical cache key: mark/lookup/store always hash the optimized plan. */
function cacheKey(plan: PlanNode): string {
  return hashPlan(optimizePlan(plan))
}

export interface PersistCacheOptions {
  /** LRU cap in bytes for this cache. Unset: the process MemoryPolicy's `maxCacheBytes` (the default cache) / no cap. */
  maxBytes?: number
}

/**
 * An LRU of materialized plan results keyed by the optimized plan's structural hash. One instance is the
 * process default (what `persist()` on the default runtime uses); `new Runtime({ persist: new PersistCache() })`
 * gives a tenant / request / test its own — nothing it caches is visible to any other runtime, and
 * `clear()` drops exactly its entries.
 */
export class PersistCache {
  private readonly cache = new Map<string, CacheEntry>()
  private readonly pending = new Set<string>()
  private readonly lru: string[] = []
  private bytes = 0

  constructor(private readonly options: PersistCacheOptions = {}) {}

  /** Mark a plan: the next execution stores its table. */
  mark(plan: PlanNode): void {
    this.pending.add(cacheKey(plan))
  }

  unmark(plan: PlanNode): void {
    const key = cacheKey(plan)
    this.pending.delete(key)
    this.dropKey(key)
  }

  lookup(plan: PlanNode): PersistLookup {
    if (this.cache.size === 0) return null
    const key = cacheKey(plan)
    const hit = this.cache.get(key)
    if (!hit) return null
    const idx = this.lru.indexOf(key)
    if (idx >= 0) this.lru.splice(idx, 1)
    this.lru.push(key)
    return { table: hit.table }
  }

  /** Store when the plan was marked (or is already cached). */
  maybeStore(plan: PlanNode, table: TableView): void {
    if (this.pending.size === 0 && this.cache.size === 0) return
    const key = cacheKey(plan)
    if (!this.pending.has(key) && !this.cache.has(key)) return
    this.store(plan, table, true)
  }

  store(plan: PlanNode, table: TableView, pinned = true): void {
    const key = cacheKey(plan)
    this.pending.add(key)
    const bytes = estimateTableBytes(table)
    const existing = this.cache.get(key)
    if (existing) {
      this.bytes -= existing.bytes
      const idx = this.lru.indexOf(key)
      if (idx >= 0) this.lru.splice(idx, 1)
    }
    this.cache.set(key, { key, table, bytes, pinned })
    this.lru.push(key)
    this.bytes += bytes
    this.evictIfNeeded()
  }

  drop(plan: PlanNode): boolean {
    return this.dropKey(cacheKey(plan))
  }

  private dropKey(key: string): boolean {
    this.pending.delete(key)
    const existing = this.cache.get(key)
    if (!existing) return false
    this.cache.delete(key)
    this.bytes -= existing.bytes
    const idx = this.lru.indexOf(key)
    if (idx >= 0) this.lru.splice(idx, 1)
    return true
  }

  clear(): void {
    this.cache.clear()
    this.lru.length = 0
    this.bytes = 0
    this.pending.clear()
  }

  stats(): { entries: number; bytes: number } {
    return { entries: this.cache.size, bytes: this.bytes }
  }

  private evictIfNeeded(): void {
    const max = this.options.maxBytes ?? (this === defaultPersistCache ? getMemoryPolicy().maxCacheBytes : undefined)
    if (max == null || max <= 0) return
    while (this.bytes > max && this.lru.length) {
      let victim = -1
      for (let i = 0; i < this.lru.length; i++) {
        const e = this.cache.get(this.lru[i]!)
        if (e && !e.pinned) {
          victim = i
          break
        }
      }
      if (victim < 0) victim = 0
      const key = this.lru.splice(victim, 1)[0]!
      const e = this.cache.get(key)
      if (e) {
        this.bytes -= e.bytes
        this.cache.delete(key)
      }
    }
  }
}

/** The process-wide cache behind the default runtime and the module-level helpers below. */
export const defaultPersistCache = new PersistCache()

export function markPlanPersist(plan: PlanNode): void {
  defaultPersistCache.mark(plan)
}
export function unmarkPlanPersist(plan: PlanNode): void {
  defaultPersistCache.unmark(plan)
}
export function lookupPersistCache(plan: PlanNode): PersistLookup {
  return defaultPersistCache.lookup(plan)
}
export function maybeStorePersist(plan: PlanNode, table: TableView): void {
  defaultPersistCache.maybeStore(plan, table)
}
export function storePersistCache(plan: PlanNode, table: TableView, pinned = true): void {
  defaultPersistCache.store(plan, table, pinned)
}
export function dropPersistCache(plan: PlanNode): boolean {
  return defaultPersistCache.drop(plan)
}
export function clearPersistCache(): void {
  defaultPersistCache.clear()
}
export function persistCacheStats(): { entries: number; bytes: number } {
  return defaultPersistCache.stats()
}
