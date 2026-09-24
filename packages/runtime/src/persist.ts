import type { TableView } from '@columna/arrow'
import { estimateTableBytes, isBorrowedTable } from '@columna/arrow'
import type { EngineKind, PlanNode } from './types.js'
import { getMemoryPolicy } from './memory.js'
import { optimizePlan } from './optimize.js'

export type PersistLookup = { table: TableView } | null

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
  return JSON.stringify(plan, (_k, v) => planReplacer(v))
}

function planReplacer(v: unknown): unknown {
  {
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
  }
}


/** What produced a cached table — checked against strict engine requests before the table is served. */
export interface PersistProvenance {
  requested: EngineKind
  dispatched: EngineKind
  strict: boolean
  /** Engines that executed at least one node of the plan that produced the table. */
  backendsUsed: EngineKind[]
}

export interface PersistCacheOptions {
  /** LRU cap on cached table bytes. Unset: the process MemoryPolicy's `maxCacheBytes` (default cache) / no cap. */
  maxBytes?: number
  /** Cap on cached tables (default 256). */
  maxEntries?: number
  /**
   * Cap on marked-but-not-yet-stored plans (default 1024). Each `persist()` records the plan's key; plans that
   * are marked and never collected would otherwise accumulate for the life of the cache.
   */
  maxPending?: number
  /** Entries older than this are dropped on lookup and on every store (default: no expiry). */
  ttlMs?: number
  /** Marks older than this are forgotten (default: no expiry). */
  pendingTtlMs?: number
  /** Clock (tests). Default `Date.now`. */
  now?: () => number
}

/** `persist()` options. */
export interface PersistOptions {
  /**
   * Cache the plan even though it calls user functions (`mapElements`). The cache key identifies a function
   * object, not what its closure reads, so this asserts the functions are pure. Default `false`: plans with
   * UDFs are executed every time and the report says why.
   */
  trustUdfs?: boolean
}

export interface PersistCacheStats {
  entries: number
  bytes: number
  pending: number
  hits: number
  misses: number
  /** Entries dropped by the byte / entry caps. */
  evictions: number
  /** Marks dropped by `maxPending`. */
  pendingEvictions: number
  /** Entries / marks dropped by `ttlMs` / `pendingTtlMs`. */
  expired: number
  /** Marked plans that were executed without caching (borrowed buffers, untrusted UDFs). */
  skipped: number
  /** Cached tables not served because a strict engine request needed another engine's result. */
  strictBypasses: number
}

/** Result of asking the cache about a plan before executing it. */
export type PersistProbe =
  | { status: 'hit'; table: TableView; provenance: PersistProvenance }
  | { status: 'miss' }
  | { status: 'skip'; reason: string }
  | { status: 'unmarked' }

type CacheEntry = { table: TableView; bytes: number; storedAt: number; provenance: PersistProvenance }
type PendingMark = { at: number; trustUdfs: boolean }

/** One pass over the optimized plan: its key, whether it calls UDFs, whether it reads caller-owned buffers. */
function analyze(plan: PlanNode): { key: string; udf: boolean; borrowed: boolean } {
  const optimized = optimizePlan(plan)
  let udf = false
  let borrowed = false
  const key = JSON.stringify(optimized, (_k, v) => {
    if (typeof v === 'function') udf = true
    else if (isTableView(v) && isBorrowedTable(v)) borrowed = true
    return planReplacer(v)
  })
  return { key, udf, borrowed }
}

/** Can a strict request for `requested` be answered with a table produced as `prov` says? */
function satisfiesStrict(requested: EngineKind, strict: boolean, prov: PersistProvenance): boolean {
  if (!strict || requested === 'auto') return true
  if (requested === 'cpu') return prov.backendsUsed.every((b) => b === 'cpu')
  return prov.backendsUsed.includes(requested)
}

/**
 * An LRU of materialized plan results keyed by the optimized plan's structural hash. One instance is the
 * process default (what `persist()` on the default runtime uses); `new Runtime({ persist: new PersistCache() })`
 * gives a tenant / request / test its own — nothing it caches is visible to any other runtime, and
 * `clear()` drops exactly its entries.
 *
 * Semantics (docs/operations.md → persist): a plan is cached only when every input is immutable — frames
 * built with `fromColumns(…, { copy: false })` alias caller memory and are refused — and it calls no UDF
 * unless `persist({ trustUdfs: true })`. Every entry records which engines produced it; a strict engine
 * request is served from the cache only when that engine produced the entry, otherwise the plan runs (and
 * either the engine executes it or `EngineStrictError` is raised, exactly as without a cache).
 */
export class PersistCache {
  private readonly entries = new Map<string, CacheEntry>() // insertion order = LRU order (oldest first)
  private readonly pending = new Map<string, PendingMark>()
  private bytes = 0
  private readonly counters = { hits: 0, misses: 0, evictions: 0, pendingEvictions: 0, expired: 0, skipped: 0, strictBypasses: 0 }
  private readonly now: () => number

  constructor(private readonly options: PersistCacheOptions = {}) {
    this.now = options.now ?? Date.now
    for (const k of ['maxBytes', 'maxEntries', 'maxPending', 'ttlMs', 'pendingTtlMs'] as const) {
      const v = options[k]
      if (v !== undefined && !(v >= 0)) throw new RangeError(`PersistCache: ${k} must be ≥ 0 (got ${v})`)
    }
  }

  /** Mark a plan: its next execution stores the result (if the plan is cacheable). */
  mark(plan: PlanNode, options: PersistOptions = {}): void {
    const { key } = analyze(plan)
    this.pending.delete(key)
    this.pending.set(key, { at: this.now(), trustUdfs: options.trustUdfs === true })
    const max = this.options.maxPending ?? 1024
    while (this.pending.size > max) {
      const oldest = this.pending.keys().next().value as string
      this.pending.delete(oldest)
      this.counters.pendingEvictions++
    }
  }

  unmark(plan: PlanNode): void {
    const { key } = analyze(plan)
    this.pending.delete(key)
    this.dropKey(key)
  }

  /**
   * Called by the runtime before executing: a hit (the table plus its provenance, already checked against a
   * strict request), a miss, a reason the marked plan cannot be cached, or "not marked".
   */
  probe(plan: PlanNode, request: { requested: EngineKind; strict: boolean } = { requested: 'auto', strict: false }): PersistProbe {
    this.expire()
    if (this.pending.size === 0 && this.entries.size === 0) return { status: 'unmarked' }
    const { key, udf, borrowed } = analyze(plan)
    const mark = this.pending.get(key)
    const entry = this.entries.get(key)
    if (!mark && !entry) return { status: 'unmarked' }
    if (borrowed) {
      this.counters.skipped++
      return { status: 'skip', reason: 'reads caller-owned buffers (fromColumns with { copy: false }); results would change with them' }
    }
    if (udf && !mark?.trustUdfs) {
      this.counters.skipped++
      return { status: 'skip', reason: 'calls a UDF (mapElements); pass persist({ trustUdfs: true }) to assert it is pure' }
    }
    if (!entry) {
      this.counters.misses++
      return { status: 'miss' }
    }
    if (!satisfiesStrict(request.requested, request.strict, entry.provenance)) {
      this.counters.strictBypasses++
      return { status: 'miss' }
    }
    this.entries.delete(key)
    this.entries.set(key, entry) // most recently used
    this.counters.hits++
    return { status: 'hit', table: entry.table, provenance: entry.provenance }
  }

  /** Back-compat: the cached table for a plan, ignoring engine semantics. */
  lookup(plan: PlanNode): PersistLookup {
    const p = this.probe(plan)
    return p.status === 'hit' ? { table: p.table } : null
  }

  /** Store after execution when the plan was marked (or is already cached) and is cacheable. */
  maybeStore(plan: PlanNode, table: TableView, provenance?: PersistProvenance): void {
    if (this.pending.size === 0 && this.entries.size === 0) return
    const { key, udf, borrowed } = analyze(plan)
    const mark = this.pending.get(key)
    if (!mark && !this.entries.has(key)) return
    if (borrowed || (udf && !mark?.trustUdfs)) return
    this.put(key, table, provenance)
  }

  /** Store unconditionally (marks the plan). Refuses plans over caller-owned buffers. */
  store(plan: PlanNode, table: TableView, _pinned = true, provenance?: PersistProvenance): void {
    const { key, borrowed } = analyze(plan)
    if (borrowed) throw new Error('PersistCache.store: the plan reads caller-owned buffers (fromColumns { copy: false }) and cannot be cached')
    if (!this.pending.has(key)) this.pending.set(key, { at: this.now(), trustUdfs: true })
    this.put(key, table, provenance)
  }

  drop(plan: PlanNode): boolean {
    return this.dropKey(analyze(plan).key)
  }

  clear(): void {
    this.entries.clear()
    this.pending.clear()
    this.bytes = 0
  }

  stats(): PersistCacheStats {
    return { entries: this.entries.size, bytes: this.bytes, pending: this.pending.size, ...this.counters }
  }

  private put(key: string, table: TableView, provenance?: PersistProvenance): void {
    const existing = this.entries.get(key)
    if (existing) {
      this.bytes -= existing.bytes
      this.entries.delete(key)
    }
    const bytes = estimateTableBytes(table)
    this.entries.set(key, {
      table,
      bytes,
      storedAt: this.now(),
      provenance: provenance ?? { requested: 'auto', dispatched: 'cpu', strict: false, backendsUsed: ['cpu'] },
    })
    this.bytes += bytes
    this.expire()
    this.evict()
  }

  private dropKey(key: string): boolean {
    this.pending.delete(key)
    const existing = this.entries.get(key)
    if (!existing) return false
    this.entries.delete(key)
    this.bytes -= existing.bytes
    return true
  }

  private evict(): void {
    const maxBytes = this.options.maxBytes ?? (this === defaultPersistCache ? getMemoryPolicy().maxCacheBytes : undefined)
    const maxEntries = this.options.maxEntries ?? 256
    const overBytes = () => maxBytes != null && maxBytes > 0 && this.bytes > maxBytes
    while (this.entries.size > 0 && (overBytes() || this.entries.size > maxEntries)) {
      const oldest = this.entries.keys().next().value as string
      const e = this.entries.get(oldest)!
      this.entries.delete(oldest)
      this.bytes -= e.bytes
      this.counters.evictions++
    }
  }

  private expire(): void {
    const now = this.now()
    const ttl = this.options.ttlMs
    if (ttl !== undefined) {
      for (const [k, e] of this.entries) {
        if (now - e.storedAt < ttl) continue
        this.entries.delete(k)
        this.bytes -= e.bytes
        this.counters.expired++
      }
    }
    const pttl = this.options.pendingTtlMs
    if (pttl !== undefined) {
      for (const [k, m] of this.pending) {
        if (now - m.at < pttl) continue
        this.pending.delete(k)
        this.counters.expired++
      }
    }
  }
}

/** The process-wide cache behind the default runtime and the module-level helpers below. */
export const defaultPersistCache = new PersistCache()

export function markPlanPersist(plan: PlanNode, options?: PersistOptions): void {
  defaultPersistCache.mark(plan, options)
}
export function unmarkPlanPersist(plan: PlanNode): void {
  defaultPersistCache.unmark(plan)
}
export function lookupPersistCache(plan: PlanNode): PersistLookup {
  return defaultPersistCache.lookup(plan)
}
export function maybeStorePersist(plan: PlanNode, table: TableView, provenance?: PersistProvenance): void {
  defaultPersistCache.maybeStore(plan, table, provenance)
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
export function persistCacheStats(): PersistCacheStats {
  return defaultPersistCache.stats()
}
