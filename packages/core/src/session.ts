/**
 * A `Session` bundles the three pieces of state that are otherwise process-wide — the execution runtime
 * (engine, strictness, memory policy), the `persist()` cache and the IO policy — into one object a server
 * can create per tenant, per request or per test. Nothing a session does is visible to another session or
 * to the process defaults: its frames run on its own `Runtime`, `persist()` fills its own `PersistCache`,
 * and every read it performs must pass its IO policy in addition to the process floor from `setIoPolicy()`.
 *
 * ```ts
 * const tenant = createSession({
 *   io: { allowedHosts: ['data.acme.example'], denyPrivateHosts: true, maxBytes: 200e6, timeoutMs: 30_000 },
 *   runtime: { engine: 'cpu', strict: true, memory: { maxBytes: 512 * 1024 * 1024 } },
 *   persist: { maxBytes: 128 * 1024 * 1024 },
 * })
 * const orders = await tenant.readCsv({ url: 'https://data.acme.example/orders.csv' })
 * const top = await orders.lazy().groupBy('country').agg({ r: col('revenue').sum() }).persist().collect()
 * tenant.close() // drops the session's cache
 * ```
 */
import { PersistCache, Runtime, getDefaultRuntime, type Backend, type RuntimeOptions, type PersistCacheOptions } from '@columna/runtime'
import type { ArrowIpcReadOptions, ArrowLike } from '@columna/arrow'
import { DataFrame, LazyFrame, type InferColumns, type Row } from './dataframe.js'
import type {
  IoLoadOptions,
  IoPolicy,
  IoSource,
  ReadCsvOptions,
  ReadExcelOptions,
  ReadJsonOptions,
  ReadParquetOptions,
} from './io/types.js'

export interface SessionOptions {
  /** IO policy every read of this session must pass (a floor on top of `setIoPolicy()`, narrowed by per-call options). */
  io?: IoPolicy
  /** Engine, strictness, thresholds and memory policy of the session's runtime. */
  runtime?: Omit<RuntimeOptions, 'persist'>
  /** The session's own `persist()` cache; `false` shares the process-wide cache. */
  persist?: PersistCacheOptions | PersistCache | false
  /**
   * Accelerator backends for the session's runtime. Default: whatever the process default runtime has
   * registered (the `columna` entry registers WASM and WebGPU there), so a session sees the same engines.
   * Pass `[]` for a CPU-only session.
   */
  backends?: Backend[]
}

export class Session {
  readonly runtime: Runtime
  readonly io: IoPolicy
  readonly persistCache: PersistCache
  private readonly ownsCache: boolean
  private closed = false

  constructor(options: SessionOptions = {}) {
    this.io = { ...(options.io ?? {}) }
    const cacheOpt = options.persist
    const cache = cacheOpt instanceof PersistCache ? cacheOpt : cacheOpt === false ? null : new PersistCache(cacheOpt ?? {})
    this.runtime = new Runtime({ ...(options.runtime ?? {}), ...(cache ? { persist: cache } : {}) })
    for (const b of options.backends ?? getDefaultRuntime().listBackends()) if (b.name !== 'cpu') this.runtime.register(b)
    this.persistCache = this.runtime.persist
    this.ownsCache = cache !== null
  }

  /** Per-call IO options with this session's policy added as a floor. */
  ioOptions<O extends IoLoadOptions>(options: O = {} as O): O {
    this.assertOpen()
    return { ...options, floors: [...(options.floors ?? []), this.io] }
  }

  /** Bind a frame or plan to this session's runtime (its engine, memory policy and cache). */
  bind<S extends Row>(frame: DataFrame<S>): DataFrame<S>
  bind<S extends Row>(frame: LazyFrame<S>): LazyFrame<S>
  bind<S extends Row>(frame: DataFrame<S> | LazyFrame<S>): DataFrame<S> | LazyFrame<S> {
    this.assertOpen()
    return frame.withRuntime(this.runtime)
  }

  // ---- constructors bound to the session ----------------------------------------------------------

  fromRows<T extends Row>(rows: readonly T[]): DataFrame<T> {
    return this.bind(DataFrame.fromRows(rows))
  }

  fromColumns<C extends Parameters<typeof DataFrame.fromColumns>[0]>(cols: C, options?: { copy?: boolean }): DataFrame<InferColumns<C>> {
    return this.bind(DataFrame.fromColumns(cols, options) as DataFrame<InferColumns<C>>)
  }

  fromCSV<S extends Row = Row>(csv: string, options: ReadCsvOptions = {}): DataFrame<S> {
    return this.bind(DataFrame.fromCSV<S>(csv, options))
  }

  fromJSON<S extends Row = Row>(data: Record<string, unknown>[] | string, options: ReadJsonOptions = {}): DataFrame<S> {
    return this.bind(DataFrame.fromJSON<S>(data, options))
  }

  fromArrowIpc<S extends Row = Row>(bytes: Uint8Array | ArrayBuffer, options?: ArrowIpcReadOptions): DataFrame<S> {
    return this.bind(DataFrame.fromArrowIpc<S>(bytes, options))
  }

  fromArrowLike<S extends Row = Row>(arrow: ArrowLike): DataFrame<S> {
    return this.bind(DataFrame.fromArrowLike<S>(arrow))
  }

  // ---- readers: session policy + process floor + per-call options ---------------------------------

  async readCsv<S extends Row = Row>(source: IoSource, options: ReadCsvOptions = {}): Promise<DataFrame<S>> {
    return this.bind(await DataFrame.readCsv<S>(source, this.ioOptions(options)))
  }

  async readJson<S extends Row = Row>(source: IoSource, options: ReadJsonOptions = {}): Promise<DataFrame<S>> {
    return this.bind(await DataFrame.readJson<S>(source, this.ioOptions(options)))
  }

  async readExcel<S extends Row = Row>(source: IoSource, options: ReadExcelOptions = {}): Promise<DataFrame<S>> {
    return this.bind(await DataFrame.readExcel<S>(source, this.ioOptions(options)))
  }

  async readParquet<S extends Row = Row>(source: IoSource, options: ReadParquetOptions = {}): Promise<DataFrame<S>> {
    return this.bind(await DataFrame.readParquet<S>(source, this.ioOptions(options)))
  }

  async readArrowIpc<S extends Row = Row>(source: IoSource, options: IoLoadOptions & ArrowIpcReadOptions = {}): Promise<DataFrame<S>> {
    return this.bind(await DataFrame.readArrowIpc<S>(source, this.ioOptions(options)))
  }

  /** Drop everything this session cached. Frames already materialized stay usable. */
  close(): void {
    if (this.ownsCache) this.persistCache.clear()
    this.closed = true
  }

  get isClosed(): boolean {
    return this.closed
  }

  private assertOpen(): void {
    if (this.closed) throw new Error('Session is closed')
  }
}

export function createSession(options: SessionOptions = {}): Session {
  return new Session(options)
}
