import type { DType } from '@columna/arrow'

/**
 * What a reader accepts. Prefer the explicit forms — `{ path }`, `{ url }`, `{ text }` (or `io.path()` etc.) —
 * whenever the string comes from outside the program; a bare string is classified heuristically
 * (http(s):// → fetch, path-like → file, else content) unless `mode` says otherwise.
 */
export type IoSource =
  | string
  | URL
  | Uint8Array
  | ArrayBuffer
  | Blob
  | { text: string }
  | { path: string }
  | { url: string | URL }

/** How to interpret a bare string source. */
export type IoSourceMode = 'auto' | 'text' | 'path' | 'url'

/**
 * Boundaries for network / filesystem loads. Set once for the process with `setIoPolicy()` and / or per
 * call; a load must satisfy both (per-call options narrow, never widen).
 */
export type IoPolicy = {
  /** Hostnames a URL may point to: exact (`api.example.com`), with port (`host:8443`) or wildcard (`*.example.com`). */
  allowedHosts?: string[]
  /** Default `['http:', 'https:']`. `file://` URLs are filesystem reads and follow `allowedDirs` instead. */
  allowedProtocols?: Array<'http:' | 'https:'>
  /**
   * Reject loopback / RFC 1918 / link-local / ULA / cloud-metadata hosts written literally or as
   * `localhost`. This is a name check, not DNS: a public name that resolves to a private address (DNS
   * rebinding) is not caught — pass a `fetch` that resolves and checks addresses if that matters.
   */
  denyPrivateHosts?: boolean
  /** Directories a path may live in (real paths are compared, so symlinks cannot escape). */
  allowedDirs?: string[]
  /** Maximum payload size in bytes; responses are read incrementally and cut off past the cap. */
  maxBytes?: number
  /** Abort the load after this many milliseconds. */
  timeoutMs?: number
  /** Caller-owned cancellation. */
  signal?: AbortSignal
  /** Maximum redirect hops for URL loads (default 5); every hop is checked against the policy. */
  maxRedirects?: number
  /** Custom fetch (proxy, DNS pinning, auth); receives `(url, { redirect: 'manual', signal })`. */
  fetch?: typeof fetch
}

/** Options shared by every reader that resolves an `IoSource`. */
export type IoLoadOptions = IoPolicy & {
  /** Interpret a bare string as content / path / URL instead of guessing (default `'auto'`). */
  mode?: IoSourceMode
  /** Treat a string source as raw content, never as a path (alias of `mode: 'text'`). */
  content?: boolean
  /** Text encoding when reading bytes/files (default utf-8). */
  encoding?: string
}

export type ReadCsvOptions = IoLoadOptions & {
  /** Column separator (default `,`). Alias of `delimiter`. */
  separator?: string
  /** Alias of `separator` (pandas-style). */
  delimiter?: string
  /** First row is header (default true). Number = header row index after skipRows. */
  header?: boolean | number
  /** Polars alias for `header !== false`. */
  hasHeader?: boolean
  /** Explicit column names (used when header is false, or to override). */
  names?: string[]
  /** Rows to skip at the start of the file (before header). */
  skipRows?: number
  /** Max data rows to read (after header). */
  nRows?: number
  /** Skip empty lines (default true). */
  skipBlankLines?: boolean
  /** Lines starting with this prefix are ignored. */
  comment?: string
  /** Quote character (default `"`). */
  quoteChar?: string
  /** Trim whitespace after delimiter (default false). */
  skipInitialSpace?: boolean
  /** Values treated as null. */
  nullValues?: string | string[]
  /** Values treated as boolean true. */
  trueValues?: string | string[]
  /** Values treated as boolean false. */
  falseValues?: string | string[]
  /** Decimal mark (default `.`). */
  decimal?: string
  /** Thousands separator to strip before parsing numbers. */
  thousands?: string
  /** Keep only these columns (names or 0-based indices). */
  usecols?: Array<string | number>
  /** Force column dtypes after parse. */
  dtypes?: Record<string, DType>
}

export type ReadJsonOptions = IoLoadOptions & {
  /** `records` = array of objects (default); `columns` = {col: values[]}; `lines` = NDJSON. */
  orient?: 'records' | 'columns' | 'values' | 'lines'
  /** Read newline-delimited JSON (NDJSON). Alias of orient:'lines'. */
  lines?: boolean
  /** Skip first N lines (NDJSON) or N array elements. */
  skipRows?: number
  /** Max records to read. */
  nRows?: number
}

export type ReadExcelOptions = IoLoadOptions & {
  /** Sheet by name or 0-based index (default 0). */
  sheet?: string | number
  /** Alias of `sheet`. */
  sheetName?: string | number
  /** Header row (default 0). `false` = no header. */
  header?: number | boolean
  names?: string[]
  skipRows?: number
  nRows?: number
  /** Column names or indices to keep. */
  usecols?: Array<string | number>
  nullValues?: string | string[]
}

export type ReadParquetOptions = IoLoadOptions & {
  /** Column subset by name. */
  columns?: string[]
  /** Inclusive start row (0-based). */
  rowStart?: number
  /** Exclusive end row. */
  rowEnd?: number
  /** Max rows (= rowEnd - rowStart when rowStart set). */
  nRows?: number
}
