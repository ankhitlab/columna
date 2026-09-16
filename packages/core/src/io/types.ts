import type { DType } from '@columna/arrow'

/** Path, URL, raw text/bytes, or browser Blob. */
export type IoSource = string | URL | Uint8Array | ArrayBuffer | Blob

export type ReadCsvOptions = {
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
  /** Text encoding when reading bytes/files (default utf-8). */
  encoding?: string
  /** Treat a string source as raw CSV content, never as a path. */
  content?: boolean
}

export type ReadJsonOptions = {
  /** `records` = array of objects (default); `columns` = {col: values[]}; `lines` = NDJSON. */
  orient?: 'records' | 'columns' | 'values' | 'lines'
  /** Read newline-delimited JSON (NDJSON). Alias of orient:'lines'. */
  lines?: boolean
  /** Skip first N lines (NDJSON) or N array elements. */
  skipRows?: number
  /** Max records to read. */
  nRows?: number
  encoding?: string
  content?: boolean
}

export type ReadExcelOptions = {
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
  encoding?: string
  content?: boolean
}

export type ReadParquetOptions = {
  /** Column subset by name. */
  columns?: string[]
  /** Inclusive start row (0-based). */
  rowStart?: number
  /** Exclusive end row. */
  rowEnd?: number
  /** Max rows (= rowEnd - rowStart when rowStart set). */
  nRows?: number
  content?: boolean
}
