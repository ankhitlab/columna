import { loadBytes, loadText, resolveSource, getIoPolicy } from './source.js'
import { parseCsvToTable, streamCsvFileToTable } from './csv-columnar.js'
import type { TableView } from '@columna/arrow'
import { parseCsvToRows } from './csv.js'
import { parseJsonToRows } from './json.js'
import { parseExcelToRows } from './excel.js'
import { parseParquetToRows } from './parquet.js'
import { readSqlRows } from './sql/index.js'
import { readKafkaRows } from './kafka/index.js'
import type { IoSource, ReadCsvOptions, ReadExcelOptions, ReadJsonOptions, ReadParquetOptions } from './types.js'
import type { ReadSqlOptions, SqlConnection } from './sql/types.js'
import type { KafkaConnection, ReadKafkaOptions } from './kafka/types.js'

export type { IoSource, IoLoadOptions, ReadCsvOptions, ReadExcelOptions, ReadJsonOptions, ReadParquetOptions } from './types.js'
export type {
  ReadSqlOptions,
  SqlClient,
  SqlConnection,
  SqlConnectionConfig,
  SqlDialect,
  SqlParams,
} from './sql/types.js'
export type {
  KafkaClient,
  KafkaCodec,
  KafkaConnection,
  KafkaFlattenOptions,
  KafkaMetaField,
  KafkaRawMessage,
  KafkaSaslOptions,
  ReadKafkaOptions,
} from './kafka/types.js'
export { parseCsvToRows, parseCsvLine, parseCsvLineInto } from './csv.js'
export { parseJsonToRows } from './json.js'
export { parseExcelToRows } from './excel.js'
export { parseParquetToRows } from './parquet.js'
export { loadBytes, loadText, resolveSource, setIoPolicy, getIoPolicy, io } from './source.js'
export { parseCsvToTable, streamCsvFileToTable, CsvTableBuilder, CsvRecordSplitter } from './csv-columnar.js'
export { readSqlRows, openSqlClient, inferDialectFromUrl, normalizeDialect } from './sql/index.js'
export { readKafkaRows, parseKafkaUrl, messageToRow, flattenObject, decodePayload } from './kafka/index.js'
export { tableToCsv, writeCsvText, tableToParquetLike, writeParquetBytes, writeParquetLikeBytes, type CsvWriteOptions } from './write.js'
export { tableToMarkdown, tableToHtml, profileTable } from './present.js'
export type { ProfileReport } from './present.js'

export async function readCsvRows(source: IoSource, options: ReadCsvOptions = {}): Promise<Record<string, unknown>[]> {
  const text = await loadText(source, options)
  return parseCsvToRows(text, options)
}

/**
 * CSV → columnar table without row objects. A Node path is streamed chunk by chunk (peak memory: the
 * columns plus one 1 MB chunk); other sources are loaded through the IO policy and parsed from memory.
 */
export async function readCsvTable(source: IoSource, options: ReadCsvOptions = {}): Promise<TableView> {
  const r = resolveSource(source, options)
  if (r.kind === 'path' && typeof process !== 'undefined' && Boolean(process.versions?.node)) {
    // policy: allowedDirs / maxBytes / signal apply here exactly as in loadBytes
    await loadBytesPolicyOnly(r.path, options)
    const global = getIoPolicy()
    const caps = [options.maxBytes, global.maxBytes].filter((v): v is number => typeof v === 'number')
    try {
      return await streamCsvFileToTable(r.path, { ...options, maxBytes: caps.length ? Math.min(...caps) : undefined })
    } catch (err) {
      const code = err && typeof err === 'object' && 'code' in err ? String((err as { code: unknown }).code) : ''
      if (code === 'ENOENT') throw new Error(`File not found: ${r.path}`)
      throw err
    }
  }
  return parseCsvToTable(await loadText(source, options), options)
}

/** Run the path policy checks (allowedDirs, maxBytes via stat) without reading the file. */
async function loadBytesPolicyOnly(path: string, options: ReadCsvOptions): Promise<void> {
  const { checkPathAllowed } = await import('./source.js')
  await checkPathAllowed(path, options)
}

export async function readJsonRows(source: IoSource, options: ReadJsonOptions = {}): Promise<Record<string, unknown>[]> {
  const text = await loadText(source, options)
  return parseJsonToRows(text, options)
}

export async function readExcelRows(source: IoSource, options: ReadExcelOptions = {}): Promise<Record<string, unknown>[]> {
  const bytes = await loadBytes(source, options)
  return await parseExcelToRows(bytes, options)
}

export async function readParquetRows(
  source: IoSource,
  options: ReadParquetOptions = {},
): Promise<Record<string, unknown>[]> {
  const bytes = await loadBytes(source, options)
  return parseParquetToRows(bytes, options)
}

export async function readDatabaseRows(
  sql: string,
  connection: SqlConnection,
  options: ReadSqlOptions = {},
): Promise<Record<string, unknown>[]> {
  return readSqlRows(sql, connection, options)
}

export async function readKafkaBatch(
  connection: KafkaConnection,
  options: Partial<ReadKafkaOptions> = {},
): Promise<Record<string, unknown>[]> {
  return readKafkaRows(connection, options)
}