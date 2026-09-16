import { loadBytes, loadText } from './source.js'
import { parseCsvToRows } from './csv.js'
import { parseJsonToRows } from './json.js'
import { parseExcelToRows } from './excel.js'
import { parseParquetToRows } from './parquet.js'
import { readSqlRows } from './sql/index.js'
import { readKafkaRows } from './kafka/index.js'
import type { IoSource, ReadCsvOptions, ReadExcelOptions, ReadJsonOptions, ReadParquetOptions } from './types.js'
import type { ReadSqlOptions, SqlConnection } from './sql/types.js'
import type { KafkaConnection, ReadKafkaOptions } from './kafka/types.js'

export type { IoSource, ReadCsvOptions, ReadExcelOptions, ReadJsonOptions, ReadParquetOptions } from './types.js'
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
export { parseCsvToRows, parseCsvLine } from './csv.js'
export { parseJsonToRows } from './json.js'
export { parseExcelToRows } from './excel.js'
export { parseParquetToRows } from './parquet.js'
export { loadBytes, loadText } from './source.js'
export { readSqlRows, inferDialectFromUrl, normalizeDialect } from './sql/index.js'
export { readKafkaRows, parseKafkaUrl, messageToRow, flattenObject, decodePayload } from './kafka/index.js'
export { tableToCsv, writeCsvText, tableToParquetLike, writeParquetBytes } from './write.js'
export { tableToMarkdown, tableToHtml, profileTable } from './present.js'
export type { ProfileReport } from './present.js'

export async function readCsvRows(source: IoSource, options: ReadCsvOptions = {}): Promise<Record<string, unknown>[]> {
  const text = await loadText(source, { content: options.content, encoding: options.encoding })
  return parseCsvToRows(text, options)
}

export async function readJsonRows(source: IoSource, options: ReadJsonOptions = {}): Promise<Record<string, unknown>[]> {
  const text = await loadText(source, { content: options.content, encoding: options.encoding })
  return parseJsonToRows(text, options)
}

export async function readExcelRows(source: IoSource, options: ReadExcelOptions = {}): Promise<Record<string, unknown>[]> {
  const bytes = await loadBytes(source, { content: options.content, encoding: options.encoding })
  return await parseExcelToRows(bytes, options)
}

export async function readParquetRows(
  source: IoSource,
  options: ReadParquetOptions = {},
): Promise<Record<string, unknown>[]> {
  const bytes = await loadBytes(source, { content: options.content })
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