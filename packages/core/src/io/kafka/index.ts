import { setRowField } from '@columna/arrow'
import { consumeWithKafkaJs, isKafkaClient } from './consumer.js'
import { decodePayload, valueToRowFields } from './decode.js'
import { resolveFlatten } from './flatten.js'
import type {
  KafkaClient,
  KafkaConnection,
  KafkaMetaField,
  KafkaRawMessage,
  ReadKafkaOptions,
} from './types.js'

export type {
  KafkaClient,
  KafkaCodec,
  KafkaConnection,
  KafkaFlattenOptions,
  KafkaMetaField,
  KafkaRawMessage,
  KafkaSaslOptions,
  ReadKafkaOptions,
} from './types.js'
export { flattenObject, resolveFlatten } from './flatten.js'
export { decodePayload, valueToRowFields } from './decode.js'

const DEFAULT_META: KafkaMetaField[] = ['topic', 'partition', 'offset', 'timestamp', 'key']

/**
 * Parse `kafka://host:9092/topic?groupId=g&fromBeginning=true` into options.
 * Multiple brokers: `kafka://b1:9092,b2:9092/topic` is not valid URL host form —
 * use `?brokers=b1:9092,b2:9092` or a config object.
 */
export function parseKafkaUrl(url: string): ReadKafkaOptions {
  const raw = url.trim()
  if (!/^kafka(\+ssl)?:\/\//i.test(raw)) {
    throw new Error(`Expected kafka:// URL, got: ${url}`)
  }
  const ssl = /^kafka\+ssl:/i.test(raw)
  const normalized = raw.replace(/^kafka\+ssl:/i, 'kafka:')
  const u = new URL(normalized)

  const topic = decodeURIComponent(u.pathname.replace(/^\//, ''))
  if (!topic) throw new Error('kafka:// URL must include a topic path, e.g. kafka://localhost:9092/events')

  const brokersParam = u.searchParams.get('brokers')
  const brokers = brokersParam
    ? brokersParam.split(',').map((b) => b.trim()).filter(Boolean)
    : [`${u.hostname}${u.port ? `:${u.port}` : ''}`]

  const fromBeginningRaw = u.searchParams.get('fromBeginning')
  const nMessagesRaw = u.searchParams.get('nMessages')
  const maxWaitMsRaw = u.searchParams.get('maxWaitMs')

  return {
    brokers,
    topic,
    ssl: ssl || u.searchParams.get('ssl') === 'true',
    groupId: u.searchParams.get('groupId') ?? undefined,
    clientId: u.searchParams.get('clientId') ?? undefined,
    fromBeginning: fromBeginningRaw === null ? undefined : fromBeginningRaw === 'true',
    nMessages: nMessagesRaw ? Number(nMessagesRaw) : undefined,
    maxWaitMs: maxWaitMsRaw ? Number(maxWaitMsRaw) : undefined,
  }
}

function resolveOptions(connection: KafkaConnection, options?: Partial<ReadKafkaOptions>): ReadKafkaOptions {
  if (isKafkaClient(connection)) {
    const topic = options?.topic
    if (!topic) throw new Error('readKafka with a custom client requires options.topic')
    return {
      brokers: options?.brokers ?? [],
      topic,
      ...options,
      client: connection,
    }
  }

  if (typeof connection === 'string') {
    return { ...parseKafkaUrl(connection), ...options }
  }

  return { ...connection, ...options }
}

function metaFields(includeMeta: boolean | KafkaMetaField[] | undefined): KafkaMetaField[] {
  if (includeMeta === false) return []
  if (Array.isArray(includeMeta)) return includeMeta
  return DEFAULT_META
}

function headersToJson(headers: KafkaRawMessage['headers']): string | null {
  if (!headers) return null
  const out: Record<string, string | null> = {}
  for (const [k, v] of Object.entries(headers)) {
    if (v === undefined) setRowField(out, k, null)
    else if (typeof v === 'string') setRowField(out, k, v)
    else setRowField(out, k, v.toString('utf8'))
  }
  return JSON.stringify(out)
}

/** Map one Kafka message to a DataFrame row (nested JSON flattened by default). */
export function messageToRow(
  message: KafkaRawMessage,
  options: Pick<
    ReadKafkaOptions,
    'valueFormat' | 'keyFormat' | 'flatten' | 'includeMeta' | 'metaPrefix'
  > = {},
): Record<string, unknown> {
  const valueFormat = options.valueFormat ?? 'auto'
  const keyFormat = options.keyFormat ?? 'string'
  const flatten = resolveFlatten(options.flatten)
  const prefix = options.metaPrefix ?? '_kafka_'

  const decodedValue = decodePayload(message.value, valueFormat)
  const row: Record<string, unknown> = { ...valueToRowFields(decodedValue, flatten) }

  const decodedKey = decodePayload(message.key, keyFormat)
  const keyScalar =
    decodedKey !== null && typeof decodedKey === 'object'
      ? JSON.stringify(decodedKey)
      : decodedKey

  for (const field of metaFields(options.includeMeta)) {
    const col = `${prefix}${field}`
    switch (field) {
      case 'topic':
        row[col] = message.topic
        break
      case 'partition':
        row[col] = message.partition
        break
      case 'offset':
        row[col] = message.offset
        break
      case 'timestamp':
        row[col] = typeof message.timestamp === 'number' ? message.timestamp : Number(message.timestamp)
        break
      case 'key':
        row[col] = keyScalar
        break
      case 'headers':
        row[col] = headersToJson(message.headers)
        break
    }
  }

  return row
}

/**
 * Consume a bounded batch from Kafka and return row objects.
 *
 * `connection` may be:
 * - URL `kafka://host:9092/topic?groupId=…`
 * - `{ brokers, topic, … }` options
 * - duck-typed `{ consume, close? }` client (+ `options.topic`)
 */
export async function readKafkaRows(
  connection: KafkaConnection,
  options: Partial<ReadKafkaOptions> = {},
): Promise<Record<string, unknown>[]> {
  const cfg = resolveOptions(connection, options)
  if (!cfg.topic) throw new Error('readKafka requires a topic')

  let messages: KafkaRawMessage[]
  if (cfg.client) {
    messages = await cfg.client.consume({
      topic: cfg.topic,
      nMessages: cfg.nMessages,
      maxWaitMs: cfg.maxWaitMs,
      fromBeginning: cfg.fromBeginning,
    })
  } else {
    messages = await consumeWithKafkaJs(cfg)
  }

  const limit = cfg.nMessages
  const sliced = limit !== undefined ? messages.slice(0, limit) : messages
  return sliced.map((m) => messageToRow(m, cfg))
}
