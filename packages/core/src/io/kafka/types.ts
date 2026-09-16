/** How to decode Kafka key / value bytes. `auto` tries UTF-8 JSON, then string. */
export type KafkaCodec = 'json' | 'string' | 'bytes' | 'auto'

export type KafkaSaslOptions = {
  mechanism: 'plain' | 'scram-sha-256' | 'scram-sha-512'
  username: string
  password: string
}

/** One consumed Kafka record before row mapping. */
export type KafkaRawMessage = {
  topic: string
  partition: number
  offset: string
  timestamp: string | number
  key: Buffer | string | null
  value: Buffer | string | null
  headers?: Record<string, Buffer | string | undefined>
}

/**
 * Duck-typed consumer for tests / custom clients.
 * Return already-fetched messages; columna maps them to rows.
 */
export type KafkaClient = {
  consume: (request: {
    topic: string
    nMessages?: number
    maxWaitMs?: number
    fromBeginning?: boolean
  }) => Promise<KafkaRawMessage[]>
  close?: () => void | Promise<void>
}

export type KafkaFlattenOptions = {
  /** Nested path separator (default `.`). */
  separator?: string
  /** Max object nesting depth (default 32). */
  maxDepth?: number
  /** Flatten arrays as `arr.0`, `arr.1`, … (default false → JSON string). */
  arrays?: boolean
}

export type KafkaMetaField = 'topic' | 'partition' | 'offset' | 'timestamp' | 'key' | 'headers'

/** Connection / consume settings for `DataFrame.readKafka`. */
export type ReadKafkaOptions = {
  /** Broker list, e.g. `['localhost:9092']` or `'host1:9092,host2:9092'`. */
  brokers: string[] | string
  /** Topic to consume from. */
  topic: string
  clientId?: string
  /** Consumer group (default `columna-<random>`). */
  groupId?: string
  /** Start from earliest offset (default false → latest, then wait for new). */
  fromBeginning?: boolean
  /** Stop after this many messages (default 1000). */
  nMessages?: number
  /** Max time to wait for messages in ms (default 10_000). */
  maxWaitMs?: number
  /** Decode message value (default `auto` → JSON when possible). */
  valueFormat?: KafkaCodec
  /** Decode message key (default `string`). */
  keyFormat?: KafkaCodec
  /**
   * Flatten nested JSON objects into dotted columns (default `true`).
   * Pass `false` to keep a single `value` column (objects stored as JSON text).
   */
  flatten?: boolean | KafkaFlattenOptions
  /**
   * Include Kafka metadata columns (default `true` → topic, partition, offset, timestamp, key).
   * Prefixed with `metaPrefix` (default `_kafka_`).
   */
  includeMeta?: boolean | KafkaMetaField[]
  /** Prefix for metadata columns (default `_kafka_`). */
  metaPrefix?: string
  ssl?: boolean
  sasl?: KafkaSaslOptions
  /** Keep the underlying KafkaJS consumer open (caller must close via returned client — not used for one-shot API). */
  keepAlive?: boolean
  /**
   * Existing duck-typed client. When set, `brokers` / SSL / SASL are ignored
   * (still require `topic` for the consume request).
   */
  client?: KafkaClient
}

/** URL form: `kafka://host:9092/topic?groupId=…&fromBeginning=true` */
export type KafkaConnection = string | ReadKafkaOptions | KafkaClient
