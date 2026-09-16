import type { KafkaClient, KafkaRawMessage, KafkaSaslOptions, ReadKafkaOptions } from './types.js'

type KafkaJsModule = {
  Kafka: new (config: {
    clientId: string
    brokers: string[]
    ssl?: boolean
    sasl?: {
      mechanism: string
      username: string
      password: string
    }
  }) => {
    consumer: (config: { groupId: string }) => {
      connect(): Promise<void>
      subscribe(config: { topic: string; fromBeginning?: boolean }): Promise<void>
      run(config: {
        eachMessage: (payload: {
          topic: string
          partition: number
          message: {
            offset: string
            timestamp: string
            key: Buffer | null
            value: Buffer | null
            headers?: Record<string, Buffer | undefined>
          }
        }) => Promise<void>
      }): Promise<void>
      disconnect(): Promise<void>
      stop(): Promise<void>
    }
  }
}

function missingDriverMessage(): string {
  return 'Kafka support requires the optional "kafkajs" package. Install it with: pnpm add kafkajs'
}

async function importKafkaJs(): Promise<KafkaJsModule> {
  try {
    return (await import('kafkajs')) as KafkaJsModule
  } catch {
    throw new Error(missingDriverMessage())
  }
}

function asBrokerList(brokers: string[] | string): string[] {
  if (Array.isArray(brokers)) return brokers.map((b) => b.trim()).filter(Boolean)
  return brokers
    .split(',')
    .map((b) => b.trim())
    .filter(Boolean)
}

function saslConfig(sasl: KafkaSaslOptions | undefined):
  | { mechanism: string; username: string; password: string }
  | undefined {
  if (!sasl) return undefined
  return {
    mechanism: sasl.mechanism,
    username: sasl.username,
    password: sasl.password,
  }
}

/** Consume a bounded batch via KafkaJS, then disconnect. */
export async function consumeWithKafkaJs(options: ReadKafkaOptions): Promise<KafkaRawMessage[]> {
  const mod = await importKafkaJs()
  const brokers = asBrokerList(options.brokers)
  if (brokers.length === 0) throw new Error('readKafka requires at least one broker')

  const nMessages = options.nMessages ?? 1000
  const maxWaitMs = options.maxWaitMs ?? 10_000
  const groupId = options.groupId ?? `columna-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  const clientId = options.clientId ?? 'columna'

  const kafka = new mod.Kafka({
    clientId,
    brokers,
    ssl: options.ssl,
    sasl: saslConfig(options.sasl),
  })

  const consumer = kafka.consumer({ groupId })
  const collected: KafkaRawMessage[] = []
  let settled = false

  const finish = async () => {
    if (settled) return
    settled = true
    try {
      await consumer.stop()
    } catch {
      /* ignore */
    }
    try {
      await consumer.disconnect()
    } catch {
      /* ignore */
    }
  }

  await consumer.connect()
  await consumer.subscribe({ topic: options.topic, fromBeginning: options.fromBeginning ?? false })

  const done = new Promise<KafkaRawMessage[]>((resolve, reject) => {
    const timer = setTimeout(() => {
      void finish().then(() => resolve(collected)).catch(reject)
    }, maxWaitMs)

    void consumer
      .run({
        eachMessage: async ({ topic, partition, message }) => {
          if (settled) return
          collected.push({
            topic,
            partition,
            offset: message.offset,
            timestamp: message.timestamp,
            key: message.key,
            value: message.value,
            headers: message.headers as Record<string, Buffer | string | undefined> | undefined,
          })
          if (collected.length >= nMessages) {
            clearTimeout(timer)
            await finish()
            resolve(collected)
          }
        },
      })
      .catch((err: unknown) => {
        clearTimeout(timer)
        settled = true
        reject(err instanceof Error ? err : new Error(String(err)))
      })
  })

  return done
}

export function isKafkaClient(v: unknown): v is KafkaClient {
  return !!v && typeof v === 'object' && typeof (v as KafkaClient).consume === 'function'
}
