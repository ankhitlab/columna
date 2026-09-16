import { describe, expect, it } from 'vitest'
import { DataFrame } from '../src/dataframe.js'
import {
  decodePayload,
  flattenObject,
  messageToRow,
  parseKafkaUrl,
  readKafkaRows,
  resolveFlatten,
} from '../src/io/kafka/index.js'
import type { KafkaClient, KafkaRawMessage } from '../src/io/kafka/types.js'

describe('kafka nested JSON flatten', () => {
  it('flattens nested objects with dotted paths', () => {
    const flat = flattenObject(
      {
        user: { id: 1, profile: { city: 'NY', tags: ['a', 'b'] } },
        ok: true,
      },
      resolveFlatten(true)!,
    )
    expect(flat).toEqual({
      'user.id': 1,
      'user.profile.city': 'NY',
      'user.profile.tags': '["a","b"]',
      ok: true,
    })
  })

  it('can flatten arrays by index', () => {
    const flat = flattenObject({ items: [{ x: 1 }, { x: 2 }] }, resolveFlatten({ arrays: true })!)
    expect(flat).toEqual({
      'items.0.x': 1,
      'items.1.x': 2,
    })
  })
})

describe('kafka decode', () => {
  it('auto-parses JSON values', () => {
    expect(decodePayload(Buffer.from('{"a":{"b":2}}'), 'auto')).toEqual({ a: { b: 2 } })
    expect(decodePayload('not-json', 'auto')).toBe('not-json')
  })
})

describe('messageToRow', () => {
  it('maps nested JSON payload + metadata', () => {
    const msg: KafkaRawMessage = {
      topic: 'events',
      partition: 0,
      offset: '42',
      timestamp: '1700000000000',
      key: Buffer.from('k1'),
      value: Buffer.from(JSON.stringify({ user: { name: 'Ada' }, n: 3 })),
    }
    const row = messageToRow(msg)
    expect(row).toMatchObject({
      'user.name': 'Ada',
      n: 3,
      _kafka_topic: 'events',
      _kafka_partition: 0,
      _kafka_offset: '42',
      _kafka_timestamp: 1700000000000,
      _kafka_key: 'k1',
    })
  })

  it('keeps value column when flatten is false', () => {
    const msg: KafkaRawMessage = {
      topic: 't',
      partition: 1,
      offset: '1',
      timestamp: 1,
      key: null,
      value: '{"x":1}',
    }
    const row = messageToRow(msg, { flatten: false, includeMeta: false })
    expect(row).toEqual({ value: '{"x":1}' })
  })
})

describe('parseKafkaUrl', () => {
  it('parses broker, topic and query', () => {
    expect(parseKafkaUrl('kafka://localhost:9092/events?groupId=g1&fromBeginning=true')).toEqual({
      brokers: ['localhost:9092'],
      topic: 'events',
      ssl: false,
      groupId: 'g1',
      clientId: undefined,
      fromBeginning: true,
      nMessages: undefined,
      maxWaitMs: undefined,
    })
  })
})

describe('DataFrame.readKafka', () => {
  it('reads via duck-typed client and flattens nested JSON', async () => {
    const client: KafkaClient = {
      async consume({ topic }) {
        expect(topic).toBe('events')
        return [
          {
            topic,
            partition: 0,
            offset: '0',
            timestamp: '1',
            key: null,
            value: Buffer.from(
              JSON.stringify({
                event: 'click',
                meta: { device: { os: 'ios' }, ip: '1.2.3.4' },
              }),
            ),
          },
        ]
      },
    }
    const df = await DataFrame.readKafka(client, { topic: 'events', includeMeta: ['partition'] })
    expect(df.toArray()).toEqual([
      {
        event: 'click',
        'meta.device.os': 'ios',
        'meta.ip': '1.2.3.4',
        _kafka_partition: 0,
      },
    ])
  })

  it('honours nRows via nMessages on client path', async () => {
    const client: KafkaClient = {
      async consume() {
        return [
          { topic: 't', partition: 0, offset: '0', timestamp: 1, key: null, value: '{"a":1}' },
          { topic: 't', partition: 0, offset: '1', timestamp: 2, key: null, value: '{"a":2}' },
          { topic: 't', partition: 0, offset: '2', timestamp: 3, key: null, value: '{"a":3}' },
        ]
      },
    }
    const rows = await readKafkaRows(client, { topic: 't', nMessages: 2, includeMeta: false })
    expect(rows).toHaveLength(2)
    expect(rows.map((r) => r.a)).toEqual([1, 2])
  })

  it('errors clearly when kafkajs is missing', async () => {
    let hasKafka = false
    try {
      await import('kafkajs')
      hasKafka = true
    } catch {
      // optional peer not installed
    }
    if (hasKafka) return
    await expect(
      DataFrame.readKafka({ brokers: ['localhost:9092'], topic: 't', maxWaitMs: 50 }),
    ).rejects.toThrow(/pnpm add kafkajs/)
  })
})
