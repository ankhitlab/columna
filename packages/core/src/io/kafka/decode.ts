import type { KafkaCodec } from './types.js'
import { flattenObject, isPlainObject, normalizeLeaf, type FlattenConfig } from './flatten.js'

export function bytesToUtf8(data: Buffer | string | null | undefined): string | null {
  if (data === null || data === undefined) return null
  if (typeof data === 'string') return data
  return data.toString('utf8')
}

/**
 * Decode Kafka key/value bytes according to codec.
 * - `json`: parse UTF-8 JSON (null if empty)
 * - `string`: UTF-8 string
 * - `bytes`: raw Buffer (or string as-is)
 * - `auto`: try JSON object/array/primitive; fall back to string
 */
export function decodePayload(
  data: Buffer | string | null | undefined,
  codec: KafkaCodec,
): unknown {
  if (data === null || data === undefined) return null

  if (codec === 'bytes') {
    if (typeof data === 'string') return Buffer.from(data)
    return data
  }

  const text = bytesToUtf8(data)
  if (text === null) return null

  if (codec === 'string') return text

  if (codec === 'json') {
    if (!text.trim()) return null
    return JSON.parse(text) as unknown
  }

  // auto
  const trimmed = text.trim()
  if (!trimmed) return null
  if (
    trimmed.startsWith('{') ||
    trimmed.startsWith('[') ||
    trimmed === 'null' ||
    trimmed === 'true' ||
    trimmed === 'false' ||
    /^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(trimmed) ||
    (trimmed.startsWith('"') && trimmed.endsWith('"'))
  ) {
    try {
      return JSON.parse(trimmed) as unknown
    } catch {
      return text
    }
  }
  return text
}

/** Turn a decoded value into row fields (optionally flattened). */
export function valueToRowFields(value: unknown, flatten: FlattenConfig | null): Record<string, unknown> {
  if (value === null || value === undefined) {
    return { value: null }
  }

  if (!flatten) {
    if (isPlainObject(value) || Array.isArray(value)) {
      return { value: JSON.stringify(value) }
    }
    return { value: normalizeLeaf(value) }
  }

  if (isPlainObject(value)) {
    return flattenObject(value, flatten)
  }

  if (Array.isArray(value)) {
    if (flatten.arrays) return flattenObject(value, flatten)
    return { value: JSON.stringify(value) }
  }

  return { value: normalizeLeaf(value) }
}
