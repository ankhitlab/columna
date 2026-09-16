import type { KafkaFlattenOptions } from './types.js'

export type FlattenConfig = {
  separator: string
  maxDepth: number
  arrays: boolean
}

export function resolveFlatten(flatten: boolean | KafkaFlattenOptions | undefined): FlattenConfig | null {
  if (flatten === false) return null
  const opts = flatten === true || flatten === undefined ? {} : flatten
  return {
    separator: opts.separator ?? '.',
    maxDepth: opts.maxDepth ?? 32,
    arrays: opts.arrays ?? false,
  }
}

/**
 * Flatten nested plain objects into dotted keys.
 * Arrays become JSON strings unless `arrays: true` (then `arr.0`, `arr.1`, …).
 */
export function flattenObject(
  input: unknown,
  config: FlattenConfig,
  prefix = '',
  depth = 0,
  out: Record<string, unknown> = {},
): Record<string, unknown> {
  if (input === null || input === undefined) {
    if (prefix) out[prefix] = null
    return out
  }

  if (depth >= config.maxDepth) {
    out[prefix || 'value'] = stableJson(input)
    return out
  }

  if (Array.isArray(input)) {
    if (!config.arrays) {
      out[prefix || 'value'] = stableJson(input)
      return out
    }
    if (input.length === 0) {
      out[prefix || 'value'] = null
      return out
    }
    for (let i = 0; i < input.length; i++) {
      const key = prefix ? `${prefix}${config.separator}${i}` : String(i)
      flattenObject(input[i], config, key, depth + 1, out)
    }
    return out
  }

  if (isPlainObject(input)) {
    const entries = Object.entries(input)
    if (entries.length === 0) {
      if (prefix) out[prefix] = null
      return out
    }
    for (const [k, v] of entries) {
      const key = prefix ? `${prefix}${config.separator}${k}` : k
      flattenObject(v, config, key, depth + 1, out)
    }
    return out
  }

  out[prefix || 'value'] = normalizeLeaf(input)
  return out
}

export function isPlainObject(v: unknown): v is Record<string, unknown> {
  if (v === null || typeof v !== 'object') return false
  if (Array.isArray(v)) return false
  if (v instanceof Date) return false
  if (typeof Buffer !== 'undefined' && Buffer.isBuffer(v)) return false
  const proto = Object.getPrototypeOf(v)
  return proto === Object.prototype || proto === null
}

export function normalizeLeaf(v: unknown): unknown {
  if (typeof v === 'bigint') {
    const n = Number(v)
    return Number.isSafeInteger(n) ? n : v.toString()
  }
  if (v instanceof Date) return v.getTime()
  if (typeof Buffer !== 'undefined' && Buffer.isBuffer(v)) return v.toString('utf8')
  return v
}

export function stableJson(v: unknown): string {
  return JSON.stringify(v)
}
