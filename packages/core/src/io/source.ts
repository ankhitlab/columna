import type { IoSource } from './types.js'

function isUrlString(s: string): boolean {
  return /^https?:\/\//i.test(s) || /^file:\/\//i.test(s)
}

function looksLikePath(s: string): boolean {
  if (s.includes('\n') || s.includes('\r')) return false
  if (isUrlString(s)) return true
  if (/[\\/]/.test(s)) return true
  return /\.(csv|tsv|txt|json|jsonl|ndjson|xlsx|xls|parquet|pq)$/i.test(s.trim())
}

async function readNodeFile(path: string): Promise<Uint8Array> {
  try {
    const fs = await import('node:fs/promises')
    return new Uint8Array(await fs.readFile(path))
  } catch (err) {
    const code = err && typeof err === 'object' && 'code' in err ? String((err as { code: unknown }).code) : ''
    if (code === 'ERR_MODULE_NOT_FOUND' || code === 'ERR_UNKNOWN_BUILTIN_MODULE') {
      throw new Error(`Local path IO requires Node.js (got path "${path}")`)
    }
    throw err
  }
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`)
  return new Uint8Array(await res.arrayBuffer())
}

/**
 * Resolve an IO source to bytes.
 * String: URL → fetch; path-like → Node fs; otherwise UTF-8 encode as content
 * (unless `forcePath` and it fails).
 */
export async function loadBytes(
  source: IoSource,
  opts: { content?: boolean; encoding?: string } = {},
): Promise<Uint8Array> {
  if (source instanceof Uint8Array) return source
  if (source instanceof ArrayBuffer) return new Uint8Array(source)
  if (typeof Blob !== 'undefined' && source instanceof Blob) {
    return new Uint8Array(await source.arrayBuffer())
  }
  if (source instanceof URL) return fetchBytes(source.href)

  if (typeof source !== 'string') {
    throw new Error(`Unsupported IO source type: ${typeof source}`)
  }

  if (opts.content) {
    return new TextEncoder().encode(source)
  }

  if (isUrlString(source)) return fetchBytes(source)

  if (looksLikePath(source)) {
    try {
      return await readNodeFile(source)
    } catch (err) {
      // Fall through to content only if it does not look like a filesystem miss
      const code = err && typeof err === 'object' && 'code' in err ? String((err as { code: unknown }).code) : ''
      if (code === 'ENOENT') throw new Error(`File not found: ${source}`)
      throw err
    }
  }

  return new TextEncoder().encode(source)
}

export async function loadText(
  source: IoSource,
  opts: { content?: boolean; encoding?: string } = {},
): Promise<string> {
  if (typeof source === 'string' && (opts.content || !looksLikePath(source))) {
    if (opts.content || !isUrlString(source)) return source
  }
  const bytes = await loadBytes(source, opts)
  return new TextDecoder(opts.encoding ?? 'utf-8').decode(bytes)
}
