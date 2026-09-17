import type { IoLoadOptions, IoPolicy, IoSource, IoSourceMode } from './types.js'

/**
 * Source resolution and the boundary every reader goes through.
 *
 * A bare string is convenient but ambiguous (content? path? URL?), so application code that receives
 * strings from the outside should say what it means — `{ path }`, `{ url }`, `{ text }` (or `mode`) —
 * and set a policy: allowed hosts / directories, byte caps, timeouts, cancellation. The library never
 * decides for you which hosts or directories are safe; without a policy, `readCsv(userString)` will
 * happily fetch any URL or read any file the process can, which is fine for a script and not for a server.
 */

// ---- process-wide policy floor -------------------------------------------------------------------

let globalPolicy: IoPolicy = {}

/**
 * Process-wide IO policy. Every load must satisfy it *and* the per-call options (both are checked), so an
 * application can lock the library down once — e.g. `setIoPolicy({ allowedDirs: ['/data'], maxBytes: 50e6,
 * denyPrivateHosts: true })` — and per-call options can only narrow it further, never widen it.
 */
export function setIoPolicy(policy: IoPolicy): void {
  globalPolicy = { ...policy }
}
export function getIoPolicy(): IoPolicy {
  return { ...globalPolicy }
}

// ---- classification ------------------------------------------------------------------------------

function isUrlString(s: string): boolean {
  return /^https?:\/\//i.test(s) || /^file:\/\//i.test(s)
}

function looksLikePath(s: string): boolean {
  if (s.includes('\n') || s.includes('\r')) return false
  if (isUrlString(s)) return true
  if (/[\\/]/.test(s)) return true
  return /\.(csv|tsv|txt|json|jsonl|ndjson|xlsx|xls|parquet|pq)$/i.test(s.trim())
}

type Resolved = { kind: 'text'; text: string } | { kind: 'path'; path: string } | { kind: 'url'; url: string } | { kind: 'bytes'; bytes: Uint8Array } | { kind: 'blob'; blob: Blob }

/** Decide what a source is: explicit tags win, then `mode`, then the (documented) heuristic for bare strings. */
export function resolveSource(source: IoSource, opts: IoLoadOptions = {}): Resolved {
  if (source instanceof Uint8Array) return { kind: 'bytes', bytes: source }
  if (source instanceof ArrayBuffer) return { kind: 'bytes', bytes: new Uint8Array(source) }
  if (typeof Blob !== 'undefined' && source instanceof Blob) return { kind: 'blob', blob: source }
  if (source instanceof URL) return urlOrFile(source.href)
  if (typeof source === 'object' && source !== null) {
    const tag = source as { text?: unknown; path?: unknown; url?: unknown }
    if (typeof tag.text === 'string') return { kind: 'text', text: tag.text }
    if (typeof tag.path === 'string') return { kind: 'path', path: tag.path }
    if (typeof tag.url === 'string') return urlOrFile(tag.url)
    if (tag.url instanceof URL) return urlOrFile(tag.url.href)
    throw new Error('Unsupported IO source object: expected { text } | { path } | { url }')
  }
  if (typeof source !== 'string') throw new Error(`Unsupported IO source type: ${typeof source}`)

  const mode: IoSourceMode = opts.content ? 'text' : (opts.mode ?? 'auto')
  switch (mode) {
    case 'text':
      return { kind: 'text', text: source }
    case 'path':
      return { kind: 'path', path: source }
    case 'url':
      return urlOrFile(source)
    case 'auto':
      if (isUrlString(source)) return urlOrFile(source)
      if (looksLikePath(source)) return { kind: 'path', path: source }
      return { kind: 'text', text: source }
  }
}

/** `file://` URLs are filesystem reads and go through the path policy, never through fetch. */
function urlOrFile(href: string): Resolved {
  if (/^file:\/\//i.test(href)) return { kind: 'path', path: fileUrlToPath(href) }
  return { kind: 'url', url: href }
}

function fileUrlToPath(href: string): string {
  const u = new URL(href)
  let p = decodeURIComponent(u.pathname)
  // file:///C:/x on Windows → C:/x
  if (/^\/[A-Za-z]:[\\/]/.test(p)) p = p.slice(1)
  return p
}

// ---- policy checks -------------------------------------------------------------------------------

const PRIVATE_HOST = /^(localhost|.*\.localhost|0\.0\.0\.0|127(\.\d{1,3}){3}|10(\.\d{1,3}){3}|192\.168(\.\d{1,3}){2}|172\.(1[6-9]|2\d|3[01])(\.\d{1,3}){2}|169\.254(\.\d{1,3}){2}|100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])(\.\d{1,3}){2}|\[?::1\]?|\[?fc[0-9a-f]{2}:.*|\[?fd[0-9a-f]{2}:.*|\[?fe80:.*|metadata\.google\.internal)$/i

function hostAllowed(host: string, port: string, allowed: string[]): boolean {
  const h = host.toLowerCase()
  return allowed.some((rule) => {
    const r = rule.toLowerCase()
    if (r.startsWith('*.')) return h.endsWith(r.slice(1)) && h.length > r.length - 1
    if (r.includes(':')) return `${h}:${port}` === r
    return h === r
  })
}

function checkUrlPolicy(url: URL, policy: IoPolicy): void {
  const protocols = policy.allowedProtocols ?? ['http:', 'https:']
  if (!protocols.includes(url.protocol as 'http:' | 'https:')) {
    throw new Error(`IO policy: protocol "${url.protocol}" is not allowed for ${url.href}`)
  }
  if (url.username || url.password) throw new Error(`IO policy: credentials in URLs are not allowed (${url.host})`)
  if (policy.denyPrivateHosts && PRIVATE_HOST.test(url.hostname)) {
    throw new Error(`IO policy: host "${url.hostname}" is a loopback / private / link-local address`)
  }
  if (policy.allowedHosts && !hostAllowed(url.hostname, url.port || defaultPort(url.protocol), policy.allowedHosts)) {
    throw new Error(`IO policy: host "${url.hostname}" is not in allowedHosts`)
  }
}

function defaultPort(protocol: string): string {
  return protocol === 'https:' ? '443' : protocol === 'http:' ? '80' : ''
}

async function checkPathPolicy(path: string, policy: IoPolicy): Promise<void> {
  if (!policy.allowedDirs) return
  const nodePath = await import('node:path')
  const fs = await import('node:fs/promises')
  // Compare real paths so symlinks cannot point outside an allowed directory.
  const real = await fs.realpath(nodePath.resolve(path)).catch(() => nodePath.resolve(path))
  const ok = await Promise.all(
    policy.allowedDirs.map(async (dir) => {
      const base = await fs.realpath(nodePath.resolve(dir)).catch(() => nodePath.resolve(dir))
      const rel = nodePath.relative(base, real)
      return rel === '' || (!rel.startsWith('..') && !nodePath.isAbsolute(rel))
    }),
  )
  if (!ok.some(Boolean)) throw new Error(`IO policy: path "${path}" is outside allowedDirs`)
}

function effectiveMaxBytes(a: IoPolicy, b: IoPolicy): number | undefined {
  const xs = [a.maxBytes, b.maxBytes].filter((v): v is number => typeof v === 'number' && v >= 0)
  return xs.length ? Math.min(...xs) : undefined
}

function combinedSignal(opts: IoLoadOptions): AbortSignal | undefined {
  const signals: AbortSignal[] = []
  if (opts.signal) signals.push(opts.signal)
  if (globalPolicy.signal) signals.push(globalPolicy.signal)
  const timeouts = [opts.timeoutMs, globalPolicy.timeoutMs].filter((v): v is number => typeof v === 'number' && v > 0)
  if (timeouts.length) signals.push(AbortSignal.timeout(Math.min(...timeouts)))
  if (signals.length === 0) return undefined
  if (signals.length === 1) return signals[0]
  if (typeof AbortSignal.any === 'function') return AbortSignal.any(signals)
  const ctl = new AbortController()
  for (const s of signals) {
    if (s.aborted) ctl.abort(s.reason)
    else s.addEventListener('abort', () => ctl.abort(s.reason), { once: true })
  }
  return ctl.signal
}

/** Path policy (allowedDirs from both the process policy and the call) plus the maxBytes stat check. */
export async function checkPathAllowed(path: string, opts: IoLoadOptions): Promise<void> {
  await checkPathPolicy(path, globalPolicy)
  await checkPathPolicy(path, opts)
  const maxBytes = effectiveMaxBytes(globalPolicy, opts)
  if (maxBytes !== undefined) {
    const fs = await import('node:fs/promises')
    const st = await fs.stat(path)
    if (st.size > maxBytes) throw new Error(`IO policy: file "${path}" is ${st.size} bytes, above maxBytes = ${maxBytes}`)
  }
}

// ---- loading -------------------------------------------------------------------------------------

async function readNodeFile(path: string, opts: IoLoadOptions): Promise<Uint8Array> {
  await checkPathPolicy(path, globalPolicy)
  await checkPathPolicy(path, opts)
  const maxBytes = effectiveMaxBytes(globalPolicy, opts)
  let fs: typeof import('node:fs/promises')
  try {
    fs = await import('node:fs/promises')
  } catch (err) {
    const code = err && typeof err === 'object' && 'code' in err ? String((err as { code: unknown }).code) : ''
    if (code === 'ERR_MODULE_NOT_FOUND' || code === 'ERR_UNKNOWN_BUILTIN_MODULE') {
      throw new Error(`Local path IO requires Node.js (got path "${path}")`)
    }
    throw err
  }
  if (maxBytes !== undefined) {
    const st = await fs.stat(path)
    if (st.size > maxBytes) throw new Error(`IO policy: file "${path}" is ${st.size} bytes, above maxBytes = ${maxBytes}`)
  }
  return new Uint8Array(await fs.readFile(path, { signal: combinedSignal(opts) }))
}

async function fetchBytes(href: string, opts: IoLoadOptions): Promise<Uint8Array> {
  const fetchImpl = opts.fetch ?? globalPolicy.fetch ?? fetch
  const maxRedirects = opts.maxRedirects ?? globalPolicy.maxRedirects ?? 5
  const maxBytes = effectiveMaxBytes(globalPolicy, opts)
  const signal = combinedSignal(opts)
  let url = new URL(href)
  // Redirects are followed by hand so that every hop — not only the first URL — passes the policy.
  for (let hop = 0; ; hop++) {
    checkUrlPolicy(url, globalPolicy)
    checkUrlPolicy(url, opts)
    const res = await fetchImpl(url.href, { redirect: 'manual', signal })
    if (res.status >= 300 && res.status < 400 && res.headers.get('location')) {
      if (hop >= maxRedirects) throw new Error(`IO policy: too many redirects fetching ${href}`)
      url = new URL(res.headers.get('location')!, url)
      continue
    }
    if (!res.ok) throw new Error(`Failed to fetch ${url.href}: ${res.status} ${res.statusText}`)
    const declared = Number(res.headers.get('content-length') ?? NaN)
    if (maxBytes !== undefined && declared > maxBytes) {
      throw new Error(`IO policy: ${url.href} declares ${declared} bytes, above maxBytes = ${maxBytes}`)
    }
    return readBodyCapped(res, maxBytes, url.href)
  }
}

/** Read a response body incrementally and stop as soon as the cap is exceeded (Content-Length can lie). */
async function readBodyCapped(res: Response, maxBytes: number | undefined, href: string): Promise<Uint8Array> {
  if (maxBytes === undefined || !res.body) return new Uint8Array(await res.arrayBuffer())
  const reader = res.body.getReader()
  const chunks: Uint8Array[] = []
  let total = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    total += value.byteLength
    if (total > maxBytes) {
      await reader.cancel().catch(() => undefined)
      throw new Error(`IO policy: ${href} exceeds maxBytes = ${maxBytes}`)
    }
    chunks.push(value)
  }
  const out = new Uint8Array(total)
  let off = 0
  for (const c of chunks) {
    out.set(c, off)
    off += c.byteLength
  }
  return out
}

/**
 * Resolve an IO source to bytes.
 * Explicit sources: `{ text }`, `{ path }`, `{ url }`, `URL`, bytes, `Blob`. A bare string is classified by
 * `mode` ('text' | 'path' | 'url'), else heuristically: http(s):// → fetch; path-like → Node fs (file:// too);
 * otherwise UTF-8 content. Policy (`allowedHosts`, `allowedDirs`, `maxBytes`, `timeoutMs`, `signal`, …) is
 * applied from both `setIoPolicy()` and the call.
 */
export async function loadBytes(source: IoSource, opts: IoLoadOptions = {}): Promise<Uint8Array> {
  const r = resolveSource(source, opts)
  const maxBytes = effectiveMaxBytes(globalPolicy, opts)
  switch (r.kind) {
    case 'bytes':
      if (maxBytes !== undefined && r.bytes.byteLength > maxBytes) throw new Error(`IO policy: input is ${r.bytes.byteLength} bytes, above maxBytes = ${maxBytes}`)
      return r.bytes
    case 'blob':
      if (maxBytes !== undefined && r.blob.size > maxBytes) throw new Error(`IO policy: blob is ${r.blob.size} bytes, above maxBytes = ${maxBytes}`)
      return new Uint8Array(await r.blob.arrayBuffer())
    case 'text': {
      const bytes = new TextEncoder().encode(r.text)
      if (maxBytes !== undefined && bytes.byteLength > maxBytes) throw new Error(`IO policy: text is ${bytes.byteLength} bytes, above maxBytes = ${maxBytes}`)
      return bytes
    }
    case 'url':
      return fetchBytes(r.url, opts)
    case 'path':
      try {
        return await readNodeFile(r.path, opts)
      } catch (err) {
        const code = err && typeof err === 'object' && 'code' in err ? String((err as { code: unknown }).code) : ''
        if (code === 'ENOENT') throw new Error(`File not found: ${r.path}`)
        throw err
      }
  }
}

export async function loadText(source: IoSource, opts: IoLoadOptions = {}): Promise<string> {
  const r = resolveSource(source, opts)
  if (r.kind === 'text') {
    const maxBytes = effectiveMaxBytes(globalPolicy, opts)
    if (maxBytes !== undefined && r.text.length > maxBytes) {
      // length ≤ bytes: a fast reject; the exact byte count is checked in loadBytes when needed
      throw new Error(`IO policy: text is longer than maxBytes = ${maxBytes}`)
    }
    return r.text
  }
  const bytes = await loadBytes(source, opts)
  return new TextDecoder(opts.encoding ?? 'utf-8').decode(bytes)
}

/** Explicit source constructors — say what a string is instead of letting the reader guess. */
export const io = {
  text: (text: string): { text: string } => ({ text }),
  path: (path: string): { path: string } => ({ path }),
  url: (url: string | URL): { url: string | URL } => ({ url }),
} as const
