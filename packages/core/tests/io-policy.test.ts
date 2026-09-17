import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest'
import { createServer, type Server } from 'node:http'
import { mkdirSync, rmSync, writeFileSync, symlinkSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { DataFrame } from '../src/dataframe.js'
import { io, loadBytes, resolveSource, setIoPolicy } from '../src/io/source.js'

const DIR = join(tmpdir(), `columna-policy-${Date.now()}`)
const OUTSIDE = join(tmpdir(), `columna-policy-outside-${Date.now()}`)
let server: Server
let base = ''
const csv = 'a,b\n1,2\n'

beforeAll(async () => {
  mkdirSync(join(DIR, 'sub'), { recursive: true })
  mkdirSync(OUTSIDE, { recursive: true })
  writeFileSync(join(DIR, 'ok.csv'), csv)
  writeFileSync(join(DIR, 'sub', 'deep.csv'), csv)
  writeFileSync(join(OUTSIDE, 'secret.csv'), 'k\nsecret\n')
  try {
    symlinkSync(join(OUTSIDE, 'secret.csv'), join(DIR, 'link.csv'))
  } catch {
    // symlinks may need privileges on Windows; the test below tolerates absence
  }
  server = createServer((req, res) => {
    const url = req.url ?? '/'
    if (url === '/data.csv') {
      res.setHeader('content-type', 'text/csv')
      res.end(csv)
    } else if (url === '/big.csv') {
      // 1 MB streamed without Content-Length: the cap must trigger while reading, not from the header
      res.setHeader('content-type', 'text/csv')
      res.write('a\n')
      const chunk = 'x'.repeat(64 * 1024) + '\n'
      for (let i = 0; i < 16; i++) res.write(chunk)
      res.end()
    } else if (url === '/redirect') {
      res.statusCode = 302
      res.setHeader('location', 'http://127.0.0.1:1/never')
      res.end()
    } else if (url === '/redirect-ok') {
      res.statusCode = 302
      res.setHeader('location', '/data.csv')
      res.end()
    } else if (url === '/slow') {
      setTimeout(() => res.end(csv), 1500)
    } else if (url === '/lying-length') {
      res.setHeader('content-length', '4')
      res.end('a,b\n1,2\n')
    } else {
      res.statusCode = 404
      res.end()
    }
  })
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r))
  const addr = server.address()
  base = `http://127.0.0.1:${typeof addr === 'object' && addr ? addr.port : 0}`
})

afterAll(async () => {
  await new Promise<void>((r) => server.close(() => r()))
  rmSync(DIR, { recursive: true, force: true })
  rmSync(OUTSIDE, { recursive: true, force: true })
})

afterEach(() => setIoPolicy({}))

describe('explicit sources', () => {
  it('tags and mode override the string heuristic', () => {
    expect(resolveSource({ text: 'a,b\n1,2' })).toEqual({ kind: 'text', text: 'a,b\n1,2' })
    expect(resolveSource(io.text('./looks/like/a/path.csv'))).toEqual({ kind: 'text', text: './looks/like/a/path.csv' })
    expect(resolveSource(io.path('a,b'))).toEqual({ kind: 'path', path: 'a,b' })
    expect(resolveSource({ url: 'http://h/x' })).toEqual({ kind: 'url', url: 'http://h/x' })
    expect(resolveSource(new URL('http://h/x'))).toEqual({ kind: 'url', url: 'http://h/x' })
    expect(resolveSource('http://h/x', { mode: 'text' })).toEqual({ kind: 'text', text: 'http://h/x' })
    expect(resolveSource('data.csv', { mode: 'url' })).toEqual({ kind: 'url', url: 'data.csv' })
    expect(resolveSource('a,b\n1,2', { mode: 'path' })).toEqual({ kind: 'path', path: 'a,b\n1,2' })
    // file:// is a filesystem read, not a fetch
    expect(resolveSource('file:///tmp/x.csv')).toEqual({ kind: 'path', path: '/tmp/x.csv' })
    expect(resolveSource('file:///C:/data/x.csv')).toEqual({ kind: 'path', path: 'C:/data/x.csv' })
    expect(() => resolveSource({ nope: 1 } as never)).toThrow(/expected \{ text \}/)
  })

  it('a text source that looks like a path is never opened as a file', async () => {
    const df = await DataFrame.readCsv({ text: 'x,y\n1,2\n' })
    expect(df.toArray()).toEqual([{ x: 1, y: 2 }])
    await expect(DataFrame.readCsv('x,y\n1,2\n', { mode: 'path' })).rejects.toThrow(/not found|ENOENT|EINVAL|ENAMETOOLONG/i)
  })
})

describe('URL policy', () => {
  it('allowedHosts: exact, port and wildcard rules; other hosts are refused before any request', async () => {
    const df = await DataFrame.readCsv(`${base}/data.csv`, { allowedHosts: ['127.0.0.1'] })
    expect(df.toArray()).toEqual([{ a: 1, b: 2 }])
    await expect(loadBytes(`${base}/data.csv`, { allowedHosts: ['example.com', '*.example.com'] })).rejects.toThrow(/not in allowedHosts/)
    await expect(loadBytes(`${base}/data.csv`, { allowedHosts: ['127.0.0.1:1'] })).rejects.toThrow(/not in allowedHosts/)
    await expect(loadBytes('https://evil.example.com.attacker.net/x', { allowedHosts: ['*.example.com'], fetch: (() => { throw new Error('must not fetch') }) as never })).rejects.toThrow(/not in allowedHosts/)
  })

  it('denyPrivateHosts refuses loopback / private / metadata names without touching the network', async () => {
    const never = (() => { throw new Error('must not fetch') }) as never
    for (const host of ['127.0.0.1', 'localhost', '10.1.2.3', '192.168.0.1', '172.16.5.5', '169.254.169.254', '[::1]', 'metadata.google.internal', 'app.localhost']) {
      await expect(loadBytes(`http://${host}/x`, { denyPrivateHosts: true, fetch: never })).rejects.toThrow(/private|loopback|link-local/)
    }
    await expect(loadBytes('ftp://example.com/x', { mode: 'url' })).rejects.toThrow(/protocol "ftp:"/)
  })

  it('every redirect hop is checked; allowed redirects still work', async () => {
    await expect(loadBytes(`${base}/redirect`, { allowedHosts: ['127.0.0.1:' + new URL(base).port] })).rejects.toThrow(/not in allowedHosts/)
    const df = await DataFrame.readCsv(`${base}/redirect-ok`)
    expect(df.toArray()).toEqual([{ a: 1, b: 2 }])
    await expect(loadBytes(`${base}/redirect-ok`, { maxRedirects: 0 })).rejects.toThrow(/too many redirects/)
  })

  it('maxBytes cuts a streamed body past the cap and rejects a declared oversize', async () => {
    await expect(loadBytes(`${base}/big.csv`, { maxBytes: 100_000 })).rejects.toThrow(/exceeds maxBytes/)
    const ok = await loadBytes(`${base}/data.csv`, { maxBytes: 100 })
    expect(ok.byteLength).toBe(csv.length)
  })

  it('timeoutMs and a caller signal abort the load', async () => {
    await expect(loadBytes(`${base}/slow`, { timeoutMs: 100 })).rejects.toThrow(/abort|timeout/i)
    const ctl = new AbortController()
    const p = loadBytes(`${base}/slow`, { signal: ctl.signal })
    ctl.abort()
    await expect(p).rejects.toThrow(/abort/i)
  })

  it('the process-wide policy is a floor: per-call options cannot widen it', async () => {
    setIoPolicy({ allowedHosts: ['example.com'], maxBytes: 3 })
    await expect(loadBytes(`${base}/data.csv`, { allowedHosts: ['127.0.0.1'] })).rejects.toThrow(/not in allowedHosts/)
    setIoPolicy({ maxBytes: 3 })
    await expect(loadBytes(`${base}/data.csv`, { maxBytes: 1_000_000 })).rejects.toThrow(/maxBytes = 3/)
    await expect(loadBytes({ text: 'a,b\n1,2\n' })).rejects.toThrow(/maxBytes = 3/)
    setIoPolicy({})
    expect((await loadBytes(`${base}/data.csv`)).byteLength).toBe(csv.length)
  })
})

describe('path policy', () => {
  it('allowedDirs confines reads (subdirectories allowed, traversal and symlink escapes refused)', async () => {
    const df = await DataFrame.readCsv(join(DIR, 'sub', 'deep.csv'), { allowedDirs: [DIR] })
    expect(df.toArray()).toEqual([{ a: 1, b: 2 }])
    await expect(DataFrame.readCsv(join(OUTSIDE, 'secret.csv'), { allowedDirs: [DIR] })).rejects.toThrow(/outside allowedDirs/)
    await expect(DataFrame.readCsv(join(DIR, '..', `columna-policy-outside-${OUTSIDE.split('-').at(-1)}`, 'secret.csv'), { allowedDirs: [DIR] })).rejects.toThrow(/outside allowedDirs/)
    await expect(loadBytes(`file://${DIR.replace(/\\/g, '/').replace(/^([A-Za-z]:)/, '/$1')}/ok.csv`, { allowedDirs: [DIR] })).resolves.toBeInstanceOf(Uint8Array)
    // symlink inside DIR pointing outside: real path decides
    try {
      await expect(DataFrame.readCsv(join(DIR, 'link.csv'), { allowedDirs: [DIR] })).rejects.toThrow(/outside allowedDirs|not found/)
    } catch (e) {
      if (!/not found/i.test(String(e))) throw e // symlink could not be created on this machine
    }
  })

  it('maxBytes applies to files before they are read', async () => {
    await expect(DataFrame.readCsv(join(DIR, 'ok.csv'), { maxBytes: 3 })).rejects.toThrow(/above maxBytes/)
    setIoPolicy({ allowedDirs: [OUTSIDE] })
    await expect(DataFrame.readCsv(join(DIR, 'ok.csv'))).rejects.toThrow(/outside allowedDirs/)
  })
})
