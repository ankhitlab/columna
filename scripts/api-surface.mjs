/**
 * Public API surface snapshot — the machine-checked half of docs/compatibility.md.
 *
 *   pnpm build && node scripts/api-surface.mjs           # rewrite docs/api-surface.json from the built package
 *   node scripts/api-surface.mjs --check                 # exit 1 if anything recorded is gone (CI does this
 *                                                        #  through packages/columna/tests/api-surface.test.ts)
 *
 * Recorded: every export of `columna`, `columna/core`, `columna/advanced`, and every method / accessor of the
 * public classes. Removing or renaming a recorded name is a breaking change and must be a deliberate edit of
 * the snapshot in the same commit as the CHANGELOG entry. Additions never fail the check.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { resolve } from 'node:path'

const root = fileURLToPath(new URL('..', import.meta.url))
const dist = resolve(root, 'packages/columna/dist')
const out = resolve(root, 'docs/api-surface.json')

export async function collectSurface(load = (entry) => import(pathToFileURL(resolve(dist, entry + '.js')).href)) {
  const surface = { entries: {}, classes: {} }
  for (const entry of ['index', 'core', 'advanced']) {
    const mod = await load(entry)
    surface.entries[entry] = Object.keys(mod)
      .filter((k) => k !== 'default')
      .sort()
  }
  const main = await load('index')
  for (const name of ['DataFrame', 'LazyFrame', 'GroupBy', 'Series', 'Expr', 'Session', 'Runtime', 'PersistCache']) {
    const cls = main[name]
    if (typeof cls !== 'function') continue
    const members = (o) => Object.getOwnPropertyNames(o).filter((n) => !['constructor', 'length', 'name', 'prototype'].includes(n) && !n.startsWith('__') && !n.startsWith('_'))
    surface.classes[name] = { instance: members(cls.prototype).sort(), static: members(cls).sort() }
  }
  return surface
}

/** Names present in `recorded` but missing from `current`, as "where: name" strings. */
export function missingNames(recorded, current) {
  const missing = []
  for (const [entry, names] of Object.entries(recorded.entries ?? {})) {
    const have = new Set(current.entries?.[entry] ?? [])
    for (const n of names) if (!have.has(n)) missing.push(`${entry}: ${n}`)
  }
  for (const [cls, sides] of Object.entries(recorded.classes ?? {})) {
    for (const side of ['instance', 'static']) {
      const have = new Set(current.classes?.[cls]?.[side] ?? [])
      for (const n of sides[side] ?? []) if (!have.has(n)) missing.push(`${cls}.${side === 'static' ? '' : 'prototype.'}${n}`)
    }
  }
  return missing
}

const isMain = process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url
if (isMain) {
  const current = await collectSurface()
  if (process.argv.includes('--check')) {
    const recorded = JSON.parse(readFileSync(out, 'utf8'))
    const missing = missingNames(recorded, current)
    if (missing.length) {
      console.error('public API surface: recorded names are missing from the build:\n  ' + missing.join('\n  '))
      process.exit(1)
    }
    console.log('public API surface: ok')
  } else {
    writeFileSync(out, JSON.stringify(current, null, 2) + '\n')
    const n = Object.values(current.entries).reduce((s, a) => s + a.length, 0)
    console.log(`docs/api-surface.json: ${n} exports, ${Object.keys(current.classes).length} classes`)
  }
}
