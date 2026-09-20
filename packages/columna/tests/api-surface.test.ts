/**
 * docs/compatibility.md, enforced: every export and public class member recorded in docs/api-surface.json must
 * still exist. Removing one is a breaking change — update the snapshot deliberately (`pnpm api:snapshot`) in the
 * same commit as the CHANGELOG "Breaking" entry. New names never fail this test; they only show up as a hint.
 */
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { collectSurface, missingNames } from '../../../scripts/api-surface.mjs'

const recorded = JSON.parse(readFileSync(new URL('../../../docs/api-surface.json', import.meta.url), 'utf8')) as {
  entries: Record<string, string[]>
  classes: Record<string, { instance: string[]; static: string[] }>
}

describe('public API surface', () => {
  it('every recorded export and class member still exists (source entry points)', async () => {
    const current = await collectSurface(async (entry: string) => {
      if (entry === 'index') return import('../src/index.js')
      if (entry === 'core') return import('../src/core.js')
      return import('../src/advanced.js')
    })
    const missing = missingNames(recorded, current)
    expect(missing, 'removed from the public API — see docs/compatibility.md').toEqual([])
    const added: string[] = []
    for (const [entry, names] of Object.entries(current.entries as Record<string, string[]>)) {
      const have = new Set(recorded.entries[entry] ?? [])
      for (const n of names) if (!have.has(n)) added.push(`${entry}: ${n}`)
    }
    if (added.length) console.info(`api-surface: ${added.length} new public name(s) not in the snapshot yet — run pnpm api:snapshot:\n  ${added.join('\n  ')}`)
  })

  it('the snapshot covers the umbrella, core and advanced entry points and the public classes', () => {
    expect(Object.keys(recorded.entries).sort()).toEqual(['advanced', 'core', 'index'])
    expect(recorded.entries.index).toEqual(expect.arrayContaining(['DataFrame', 'LazyFrame', 'col', 'createSession', 'init', 'toArrowIpc']))
    expect(recorded.entries.advanced.length).toBeGreaterThan(200)
    expect(recorded.classes.DataFrame.instance).toEqual(expect.arrayContaining(['collect', 'filter', 'groupBy', 'join', 'toArrowIpc', 'withRuntime']))
    expect(recorded.classes.LazyFrame.instance).toEqual(expect.arrayContaining(['collect', 'collectWithReport', 'persist']))
  })
})
