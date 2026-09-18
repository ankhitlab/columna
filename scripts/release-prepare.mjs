/**
 * Prepare a release commit:
 *   node scripts/release-prepare.mjs 0.3.0
 * - sets "version" in package.json and packages/columna/package.json (the only published package),
 * - moves the CHANGELOG "[Unreleased]" entries under "## [0.3.0] - <today>" and leaves an empty [Unreleased],
 * - verifies the section is non-empty.
 * Then: git commit -am "Release 0.3.0" && git tag v0.3.0 && git push --follow-tags  → .github/workflows/release.yml
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const version = process.argv[2]
if (!version || !/^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$/.test(version)) {
  console.error('usage: node scripts/release-prepare.mjs <semver>')
  process.exit(2)
}
const root = new URL('../', import.meta.url)

for (const rel of ['package.json', 'packages/columna/package.json']) {
  const url = new URL(rel, root)
  const raw = readFileSync(url, 'utf8')
  const eol = raw.includes('\r\n') ? '\r\n' : '\n'
  const json = JSON.parse(raw)
  const before = json.version
  json.version = version
  writeFileSync(url, JSON.stringify(json, null, 2).replace(/\n/g, eol) + eol)
  console.log(`${rel}: ${before} → ${version}`)
}

const clUrl = new URL('CHANGELOG.md', root)
const raw = readFileSync(clUrl, 'utf8')
const eol = raw.includes('\r\n') ? '\r\n' : '\n'
const text = raw.replace(/\r\n/g, '\n')
const marker = '## [Unreleased]'
const at = text.indexOf(marker)
if (at < 0) {
  console.error('CHANGELOG.md: no [Unreleased] section')
  process.exit(1)
}
const afterMarker = at + marker.length
const nextHeading = text.indexOf('\n## [', afterMarker)
const body = text.slice(afterMarker, nextHeading < 0 ? text.length : nextHeading).trim()
if (!body) {
  console.error('CHANGELOG.md: [Unreleased] is empty — nothing to release')
  process.exit(1)
}
const today = new Date().toISOString().slice(0, 10)
const rebuilt =
  text.slice(0, at) +
  `${marker}\n\n## [${version}] - ${today}\n\n${body}\n` +
  (nextHeading < 0 ? '' : text.slice(nextHeading))
writeFileSync(clUrl, rebuilt.replace(/\n/g, eol))
console.log(`CHANGELOG.md: [Unreleased] → [${version}] - ${today}`)

const check = spawnSync(process.execPath, [fileURLToPath(new URL('changelog-section.mjs', import.meta.url)), version, '--check'], { stdio: 'inherit' })
process.exit(check.status ?? 1)
