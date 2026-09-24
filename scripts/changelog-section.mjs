/**
 * Print the CHANGELOG.md section for one version (release notes for GitHub Releases).
 *   node scripts/changelog-section.mjs 0.3.0            → the "## [0.3.0] - …" section body
 *   node scripts/changelog-section.mjs 0.3.0 --check    → exit 1 when the section is missing or empty
 *   node scripts/changelog-section.mjs 0.3.0 --links https://github.com/<owner>/<repo>/blob/v0.3.0
 *        → relative links (docs/x.md) rewritten to absolute ones; a release page resolves relative links
 *          against /releases/tag/, where they 404
 * Output is always UTF-8 without a BOM — write it with a UTF-8 aware redirect (bash, or `> file` in PowerShell 7);
 * Windows PowerShell 5.1's `>` re-encodes to UTF-16LE, which is how the v0.3.0 notes got garbled.
 */
import { readFileSync } from 'node:fs'

const version = process.argv[2]
const check = process.argv.includes('--check')
const linksAt = process.argv.indexOf('--links')
const linkBase = linksAt > 0 ? process.argv[linksAt + 1]?.replace(/\/+$/, '') : undefined
if (!version) {
  console.error('usage: node scripts/changelog-section.mjs <version> [--check]')
  process.exit(2)
}
const text = readFileSync(new URL('../CHANGELOG.md', import.meta.url), 'utf8').replace(/\r\n/g, '\n')
const lines = text.split('\n')
const start = lines.findIndex((l) => l.startsWith(`## [${version}]`))
if (start < 0) {
  console.error(`CHANGELOG.md has no "## [${version}]" section (move the [Unreleased] entries under it first)`)
  process.exit(1)
}
let end = lines.length
for (let i = start + 1; i < lines.length; i++) {
  if (lines[i].startsWith('## ')) {
    end = i
    break
  }
}
let body = lines
  .slice(start + 1, end)
  .join('\n')
  .trim()
if (linkBase) {
  // [text](relative/path) → [text](<base>/relative/path); absolute URLs, anchors and mailto: are left alone
  body = body.replace(/\]\((?![a-z][a-z0-9+.-]*:|#|\/)([^)\s]+)\)/gi, (_m, p) => `](${linkBase}/${p.replace(/^\.\//, '')})`)
}
if (check) {
  if (!body) {
    console.error(`CHANGELOG.md section for ${version} is empty`)
    process.exit(1)
  }
  console.log(`CHANGELOG.md: section ${version} ok (${body.split('\n').length} lines)`)
} else {
  process.stdout.write(body + '\n')
}
