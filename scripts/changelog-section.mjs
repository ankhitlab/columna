/**
 * Print the CHANGELOG.md section for one version (release notes for GitHub Releases).
 *   node scripts/changelog-section.mjs 0.3.0            → the "## [0.3.0] - …" section body
 *   node scripts/changelog-section.mjs 0.3.0 --check    → exit 1 when the section is missing or empty
 */
import { readFileSync } from 'node:fs'

const version = process.argv[2]
const check = process.argv.includes('--check')
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
const body = lines
  .slice(start + 1, end)
  .join('\n')
  .trim()
if (check) {
  if (!body) {
    console.error(`CHANGELOG.md section for ${version} is empty`)
    process.exit(1)
  }
  console.log(`CHANGELOG.md: section ${version} ok (${body.split('\n').length} lines)`)
} else {
  console.log(body)
}
