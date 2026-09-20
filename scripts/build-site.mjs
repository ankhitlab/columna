/**
 * Versioned documentation site (what the `pages` workflow publishes):
 *
 *   site/                 landing page + version switcher, redirects to latest/
 *   site/latest/          typedoc + guides for the current checkout (main)
 *   site/v0.3.0/ …        the same for every release tag that can build them (typedoc.json present)
 *   site/versions.json    [{ version, path, date, current }] — the switcher and external tools read this
 *   site/coverage.json    shields.io endpoint (written by the workflow next to this)
 *
 *   node scripts/build-site.mjs [--out site] [--tags v0.3.0,v0.4.0] [--no-tags]
 *
 * Each tag is checked out into a temporary git worktree, dependencies installed with `--frozen-lockfile`,
 * built and documented; a tag whose checkout cannot build docs (older than the docs tooling) is listed as
 * "not available" instead of failing the site.
 */
import { execFileSync, spawnSync } from 'node:child_process'
import { cpSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('..', import.meta.url))
const args = process.argv.slice(2)
const opt = (name, dflt) => {
  const i = args.indexOf(name)
  return i >= 0 ? args[i + 1] : dflt
}
const out = resolve(root, opt('--out', 'site'))
const noTags = args.includes('--no-tags')
const explicitTags = opt('--tags', null)

const git = (...a) => execFileSync('git', a, { cwd: root, encoding: 'utf8' }).trim()
const sh = (cmd, a, cwd) => {
  const r = spawnSync(cmd, a, { cwd, stdio: 'inherit', shell: process.platform === 'win32' })
  if (r.status !== 0) throw new Error(`${cmd} ${a.join(' ')} failed in ${cwd}`)
}

rmSync(out, { recursive: true, force: true })
mkdirSync(out, { recursive: true })

const current = JSON.parse(readFileSync(resolve(root, 'package.json'), 'utf8')).version
const versions = []

// latest = this checkout
sh('pnpm', ['docs:api'], root)
cpSync(resolve(root, 'docs/api'), join(out, 'latest'), { recursive: true })
versions.push({ version: `main (${current}-dev)`, path: 'latest/', date: git('log', '-1', '--format=%cs'), current: true })

if (!noTags) {
  const tags = (explicitTags ? explicitTags.split(',') : git('tag', '-l', 'v*.*.*').split('\n').filter(Boolean)).sort((a, b) =>
    b.localeCompare(a, undefined, { numeric: true }),
  )
  for (const tag of tags) {
    const wt = mkdtempSync(join(tmpdir(), `columna-site-${tag}-`))
    try {
      git('worktree', 'add', '--detach', wt, tag)
      const date = git('log', '-1', '--format=%cs', tag)
      if (!existsSync(join(wt, 'typedoc.json'))) {
        versions.push({ version: tag, path: null, date, note: 'predates the documentation tooling' })
        continue
      }
      sh('pnpm', ['install', '--frozen-lockfile', '--ignore-scripts'], wt)
      sh('pnpm', ['build'], wt)
      sh('pnpm', ['docs:api'], wt)
      cpSync(join(wt, 'docs/api'), join(out, tag), { recursive: true })
      versions.push({ version: tag, path: `${tag}/`, date })
    } catch (err) {
      console.error(`site: ${tag} skipped — ${err instanceof Error ? err.message : err}`)
      versions.push({ version: tag, path: null, date: null, note: 'build failed' })
    } finally {
      try {
        git('worktree', 'remove', '--force', wt)
      } catch {
        rmSync(wt, { recursive: true, force: true })
      }
    }
  }
}

writeFileSync(join(out, 'versions.json'), JSON.stringify(versions, null, 2))

// version switcher injected into every page: a small script that reads versions.json relative to the site root
const switcher = `(function(){var m=location.pathname.match(/^(.*?\\/)(latest|v\\d+\\.\\d+\\.\\d+[^/]*)\\//);if(!m)return;var base=m[1],cur=m[2];
fetch(base+'versions.json').then(function(r){return r.json()}).then(function(vs){var sel=document.createElement('select');sel.style.cssText='position:fixed;top:8px;right:8px;z-index:9999;font:13px system-ui;padding:2px 6px';
vs.forEach(function(v){var o=document.createElement('option');o.textContent=v.version+(v.date?' · '+v.date:'')+(v.path?'':' (n/a)');o.value=v.path||'';o.disabled=!v.path;o.selected=v.path&&v.path.replace('/','')===cur;sel.appendChild(o)});
sel.onchange=function(){if(sel.value)location.href=base+sel.value+location.pathname.slice((base+cur+'/').length)};document.body.appendChild(sel)}).catch(function(){})})();`
writeFileSync(join(out, 'version-switcher.js'), switcher)
for (const v of versions) {
  if (!v.path) continue
  const dir = join(out, v.path)
  injectSwitcher(dir)
}

const rows = versions
  .map((v) => `<li>${v.path ? `<a href="${v.path}">${v.version}</a>` : `<span>${v.version}</span>`}${v.date ? ` <small>${v.date}</small>` : ''}${v.note ? ` <em>— ${v.note}</em>` : ''}${v.current ? ' <strong>(development)</strong>' : ''}</li>`)
  .join('\n')
writeFileSync(
  join(out, 'index.html'),
  `<!doctype html><meta charset="utf-8"><title>columna documentation</title>
<meta http-equiv="refresh" content="3; url=latest/">
<style>body{font:16px system-ui;max-width:40rem;margin:4rem auto;padding:0 1rem}li{margin:.4rem 0}</style>
<h1>columna documentation</h1>
<p>Redirecting to the <a href="latest/">development version</a>… Released versions:</p>
<ul>${rows}</ul>
<p><a href="versions.json">versions.json</a> · <a href="https://github.com/ankhitlab/columna">repository</a></p>`,
)
console.log(`site: ${versions.filter((v) => v.path).length} version(s) at ${out}`)

function injectSwitcher(dir) {
  const walk = (d) => {
    for (const e of readdirSyncSafe(d)) {
      const p = join(d, e.name)
      if (e.isDirectory()) walk(p)
      else if (e.name.endsWith('.html')) {
        const html = readFileSync(p, 'utf8')
        if (!html.includes('version-switcher.js')) writeFileSync(p, html.replace('</body>', '<script src="' + relToSiteRoot(p) + 'version-switcher.js"></script></body>'))
      }
    }
  }
  walk(dir)
}
function readdirSyncSafe(d) {
  try {
    return readdirSync(d, { withFileTypes: true })
  } catch {
    return []
  }
}
function relToSiteRoot(file) {
  const depth = resolve(file).slice(out.length + 1).split(/[\\/]/).length - 1
  return '../'.repeat(depth)
}
