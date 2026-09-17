/**
 * Serve ./dist and run it in headless Chromium via Playwright. Chromium is launched with WebGPU enabled;
 * on CI runners without a GPU the page skips the GPU checks and still verifies the CPU path of the
 * *built* bundle. Exit code 1 on any failed check.
 *
 *   pnpm --filter @columna/browser-smoke run test:ci
 *   SMOKE_HEADED=1 node run.mjs     # watch it
 */
import { createServer } from 'node:http'
import { readFile, stat } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const dist = path.join(path.dirname(fileURLToPath(import.meta.url)), 'dist')
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm', '.json': 'application/json', '.map': 'application/json' }

const server = createServer(async (req, res) => {
  try {
    let p = path.join(dist, decodeURIComponent((req.url ?? '/').split('?')[0]))
    if ((await stat(p)).isDirectory()) p = path.join(p, 'index.html')
    res.setHeader('content-type', types[path.extname(p)] ?? 'application/octet-stream')
    res.end(await readFile(p))
  } catch {
    res.statusCode = 404
    res.end()
  }
})
await new Promise((r) => server.listen(0, '127.0.0.1', r))
const port = server.address().port
const url = `http://127.0.0.1:${port}/`

let playwright
try {
  playwright = await import('playwright')
} catch {
  console.error('playwright is not installed: pnpm --filter @columna/browser-smoke add -D playwright && npx playwright install chromium')
  process.exit(2)
}
// Prefer the full Chromium (new headless mode: WebGPU via SwiftShader); fall back to the headless shell.
const launchOpts = {
  headless: !process.env.SMOKE_HEADED,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist'],
}
// SMOKE_CHANNEL=chrome|msedge uses the installed system browser (a real GPU → the WebGPU checks run).
const channel = process.env.SMOKE_CHANNEL ?? 'chromium'
const browser = await playwright.chromium.launch({ ...launchOpts, channel }).catch(() => playwright.chromium.launch(launchOpts))
const page = await browser.newPage()
const console_ = []
page.on('console', (m) => console_.push(`[${m.type()}] ${m.text()}`))
page.on('pageerror', (e) => console_.push(`[pageerror] ${e.message}`))
await page.goto(url)
await page.waitForFunction(() => window.__smoke?.done === true, null, { timeout: 120_000 })
const result = await page.evaluate(() => window.__smoke)
await browser.close()
server.close()

console.log(`browser: ${result.userAgent}`)
console.log(`webgpu: ${result.webgpu ? 'available — GPU checks ran' : 'not available — GPU checks skipped'}`)
for (const c of result.checks) console.log(`${c.ok ? '  ok  ' : '  FAIL'} ${c.name}${c.ms !== undefined ? ` (${c.ms.toFixed(0)} ms)` : ''}${c.detail ? ` — ${c.detail}` : ''}`)
const failed = result.checks.filter((c) => !c.ok)
if (failed.length) {
  console.error(`\n${failed.length} check(s) failed`)
  for (const line of console_) console.error(line)
  process.exit(1)
}
console.log(`\nall ${result.checks.length} checks passed`)
