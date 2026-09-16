import { isDataFrame } from '../types'
import type { ProjectFS } from '../fs/ProjectFS'

export interface MagicContext {
  getVars: () => Record<string, unknown>
  clearConsole: () => void
  clearVars: () => void
  getHistory: () => string[]
  project: ProjectFS | null
  log: (text: string) => void
  runCode: (code: string) => Promise<unknown>
}

export interface MagicResult {
  handled: boolean
  result?: unknown
}

/** Handle IPython-like %magics. Returns handled=false if not a magic line. */
export async function tryMagic(source: string, ctx: MagicContext): Promise<MagicResult> {
  const trimmed = source.trim()
  if (!trimmed.startsWith('%')) return { handled: false }

  const [cmdRaw, ...rest] = trimmed.slice(1).split(/\s+/)
  const cmd = (cmdRaw ?? '').toLowerCase()
  const arg = rest.join(' ').trim()

  switch (cmd) {
    case 'who': {
      const names = Object.keys(ctx.getVars()).sort()
      ctx.log(names.length ? names.join('  ') : '(no variables)')
      return { handled: true }
    }
    case 'whos': {
      const entries = Object.entries(ctx.getVars()).sort(([a], [b]) => a.localeCompare(b))
      if (!entries.length) {
        ctx.log('(no variables)')
        return { handled: true }
      }
      const lines = entries.map(([name, value]) => {
        if (isDataFrame(value)) {
          const [r, c] = value.shape
          return `${name.padEnd(16)} DataFrame  ${r}×${c}`
        }
        return `${name.padEnd(16)} ${typeof value}`.padEnd(28) + String(value).slice(0, 40)
      })
      ctx.log(['Name             Type        Info', '-'.repeat(40), ...lines].join('\n'))
      return { handled: true }
    }
    case 'clear': {
      if (arg === 'vars' || arg === '-v') ctx.clearVars()
      else ctx.clearConsole()
      ctx.log('cleared')
      return { handled: true }
    }
    case 'hist':
    case 'history': {
      const hist = ctx.getHistory()
      ctx.log(hist.map((h, i) => `${String(i).padStart(3)}  ${h.split('\n')[0]}`).join('\n') || '(empty)')
      return { handled: true }
    }
    case 'pwd': {
      const cwd = ctx.project?.cwd ?? '.'
      const root = ctx.project?.rootName ?? '(no project)'
      ctx.log(cwd === '.' ? root : `${root}/${cwd}`)
      return { handled: true }
    }
    case 'cd': {
      if (!ctx.project?.isOpen) {
        ctx.log('No project folder open')
        return { handled: true }
      }
      const next = arg || '.'
      ctx.project.setCwd(ctx.project.resolvePath(next))
      ctx.log(`cwd → ${ctx.project.cwd}`)
      return { handled: true }
    }
    case 'time': {
      if (!arg) {
        ctx.log('Usage: %time <expression>')
        return { handled: true }
      }
      const t0 = performance.now()
      const result = await ctx.runCode(arg)
      const ms = performance.now() - t0
      ctx.log(`Wall time: ${ms.toFixed(2)} ms`)
      return { handled: true, result }
    }
    case 'timeit': {
      if (!arg) {
        ctx.log('Usage: %timeit <expression>')
        return { handled: true }
      }
      const n = 5
      const times: number[] = []
      for (let i = 0; i < n; i++) {
        const t0 = performance.now()
        await ctx.runCode(arg)
        times.push(performance.now() - t0)
      }
      const avg = times.reduce((a, b) => a + b, 0) / times.length
      ctx.log(`${n} loops, avg ${avg.toFixed(2)} ms (min ${Math.min(...times).toFixed(2)} ms)`)
      return { handled: true }
    }
    default:
      ctx.log(`Unknown magic %${cmd}. Try %who %whos %time %timeit %clear %hist %pwd %cd`)
      return { handled: true }
  }
}
