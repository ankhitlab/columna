import { transform } from 'sucrase'
import {
  DataFrame,
  LazyFrame,
  Expr,
  col,
  lit,
  init,
  aggExpr,
  dt,
  daysBetween,
} from 'columna'
import { loadCommandHistory, saveCommandHistory } from '../shell/layoutPersist'
import {
  describeValue,
  isDataFrame,
  uid,
  type ConsoleEntry,
  type ConsoleLevel,
  type RichPayload,
  type VarInfo,
} from '../types'
import type { PlotApi } from '../plots/plotApi'
import type { ProjectFS } from '../fs/ProjectFS'
import type { DebuggerRuntime } from '../debug/Debugger'
import { instrumentForDebug } from '../debug/instrument'
import { tryMagic } from './magics'

const BUILTIN_KEYS = new Set([
  'DataFrame',
  'LazyFrame',
  'Expr',
  'col',
  'lit',
  'init',
  'aggExpr',
  'dt',
  'daysBetween',
  'plot',
  'hist',
  'clear',
  'help',
  'console',
  '__dbg',
])

export interface ReplRunOptions {
  debug?: boolean
  debugger?: DebuggerRuntime
}

export interface ReplRunResult {
  ok: boolean
  result?: unknown
  error?: string
  entries: ConsoleEntry[]
  vars: VarInfo[]
}

export class ReplEngine {
  private userScope: Record<string, unknown> = {}
  private entries: ConsoleEntry[] = []
  private history: string[] = loadCommandHistory()
  private plotApi: PlotApi
  private project: ProjectFS | null = null
  private listeners = new Set<() => void>()

  constructor(plotApi: PlotApi) {
    this.plotApi = plotApi
  }

  setProject(project: ProjectFS | null): void {
    this.project = project
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn)
    return () => this.listeners.delete(fn)
  }

  private emit(): void {
    for (const fn of this.listeners) fn()
  }

  getConsoleEntries(): ConsoleEntry[] {
    return [...this.entries]
  }

  getHistory(): string[] {
    return [...this.history]
  }

  getCompletionNames(): string[] {
    return [...BUILTIN_KEYS, ...Object.keys(this.userScope)].filter((k) => k !== '__dbg')
  }

  clearHistory(): void {
    this.history = []
    saveCommandHistory(this.history)
    this.emit()
  }

  private recordHistory(source: string): void {
    const trimmed = source.trim()
    if (!trimmed) return
    this.history = [trimmed, ...this.history.filter((h) => h !== trimmed)].slice(0, 200)
    saveCommandHistory(this.history)
  }

  getVars(): VarInfo[] {
    return Object.entries(this.userScope)
      .map(([name, value]) => ({ name, value, ...describeValue(value) }))
      .sort((a, b) => a.name.localeCompare(b.name))
  }

  getScope(): Record<string, unknown> {
    return { ...this.userScope }
  }

  getVar(name: string): unknown {
    return this.userScope[name]
  }

  setVar(name: string, value: unknown): void {
    if (!name || BUILTIN_KEYS.has(name)) return
    this.userScope[name] = value
    this.emit()
  }

  removeVar(name: string): void {
    delete this.userScope[name]
    this.emit()
  }

  clearConsole(): void {
    this.entries = []
    this.emit()
  }

  clearUserVars(): void {
    this.userScope = {}
    this.emit()
  }

  refresh(): void {
    this.emit()
  }

  push(level: ConsoleLevel, text: string, rich?: RichPayload): void {
    this.entries.push({ id: uid('log'), level, text, ts: Date.now(), rich })
    this.emit()
  }

  private makeConsole() {
    return {
      log: (...args: unknown[]) => this.push('log', args.map(stringify).join(' ')),
      info: (...args: unknown[]) => this.push('info', args.map(stringify).join(' ')),
      warn: (...args: unknown[]) => this.push('warn', args.map(stringify).join(' ')),
      error: (...args: unknown[]) => this.push('error', args.map(stringify).join(' ')),
    }
  }

  private builtins(dbg?: DebuggerRuntime): Record<string, unknown> {
    const consoleApi = this.makeConsole()
    const base: Record<string, unknown> = {
      DataFrame,
      LazyFrame,
      Expr,
      col,
      lit,
      init,
      aggExpr,
      dt,
      daysBetween,
      plot: this.plotApi.plot,
      hist: this.plotApi.hist,
      clear: () => {
        this.clearConsole()
        this.plotApi.clear()
      },
      help: () => {
        consoleApi.log(
          [
            'Columna Studio REPL',
            '- DataFrame, col, lit, init · plot / hist',
            '- Magics: %who %whos %time %timeit %clear %hist %pwd %cd',
            '- Cells: // %% · Debug Run (F9) · F2 rename',
          ].join('\n'),
        )
      },
      console: consoleApi,
    }
    if (dbg) base.__dbg = dbg.api()
    return base
  }

  async run(source: string, options: ReplRunOptions = {}): Promise<ReplRunResult> {
    const trimmed = source.trim()
    if (!trimmed) {
      return { ok: true, entries: this.getConsoleEntries(), vars: this.getVars() }
    }

    // Magics (skip recording inner %time code twice via runCode callback)
    if (trimmed.startsWith('%')) {
      this.recordHistory(trimmed)
      this.push('info', `>>> ${preview(trimmed)}`)
      const magic = await tryMagic(trimmed, {
        getVars: () => this.userScope,
        clearConsole: () => this.clearConsole(),
        clearVars: () => this.clearUserVars(),
        getHistory: () => this.getHistory(),
        project: this.project,
        log: (t) => this.push('info', t),
        runCode: async (code) => {
          const r = await this.run(code, { debug: false })
          if (!r.ok) throw new Error(r.error ?? 'magic failed')
          return r.result
        },
      })
      if (magic.handled) {
        if (magic.result !== undefined) this.pushRichResult(magic.result)
        this.emit()
        return { ok: true, result: magic.result, entries: this.getConsoleEntries(), vars: this.getVars() }
      }
    }

    this.recordHistory(trimmed)
    this.push('info', `>>> ${preview(trimmed)}`)

    const debug = Boolean(options.debug && options.debugger)
    if (debug) options.debugger!.beginRun()

    let jsCode: string
    try {
      jsCode = transform(trimmed, {
        transforms: ['typescript', 'imports'],
        disableESTransforms: true,
      }).code
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      this.push('error', `transpile: ${message}`)
      if (debug) options.debugger!.endRun()
      this.emit()
      return { ok: false, error: message, entries: this.getConsoleEntries(), vars: this.getVars() }
    }

    jsCode = jsCode
      .replace(/^["']use strict["'];?/, '')
      .split('\n')
      .filter((line) => !/^\s*import\b/.test(line) && !/^\s*export\b/.test(line))
      .join('\n')

    const declared = collectDeclarations(trimmed)
    const builtins = this.builtins(debug ? options.debugger : undefined)
    const env: Record<string, unknown> = { ...builtins, ...this.userScope }
    const allNames = [...new Set([...Object.keys(env), ...declared])]

    const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor as new (
      ...args: string[]
    ) => (...args: unknown[]) => Promise<unknown>

    let code = stripDeclKeywords(instrumentLastExpression(jsCode))
    if (debug) {
      code = instrumentForDebug(code, allNames.filter((n) => n !== '__dbg'))
    }

    const body = `
const __env = arguments[0];
let __result = undefined;
${allNames.map((k) => `let ${k} = __env[${JSON.stringify(k)}];`).join('\n')}
${code}
const __captured = {};
${declared.map((n) => `try { __captured[${JSON.stringify(n)}] = ${n}; } catch {}`).join('\n')}
${Object.keys(this.userScope)
  .map((n) => `try { __captured[${JSON.stringify(n)}] = ${n}; } catch {}`)
  .join('\n')}
return { __captured, __result };
`

    try {
      // Trust boundary note: this is a REPL, not a sandbox. The code runs in the page's realm with the
      // page's privileges — origin storage, opened folder handles, network as the page — exactly like the
      // browser devtools console. A Worker or iframe would not change that (same origin, same handles,
      // same fetch). Never feed Studio code you would not paste into devtools; see packages/studio/README.md.
      const fn = new AsyncFunction(body)
      const out = (await fn(env)) as { __captured: Record<string, unknown>; __result: unknown }

      for (const [k, v] of Object.entries(out.__captured)) {
        if (BUILTIN_KEYS.has(k)) continue
        this.userScope[k] = v
      }

      if (out.__result !== undefined) {
        this.pushRichResult(out.__result)
      }

      if (debug) options.debugger!.endRun()
      this.emit()
      return {
        ok: true,
        result: out.__result,
        entries: this.getConsoleEntries(),
        vars: this.getVars(),
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      this.push('error', message)
      if (debug) options.debugger!.endRun()
      this.emit()
      return { ok: false, error: message, entries: this.getConsoleEntries(), vars: this.getVars() }
    }
  }

  private pushRichResult(value: unknown): void {
    if (isDataFrame(value)) {
      const [r, c] = value.shape
      const head = value.toArray().slice(0, 8)
      this.push('result', `DataFrame ${r}×${c}`, {
        kind: 'dataframe',
        columns: value.columns,
        rows: head,
      })
      return
    }
    if (value !== null && typeof value === 'object') {
      try {
        this.push('result', stringify(value), { kind: 'json', json: value })
        return
      } catch {
        /* fallthrough */
      }
    }
    this.push('result', stringify(value))
  }

  seedDataFrame(name: string, df: DataFrame): void {
    this.userScope[name] = df
    this.emit()
  }
}

function preview(text: string): string {
  const first = text.split('\n')[0] ?? text
  return text.includes('\n') ? `${first} …` : first
}

function stringify(value: unknown): string {
  if (isDataFrame(value)) {
    const [rows, cols] = value.shape
    return `DataFrame ${rows}×${cols}`
  }
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value, null, 2) ?? String(value)
  } catch {
    return String(value)
  }
}

function collectDeclarations(source: string): string[] {
  const names = new Set<string>()
  for (const m of source.matchAll(/\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)/g)) {
    names.add(m[1]!)
  }
  return [...names]
}

function stripDeclKeywords(code: string): string {
  return code
    .replace(/^["']use strict["'];?/, '')
    .replace(/\b(?:const|let|var)\s+/g, '')
}

function instrumentLastExpression(jsCode: string): string {
  const code = jsCode.trim()
  if (!code) return ''
  const lines = code.split('\n')
  let lastIdx = lines.length - 1
  while (lastIdx >= 0 && lines[lastIdx]!.trim() === '') lastIdx--
  if (lastIdx < 0) return ''
  const last = lines[lastIdx]!.trim()
  const isDecl =
    /^(const|let|var|function|class|if|for|while|switch|return|throw|try|type|interface|export|import)\b/.test(
      last,
    )
  const isBlock = last.endsWith('{') || last === '}' || last.endsWith(';')
  if (isDecl || isBlock) return code
  lines[lastIdx] = `__result = (${last.replace(/;$/, '')});`
  return lines.join('\n')
}
