import { describe, expect, it } from 'vitest'
import { DataFrame } from 'columna'
import { cellAtLine, parseCells } from '../src/editor/cells'
import { parseOutline } from '../src/outline/outlineParse'
import { createPlotApi } from '../src/plots/plotApi'
import { ReplEngine } from '../src/repl/ReplEngine'
import { DebuggerRuntime } from '../src/debug/Debugger'
import { instrumentForDebug } from '../src/debug/instrument'
import { searchInContent, renameIdentifierInContent } from '../src/search/searchLib'
import { analyzeStudioHeuristics } from '../src/analysis/Analysis'

describe('ReplEngine', () => {
  it('assigns DataFrame into scope with shape preview', async () => {
    const plots = createPlotApi()
    const repl = new ReplEngine(plots)
    const result = await repl.run(`
const df = DataFrame.fromRows([
  { city: 'Berlin', age: 30 },
  { city: 'Paris', age: 41 },
])
df
`)
    expect(result.ok).toBe(true)
    const dfVar = result.vars.find((v) => v.name === 'df')
    expect(dfVar?.isDataFrame).toBe(true)
    expect(dfVar?.preview).toContain('2×2')
  })

  it('supports filter collect pipeline', async () => {
    const repl = new ReplEngine(createPlotApi())
    const result = await repl.run(`
const df = DataFrame.fromRows([
  { age: 10, v: 1 },
  { age: 20, v: 2 },
])
const out = await df.filter(col('age').gt(15)).collect()
out.shape
`)
    expect(result.ok).toBe(true)
    expect(result.result).toEqual([1, 2])
  })

  it('setVar / removeVar / clearUserVars', async () => {
    const repl = new ReplEngine(createPlotApi())
    const r = await repl.run('const answer = 42')
    expect(r.ok).toBe(true)
    expect(repl.getVar('answer')).toBe(42)
    repl.setVar('answer', 7)
    expect(repl.getVar('answer')).toBe(7)
    repl.removeVar('answer')
    expect(repl.getVar('answer')).toBeUndefined()
    await repl.run('const a = 1\nconst b = 2')
    repl.clearUserVars()
    expect(repl.getVars()).toHaveLength(0)
  })

  it('records command history', async () => {
    const repl = new ReplEngine(createPlotApi())
    await repl.run('const x = 1')
    await repl.run('const y = 2')
    const hist = repl.getHistory()
    expect(hist[0]).toContain('const y = 2')
    expect(hist.some((h) => h.includes('const x = 1'))).toBe(true)
    repl.clearHistory()
    expect(repl.getHistory()).toHaveLength(0)
  })

  it('handles %who and %whos magics', async () => {
    const repl = new ReplEngine(createPlotApi())
    await repl.run('const alpha = 1')
    const who = await repl.run('%who')
    expect(who.ok).toBe(true)
    const whos = await repl.run('%whos')
    expect(whos.ok).toBe(true)
    expect(repl.getConsoleEntries().some((e) => e.text.includes('alpha'))).toBe(true)
  })

  it('%time runs expression', async () => {
    const repl = new ReplEngine(createPlotApi())
    const r = await repl.run('%time 1+1')
    expect(r.ok).toBe(true)
    expect(repl.getConsoleEntries().some((e) => e.text.includes('Wall time'))).toBe(true)
  })

  it('plot api records specs and remove', () => {
    const api = createPlotApi()
    const df = DataFrame.fromRows([
      { x: 1, y: 2 },
      { x: 2, y: 4 },
    ])
    const p = api.plot(df, { x: 'x', y: 'y', type: 'scatter' })
    expect(api.getPlots()).toHaveLength(1)
    api.hist(df, 'y', { bins: 5, title: 'ages' })
    expect(api.getPlots()[0]?.title).toBe('ages')
    api.remove(p.id)
    expect(api.getPlots().every((x) => x.id !== p.id)).toBe(true)
  })
})

describe('debugger', () => {
  it('instruments lines with __dbg.line', () => {
    const code = instrumentForDebug('x = 1\ny = 2', ['x', 'y'])
    expect(code).toContain('await __dbg.line(1')
    expect(code).toContain('await __dbg.line(2')
  })

  it('pauses on breakpoint and continues', async () => {
    const dbg = new DebuggerRuntime()
    dbg.toggleBreakpoint(2)
    const repl = new ReplEngine(createPlotApi())
    const runPromise = repl.run('const a = 1\nconst b = a + 1\nb', { debug: true, debugger: dbg })

    // wait until paused
    await new Promise<void>((resolve) => {
      const t = setInterval(() => {
        if (dbg.paused) {
          clearInterval(t)
          resolve()
        }
      }, 10)
    })
    expect(dbg.paused?.line).toBe(2)
    dbg.action('continue')
    const result = await runPromise
    expect(result.ok).toBe(true)
    expect(repl.getVar('b')).toBe(2)
  })
})

describe('search', () => {
  it('finds text in content', () => {
    const hits = searchInContent('a.ts', 'hello\nworld hello', { query: 'hello' })
    expect(hits).toHaveLength(2)
    expect(hits[0]?.line).toBe(1)
  })

  it('renames identifier', () => {
    const { content, count } = renameIdentifierInContent('const foo = 1\nfoo + foo', 'foo', 'bar')
    expect(count).toBe(3)
    expect(content).toContain('bar')
    expect(content).not.toContain('foo')
  })
})

describe('analysis', () => {
  it('flags debugger leftover', () => {
    const issues = analyzeStudioHeuristics('const x = 1\ndebugger\n')
    expect(issues.some((i) => i.message.includes('debugger'))).toBe(true)
  })
})

describe('cells', () => {
  it('parses // %% markers', () => {
    const src = `// %% one
const a = 1
// %% two
const b = 2
`
    const cells = parseCells(src)
    expect(cells).toHaveLength(2)
    expect(cells[0]?.title).toBe('one')
    expect(cellAtLine(src, 4).title).toBe('two')
  })
})

describe('outline', () => {
  it('finds cells functions and consts', () => {
    const src = `// %% Load
const df = 1
function foo() {}
class Bar {}
`
    const items = parseOutline(src)
    expect(items.some((i) => i.kind === 'cell' && i.name === 'Load')).toBe(true)
    expect(items.some((i) => i.kind === 'function' && i.name === 'foo')).toBe(true)
  })
})
