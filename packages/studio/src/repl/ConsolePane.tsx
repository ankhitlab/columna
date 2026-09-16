import { useEffect, useRef, useState } from 'react'
import type { ConsoleEntry } from '../types'

export interface ConsolePaneProps {
  entries: ConsoleEntry[]
  history: string[]
  completions?: string[]
  draft?: string
  onDraftConsumed?: () => void
  onSubmit: (code: string) => void
  onClear: () => void
}

export function ConsolePane({
  entries,
  history,
  completions = [],
  draft,
  onDraftConsumed,
  onSubmit,
  onClear,
}: ConsolePaneProps) {
  const [input, setInput] = useState('')
  const [histIdx, setHistIdx] = useState(-1)
  const bottomRef = useRef<HTMLDivElement>(null)
  const areaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [entries])

  useEffect(() => {
    if (draft == null) return
    setInput(draft)
    onDraftConsumed?.()
    areaRef.current?.focus()
  }, [draft, onDraftConsumed])

  const submit = () => {
    if (!input.trim()) return
    onSubmit(input)
    setInput('')
    setHistIdx(-1)
  }

  const complete = () => {
    const el = areaRef.current
    if (!el) return
    const pos = el.selectionStart
    const before = input.slice(0, pos)
    const m = before.match(/([A-Za-z_$][\w$]*)$/)
    if (!m) return
    const prefix = m[1]!
    const hits = completions.filter((c) => c.startsWith(prefix) && c !== prefix)
    if (hits.length === 1) {
      const rest = input.slice(pos)
      const next = before.slice(0, before.length - prefix.length) + hits[0] + rest
      setInput(next)
      return
    }
    if (hits.length > 1) {
      // common prefix
      let common = hits[0]!
      for (const h of hits.slice(1)) {
        let i = 0
        while (i < common.length && i < h.length && common[i] === h[i]) i++
        common = common.slice(0, i)
      }
      if (common.length > prefix.length) {
        const rest = input.slice(pos)
        setInput(before.slice(0, before.length - prefix.length) + common + rest)
      }
    }
  }

  return (
    <div className="console">
      <div className="panel-header" style={{ background: '#151a24', color: '#9aa7b8', borderColor: '#323b4a' }}>
        <span>Console · Tab complete · %magics</span>
        <button className="btn btn-ghost" style={{ color: '#9aa7b8' }} type="button" onClick={onClear}>
          Clear
        </button>
      </div>
      <div className="console-out">
        {entries.map((e) => (
          <div key={e.id} className={`console-line ${e.level}`}>
            <pre className="console-text">{e.text}</pre>
            {e.rich?.kind === 'dataframe' && (
              <div className="rich-df">
                <table className="df-table">
                  <thead>
                    <tr>
                      {e.rich.columns.map((c) => (
                        <th key={c}>{c}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {e.rich.rows.map((row, ri) => {
                      const cols = (e.rich as { kind: 'dataframe'; columns: string[] }).columns
                      return (
                        <tr key={ri}>
                          {cols.map((c) => (
                            <td key={c}>{row[c] == null ? '∅' : String(row[c])}</td>
                          ))}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
            {e.rich?.kind === 'json' && (
              <details className="rich-json">
                <summary>object</summary>
                <pre>{JSON.stringify(e.rich.json, null, 2)}</pre>
              </details>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      <div className="console-input-row">
        <textarea
          ref={areaRef}
          className="console-input console-textarea"
          value={input}
          rows={Math.min(6, Math.max(1, input.split('\n').length))}
          onChange={(e) => setInput(e.target.value)}
          placeholder=">>> expression · %who · Tab complete · Ctrl+Enter"
          spellCheck={false}
          onKeyDown={(e) => {
            if (e.key === 'Tab') {
              e.preventDefault()
              complete()
              return
            }
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
              e.preventDefault()
              submit()
              return
            }
            if (e.key === 'Enter' && !e.shiftKey && !input.includes('\n')) {
              e.preventDefault()
              submit()
              return
            }
            if (e.key === 'ArrowUp' && !e.shiftKey && (histIdx >= 0 || input === '' || !input.includes('\n'))) {
              if (history.length === 0) return
              const caretAtStart = areaRef.current?.selectionStart === 0
              if (!caretAtStart && histIdx < 0 && input.includes('\n')) return
              e.preventDefault()
              const next = Math.min(history.length - 1, histIdx + 1)
              setHistIdx(next)
              setInput(history[next] ?? '')
            }
            if (e.key === 'ArrowDown' && histIdx >= 0) {
              e.preventDefault()
              const next = histIdx - 1
              setHistIdx(next)
              setInput(next < 0 ? '' : (history[next] ?? ''))
            }
          }}
        />
        <button className="btn btn-primary" type="button" onClick={submit}>
          Run
        </button>
      </div>
    </div>
  )
}
