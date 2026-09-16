import { useState } from 'react'
import type { DebuggerRuntime, DebugPauseState } from './Debugger'

export interface DebugPaneProps {
  dbg: DebuggerRuntime
  paused: DebugPauseState | null
  running: boolean
  breakpoints: number[]
  onAction: (a: 'continue' | 'stepOver' | 'stepInto' | 'stepOut' | 'stop') => void
  onEvalWatch: (expr: string) => Promise<string>
  onJumpLine?: (line: number) => void
}

export function DebugPane({
  dbg,
  paused,
  running,
  breakpoints,
  onAction,
  onEvalWatch,
  onJumpLine,
}: DebugPaneProps) {
  const [watchExpr, setWatchExpr] = useState('df')
  const [watchOut, setWatchOut] = useState('')
  const [watches, setWatches] = useState<string[]>([''])

  void dbg

  return (
    <div className="debug-pane">
      <div className="panel-header">
        <span>Debug</span>
        <div className="panel-tools">
          <button className="btn" type="button" disabled={!paused} onClick={() => onAction('continue')}>
            Continue
          </button>
          <button className="btn" type="button" disabled={!paused} onClick={() => onAction('stepOver')}>
            Over
          </button>
          <button className="btn" type="button" disabled={!paused} onClick={() => onAction('stepInto')}>
            Into
          </button>
          <button className="btn" type="button" disabled={!paused} onClick={() => onAction('stepOut')}>
            Out
          </button>
          <button className="btn" type="button" disabled={!running} onClick={() => onAction('stop')}>
            Stop
          </button>
        </div>
      </div>
      <div className="debug-body">
        <div className="debug-col">
          <h4>Status</h4>
          <p className="muted">
            {paused
              ? `Paused at line ${paused.line}`
              : running
                ? 'Running…'
                : 'Idle — use Debug Run (F9). Click gutter to toggle breakpoints.'}
          </p>
          <p className="muted">Breakpoints: {breakpoints.length ? breakpoints.join(', ') : 'none'}</p>
          <h4>Call stack</h4>
          {(paused?.frames.length ? paused.frames : []).length === 0 ? (
            <p className="muted">{paused ? `<script>:${paused.line}` : '—'}</p>
          ) : (
            <ul className="debug-stack">
              {(paused?.frames ?? []).map((f, i) => (
                <li key={`${f.name}-${i}`}>
                  <button type="button" className="linkish" onClick={() => onJumpLine?.(f.line)}>
                    {f.name}:{f.line}
                  </button>
                </li>
              ))}
            </ul>
          )}
          <h4>Locals</h4>
          <pre className="help-dump">
            {paused
              ? Object.entries(paused.locals)
                  .map(([k, v]) => `${k} = ${fmt(v)}`)
                  .join('\n') || '(empty)'
              : '—'}
          </pre>
        </div>
        <div className="debug-col">
          <h4>Watch</h4>
          {watches.map((w, i) => (
            <div key={i} className="watch-row">
              <input
                className="toolbar-input"
                value={w}
                placeholder="expression"
                onChange={(e) => {
                  const next = [...watches]
                  next[i] = e.target.value
                  setWatches(next)
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && w.trim() && paused) {
                    void onEvalWatch(w).then(setWatchOut)
                  }
                }}
              />
            </div>
          ))}
          <button className="btn" type="button" onClick={() => setWatches((w) => [...w, ''])}>
            + watch
          </button>
          <div className="watch-eval">
            <input
              className="toolbar-input"
              value={watchExpr}
              onChange={(e) => setWatchExpr(e.target.value)}
              placeholder="Evaluate…"
            />
            <button
              className="btn"
              type="button"
              disabled={!paused}
              onClick={() => void onEvalWatch(watchExpr).then(setWatchOut)}
            >
              Eval
            </button>
          </div>
          <pre className="help-dump">{watchOut || '—'}</pre>
          <p className="muted" style={{ fontSize: 12 }}>
            Cooperative debugger: steps user script lines only (not columna internals).
          </p>
        </div>
      </div>
    </div>
  )
}

function fmt(v: unknown): string {
  try {
    if (v && typeof v === 'object' && 'shape' in (v as object)) {
      const s = (v as { shape: [number, number] }).shape
      return `DataFrame ${s[0]}×${s[1]}`
    }
    return JSON.stringify(v) ?? String(v)
  } catch {
    return String(v)
  }
}
