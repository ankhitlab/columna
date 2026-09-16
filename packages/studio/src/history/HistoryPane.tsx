import { useMemo, useState } from 'react'

export interface HistoryPaneProps {
  commands: string[]
  onRerun: (code: string) => void
  onPaste: (code: string) => void
  onClear: () => void
}

export function HistoryPane({ commands, onRerun, onPaste, onClear }: HistoryPaneProps) {
  const [filter, setFilter] = useState('')

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase()
    if (!q) return commands
    return commands.filter((c) => c.toLowerCase().includes(q))
  }, [commands, filter])

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="panel-header">
        <span>History</span>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <input
            className="toolbar-input"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter…"
          />
          <button className="btn" type="button" onClick={onClear}>
            Clear
          </button>
        </div>
      </div>
      <div className="panel-body history-list">
        {filtered.length === 0 ? (
          <p className="muted" style={{ padding: 12 }}>
            No commands yet. Run code from the editor or console.
          </p>
        ) : (
          filtered.map((cmd, i) => (
            <div key={`${i}-${cmd.slice(0, 24)}`} className="history-item">
              <pre className="history-code">{cmd}</pre>
              <div className="history-actions">
                <button className="btn" type="button" onClick={() => onPaste(cmd)}>
                  Paste
                </button>
                <button className="btn btn-primary" type="button" onClick={() => onRerun(cmd)}>
                  Run
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
