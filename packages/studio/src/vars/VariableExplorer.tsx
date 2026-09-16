import { useMemo, useState } from 'react'
import type { VarInfo } from '../types'

export interface VariableExplorerProps {
  vars: VarInfo[]
  selected?: string | null
  onSelect: (name: string) => void
  onInspect: (name: string) => void
  /** Open DataFrame / variable in detached Spyder-style window */
  onOpenWindow?: (name: string) => void
  onRemove: (name: string) => void
  onClearAll: () => void
  onRefresh: () => void
  onEdit: (name: string, value: unknown) => void
}

export function VariableExplorer({
  vars,
  selected,
  onSelect,
  onInspect,
  onOpenWindow,
  onRemove,
  onClearAll,
  onRefresh,
  onEdit,
}: VariableExplorerProps) {
  const [filter, setFilter] = useState('')
  const [editing, setEditing] = useState<string | null>(null)
  const [editText, setEditText] = useState('')

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase()
    if (!q) return vars
    return vars.filter(
      (v) =>
        v.name.toLowerCase().includes(q) ||
        v.type.toLowerCase().includes(q) ||
        v.preview.toLowerCase().includes(q),
    )
  }, [vars, filter])

  const copyName = async (name: string) => {
    try {
      await navigator.clipboard.writeText(name)
    } catch {
      /* ignore */
    }
  }

  const startEdit = (v: VarInfo) => {
    if (!isPrimitive(v.value)) return
    setEditing(v.name)
    setEditText(String(v.value))
  }

  const commitEdit = (name: string, original: unknown) => {
    setEditing(null)
    const next = parsePrimitive(editText, original)
    if (next !== undefined) onEdit(name, next)
  }

  return (
    <div className="panel" style={{ border: 'none', height: '100%' }}>
      <div className="panel-header">
        <span>Variable Explorer</span>
        <div className="panel-tools">
          <input
            className="toolbar-input"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter…"
          />
          <button className="btn" type="button" title="Refresh" onClick={onRefresh}>
            ↻
          </button>
          <button
            className="btn"
            type="button"
            title="Clear namespace"
            disabled={vars.length === 0}
            onClick={() => {
              if (confirm('Remove all user variables?')) onClearAll()
            }}
          >
            Clear
          </button>
        </div>
      </div>
      <div className="panel-body">
        {filtered.length === 0 ? (
          <p className="muted" style={{ padding: 12 }}>
            No user variables yet. Run code that assigns a DataFrame.
          </p>
        ) : (
          <table className="var-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Type</th>
                <th>Size / Preview</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {filtered.map((v) => (
                <tr
                  key={v.name}
                  className={selected === v.name ? 'active' : undefined}
                  onClick={() => onSelect(v.name)}
                  onDoubleClick={() => {
                    if (onOpenWindow) onOpenWindow(v.name)
                    else onInspect(v.name)
                  }}
                >
                  <td>
                    {editing === v.name ? (
                      <input
                        className="toolbar-input"
                        autoFocus
                        value={editText}
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => setEditText(e.target.value)}
                        onBlur={() => commitEdit(v.name, v.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') commitEdit(v.name, v.value)
                          if (e.key === 'Escape') setEditing(null)
                        }}
                      />
                    ) : (
                      <button
                        type="button"
                        className="linkish"
                        title={isPrimitive(v.value) ? 'Click to edit' : v.name}
                        onClick={(e) => {
                          e.stopPropagation()
                          onSelect(v.name)
                          startEdit(v)
                        }}
                      >
                        {v.name}
                      </button>
                    )}
                  </td>
                  <td>{v.type}</td>
                  <td className="muted">{v.preview}</td>
                  <td className="var-actions" onClick={(e) => e.stopPropagation()}>
                    <button className="btn btn-ghost" type="button" title="Copy name" onClick={() => void copyName(v.name)}>
                      ⎘
                    </button>
                    <button className="btn btn-ghost" type="button" title="Inspect in Help" onClick={() => onInspect(v.name)}>
                      ?
                    </button>
                    {onOpenWindow && (
                      <button
                        className="btn btn-ghost"
                        type="button"
                        title="Open in window"
                        onClick={() => onOpenWindow(v.name)}
                      >
                        ⧉
                      </button>
                    )}
                    <button className="btn btn-ghost" type="button" title="Remove" onClick={() => onRemove(v.name)}>
                      ×
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

function isPrimitive(value: unknown): boolean {
  return typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean'
}

function parsePrimitive(text: string, original: unknown): unknown | undefined {
  if (typeof original === 'boolean') {
    const t = text.trim().toLowerCase()
    if (t === 'true') return true
    if (t === 'false') return false
    return undefined
  }
  if (typeof original === 'number') {
    const n = Number(text)
    return Number.isFinite(n) ? n : undefined
  }
  return text
}
