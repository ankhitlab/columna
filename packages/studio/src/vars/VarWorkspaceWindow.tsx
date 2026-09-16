import { useEffect, useState } from 'react'
import { describeValue, isDataFrame } from '../types'
import type { VarInfo } from '../types'
import { VariableExplorer } from './VariableExplorer'
import { DataFrameViewer } from './DataFrameViewer'

export interface VarWorkspaceWindowProps {
  vars: VarInfo[]
  selected: string | null
  selectedValue: unknown
  onSelect: (name: string) => void
  onRemove: (name: string) => void
  onClearAll: () => void
  onRefresh: () => void
  onEdit: (name: string, value: unknown) => void
  onPlot: (name: string) => void
  onClose: () => void
}

/** Full Variable Explorer + DataFrame/object editor hosted in a detached window. */
export function VarWorkspaceWindow({
  vars,
  selected,
  selectedValue,
  onSelect,
  onRemove,
  onClearAll,
  onRefresh,
  onEdit,
  onPlot,
  onClose,
}: VarWorkspaceWindowProps) {
  const dfSelected = isDataFrame(selectedValue)

  return (
    <div className="var-window">
      <header className="var-window-chrome">
        <div>
          <strong>Variable Explorer</strong>
          <span className="muted"> · Columna Studio</span>
        </div>
        <div className="panel-tools">
          <button className="btn" type="button" onClick={onRefresh}>
            Refresh
          </button>
          <button className="btn" type="button" onClick={onClose}>
            Close
          </button>
        </div>
      </header>
      <div className="var-window-body">
        <aside className="var-window-sidebar">
          <VariableExplorer
            vars={vars}
            selected={selected}
            onSelect={onSelect}
            onInspect={onSelect}
            onRemove={onRemove}
            onClearAll={onClearAll}
            onRefresh={onRefresh}
            onEdit={onEdit}
          />
        </aside>
        <main className="var-window-main">
          {dfSelected ? (
            <DataFrameViewer
              name={selected}
              value={selectedValue}
              onPlot={onPlot}
              expanded
              editable
              onChange={(n, next) => onEdit(n, next)}
            />
          ) : selected ? (
            <ObjectPane name={selected} value={selectedValue} onEdit={onEdit} />
          ) : (
            <div className="panel" style={{ border: 'none', height: '100%' }}>
              <div className="panel-header">
                <span>Viewer</span>
              </div>
              <div className="panel-body">
                <p className="muted" style={{ padding: 16 }}>
                  Select a variable on the left. Double-click a DataFrame in the main IDE to open this
                  window. Double-click cells to edit values.
                </p>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

function ObjectPane({
  name,
  value,
  onEdit,
}: {
  name: string
  value: unknown
  onEdit: (name: string, value: unknown) => void
}) {
  const meta = describeValue(value)
  const primitive =
    typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean'
  const [text, setText] = useState(() => (value == null ? '' : String(value)))

  useEffect(() => {
    setText(value == null ? '' : String(value))
  }, [name, value])

  let dump: string
  try {
    if (typeof value === 'function') dump = value.toString()
    else dump = JSON.stringify(value, null, 2) ?? String(value)
  } catch {
    dump = String(value)
  }

  const commitPrimitive = () => {
    if (!primitive) return
    if (typeof value === 'boolean') {
      const t = text.trim().toLowerCase()
      if (t === 'true' || t === '1') onEdit(name, true)
      else if (t === 'false' || t === '0') onEdit(name, false)
      return
    }
    if (typeof value === 'number') {
      const n = Number(text)
      if (Number.isFinite(n)) onEdit(name, n)
      return
    }
    onEdit(name, text)
  }

  return (
    <div className="panel" style={{ border: 'none', height: '100%' }}>
      <div className="panel-header">
        <span>
          {name} · {meta.type}
        </span>
      </div>
      <div className="df-meta">
        <span className="chip">{meta.preview}</span>
        {primitive && <span className="chip">editable</span>}
      </div>
      <div className="panel-body help-body">
        {primitive ? (
          <label className="form-row" style={{ gridTemplateColumns: '1fr', maxWidth: 480 }}>
            <span className="muted">Value</span>
            <input
              className="toolbar-input"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onBlur={commitPrimitive}
              onKeyDown={(e) => {
                if (e.key === 'Enter') commitPrimitive()
              }}
            />
          </label>
        ) : (
          <pre className="help-dump">{dump}</pre>
        )}
      </div>
    </div>
  )
}
