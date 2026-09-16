import { useMemo } from 'react'
import { parseOutline } from './outlineParse'

export interface OutlinePaneProps {
  source: string
  onJump: (line: number) => void
}

export function OutlinePane({ source, onJump }: OutlinePaneProps) {
  const items = useMemo(() => parseOutline(source), [source])

  return (
    <div className="panel" style={{ border: 'none', height: '100%' }}>
      <div className="panel-header">
        <span>Outline</span>
        <span className="muted">{items.length}</span>
      </div>
      <div className="panel-body">
        {items.length === 0 ? (
          <p className="muted" style={{ padding: 12 }}>
            No symbols. Add <code>// %%</code> cells or declarations.
          </p>
        ) : (
          <ul className="outline-list">
            {items.map((item) => (
              <li key={`${item.kind}-${item.line}-${item.name}`}>
                <button type="button" className="outline-item" onClick={() => onJump(item.line)}>
                  <span className={`outline-kind kind-${item.kind}`}>{item.kind[0]!.toUpperCase()}</span>
                  <span className="outline-name">{item.name}</span>
                  <span className="muted outline-line">:{item.line}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
