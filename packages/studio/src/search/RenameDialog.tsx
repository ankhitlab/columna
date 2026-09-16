import { useMemo, useState } from 'react'
import { renameIdentifierInContent } from './searchLib'

export interface RenameDialogProps {
  open: boolean
  symbol: string
  files: { path: string; content: string }[]
  onClose: () => void
  onApply: (edits: { path: string; content: string }[]) => void
}

export function RenameDialog({ open, symbol, files, onClose, onApply }: RenameDialogProps) {
  const [to, setTo] = useState(symbol)

  const preview = useMemo(() => {
    if (!open || !symbol || !to || symbol === to) return []
    return files
      .map((f) => {
        const r = renameIdentifierInContent(f.content, symbol, to)
        return { path: f.path, count: r.count, content: r.content }
      })
      .filter((x) => x.count > 0)
  }, [open, symbol, to, files])

  if (!open) return null

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()} role="dialog">
        <div className="modal-header">
          <strong>Rename symbol</strong>
          <button className="btn btn-ghost" type="button" onClick={onClose}>
            ×
          </button>
        </div>
        <div className="modal-body">
          <label className="form-row">
            <span>From</span>
            <input className="toolbar-input" value={symbol} readOnly />
          </label>
          <label className="form-row">
            <span>To</span>
            <input className="toolbar-input" value={to} onChange={(e) => setTo(e.target.value)} autoFocus />
          </label>
          <p className="muted">
            {preview.reduce((a, b) => a + b.count, 0)} replacements in {preview.length} files
          </p>
          <ul className="find-results" style={{ maxHeight: 160, overflow: 'auto' }}>
            {preview.map((p) => (
              <li key={p.path} className="muted">
                {p.path}: {p.count}
              </li>
            ))}
          </ul>
          <div className="modal-footer">
            <button className="btn" type="button" onClick={onClose}>
              Cancel
            </button>
            <button
              className="btn btn-primary"
              type="button"
              disabled={!preview.length}
              onClick={() => {
                onApply(preview.map((p) => ({ path: p.path, content: p.content })))
                onClose()
              }}
            >
              Apply
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
