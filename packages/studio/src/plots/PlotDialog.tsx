import { useEffect, useMemo, useState } from 'react'
import type { PlotType } from '../types'
import { isDataFrame } from '../types'

export interface PlotDialogProps {
  open: boolean
  frameName: string | null
  value: unknown
  onClose: () => void
  onSubmit: (opts: {
    x: string
    y: string
    type: PlotType
    title: string
    bins: number
  }) => void
}

export function PlotDialog({ open, frameName, value, onClose, onSubmit }: PlotDialogProps) {
  const df = isDataFrame(value) ? value : null
  const columns = useMemo(() => df?.columns ?? [], [df])

  const [x, setX] = useState('')
  const [y, setY] = useState('')
  const [type, setType] = useState<PlotType>('scatter')
  const [title, setTitle] = useState('')
  const [bins, setBins] = useState(20)

  useEffect(() => {
    if (!open || !df) return
    setX(columns[0] ?? '')
    setY(columns[1] ?? columns[0] ?? '')
    setType('scatter')
    setTitle(frameName ? `${frameName}` : '')
    setBins(20)
  }, [open, df, columns, frameName])

  if (!open) return null

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label="Plot dialog"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <strong>Create plot{frameName ? ` · ${frameName}` : ''}</strong>
          <button className="btn btn-ghost" type="button" onClick={onClose}>
            ×
          </button>
        </div>
        {!df ? (
          <p className="muted" style={{ padding: 16 }}>
            Select a DataFrame first.
          </p>
        ) : (
          <form
            className="modal-body"
            onSubmit={(e) => {
              e.preventDefault()
              onSubmit({ x, y, type, title, bins })
            }}
          >
            <label className="form-row">
              <span>Type</span>
              <select value={type} onChange={(e) => setType(e.target.value as PlotType)}>
                <option value="scatter">Scatter</option>
                <option value="line">Line</option>
                <option value="bar">Bar</option>
                <option value="hist">Histogram</option>
              </select>
            </label>
            {type === 'hist' ? (
              <>
                <label className="form-row">
                  <span>Column</span>
                  <select value={x} onChange={(e) => setX(e.target.value)}>
                    {columns.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="form-row">
                  <span>Bins</span>
                  <input
                    type="number"
                    min={2}
                    max={200}
                    value={bins}
                    onChange={(e) => setBins(Number(e.target.value) || 20)}
                  />
                </label>
              </>
            ) : (
              <>
                <label className="form-row">
                  <span>X</span>
                  <select value={x} onChange={(e) => setX(e.target.value)}>
                    {columns.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="form-row">
                  <span>Y</span>
                  <select value={y} onChange={(e) => setY(e.target.value)}>
                    {columns.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </label>
              </>
            )}
            <label className="form-row">
              <span>Title</span>
              <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Optional" />
            </label>
            <div className="modal-footer">
              <button className="btn" type="button" onClick={onClose}>
                Cancel
              </button>
              <button className="btn btn-primary" type="submit" disabled={!x || (type !== 'hist' && !y)}>
                Plot
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  )
}
