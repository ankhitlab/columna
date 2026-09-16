import { useEffect, useMemo, useState } from 'react'
import { DataFrame } from 'columna'
import { headRows, toCsvText } from '../dfAccess'
import { isDataFrame } from '../types'

export interface DataFrameViewerProps {
  name: string | null
  value: unknown
  onPlot?: (name: string) => void
  onOpenWindow?: (name: string) => void
  /** Full-height layout for detached Variable Explorer window */
  expanded?: boolean
  /** Spyder-style in-place cell editing (writes back via onChange) */
  editable?: boolean
  onChange?: (name: string, next: DataFrame) => void
}

type SortDir = 'asc' | 'desc' | null

interface EditCell {
  rowIndex: number
  col: string
  text: string
}

const EDIT_ROW_CAP = 50_000

export function DataFrameViewer({
  name,
  value,
  onPlot,
  onOpenWindow,
  expanded,
  editable,
  onChange,
}: DataFrameViewerProps) {
  const [limit, setLimit] = useState(200)
  const [filter, setFilter] = useState('')
  const [sortCol, setSortCol] = useState<string | null>(null)
  const [sortDir, setSortDir] = useState<SortDir>(null)
  const [showDescribe, setShowDescribe] = useState(false)
  const [draftRows, setDraftRows] = useState<Record<string, unknown>[] | null>(null)
  const [editing, setEditing] = useState<EditCell | null>(null)

  const df = isDataFrame(value) ? value : null
  const tooLarge = Boolean(df && df.shape[0] > EDIT_ROW_CAP)
  const canEdit = Boolean(editable && onChange && name && df && !tooLarge)

  useEffect(() => {
    setLimit(200)
    setFilter('')
    setSortCol(null)
    setSortDir(null)
    setShowDescribe(false)
    setEditing(null)
  }, [name])

  useEffect(() => {
    if (!df || !canEdit) {
      setDraftRows(null)
      return
    }
    // Don't clobber an in-progress cell edit
    if (editing) return
    setDraftRows(headRows(df, Math.min(df.shape[0], EDIT_ROW_CAP)))
  }, [df, canEdit, editing])

  const sourceRows = useMemo(() => {
    if (draftRows) return draftRows
    if (!df) return []
    return headRows(df, Math.min(df.shape[0], 5000))
  }, [df, draftRows])

  const indexed = useMemo(
    () => sourceRows.map((row, i) => ({ i, row })),
    [sourceRows],
  )

  const processed = useMemo(() => {
    let rows = indexed
    const q = filter.trim().toLowerCase()
    if (q) {
      rows = rows.filter(({ row }) =>
        Object.values(row).some((v) => String(v ?? '').toLowerCase().includes(q)),
      )
    }
    if (sortCol && sortDir) {
      const dir = sortDir === 'asc' ? 1 : -1
      rows = [...rows].sort((a, b) => compareCells(a.row[sortCol], b.row[sortCol]) * dir)
    }
    return rows
  }, [indexed, filter, sortCol, sortDir])

  const rows = processed.slice(0, limit)
  const describeStats = useMemo(
    () => (df && showDescribe ? describeFrame(df, sourceRows) : []),
    [df, showDescribe, sourceRows],
  )

  if (!name || !df) {
    return (
      <div className="panel" style={{ border: 'none', height: '100%' }}>
        <div className="panel-header">
          <span>DataFrame Viewer</span>
        </div>
        <div className="panel-body">
          <p className="muted" style={{ padding: 12 }}>
            Select a DataFrame in Variable Explorer.
          </p>
        </div>
      </div>
    )
  }

  const [r, c] = df.shape
  const truncated = r > sourceRows.length

  const copyCsv = async () => {
    await navigator.clipboard.writeText(toCsvText(df))
  }

  const downloadCsv = () => {
    const blob = new Blob([toCsvText(df)], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${name}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const toggleSort = (col: string) => {
    if (sortCol !== col) {
      setSortCol(col)
      setSortDir('asc')
      return
    }
    if (sortDir === 'asc') setSortDir('desc')
    else if (sortDir === 'desc') {
      setSortCol(null)
      setSortDir(null)
    } else setSortDir('asc')
  }

  const commitEdit = () => {
    if (!editing || !canEdit || !draftRows) {
      setEditing(null)
      return
    }
    const { rowIndex, col, text } = editing
    const dtype = String(df.dtypes[col] ?? 'utf8')
    const prev = draftRows[rowIndex]?.[col]
    const nextVal = parseCellInput(text, dtype, prev)
    if (Object.is(nextVal, prev) || (nextVal === prev)) {
      setEditing(null)
      return
    }
    const nextRows = draftRows.map((row, i) =>
      i === rowIndex ? { ...row, [col]: nextVal } : row,
    )
    setDraftRows(nextRows)
    setEditing(null)
    onChange!(name, DataFrame.fromRows(nextRows))
  }

  const startEdit = (rowIndex: number, col: string) => {
    if (!canEdit) return
    const current = draftRows?.[rowIndex]?.[col]
    setEditing({
      rowIndex,
      col,
      text: current == null ? '' : String(current),
    })
  }

  return (
    <div className={`panel df-viewer ${expanded ? 'df-viewer-expanded' : ''}`} style={{ border: 'none', height: '100%' }}>
      <div className="panel-header">
        <span>
          DataFrame · {name}
          {canEdit ? ' · editable' : ''}
        </span>
        <div className="panel-tools">
          <input
            className="toolbar-input"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter rows…"
          />
          <button className="btn" type="button" onClick={() => setShowDescribe((v) => !v)}>
            {showDescribe ? 'Hide describe' : 'Describe'}
          </button>
          <button className="btn" type="button" onClick={() => void copyCsv()}>
            Copy CSV
          </button>
          <button className="btn" type="button" onClick={downloadCsv}>
            Download
          </button>
          {onPlot && (
            <button className="btn" type="button" onClick={() => onPlot(name)}>
              Plot
            </button>
          )}
          {onOpenWindow && (
            <button
              className="btn btn-primary"
              type="button"
              title="Open in separate window"
              onClick={() => onOpenWindow(name)}
            >
              Window
            </button>
          )}
        </div>
      </div>
      <div className="df-meta">
        <span className="chip">
          {r} × {c}
        </span>
        {filter && <span className="chip">{processed.length} matched</span>}
        {truncated && <span className="chip">showing {sourceRows.length.toLocaleString()} rows</span>}
        {editable && tooLarge && (
          <span className="chip">editing disabled (&gt;{EDIT_ROW_CAP.toLocaleString()} rows)</span>
        )}
        {canEdit && <span className="chip">double-click cell to edit</span>}
        {df.columns.map((col) => (
          <span className="chip" key={col}>
            {col}:{df.dtypes[col]}
          </span>
        ))}
      </div>
      {showDescribe && (
        <div className="df-describe">
          <table className="df-table">
            <thead>
              <tr>
                <th>column</th>
                <th>dtype</th>
                <th>count</th>
                <th>nulls</th>
                <th>min</th>
                <th>max</th>
                <th>mean</th>
              </tr>
            </thead>
            <tbody>
              {describeStats.map((s) => (
                <tr key={s.column}>
                  <td>{s.column}</td>
                  <td>{s.dtype}</td>
                  <td>{s.count}</td>
                  <td>{s.nulls}</td>
                  <td>{s.min ?? '—'}</td>
                  <td>{s.max ?? '—'}</td>
                  <td>{s.mean ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="df-table-wrap panel-body">
        <table className={`df-table ${canEdit ? 'df-table-editable' : ''}`}>
          <thead>
            <tr>
              <th>#</th>
              {df.columns.map((col) => (
                <th key={col} className="sortable" onClick={() => toggleSort(col)}>
                  {col}
                  {sortCol === col ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ i, row }) => (
              <tr key={i}>
                <td className="muted">{i}</td>
                {df.columns.map((col) => {
                  const isEditing = editing?.rowIndex === i && editing.col === col
                  return (
                    <td
                      key={col}
                      className={canEdit ? 'df-cell-editable' : undefined}
                      onDoubleClick={() => startEdit(i, col)}
                      title={canEdit ? 'Double-click to edit' : undefined}
                    >
                      {isEditing ? (
                        <input
                          className="df-cell-input"
                          autoFocus
                          value={editing.text}
                          onChange={(e) => setEditing({ ...editing, text: e.target.value })}
                          onBlur={commitEdit}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                              e.preventDefault()
                              commitEdit()
                            }
                            if (e.key === 'Escape') {
                              e.preventDefault()
                              setEditing(null)
                            }
                            e.stopPropagation()
                          }}
                          onClick={(e) => e.stopPropagation()}
                        />
                      ) : (
                        formatCell(row[col])
                      )}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {processed.length > limit && (
        <div style={{ padding: 8 }}>
          <button className="btn" type="button" onClick={() => setLimit((n) => n + 200)}>
            Load more
          </button>
        </div>
      )}
    </div>
  )
}

function parseCellInput(text: string, dtype: string, original: unknown): unknown {
  const t = text.trim()
  if (t === '' || t === '∅' || t.toLowerCase() === 'null' || t.toLowerCase() === 'none') return null

  const d = dtype.toLowerCase()
  if (d.includes('bool') || typeof original === 'boolean') {
    const low = t.toLowerCase()
    if (low === 'true' || low === '1' || low === 'yes') return true
    if (low === 'false' || low === '0' || low === 'no') return false
    return original
  }
  if (
    d.includes('int') ||
    d.includes('float') ||
    d.includes('double') ||
    d === 'number' ||
    typeof original === 'number'
  ) {
    const n = Number(t.replace(',', '.'))
    return Number.isFinite(n) ? n : original
  }
  return text
}

function formatCell(v: unknown): string {
  if (v == null) return '∅'
  if (typeof v === 'number') return Number.isFinite(v) ? String(v) : String(v)
  return String(v)
}

function compareCells(a: unknown, b: unknown): number {
  if (a == null && b == null) return 0
  if (a == null) return -1
  if (b == null) return 1
  if (typeof a === 'number' && typeof b === 'number') return a - b
  return String(a).localeCompare(String(b), undefined, { numeric: true })
}

interface ColStat {
  column: string
  dtype: string
  count: number
  nulls: number
  min: string | null
  max: string | null
  mean: string | null
}

function describeFrame(df: DataFrame, rows: Record<string, unknown>[]): ColStat[] {
  return df.columns.map((column) => {
    const dtype = String(df.dtypes[column] ?? 'unknown')
    let nulls = 0
    const nums: number[] = []
    let minS: string | null = null
    let maxS: string | null = null
    for (const row of rows) {
      const v = row[column]
      if (v == null || v === '') {
        nulls += 1
        continue
      }
      const n = typeof v === 'number' ? v : Number(v)
      if (
        Number.isFinite(n) &&
        (dtype.includes('int') || dtype.includes('float') || dtype === 'number' || typeof v === 'number')
      ) {
        nums.push(n)
      }
      const s = String(v)
      if (minS == null || s < minS) minS = s
      if (maxS == null || s > maxS) maxS = s
    }
    const count = rows.length - nulls
    const mean = nums.length > 0 ? (nums.reduce((a, b) => a + b, 0) / nums.length).toFixed(4) : null
    return {
      column,
      dtype,
      count,
      nulls,
      min: nums.length ? String(Math.min(...nums)) : minS,
      max: nums.length ? String(Math.max(...nums)) : maxS,
      mean,
    }
  })
}
