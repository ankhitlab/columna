import { useMemo, useState } from 'react'
import { searchInContent, type SearchHit } from './searchLib'

export interface FindInFilesProps {
  files: { path: string; content: string }[]
  onOpen: (path: string, line: number) => void
}

export function FindInFiles({ files, onOpen }: FindInFilesProps) {
  const [query, setQuery] = useState('')
  const [regex, setRegex] = useState(false)
  const [caseSensitive, setCaseSensitive] = useState(false)
  const [submitted, setSubmitted] = useState('')

  const hits = useMemo(() => {
    if (!submitted) return [] as SearchHit[]
    return files.flatMap((f) =>
      searchInContent(f.path, f.content, { query: submitted, regex, caseSensitive }),
    )
  }, [files, submitted, regex, caseSensitive])

  return (
    <div className="find-pane">
      <div className="panel-header">
        <span>Find in files</span>
      </div>
      <div className="find-form">
        <input
          className="toolbar-input"
          style={{ flex: 1 }}
          value={query}
          placeholder="Search…"
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') setSubmitted(query)
          }}
        />
        <label className="muted">
          <input type="checkbox" checked={regex} onChange={(e) => setRegex(e.target.checked)} /> regex
        </label>
        <label className="muted">
          <input
            type="checkbox"
            checked={caseSensitive}
            onChange={(e) => setCaseSensitive(e.target.checked)}
          />{' '}
          case
        </label>
        <button className="btn btn-primary" type="button" onClick={() => setSubmitted(query)}>
          Find
        </button>
      </div>
      <div className="panel-body find-results">
        {!submitted ? (
          <p className="muted" style={{ padding: 12 }}>
            Search across project / session files.
          </p>
        ) : hits.length === 0 ? (
          <p className="muted" style={{ padding: 12 }}>
            No matches.
          </p>
        ) : (
          hits.slice(0, 500).map((h, i) => (
            <button
              key={`${h.path}:${h.line}:${i}`}
              type="button"
              className="find-hit"
              onClick={() => onOpen(h.path, h.line)}
            >
              <span className="find-loc">
                {h.path}:{h.line}
              </span>
              <span className="find-text">{h.text}</span>
            </button>
          ))
        )}
      </div>
    </div>
  )
}
