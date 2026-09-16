import type { AnalysisIssue } from './Analysis'

export interface AnalysisPaneProps {
  issues: AnalysisIssue[]
  onJump: (line: number) => void
  onRefresh: () => void
}

export function AnalysisPane({ issues, onJump, onRefresh }: AnalysisPaneProps) {
  return (
    <div className="analysis-pane">
      <div className="panel-header">
        <span>Analysis ({issues.length})</span>
        <button className="btn" type="button" onClick={onRefresh}>
          Refresh
        </button>
      </div>
      <div className="panel-body">
        {issues.length === 0 ? (
          <p className="muted" style={{ padding: 12 }}>
            No issues. TypeScript diagnostics + Studio heuristics.
          </p>
        ) : (
          <ul className="analysis-list">
            {issues.map((iss) => (
              <li key={iss.id}>
                <button type="button" className={`analysis-item sev-${iss.severity}`} onClick={() => onJump(iss.line)}>
                  <span className="analysis-sev">{iss.severity}</span>
                  <span className="analysis-msg">{iss.message}</span>
                  <span className="muted">
                    :{iss.line} · {iss.source}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
