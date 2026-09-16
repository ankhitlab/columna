export type AnalysisSeverity = 'error' | 'warning' | 'info'

export interface AnalysisIssue {
  id: string
  severity: AnalysisSeverity
  message: string
  line: number
  column: number
  source: 'typescript' | 'studio'
}

/** Studio heuristics on top of Monaco TS markers. */
export function analyzeStudioHeuristics(source: string): AnalysisIssue[] {
  const issues: AnalysisIssue[] = []
  const lines = source.split('\n')

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!
    const lineNo = i + 1
    if (/\bdebugger\b/.test(line)) {
      issues.push({
        id: `dbg-${lineNo}`,
        severity: 'warning',
        message: 'Leftover debugger statement',
        line: lineNo,
        column: line.indexOf('debugger') + 1,
        source: 'studio',
      })
    }
    // LazyFrame-ish chain ending without await/collect
    if (
      /\.(filter|select|groupBy|sort|withColumns)\s*\(/.test(line) &&
      !/\bawait\b/.test(line) &&
      !/\.collect\s*\(/.test(line) &&
      !/;\s*$/.test(line.trim()) &&
      i + 1 < lines.length &&
      !/^\s*\./.test(lines[i + 1]!) &&
      !/\.collect\b/.test(lines.slice(i, i + 6).join('\n'))
    ) {
      // only flag if looks like assignment of chain that never collects in next few lines
      if (/^\s*(const|let|var)\s+\w+\s*=/.test(line) && !/\.collect\b/.test(source.slice(source.indexOf(line)))) {
        issues.push({
          id: `await-${lineNo}`,
          severity: 'info',
          message: 'Lazy pipeline may need .collect() / await',
          line: lineNo,
          column: 1,
          source: 'studio',
        })
      }
    }
  }

  // Empty cells
  const cellStarts: number[] = []
  for (let i = 0; i < lines.length; i++) {
    if (/^\s*(?:\/\/|#)\s*%%/.test(lines[i]!)) cellStarts.push(i)
  }
  for (let c = 0; c < cellStarts.length; c++) {
    const start = cellStarts[c]!
    const end = c + 1 < cellStarts.length ? cellStarts[c + 1]! : lines.length
    const body = lines.slice(start + 1, end).join('\n').trim()
    if (!body) {
      issues.push({
        id: `cell-${start + 1}`,
        severity: 'info',
        message: 'Empty cell',
        line: start + 1,
        column: 1,
        source: 'studio',
      })
    }
  }

  // Unused simple const (declared never referenced again) — very light heuristic
  for (const m of source.matchAll(/\b(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=/g)) {
    const name = m[1]!
    if (name.startsWith('_')) continue
    const re = new RegExp(`\\b${name}\\b`, 'g')
    const count = [...source.matchAll(re)].length
    if (count <= 1) {
      const before = source.slice(0, m.index ?? 0)
      const line = before.split('\n').length
      issues.push({
        id: `unused-${name}-${line}`,
        severity: 'info',
        message: `'${name}' is assigned but never used`,
        line,
        column: 1,
        source: 'studio',
      })
    }
  }

  return issues
}

export function markersToIssues(
  markers: { message: string; startLineNumber: number; startColumn: number; severity: number }[],
): AnalysisIssue[] {
  return markers.map((m, i) => ({
    id: `ts-${i}-${m.startLineNumber}`,
    severity: m.severity >= 8 ? 'error' : m.severity >= 4 ? 'warning' : 'info',
    message: m.message,
    line: m.startLineNumber,
    column: m.startColumn,
    source: 'typescript',
  }))
}
