export interface CodeCell {
  index: number
  startLine: number
  endLine: number
  title: string
  code: string
}

const CELL_RE = /^\s*(?:\/\/|#)\s*%%\s*(.*)$/

export function isCellMarker(line: string): boolean {
  return CELL_RE.test(line)
}

export function parseCells(source: string): CodeCell[] {
  const lines = source.split('\n')
  const markers: { line: number; title: string }[] = []

  for (let i = 0; i < lines.length; i++) {
    const m = lines[i]!.match(CELL_RE)
    if (m) markers.push({ line: i + 1, title: (m[1] ?? '').trim() || `Cell ${markers.length + 1}` })
  }

  if (markers.length === 0) {
    return [
      {
        index: 0,
        startLine: 1,
        endLine: Math.max(1, lines.length),
        title: 'Script',
        code: source,
      },
    ]
  }

  const cells: CodeCell[] = []
  for (let i = 0; i < markers.length; i++) {
    const startLine = markers[i]!.line
    const endLine = i + 1 < markers.length ? markers[i + 1]!.line - 1 : lines.length
    const body = lines.slice(startLine, endLine).join('\n')
    cells.push({
      index: i,
      startLine,
      endLine: Math.max(startLine, endLine),
      title: markers[i]!.title,
      code: body,
    })
  }
  return cells
}

export function cellAtLine(source: string, lineNumber: number): CodeCell {
  const cells = parseCells(source)
  for (let i = cells.length - 1; i >= 0; i--) {
    const c = cells[i]!
    if (lineNumber >= c.startLine) return c
  }
  return cells[0]!
}

export function nextCell(source: string, lineNumber: number): CodeCell | null {
  const cells = parseCells(source)
  const current = cellAtLine(source, lineNumber)
  return cells[current.index + 1] ?? null
}
