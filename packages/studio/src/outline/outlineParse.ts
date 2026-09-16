import { isCellMarker, parseCells } from '../editor/cells'

export type OutlineKind = 'cell' | 'function' | 'class' | 'const'

export interface OutlineItem {
  kind: OutlineKind
  name: string
  line: number
}

export function parseOutline(source: string): OutlineItem[] {
  const items: OutlineItem[] = []
  const lines = source.split('\n')

  for (const cell of parseCells(source)) {
    if (cell.title !== 'Script' || isCellMarker(lines[0] ?? '')) {
      const first = lines[cell.startLine - 1] ?? ''
      if (isCellMarker(first) || cell.title !== 'Script') {
        items.push({ kind: 'cell', name: cell.title, line: cell.startLine })
      }
    }
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!
    const lineNo = i + 1

    let m = line.match(/^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)/)
    if (m) {
      items.push({ kind: 'function', name: m[1]!, line: lineNo })
      continue
    }

    m = line.match(/^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)/)
    if (m) {
      items.push({ kind: 'class', name: m[1]!, line: lineNo })
      continue
    }

    m = line.match(/^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=/)
    if (m) {
      items.push({ kind: 'const', name: m[1]!, line: lineNo })
    }
  }

  return items.sort((a, b) => a.line - b.line)
}
