export interface SearchHit {
  path: string
  line: number
  column: number
  text: string
}

export interface SearchOptions {
  query: string
  regex?: boolean
  caseSensitive?: boolean
}

export function searchInContent(
  path: string,
  content: string,
  opts: SearchOptions,
): SearchHit[] {
  const q = opts.query
  if (!q) return []
  const hits: SearchHit[] = []
  const lines = content.split('\n')
  let re: RegExp | null = null
  if (opts.regex) {
    try {
      re = new RegExp(q, opts.caseSensitive ? 'g' : 'gi')
    } catch {
      return []
    }
  }
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!
    if (re) {
      re.lastIndex = 0
      const m = re.exec(line)
      if (m) hits.push({ path, line: i + 1, column: (m.index ?? 0) + 1, text: line.trimEnd() })
    } else {
      const hay = opts.caseSensitive ? line : line.toLowerCase()
      const needle = opts.caseSensitive ? q : q.toLowerCase()
      const idx = hay.indexOf(needle)
      if (idx >= 0) hits.push({ path, line: i + 1, column: idx + 1, text: line.trimEnd() })
    }
  }
  return hits
}

export function renameIdentifierInContent(
  content: string,
  from: string,
  to: string,
): { content: string; count: number } {
  if (!from || !to || from === to) return { content, count: 0 }
  if (!/^[A-Za-z_$][\w$]*$/.test(from) || !/^[A-Za-z_$][\w$]*$/.test(to)) {
    return { content, count: 0 }
  }
  const re = new RegExp(`\\b${from.replace(/\$/g, '\\$')}\\b`, 'g')
  let count = 0
  const next = content.replace(re, () => {
    count += 1
    return to
  })
  return { content: next, count }
}
