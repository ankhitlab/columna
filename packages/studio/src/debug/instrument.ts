/** Inject await __dbg.line(N, () => ({...})) before each runnable line. */
export function instrumentForDebug(jsCode: string, localNames: string[]): string {
  const lines = jsCode.split('\n')
  const locObj = `{ ${localNames
    .filter((n) => /^[A-Za-z_$][\w$]*$/.test(n))
    .map((n) => `${JSON.stringify(n)}: (typeof ${n}!=='undefined'?${n}:undefined)`)
    .join(', ')} }`
  const out: string[] = []

  for (let i = 0; i < lines.length; i++) {
    const lineNo = i + 1
    const raw = lines[i]!
    const trimmed = raw.trim()
    if (!trimmed || trimmed.startsWith('//') || trimmed.startsWith('*') || trimmed.startsWith('/*')) {
      out.push(raw)
      continue
    }

    const fnMatch = trimmed.match(/^(async\s+)?function\s+([A-Za-z_$][\w$]*)/)
    out.push(`await __dbg.line(${lineNo}, () => (${locObj}));`)
    if (fnMatch) {
      out.push(`__dbg.push(${JSON.stringify(fnMatch[2])}, ${lineNo});`)
    }
    out.push(raw)
    // Heuristic: closing brace alone after a function body
    if (trimmed === '}' && fnMatch === null) {
      out.push(`try { __dbg.pop(); } catch {}`)
    }
  }

  return out.join('\n')
}
