/**
 * `JSON.parse` that does not round large integers.
 *
 * `JSON.parse('{"id": 9007199254740993}')` yields 9007199254740992 — silently. Here every integer literal
 * (no fraction, no exponent) outside ±(2^53 − 1) comes back as a `BigInt` instead, so the reader's Int64Policy
 * (`int64: 'error' | 'string' | 'number'`) decides what happens to it. Texts without a run of 16+ digits take
 * `JSON.parse` directly (a safe integer has at most 16 digits, and `2^53 − 1` has 16).
 *
 * How: a token-aware pass quotes each unsafe literal with a per-call random sentinel, `JSON.parse` runs, and
 * the sentinel strings are turned into BigInts. The number of sentinels found must equal the number written,
 * so a genuine string that happened to contain the sentinel is detected rather than misread.
 */
export function parseJsonExact(text: string): unknown {
  if (!/\d{16}/.test(text)) return JSON.parse(text)
  let sentinel = ''
  do sentinel = `\u0001i64_${Math.random().toString(36).slice(2)}${Math.random().toString(36).slice(2)}:`
  while (text.includes(sentinel))

  let out = ''
  let last = 0
  let written = 0
  const n = text.length
  let i = 0
  while (i < n) {
    const c = text.charCodeAt(i)
    if (c === 34 /* " */) {
      i++
      while (i < n) {
        const d = text.charCodeAt(i)
        if (d === 92 /* \ */) i += 2
        else if (d === 34) {
          i++
          break
        } else i++
      }
      continue
    }
    if (c === 45 /* - */ || (c >= 48 && c <= 57)) {
      const start = i
      if (c === 45) i++
      while (i < n && text.charCodeAt(i) >= 48 && text.charCodeAt(i) <= 57) i++
      const intEnd = i
      let isInt = true
      while (i < n) {
        const d = text.charCodeAt(i)
        if (d === 46 || d === 101 || d === 69 || d === 43 || d === 45 || (d >= 48 && d <= 57)) {
          if (d === 46 || d === 101 || d === 69) isInt = false
          i++
        } else break
      }
      if (isInt && intEnd === i) {
        const digits = intEnd - start - (c === 45 ? 1 : 0)
        if (digits >= 16) {
          const literal = text.slice(start, intEnd)
          const v = BigInt(literal)
          if (v > 9007199254740991n || v < -9007199254740991n) {
            out += text.slice(last, start) + JSON.stringify(sentinel + literal)
            last = intEnd
            written++
          }
        }
      }
      continue
    }
    i++
  }
  if (written === 0) return JSON.parse(text)
  out += text.slice(last)

  let found = 0
  const revived = JSON.parse(out, (_k, v: unknown) => {
    if (typeof v === 'string' && v.startsWith(sentinel)) {
      found++
      return BigInt(v.slice(sentinel.length))
    }
    return v
  })
  if (found !== written) throw new Error('parseJsonExact: large-integer rewrite could not be verified (sentinel collision); refusing to guess')
  return revived
}
