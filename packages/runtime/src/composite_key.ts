/**
 * Ambiguity-free composite key encoding for Map/Set join / groupBy / unique paths.
 *
 * Avoids classic collisions from joining with raw `\0` and a null sentinel `∅`:
 * - null vs the literal string "∅"
 * - ["a\0b","c"] vs ["a","b\0c"]
 *
 * Wire format per part: `<tag><decimalLength>:<payload>`
 *   N0:          null
 *   S<n>:<chars> string
 *   n<n>:<chars> number (canonical String(n); NaN / -0 preserved)
 *   B1:0|B1:1    boolean
 */
export type KeyPart = string | number | boolean | null

export function encodeCompositeKey(parts: readonly KeyPart[]): string {
  let out = ''
  for (const part of parts) {
    if (part === null) {
      out += 'N0:'
      continue
    }
    if (typeof part === 'string') {
      out += `S${part.length}:${part}`
      continue
    }
    if (typeof part === 'number') {
      const payload = Object.is(part, -0) ? '-0' : String(part)
      out += `n${payload.length}:${payload}`
      continue
    }
    out += part ? 'B1:1' : 'B1:0'
  }
  return out
}
