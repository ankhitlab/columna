/**
 * Alias structure of two-level fractional factorial designs (Minitab DOE › Create Factorial Design ›
 * "Display alias table"): the defining relation from the generators, resolution, and the aliases of every
 * effect up to a chosen order. Letters A, B, C, … name the factors; generators like 'E=ABCD' or 'ABCDE'.
 */

export interface AliasStructure {
  factors: string[]
  /** Words of the defining relation (I = ABCE = BCDF = …). */
  definingRelation: string[]
  resolution: number
  /** Aliases per effect (words up to `maxOrder`), e.g. { A: ['BCE', 'ABCDF'] }. */
  aliases: Record<string, string[]>
  /** Alias chains as Minitab prints them: 'A + BCE + ...'. */
  chains: string[]
  runs: number
  fraction: string
}

function wordToMask(word: string, k: number): number {
  let m = 0
  for (const ch of word.toUpperCase()) {
    const i = ch.charCodeAt(0) - 65
    if (i < 0 || i >= k) throw new RangeError(`aliasStructure: unknown factor letter ${ch}`)
    m ^= 1 << i
  }
  return m
}

function maskToWord(mask: number): string {
  let w = ''
  for (let i = 0; i < 26; i++) if (mask & (1 << i)) w += String.fromCharCode(65 + i)
  return w || 'I'
}

function bits(mask: number): number {
  let c = 0
  while (mask) {
    c += mask & 1
    mask >>= 1
  }
  return c
}

/**
 * Alias structure for k factors and p generators (2^(k−p) design). maxOrder (default 2) limits the effects
 * listed; chainOrder (default max(3, maxOrder)) limits the interaction order shown in each alias chain.
 *   aliasStructure(5, ['E=ABCD'])                 → resolution V, A aliased with BCDE, …
 *   aliasStructure(['A','B','C','D','E','F'], ['E=ABC', 'F=BCD'], { maxOrder: 2 })
 */
export function aliasStructure(factors: number | string[], generators: string[], options: { maxOrder?: number; chainOrder?: number } = {}): AliasStructure {
  const names = typeof factors === 'number' ? Array.from({ length: factors }, (_, i) => String.fromCharCode(65 + i)) : factors
  const k = names.length
  if (k < 2 || k > 20) throw new RangeError('aliasStructure: 2…20 factors')
  const p = generators.length
  if (p < 1 || p >= k) throw new RangeError('aliasStructure: 1 ≤ generators < factors')
  const genMasks = generators.map((g) => wordToMask(g.replace(/=/g, ''), k))
  // defining relation: all non-empty products of the generator words
  const words = new Set<number>()
  for (let m = 1; m < 1 << p; m++) {
    let w = 0
    for (let j = 0; j < p; j++) if (m & (1 << j)) w ^= genMasks[j]!
    if (w) words.add(w)
  }
  const byWord = (a: number, b: number) => bits(a) - bits(b) || (maskToWord(a) < maskToWord(b) ? -1 : 1)
  const relation = [...words].sort(byWord)
  const resolution = Math.min(...relation.map(bits))
  const maxOrder = options.maxOrder ?? 2
  const aliases: Record<string, string[]> = {}
  const chains: string[] = []
  const seen = new Set<number>()
  const effects: number[] = []
  for (let m = 1; m < 1 << k; m++) if (bits(m) <= maxOrder) effects.push(m)
  effects.sort(byWord)
  const chainOrder = options.chainOrder ?? Math.max(3, maxOrder)
  const label = maskToWord
  for (const e of effects) {
    if (seen.has(e)) continue
    const group = relation.map((w) => e ^ w).filter((a) => a !== 0)
    const all = [e, ...group].sort(byWord)
    for (const a of all) seen.add(a)
    const partners = group.filter((a) => bits(a) <= chainOrder)
    aliases[label(e)] = group.sort(byWord).map(label)
    chains.push([e, ...partners.sort(byWord)].map(label).join(' + '))
  }
  return {
    factors: names,
    definingRelation: relation.map(maskToWord),
    resolution,
    aliases,
    chains,
    runs: 1 << (k - p),
    fraction: `2^(${k}-${p})`,
  }
}
