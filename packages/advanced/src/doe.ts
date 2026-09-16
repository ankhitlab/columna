/**
 * Tier 5.2 — Design of Experiments:
 * full / fractional factorial, Plackett–Burman, CCD, Box–Behnken, Taguchi L-arrays;
 * effect estimates with Lenth PSE and Pareto / normal-plot coordinates.
 */
import { f as fDist, normal } from './dist.js'
import { lstsq, matrix } from './linalg.js'
import { linearModel, type LinearModelResult } from './lm.js'

const STD = normal()

export interface DoeDesign {
  type: string
  /** Factor names. */
  factors: string[]
  /** Coded design matrix (−1 / 0 / +1 or Taguchi levels 1…L), rows = runs. */
  matrix: number[][]
  /** Generator / resolution info when applicable. */
  info?: Record<string, unknown>
}

function fullFactorialLevels(k: number): number[][] {
  const n = 1 << k
  const rows: number[][] = []
  for (let i = 0; i < n; i++) {
    const row: number[] = []
    for (let j = 0; j < k; j++) row.push(i & (1 << (k - 1 - j)) ? 1 : -1)
    rows.push(row)
  }
  return rows
}

/** 2^k full factorial in standard order. */
export function fullFactorial(factors: string[]): DoeDesign {
  if (factors.length < 1 || factors.length > 12) throw new RangeError('fullFactorial: 1…12 factors')
  return { type: 'full-factorial', factors: factors.slice(), matrix: fullFactorialLevels(factors.length), info: { runs: 1 << factors.length } }
}

/**
 * 2^{k−p} fractional factorial. Generators like `['ABD', 'ACE']` define column D = A·B, E = A·C
 * (letters map to factor indices in order). Resolution is inferred from the shortest word.
 */
export function fractionalFactorial(factors: string[], generators: string[]): DoeDesign {
  const k = factors.length
  const p = generators.length
  if (p < 1) throw new RangeError('fractionalFactorial: provide generators')
  const basic = k - p
  if (basic < 1) throw new RangeError('fractionalFactorial: too many generators')
  const base = fullFactorialLevels(basic)
  const letter = (ch: string) => {
    const i = ch.toUpperCase().charCodeAt(0) - 65
    if (i < 0 || i >= k) throw new RangeError(`fractionalFactorial: unknown letter ${ch}`)
    return i
  }
  const runs = base.map((row) => {
    const full = new Array(k).fill(0)
    for (let j = 0; j < basic; j++) full[j] = row[j]!
    for (let g = 0; g < p; g++) {
      const word = generators[g]!.replace(/=/g, '')
      // last letter is the generated factor; product of the preceding
      const letters = [...word.toUpperCase()]
      const genIdx = letter(letters[letters.length - 1]!)
      let prod = 1
      for (let i = 0; i < letters.length - 1; i++) prod *= full[letter(letters[i]!)]!
      full[genIdx] = prod
    }
    return full
  })
  // resolution = length of shortest word in generators (approx)
  const res = Math.min(...generators.map((g) => g.replace(/=/g, '').length))
  return { type: 'fractional-factorial', factors: factors.slice(), matrix: runs, info: { generators, resolution: res, runs: runs.length } }
}

/** Plackett–Burman designs for n − 1 factors in n runs (n ≡ 0 mod 4, n ≤ 48). */
export function plackettBurman(factors: string[]): DoeDesign {
  if (!Array.isArray(factors) || factors.length === 0) throw new RangeError('plackettBurman: factors must be a non-empty array of names')
  const f = factors.length
  const n = Math.ceil((f + 1) / 4) * 4
  if (n > 48) throw new RangeError('plackettBurman: too many factors')
  // cyclic generators from known first rows (Montgomery / NIST)
  const firstRows: Record<number, number[]> = {
    8: [1, 1, 1, -1, 1, -1, -1],
    12: [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],
    16: [1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1],
    20: [1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1],
    24: [1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1],
  }
  if (!(n in firstRows) && n !== 4) {
    // construct Hadamard-like cyclic for n=4: ++-
    if (n === 4) firstRows[4] = [1, 1, -1]
    else throw new RangeError(`plackettBurman: no generator for ${n} runs`)
  }
  if (n === 4) firstRows[4] = [1, 1, -1]
  const first = firstRows[n]!
  const cols = n - 1
  const mat: number[][] = []
  let row = first.slice()
  for (let i = 0; i < cols; i++) {
    mat.push(row.slice(0, f).concat(new Array(Math.max(0, f - cols)).fill(0)).slice(0, f))
    // actually store full then trim columns
    void 0
    row = [row[row.length - 1]!, ...row.slice(0, row.length - 1)]
  }
  // rebuild properly
  const cycles: number[][] = []
  row = first.slice()
  for (let i = 0; i < cols; i++) {
    cycles.push(row.slice())
    row = [row[row.length - 1]!, ...row.slice(0, row.length - 1)]
  }
  // transpose-ish: each cycle is a row of the (n-1)×(n-1) cyclic; add a final −1 row
  const design: number[][] = cycles.map((r) => r.slice(0, f))
  design.push(new Array(f).fill(-1))
  return { type: 'plackett-burman', factors: factors.slice(), matrix: design, info: { runs: n } }
}

/** Central Composite Design (CCD): factorial + axial ±α + centre points. */
export function ccd(factors: string[], options: { alpha?: number | 'orthogonal' | 'rotatable'; centerPoints?: number } = {}): DoeDesign {
  const k = factors.length
  if (k < 2 || k > 6) throw new RangeError('ccd: 2…6 factors')
  const cube = fullFactorialLevels(k)
  const centerPoints = options.centerPoints ?? 4
  let alpha: number
  if (options.alpha === undefined || options.alpha === 'rotatable') alpha = 2 ** (k / 4)
  else if (options.alpha === 'orthogonal') {
    const nF = 1 << k
    const nC = centerPoints
    alpha = Math.sqrt(Math.sqrt(nF) * (Math.sqrt(nF + nC) - Math.sqrt(nF)))
  } else alpha = options.alpha
  const axial: number[][] = []
  for (let j = 0; j < k; j++) {
    const up = new Array(k).fill(0)
    const dn = new Array(k).fill(0)
    up[j] = alpha
    dn[j] = -alpha
    axial.push(up, dn)
  }
  const centres = Array.from({ length: centerPoints }, () => new Array(k).fill(0))
  return {
    type: 'ccd',
    factors: factors.slice(),
    matrix: [...cube, ...axial, ...centres],
    info: { alpha, centerPoints, runs: cube.length + axial.length + centerPoints },
  }
}

/** Box–Behnken design for 3…7 factors. */
export function boxBehnken(factors: string[], options: { centerPoints?: number } = {}): DoeDesign {
  const k = factors.length
  if (k < 3 || k > 7) throw new RangeError('boxBehnken: 3…7 factors')
  const centerPoints = options.centerPoints ?? 3
  const rows: number[][] = []
  for (let i = 0; i < k; i++) {
    for (let j = i + 1; j < k; j++) {
      for (const a of [-1, 1]) for (const b of [-1, 1]) {
        const row = new Array(k).fill(0)
        row[i] = a
        row[j] = b
        rows.push(row)
      }
    }
  }
  for (let c = 0; c < centerPoints; c++) rows.push(new Array(k).fill(0))
  return { type: 'box-behnken', factors: factors.slice(), matrix: rows, info: { centerPoints, runs: rows.length } }
}

/** Taguchi orthogonal arrays L4…L243 (standard + extended subset). */
export function taguchi(
  array:
    | 'L4' | 'L8' | 'L9' | 'L12' | 'L16' | 'L18' | 'L20' | 'L24' | 'L25' | 'L27' | 'L28'
    | 'L32' | 'L36' | 'L40' | 'L44' | 'L48' | 'L50' | 'L54' | 'L64' | 'L81'
    | 'L108' | 'L121' | 'L128' | 'L243',
  factorNames?: string[],
): DoeDesign {
  const pbToTaguchi = (first: number[]): number[][] => {
    const cols = first.length
    const cycles: number[][] = []
    let row = first.slice()
    for (let i = 0; i < cols; i++) {
      cycles.push(row.map((v) => (v === 1 ? 1 : 2)))
      row = [row[row.length - 1]!, ...row.slice(0, row.length - 1)]
    }
    cycles.push(new Array(cols).fill(2))
    return cycles
  }
  const tables: Record<string, number[][]> = {
    L4: [
      [1, 1, 1],
      [1, 2, 2],
      [2, 1, 2],
      [2, 2, 1],
    ],
    L8: [
      [1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 2, 2, 2, 2],
      [1, 2, 2, 1, 1, 2, 2],
      [1, 2, 2, 2, 2, 1, 1],
      [2, 1, 2, 1, 2, 1, 2],
      [2, 1, 2, 2, 1, 2, 1],
      [2, 2, 1, 1, 2, 2, 1],
      [2, 2, 1, 2, 1, 1, 2],
    ],
    L9: [
      [1, 1, 1, 1],
      [1, 2, 2, 2],
      [1, 3, 3, 3],
      [2, 1, 2, 3],
      [2, 2, 3, 1],
      [2, 3, 1, 2],
      [3, 1, 3, 2],
      [3, 2, 1, 3],
      [3, 3, 2, 1],
    ],
    L12: [
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
      [1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
      [1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2],
      [1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1],
      [1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1],
      [2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1],
      [2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2],
      [2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1],
      [2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2],
      [2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2],
      [2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1],
    ],
  }
  if (array === 'L20') {
    tables.L20 = pbToTaguchi([1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1])
  }
  if (array === 'L24') {
    tables.L24 = pbToTaguchi([
      1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1,
    ])
  }
  if (array === 'L28') {
    // PB(28)-like cyclic generator (length 27)
    tables.L28 = pbToTaguchi([
      1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1,
    ])
  }
  // L16 = 2-level 15 columns from full 2^4 with interactions as columns — use PB-like
  if (array === 'L16') {
    const ff = fullFactorialLevels(4)
    const cols: number[][] = []
    for (let mask = 1; mask < 16; mask++) {
      cols.push(ff.map((row) => {
        let p = 1
        for (let j = 0; j < 4; j++) if (mask & (1 << j)) p *= row[j]!
        return p === 1 ? 1 : 2
      }))
    }
    tables.L16 = ff.map((_, i) => cols.map((c) => c[i]!))
  }
  if (array === 'L18') {
    tables.L18 = [
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 2, 2, 2, 2, 2, 2],
      [1, 1, 3, 3, 3, 3, 3, 3],
      [1, 2, 1, 1, 2, 2, 3, 3],
      [1, 2, 2, 2, 3, 3, 1, 1],
      [1, 2, 3, 3, 1, 1, 2, 2],
      [1, 3, 1, 2, 1, 3, 2, 3],
      [1, 3, 2, 3, 2, 1, 3, 1],
      [1, 3, 3, 1, 3, 2, 1, 2],
      [2, 1, 1, 3, 3, 2, 2, 1],
      [2, 1, 2, 1, 1, 3, 3, 2],
      [2, 1, 3, 2, 2, 1, 1, 3],
      [2, 2, 1, 2, 3, 1, 3, 2],
      [2, 2, 2, 3, 1, 2, 1, 3],
      [2, 2, 3, 1, 2, 3, 2, 1],
      [2, 3, 1, 3, 2, 3, 1, 2],
      [2, 3, 2, 1, 3, 1, 2, 3],
      [2, 3, 3, 2, 1, 2, 3, 1],
    ]
  }
  if (array === 'L25') {
    // 5-level OA: 6 columns from 5² lattice
    const rows: number[][] = []
    for (let a = 0; a < 5; a++) for (let b = 0; b < 5; b++) {
      rows.push([
        a + 1,
        b + 1,
        ((a + b) % 5) + 1,
        ((a + 2 * b) % 5) + 1,
        ((a + 3 * b) % 5) + 1,
        ((a + 4 * b) % 5) + 1,
      ])
    }
    tables.L25 = rows
  }
  if (array === 'L27') {
    const rows: number[][] = []
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) for (let c = 0; c < 3; c++) {
      const cols = [
        a, b, (a + b) % 3, (a + 2 * b) % 3, c,
        (a + c) % 3, (a + 2 * c) % 3, (b + c) % 3, (b + 2 * c) % 3,
        (a + b + c) % 3, (a + b + 2 * c) % 3, (a + 2 * b + c) % 3, (a + 2 * b + 2 * c) % 3,
      ]
      rows.push(cols.map((v) => v + 1))
    }
    tables.L27 = rows
  }
  if (array === 'L32') {
    // 2-level OA: 31 columns from 2^5 factorial contrasts
    const ff = fullFactorialLevels(5)
    const cols: number[][] = []
    for (let mask = 1; mask < 32; mask++) {
      cols.push(ff.map((row) => {
        let p = 1
        for (let j = 0; j < 5; j++) if (mask & (1 << j)) p *= row[j]!
        return p === 1 ? 1 : 2
      }))
    }
    tables.L32 = ff.map((_, i) => cols.map((c) => c[i]!))
  }
  if (array === 'L36') {
    // Abbreviated mixed OA: col0 2-level, cols 1..11 three-level (3^2 lattice × repeats)
    const rows: number[][] = []
    for (let a = 0; a < 2; a++) {
      for (let b = 0; b < 3; b++) for (let c = 0; c < 3; c++) {
        const base = [
          a + 1,
          b + 1,
          c + 1,
          ((b + c) % 3) + 1,
          ((b + 2 * c) % 3) + 1,
          ((2 * b + c) % 3) + 1,
          ((2 * b + 2 * c) % 3) + 1,
          ((b + a) % 3) + 1,
          ((c + a) % 3) + 1,
          ((b + c + a) % 3) + 1,
          ((b + 2 * c + a) % 3) + 1,
          ((2 * b + c + a) % 3) + 1,
        ]
        rows.push(base)
        // duplicate pattern with shift for 36 runs (2×3×3×2)
        rows.push(base.map((v, j) => (j === 0 ? v : ((v + 1) % 3) + 1)))
      }
    }
    tables.L36 = rows.slice(0, 36)
  }
  if (array === 'L40') {
    // 2×L20 stack with column shift on second block
    const base = pbToTaguchi([1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1])
    const shifted = base.map((r) => r.map((v, j) => (j === 0 ? v : v === 1 ? 2 : 1)))
    tables.L40 = [...base, ...shifted]
  }
  if (array === 'L44') {
    tables.L44 = pbToTaguchi([
      1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1,
    ])
  }
  if (array === 'L48') {
    // 2×L24 stack
    const base = pbToTaguchi([
      1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1,
    ])
    const shifted = base.map((r) => r.map((v, j) => (j % 2 === 0 ? v : v === 1 ? 2 : 1)))
    tables.L48 = [...base, ...shifted]
  }
  if (array === 'L50') {
    // 5-level OA: 11 columns from 5² lattice + linear combinations mod 5
    const rows: number[][] = []
    for (let a = 0; a < 5; a++) for (let b = 0; b < 5; b++) {
      const cols = [
        a, b,
        (a + b) % 5, (a + 2 * b) % 5, (a + 3 * b) % 5, (a + 4 * b) % 5,
        (2 * a + b) % 5, (2 * a + 2 * b) % 5, (2 * a + 3 * b) % 5, (2 * a + 4 * b) % 5,
        (3 * a + b) % 5,
      ]
      rows.push(cols.map((v) => v + 1))
    }
    // 25 + 25 with shift for 50 runs
    const shifted = rows.map((r) => r.map((v) => ((v % 5) + 1)))
    tables.L50 = [...rows, ...shifted].slice(0, 50)
  }
  if (array === 'L54') {
    // Mixed OA: 1 two-level + 25 three-level columns (abbreviated 2×3³ construction)
    const rows: number[][] = []
    for (let a = 0; a < 2; a++) {
      for (let b = 0; b < 3; b++) for (let c = 0; c < 3; c++) for (let d = 0; d < 3; d++) {
        const cols = [
          a + 1,
          b + 1, c + 1, d + 1,
          ((b + c) % 3) + 1, ((b + 2 * c) % 3) + 1,
          ((b + d) % 3) + 1, ((b + 2 * d) % 3) + 1,
          ((c + d) % 3) + 1, ((c + 2 * d) % 3) + 1,
          ((b + c + d) % 3) + 1, ((b + c + 2 * d) % 3) + 1,
          ((b + 2 * c + d) % 3) + 1, ((b + 2 * c + 2 * d) % 3) + 1,
          ((2 * b + c + d) % 3) + 1, ((2 * b + c + 2 * d) % 3) + 1,
          ((2 * b + 2 * c + d) % 3) + 1, ((2 * b + 2 * c + 2 * d) % 3) + 1,
          ((a + b) % 3) + 1, ((a + c) % 3) + 1, ((a + d) % 3) + 1,
          ((a + b + c) % 3) + 1, ((a + b + d) % 3) + 1, ((a + c + d) % 3) + 1,
          ((a + b + c + d) % 3) + 1,
        ]
        rows.push(cols)
      }
    }
    tables.L54 = rows.slice(0, 54)
  }
  if (array === 'L64') {
    // 2-level OA: 63 columns from 2^6 factorial contrasts
    const ff = fullFactorialLevels(6)
    const cols: number[][] = []
    for (let mask = 1; mask < 64; mask++) {
      cols.push(ff.map((row) => {
        let p = 1
        for (let j = 0; j < 6; j++) if (mask & (1 << j)) p *= row[j]!
        return p === 1 ? 1 : 2
      }))
    }
    tables.L64 = ff.map((_, i) => cols.map((c) => c[i]!))
  }
  if (array === 'L81') {
    // 3-level OA: 40 columns from 3^4 lattice (abbreviated to first 40 contrasts)
    const rows: number[][] = []
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) for (let c = 0; c < 3; c++) for (let d = 0; d < 3; d++) {
      const cols = [
        a, b, c, d,
        (a + b) % 3, (a + 2 * b) % 3,
        (a + c) % 3, (a + 2 * c) % 3,
        (a + d) % 3, (a + 2 * d) % 3,
        (b + c) % 3, (b + 2 * c) % 3,
        (b + d) % 3, (b + 2 * d) % 3,
        (c + d) % 3, (c + 2 * d) % 3,
        (a + b + c) % 3, (a + b + 2 * c) % 3, (a + 2 * b + c) % 3, (a + 2 * b + 2 * c) % 3,
        (a + b + d) % 3, (a + b + 2 * d) % 3, (a + 2 * b + d) % 3, (a + 2 * b + 2 * d) % 3,
        (a + c + d) % 3, (a + c + 2 * d) % 3, (a + 2 * c + d) % 3, (a + 2 * c + 2 * d) % 3,
        (b + c + d) % 3, (b + c + 2 * d) % 3, (b + 2 * c + d) % 3, (b + 2 * c + 2 * d) % 3,
        (a + b + c + d) % 3, (a + b + c + 2 * d) % 3, (a + b + 2 * c + d) % 3, (a + b + 2 * c + 2 * d) % 3,
        (a + 2 * b + c + d) % 3, (a + 2 * b + c + 2 * d) % 3, (a + 2 * b + 2 * c + d) % 3, (a + 2 * b + 2 * c + 2 * d) % 3,
      ]
      rows.push(cols.map((v) => v + 1))
    }
    tables.L81 = rows
  }
  if (array === 'L108') {
    // Mixed OA: 1 two-level + 3-level columns from 2×3³×3 lattice (108 = 2×54)
    const rows: number[][] = []
    for (let a = 0; a < 2; a++) {
      for (let b = 0; b < 3; b++) for (let c = 0; c < 3; c++) for (let d = 0; d < 3; d++) for (let e = 0; e < 3; e++) {
        const cols = [
          a + 1,
          b + 1, c + 1, d + 1, e + 1,
          ((b + c) % 3) + 1, ((b + 2 * c) % 3) + 1,
          ((b + d) % 3) + 1, ((b + 2 * d) % 3) + 1,
          ((b + e) % 3) + 1, ((b + 2 * e) % 3) + 1,
          ((c + d) % 3) + 1, ((c + 2 * d) % 3) + 1,
          ((c + e) % 3) + 1, ((c + 2 * e) % 3) + 1,
          ((d + e) % 3) + 1, ((d + 2 * e) % 3) + 1,
          ((b + c + d) % 3) + 1, ((b + c + e) % 3) + 1, ((b + d + e) % 3) + 1, ((c + d + e) % 3) + 1,
          ((a + b) % 3) + 1, ((a + c) % 3) + 1, ((a + d) % 3) + 1, ((a + e) % 3) + 1,
          ((a + b + c + d + e) % 3) + 1,
        ]
        rows.push(cols)
      }
    }
    tables.L108 = rows.slice(0, 108)
  }
  if (array === 'L121') {
    // 11-level OA: 12 columns from 11² lattice
    const rows: number[][] = []
    for (let a = 0; a < 11; a++) for (let b = 0; b < 11; b++) {
      const cols = [a, b]
      for (let k = 1; k <= 10; k++) cols.push((a + k * b) % 11)
      rows.push(cols.map((v) => v + 1))
    }
    tables.L121 = rows
  }
  if (array === 'L128') {
    // 2-level OA: 127 columns from 2^7 factorial contrasts
    const ff = fullFactorialLevels(7)
    const cols: number[][] = []
    for (let mask = 1; mask < 128; mask++) {
      cols.push(ff.map((row) => {
        let p = 1
        for (let j = 0; j < 7; j++) if (mask & (1 << j)) p *= row[j]!
        return p === 1 ? 1 : 2
      }))
    }
    tables.L128 = ff.map((_, i) => cols.map((c) => c[i]!))
  }
  if (array === 'L243') {
    // 3-level OA: 3^5 = 243 runs; first 40 lattice contrasts (same family as L81)
    const rows: number[][] = []
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) for (let c = 0; c < 3; c++) {
      for (let d = 0; d < 3; d++) for (let e = 0; e < 3; e++) {
        const cols = [
          a, b, c, d, e,
          (a + b) % 3, (a + 2 * b) % 3,
          (a + c) % 3, (a + 2 * c) % 3,
          (a + d) % 3, (a + 2 * d) % 3,
          (a + e) % 3, (a + 2 * e) % 3,
          (b + c) % 3, (b + 2 * c) % 3,
          (b + d) % 3, (b + 2 * d) % 3,
          (b + e) % 3, (b + 2 * e) % 3,
          (c + d) % 3, (c + 2 * d) % 3,
          (c + e) % 3, (c + 2 * e) % 3,
          (d + e) % 3, (d + 2 * e) % 3,
          (a + b + c) % 3, (a + b + d) % 3, (a + b + e) % 3,
          (a + c + d) % 3, (a + c + e) % 3, (a + d + e) % 3,
          (b + c + d) % 3, (b + c + e) % 3, (b + d + e) % 3, (c + d + e) % 3,
          (a + b + c + d) % 3, (a + b + c + e) % 3, (a + b + d + e) % 3, (a + c + d + e) % 3,
          (a + b + c + d + e) % 3,
        ]
        rows.push(cols.map((v) => v + 1))
      }
    }
    tables.L243 = rows
  }
  const mat = tables[array]
  if (!mat) throw new RangeError(`taguchi: unsupported array ${array}`)
  const nCol = mat[0]!.length
  const factors = factorNames?.slice(0, nCol) ?? Array.from({ length: nCol }, (_, i) => `F${i + 1}`)
  return { type: `taguchi-${array}`, factors, matrix: mat.map((r) => r.slice(0, factors.length)), info: { array, runs: mat.length } }
}

export interface EffectEstimate {
  term: string
  effect: number
  coefficient: number
  /** Half-normal plot coordinate: |effect|. */
  absEffect: number
  /** Normal plot: Φ⁻¹((i−0.375)/(m+0.25)) paired with signed effect after ranking. */
  normalScore?: number
}

export interface EffectsAnalysis {
  effects: EffectEstimate[]
  /** Lenth's pseudo standard error. */
  pse: number
  /** Margin of error (ME) ≈ t_{0.975, d} · PSE with d ≈ m/3. */
  me: number
  /** Simultaneous margin of error (SME). */
  sme: number
  significant: string[]
}

/**
 * Effect estimates for a two-level design (coded ±1) with response `y`.
 * Lenth PSE for unreplicated factorials (Minitab / JMP default).
 */
export function analyzeEffects(design: DoeDesign, y: ArrayLike<number>, options: { interactions?: boolean } = {}): EffectsAnalysis {
  const X = design.matrix
  const n = X.length
  const k = design.factors.length
  if (y.length !== n) throw new RangeError('analyzeEffects: y length must match runs')
  const ys = Array.from(y)
  const terms: Array<{ name: string; col: number[] }> = []
  for (let j = 0; j < k; j++) terms.push({ name: design.factors[j]!, col: X.map((r) => r[j]!) })
  if (options.interactions !== false && k <= 6) {
    for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) {
      terms.push({ name: `${design.factors[i]}*${design.factors[j]}`, col: X.map((r) => r[i]! * r[j]!) })
    }
  }
  // only keep columns that are not constant (fractional / PB)
  const usable = terms.filter((t) => {
    const u = new Set(t.col.map((c) => Math.round(c * 1e10) / 1e10))
    return u.size > 1
  })
  const effects: EffectEstimate[] = usable.map((t) => {
    let pos = 0
    let nPos = 0
    let neg = 0
    let nNeg = 0
    for (let i = 0; i < n; i++) {
      if (t.col[i]! > 0) {
        pos += ys[i]!
        nPos++
      } else if (t.col[i]! < 0) {
        neg += ys[i]!
        nNeg++
      }
    }
    const effect = pos / nPos - neg / nNeg
    return { term: t.name, effect, coefficient: effect / 2, absEffect: Math.abs(effect) }
  })

  // Lenth PSE
  const abs = effects.map((e) => e.absEffect).sort((a, b) => a - b)
  const m = abs.length
  const s0 = 1.5 * abs[Math.floor(m / 2)]!
  const trimmed = abs.filter((a) => a <= 2.5 * s0)
  const pse = 1.5 * trimmed[Math.floor(trimmed.length / 2)]!
  const d = Math.max(1, Math.floor(m / 3))
  // t critical approx via normal for simplicity + small df correction
  const t975 = d > 30 ? 1.96 : STD.ppf(0.975) * (1 + 1 / (4 * d))
  const me = t975 * pse
  const sme = STD.ppf(1 - 0.025 / m) * pse * (1 + 1 / (4 * d))
  // normal scores for half-normal plot
  const ranked = effects.slice().sort((a, b) => a.absEffect - b.absEffect)
  ranked.forEach((e, i) => {
    e.normalScore = STD.ppf((i + 1 - 0.375) / (m + 0.25))
  })
  const significant = effects.filter((e) => e.absEffect > me).map((e) => e.term)
  return { effects, pse, me, sme, significant }
}

/** Fit a response-surface / factorial model via `linearModel` on a design + response. */
export function analyzeDoe(
  design: DoeDesign,
  y: ArrayLike<number>,
  options: { formula?: string } = {},
): LinearModelResult & { design: DoeDesign } {
  const data: Record<string, Array<number | string | boolean | null>> = { y: Array.from(y) }
  for (let j = 0; j < design.factors.length; j++) {
    data[design.factors[j]!] = design.matrix.map((r) => r[j]!)
  }
  const formula =
    options.formula ??
    (design.type.startsWith('ccd') || design.type === 'box-behnken'
      ? `y ~ ${design.factors.map((f) => `${f}*${f}`).join(' + ')} + ${design.factors.join('*')}`
      : `y ~ ${design.factors.join('*')}`)
  // For RSM prefer quadratic mains: f*f gives f:f; also include two-fi via *
  const lm = linearModel(data, formula, { factors: [] })
  return Object.assign(lm, { design })
}

/** Expand a design to a model matrix for custom OLS (intercept + mains + optional interactions). */
export function designMatrix(design: DoeDesign, options: { interactions?: boolean; intercept?: boolean } = {}): { names: string[]; X: number[][] } {
  const k = design.factors.length
  const names: string[] = []
  const cols: number[][] = []
  if (options.intercept !== false) {
    names.push('Intercept')
    cols.push(design.matrix.map(() => 1))
  }
  for (let j = 0; j < k; j++) {
    names.push(design.factors[j]!)
    cols.push(design.matrix.map((r) => r[j]!))
  }
  if (options.interactions) {
    for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) {
      names.push(`${design.factors[i]}*${design.factors[j]}`)
      cols.push(design.matrix.map((r) => r[i]! * r[j]!))
    }
  }
  const X = design.matrix.map((_, i) => cols.map((c) => c[i]!))
  return { names, X }
}

/** Convenience OLS on a design matrix. */
export function fitDesign(design: DoeDesign, y: ArrayLike<number>, options: { interactions?: boolean } = {}) {
  const { names, X } = designMatrix(design, { interactions: options.interactions ?? true })
  const M = matrix(X.length, names.length)
  for (let i = 0; i < X.length; i++) for (let j = 0; j < names.length; j++) M.data[i * names.length + j] = X[i]![j]!
  const fit = lstsq(M, Array.from(y))
  return { names, coefficients: Array.from(fit.coef), sse: fit.sse, rank: fit.rank }
}

export type SnRatio = 'nominal' | 'larger' | 'smaller'

export interface TaguchiFactorSummary {
  factor: string
  levels: number[]
  /** Mean response at each level. */
  meanResponse: number[]
  /** Mean S/N at each level. */
  meanSN: number[]
  /** Δ = max(meanSN) − min(meanSN). */
  deltaSN: number
  /** Level with the best (largest) S/N. */
  bestLevel: number
}

export interface TaguchiAnalysis {
  snRatio: SnRatio
  /** Per-run S/N (one value per inner row; if outer provided, S/N pools outer replicates). */
  sn: number[]
  /** Per-run mean response (across outer, or the single response). */
  mean: number[]
  factors: TaguchiFactorSummary[]
  /** Factors ranked by Δ(S/N) descending. */
  ranking: string[]
}

function snValue(reps: number[], kind: SnRatio): number {
  const n = reps.length
  if (n < 1) return NaN
  if (kind === 'larger') {
    // −10 log₁₀( (1/n) Σ 1/y² )
    let s = 0
    for (const y of reps) s += 1 / (y * y || 1e-300)
    return -10 * Math.log10(s / n)
  }
  if (kind === 'smaller') {
    // −10 log₁₀( (1/n) Σ y² )
    let s = 0
    for (const y of reps) s += y * y
    return -10 * Math.log10(s / n)
  }
  // nominal-the-best (Type I): 10 log₁₀(ȳ² / s²)
  let m = 0
  for (const y of reps) m += y
  m /= n
  let s2 = 0
  for (const y of reps) s2 += (y - m) ** 2
  s2 /= Math.max(1, n - 1)
  return 10 * Math.log10((m * m) / Math.max(s2, 1e-300))
}

/**
 * Taguchi analysis: S/N ratios and average response / S/N tables by factor level.
 * `responses` is either one value per inner run, or a matrix [inner][outer] when an outer array is used.
 */
export function analyzeTaguchi(
  design: DoeDesign,
  responses: ArrayLike<number> | ArrayLike<ArrayLike<number>>,
  options: { snRatio?: SnRatio } = {},
): TaguchiAnalysis {
  const snRatio = options.snRatio ?? 'nominal'
  const n = design.matrix.length
  const rows: number[][] = []
  if (Array.isArray(responses) && responses.length > 0 && Array.isArray((responses as ArrayLike<number>[])[0])) {
    for (let i = 0; i < n; i++) rows.push(Array.from((responses as ArrayLike<number>[])[i]!))
  } else {
    const flat = Array.from(responses as ArrayLike<number>)
    if (flat.length !== n) throw new RangeError('analyzeTaguchi: responses length must match design runs')
    for (let i = 0; i < n; i++) rows.push([flat[i]!])
  }
  if (rows.length !== n) throw new RangeError('analyzeTaguchi: responses rows must match design runs')

  const sn = rows.map((r) => snValue(r, snRatio))
  const mean = rows.map((r) => r.reduce((a, b) => a + b, 0) / r.length)

  const factors: TaguchiFactorSummary[] = []
  for (let j = 0; j < design.factors.length; j++) {
    const levelSet = [...new Set(design.matrix.map((r) => r[j]!))].sort((a, b) => a - b)
    const meanResponse: number[] = []
    const meanSN: number[] = []
    for (const lev of levelSet) {
      let sY = 0
      let sSN = 0
      let c = 0
      for (let i = 0; i < n; i++) {
        if (design.matrix[i]![j] !== lev) continue
        sY += mean[i]!
        sSN += sn[i]!
        c++
      }
      meanResponse.push(sY / c)
      meanSN.push(sSN / c)
    }
    const deltaSN = Math.max(...meanSN) - Math.min(...meanSN)
    const bestIdx = meanSN.indexOf(Math.max(...meanSN))
    factors.push({
      factor: design.factors[j]!,
      levels: levelSet,
      meanResponse,
      meanSN,
      deltaSN,
      bestLevel: levelSet[bestIdx]!,
    })
  }
  const ranking = factors
    .slice()
    .sort((a, b) => b.deltaSN - a.deltaSN)
    .map((f) => f.factor)
  return { snRatio, sn, mean, factors, ranking }
}

/**
 * Jones–Nachtsheim definitive screening design: 2k+1 foldover pairs + center
 * (optionally +2 extra centers → 2k+3). Levels ∈ {−1, 0, +1}.
 */
export function definitiveScreening(factorNames: string[], options: { extraCenters?: number } = {}): DoeDesign {
  const k = factorNames.length
  if (k < 4 || k > 12) throw new RangeError('definitiveScreening: 4…12 factors')
  const rows: number[][] = []
  // Foldover construction: for i=1..k, run with factor i at 0 and others ±1 in a cyclic pattern
  for (let i = 0; i < k; i++) {
    const pos = new Array(k).fill(1)
    const neg = new Array(k).fill(-1)
    pos[i] = 0
    neg[i] = 0
    // alternate signs by distance from i
    for (let j = 0; j < k; j++) {
      if (j === i) continue
      const s = ((j - i + k) % k) % 2 === 0 ? 1 : -1
      pos[j] = s
      neg[j] = -s
    }
    rows.push(pos)
    rows.push(neg)
  }
  const nCenters = 1 + (options.extraCenters ?? 0)
  for (let c = 0; c < nCenters; c++) rows.push(new Array(k).fill(0))
  return {
    type: 'definitive-screening',
    factors: factorNames.slice(),
    matrix: rows,
    info: { type: 'dsd', runs: rows.length, k },
  }
}

/** Simplex-lattice / simplex-centroid mixture design (compositions sum to 1). Optional process factors crossed in. */
export function mixtureDesign(
  factorNames: string[],
  options: {
    type?: 'lattice' | 'centroid'
    degree?: 2 | 3
    /** Process variable names — crossed with mixture points at ±1 (or −1,0,1 if processLevels=3). */
    process?: string[]
    processLevels?: 2 | 3
  } = {},
): DoeDesign {
  const q = factorNames.length
  if (q < 2 || q > 8) throw new RangeError('mixtureDesign: 2…8 components')
  const type = options.type ?? 'lattice'
  const degree = options.degree ?? 2
  const process = options.process ?? []
  const processLevels = options.processLevels ?? 2
  const rows: number[][] = []
  const pushUnique = (row: number[]) => {
    const key = row.map((v) => v.toFixed(6)).join(',')
    if (!rows.some((r) => r.map((v) => v.toFixed(6)).join(',') === key)) rows.push(row)
  }

  if (type === 'centroid' || degree >= 2) {
    // vertices
    for (let i = 0; i < q; i++) {
      const r = new Array(q).fill(0)
      r[i] = 1
      pushUnique(r)
    }
  }
  if (type === 'lattice') {
    const m = degree
    // all nonneg compositions with denominators m
    const rec = (left: number, start: number, cur: number[]) => {
      if (cur.length === q - 1) {
        const last = left
        pushUnique([...cur, last / m])
        return
      }
      for (let v = 0; v <= left; v++) {
        cur.push(v / m)
        rec(left - v, start + 1, cur)
        cur.pop()
      }
    }
    rec(m, 0, [])
  } else {
    // simplex centroid: all subset centroids
    const n = 1 << q
    for (let mask = 1; mask < n; mask++) {
      const idx: number[] = []
      for (let j = 0; j < q; j++) if (mask & (1 << j)) idx.push(j)
      const r = new Array(q).fill(0)
      for (const j of idx) r[j] = 1 / idx.length
      pushUnique(r)
    }
  }

  let matrixRows = rows
  const factors = [...factorNames, ...process]
  if (process.length > 0) {
    const levels = processLevels === 3 ? [-1, 0, 1] : [-1, 1]
    const procRows: number[][] = [[]]
    for (let j = 0; j < process.length; j++) {
      const next: number[][] = []
      for (const prev of procRows) for (const lv of levels) next.push([...prev, lv])
      procRows.length = 0
      procRows.push(...next)
    }
    matrixRows = []
    for (const mix of rows) for (const pr of procRows) matrixRows.push([...mix, ...pr])
  }

  return {
    type: process.length ? `mixture-${type}-process` : `mixture-${type}`,
    factors,
    matrix: matrixRows,
    info: {
      type,
      degree,
      runs: matrixRows.length,
      mixtureComponents: q,
      process,
      processLevels: process.length ? processLevels : undefined,
    },
  }
}

export interface MixtureAnalysis {
  model: 'linear' | 'quadratic' | 'special-cubic'
  names: string[]
  coefficients: number[]
  r2: number
  sse: number
  nProcess?: number
  predict(x: ArrayLike<number>): number
}

function scheffeColumns(
  x: number[],
  model: 'linear' | 'quadratic' | 'special-cubic',
  nProcess = 0,
  processModel: 'linear' | 'quadratic' = 'linear',
): { names: string[]; cols: number[] } {
  const q = x.length - nProcess
  const mix = x.slice(0, q)
  const proc = x.slice(q)
  const names: string[] = []
  const cols: number[] = []
  for (let i = 0; i < q; i++) {
    names.push(`X${i + 1}`)
    cols.push(mix[i]!)
  }
  if (model === 'quadratic' || model === 'special-cubic') {
    for (let i = 0; i < q; i++) for (let j = i + 1; j < q; j++) {
      names.push(`X${i + 1}*X${j + 1}`)
      cols.push(mix[i]! * mix[j]!)
    }
  }
  if (model === 'special-cubic' && q >= 3) {
    for (let i = 0; i < q; i++) for (let j = i + 1; j < q; j++) for (let k = j + 1; k < q; k++) {
      names.push(`X${i + 1}*X${j + 1}*X${k + 1}`)
      cols.push(mix[i]! * mix[j]! * mix[k]!)
    }
  }
  // process main effects + mixture×process
  for (let k = 0; k < nProcess; k++) {
    names.push(`Z${k + 1}`)
    cols.push(proc[k]!)
    for (let i = 0; i < q; i++) {
      names.push(`X${i + 1}*Z${k + 1}`)
      cols.push(mix[i]! * proc[k]!)
    }
    if (processModel === 'quadratic') {
      names.push(`Z${k + 1}^2`)
      cols.push(proc[k]! * proc[k]!)
    }
  }
  return { names, cols }
}

/** Scheffé polynomial fit on mixture design (no intercept). Supports mixture × process terms. */
export function analyzeMixture(
  design: DoeDesign,
  y: ArrayLike<number>,
  options: {
    model?: 'linear' | 'quadratic' | 'special-cubic'
    processModel?: 'linear' | 'quadratic'
  } = {},
): MixtureAnalysis {
  const model = options.model ?? 'quadratic'
  const processModel = options.processModel ?? 'linear'
  const ys = Array.from(y)
  if (ys.length !== design.matrix.length) throw new RangeError('analyzeMixture: y length mismatch')
  const nProcess = Array.isArray(design.info?.process) ? (design.info!.process as string[]).length : 0
  const q = design.factors.length - nProcess
  const sample = scheffeColumns(design.matrix[0]!, model, nProcess, processModel)
  const p = sample.cols.length
  const M = matrix(ys.length, p)
  for (let i = 0; i < ys.length; i++) {
    const { cols } = scheffeColumns(design.matrix[i]!, model, nProcess, processModel)
    for (let j = 0; j < p; j++) M.data[i * p + j] = cols[j]!
  }
  const fit = lstsq(M, ys)
  const coef = Array.from(fit.coef)
  let sst = 0
  const my = ys.reduce((a, b) => a + b, 0) / ys.length
  for (const yi of ys) sst += (yi - my) ** 2
  return {
    model,
    names: sample.names,
    coefficients: coef,
    r2: sst > 0 ? 1 - fit.sse / sst : NaN,
    sse: fit.sse,
    nProcess: nProcess || undefined,
    predict: (x) => {
      const xx = Array.from(x)
      if (xx.length !== q + nProcess) throw new RangeError('predict: wrong dimension')
      const { cols } = scheffeColumns(xx, model, nProcess, processModel)
      let s = 0
      for (let j = 0; j < coef.length; j++) s += coef[j]! * cols[j]!
      return s
    },
  }
}

export interface OptimizerGoal {
  predict: (x: number[]) => number
  goal: 'maximize' | 'minimize' | 'target'
  target?: number
  lower: number
  upper: number
  weight?: number
}

export interface ResponseOptimizerResult {
  x: number[]
  D: number
  desirabilities: number[]
  yhat: number[]
}

function desirability(y: number, g: OptimizerGoal): number {
  const { lower: L, upper: U, goal, target } = g
  if (!(U > L)) return 0
  if (goal === 'maximize') {
    if (y <= L) return 0
    if (y >= U) return 1
    return (y - L) / (U - L)
  }
  if (goal === 'minimize') {
    if (y >= U) return 0
    if (y <= L) return 1
    return (U - y) / (U - L)
  }
  const T = target ?? (L + U) / 2
  if (y < L || y > U) return 0
  if (y <= T) return (y - L) / (T - L || 1e-12)
  return (U - y) / (U - T || 1e-12)
}

/**
 * Derringer–Suich response optimizer.
 * For `simplex: true`, x lives on the composition simplex (mixture).
 */
export function responseOptimizer(
  goals: OptimizerGoal[],
  options: {
    bounds?: Array<[number, number]>
    simplex?: boolean
    nComponents?: number
    grid?: number
  } = {},
): ResponseOptimizerResult {
  if (goals.length < 1) throw new RangeError('responseOptimizer: need ≥1 goal')
  const grid = options.grid ?? 11
  const simplex = options.simplex ?? false
  const dim = simplex ? (options.nComponents ?? goals.length) : (options.bounds?.length ?? 2)

  const evalD = (x: number[]) => {
    const yhat = goals.map((g) => g.predict(x))
    const ds = goals.map((g, i) => desirability(yhat[i]!, g))
    const ws = goals.map((g) => g.weight ?? 1)
    const wSum = ws.reduce((a, b) => a + b, 0)
    let logD = 0
    for (let i = 0; i < ds.length; i++) {
      if (ds[i]! <= 0) return { D: 0, ds, yhat }
      logD += (ws[i]! / wSum) * Math.log(ds[i]!)
    }
    return { D: Math.exp(logD), ds, yhat }
  }

  let bestX = new Array(dim).fill(simplex ? 1 / dim : 0)
  let best = evalD(bestX)

  if (simplex) {
    // grid over simplex lattice
    const rec = (left: number, cur: number[]) => {
      if (cur.length === dim - 1) {
        const x = [...cur, left / (grid - 1)]
        const e = evalD(x)
        if (e.D > best.D) {
          best = e
          bestX = x
        }
        return
      }
      for (let v = 0; v <= left; v++) {
        cur.push(v / (grid - 1))
        rec(left - v, cur)
        cur.pop()
      }
    }
    rec(grid - 1, [])
  } else {
    const bounds = options.bounds ?? Array.from({ length: dim }, () => [-1, 1] as [number, number])
    const rec = (j: number, cur: number[]) => {
      if (j === dim) {
        const e = evalD(cur)
        if (e.D > best.D) {
          best = e
          bestX = cur.slice()
        }
        return
      }
      const [lo, hi] = bounds[j]!
      for (let t = 0; t < grid; t++) {
        cur[j] = lo + (t / Math.max(1, grid - 1)) * (hi - lo)
        rec(j + 1, cur)
      }
    }
    rec(0, new Array(dim).fill(0))
  }

  return { x: bestX, D: best.D, desirabilities: best.ds, yhat: best.yhat }
}

export interface NestedAnovaComponent {
  source: string
  df: number
  ss: number
  ms: number
  f: number
  pValue: number
  /** Estimated variance component. */
  varComponent: number
}

export interface NestedAnovaResult {
  components: NestedAnovaComponent[]
  residual: { df: number; ss: number; ms: number }
  n: number
  /** Total of variance component estimates. */
  totalVar: number
}

/**
 * Fully nested ANOVA (sequential EMS / F) for factors nested A/B/C/…
 * `factors[0]` is outermost; last is innermost before residual.
 */
export function nestedAnova(
  y: ArrayLike<number>,
  factors: Array<ArrayLike<string | number>>,
): NestedAnovaResult {
  const yy = Array.from(y)
  const n = yy.length
  if (factors.length < 1) throw new RangeError('nestedAnova: need ≥1 factor')
  const cols = factors.map((f) => Array.from(f, String))
  for (const f of cols) {
    if (f.length !== n) throw new RangeError('nestedAnova: factor length mismatch')
  }
  // Build nested keys at each level: f0, f0::f1, ... (each level extends the previous key)
  const levelKeys: string[][] = []
  for (let L = 0; L < factors.length; L++) {
    const prev = L ? levelKeys[L - 1]! : null
    const keys: string[] = new Array(n)
    for (let i = 0; i < n; i++) keys[i] = prev ? `${prev[i]}::${cols[L]![i]}` : cols[L]![i]!
    levelKeys.push(keys)
  }

  const unique = (keys: string[]) => [...new Set(keys)]
  const groupMeans = (keys: string[]) => {
    const u = unique(keys)
    const sum = new Map(u.map((k) => [k, 0]))
    const cnt = new Map(u.map((k) => [k, 0]))
    for (let i = 0; i < n; i++) {
      sum.set(keys[i]!, sum.get(keys[i]!)! + yy[i]!)
      cnt.set(keys[i]!, cnt.get(keys[i]!)! + 1)
    }
    const mean = new Map(u.map((k) => [k, sum.get(k)! / cnt.get(k)!]))
    return { u, mean, cnt }
  }

  const grand = yy.reduce((a, b) => a + b, 0) / n
  const components: NestedAnovaComponent[] = []
  let ssPrev = 0 // SS explained by coarser levels (for residual within)

  // For each level L, SS_L = sum n_g (ȳ_g - ȳ_parent)^2
  for (let L = 0; L < factors.length; L++) {
    const { u, mean, cnt } = groupMeans(levelKeys[L]!)
    let ss = 0
    if (L === 0) {
      for (const k of u) {
        const d = mean.get(k)! - grand
        ss += cnt.get(k)! * d * d
      }
    } else {
      const parent = groupMeans(levelKeys[L - 1]!)
      for (const k of u) {
        const parentKey = k.split('::').slice(0, L).join('::')
        const d = mean.get(k)! - parent.mean.get(parentKey)!
        ss += cnt.get(k)! * d * d
      }
    }
    const df = L === 0 ? u.length - 1 : u.length - unique(levelKeys[L - 1]!).length
    const ms = ss / Math.max(1, df)
    components.push({
      source: `Factor${L + 1}`,
      df: Math.max(0, df),
      ss,
      ms,
      f: NaN,
      pValue: NaN,
      varComponent: 0,
    })
    ssPrev += ss
  }

  // Residual
  const innermost = groupMeans(levelKeys[factors.length - 1]!)
  let ssRes = 0
  for (let i = 0; i < n; i++) {
    const d = yy[i]! - innermost.mean.get(levelKeys[factors.length - 1]![i]!)!
    ssRes += d * d
  }
  const dfRes = n - innermost.u.length
  const msRes = ssRes / Math.max(1, dfRes)

  // F tests: each level vs next (innermost vs residual)
  for (let L = factors.length - 1; L >= 0; L--) {
    const nextMs = L === factors.length - 1 ? msRes : components[L + 1]!.ms
    const nextDf = L === factors.length - 1 ? dfRes : components[L + 1]!.df
    const f = nextMs > 0 ? components[L]!.ms / nextMs : NaN
    components[L]!.f = f
    components[L]!.pValue =
      Number.isFinite(f) && components[L]!.df > 0 && nextDf > 0
        ? fDist(components[L]!.df, nextDf).sf(f)
        : NaN
    // MoM variance component: (MS_L - MS_next) / n̄
    const nBar = n / unique(levelKeys[L]!).length
    components[L]!.varComponent = Math.max(0, (components[L]!.ms - nextMs) / Math.max(1e-8, nBar))
  }

  const totalVar = components.reduce((s, c) => s + c.varComponent, 0) + msRes
  void ssPrev
  return {
    components,
    residual: { df: Math.max(0, dfRes), ss: ssRes, ms: msRes },
    n,
    totalVar,
  }
}

