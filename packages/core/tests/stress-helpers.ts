/**
 * Shared fixtures for stress / load correctness tests.
 * Scale: default CI sizes; set STRESS_HEAVY=1 for ~10× rows (capped).
 */
import { DataFrame } from '../src/dataframe.js'

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** CI base size; STRESS_HEAVY multiplies by 10 (capped at 500k). */
export function stressN(base: number): number {
  if (!process.env.STRESS_HEAVY) return base
  return Math.min(Math.max(1, Math.floor(base * 10)), 500_000)
}

export const STRESS_TIMEOUT = process.env.STRESS_HEAVY ? 600_000 : 120_000

export type StressRow = {
  id: number
  k: number
  city: string
  seg: string
  salary: number | null
  age: number | null
  x: number
  y: number
  flag: boolean
  note: string | null
}

const CITIES = ['Berlin', 'Paris', 'Rome', 'Madrid', '', 'Wien', 'ünï', 'x"y']
const SEGS = ['A', 'B', 'C', 'B2B', 'B2C']

/** Adversarial frame: nulls, ties, empty/unicode/quoted strings, mixed null keys. */
export function adversarialRows(seed: number, n: number): StressRow[] {
  const rnd = mulberry32(seed)
  return Array.from({ length: n }, (_, id) => {
    const nullSalary = rnd() < 0.12
    const nullAge = rnd() < 0.08
    const nullNote = rnd() < 0.2
    return {
      id,
      k: Math.floor(rnd() * 11), // heavy ties
      city: CITIES[Math.floor(rnd() * CITIES.length)]!,
      seg: SEGS[Math.floor(rnd() * SEGS.length)]!,
      salary: nullSalary ? null : Math.round((20_000 + rnd() * 120_000) * 100) / 100,
      age: nullAge ? null : 18 + Math.floor(rnd() * 60),
      x: Math.round(rnd() * 1e6) / 1e4,
      y: Math.round((rnd() - 0.5) * 1e5) / 1e3,
      flag: rnd() < 0.5,
      note: nullNote ? null : rnd() < 0.15 ? `dup` : `n${Math.floor(rnd() * 8000)}`,
    }
  })
}

export function adversarialFrame(seed: number, n: number): DataFrame {
  return DataFrame.fromRows(adversarialRows(seed, n))
}

/** Unquoted-safe subset (no `"` / `,` / newlines in text) for native CSV write path. */
export function unquotedSafeRows(seed: number, n: number): Array<Record<string, unknown>> {
  const rnd = mulberry32(seed)
  const cities = ['Berlin', 'Paris', 'Rome', 'Madrid', 'Wien']
  return Array.from({ length: n }, (_, id) => ({
    id,
    age: 18 + Math.floor(rnd() * 60),
    salary: Math.round((20_000 + rnd() * 120_000) * 100) / 100,
    x: Math.round(rnd() * 1e4) / 100,
    y: Math.round(rnd() * 1e4) / 100,
    city: cities[id % cities.length]!,
    seg: id % 7 ? 'B2B' : 'B2C',
  }))
}

export function sum(xs: Array<number | null | undefined>): number {
  let s = 0
  for (const v of xs) if (v != null && Number.isFinite(v)) s += v
  return s
}

export function sortById(rows: Array<Record<string, unknown>>): Array<Record<string, unknown>> {
  return [...rows].sort((a, b) => Number(a.id) - Number(b.id))
}

/** Stable multiset fingerprint of rows (order-insensitive). */
export function rowMultisetKey(rows: Array<Record<string, unknown>>, cols: string[]): string {
  const keys = rows.map((r) => cols.map((c) => JSON.stringify(r[c] ?? null)).join('\t'))
  keys.sort()
  return keys.join('\n')
}

export function checksumNumeric(rows: Array<Record<string, unknown>>, col: string): number {
  return sum(rows.map((r) => (r[col] == null ? null : Number(r[col]))))
}
