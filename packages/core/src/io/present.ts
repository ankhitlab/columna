import type { Column, TableView } from '@columna/arrow'
import { isValid } from '@columna/arrow'

function cellAt(table: TableView, colIdx: number, row: number): string {
  const col = table.columns[colIdx]!
  if (!isValid(col.nullBitmap, row)) return ''
  const dtype = col.field.dtype
  if (dtype === 'utf8') return (col.data as string[])[row] ?? ''
  if (dtype === 'bool') return (col.data as Uint8Array)[row] ? 'true' : 'false'
  if (dtype === 'category' && col.dictionary) {
    return col.dictionary[(col.data as Uint32Array)[row]!] ?? ''
  }
  return String((col.data as Float64Array | Int32Array | Uint32Array)[row]!)
}

const mdCell = (s: string) => s.replace(/\|/g, '\\|').replace(/\r?\n/g, ' ')

export function tableToMarkdown(table: TableView, maxRows = 50): string {
  const names = table.schema.map((f) => f.name)
  const n = Math.min(table.numRows, maxRows)
  const header = `| ${names.map(mdCell).join(' | ')} |`
  const sep = `| ${names.map(() => '---').join(' | ')} |`
  const rows: string[] = [header, sep]
  for (let i = 0; i < n; i++) {
    rows.push(`| ${names.map((_, ci) => mdCell(cellAt(table, ci, i))).join(' | ')} |`)
  }
  if (table.numRows > maxRows) rows.push(`| … (${table.numRows - maxRows} more rows) |`)
  return rows.join('\n')
}

export function tableToHtml(table: TableView, maxRows = 100): string {
  const names = table.schema.map((f) => f.name)
  const n = Math.min(table.numRows, maxRows)
  const thead = `<thead><tr>${names.map((nm) => `<th>${escapeHtml(nm)}</th>`).join('')}</tr></thead>`
  const bodyRows: string[] = []
  for (let i = 0; i < n; i++) {
    bodyRows.push(
      `<tr>${names.map((_, ci) => `<td>${escapeHtml(cellAt(table, ci, i))}</td>`).join('')}</tr>`,
    )
  }
  return `<table>${thead}<tbody>${bodyRows.join('')}</tbody></table>`
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

export type ProfileReport = {
  rows: number
  columns: Array<{
    name: string
    dtype: string
    nullCount: number
    approxUnique: number
    min?: number | string
    max?: number | string
  }>
}

const f64buf = new ArrayBuffer(8)
const f64view = new Float64Array(f64buf)
const u32view = new Uint32Array(f64buf)

/** Open-addressing set for numbers — avoids boxing every value in Set<number>. */
function countUniqueNums(data: ArrayLike<number>, bm: Uint8Array | undefined, n: number, validHint: number): number {
  let cap = 16
  while (cap < validHint * 2 + 16) cap <<= 1
  let keys = new Float64Array(cap)
  let used = new Uint8Array(cap)
  let mask = cap - 1
  let size = 0

  const hash = (v: number): number => {
    f64view[0] = v
    let h = (u32view[0]! ^ u32view[1]!) | 0
    h = Math.imul(h ^ (h >>> 16), 0x7feb352d)
    h = Math.imul(h ^ (h >>> 15), 0x846ca68b)
    return (h ^ (h >>> 16)) >>> 0
  }

  const insert = (v: number): void => {
    if (size * 2 >= cap) {
      const oldKeys = keys
      const oldUsed = used
      const oldCap = oldKeys.length
      cap <<= 1
      keys = new Float64Array(cap)
      used = new Uint8Array(cap)
      mask = cap - 1
      size = 0
      for (let i = 0; i < oldCap; i++) {
        if (!oldUsed[i]) continue
        insert(oldKeys[i]!)
      }
    }
    let i = hash(v) & mask
    for (;;) {
      if (!used[i]) {
        used[i] = 1
        keys[i] = v
        size++
        return
      }
      if (keys[i] === v) return
      i = (i + 1) & mask
    }
  }

  for (let i = 0; i < n; i++) {
    if (bm && !isValid(bm, i)) continue
    insert(data[i]!)
  }
  return size
}

function profileNumeric(col: Column, n: number): {
  nullCount: number
  approxUnique: number
  min?: number
  max?: number
} {
  const bm = col.nullBitmap
  const data = col.data as Float64Array | Float32Array | Int32Array | Uint32Array
  let nullCount = 0
  let min = Infinity
  let max = -Infinity
  let valid = 0
  let allInt = true

  for (let i = 0; i < n; i++) {
    if (bm && !isValid(bm, i)) {
      nullCount++
      continue
    }
    const v = data[i]!
    valid++
    if (v < min) min = v
    if (v > max) max = v
    if (allInt && !Number.isInteger(v)) allInt = false
  }
  if (valid === 0) return { nullCount, approxUnique: 0 }

  if (allInt && Number.isFinite(min) && Number.isFinite(max)) {
    const span = max - min + 1
    if (span > 0 && span <= Math.max(valid * 4, 1_048_576) && span <= 16_000_000) {
      const seen = new Uint8Array(span)
      let approxUnique = 0
      for (let i = 0; i < n; i++) {
        if (bm && !isValid(bm, i)) continue
        const off = data[i]! - min
        if (!seen[off]) {
          seen[off] = 1
          approxUnique++
        }
      }
      return { nullCount, approxUnique, min, max }
    }
  }

  // Near-unique columns (timestamps, ids with huge span): sample then extrapolate.
  const sampleN = Math.min(n, 16_384)
  let sampleValid = 0
  const probe = countUniqueNums(data, bm, sampleN, Math.min(valid, sampleN))
  for (let i = 0; i < sampleN; i++) {
    if (!(bm && !isValid(bm, i))) sampleValid++
  }
  if (sampleValid > 0 && probe / sampleValid >= 0.99) {
    return { nullCount, approxUnique: valid, min, max }
  }
  if (sampleN >= n) return { nullCount, approxUnique: probe, min, max }
  return { nullCount, approxUnique: countUniqueNums(data, bm, n, valid), min, max }
}

function profileUtf8(col: Column, n: number): {
  nullCount: number
  approxUnique: number
  min?: string
  max?: string
} {
  const bm = col.nullBitmap
  const data = col.data as string[]
  const uniq = new Map<string, 1>()
  let nullCount = 0
  let min: string | undefined
  let max: string | undefined
  for (let i = 0; i < n; i++) {
    if (bm && !isValid(bm, i)) {
      nullCount++
      continue
    }
    const v = data[i]!
    if (!uniq.has(v)) uniq.set(v, 1)
    if (min === undefined || v < min) min = v
    if (max === undefined || v > max) max = v
  }
  return { nullCount, approxUnique: uniq.size, min, max }
}

function profileCategory(col: Column, n: number): {
  nullCount: number
  approxUnique: number
  min?: string
  max?: string
} {
  const bm = col.nullBitmap
  const codes = col.data as Uint32Array
  const dict = col.dictionary ?? []
  let nullCount = 0
  const used = new Uint8Array(Math.max(dict.length, 1))
  let approxUnique = 0
  for (let i = 0; i < n; i++) {
    if (bm && !isValid(bm, i)) {
      nullCount++
      continue
    }
    const code = codes[i]!
    if (code >= dict.length || used[code]) continue
    used[code] = 1
    approxUnique++
  }
  let min: string | undefined
  let max: string | undefined
  for (let c = 0; c < dict.length; c++) {
    if (!used[c]) continue
    const s = dict[c]!
    if (min === undefined || s < min) min = s
    if (max === undefined || s > max) max = s
  }
  return { nullCount, approxUnique, min, max }
}

function profileBool(col: Column, n: number): {
  nullCount: number
  approxUnique: number
  min?: number
  max?: number
} {
  const bm = col.nullBitmap
  const data = col.data as Uint8Array
  let nullCount = 0
  let has0 = false
  let has1 = false
  for (let i = 0; i < n; i++) {
    if (bm && !isValid(bm, i)) {
      nullCount++
      continue
    }
    if (data[i]) has1 = true
    else has0 = true
  }
  const approxUnique = (has0 ? 1 : 0) + (has1 ? 1 : 0)
  return {
    nullCount,
    approxUnique,
    min: has0 ? 0 : has1 ? 1 : undefined,
    max: has1 ? 1 : has0 ? 0 : undefined,
  }
}

export function profileTable(table: TableView): ProfileReport {
  const n = table.numRows
  const columns = table.columns.map((col) => {
    const dtype = col.field.dtype
    let stats: {
      nullCount: number
      approxUnique: number
      min?: number | string
      max?: number | string
    }
    if (dtype === 'utf8') stats = profileUtf8(col, n)
    else if (dtype === 'bool') stats = profileBool(col, n)
    else if (dtype === 'category') stats = profileCategory(col, n)
    else stats = profileNumeric(col, n)

    return {
      name: col.field.name,
      dtype,
      nullCount: stats.nullCount,
      approxUnique: stats.approxUnique,
      min: stats.min,
      max: stats.max,
    }
  })
  return { rows: n, columns }
}
