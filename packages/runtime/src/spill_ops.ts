import {
  allocateData,
  estimateTableBytes,
  getColumn,
  getValue,
  isValid,
  setValid,
  setValue,
  sliceTable,
  tableFromColumns,
  takeColumn,
  type Column,
  type TableView,
} from '@columna/arrow'
import type { ExprNode } from './types.js'
import { memoryBudget, recordLiveBytes, spillEnabled } from './memory.js'
import { concatTables, spillRead, spillTempPath, spillUnlink, spillUnlinkMany, spillWrite } from './spill.js'

type SortKey = { expr: ExprNode; descending: boolean; nullsLast?: boolean }

type EvalRow = (expr: ExprNode, table: TableView, row: number) => unknown

/**
 * Decide whether an operator should take the spill path for this table.
 * Uses ~2× live estimate as a stand-in for input + index/output working set.
 */
export function needsSpill(table: TableView, factor = 2): boolean {
  const budget = memoryBudget()
  if (!budget || !spillEnabled()) return false
  const bytes = estimateTableBytes(table)
  recordLiveBytes(bytes)
  return bytes * factor > budget
}

export function chunkRowCount(table: TableView, budget: number): number {
  const bytes = estimateTableBytes(table)
  const perRow = Math.max(1, Math.ceil(bytes / Math.max(1, table.numRows)))
  // Leave headroom for sorted copy + merge buffers.
  return Math.max(1, Math.floor(budget / (perRow * 4)))
}

/**
 * External sort: sorted runs spilled to disk, then k-way merge.
 * `sortChunk` must sort a small in-memory table without re-entering spill.
 */
export function externalSortTable(
  table: TableView,
  by: SortKey[],
  limit: number | undefined,
  sortChunk: (chunk: TableView, by: SortKey[], limit?: number) => TableView,
  evalRow: EvalRow,
): TableView {
  const budget = memoryBudget()!
  const chunkRows = chunkRowCount(table, budget)
  const runs: string[] = []
  try {
    for (let start = 0; start < table.numRows; start += chunkRows) {
      const end = Math.min(table.numRows, start + chunkRows)
      const chunk = sliceTable(table, start, end)
      const sorted = sortChunk(chunk, by)
      runs.push(spillWrite(sorted, spillTempPath('sort')))
    }
    if (runs.length === 0) return tableFromColumns(table.columns.map((c) => takeColumn(c, [])))
    if (runs.length === 1) {
      const only = spillRead(runs[0]!)
      spillUnlink(runs[0]!)
      runs.length = 0
      return limit !== undefined ? sliceTable(only, 0, Math.min(limit, only.numRows)) : only
    }
    return mergeSortedRuns(runs, by, limit, evalRow)
  } finally {
    spillUnlinkMany(runs)
  }
}

function compareHead(
  tables: TableView[],
  cursors: number[],
  by: SortKey[],
  evalRow: EvalRow,
  a: number,
  b: number,
): number {
  const ta = tables[a]!
  const tb = tables[b]!
  const ia = cursors[a]!
  const ib = cursors[b]!
  for (const key of by) {
    const va = evalRow(key.expr, ta, ia)
    const vb = evalRow(key.expr, tb, ib)
    if (va === vb) continue
    const nullsLast = key.nullsLast !== false
    if (va === null || va === undefined) return nullsLast ? 1 : -1
    if (vb === null || vb === undefined) return nullsLast ? -1 : 1
    const cmp = (va as number | string | boolean) < (vb as number | string | boolean) ? -1 : 1
    return key.descending ? -cmp : cmp
  }
  // Stable: prefer lower run index on ties.
  return a - b
}

function mergeSortedRuns(
  runPaths: string[],
  by: SortKey[],
  limit: number | undefined,
  evalRow: EvalRow,
): TableView {
  const tables = runPaths.map((p) => spillRead(p))
  const cursors = tables.map(() => 0)
  const alive = tables.map((t) => t.numRows > 0)
  const maxOut = limit ?? Number.POSITIVE_INFINITY

  // Per-run gather lists in merge order — one take per run at the end, then interleave via a thin index table.
  const picks: Array<{ run: number; row: number }> = []
  while (picks.length < maxOut) {
    let best = -1
    for (let r = 0; r < tables.length; r++) {
      if (!alive[r]) continue
      if (best < 0 || compareHead(tables, cursors, by, evalRow, r, best) < 0) best = r
    }
    if (best < 0) break
    picks.push({ run: best, row: cursors[best]! })
    cursors[best]! += 1
    if (cursors[best]! >= tables[best]!.numRows) alive[best] = false
  }

  if (picks.length === 0) {
    return tableFromColumns(tables[0]!.columns.map((c) => takeColumn(c, [])))
  }

  // Materialize by copying each picked row in order (correctness over micro-opts).
  const schema = tables[0]!.schema
  const n = picks.length
  const outCols: Column[] = schema.map((field, ci) => {
    const sample = tables[0]!.columns[ci]!
    const data = allocateData(field.dtype, n)
    const nullBitmap = new Uint8Array(Math.ceil(n / 8) || 1)
    let anyNull = false
    for (let i = 0; i < n; i++) {
      const { run, row } = picks[i]!
      const src = tables[run]!.columns[ci]!
      if (!isValid(src.nullBitmap, row)) {
        anyNull = true
      } else {
        setValid(nullBitmap, i, true)
        setValue(data, i, getValue(src.data, row) as never, field.dtype)
      }
    }
    return {
      field: { ...field },
      data,
      nullBitmap: anyNull ? nullBitmap : undefined,
      dictionary: sample.dictionary ? [...sample.dictionary] : undefined,
    }
  })
  return tableFromColumns(outCols)
}

/**
 * Chunked unique: unique each chunk, spill partials, then unique the concat.
 */
export function uniqueTableSpilled(
  table: TableView,
  uniqueChunk: (chunk: TableView) => TableView,
): TableView {
  const budget = memoryBudget()!
  const chunkRows = chunkRowCount(table, budget)
  const paths: string[] = []
  try {
    for (let start = 0; start < table.numRows; start += chunkRows) {
      const end = Math.min(table.numRows, start + chunkRows)
      const partial = uniqueChunk(sliceTable(table, start, end))
      paths.push(spillWrite(partial, spillTempPath('uniq')))
    }
    const partials = paths.map((p) => spillRead(p))
    const merged = concatTables(partials)
    return uniqueChunk(merged)
  } finally {
    spillUnlinkMany(paths)
  }
}

/**
 * Spill the right side (freeing pressure / recording spilledBytes), reload, then join once in memory.
 * Chunked inner-join concat is easy to get wrong with duplicate emit; keep a single join after spill I/O.
 */
export function joinTablesSpilled(
  left: TableView,
  right: TableView,
  joinFull: (left: TableView, right: TableView) => TableView,
): TableView {
  recordLiveBytes(estimateTableBytes(left) + estimateTableBytes(right))
  const rightPath = spillWrite(right, spillTempPath('join-r'))
  try {
    const reloaded = spillRead(rightPath)
    return joinFull(left, reloaded)
  } finally {
    spillUnlink(rightPath)
  }
}

/** Row key helper shared with unique/join spill paths. */
export function rowKey(table: TableView, columns: string[], row: number): string {
  return columns
    .map((n) => {
      const c = getColumn(table, n)
      if (!isValid(c.nullBitmap, row)) return '∅'
      return String(getValue(c.data, row))
    })
    .join('\0')
}
