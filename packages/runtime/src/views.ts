import {
  getColumn,
  isValid,
  tableFromColumns,
  takeColumn,
  type Column,
  type TableView,
} from '@columna/arrow'

const VIEW = Symbol.for('columna.filterView')

type FilterViewState = {
  base: TableView
  /** Selected row indices into `base` column buffers. */
  indices: Uint32Array
}

function viewState(table: TableView): FilterViewState | undefined {
  return (table as TableView & { [VIEW]?: FilterViewState })[VIEW]
}

/**
 * Build a filter view: shared column buffer references + a selection index.
 * `numRows` reflects the selection; column `.data.length` still matches the base
 * until `materializeView` / gather.
 */
export function createFilterView(base: TableView, indices: Uint32Array, keep?: readonly string[]): TableView {
  const cols =
    keep && keep.length
      ? keep.map((n) => getColumn(base, n))
      : [...base.columns]
  const table: TableView & { [VIEW]?: FilterViewState } = {
    schema: cols.map((c) => c.field),
    numRows: indices.length,
    columns: cols,
  }
  table[VIEW] = { base: { schema: base.schema, numRows: base.numRows, columns: base.columns }, indices }
  return table
}

export function isFilterView(table: TableView): boolean {
  return viewState(table) != null
}

/** Gather shared buffers into a dense table (required before sort/join/unique/mutate). */
export function materializeView(table: TableView): TableView {
  const st = viewState(table)
  if (!st) return table
  const cols = table.columns.map((c) => {
    // Prefer the base column with the same name so we gather from full buffers.
    const baseCol = st.base.columns.find((b) => b.field.name === c.field.name) ?? c
    return takeColumn(baseCol, st.indices)
  })
  return tableFromColumns(cols)
}

/** Ensure a table is dense (no deferred filter view). */
export function ensureMaterialized(table: TableView): TableView {
  return isFilterView(table) ? materializeView(table) : table
}

/**
 * Selection bitmap (1 = keep) over `numRows`. Used by consumers that prefer bit tests
 * over an index list.
 */
export function selectionBitmap(table: TableView): Uint8Array | null {
  const st = viewState(table)
  if (!st) return null
  const bits = new Uint8Array(Math.ceil(st.base.numRows / 8) || 1)
  for (let i = 0; i < st.indices.length; i++) {
    const row = st.indices[i]!
    bits[row >> 3]! |= 1 << (row & 7)
  }
  return bits
}

export function viewRowValid(table: TableView, logicalRow: number): boolean {
  const st = viewState(table)
  if (!st) return logicalRow >= 0 && logicalRow < table.numRows
  return logicalRow >= 0 && logicalRow < st.indices.length
}

export function viewPhysicalRow(table: TableView, logicalRow: number): number {
  const st = viewState(table)
  if (!st) return logicalRow
  return st.indices[logicalRow]!
}

/** Shallow select: reuse column object references (no data copy). */
export function selectView(table: TableView, names: string[]): TableView {
  const dense = ensureMaterialized(table)
  const cols: Column[] = names.map((n) => getColumn(dense, n))
  return tableFromColumns(cols)
}
