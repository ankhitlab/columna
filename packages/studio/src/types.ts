import type { DataFrame } from 'columna'

export type ConsoleLevel = 'log' | 'info' | 'warn' | 'error' | 'result'

export type RichPayload =
  | { kind: 'dataframe'; columns: string[]; rows: Record<string, unknown>[] }
  | { kind: 'json'; json: unknown }
  | { kind: 'html'; html: string }

export interface ConsoleEntry {
  id: string
  level: ConsoleLevel
  text: string
  ts: number
  rich?: RichPayload
}

export interface SessionFile {
  id: string
  name: string
  content: string
  handle?: FileSystemFileHandle
}

export type PlotType = 'scatter' | 'line' | 'bar' | 'hist'

export interface PlotSpec {
  id: string
  title: string
  type: PlotType
  x: (number | string)[]
  y: number[]
  createdAt: number
}

export interface VarInfo {
  name: string
  type: string
  preview: string
  value: unknown
  isDataFrame: boolean
}

export function isDataFrame(value: unknown): value is DataFrame {
  return (
    !!value &&
    typeof value === 'object' &&
    'table' in value &&
    'shape' in value &&
    typeof (value as DataFrame).toArray === 'function'
  )
}

export function describeValue(value: unknown): Pick<VarInfo, 'type' | 'preview' | 'isDataFrame'> {
  if (isDataFrame(value)) {
    const [rows, cols] = value.shape
    return {
      type: 'DataFrame',
      preview: `${rows}×${cols} [${value.columns.join(', ')}]`,
      isDataFrame: true,
    }
  }
  if (value === null) return { type: 'null', preview: 'null', isDataFrame: false }
  if (Array.isArray(value)) {
    return { type: 'Array', preview: `Array(${value.length})`, isDataFrame: false }
  }
  const t = typeof value
  if (t === 'object') {
    try {
      const json = JSON.stringify(value)
      return { type: 'Object', preview: (json ?? '[object]').slice(0, 80), isDataFrame: false }
    } catch {
      return { type: 'Object', preview: '[object]', isDataFrame: false }
    }
  }
  return { type: t, preview: String(value), isDataFrame: false }
}

let idCounter = 0
export function uid(prefix = 'id'): string {
  idCounter += 1
  return `${prefix}_${Date.now().toString(36)}_${idCounter}`
}
