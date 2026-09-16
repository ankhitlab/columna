import { uid, type SessionFile } from '../types'

const DEFAULT_CODE = `// Columna Studio — Spyder-like playground
// %% Load data
const df = DataFrame.fromRows([
  { city: 'Berlin', age: 30, salary: 72000 },
  { city: 'Berlin', age: 22, salary: 48000 },
  { city: 'Paris', age: 41, salary: 91000 },
  { city: 'Paris', age: 35, salary: 80000 },
])

// %% Aggregate
const summary = await df
  .filter(col('age').gt(18))
  .groupBy('city')
  .agg({ salary: 'mean', age: 'count' })
  .sort(col('salary').desc())
  .collect()

console.log(summary.toArray())

// %% Plots
plot(df, { x: 'age', y: 'salary', type: 'scatter' })
hist(df, 'age')
summary
`

/** Soft cap to avoid unbounded session retention of large scripts. */
const MAX_SESSION_FILES = 24

type FilePickerWindow = Window & {
  showOpenFilePicker?: (options?: unknown) => Promise<FileSystemFileHandle[]>
  showSaveFilePicker?: (options?: unknown) => Promise<FileSystemFileHandle>
}

export type CsvImportResult = { varName: string; csv: string; fileName: string }

export function csvVarName(filename: string): string {
  return filename.replace(/\.csv$/i, '').replace(/\W+/g, '_') || 'imported'
}

function csvStub(varName: string, filename: string): string {
  return [
    `// CSV "${filename}" is loaded into Variable Explorer as \`${varName}\``,
    `// (raw text is not embedded here to avoid duplicate memory).`,
    varName,
    '',
  ].join('\n')
}

export class SessionFiles {
  files: SessionFile[] = []
  activeId: string | null = null
  private listeners = new Set<() => void>()

  constructor() {
    const id = uid('file')
    this.files = [{ id, name: 'main.ts', content: DEFAULT_CODE }]
    this.activeId = id
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn)
    return () => this.listeners.delete(fn)
  }

  private emit(): void {
    for (const fn of this.listeners) fn()
  }

  private enforceLimit(): void {
    while (this.files.length > MAX_SESSION_FILES) {
      const victim = this.files.find((f) => f.id !== this.activeId && f.name !== 'main.ts')
      if (!victim) break
      this.files = this.files.filter((f) => f.id !== victim.id)
    }
  }

  getActive(): SessionFile | undefined {
    return this.files.find((f) => f.id === this.activeId)
  }

  setActive(id: string): void {
    this.activeId = id
    this.emit()
  }

  updateContent(id: string, content: string): void {
    this.files = this.files.map((f) => (f.id === id ? { ...f, content } : f))
    this.emit()
  }

  create(name = `script_${this.files.length + 1}.ts`): SessionFile {
    const file: SessionFile = { id: uid('file'), name, content: '// new script\n' }
    this.files = [...this.files, file]
    this.activeId = file.id
    this.enforceLimit()
    this.emit()
    return file
  }

  remove(id: string): void {
    this.files = this.files.filter((f) => f.id !== id)
    if (this.activeId === id) this.activeId = this.files[0]?.id ?? null
    this.emit()
  }

  /** Add a short stub script for a CSV; caller should seed the DataFrame separately. */
  importCsvStub(filename: string, varName = csvVarName(filename)): SessionFile {
    const sf: SessionFile = {
      id: uid('file'),
      name: filename.replace(/\.csv$/i, '.ts'),
      content: csvStub(varName, filename),
    }
    this.files = [...this.files, sf]
    this.activeId = sf.id
    this.enforceLimit()
    this.emit()
    return sf
  }

  async openFromPicker(): Promise<CsvImportResult | null> {
    const w = window as FilePickerWindow

    if (typeof w.showOpenFilePicker === 'function') {
      const [handle] = await w.showOpenFilePicker({
        multiple: false,
        types: [
          {
            description: 'Scripts & CSV',
            accept: {
              'text/typescript': ['.ts'],
              'text/javascript': ['.js'],
              'text/csv': ['.csv'],
              'text/plain': ['.txt'],
            },
          },
        ],
      })
      const file = await handle.getFile()
      const content = await file.text()
      if (file.name.toLowerCase().endsWith('.csv')) {
        const varName = csvVarName(file.name)
        const sf: SessionFile = {
          id: uid('file'),
          name: file.name.replace(/\.csv$/i, '.ts'),
          content: csvStub(varName, file.name),
          handle,
        }
        this.files = [...this.files, sf]
        this.activeId = sf.id
        this.enforceLimit()
        this.emit()
        return { varName, csv: content, fileName: file.name }
      }
      const sf: SessionFile = { id: uid('file'), name: file.name, content, handle }
      this.files = [...this.files, sf]
      this.activeId = sf.id
      this.enforceLimit()
      this.emit()
      return null
    }

    return await new Promise<CsvImportResult | null>((resolve) => {
      const input = document.createElement('input')
      input.type = 'file'
      input.accept = '.ts,.js,.csv,.txt'
      input.onchange = async () => {
        const file = input.files?.[0]
        if (!file) return resolve(null)
        const content = await file.text()
        if (file.name.toLowerCase().endsWith('.csv')) {
          const varName = csvVarName(file.name)
          const sf: SessionFile = {
            id: uid('file'),
            name: file.name.replace(/\.csv$/i, '.ts'),
            content: csvStub(varName, file.name),
          }
          this.files = [...this.files, sf]
          this.activeId = sf.id
          this.enforceLimit()
          this.emit()
          resolve({ varName, csv: content, fileName: file.name })
          return
        }
        const sf: SessionFile = { id: uid('file'), name: file.name, content }
        this.files = [...this.files, sf]
        this.activeId = sf.id
        this.enforceLimit()
        this.emit()
        resolve(null)
      }
      input.oncancel = () => resolve(null)
      input.click()
    })
  }

  async saveActive(): Promise<void> {
    const active = this.getActive()
    if (!active) return

    if (active.handle && 'createWritable' in active.handle) {
      const writable = await active.handle.createWritable()
      await writable.write(active.content)
      await writable.close()
      return
    }

    const w = window as FilePickerWindow
    if (typeof w.showSaveFilePicker === 'function') {
      const handle = await w.showSaveFilePicker({
        suggestedName: active.name,
        types: [{ description: 'TypeScript', accept: { 'text/typescript': ['.ts'] } }],
      })
      const writable = await handle.createWritable()
      await writable.write(active.content)
      await writable.close()
      this.files = this.files.map((f) => (f.id === active.id ? { ...f, handle, name: handle.name } : f))
      this.emit()
      return
    }

    const blob = new Blob([active.content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = active.name
    a.click()
    URL.revokeObjectURL(url)
  }
}
