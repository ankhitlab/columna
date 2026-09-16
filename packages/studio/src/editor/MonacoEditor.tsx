import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'
import * as monaco from 'monaco-editor'
import editorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker'
import tsWorker from 'monaco-editor/esm/vs/language/typescript/ts.worker?worker'
import { cellAtLine, isCellMarker, nextCell, parseCells } from './cells'

let monacoReady = false

function ensureMonacoEnv() {
  if (monacoReady) return
  ;(globalThis as typeof globalThis & { MonacoEnvironment?: unknown }).MonacoEnvironment = {
    getWorker(_moduleId: string, label: string) {
      if (label === 'typescript' || label === 'javascript') return new tsWorker()
      return new editorWorker()
    },
  }
  monaco.languages.typescript.typescriptDefaults.setCompilerOptions({
    target: monaco.languages.typescript.ScriptTarget.ES2020,
    allowNonTsExtensions: true,
    moduleResolution: monaco.languages.typescript.ModuleResolutionKind.NodeJs,
    module: monaco.languages.typescript.ModuleKind.ESNext,
    noEmit: true,
    esModuleInterop: true,
    strict: false,
  })
  monaco.languages.typescript.typescriptDefaults.setDiagnosticsOptions({
    noSemanticValidation: false,
    noSyntaxValidation: false,
  })
  monaco.languages.typescript.typescriptDefaults.addExtraLib(
    `
declare class DataFrame {
  static fromRows(rows: Record<string, unknown>[]): DataFrame;
  static fromCSV(csv: string, options?: Record<string, unknown>): DataFrame;
  readonly shape: [number, number];
  readonly columns: string[];
  readonly dtypes: Record<string, string>;
  filter(predicate: unknown): LazyFrame;
  select(...cols: any[]): LazyFrame;
  groupBy(...keys: string[]): GroupBy;
  sort(...by: any[]): LazyFrame;
  toArray(): Record<string, unknown>[];
}
declare class LazyFrame {
  filter(predicate: unknown): LazyFrame;
  groupBy(...keys: string[]): GroupBy;
  agg(spec: Record<string, any>): LazyFrame;
  sort(...by: any[]): LazyFrame;
  collect(): Promise<DataFrame>;
}
declare class GroupBy { agg(spec: Record<string, any>): LazyFrame; }
declare function col(name: string): any;
declare function lit(value: any): any;
declare function init(): Promise<any>;
declare function plot(df: any, opts: { x: string; y: string; type?: string; title?: string }): any;
declare function hist(df: any, column: string, bins?: number): any;
declare function clear(): void;
declare function help(): void;
`,
    'ts:columna-ambient.d.ts',
  )
  monacoReady = true
}

export interface MonacoEditorHandle {
  jumpToLine: (line: number) => void
  getValue: () => string
  getCurrentCellCode: () => string
  runCurrentCell: () => void
  runCurrentCellAndAdvance: () => void
  runAllCells: () => void
  runCurrentLine: () => void
  getWordAtCursor: () => string | null
  getModelMarkers: () => monaco.editor.IMarker[]
}

export interface MonacoEditorProps {
  value: string
  onChange: (value: string) => void
  onRun: (code: string) => void | Promise<void>
  breakpoints?: number[]
  pauseLine?: number | null
  onToggleBreakpoint?: (line: number) => void
  onRenameRequest?: (symbol: string) => void
  completionNames?: string[]
}

export const MonacoEditor = forwardRef<MonacoEditorHandle, MonacoEditorProps>(function MonacoEditor(
  { value, onChange, onRun, breakpoints = [], pauseLine = null, onToggleBreakpoint, onRenameRequest, completionNames = [] },
  ref,
) {
  const hostRef = useRef<HTMLDivElement>(null)
  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null)
  const cellDecoRef = useRef<string[]>([])
  const bpDecoRef = useRef<string[]>([])
  const onRunRef = useRef(onRun)
  const onChangeRef = useRef(onChange)
  const onToggleBpRef = useRef(onToggleBreakpoint)
  const onRenameRef = useRef(onRenameRequest)
  const completionRef = useRef(completionNames)
  onRunRef.current = onRun
  onChangeRef.current = onChange
  onToggleBpRef.current = onToggleBreakpoint
  onRenameRef.current = onRenameRequest
  completionRef.current = completionNames

  const updateCellDecorations = (editor: monaco.editor.IStandaloneCodeEditor) => {
    const model = editor.getModel()
    if (!model) return
    const lines = model.getLinesContent()
    const decos: monaco.editor.IModelDeltaDecoration[] = []
    for (let i = 0; i < lines.length; i++) {
      if (!isCellMarker(lines[i]!)) continue
      decos.push({
        range: new monaco.Range(i + 1, 1, i + 1, 1),
        options: {
          isWholeLine: true,
          className: 'cell-marker-line',
          glyphMarginClassName: 'cell-marker-glyph',
          overviewRuler: {
            color: '#0f6e56',
            position: monaco.editor.OverviewRulerLane.Left,
          },
        },
      })
    }
    cellDecoRef.current = editor.deltaDecorations(cellDecoRef.current, decos)
  }

  const updateBpDecorations = (
    editor: monaco.editor.IStandaloneCodeEditor,
    bps: number[],
    pause: number | null,
  ) => {
    const decos: monaco.editor.IModelDeltaDecoration[] = []
    for (const line of bps) {
      decos.push({
        range: new monaco.Range(line, 1, line, 1),
        options: {
          isWholeLine: false,
          glyphMarginClassName: 'breakpoint-glyph',
          overviewRuler: { color: '#a33b2b', position: monaco.editor.OverviewRulerLane.Left },
        },
      })
    }
    if (pause && pause > 0) {
      decos.push({
        range: new monaco.Range(pause, 1, pause, 1),
        options: {
          isWholeLine: true,
          className: 'debug-pause-line',
          glyphMarginClassName: 'debug-pause-glyph',
        },
      })
    }
    bpDecoRef.current = editor.deltaDecorations(bpDecoRef.current, decos)
  }

  useImperativeHandle(ref, () => ({
    jumpToLine(line: number) {
      const editor = editorRef.current
      if (!editor) return
      editor.revealLineInCenter(line)
      editor.setPosition({ lineNumber: line, column: 1 })
      editor.focus()
    },
    getValue() {
      return editorRef.current?.getValue() ?? ''
    },
    getCurrentCellCode() {
      const editor = editorRef.current
      if (!editor) return ''
      const pos = editor.getPosition()
      if (!pos) return editor.getValue()
      return cellAtLine(editor.getValue(), pos.lineNumber).code
    },
    runCurrentCell() {
      const editor = editorRef.current
      if (!editor) return
      const selection = editor.getSelection()
      const model = editor.getModel()
      if (selection && !selection.isEmpty() && model) {
        onRunRef.current(model.getValueInRange(selection))
        return
      }
      const pos = editor.getPosition()
      const source = editor.getValue()
      const cell = pos ? cellAtLine(source, pos.lineNumber) : parseCells(source)[0]
      if (cell) onRunRef.current(cell.code)
    },
    runCurrentCellAndAdvance() {
      const editor = editorRef.current
      if (!editor) return
      const pos = editor.getPosition()
      const source = editor.getValue()
      if (!pos) return
      const cell = cellAtLine(source, pos.lineNumber)
      onRunRef.current(cell.code)
      const nxt = nextCell(source, pos.lineNumber)
      if (nxt) {
        editor.revealLineInCenter(nxt.startLine)
        editor.setPosition({ lineNumber: nxt.startLine, column: 1 })
      }
    },
    runAllCells() {
      const editor = editorRef.current
      if (!editor) return
      void (async () => {
        for (const cell of parseCells(editor.getValue())) {
          if (cell.code.trim()) await onRunRef.current(cell.code)
        }
      })()
    },
    runCurrentLine() {
      const editor = editorRef.current
      if (!editor) return
      const pos = editor.getPosition()
      const model = editor.getModel()
      if (!pos || !model) return
      onRunRef.current(model.getLineContent(pos.lineNumber))
    },
    getWordAtCursor() {
      const editor = editorRef.current
      const model = editor?.getModel()
      const pos = editor?.getPosition()
      if (!editor || !model || !pos) return null
      const word = model.getWordAtPosition(pos)
      return word?.word ?? null
    },
    getModelMarkers() {
      const model = editorRef.current?.getModel()
      if (!model) return []
      return monaco.editor.getModelMarkers({ resource: model.uri })
    },
  }))

  useEffect(() => {
    ensureMonacoEnv()
    if (!hostRef.current) return

    const editor = monaco.editor.create(hostRef.current, {
      value,
      language: 'typescript',
      theme: 'vs',
      automaticLayout: true,
      minimap: { enabled: false },
      fontFamily: 'IBM Plex Mono, ui-monospace, monospace',
      fontSize: 13,
      lineNumbers: 'on',
      glyphMargin: true,
      scrollBeyondLastLine: false,
      tabSize: 2,
      padding: { top: 8 },
    })
    editorRef.current = editor
    updateCellDecorations(editor)
    updateBpDecorations(editor, breakpoints, pauseLine)

    const sub = editor.onDidChangeModelContent(() => {
      onChangeRef.current(editor.getValue())
      updateCellDecorations(editor)
    })

    const glyph = editor.onMouseDown((e) => {
      if (e.target.type !== monaco.editor.MouseTargetType.GUTTER_GLYPH_MARGIN) return
      const line = e.target.position?.lineNumber
      if (line) onToggleBpRef.current?.(line)
    })

    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => {
      const selection = editor.getSelection()
      const model = editor.getModel()
      if (selection && !selection.isEmpty() && model) {
        onRunRef.current(model.getValueInRange(selection))
        return
      }
      const pos = editor.getPosition()
      if (!pos) return
      const cell = cellAtLine(editor.getValue(), pos.lineNumber)
      onRunRef.current(cell.code)
    })

    editor.addCommand(monaco.KeyMod.Shift | monaco.KeyCode.Enter, () => {
      const pos = editor.getPosition()
      if (!pos) return
      const source = editor.getValue()
      const cell = cellAtLine(source, pos.lineNumber)
      onRunRef.current(cell.code)
      const nxt = nextCell(source, pos.lineNumber)
      if (nxt) {
        editor.revealLineInCenter(nxt.startLine)
        editor.setPosition({ lineNumber: nxt.startLine, column: 1 })
      }
    })

    editor.addCommand(monaco.KeyCode.F2, () => {
      const model = editor.getModel()
      const pos = editor.getPosition()
      if (!model || !pos) return
      const word = model.getWordAtPosition(pos)
      if (word?.word) onRenameRef.current?.(word.word)
    })

    const completion = monaco.languages.registerCompletionItemProvider('typescript', {
      triggerCharacters: ['.'],
      provideCompletionItems: (model, position) => {
        const word = model.getWordUntilPosition(position)
        const range = {
          startLineNumber: position.lineNumber,
          endLineNumber: position.lineNumber,
          startColumn: word.startColumn,
          endColumn: word.endColumn,
        }
        const suggestions = completionRef.current.map((name) => ({
          label: name,
          kind: monaco.languages.CompletionItemKind.Variable,
          insertText: name,
          range,
        }))
        return { suggestions }
      },
    })

    return () => {
      sub.dispose()
      glyph.dispose()
      completion.dispose()
      editor.dispose()
      editorRef.current = null
    }
    // intentionally mount once
  }, [])

  useEffect(() => {
    const editor = editorRef.current
    if (!editor) return
    if (editor.getValue() !== value) {
      const pos = editor.getPosition()
      editor.setValue(value)
      if (pos) editor.setPosition(pos)
      updateCellDecorations(editor)
    }
  }, [value])

  useEffect(() => {
    const editor = editorRef.current
    if (!editor) return
    updateBpDecorations(editor, breakpoints, pauseLine)
  }, [breakpoints, pauseLine])

  return <div className="editor-shell" ref={hostRef} />
})
