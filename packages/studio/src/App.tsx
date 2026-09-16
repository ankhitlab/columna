import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { DataFrame } from 'columna'
import { MonacoEditor, type MonacoEditorHandle } from './editor/MonacoEditor'
import { ConsolePane } from './repl/ConsolePane'
import { ConsoleManager } from './repl/ConsoleManager'
import { createPlotApi } from './plots/plotApi'
import { PlotsPane } from './plots/PlotsPane'
import { PlotDialog } from './plots/PlotDialog'
import { VariableExplorer } from './vars/VariableExplorer'
import { DataFrameViewer } from './vars/DataFrameViewer'
import { ExternalWindow } from './vars/ExternalWindow'
import { VarWorkspaceWindow } from './vars/VarWorkspaceWindow'
import { SessionFiles, csvVarName } from './fs/SessionFiles'
import { FileSidebar } from './fs/FileSidebar'
import { ProjectFS, supportsDirectoryPicker } from './fs/ProjectFS'
import { ProjectTree } from './fs/ProjectTree'
import { OutlinePane } from './outline/OutlinePane'
import { HistoryPane } from './history/HistoryPane'
import { HelpPane } from './help/HelpPane'
import { MenuBar, type MenuDef } from './shell/MenuBar'
import { DebuggerRuntime } from './debug/Debugger'
import { DebugPane } from './debug/DebugPane'
import { FindInFiles } from './search/FindInFiles'
import { RenameDialog } from './search/RenameDialog'
import { analyzeStudioHeuristics, markersToIssues, type AnalysisIssue } from './analysis/Analysis'
import { AnalysisPane } from './analysis/AnalysisPane'
import {
  loadLayout,
  saveLayout,
  type BottomTab,
  type LeftTab,
  type RightTab,
} from './shell/layoutPersist'
import { isDataFrame } from './types'

export function App() {
  const plotApi = useMemo(() => createPlotApi(), [])
  const consoles = useMemo(() => new ConsoleManager(plotApi), [plotApi])
  const files = useMemo(() => new SessionFiles(), [])
  const project = useMemo(() => new ProjectFS(), [])
  const dbg = useMemo(() => new DebuggerRuntime(), [])
  const editorRef = useRef<MonacoEditorHandle>(null)

  const initial = useMemo(() => loadLayout(), [])
  const [tick, setTick] = useState(0)
  const bump = () => setTick((n) => n + 1)

  const [selectedVar, setSelectedVar] = useState<string | null>(null)
  const [leftTab, setLeftTab] = useState<LeftTab>(initial.leftTab)
  const [bottomTab, setBottomTab] = useState<BottomTab>(initial.bottomTab)
  const [rightTab, setRightTab] = useState<RightTab>(initial.rightTab)
  const [helpMode, setHelpMode] = useState<'guide' | 'inspect'>('guide')
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(false)
  const [editorRatio, setEditorRatio] = useState(initial.editorRatio)
  const [plotOpen, setPlotOpen] = useState(false)
  const [consoleDraft, setConsoleDraft] = useState<string | undefined>(undefined)
  const [varWindowOpen, setVarWindowOpen] = useState(false)
  const [windowVar, setWindowVar] = useState<string | null>(null)
  const [renameSymbol, setRenameSymbol] = useState<string | null>(null)
  const [analysisIssues, setAnalysisIssues] = useState<AnalysisIssue[]>([])

  useEffect(() => {
    saveLayout({ editorRatio, leftTab, bottomTab, rightTab })
  }, [editorRatio, leftTab, bottomTab, rightTab])

  useEffect(() => files.subscribe(bump), [files])
  useEffect(() => project.subscribe(bump), [project])
  useEffect(() => consoles.subscribe(bump), [consoles])
  useEffect(() => plotApi.subscribe(bump), [plotApi])
  useEffect(() => dbg.subscribe(bump), [dbg])
  useEffect(() => {
    consoles.setProject(project)
  }, [consoles, project])

  void tick

  const repl = consoles.getEngine()
  const sessionActive = files.getActive()
  const projectActive = project.getActive()
  const useProject = project.isOpen && !!projectActive
  const activeName = useProject ? projectActive!.name : (sessionActive?.name ?? '—')
  const activeContent = useProject ? projectActive!.content : (sessionActive?.content ?? '')
  const activePath = useProject ? projectActive!.path : (sessionActive?.name ?? null)

  const vars = repl.getVars()
  const entries = repl.getConsoleEntries()
  const history = repl.getHistory()
  const plots = plotApi.getPlots()
  const selectedValue = selectedVar ? repl.getVar(selectedVar) : null
  const breakpoints = [...dbg.breakpoints].sort((a, b) => a - b)
  const pauseLine = dbg.paused?.line ?? null

  const updateActiveContent = (content: string) => {
    if (useProject && projectActive) project.updateContent(projectActive.id, content)
    else if (sessionActive) files.updateContent(sessionActive.id, content)
  }

  const saveActive = async () => {
    if (useProject) await project.saveActive()
    else await files.saveActive()
  }

  const runCode = useCallback(
    async (code: string, opts?: { debug?: boolean }) => {
      setBusy(true)
      try {
        const cleaned = code
          .split('\n')
          .filter((line) => !/^\s*import\s+.*from\s+['"]columna['"]\s*;?\s*$/.test(line))
          .join('\n')
        const result = await repl.run(cleaned, {
          debug: opts?.debug,
          debugger: opts?.debug ? dbg : undefined,
        })
        const dfVar = result.vars.find((v) => v.isDataFrame)
        if (dfVar && !selectedVar) setSelectedVar(dfVar.name)
        if (!opts?.debug) setBottomTab('console')
        else setBottomTab('debug')
      } finally {
        setBusy(false)
      }
    },
    [repl, selectedVar, dbg],
  )

  const seedCsv = useCallback(
    (fileName: string, csv: string, varName = csvVarName(fileName)) => {
      files.importCsvStub(fileName, varName)
      const df = DataFrame.fromCSV(csv)
      repl.seedDataFrame(varName, df)
      setSelectedVar(varName)
      repl.push('info', `Loaded CSV into DataFrame \`${varName}\` (${df.shape[0]}×${df.shape[1]})`)
    },
    [files, repl],
  )

  const openFile = useCallback(async () => {
    const csv = await files.openFromPicker()
    if (csv) seedCsv(csv.fileName, csv.csv, csv.varName)
  }, [files, seedCsv])

  const openFolder = async () => {
    try {
      await project.openFolder()
      setLeftTab('project')
    } catch (err) {
      repl.push('error', err instanceof Error ? err.message : String(err))
    }
  }

  const onDrop = async (fileList: FileList) => {
    for (const file of Array.from(fileList)) {
      const text = await file.text()
      if (file.name.toLowerCase().endsWith('.csv')) seedCsv(file.name, text)
      else {
        files.create(file.name)
        const created = files.getActive()
        if (created) files.updateContent(created.id, text)
      }
    }
  }

  const openPlotDialog = (name: string) => {
    setSelectedVar(name)
    setPlotOpen(true)
  }

  const inspectVar = (name: string) => {
    setSelectedVar(name)
    setHelpMode('inspect')
    setRightTab('help')
  }

  const openVarWindow = (name: string) => {
    setSelectedVar(name)
    setWindowVar(name)
    setVarWindowOpen(true)
  }

  const windowSelected = windowVar ?? selectedVar
  const windowValue = windowSelected ? repl.getVar(windowSelected) : null

  const searchableFiles = useMemo(() => {
    if (project.isOpen) {
      return project.buffers.map((b) => ({ path: b.path, content: b.content }))
    }
    return files.files.map((f) => ({ path: f.name, content: f.content }))
  }, [project.isOpen, project.buffers, files.files, tick])

  const refreshAnalysis = () => {
    const studio = analyzeStudioHeuristics(activeContent)
    const markers = editorRef.current?.getModelMarkers() ?? []
    const ts = markersToIssues(markers)
    setAnalysisIssues([...ts, ...studio].sort((a, b) => a.line - b.line))
  }

  const jumpOpen = async (path: string, line: number) => {
    if (project.isOpen) {
      await project.openFile(path)
      setLeftTab('project')
    } else {
      const f = files.files.find((x) => x.name === path)
      if (f) files.setActive(f.id)
    }
    setTimeout(() => editorRef.current?.jumpToLine(line), 50)
  }

  const menus: MenuDef[] = [
    {
      id: 'file',
      label: 'File',
      items: [
        { id: 'file.new', label: 'New script' },
        { id: 'file.open', label: 'Open…', shortcut: 'Ctrl+O' },
        { id: 'file.openFolder', label: 'Open folder…', disabled: !supportsDirectoryPicker() },
        { id: 'file.closeFolder', label: 'Close folder', disabled: !project.isOpen },
        { id: 'file.save', label: 'Save', shortcut: 'Ctrl+S' },
      ],
    },
    {
      id: 'edit',
      label: 'Edit',
      items: [
        { id: 'edit.rename', label: 'Rename symbol', shortcut: 'F2' },
        { id: 'edit.find', label: 'Find in files' },
        { id: 'edit.clearConsole', label: 'Clear console' },
        { id: 'edit.clearHistory', label: 'Clear history' },
        { id: 'edit.clearVars', label: 'Clear variables' },
      ],
    },
    {
      id: 'run',
      label: 'Run',
      items: [
        { id: 'run.file', label: 'Run file', shortcut: 'F5', disabled: busy },
        { id: 'run.debug', label: 'Debug Run', shortcut: 'F9', disabled: busy },
        { id: 'run.cell', label: 'Run cell', shortcut: 'Ctrl+Enter', disabled: busy },
        { id: 'run.cellAdvance', label: 'Run cell and advance', shortcut: 'Shift+Enter', disabled: busy },
        { id: 'run.allCells', label: 'Run all cells', disabled: busy },
        { id: 'run.line', label: 'Run current line', disabled: busy },
      ],
    },
    {
      id: 'view',
      label: 'View',
      items: [
        { id: 'view.project', label: 'Project' },
        { id: 'view.files', label: 'Session files' },
        { id: 'view.outline', label: 'Outline' },
        { id: 'view.console', label: 'Console' },
        { id: 'view.debug', label: 'Debug' },
        { id: 'view.find', label: 'Find' },
        { id: 'view.analysis', label: 'Analysis' },
        { id: 'view.history', label: 'History' },
        { id: 'view.plots', label: 'Plots' },
        { id: 'view.vars', label: 'Variable Explorer' },
        { id: 'view.varWindow', label: 'Open Variable Explorer window' },
        { id: 'view.help', label: 'Help' },
      ],
    },
    {
      id: 'help',
      label: 'Help',
      items: [
        { id: 'help.studio', label: 'Studio Help' },
        { id: 'help.repl', label: 'REPL help()' },
      ],
    },
  ]

  const onMenuAction = (id: string) => {
    switch (id) {
      case 'file.new':
        files.create()
        break
      case 'file.open':
        void openFile()
        break
      case 'file.openFolder':
        void openFolder()
        break
      case 'file.closeFolder':
        project.closeFolder()
        break
      case 'file.save':
        void saveActive()
        break
      case 'edit.rename': {
        const w = editorRef.current?.getWordAtCursor()
        if (w) setRenameSymbol(w)
        break
      }
      case 'edit.find':
        setBottomTab('find')
        break
      case 'edit.clearConsole':
        repl.clearConsole()
        break
      case 'edit.clearHistory':
        repl.clearHistory()
        break
      case 'edit.clearVars':
        if (confirm('Remove all user variables?')) {
          repl.clearUserVars()
          setSelectedVar(null)
        }
        break
      case 'run.file':
        void runCode(activeContent)
        break
      case 'run.debug':
        void runCode(activeContent, { debug: true })
        break
      case 'run.cell':
        editorRef.current?.runCurrentCell()
        break
      case 'run.cellAdvance':
        editorRef.current?.runCurrentCellAndAdvance()
        break
      case 'run.allCells':
        editorRef.current?.runAllCells()
        break
      case 'run.line':
        editorRef.current?.runCurrentLine()
        break
      case 'view.project':
        setLeftTab('project')
        break
      case 'view.files':
        setLeftTab('files')
        break
      case 'view.outline':
        setLeftTab('outline')
        break
      case 'view.console':
        setBottomTab('console')
        break
      case 'view.debug':
        setBottomTab('debug')
        break
      case 'view.find':
        setBottomTab('find')
        break
      case 'view.analysis':
        refreshAnalysis()
        setBottomTab('analysis')
        break
      case 'view.history':
        setBottomTab('history')
        break
      case 'view.plots':
        setBottomTab('plots')
        break
      case 'view.vars':
        setRightTab('vars')
        break
      case 'view.varWindow':
        setVarWindowOpen(true)
        if (selectedVar) setWindowVar(selectedVar)
        break
      case 'view.help':
      case 'help.studio':
        setHelpMode('guide')
        setRightTab('help')
        break
      case 'help.repl':
        void runCode('help()')
        break
      default:
        break
    }
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault()
        void saveActive()
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'o') {
        e.preventDefault()
        void openFile()
      }
      if (e.key === 'F5') {
        e.preventDefault()
        void runCode(activeContent)
      }
      if (e.key === 'F9') {
        e.preventDefault()
        void runCode(activeContent, { debug: true })
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [activeContent, openFile, runCode])

  const onSplitPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    e.preventDefault()
    const stack = e.currentTarget.parentElement
    if (!stack) return
    const startY = e.clientY
    const startRatio = editorRatio
    const rect = stack.getBoundingClientRect()
    const onMove = (ev: PointerEvent) => {
      const delta = ev.clientY - startY
      setEditorRatio(Math.min(0.8, Math.max(0.25, startRatio + delta / rect.height)))
    }
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  return (
    <div
      className="app"
      onDragEnter={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        if (e.dataTransfer.files?.length) void onDrop(e.dataTransfer.files)
      }}
    >
      {dragging && <div className="drop-overlay">Drop CSV or .ts files</div>}

      <header className="chrome">
        <div className="chrome-top">
          <div className="brand">Columna Studio</div>
          <MenuBar menus={menus} onAction={onMenuAction} />
          <div className="hint">F5 run · F9 debug · F2 rename · gutter BP</div>
        </div>
        <div className="toolbar">
          <div className="toolbar-actions">
            <button className="btn btn-primary" disabled={busy} type="button" onClick={() => void runCode(activeContent)}>
              {busy ? 'Running…' : 'Run'}
            </button>
            <button className="btn" disabled={busy} type="button" onClick={() => void runCode(activeContent, { debug: true })}>
              Debug
            </button>
            <button className="btn" disabled={busy} type="button" onClick={() => editorRef.current?.runCurrentCell()}>
              Run cell
            </button>
            <button className="btn" type="button" onClick={() => void openFolder()} disabled={!supportsDirectoryPicker()}>
              Open folder
            </button>
            <button className="btn" type="button" onClick={() => void openFile()}>
              Open
            </button>
            <button className="btn" type="button" onClick={() => void saveActive()}>
              Save
            </button>
            {project.isOpen && (
              <span className="chip">
                cwd: {project.cwd === '.' ? project.rootName : `${project.rootName}/${project.cwd}`}
              </span>
            )}
          </div>
        </div>
      </header>

      <div className="workspace">
        <div className="left-stack">
          <div className="side-tabs">
            <button type="button" className={`tab ${leftTab === 'project' ? 'active' : ''}`} onClick={() => setLeftTab('project')}>
              Project
            </button>
            <button type="button" className={`tab ${leftTab === 'files' ? 'active' : ''}`} onClick={() => setLeftTab('files')}>
              Session
            </button>
            <button type="button" className={`tab ${leftTab === 'outline' ? 'active' : ''}`} onClick={() => setLeftTab('outline')}>
              Outline
            </button>
          </div>
          {leftTab === 'project' && (
            <ProjectTree
              tree={project.tree}
              rootName={project.rootName}
              cwd={project.cwd}
              activePath={projectActive?.path ?? null}
              onOpenFolder={() => void openFolder()}
              onCloseFolder={() => project.closeFolder()}
              onRefresh={() => void project.refreshTree()}
              onOpenFile={(path) => void project.openFile(path)}
              onSetCwd={(path) => project.setCwd(path)}
            />
          )}
          {leftTab === 'files' && (
            <FileSidebar
              files={files.files}
              activeId={files.activeId}
              onSelect={(id) => files.setActive(id)}
              onCreate={() => files.create()}
              onRemove={(id) => files.remove(id)}
            />
          )}
          {leftTab === 'outline' && (
            <OutlinePane source={activeContent} onJump={(line) => editorRef.current?.jumpToLine(line)} />
          )}
        </div>

        <div
          className="center-stack"
          style={{ gridTemplateRows: `minmax(0, ${editorRatio}fr) 8px minmax(160px, ${1 - editorRatio}fr)` }}
        >
          <div className="panel" style={{ borderRight: 'none' }}>
            <div className="panel-header">
              <span>
                Editor · {activeName}
                {activePath && useProject ? ` · ${activePath}` : ''}
              </span>
            </div>
            <div className="panel-body" style={{ overflow: 'hidden' }}>
              <MonacoEditor
                ref={editorRef}
                value={activeContent}
                onChange={updateActiveContent}
                onRun={(code) => void runCode(code)}
                breakpoints={breakpoints}
                pauseLine={pauseLine}
                onToggleBreakpoint={(line) => dbg.toggleBreakpoint(line)}
                onRenameRequest={(sym) => setRenameSymbol(sym)}
                completionNames={repl.getCompletionNames()}
              />
            </div>
          </div>

          <div className="split-handle" title="Drag to resize" onPointerDown={onSplitPointerDown} />

          <div className="panel" style={{ borderRight: 'none' }}>
            <div className="panel-header">
              <div className="tabs" style={{ flexWrap: 'wrap' }}>
                {consoles.slots.map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    className={`tab ${bottomTab === 'console' && consoles.activeId === s.id ? 'active' : ''}`}
                    onClick={() => {
                      consoles.setActive(s.id)
                      setBottomTab('console')
                    }}
                  >
                    {s.name}
                  </button>
                ))}
                <button type="button" className="tab" onClick={() => consoles.addConsole()} title="New console">
                  +
                </button>
                <button type="button" className={`tab ${bottomTab === 'history' ? 'active' : ''}`} onClick={() => setBottomTab('history')}>
                  History
                </button>
                <button type="button" className={`tab ${bottomTab === 'plots' ? 'active' : ''}`} onClick={() => setBottomTab('plots')}>
                  Plots ({plots.length})
                </button>
                <button type="button" className={`tab ${bottomTab === 'debug' ? 'active' : ''}`} onClick={() => setBottomTab('debug')}>
                  Debug
                </button>
                <button type="button" className={`tab ${bottomTab === 'find' ? 'active' : ''}`} onClick={() => setBottomTab('find')}>
                  Find
                </button>
                <button
                  type="button"
                  className={`tab ${bottomTab === 'analysis' ? 'active' : ''}`}
                  onClick={() => {
                    refreshAnalysis()
                    setBottomTab('analysis')
                  }}
                >
                  Analysis
                </button>
              </div>
            </div>
            <div className="panel-body" style={{ overflow: 'hidden' }}>
              {bottomTab === 'console' && (
                <ConsolePane
                  entries={entries}
                  history={history}
                  completions={repl.getCompletionNames()}
                  draft={consoleDraft}
                  onDraftConsumed={() => setConsoleDraft(undefined)}
                  onSubmit={(code) => void runCode(code)}
                  onClear={() => repl.clearConsole()}
                />
              )}
              {bottomTab === 'history' && (
                <HistoryPane
                  commands={history}
                  onRerun={(code) => void runCode(code)}
                  onPaste={(code) => {
                    setConsoleDraft(code)
                    setBottomTab('console')
                  }}
                  onClear={() => repl.clearHistory()}
                />
              )}
              {bottomTab === 'plots' && (
                <PlotsPane plots={plots} onClear={() => plotApi.clear()} onRemove={(id) => plotApi.remove(id)} />
              )}
              {bottomTab === 'debug' && (
                <DebugPane
                  dbg={dbg}
                  paused={dbg.paused}
                  running={dbg.running}
                  breakpoints={breakpoints}
                  onAction={(a) => dbg.action(a)}
                  onJumpLine={(line) => editorRef.current?.jumpToLine(line)}
                  onEvalWatch={async (expr) => {
                    try {
                      const scope = { ...repl.getScope(), ...(dbg.paused?.locals ?? {}) }
                      const v = await dbg.evalWatch(expr, scope)
                      return typeof v === 'string' ? v : JSON.stringify(v, null, 2) ?? String(v)
                    } catch (err) {
                      return err instanceof Error ? err.message : String(err)
                    }
                  }}
                />
              )}
              {bottomTab === 'find' && (
                <FindInFiles files={searchableFiles} onOpen={(path, line) => void jumpOpen(path, line)} />
              )}
              {bottomTab === 'analysis' && (
                <AnalysisPane
                  issues={analysisIssues}
                  onJump={(line) => editorRef.current?.jumpToLine(line)}
                  onRefresh={refreshAnalysis}
                />
              )}
            </div>
          </div>
        </div>

        <div className="right-stack">
          <div className="side-tabs">
            <button type="button" className={`tab ${rightTab === 'vars' ? 'active' : ''}`} onClick={() => setRightTab('vars')}>
              Variables
            </button>
            <button
              type="button"
              className={`tab ${rightTab === 'help' ? 'active' : ''}`}
              onClick={() => {
                setHelpMode('guide')
                setRightTab('help')
              }}
            >
              Help
            </button>
          </div>
          <div className="right-top">
            {rightTab === 'vars' ? (
              <VariableExplorer
                vars={vars}
                selected={selectedVar}
                onSelect={setSelectedVar}
                onInspect={inspectVar}
                onOpenWindow={openVarWindow}
                onRemove={(name) => {
                  repl.removeVar(name)
                  if (selectedVar === name) setSelectedVar(null)
                  if (windowVar === name) setWindowVar(null)
                }}
                onClearAll={() => {
                  repl.clearUserVars()
                  setSelectedVar(null)
                  setWindowVar(null)
                }}
                onRefresh={() => repl.refresh()}
                onEdit={(name, value) => repl.setVar(name, value)}
              />
            ) : (
              <HelpPane selectedName={selectedVar} selectedValue={selectedValue} mode={helpMode} />
            )}
          </div>
          <DataFrameViewer
            name={selectedVar}
            value={selectedValue}
            onPlot={openPlotDialog}
            onOpenWindow={openVarWindow}
          />
        </div>
      </div>

      <ExternalWindow
        open={varWindowOpen}
        title={`Variable Explorer${windowSelected ? ` · ${windowSelected}` : ''} — Columna Studio`}
        onClose={() => setVarWindowOpen(false)}
      >
        <VarWorkspaceWindow
          vars={vars}
          selected={windowSelected}
          selectedValue={windowValue}
          onSelect={(name) => {
            setWindowVar(name)
            setSelectedVar(name)
          }}
          onRemove={(name) => {
            repl.removeVar(name)
            if (selectedVar === name) setSelectedVar(null)
            if (windowVar === name) setWindowVar(null)
          }}
          onClearAll={() => {
            repl.clearUserVars()
            setSelectedVar(null)
            setWindowVar(null)
          }}
          onRefresh={() => repl.refresh()}
          onEdit={(name, value) => repl.setVar(name, value)}
          onPlot={(name) => {
            openPlotDialog(name)
            window.focus()
          }}
          onClose={() => setVarWindowOpen(false)}
        />
      </ExternalWindow>

      <RenameDialog
        open={!!renameSymbol}
        symbol={renameSymbol ?? ''}
        files={searchableFiles}
        onClose={() => setRenameSymbol(null)}
        onApply={(edits) => {
          for (const e of edits) {
            if (project.isOpen) {
              const buf = project.buffers.find((b) => b.path === e.path)
              if (buf) project.updateContent(buf.id, e.content)
              else void project.writePath(e.path, e.content)
            } else {
              const f = files.files.find((x) => x.name === e.path)
              if (f) files.updateContent(f.id, e.content)
            }
          }
        }}
      />

      <PlotDialog
        open={plotOpen}
        frameName={selectedVar}
        value={selectedValue}
        onClose={() => setPlotOpen(false)}
        onSubmit={(opts) => {
          const df = selectedValue
          if (!isDataFrame(df)) return
          if (opts.type === 'hist') {
            plotApi.hist(df, opts.x, { bins: opts.bins, title: opts.title || undefined })
          } else {
            plotApi.plot(df, {
              x: opts.x,
              y: opts.y,
              type: opts.type,
              title: opts.title || undefined,
            })
          }
          setPlotOpen(false)
          setBottomTab('plots')
        }}
      />
    </div>
  )
}
