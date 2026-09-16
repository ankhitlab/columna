import { describeValue, isDataFrame } from '../types'

export interface HelpPaneProps {
  selectedName: string | null
  selectedValue: unknown
  mode?: 'guide' | 'inspect'
}

export function HelpPane({ selectedName, selectedValue, mode = 'guide' }: HelpPaneProps) {
  const showInspect = mode === 'inspect' && selectedName

  return (
    <div className="panel" style={{ border: 'none', height: '100%' }}>
      <div className="panel-header">
        <span>Help</span>
        {showInspect && <span className="muted">{selectedName}</span>}
      </div>
      <div className="panel-body help-body">
        {showInspect ? (
          <ObjectInspector name={selectedName!} value={selectedValue} />
        ) : (
          <StudioGuide />
        )}
      </div>
    </div>
  )
}

function StudioGuide() {
  return (
    <div className="help-guide">
      <h3>Columna Studio</h3>
      <p>Spyder-like IDE for the columna DataFrame library (browser REPL).</p>
      <h4>Run</h4>
      <ul>
        <li>
          <kbd>Ctrl+Enter</kbd> — run selection or current cell
        </li>
        <li>
          <kbd>Shift+Enter</kbd> — run cell and advance
        </li>
        <li>
          Cells start with <code>// %%</code> or <code># %%</code>
        </li>
      </ul>
      <h4>API</h4>
      <ul>
        <li>
          <code>DataFrame</code>, <code>col</code>, <code>lit</code>, <code>init</code>
        </li>
        <li>
          <code>plot(df, {'{'} x, y, type {'}'})</code> / <code>hist(df, column)</code>
        </li>
        <li>
          <code>help()</code> / <code>clear()</code>
        </li>
      </ul>
      <h4>Panes</h4>
      <ul>
        <li>Files / Outline — session scripts and symbols</li>
        <li>Variable Explorer — inspect, edit primitives, clear namespace</li>
        <li>Console / History / Plots — REPL, command log, charts</li>
      </ul>
      <p className="muted">Double-click a variable to inspect it here.</p>
    </div>
  )
}

function ObjectInspector({ name, value }: { name: string; value: unknown }) {
  const meta = describeValue(value)
  let dump: string
  try {
    if (isDataFrame(value)) {
      const [r, c] = value.shape
      dump = [
        `DataFrame ${name}`,
        `shape: ${r} × ${c}`,
        `columns: ${value.columns.join(', ')}`,
        `dtypes: ${JSON.stringify(value.dtypes, null, 2)}`,
      ].join('\n')
    } else if (typeof value === 'function') {
      dump = value.toString()
    } else {
      dump = JSON.stringify(value, null, 2) ?? String(value)
    }
  } catch {
    dump = String(value)
  }

  return (
    <div className="help-inspect">
      <div className="df-meta">
        <span className="chip">{name}</span>
        <span className="chip">{meta.type}</span>
        <span className="chip">{meta.preview}</span>
      </div>
      <pre className="help-dump">{dump}</pre>
    </div>
  )
}
