export type LeftTab = 'project' | 'files' | 'outline'
export type BottomTab = 'console' | 'history' | 'plots' | 'debug' | 'find' | 'analysis'
export type RightTab = 'vars' | 'help'

export interface StudioLayout {
  editorRatio: number
  leftTab: LeftTab
  bottomTab: BottomTab
  rightTab: RightTab
}

const KEY = 'columna-studio-layout'
const HISTORY_KEY = 'columna-studio-cmd-history'

const DEFAULTS: StudioLayout = {
  editorRatio: 0.58,
  leftTab: 'files',
  bottomTab: 'console',
  rightTab: 'vars',
}

const LEFT: LeftTab[] = ['project', 'files', 'outline']
const BOTTOM: BottomTab[] = ['console', 'history', 'plots', 'debug', 'find', 'analysis']

export function loadLayout(): StudioLayout {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return { ...DEFAULTS }
    const parsed = JSON.parse(raw) as Partial<StudioLayout>
    return {
      editorRatio: clampRatio(parsed.editorRatio ?? DEFAULTS.editorRatio),
      leftTab: LEFT.includes(parsed.leftTab as LeftTab) ? (parsed.leftTab as LeftTab) : 'files',
      bottomTab: BOTTOM.includes(parsed.bottomTab as BottomTab)
        ? (parsed.bottomTab as BottomTab)
        : 'console',
      rightTab: parsed.rightTab === 'help' ? 'help' : 'vars',
    }
  } catch {
    return { ...DEFAULTS }
  }
}

export function saveLayout(layout: StudioLayout): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(layout))
  } catch {
    /* ignore quota */
  }
}

export function loadCommandHistory(limit = 200): string[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as unknown
    if (!Array.isArray(parsed)) return []
    return parsed.filter((x): x is string => typeof x === 'string').slice(0, limit)
  } catch {
    return []
  }
}

export function saveCommandHistory(commands: string[], limit = 200): void {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(commands.slice(0, limit)))
  } catch {
    /* ignore */
  }
}

function clampRatio(n: number): number {
  if (!Number.isFinite(n)) return DEFAULTS.editorRatio
  return Math.min(0.8, Math.max(0.25, n))
}
