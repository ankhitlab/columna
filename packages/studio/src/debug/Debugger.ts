export type DebugAction = 'continue' | 'stepOver' | 'stepInto' | 'stepOut' | 'stop'

export interface DebugFrame {
  name: string
  line: number
  locals: Record<string, unknown>
}

export interface DebugPauseState {
  line: number
  frames: DebugFrame[]
  locals: Record<string, unknown>
}

export type StepMode = 'none' | 'over' | 'into' | 'out'

export class DebuggerRuntime {
  breakpoints = new Set<number>()
  paused: DebugPauseState | null = null
  running = false
  private stepMode: StepMode = 'none'
  private stack: DebugFrame[] = []
  private resume: ((action: DebugAction) => void) | null = null
  private depthAtStepOver = 0
  private listeners = new Set<() => void>()
  private stopRequested = false

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn)
    return () => this.listeners.delete(fn)
  }

  private emit(): void {
    for (const fn of this.listeners) fn()
  }

  toggleBreakpoint(line: number): void {
    if (this.breakpoints.has(line)) this.breakpoints.delete(line)
    else this.breakpoints.add(line)
    this.emit()
  }

  setBreakpoints(lines: Iterable<number>): void {
    this.breakpoints = new Set(lines)
    this.emit()
  }

  clearBreakpoints(): void {
    this.breakpoints.clear()
    this.emit()
  }

  beginRun(): void {
    this.running = true
    this.stopRequested = false
    this.paused = null
    this.stack = []
    this.stepMode = 'none'
    this.emit()
  }

  endRun(): void {
    this.running = false
    this.paused = null
    this.resume = null
    this.stepMode = 'none'
    this.emit()
  }

  requestStop(): void {
    this.stopRequested = true
    if (this.resume) {
      const r = this.resume
      this.resume = null
      r('stop')
    }
  }

  action(a: DebugAction): void {
    if (a === 'stop') {
      this.requestStop()
      return
    }
    if (!this.resume) return
    if (a === 'stepOver') {
      this.stepMode = 'over'
      this.depthAtStepOver = this.stack.length
    } else if (a === 'stepInto') {
      this.stepMode = 'into'
    } else if (a === 'stepOut') {
      this.stepMode = 'out'
      this.depthAtStepOver = Math.max(0, this.stack.length - 1)
    } else {
      this.stepMode = 'none'
    }
    const r = this.resume
    this.resume = null
    this.paused = null
    this.emit()
    r(a)
  }

  /** Builtin injected into instrumented code */
  api() {
    return {
      line: async (line: number, getLocals: () => Record<string, unknown>) => {
        if (this.stopRequested) throw new Error('Debug stopped')
        const locals = safeLocals(getLocals)
        if (this.stack.length) {
          this.stack[this.stack.length - 1] = {
            ...this.stack[this.stack.length - 1]!,
            line,
            locals,
          }
        }

        const hitBp = this.breakpoints.has(line)
        const stepHit =
          this.stepMode === 'into' ||
          (this.stepMode === 'over' && this.stack.length <= this.depthAtStepOver) ||
          (this.stepMode === 'out' && this.stack.length <= this.depthAtStepOver)

        if (!hitBp && !stepHit) return

        this.stepMode = 'none'
        this.paused = {
          line,
          frames: [...this.stack],
          locals,
        }
        this.emit()

        const action = await new Promise<DebugAction>((resolve) => {
          this.resume = resolve
        })
        if (action === 'stop') throw new Error('Debug stopped')
      },

      push: (name: string, line: number) => {
        this.stack.push({ name, line, locals: {} })
        this.emit()
      },

      pop: () => {
        this.stack.pop()
        this.emit()
      },
    }
  }

  async evalWatch(expr: string, scope: Record<string, unknown>): Promise<unknown> {
    const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor as new (
      ...args: string[]
    ) => (...args: unknown[]) => Promise<unknown>
    const keys = Object.keys(scope)
    const body = `
      const __s = arguments[0];
      ${keys.map((k) => `const ${k} = __s[${JSON.stringify(k)}];`).join('\n')}
      return (${expr});
    `
    const fn = new AsyncFunction(body)
    return fn(scope)
  }
}

function safeLocals(getLocals: () => Record<string, unknown>): Record<string, unknown> {
  try {
    return getLocals() ?? {}
  } catch {
    return {}
  }
}
