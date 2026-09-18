import type { PlanNode } from './types.js'

/**
 * Cooperative cancellation and deadlines for plan execution.
 *
 * A plan runs as synchronous kernels; nothing can interrupt one kernel half-way. What the runtime can do is
 * check between operators — and, so that an `AbortSignal` from a click handler or a timer actually has a
 * chance to fire, yield to the event loop between operators when a guard is present. `collect()` without a
 * guard runs exactly as before (no yields, fused fast paths untouched).
 */
export interface ExecGuard {
  readonly signal?: AbortSignal
  /** Absolute `performance.now()` deadline. */
  readonly deadline?: number
  readonly timeoutMs?: number
  readonly t0: number
}

export class ExecutionAbortedError extends Error {
  constructor(
    /** `'signal'` — the AbortSignal fired; `'timeout'` — `timeoutMs` elapsed. */
    readonly reason: 'signal' | 'timeout',
    /** Operator at whose boundary the abort was observed. */
    readonly node: string,
    readonly elapsedMs: number,
    readonly cause?: unknown,
  ) {
    super(
      reason === 'timeout'
        ? `execution exceeded timeoutMs at "${node}" after ${elapsedMs.toFixed(0)} ms`
        : `execution aborted at "${node}" after ${elapsedMs.toFixed(0)} ms${cause !== undefined ? `: ${describeCause(cause)}` : ''}`,
    )
    this.name = 'ExecutionAbortedError'
  }
}

function describeCause(c: unknown): string {
  if (c instanceof Error) return c.message
  if (typeof c === 'string') return c
  try {
    return JSON.stringify(c)
  } catch {
    return String(c)
  }
}

export const nowMs = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now())

export function makeGuard(opts: { signal?: AbortSignal; timeoutMs?: number } | undefined): ExecGuard | undefined {
  if (!opts || (!opts.signal && opts.timeoutMs === undefined)) return undefined
  if (opts.timeoutMs !== undefined && !(opts.timeoutMs > 0)) throw new RangeError(`timeoutMs must be > 0 (got ${opts.timeoutMs})`)
  const t0 = nowMs()
  return { signal: opts.signal, timeoutMs: opts.timeoutMs, deadline: opts.timeoutMs !== undefined ? t0 + opts.timeoutMs : undefined, t0 }
}

/** Throws `ExecutionAbortedError` when the signal fired or the deadline passed. Cheap: two comparisons. */
export function checkGuard(guard: ExecGuard | undefined, node: string): void {
  if (!guard) return
  if (guard.signal?.aborted) throw new ExecutionAbortedError('signal', node, nowMs() - guard.t0, guard.signal.reason)
  if (guard.deadline !== undefined && nowMs() > guard.deadline) throw new ExecutionAbortedError('timeout', node, nowMs() - guard.t0)
}

/** Let timers, aborts and UI events run: `scheduler.yield()` → `setImmediate` → `MessageChannel`. */
export function yieldToEventLoop(): Promise<void> {
  const g = globalThis as { scheduler?: { yield?: () => Promise<void> }; setImmediate?: (cb: () => void) => unknown }
  if (g.scheduler && typeof g.scheduler.yield === 'function') return g.scheduler.yield()
  if (typeof g.setImmediate === 'function') return new Promise((r) => g.setImmediate!(() => r()))
  if (typeof MessageChannel !== 'undefined') {
    return new Promise((r) => {
      const ch = new MessageChannel()
      ch.port1.onmessage = () => {
        ch.port1.close()
        r()
      }
      ch.port2.postMessage(null)
    })
  }
  return new Promise((r) => setTimeout(r, 0))
}

export function planChildren(n: PlanNode): PlanNode[] {
  if (n.type === 'scan') return []
  if ('frames' in n) return n.frames
  if ('left' in n) return [n.left, n.right]
  return [n.input]
}

export function withPlanChildren(n: PlanNode, kids: PlanNode[]): PlanNode {
  if (n.type === 'scan') return n
  if ('frames' in n) return { ...n, frames: kids }
  if ('left' in n) return { ...n, left: kids[0]!, right: kids[1]! }
  return { ...n, input: kids[0]! }
}

/**
 * The smallest subtree the CPU engine executes as one fused kernel: `project → filter` and `project → join`
 * are pruned/gathered together, so cutting between them would change the kernel. Returns the unit's leaf
 * children and a function that rebuilds the unit over materialized (scan) children.
 */
export function executionUnit(plan: PlanNode): { children: PlanNode[]; rebuild: (kids: PlanNode[]) => PlanNode } {
  if (plan.type === 'project' && (plan.input.type === 'filter' || plan.input.type === 'join')) {
    const inner = plan.input
    return { children: planChildren(inner), rebuild: (kids) => ({ ...plan, input: withPlanChildren(inner, kids) }) }
  }
  return { children: planChildren(plan), rebuild: (kids) => withPlanChildren(plan, kids) }
}
