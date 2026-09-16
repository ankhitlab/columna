import { ReplEngine } from './ReplEngine'
import type { PlotApi } from '../plots/plotApi'
import type { ProjectFS } from '../fs/ProjectFS'
import { uid } from '../types'

export interface ConsoleSlot {
  id: string
  name: string
  engine: ReplEngine
}

/** Multiple independent REPL kernels sharing plots / project cwd. */
export class ConsoleManager {
  slots: ConsoleSlot[] = []
  activeId: string | null = null
  private plotApi: PlotApi
  private project: ProjectFS | null = null
  private listeners = new Set<() => void>()

  constructor(plotApi: PlotApi) {
    this.plotApi = plotApi
    this.addConsole()
  }

  setProject(project: ProjectFS | null): void {
    this.project = project
    for (const s of this.slots) s.engine.setProject(project)
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn)
    return () => this.listeners.delete(fn)
  }

  private emit(): void {
    for (const fn of this.listeners) fn()
  }

  get active(): ConsoleSlot {
    return this.slots.find((s) => s.id === this.activeId) ?? this.slots[0]!
  }

  getEngine(): ReplEngine {
    return this.active.engine
  }

  addConsole(): ConsoleSlot {
    const n = this.slots.length + 1
    const engine = new ReplEngine(this.plotApi)
    engine.setProject(this.project)
    const slot: ConsoleSlot = { id: uid('console'), name: `Console ${n}`, engine }
    this.slots = [...this.slots, slot]
    this.activeId = slot.id
    // Forward engine ticks
    engine.subscribe(() => this.emit())
    this.emit()
    return slot
  }

  setActive(id: string): void {
    if (this.slots.some((s) => s.id === id)) {
      this.activeId = id
      this.emit()
    }
  }

  removeConsole(id: string): void {
    if (this.slots.length <= 1) return
    this.slots = this.slots.filter((s) => s.id !== id)
    if (this.activeId === id) this.activeId = this.slots[0]?.id ?? null
    this.emit()
  }
}
