import { numericColumnValues, plotSeries } from '../dfAccess'
import { uid, type PlotSpec, type PlotType } from '../types'

export interface PlotOptions {
  x: string
  y: string
  type?: PlotType
  title?: string
}

export interface HistOptions {
  bins?: number
  title?: string
}

export interface PlotApi {
  plot: (data: unknown, opts: PlotOptions) => PlotSpec
  hist: (data: unknown, column: string, binsOrOpts?: number | HistOptions) => PlotSpec
  clear: () => void
  remove: (id: string) => void
  getPlots: () => PlotSpec[]
  subscribe: (fn: () => void) => () => void
}

export function createPlotApi(): PlotApi {
  let plots: PlotSpec[] = []
  const listeners = new Set<() => void>()

  const emit = () => {
    for (const fn of listeners) fn()
  }

  const push = (spec: Omit<PlotSpec, 'id' | 'createdAt'>) => {
    const full: PlotSpec = { ...spec, id: uid('plot'), createdAt: Date.now() }
    plots = [full, ...plots].slice(0, 20)
    emit()
    return full
  }

  return {
    subscribe(fn) {
      listeners.add(fn)
      return () => listeners.delete(fn)
    },
    getPlots: () => plots,
    clear() {
      plots = []
      emit()
    },
    remove(id) {
      plots = plots.filter((p) => p.id !== id)
      emit()
    },
    plot(data, opts) {
      const { x, y } = plotSeries(data, opts.x, opts.y)
      return push({
        title: opts.title ?? `${opts.type ?? 'scatter'}: ${opts.y} ~ ${opts.x}`,
        type: opts.type ?? 'scatter',
        x,
        y,
      })
    },
    hist(data, column, binsOrOpts = 20) {
      const opts: HistOptions =
        typeof binsOrOpts === 'number' ? { bins: binsOrOpts } : (binsOrOpts ?? {})
      const bins = opts.bins ?? 20
      const title = opts.title ?? `hist: ${column}`
      const values = numericColumnValues(data, column)
      if (values.length === 0) {
        return push({ title, type: 'hist', x: [], y: [] })
      }
      let min = values[0]!
      let max = values[0]!
      for (let i = 1; i < values.length; i++) {
        const v = values[i]!
        if (v < min) min = v
        if (v > max) max = v
      }
      const width = max === min ? 1 : (max - min) / bins
      const counts = new Array<number>(bins).fill(0)
      const labels: string[] = []
      for (let i = 0; i < bins; i++) labels.push(`${(min + i * width).toFixed(1)}`)
      for (let i = 0; i < values.length; i++) {
        const v = values[i]!
        let bin = max === min ? 0 : Math.floor((v - min) / width)
        if (bin >= bins) bin = bins - 1
        if (bin < 0) bin = 0
        counts[bin]! += 1
      }
      return push({
        title,
        type: 'hist',
        x: labels,
        y: counts,
      })
    },
  }
}
