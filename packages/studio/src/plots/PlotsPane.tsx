import { useEffect, useRef } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import type { PlotSpec } from '../types'

export interface PlotsPaneProps {
  plots: PlotSpec[]
  onClear: () => void
  onRemove: (id: string) => void
}

export function PlotsPane({ plots, onClear, onRemove }: PlotsPaneProps) {
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="panel-header">
        <span>Plots</span>
        <button className="btn" type="button" onClick={onClear}>
          Clear
        </button>
      </div>
      <div className="plots panel-body">
        {plots.length === 0 ? (
          <p className="muted">
            No plots yet. Try plot(df, {'{'} x: &apos;age&apos;, y: &apos;salary&apos; {'}'}) or use Plot in the DataFrame viewer.
          </p>
        ) : (
          plots.map((p) => <PlotCard key={p.id} spec={p} onRemove={() => onRemove(p.id)} />)
        )}
      </div>
    </div>
  )
}

function PlotCard({ spec, onRemove }: { spec: PlotSpec; onRemove: () => void }) {
  const hostRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<uPlot | null>(null)

  useEffect(() => {
    const el = hostRef.current
    if (!el) return

    const xs = spec.x.map((v, i) => (typeof v === 'number' ? v : i))
    const ys = spec.y.map((v) => Number(v))

    const opts: uPlot.Options = {
      width: Math.max(320, el.clientWidth || 480),
      height: 220,
      title: undefined,
      series: [
        {},
        {
          label: spec.title,
          stroke: '#0f6e56',
          width: 2,
          points: { show: spec.type === 'scatter', size: 6 },
          paths: spec.type === 'bar' || spec.type === 'hist' ? uPlot.paths.bars!({ size: [0.6, 100] }) : undefined,
        },
      ],
      axes: [{ stroke: '#5d6673' }, { stroke: '#5d6673' }],
      scales: {
        x: { time: false },
      },
    }

    const chart = new uPlot(opts, [xs, ys], el)
    chartRef.current = chart
    const ro = new ResizeObserver(() => {
      chart.setSize({ width: Math.max(320, el.clientWidth || 480), height: 220 })
    })
    ro.observe(el)
    return () => {
      ro.disconnect()
      chart.destroy()
      chartRef.current = null
    }
  }, [spec])

  const exportPng = () => {
    const root = hostRef.current
    if (!root) return
    const canvas = root.querySelector('canvas')
    if (!canvas) return
    const url = canvas.toDataURL('image/png')
    const a = document.createElement('a')
    a.href = url
    a.download = `${sanitize(spec.title) || 'plot'}.png`
    a.click()
  }

  return (
    <div className="plot-card">
      <div className="plot-card-header">
        <div className="plot-title">{spec.title}</div>
        <div className="panel-tools">
          <button className="btn" type="button" onClick={exportPng}>
            PNG
          </button>
          <button className="btn btn-ghost" type="button" onClick={onRemove}>
            ×
          </button>
        </div>
      </div>
      <div className="plot-host" ref={hostRef} />
    </div>
  )
}

function sanitize(name: string): string {
  return name.replace(/[^\w.-]+/g, '_').slice(0, 64)
}
