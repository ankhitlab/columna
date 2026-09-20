/**
 * Plot-ready series export — pure data contract for SPC / time-series results.
 * No renderer; consumers map `role` to chart layers.
 */
import type { ControlChartResult, CusumResult, EwmaResult, MovingAverageResult } from './spc.js'
import type { ArimaResult, EtsResult } from './timeseries.js'

export type PlotRole = 'data' | 'center' | 'ucl' | 'lcl' | 'forecast' | 'band' | 'companion'

export interface PlotSeries {
  name: string
  x: number[]
  y: number[]
  role: PlotRole
}

export interface PlotSeriesResult {
  series: PlotSeries[]
}

export type PlottableResult =
  | ControlChartResult
  | EwmaResult
  | CusumResult
  | MovingAverageResult
  | EtsResult
  | ArimaResult

function idx(n: number, offset = 0): number[] {
  return Array.from({ length: n }, (_, i) => i + offset)
}

function fromControlChart(r: ControlChartResult, prefix = ''): PlotSeries[] {
  const x = r.points.map((p) => p.index)
  const out: PlotSeries[] = [
    { name: `${prefix}data`, x, y: r.points.map((p) => p.value), role: 'data' },
    { name: `${prefix}center`, x, y: r.points.map((p) => p.center), role: 'center' },
    { name: `${prefix}ucl`, x, y: r.points.map((p) => p.ucl), role: 'ucl' },
    { name: `${prefix}lcl`, x, y: r.points.map((p) => p.lcl), role: 'lcl' },
  ]
  if (r.companion) {
    const c = fromControlChart(r.companion, `${prefix}companion.`)
    for (const s of c) {
      out.push({ ...s, role: s.role === 'data' ? 'companion' : s.role })
    }
  }
  return out
}

function fromEwma(r: EwmaResult): PlotSeries[] {
  const n = r.z.length
  const x = idx(n)
  return [
    { name: 'ewma', x, y: r.z.slice(), role: 'data' },
    { name: 'center', x, y: Array(n).fill(r.center), role: 'center' },
    { name: 'ucl', x, y: r.ucl.slice(), role: 'ucl' },
    { name: 'lcl', x, y: r.lcl.slice(), role: 'lcl' },
  ]
}

function fromCusum(r: CusumResult): PlotSeries[] {
  const n = r.cPlus.length
  const x = idx(n)
  return [
    { name: 'cPlus', x, y: r.cPlus.slice(), role: 'data' },
    { name: 'cMinus', x, y: r.cMinus.slice(), role: 'companion' },
    { name: 'h', x, y: Array(n).fill(r.h), role: 'ucl' },
    { name: 'negH', x, y: Array(n).fill(-r.h), role: 'lcl' },
  ]
}

function fromMa(r: MovingAverageResult): PlotSeries[] {
  const n = r.ma.length
  const x = idx(n)
  return [
    { name: 'ma', x, y: r.ma.slice(), role: 'data' },
    { name: 'center', x, y: Array(n).fill(r.center), role: 'center' },
    { name: 'ucl', x, y: r.ucl.slice(), role: 'ucl' },
    { name: 'lcl', x, y: r.lcl.slice(), role: 'lcl' },
  ]
}

function fromEts(r: EtsResult): PlotSeries[] {
  const n = r.fitted.length
  const xFit = idx(n)
  const out: PlotSeries[] = [
    { name: 'fitted', x: xFit, y: r.fitted.slice(), role: 'data' },
  ]
  if (r.forecast?.length) {
    const h = r.forecast.length
    const xF = idx(h, n)
    out.push({ name: 'forecast', x: xF, y: r.forecast.slice(), role: 'forecast' })
    if (r.forecastLower && r.forecastUpper) {
      out.push({ name: 'forecastLower', x: xF, y: r.forecastLower.slice(), role: 'band' })
      out.push({ name: 'forecastUpper', x: xF, y: r.forecastUpper.slice(), role: 'band' })
    }
  }
  return out
}

function fromArima(r: ArimaResult): PlotSeries[] {
  const n = r.fitted.length
  const xFit = idx(n)
  const out: PlotSeries[] = [
    { name: 'fitted', x: xFit, y: r.fitted.slice(), role: 'data' },
  ]
  if (r.forecast?.length) {
    const h = r.forecast.length
    const xF = idx(h, n)
    out.push({ name: 'forecast', x: xF, y: r.forecast.slice(), role: 'forecast' })
    if (r.forecastLower && r.forecastUpper) {
      out.push({ name: 'forecastLower', x: xF, y: r.forecastLower.slice(), role: 'band' })
      out.push({ name: 'forecastUpper', x: xF, y: r.forecastUpper.slice(), role: 'band' })
    }
  }
  return out
}

function isControlChart(r: PlottableResult): r is ControlChartResult {
  return 'points' in r && Array.isArray((r as ControlChartResult).points)
}

function isEwma(r: PlottableResult): r is EwmaResult {
  return (r as EwmaResult).type === 'ewma'
}

function isCusum(r: PlottableResult): r is CusumResult {
  return (r as CusumResult).type === 'cusum'
}

function isMa(r: PlottableResult): r is MovingAverageResult {
  return (r as MovingAverageResult).type === 'ma'
}

function isEts(r: PlottableResult): r is EtsResult {
  return 'method' in r && typeof (r as EtsResult).method === 'string' && 'fitted' in r && 'sse' in r
}

function isArima(r: PlottableResult): r is ArimaResult {
  return 'order' in r && 'ar' in r && 'fitted' in r
}

/**
 * Convert an SPC or time-series fit result into plot-ready series
 * (`data` / limits / forecast / bands). No rendering.
 */
export function plotSeries(result: PlottableResult): PlotSeriesResult {
  if (isControlChart(result)) return { series: fromControlChart(result) }
  if (isEwma(result)) return { series: fromEwma(result) }
  if (isCusum(result)) return { series: fromCusum(result) }
  if (isMa(result)) return { series: fromMa(result) }
  if (isArima(result)) return { series: fromArima(result) }
  if (isEts(result)) return { series: fromEts(result) }
  throw new TypeError('plotSeries: unsupported result type')
}

const ROLE_STROKE: Record<PlotRole, string> = {
  data: '#1f2937',
  center: '#2563eb',
  ucl: '#dc2626',
  lcl: '#dc2626',
  forecast: '#7c3aed',
  band: '#a78bfa',
  companion: '#059669',
}

function escapeXml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

/**
 * SVG renderer over `plotSeries` / `PlotSeriesResult`.
 * With `interactive: true`: role/data attrs, point tooltips, hover CSS, legend (no Graph Builder product).
 */
export function renderPlotSeries(
  input: PlottableResult | PlotSeriesResult,
  options: { width?: number; height?: number; title?: string; interactive?: boolean } = {},
): string {
  const series =
    'series' in input &&
    Array.isArray((input as PlotSeriesResult).series) &&
    !('points' in input) &&
    !('order' in input) &&
    !('fitted' in input)
      ? (input as PlotSeriesResult).series
      : plotSeries(input as PlottableResult).series
  if (series.length < 1) throw new RangeError('renderPlotSeries: no series')
  const interactive = options.interactive === true
  const width = options.width ?? 640
  const height = options.height ?? 360
  const padL = 48
  const padR = interactive ? 110 : 16
  const padT = options.title ? 36 : 16
  const padB = 36
  let xMin = Infinity
  let xMax = -Infinity
  let yMin = Infinity
  let yMax = -Infinity
  for (const s of series) {
    for (let i = 0; i < s.x.length; i++) {
      const xv = s.x[i]!
      const yv = s.y[i]!
      if (!Number.isFinite(xv) || !Number.isFinite(yv)) continue
      if (xv < xMin) xMin = xv
      if (xv > xMax) xMax = xv
      if (yv < yMin) yMin = yv
      if (yv > yMax) yMax = yv
    }
  }
  if (!(xMax > xMin)) {
    xMin = 0
    xMax = 1
  }
  if (!(yMax > yMin)) {
    yMin -= 1
    yMax += 1
  }
  const yPad = 0.05 * (yMax - yMin) || 1
  yMin -= yPad
  yMax += yPad
  const iw = width - padL - padR
  const ih = height - padT - padB
  const sx = (x: number) => padL + ((x - xMin) / (xMax - xMin)) * iw
  const sy = (y: number) => padT + ((yMax - y) / (yMax - yMin)) * ih
  const parts: string[] = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
  ]
  if (interactive) {
    parts.push(
      `<style><![CDATA[
.plot-layer{opacity:1;transition:opacity .15s}
.plot-layer:hover{opacity:.55}
.plot-pt{opacity:0}
.plot-layer:hover .plot-pt{opacity:1}
]]></style>`,
    )
  }
  parts.push(`<rect width="100%" height="100%" fill="#fff"/>`)
  if (options.title) {
    parts.push(`<text x="${width / 2}" y="22" text-anchor="middle" font-family="sans-serif" font-size="14">${escapeXml(options.title)}</text>`)
  }
  // axes
  parts.push(`<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + ih}" stroke="#9ca3af"/>`)
  parts.push(`<line x1="${padL}" y1="${padT + ih}" x2="${padL + iw}" y2="${padT + ih}" stroke="#9ca3af"/>`)
  parts.push(`<text x="${padL}" y="${height - 8}" font-family="sans-serif" font-size="10">${xMin.toFixed(2)}</text>`)
  parts.push(`<text x="${padL + iw}" y="${height - 8}" text-anchor="end" font-family="sans-serif" font-size="10">${xMax.toFixed(2)}</text>`)
  parts.push(`<text x="8" y="${padT + 8}" font-family="sans-serif" font-size="10">${yMax.toFixed(2)}</text>`)
  parts.push(`<text x="8" y="${padT + ih}" font-family="sans-serif" font-size="10">${yMin.toFixed(2)}</text>`)

  for (const s of series) {
    const pts: string[] = []
    const valid: Array<{ x: number; y: number; px: number; py: number }> = []
    for (let i = 0; i < s.x.length; i++) {
      if (!Number.isFinite(s.x[i]!) || !Number.isFinite(s.y[i]!)) continue
      const px = sx(s.x[i]!)
      const py = sy(s.y[i]!)
      pts.push(`${px.toFixed(2)},${py.toFixed(2)}`)
      valid.push({ x: s.x[i]!, y: s.y[i]!, px, py })
    }
    if (pts.length < 2) continue
    const stroke = ROLE_STROKE[s.role] ?? '#111'
    const dash = s.role === 'ucl' || s.role === 'lcl' || s.role === 'center' ? ' stroke-dasharray="4 3"' : ''
    const widthAttr = s.role === 'data' || s.role === 'forecast' ? 1.75 : 1.25
    const attrs = interactive
      ? ` class="plot-layer" role="${escapeXml(s.role)}" data-role="${escapeXml(s.role)}" data-name="${escapeXml(s.name)}"`
      : ''
    if (interactive) parts.push(`<g${attrs}>`)
    parts.push(
      `<polyline fill="none" stroke="${stroke}" stroke-width="${widthAttr}"${dash} points="${pts.join(' ')}">${
        interactive ? `<title>${escapeXml(`${s.name} (${s.role})`)}</title>` : ''
      }</polyline>`,
    )
    if (interactive) {
      const step = Math.max(1, Math.floor(valid.length / 40))
      for (let i = 0; i < valid.length; i += step) {
        const p = valid[i]!
        parts.push(
          `<circle class="plot-pt" cx="${p.px.toFixed(2)}" cy="${p.py.toFixed(2)}" r="3" fill="${stroke}"><title>${escapeXml(
            `${s.name}: ${p.x.toFixed(3)}, ${p.y.toFixed(3)}`,
          )}</title></circle>`,
        )
      }
      parts.push('</g>')
    }
  }

  if (interactive) {
    parts.push(`<g class="legend" font-family="sans-serif" font-size="10">`)
    let ly = padT + 4
    for (const s of series) {
      const stroke = ROLE_STROKE[s.role] ?? '#111'
      parts.push(
        `<g><title>${escapeXml(`${s.name} (${s.role})`)}</title>`,
        `<line x1="${width - padR + 8}" y1="${ly}" x2="${width - padR + 28}" y2="${ly}" stroke="${stroke}" stroke-width="2"/>`,
        `<text x="${width - padR + 32}" y="${ly + 3}" fill="#374151">${escapeXml(s.name)}</text></g>`,
      )
      ly += 14
      if (ly > padT + ih) break
    }
    parts.push('</g>')
  }
  parts.push('</svg>')
  return parts.join('')
}
