/**
 * Notebook-friendly summaries for common @columna/advanced result objects.
 */
import type { AnovaResult, TTestResult } from './tests.js'
import type { OlsResult } from './regression.js'

export type ReportFormat = 'markdown' | 'html'

type Formattable = TTestResult | AnovaResult | OlsResult | { test?: string; method?: string }

function fmt(x: number, digits = 4): string {
  if (!Number.isFinite(x)) return String(x)
  const s = x.toPrecision(digits + 1)
  if (s.includes('e') || s.includes('E')) return s
  const n = Number(s)
  if (Math.abs(n) >= 1e6 || (Math.abs(n) > 0 && Math.abs(n) < 1e-4)) return n.toExponential(3)
  return String(Number(n.toFixed(digits)))
}

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

function mdTable(headers: string[], rows: string[][]): string {
  const line = (cells: string[]) => `| ${cells.join(' | ')} |`
  return [line(headers), line(headers.map(() => '---')), ...rows.map((r) => line(r))].join('\n')
}

function htmlTable(headers: string[], rows: string[][]): string {
  const th = headers.map((h) => `<th>${escHtml(h)}</th>`).join('')
  const body = rows.map((r) => `<tr>${r.map((c) => `<td>${escHtml(c)}</td>`).join('')}</tr>`).join('')
  return `<table><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table>`
}

function formatTTest(r: TTestResult, format: ReportFormat): string {
  const title = r.test
  const bullets = [
    `**Estimate:** ${fmt(r.estimate)} (SE ${fmt(r.se)})`,
    `**t:** ${fmt(r.statistic)} (df ${fmt(r.df, 2)})`,
    `**p-value:** ${fmt(r.pValue)} (${r.alternative})`,
    `**${Math.round(r.confidence * 100)}% CI:** [${fmt(r.ci[0])}, ${fmt(r.ci[1])}]`,
    `**n:** ${r.n}`,
  ]
  const sampleRows = r.samples.map((s) => [String(s.n), fmt(s.mean), fmt(s.sd)])
  if (format === 'html') {
    return `<h2>${escHtml(title)}</h2><ul>${bullets.map((b) => `<li>${escHtml(b.replace(/\*\*/g, ''))}</li>`).join('')}</ul>${htmlTable(['n', 'Mean', 'SD'], sampleRows)}`
  }
  return `## ${title}\n\n${bullets.map((b) => `- ${b}`).join('\n')}\n\n${mdTable(['n', 'Mean', 'SD'], sampleRows)}`
}

function formatAnova(r: AnovaResult, format: ReportFormat): string {
  const summary = [
    `**F(${r.dfBetween}, ${r.dfWithin}):** ${fmt(r.statistic)}`,
    `**p-value:** ${fmt(r.pValue)}`,
    `**η²:** ${fmt(r.etaSquared)}`,
    `**SS between / within:** ${fmt(r.ssBetween)} / ${fmt(r.ssWithin)}`,
  ]
  const groupRows = r.groups.map((g) => [g.name, String(g.n), fmt(g.mean), fmt(g.sd)])
  if (format === 'html') {
    return `<h2>${escHtml(r.test)}</h2><ul>${summary.map((b) => `<li>${escHtml(b.replace(/\*\*/g, ''))}</li>`).join('')}</ul>${htmlTable(['Group', 'n', 'Mean', 'SD'], groupRows)}`
  }
  return `## ${r.test}\n\n${summary.map((b) => `- ${b}`).join('\n')}\n\n${mdTable(['Group', 'n', 'Mean', 'SD'], groupRows)}`
}

function formatOls(r: OlsResult, format: ReportFormat): string {
  const summary = [
    `**S:** ${fmt(r.s)}`,
    `**R²:** ${fmt(r.r2)} (adj ${fmt(r.r2adj)})`,
    `**n:** ${r.n}, **p:** ${r.p}`,
  ]
  const coefRows = r.coefficients.map((c) => [c.name, fmt(c.coef), fmt(c.se), fmt(c.t), fmt(c.pValue)])
  const anovaRows = [
    ['Regression', String(r.anova.regression.df), fmt(r.anova.regression.ss), fmt(r.anova.regression.ms ?? NaN), fmt(r.anova.regression.f ?? NaN), fmt(r.anova.regression.pValue ?? NaN)],
    ['Error', String(r.anova.error.df), fmt(r.anova.error.ss), fmt(r.anova.error.ms)],
    ['Total', String(r.anova.total.df), fmt(r.anova.total.ss), '—', '—', '—'],
  ]
  if (format === 'html') {
    return `<h2>Regression</h2><ul>${summary.map((b) => `<li>${escHtml(b.replace(/\*\*/g, ''))}</li>`).join('')}</ul>${htmlTable(['Term', 'Coef', 'SE', 't', 'p'], coefRows)}${htmlTable(['Source', 'df', 'SS', 'MS', 'F', 'p'], anovaRows)}`
  }
  return `## Regression\n\n${summary.map((b) => `- ${b}`).join('\n')}\n\n### Coefficients\n\n${mdTable(['Term', 'Coef', 'SE', 't', 'p'], coefRows)}\n\n### ANOVA\n\n${mdTable(['Source', 'df', 'SS', 'MS', 'F', 'p'], anovaRows)}`
}

/**
 * Render a test / model result as Markdown or simple HTML tables.
 */
export function formatReport(result: Formattable, format: ReportFormat = 'markdown'): string {
  const tag = 'test' in result && result.test ? result.test : 'method' in result && result.method ? result.method : ''
  if (tag === 'regression' || ('coefficients' in result && Array.isArray((result as OlsResult).coefficients))) {
    return formatOls(result as OlsResult, format)
  }
  if (tag === 'one-way ANOVA') return formatAnova(result as AnovaResult, format)
  if (typeof tag === 'string' && tag.includes('t')) return formatTTest(result as TTestResult, format)
  throw new RangeError(`formatReport: unsupported result type (${String(tag) || 'unknown'})`)
}
