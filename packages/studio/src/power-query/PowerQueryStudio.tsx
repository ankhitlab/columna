import { useMemo, useState } from 'react'
import { aggExpr, col, DataFrame, lit } from 'columna'

type StepKind = 'filter' | 'select' | 'sort' | 'rename' | 'limit' | 'groupBy' | 'addColumn' | 'replace' | 'split' | 'pivot' | 'join' | 'fillMissing' | 'changeType'
type FilterOp = 'eq' | 'neq' | 'gt' | 'gte' | 'lt' | 'lte' | 'contains' | 'startsWith' | 'endsWith'
type AggregateFn = 'sum' | 'mean' | 'min' | 'max' | 'count'
type AddOp = 'copy' | 'add' | 'sub' | 'mul' | 'div'
type FillValueType = 'text' | 'number'
type DataType = 'text' | 'integer' | 'decimal' | 'boolean' | 'datetime'
type JoinType = 'left' | 'inner' | 'outer' | 'cross' | 'semi' | 'anti'

interface BaseStep {
  id: string
  kind: StepKind
}

interface FilterStep extends BaseStep {
  kind: 'filter'
  column: string
  operator: FilterOp
  value: string
}

interface SelectStep extends BaseStep {
  kind: 'select'
  columns: string[]
}

interface SortStep extends BaseStep {
  kind: 'sort'
  column: string
  direction: 'asc' | 'desc'
}

interface RenameStep extends BaseStep {
  kind: 'rename'
  oldName: string
  newName: string
}

interface LimitStep extends BaseStep {
  kind: 'limit'
  rows: number
}

interface GroupByStep extends BaseStep {
  kind: 'groupBy'
  keys: string[]
  aggColumn: string
  aggFn: AggregateFn
  outputColumn: string
}

interface AddColumnStep extends BaseStep {
  kind: 'addColumn'
  sourceColumn: string
  outputColumn: string
  op: AddOp
  value: string
}

interface ReplaceStep extends BaseStep {
  kind: 'replace'
  column: string
  search: string
  replacement: string
}

interface SplitStep extends BaseStep {
  kind: 'split'
  column: string
  delimiter: string
  leftColumn: string
  rightColumn: string
}

interface PivotStep extends BaseStep {
  kind: 'pivot'
  index: string
  columns: string
  values: string
  aggFn: AggregateFn
}

interface JoinStep extends BaseStep {
  kind: 'join'
  leftOn: string
  rightOn: string
  how: JoinType
}

interface FillMissingStep extends BaseStep {
  kind: 'fillMissing'
  column: string
  value: string
}

interface ChangeTypeStep extends BaseStep {
  kind: 'changeType'
  column: string
  targetType: DataType
}

type TransformStep =
  | FilterStep
  | SelectStep
  | SortStep
  | RenameStep
  | LimitStep
  | GroupByStep
  | AddColumnStep
  | ReplaceStep
  | SplitStep
  | PivotStep
  | JoinStep
  | FillMissingStep
  | ChangeTypeStep

interface StepPreview {
  title: string
  rowCount: number
  columns: string[]
  head: Array<Record<string, unknown>>
}

interface SourceValidation {
  errors: string[]
  warnings: string[]
}

interface StepValidation {
  errors: string[]
  warnings: string[]
}

interface ValidationResult {
  source: SourceValidation
  hasErrors: boolean
  byStep: Record<string, StepValidation>
}

const SAMPLE_CSV = `Region,Product,Sales,Date
West,Lamp,1820,2024-01-01
West,Keyboard,1140,2024-01-02
North,Monitor,930,2024-01-03
East,Keyboard,760,2024-01-04
South,Lamp,410,2024-01-05
West,Mouse,620,2024-01-06
North,Monitor,1330,2024-01-07
South,Speaker,840,2024-01-08
East,Monitor,980,2024-01-09
West,Tablet,2200,2024-01-10`

const JOIN_SOURCE_CSV = `RegionCode,Manager,Segment
West,Williams,Enterprise
North,Petrov,Retail
East,Kent,SMB
South,Davis,Retail
West,Williams,Government`

const PROJECT_TREE: { title: string; items: string[] }[] = [
  { title: 'Sources', items: ['sales.csv', 'regions.csv'] },
  { title: 'Queries', items: ['Clean Sales', 'Monthly Sales'] },
  { title: 'Datasets', items: ['Sales Clean'] },
  { title: 'Analyses', items: ['Descriptive', 'Regression'] },
  { title: 'Models', items: ['Forecast (stub)'] },
]

const FILTER_OPERATORS: FilterOp[] = ['eq', 'neq', 'gt', 'gte', 'lt', 'lte', 'contains', 'startsWith', 'endsWith']
const AGG_FUNCTIONS: AggregateFn[] = ['sum', 'mean', 'min', 'max', 'count']
const ADD_OPERATORS: AddOp[] = ['copy', 'add', 'sub', 'mul', 'div']
const DATA_TYPES: DataType[] = ['text', 'integer', 'decimal', 'boolean', 'datetime']
const JOIN_TYPES: JoinType[] = ['left', 'inner', 'outer', 'cross', 'semi', 'anti']
const COLUMN_QUICK_ACTIONS: Array<{
  kind: 'filter' | 'replace' | 'split' | 'groupBy' | 'pivot' | 'join' | 'fillMissing' | 'changeType'
  label: string
}> = [
  { kind: 'filter', label: 'Filter' },
  { kind: 'replace', label: 'Replace' },
  { kind: 'split', label: 'Split' },
  { kind: 'groupBy', label: 'Group' },
  { kind: 'pivot', label: 'Pivot' },
  { kind: 'join', label: 'Join' },
  { kind: 'fillMissing', label: 'Fill missing' },
  { kind: 'changeType', label: 'Change type' },
]

type QuickActionKind = (typeof COLUMN_QUICK_ACTIONS)[number]['kind']

let stepId = 0
function nextId(prefix: string): string {
  stepId += 1
  return `${prefix}_${Date.now().toString(36)}_${stepId}`
}

function splitCSVLine(line: string): string[] {
  const out: string[] = []
  let token = ''
  let quoted = false

  for (let i = 0; i < line.length; i++) {
    const ch = line[i]!
    if (ch === '"') {
      quoted = !quoted
      continue
    }
    if (ch === ',' && !quoted) {
      out.push(token.trim())
      token = ''
      continue
    }
    token += ch
  }
  out.push(token.trim())
  return out.map((v) => v.replace(/^"|"$/g, ''))
}

function parseColumns(csv: string): string[] {
  const first = csv
    .split(/\r?\n/)
    .find((line) => line.trim())
  if (!first) return []
  return splitCSVLine(first)
}

function parseRows(csv: string): string[][] {
  const lines = csv.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
  if (lines.length <= 1) return []
  return lines.slice(1).map(splitCSVLine)
}

function parseRecords(csv: string): Array<Record<string, string>> {
  const columns = parseColumns(csv)
  if (!columns.length) return []

  return parseRows(csv).map((row) => {
    const record: Record<string, string> = {}
    for (let i = 0; i < columns.length; i++) {
      record[columns[i]!] = row[i] ?? ''
    }
    return record
  })
}

function validateSourceCsv(csv: string, rawColumns: string[], rows: string[][]): SourceValidation {
  const errors: string[] = []
  const warnings: string[] = []
  const trimmed = csv.trim()

  if (!trimmed) {
    errors.push('Введите CSV-данные в поле источника.')
    return { errors, warnings }
  }

  if (!rawColumns.length) {
    errors.push('В CSV должна быть строка заголовков.')
    return { errors, warnings }
  }

  if (new Set(rawColumns).size !== rawColumns.length) {
    const seen = new Set<string>()
    const duplicates = new Set<string>()
    for (const col of rawColumns) {
      if (seen.has(col)) duplicates.add(col)
      seen.add(col)
    }
    errors.push(`Дублируются колонки в исходнике: ${[...duplicates].join(', ')}`)
  }

  if (!rows.length) {
    warnings.push('В исходнике пока только заголовки, нет строк данных.')
  }

  for (let i = 0; i < rows.length; i++) {
    if (rows[i]!.length !== rawColumns.length) {
      warnings.push(`Строка ${i + 2}: количество колонок ${rows[i]!.length} вместо ${rawColumns.length}.`)
    }
  }

  return { errors, warnings }
}

function parseStepValue(raw: string): string | number | boolean {
  const text = raw.trim()
  if (!text) return ''
  if (/^(true|false)$/i.test(text)) return text.toLowerCase() === 'true'
  const asNumber = Number(text)
  if (Number.isFinite(asNumber) && text !== '') return asNumber
  if ((text[0] === '"' && text.at(-1) === '"') || (text[0] === "'" && text.at(-1) === "'")) {
    return text.slice(1, -1)
  }
  return text
}

function parseNumeric(value: string): number | null {
  const parsed = parseStepValue(value)
  return typeof parsed === 'number' && Number.isFinite(parsed) ? parsed : null
}

function parseBoolean(value: string): boolean | null {
  if (/^(true|false)$/i.test(value.trim())) return value.trim().toLowerCase() === 'true'
  return null
}

function parseCellValueAsScalar(value: string): number | boolean | string {
  const boolValue = parseBoolean(value)
  if (boolValue !== null) return boolValue
  const numeric = parseNumeric(value)
  if (numeric !== null) return numeric
  return value
}

function castTypeFromTarget(targetType: DataType): 'utf8' | 'f64' | 'i32' | 'bool' | 'datetime' {
  if (targetType === 'text') return 'utf8'
  if (targetType === 'integer') return 'i32'
  if (targetType === 'decimal') return 'f64'
  if (targetType === 'boolean') return 'bool'
  return 'datetime'
}

function normalizeColumns(value: string): string[] {
  return value
    .split(',')
    .map((v) => v.trim())
    .filter(Boolean)
}

function unique(values: string[]): string[] {
  return [...new Set(values)]
}

function createStep(kind: StepKind, columns: string[]): TransformStep {
  switch (kind) {
    case 'filter':
      return {
        id: nextId('filt'),
        kind,
        column: columns[0] ?? '',
        operator: 'eq',
        value: '',
      }
    case 'select':
      return {
        id: nextId('sel'),
        kind,
        columns: columns.slice(0, 2),
      }
    case 'sort':
      return {
        id: nextId('sort'),
        kind,
        column: columns[0] ?? '',
        direction: 'asc',
      }
    case 'rename':
      return {
        id: nextId('ren'),
        kind,
        oldName: columns[0] ?? '',
        newName: '',
      }
    case 'limit':
      return {
        id: nextId('limit'),
        kind,
        rows: 5,
      }
    case 'groupBy':
      return {
        id: nextId('group'),
        kind,
        keys: columns.slice(0, 1),
        aggColumn: columns[0] ?? '',
        aggFn: 'sum',
        outputColumn: 'SalesAgg',
      }
    case 'addColumn':
      return {
        id: nextId('addc'),
        kind,
        sourceColumn: columns[0] ?? '',
        outputColumn: columns[1] ?? 'NewColumn',
        op: 'copy',
        value: '1',
      }
    case 'replace':
      return {
        id: nextId('rep'),
        kind,
        column: columns[0] ?? '',
        search: '',
        replacement: '',
      }
    case 'split':
      return {
        id: nextId('split'),
        kind,
        column: columns[0] ?? '',
        delimiter: ',',
        leftColumn: `${columns[0] ?? 'Column'}_left`,
        rightColumn: `${columns[0] ?? 'Column'}_right`,
      }
    case 'pivot':
      return {
        id: nextId('pivot'),
        kind,
        index: columns[0] ?? '',
        columns: columns[1] ?? '',
        values: columns[2] ?? '',
        aggFn: 'sum',
      }
    case 'join':
      return {
        id: nextId('join'),
        kind,
        leftOn: columns[0] ?? '',
        rightOn: columns[0] ?? '',
        how: 'left',
      }
    case 'fillMissing':
      return {
        id: nextId('fill'),
        kind,
        column: columns[0] ?? '',
        value: '0',
      }
    case 'changeType':
      return {
        id: nextId('type'),
        kind,
        column: columns[0] ?? '',
        targetType: 'text',
      }
    default: {
      return {
        id: nextId('filt'),
        kind: 'filter',
        column: columns[0] ?? '',
        operator: 'eq',
        value: '',
      }
    }
  }
}

function filterOperatorLabel(op: FilterOp): string {
  if (op === 'eq') return '='
  if (op === 'neq') return '!='
  if (op === 'gt') return '>'
  if (op === 'gte') return '>='
  if (op === 'lt') return '<'
  if (op === 'lte') return '<='
  if (op === 'startsWith') return 'startsWith'
  if (op === 'endsWith') return 'endsWith'
  return 'contains'
}

function stepLabel(step: TransformStep): string {
  if (step.kind === 'filter') return 'Фильтр'
  if (step.kind === 'select') return 'Выбор колонок'
  if (step.kind === 'sort') return 'Сортировка'
  if (step.kind === 'rename') return 'Переименование'
  if (step.kind === 'limit') return 'Ограничение строк'
  if (step.kind === 'groupBy') return 'Агрегация по группам'
  if (step.kind === 'replace') return 'Замена значений'
  if (step.kind === 'split') return 'Разделение колонки'
  if (step.kind === 'pivot') return 'Свод'
  if (step.kind === 'join') return 'Объединение'
  if (step.kind === 'fillMissing') return 'Заполнить пропуски'
  if (step.kind === 'changeType') return 'Смена типа'
  return 'Новая колонка'
}

function buildFilterExpr(step: FilterStep) {
  const left = col(step.column)
  const raw = parseStepValue(step.value)
  const right = typeof raw === 'string' ? lit(raw) : raw

  if (step.operator === 'eq') return left.eq(right)
  if (step.operator === 'neq') return left.neq(right)
  if (step.operator === 'gt') return left.gt(raw as number | string | boolean)
  if (step.operator === 'gte') return left.gte(raw as number | string | boolean)
  if (step.operator === 'lt') return left.lt(raw as number | string | boolean)
  if (step.operator === 'lte') return left.lte(raw as number | string | boolean)
  if (step.operator === 'contains') return left.str.contains(String(raw))
  if (step.operator === 'startsWith') return left.str.startsWith(String(raw))
  return left.str.endsWith(String(raw))
}

function buildFilterExprCode(step: FilterStep): string {
  const raw = parseStepValue(step.value)
  const renderLiteral = () => {
    if (typeof raw === 'string') return JSON.stringify(raw)
    return String(raw)
  }

  if (step.operator === 'contains') {
    return `col(${JSON.stringify(step.column)}).str.contains(${renderLiteral()})`
  }
  if (step.operator === 'startsWith') {
    return `col(${JSON.stringify(step.column)}).str.startsWith(${renderLiteral()})`
  }
  if (step.operator === 'endsWith') {
    return `col(${JSON.stringify(step.column)}).str.endsWith(${renderLiteral()})`
  }

  if (['eq', 'neq', 'gt', 'gte', 'lt', 'lte'].includes(step.operator)) {
    const op = step.operator === 'eq' ? 'eq' : step.operator === 'neq' ? 'neq' : step.operator
    const column = `col(${JSON.stringify(step.column)})`
    return `${column}.${op}(${renderLiteral()})`
  }

  return `col(${JSON.stringify(step.column)}).str.contains(${renderLiteral()})`
}

function replaceExpr(step: ReplaceStep) {
  return col(step.column).str.replace(step.search, step.replacement)
}

function replaceExprCode(step: ReplaceStep): string {
  return `col(${JSON.stringify(step.column)}).str.replace(${JSON.stringify(step.search)}, ${JSON.stringify(step.replacement)})`
}

function changeTypeExpr(step: ChangeTypeStep) {
  return col(step.column).cast(castTypeFromTarget(step.targetType))
}

function changeTypeExprCode(step: ChangeTypeStep): string {
  return `col(${JSON.stringify(step.column)}).cast(${JSON.stringify(castTypeFromTarget(step.targetType))})`
}

function pivotExprCode(step: PivotStep): string {
  return `\n    index: ${JSON.stringify(step.index)},\n    columns: ${JSON.stringify(step.columns)},\n    values: ${JSON.stringify(step.values)},\n    agg: ${JSON.stringify(step.aggFn)}`
}

function joinSourceColumns(): string[] {
  return unique(parseColumns(JOIN_SOURCE_CSV))
}

function addColumnExpr(step: AddColumnStep) {
  const src = col(step.sourceColumn)
  if (step.op === 'copy') return src
  const value = parseStepValue(step.value)
  const number = typeof value === 'number' ? value : 0
  if (step.op === 'add') return src.add(number)
  if (step.op === 'sub') return src.sub(number)
  if (step.op === 'mul') return src.mul(number)
  return src.div(number)
}

function addColumnExprCode(step: AddColumnStep): string {
  if (step.op === 'copy') {
    return `col(${JSON.stringify(step.sourceColumn)})`
  }
  const value = parseNumeric(step.value)
  return `col(${JSON.stringify(step.sourceColumn)}).${step.op}(${value === null ? '0' : value})`
}

function stepToQueryMethod(step: TransformStep): string {
  if (step.kind === 'filter') {
    return `.filter(${buildFilterExprCode(step)})`
  }
  if (step.kind === 'select') {
    if (!step.columns.length) return ''
    return `.select(${step.columns.map((column) => JSON.stringify(column)).join(', ')})`
  }
  if (step.kind === 'sort') {
    const order = step.direction === 'asc' ? 'asc' : 'desc'
    return `.sort(col(${JSON.stringify(step.column)}).${order}())`
  }
  if (step.kind === 'rename') {
    return `.rename({${JSON.stringify(step.oldName)}: ${JSON.stringify(step.newName)})`
  }
  if (step.kind === 'limit') {
    return `.limit(${Math.max(1, Math.floor(step.rows))})`
  }
  if (step.kind === 'groupBy') {
    return `.groupBy(${step.keys.map((k) => JSON.stringify(k)).join(', ')}).agg({${JSON.stringify(step.outputColumn)}: aggExpr(${JSON.stringify(step.aggFn)}, ${JSON.stringify(step.aggColumn)})})`
  }
  if (step.kind === 'addColumn') {
    return `.withColumn(${JSON.stringify(step.outputColumn)}, ${addColumnExprCode(step)})`
  }
  if (step.kind === 'replace') {
    return `.withColumn(${JSON.stringify(step.column)}, ${replaceExprCode(step)})`
  }
  if (step.kind === 'changeType') {
    return `.withColumn(${JSON.stringify(step.column)}, ${changeTypeExprCode(step)})`
  }
  if (step.kind === 'pivot') {
    return `.pivot({ ${pivotExprCode(step)} })`
  }
  if (step.kind === 'join') {
    return `.join(DataFrame.fromCSV(JOIN_SOURCE_CSV), { leftOn: ${JSON.stringify(step.leftOn)}, rightOn: ${JSON.stringify(step.rightOn)}, how: ${JSON.stringify(step.how)} })`
  }
  if (step.kind === 'fillMissing') {
    return `.fillNull(${JSON.stringify(parseCellValueAsScalar(step.value))}, [${JSON.stringify(step.column)}])`
  }
  return ''
}

function stepToRuntimeCode(steps: TransformStep[]): string {
  if (!steps.length) {
    return 'DataFrame.fromCSV(sourceCsv).collect()'
  }
  const runtimeSteps = steps.filter((step) => step.kind !== 'split')
  const manualSteps = steps.filter((step) => step.kind === 'split')
  const runtimeChain = runtimeSteps.map(stepToQueryMethod).join('')
  const manual = manualSteps
    .map((step) => ` // ручной шаг: split(${JSON.stringify(step.column)}, ${JSON.stringify(step.delimiter)}) -> ${JSON.stringify(step.leftColumn)}, ${JSON.stringify(step.rightColumn)}`)
    .join('\n')
  return `DataFrame.fromCSV(sourceCsv)${runtimeChain}.collect();\n${manual ? `// ${manual}` : ''}`
}

function renderPipelineCode(csvText: string, steps: TransformStep[]): string {
  return `const sourceCsv = ${JSON.stringify(csvText)}\nconst output = await ${stepToRuntimeCode(steps)}`
}

function findNumericColumns(columns: string[], records: Array<Record<string, string>>): string[] {
  const candidates = new Set<string>()

  for (const column of columns) {
    const values = records
      .map((row) => parseStepValue(row[column] ?? '').toString())
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value))

    if (records.length > 0 && values.length >= Math.min(records.length, 2)) {
      candidates.add(column)
    }
  }

  return [...candidates]
}

function validateTransformSteps(steps: TransformStep[], columns: string[]): Record<string, StepValidation> {
  const byStep: Record<string, StepValidation> = {}

  let currentColumns = [...columns]

  for (const step of steps) {
    const entry: StepValidation = { errors: [], warnings: [] }

    if (step.kind === 'filter') {
      if (!step.column) {
        entry.errors.push('Выберите колонку для фильтрации.')
      } else if (!currentColumns.includes(step.column)) {
        entry.errors.push(`Колонки ${step.column} нет в текущем результате.`)
      }
      if (!step.value.trim()) {
        entry.errors.push('Укажите значение для условия фильтрации.')
      }
      if (['gt', 'gte', 'lt', 'lte'].includes(step.operator)) {
        if (parseNumeric(step.value) === null) {
          entry.errors.push('Для этих операторов нужно число в поле значения.')
        }
      }
    }

    if (step.kind === 'select') {
      if (!step.columns.length) {
        entry.errors.push('Укажите хотя бы одну колонку.')
      }
      const unknown = step.columns.filter((colName) => !currentColumns.includes(colName))
      if (unknown.length) {
        entry.errors.push(`Неизвестные колонки: ${unknown.join(', ')}`)
      }
      if (unique(step.columns).length !== step.columns.length) {
        entry.warnings.push('Колонки повторяются. Повторы будут убраны.')
      }
      if (!entry.errors.length) {
        const selected = unique(step.columns)
        currentColumns = selected
      }
    }

    if (step.kind === 'sort') {
      if (!step.column) {
        entry.errors.push('Выберите колонку для сортировки.')
      } else if (!currentColumns.includes(step.column)) {
        entry.errors.push(`Колонки ${step.column} нет в текущем результате.`)
      }
    }

    if (step.kind === 'rename') {
      if (!step.oldName) {
        entry.errors.push('Укажите исходную колонку.')
      } else if (!currentColumns.includes(step.oldName)) {
        entry.errors.push(`Колонки ${step.oldName} нет в текущем результате.`)
      }
      if (!step.newName) {
        entry.errors.push('Укажите имя новой колонки.')
      }
      if (!entry.errors.length && step.oldName === step.newName) {
        entry.warnings.push('Новое имя совпадает со старым.')
      }
      if (!entry.errors.length) {
        currentColumns = unique(currentColumns.map((column) => (column === step.oldName ? step.newName : column)))
      }
    }

    if (step.kind === 'limit') {
      if (Number.isNaN(step.rows) || step.rows <= 0 || !Number.isInteger(step.rows)) {
        entry.errors.push('Лимит должен быть целым числом больше 0.')
      }
    }

    if (step.kind === 'groupBy') {
      if (!step.keys.length) {
        entry.errors.push('Укажите хотя бы один ключ группировки.')
      }
      const missingKeys = step.keys.filter((key) => !currentColumns.includes(key))
      if (missingKeys.length) {
        entry.errors.push(`Неизвестные ключи группировки: ${missingKeys.join(', ')}`)
      }
      if (!step.aggColumn) {
        entry.errors.push('Выберите колонку для агрегирования.')
      } else if (!currentColumns.includes(step.aggColumn)) {
        entry.errors.push(`Колонки ${step.aggColumn} нет в текущем результате.`)
      }
      if (!step.outputColumn) {
        entry.errors.push('Укажите имя результата агрегации.')
      }
      if (!entry.errors.length) {
        currentColumns = unique([...step.keys, step.outputColumn])
      }
    }

    if (step.kind === 'addColumn') {
      if (!step.sourceColumn) {
        entry.errors.push('Укажите базовую колонку.')
      } else if (!currentColumns.includes(step.sourceColumn)) {
        entry.errors.push(`Колонки ${step.sourceColumn} нет в текущем результате.`)
      }
      if (!step.outputColumn) {
        entry.errors.push('Укажите имя новой колонки.')
      }
      if (step.op !== 'copy' && parseNumeric(step.value) === null) {
        entry.errors.push('Для арифметики нужно число в поле значения.')
      }
      if (!entry.errors.length) {
        currentColumns = unique([...currentColumns, step.outputColumn])
      }
    }

    if (step.kind === 'replace') {
      if (!step.column) {
        entry.errors.push('Укажите колонку для замены.')
      } else if (!currentColumns.includes(step.column)) {
        entry.errors.push(`Колонки ${step.column} нет в текущем результате.`)
      }
      if (step.search === '') {
        entry.warnings.push('Пустая строка поиска заменит все пустые значения.')
      }
      if (!step.replacement) {
        entry.warnings.push('Пустая замена очистит значения.')
      }
    }

    if (step.kind === 'split') {
      if (!step.column) {
        entry.errors.push('Укажите колонку для разделения.')
      } else if (!currentColumns.includes(step.column)) {
        entry.errors.push(`Колонки ${step.column} нет в текущем результате.`)
      }
      if (step.delimiter === '') {
        entry.warnings.push('Разделитель пустой: будет разделение на символы.')
      }
      if (!step.leftColumn || !step.rightColumn) {
        entry.errors.push('Укажите имена выходных колонок.')
      }
      if (!entry.errors.length) {
        currentColumns = unique([...currentColumns, step.leftColumn, step.rightColumn])
      }
    }

    if (step.kind === 'pivot') {
      if (!step.index) {
        entry.errors.push('Укажите колонку индекса.')
      } else if (!currentColumns.includes(step.index)) {
        entry.errors.push(`Колонки ${step.index} нет в текущем результате.`)
      }
      if (!step.columns) {
        entry.errors.push('Укажите колонку для формирования колонок.')
      } else if (!currentColumns.includes(step.columns)) {
        entry.errors.push(`Колонки ${step.columns} нет в текущем результате.`)
      }
      if (!step.values) {
        entry.errors.push('Укажите колонку значений.')
      } else if (!currentColumns.includes(step.values)) {
        entry.errors.push(`Колонки ${step.values} нет в текущем результате.`)
      }
      if (!entry.errors.length) {
        const indexColumns = unique([step.index, ...currentColumns.filter((column) => ![step.columns, step.values].includes(column))])
        currentColumns = unique(indexColumns)
      }
    }

    if (step.kind === 'join') {
      if (!step.leftOn) {
        entry.errors.push('Укажите колонку ключа слева.')
      } else if (!currentColumns.includes(step.leftOn)) {
        entry.errors.push(`Колонки ${step.leftOn} нет в текущем результате.`)
      }
      if (!step.rightOn) {
        entry.errors.push('Укажите колонку ключа справа.')
      } else {
        const rightColumns = joinSourceColumns()
        if (!rightColumns.includes(step.rightOn)) {
          entry.errors.push(`Колонки ${step.rightOn} нет во вспомогательном источнике join.`)
        }
      }
      if (!entry.errors.length) {
        const joinedColumns = unique([...currentColumns, ...joinSourceColumns()])
        currentColumns = joinedColumns
      }
    }

    if (step.kind === 'fillMissing') {
      if (!step.column) {
        entry.errors.push('Укажите колонку для заполнения.')
      } else if (!currentColumns.includes(step.column)) {
        entry.errors.push(`Колонки ${step.column} нет в текущем результате.`)
      }
    }

    if (step.kind === 'changeType') {
      if (!step.column) {
        entry.errors.push('Укажите колонку для смены типа.')
      } else if (!currentColumns.includes(step.column)) {
        entry.errors.push(`Колонки ${step.column} нет в текущем результате.`)
      }
      if (!step.targetType) {
        entry.errors.push('Укажите целевой тип.')
      }
    }

    byStep[step.id] = entry
  }

  return byStep
}

function validatePipeline(csvText: string, rawColumns: string[], rows: string[][], steps: TransformStep[]): ValidationResult {
  const source = validateSourceCsv(csvText, rawColumns, rows)
  const byStep = validateTransformSteps(steps, unique(rawColumns))
  return {
    source,
    hasErrors: source.errors.length > 0 || Object.values(byStep).some((v) => v.errors.length > 0),
    byStep,
  }
}

async function applyStep(df: DataFrame, step: TransformStep): Promise<DataFrame> {
  if (step.kind === 'filter') {
    return df.filter(buildFilterExpr(step)).collect()
  }
  if (step.kind === 'select') {
    return df.select(...step.columns).collect()
  }
  if (step.kind === 'sort') {
    return df.sort(step.direction === 'asc' ? col(step.column).asc() : col(step.column).desc()).collect()
  }
  if (step.kind === 'rename') {
    return df.rename({ [step.oldName]: step.newName }).collect()
  }
  if (step.kind === 'limit') {
    return df.limit(Math.max(1, Math.floor(step.rows))).collect()
  }
  if (step.kind === 'groupBy') {
    return df.groupBy(...step.keys).agg({ [step.outputColumn]: aggExpr(step.aggFn, step.aggColumn) }).collect()
  }
  if (step.kind === 'addColumn') {
    return df.withColumn(step.outputColumn, addColumnExpr(step)).collect()
  }
  if (step.kind === 'replace') {
    return df.withColumn(step.column, replaceExpr(step)).collect()
  }
  if (step.kind === 'split') {
    const rows = await df.toArray()
    const out = rows.map((row) => {
      const raw = row[step.column]
      const text = raw == null ? '' : String(raw)
      const parts = text.split(step.delimiter || ',')
      const left = parts[0] ?? ''
      const right = parts.slice(1).join(step.delimiter || ',')
      return {
        ...row,
        [step.leftColumn]: left,
        [step.rightColumn]: right,
      }
    })
    return DataFrame.fromRows(out)
  }
  if (step.kind === 'pivot') {
    return df
      .pivot({
        index: step.index,
        columns: step.columns,
        values: step.values,
        agg: step.aggFn,
      })
      .collect()
  }
  if (step.kind === 'join') {
    const joinDf = DataFrame.fromCSV(JOIN_SOURCE_CSV)
    return df.join(joinDf, { leftOn: step.leftOn, rightOn: step.rightOn, how: step.how }).collect()
  }
  if (step.kind === 'fillMissing') {
    return df.fillNull(parseCellValueAsScalar(step.value), [step.column]).collect()
  }
  if (step.kind === 'changeType') {
    return df.withColumn(step.column, changeTypeExpr(step)).collect()
  }
  return df
}

function flattenErrors(validation: ValidationResult): string[] {
  const fromSource = validation.source.errors.map((message) => `Источник: ${message}`)
  const fromSteps = Object.entries(validation.byStep).flatMap(([id, info]) =>
    info.errors.map((error) => `${id}: ${error}`),
  )
  return [...fromSource, ...fromSteps]
}

function flattenWarnings(validation: ValidationResult): string[] {
  const fromSource = validation.source.warnings.map((message) => `Источник: ${message}`)
  const fromSteps = Object.entries(validation.byStep).flatMap(([id, info]) =>
    info.warnings.map((warning) => `${id}: ${warning}`),
  )
  return [...fromSource, ...fromSteps]
}
interface SuggestedStep {
  title: string
  description: string
  create: () => TransformStep
}

export function PowerQueryStudio() {
  const [csvText, setCsvText] = useState(SAMPLE_CSV)
  const [steps, setSteps] = useState<TransformStep[]>([])
  const [newKind, setNewKind] = useState<StepKind>('filter')
  const [outputMode, setOutputMode] = useState<'results' | 'charts' | 'diagnostics' | 'log' | 'warnings'>('results')
  const [activeColumn, setActiveColumn] = useState<string | null>(null)
  const [activeStepId, setActiveStepId] = useState<string | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)
  const [preview, setPreview] = useState<StepPreview | null>(null)
  const [executionLog, setExecutionLog] = useState<string[]>([])

  const rawColumns = useMemo(() => parseColumns(csvText), [csvText])
  const rows = useMemo(() => parseRows(csvText), [csvText])
  const records = useMemo(() => parseRecords(csvText), [csvText])
  const columns = useMemo(() => unique(rawColumns), [rawColumns])
  const validation = useMemo(() => validatePipeline(csvText, rawColumns, rows, steps), [csvText, rawColumns, rows, steps])
  const numericColumns = useMemo(() => findNumericColumns(rawColumns, records), [rawColumns, records])
  const runtimeCode = useMemo(() => renderPipelineCode(csvText, steps), [csvText, steps])
  const dataColumns = preview?.columns ?? columns
  const dataRows = preview ? preview.head : records.slice(0, 24)
  const sourceJoinColumns = useMemo(() => joinSourceColumns(), [])

  const validationErrors = useMemo(() => flattenErrors(validation), [validation])
  const validationWarnings = useMemo(() => flattenWarnings(validation), [validation])
  const canRun = !validation.hasErrors && csvText.trim().length > 0 && rawColumns.length > 0

  const stepTrail = steps.length
    ? steps.map((step, index) => `Step ${index + 1}: ${stepLabel(step)}`).join(' → ')
    : 'Шагов пока нет'
  const activeStep = steps.find((step) => step.id === activeStepId) ?? steps.at(-1) ?? null
  const outputTabs: Array<{ key: 'results' | 'charts' | 'diagnostics' | 'log' | 'warnings'; label: string }> = [
    { key: 'results', label: 'Results' },
    { key: 'charts', label: 'Charts' },
    { key: 'diagnostics', label: 'Diagnostics' },
    { key: 'log', label: 'Execution log' },
    { key: 'warnings', label: 'Warnings' },
  ]

  const suggestedSteps = useMemo((): SuggestedStep[] => {
    const groupKey = columns[0] ?? ''
    const groupAggSource = numericColumns[0] ?? columns[1] ?? columns[0] ?? ''
    const multiplySource = numericColumns[0] ?? columns[0] ?? ''

    if (!columns.length) return []

    const suggestionAggregate: SuggestedStep = {
      title: 'Агрегировать по региону',
      description: `Сгруппировать по ${groupKey || 'колонке'} и посчитать сумму по ${groupAggSource || 'числовой колонке'}`,
      create: () => ({
        ...createStep('groupBy', columns),
        keys: groupKey ? [groupKey] : [],
        aggColumn: groupAggSource,
        aggFn: 'sum',
        outputColumn: `${groupAggSource || 'Sum'}By${groupKey || 'Key'}`,
      }),
    }

    const suggestionAdd: SuggestedStep = {
      title: 'Добавить вычисляемую колонку',
      description: multiplySource
        ? `Скопировать ${multiplySource} и умножить на 2 в новой колонке`
        : 'Скопировать любое значение в новую колонку',
      create: () => ({
        ...createStep('addColumn', columns),
        sourceColumn: multiplySource,
        outputColumn: multiplySource ? `${multiplySource}_x2` : 'NewColumn',
        op: multiplySource ? 'mul' : 'copy',
        value: multiplySource ? '2' : '1',
      }),
    }

    return [suggestionAggregate, suggestionAdd]
  }, [columns, numericColumns])

  function createQuickStep(kind: QuickActionKind, anchorColumn: string): TransformStep {
    if (kind === 'filter') {
      const source = createStep('filter', columns)
      return { ...(source as FilterStep), kind: 'filter', column: anchorColumn, value: '' }
    }
    if (kind === 'replace') {
      const source = createStep('replace', columns)
      return { ...(source as ReplaceStep), kind: 'replace', column: anchorColumn, search: '', replacement: '' }
    }
    if (kind === 'split') {
      const source = createStep('split', columns)
      return {
        ...(source as SplitStep),
        kind: 'split',
        column: anchorColumn,
        delimiter: ',',
        leftColumn: `${anchorColumn}_left`,
        rightColumn: `${anchorColumn}_right`,
      }
    }
    if (kind === 'groupBy') {
      const source = createStep('groupBy', columns)
      return {
        ...(source as GroupByStep),
        kind: 'groupBy',
        keys: [anchorColumn],
        aggColumn: columns.find((column) => column !== anchorColumn) ?? anchorColumn,
        outputColumn: `${anchorColumn}_agg`,
      }
    }
    if (kind === 'pivot') {
      const source = createStep('pivot', columns)
      const nextColumn = columns.find((column) => column !== anchorColumn) ?? columns[0] ?? ''
      return { ...(source as PivotStep), kind: 'pivot', index: anchorColumn, columns: nextColumn, values: nextColumn, aggFn: 'sum' }
    }
    if (kind === 'join') {
      const source = createStep('join', columns)
      return {
        ...(source as JoinStep),
        kind: 'join',
        leftOn: anchorColumn,
        rightOn: sourceJoinColumns[0] ?? '',
        how: 'left',
      }
    }
    if (kind === 'fillMissing') {
      const source = createStep('fillMissing', columns)
      return { ...(source as FillMissingStep), kind: 'fillMissing', column: anchorColumn, value: '0' }
    }
    const source = createStep('changeType', columns)
    return { ...(source as ChangeTypeStep), kind: 'changeType', column: anchorColumn, targetType: 'text' }
  }

  function addStep() {
    const created = createStep(newKind, columns)
    setSteps((current) => [...current, created])
    setActiveStepId(created.id)
  }

  function addSuggestedStep(suggested: SuggestedStep) {
    const created = suggested.create()
    setSteps((current) => [...current, created])
    setActiveStepId(created.id)
  }

  function addStepFromColumn(kind: QuickActionKind, anchorColumn: string) {
    const created = createQuickStep(kind, anchorColumn)
    setSteps((current) => [...current, created])
    setActiveStepId(created.id)
    setActiveColumn(anchorColumn)
  }

  function removeStep(id: string) {
    setSteps((current) => {
      const next = current.filter((step) => step.id !== id)
      if (activeStepId === id) {
        setActiveStepId(next.at(-1)?.id ?? null)
      }
      return next
    })
  }

  function updateStep(id: string, patch: Partial<TransformStep>) {
    setSteps((current) =>
      current.map((step) => {
        if (step.id !== id) return step
        return { ...(step as object), ...patch, id: step.id } as TransformStep
      }),
    )
  }

  function renderStepEditor(step: TransformStep) {
    return (
      <div className="dq-step-fields">
        {step.kind === 'filter' ? (
          <>
            <div className="form-row">
              <span>Колонка</span>
              <select
                className="toolbar-input"
                value={step.column}
                onChange={(event) => updateStep(step.id, { kind: 'filter', column: event.target.value } as Partial<TransformStep>)}
              >
                <option value="">Выберите</option>
                {columns.map((column) => (
                  <option key={`${step.id}-${column}`} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <span>Оператор</span>
              <select
                className="toolbar-input"
                value={step.operator}
                onChange={(event) => updateStep(step.id, { kind: 'filter', operator: event.target.value as FilterOp } as Partial<TransformStep>)}
              >
                {FILTER_OPERATORS.map((operator) => (
                  <option key={`${step.id}-${operator}`} value={operator}>
                    {filterOperatorLabel(operator)}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <span>Значение</span>
              <input
                className="toolbar-input"
                value={step.value}
                onChange={(event) => updateStep(step.id, { kind: 'filter', value: event.target.value } as Partial<TransformStep>)}
              />
            </div>
          </>
        ) : null}

        {step.kind === 'select' ? (
          <div className="form-row">
            <span>Колонки</span>
            <input
              className="toolbar-input"
              value={step.columns.join(', ')}
              onChange={(event) => updateStep(step.id, { kind: 'select', columns: normalizeColumns(event.target.value) } as Partial<TransformStep>)}
            />
          </div>
        ) : null}

        {step.kind === 'sort' ? (
          <>
            <div className="form-row">
              <span>Колонка</span>
              <select
                className="toolbar-input"
                value={step.column}
                onChange={(event) => updateStep(step.id, { kind: 'sort', column: event.target.value } as Partial<TransformStep>)}
              >
                <option value="">Выберите</option>
                {columns.map((column) => (
                  <option key={`${step.id}-${column}`} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <span>Порядок</span>
              <select
                className="toolbar-input"
                value={step.direction}
                onChange={(event) => updateStep(step.id, { kind: 'sort', direction: event.target.value as 'asc' | 'desc' } as Partial<TransformStep>)}
              >
                <option value="asc">Возрастание</option>
                <option value="desc">Убывание</option>
              </select>
            </div>
          </>
        ) : null}

        {step.kind === 'rename' ? (
          <>
            <div className="form-row">
              <span>Исходная</span>
              <select
                className="toolbar-input"
                value={step.oldName}
                onChange={(event) => updateStep(step.id, { kind: 'rename', oldName: event.target.value } as Partial<TransformStep>)}
              >
                <option value="">Выберите</option>
                {columns.map((column) => (
                  <option key={`${step.id}-${column}`} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <span>Новая</span>
              <input
                className="toolbar-input"
                value={step.newName}
                onChange={(event) => updateStep(step.id, { kind: 'rename', newName: event.target.value } as Partial<TransformStep>)}
              />
            </div>
          </>
        ) : null}

        {step.kind === 'limit' ? (
          <div className="form-row">
            <span>Количество</span>
            <input
              className="toolbar-input"
              type="number"
              min="1"
              value={step.rows}
              onChange={(event) => updateStep(step.id, { kind: 'limit', rows: Number(event.target.value) } as Partial<TransformStep>)}
            />
          </div>
        ) : null}

        {step.kind === 'groupBy' ? (
          <>
            <div className="form-row">
              <span>Ключи</span>
              <input
                className="toolbar-input"
                value={step.keys.join(', ')}
                onChange={(event) => updateStep(step.id, { kind: 'groupBy', keys: normalizeColumns(event.target.value) } as Partial<TransformStep>)}
              />
            </div>
            <div className="form-row">
              <span>Агрегируемая колонка</span>
              <select
                className="toolbar-input"
                value={step.aggColumn}
                onChange={(event) => updateStep(step.id, { kind: 'groupBy', aggColumn: event.target.value } as Partial<TransformStep>)}
              >
                <option value="">Выберите</option>
                {columns.map((column) => (
                  <option key={`${step.id}-${column}`} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <span>Агрегация</span>
              <select
                className="toolbar-input"
                value={step.aggFn}
                onChange={(event) => updateStep(step.id, { kind: 'groupBy', aggFn: event.target.value as AggregateFn } as Partial<TransformStep>)}
              >
                {AGG_FUNCTIONS.map((agg) => (
                  <option key={`${step.id}-${agg}`} value={agg}>
                    {agg}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <span>Результат</span>
              <input
                className="toolbar-input"
                value={step.outputColumn}
                onChange={(event) => updateStep(step.id, { kind: 'groupBy', outputColumn: event.target.value } as Partial<TransformStep>)}
              />
            </div>
          </>
        ) : null}

        {step.kind === 'addColumn' ? (
          <>
            <div className="form-row">
              <span>Базовая колонка</span>
              <select
                className="toolbar-input"
                value={step.sourceColumn}
                onChange={(event) => updateStep(step.id, { kind: 'addColumn', sourceColumn: event.target.value } as Partial<TransformStep>)}
              >
                <option value="">Выберите</option>
                {columns.map((column) => (
                  <option key={`${step.id}-${column}`} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <span>Новая колонка</span>
              <input
                className="toolbar-input"
                value={step.outputColumn}
                onChange={(event) => updateStep(step.id, { kind: 'addColumn', outputColumn: event.target.value } as Partial<TransformStep>)}
              />
            </div>
            <div className="form-row">
              <span>Оператор</span>
              <select
                className="toolbar-input"
                value={step.op}
                onChange={(event) => updateStep(step.id, { kind: 'addColumn', op: event.target.value as AddOp } as Partial<TransformStep>)}
              >
                {ADD_OPERATORS.map((op) => (
                  <option key={`${step.id}-${op}`} value={op}>
                    {op}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <span>Значение</span>
              <input
                className="toolbar-input"
                disabled={step.op === 'copy'}
                value={step.value}
                onChange={(event) => updateStep(step.id, { kind: 'addColumn', value: event.target.value } as Partial<TransformStep>)}
              />
            </div>
          </>
        ) : null}

        {step.kind === 'replace' ? (
          <>
            <div className="form-row">
              <span>Колонка</span>
              <select
                className="toolbar-input"
                value={step.column}
                onChange={(event) => updateStep(step.id, { kind: 'replace', column: event.target.value } as Partial<TransformStep>)}
              >
                <option value="">Выберите</option>
                {columns.map((column) => (
                  <option key={`${step.id}-${column}`} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <span>Найти</span>
              <input
                className="toolbar-input"
                value={step.search}
                onChange={(event) => updateStep(step.id, { kind: 'replace', search: event.target.value } as Partial<TransformStep>)}
              />
            </div>
            <div className="form-row">
              <span>Заменить</span>
              <input
                className="toolbar-input"
                value={step.replacement}
                onChange={(event) => updateStep(step.id, { kind: 'replace', replacement: event.target.value } as Partial<TransformStep>)}
              />
            </div>
          </>
        ) : null}

        {step.kind === 'split' ? (
          <>
            <div className="form-row">
              <span>Колонка</span>
              <select
                className="toolbar-input"
                value={step.column}
                onChange={(event) => updateStep(step.id, { kind: 'split', column: event.target.value } as Partial<TransformStep>)}
              >
                <option value="">Выберите</option>
                {columns.map((column) => (
                  <option key={`${step.id}-${column}`} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <span>Разделитель</span>
              <input
                className="toolbar-input"
                value={step.delimiter}
                onChange={(event) => updateStep(step.id, { kind: 'split', delimiter: event.target.value } as Partial<TransformStep>)}
              />
            </div>
            <div className="form-row">
              <span>Левая колонка</span>
              <input
                className="toolbar-input"
                value={step.leftColumn}
                onChange={(event) => updateStep(step.id, { kind: 'split', leftColumn: event.target.value } as Partial<TransformStep>)}
              />
            </div>
            <div className="form-row">
              <span>Правая колонка</span>
              <input
                className="toolbar-input"
                value={step.rightColumn}
                onChange={(event) => updateStep(step.id, { kind: 'split', rightColumn: event.target.value } as Partial<TransformStep>)}
              />
            </div>
          </>
        ) : null}

        {step.kind === 'pivot' ? (
          <>
            <div className="form-row">
              <span>Индекс</span>
              <select
                className="toolbar-input"
                value={step.index}
                onChange={(event) => updateStep(step.id, { kind: 'pivot', index: event.target.value } as Partial<TransformStep>)}
              >
                <option value="">Выберите</option>
                {columns.map((column) => (
                  <option key={`${step.id}-${column}`} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <span>Колонки</span>
              <select
                className="toolbar-input"
                value={step.columns}
                onChange={(event) => updateStep(step.id, { kind: 'pivot', columns: event.target.value } as Partial<TransformStep>)}
              >
                <option value="">Выберите</option>
                {columns.map((column) => (
                  <option key={`${step.id}-${column}`} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <span>Значение</span>
              <select
                className="toolbar-input"
                value={step.values}
                onChange={(event) => updateStep(step.id, { kind: 'pivot', values: event.target.value } as Partial<TransformStep>)}
              >
                <option value="">Выберите</option>
                {columns.map((column) => (
                  <option key={`${step.id}-${column}`} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <span>Агрегация</span>
              <select
                className="toolbar-input"
                value={step.aggFn}
                onChange={(event) => updateStep(step.id, { kind: 'pivot', aggFn: event.target.value as AggregateFn } as Partial<TransformStep>)}
              >
                {AGG_FUNCTIONS.map((agg) => (
                  <option key={`${step.id}-${agg}`} value={agg}>
                    {agg}
                  </option>
                ))}
              </select>
            </div>
          </>
        ) : null}

        {step.kind === 'join' ? (
          <>
            <div className="form-row">
              <span>Левый ключ</span>
              <select
                className="toolbar-input"
                value={step.leftOn}
                onChange={(event) => updateStep(step.id, { kind: 'join', leftOn: event.target.value } as Partial<TransformStep>)}
              >
                <option value="">Выберите</option>
                {dataColumns.map((column) => (
                  <option key={`${step.id}-${column}`} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <span>Правый ключ</span>
              <select
                className="toolbar-input"
                value={step.rightOn}
                onChange={(event) => updateStep(step.id, { kind: 'join', rightOn: event.target.value } as Partial<TransformStep>)}
              >
                <option value="">Выберите</option>
                {sourceJoinColumns.map((column) => (
                  <option key={`${step.id}-${column}`} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <span>Тип</span>
              <select
                className="toolbar-input"
                value={step.how}
                onChange={(event) => updateStep(step.id, { kind: 'join', how: event.target.value as JoinType } as Partial<TransformStep>)}
              >
                {JOIN_TYPES.map((joinType) => (
                  <option key={`${step.id}-${joinType}`} value={joinType}>
                    {joinType}
                  </option>
                ))}
              </select>
            </div>
          </>
        ) : null}

        {step.kind === 'fillMissing' ? (
          <>
            <div className="form-row">
              <span>Колонка</span>
              <select
                className="toolbar-input"
                value={step.column}
                onChange={(event) => updateStep(step.id, { kind: 'fillMissing', column: event.target.value } as Partial<TransformStep>)}
              >
                <option value="">Выберите</option>
                {columns.map((column) => (
                  <option key={`${step.id}-${column}`} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <span>Значение</span>
              <input
                className="toolbar-input"
                value={step.value}
                onChange={(event) => updateStep(step.id, { kind: 'fillMissing', value: event.target.value } as Partial<TransformStep>)}
              />
            </div>
          </>
        ) : null}

        {step.kind === 'changeType' ? (
          <>
            <div className="form-row">
              <span>Колонка</span>
              <select
                className="toolbar-input"
                value={step.column}
                onChange={(event) => updateStep(step.id, { kind: 'changeType', column: event.target.value } as Partial<TransformStep>)}
              >
                <option value="">Выберите</option>
                {columns.map((column) => (
                  <option key={`${step.id}-${column}`} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <span>Тип</span>
              <select
                className="toolbar-input"
                value={step.targetType}
                onChange={(event) => updateStep(step.id, { kind: 'changeType', targetType: event.target.value as DataType } as Partial<TransformStep>)}
              >
                {DATA_TYPES.map((dataType) => (
                  <option key={`${step.id}-${dataType}`} value={dataType}>
                    {dataType}
                  </option>
                ))}
              </select>
            </div>
          </>
        ) : null}
      </div>
    )
  }

  async function runPipeline() {
    if (validation.hasErrors) {
      setRunError('Перед запуском исправьте ошибки валидации.')
      setOutputMode('diagnostics')
      return
    }

    const started = new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    setExecutionLog((current) => [ ...current, `[${started}] Запуск: выполнен старт`].slice(-40))

    setIsRunning(true)
    setRunError(null)
    setPreview(null)

    try {
      let current = DataFrame.fromCSV(csvText)
      for (const step of steps) {
        current = await applyStep(current, step)
      }
      const head = current.toArray().slice(0, 25)
      const result: StepPreview = {
        title: 'Результат',
        rowCount: current.shape[0],
        columns: current.columns,
        head,
      }
      setPreview(result)
      const finished = new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      setExecutionLog((currentLog) => [
        ...currentLog,
        `[${finished}] Выполнено: ${result.rowCount} строк, ${result.columns.length} колонок`,
      ].slice(-40))
      setOutputMode('results')
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setRunError(message)
      const failed = new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      setExecutionLog((currentLog) => [...currentLog, `[${failed}] Ошибка: ${message}`].slice(-40))
      setOutputMode('log')
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="app">
      <div className="chrome">
        <div className="chrome-top">
          <div className="brand">DataLab Studio</div>
          <div className="toolbar-actions">
            <span className="hint">Подготовка и трансформация данных</span>
          </div>
        </div>
        <div className="toolbar dq-header-toolbar">
          <div className="dq-tabs">
            {['Import', 'Transform', 'Analyze', 'Visualize', 'Model'].map((title) => (
              <button key={title} className="btn btn-ghost" type="button">
                {title}
              </button>
            ))}
            <button className="btn btn-primary" type="button" onClick={runPipeline} disabled={isRunning || !canRun}>
              {isRunning ? 'Выполняется…' : 'Run'}
            </button>
          </div>
          <div className="dq-run-status">{columns.length ? <span className="hint">Колонки: {columns.join(', ')}</span> : null}</div>
        </div>
      </div>

      <main className="dq-shell">
        <section className="dq-project">
          <div className="panel-header">PROJECT</div>
          <div className="dq-tree">
            {PROJECT_TREE.map((group) => (
              <details key={group.title} className="dq-tree-group" open>
                <summary>{group.title}</summary>
                {group.items.map((item) => (
                  <button key={`${group.title}-${item}`} className="dq-tree-item" type="button">
                    {item}
                  </button>
                ))}
              </details>
            ))}
          </div>
        </section>

        <section className="dq-workspace">
          <div className="panel-header">WORKSPACE</div>
          <div className="dq-workspace-head">
            <div className="dq-stage">Data | Query</div>
            <div className="dq-stage-path">{stepTrail}</div>
          </div>

          <div className="dq-workspace-body">
            <div className="dq-data-card">
              <div className="dq-card-header">
                <div>Data</div>
              </div>
              <label className="dq-label">
                <span>Источник CSV</span>
                <textarea className="dq-textarea" value={csvText} onChange={(event) => setCsvText(event.target.value)} />
              </label>

              <div className="dq-card-header">Query grid</div>
              <div className="dq-grid-wrap">
                <table className="dq-table">
                  <thead>
                    <tr>
                      {dataColumns.map((column) => (
                        <th key={column}>
                          <button
                            className={`dq-col-btn ${activeColumn === column ? 'is-active' : ''}`}
                            type="button"
                            onClick={() => setActiveColumn(column)}
                          >
                            {column}
                          </button>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {dataRows.map((row, rowIndex) => (
                      <tr key={`${preview?.title ?? 'source'}-${rowIndex}`}>
                        {dataColumns.map((column) => {
                          const value = row[column]
                          const text = value === null || value === undefined ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value)
                          return <td key={`${column}-${rowIndex}`}>{text}</td>
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
                {!dataRows.length ? <div className="hint">Нет данных для отображения.</div> : null}
              </div>
              <div className="dq-column-bar">
                {activeColumn ? (
                  <div className="dq-quick-actions">
                    <span className="dq-quick-title">Колонка: {activeColumn}</span>
                    {COLUMN_QUICK_ACTIONS.map((action) => (
                      <button
                        className="dq-chip"
                        key={action.kind}
                        type="button"
                        onClick={() => addStepFromColumn(action.kind, activeColumn)}
                      >
                        {action.label}
                      </button>
                    ))}
                    <button className="dq-chip btn-ghost" type="button" onClick={() => setActiveColumn(null)}>
                      Снять выбор
                    </button>
                  </div>
                ) : (
                  <div className="hint">Кликните по колонке в сетке, затем выберите действие.</div>
                )}
              </div>
            </div>

            <div className="dq-query-card">
              <div className="dq-card-header">Query</div>
              <div className="dq-query-controls">
                <select className="toolbar-input" value={newKind} onChange={(event) => setNewKind(event.target.value as StepKind)}>
                  <option value="filter">Фильтр</option>
                  <option value="select">Выбор колонок</option>
                  <option value="sort">Сортировка</option>
                  <option value="rename">Переименование</option>
                  <option value="limit">Лимит</option>
                  <option value="groupBy">Агрегация</option>
                  <option value="addColumn">Колонка</option>
                  <option value="replace">Замена</option>
                  <option value="split">Разделение</option>
                  <option value="pivot">Свод</option>
                  <option value="join">Объединение</option>
                  <option value="fillMissing">Заполнить пропуски</option>
                  <option value="changeType">Смена типа</option>
                </select>
                <button className="btn btn-primary" type="button" onClick={addStep}>
                  Добавить шаг
                </button>
              </div>

              <div className="dq-stage">Предложенные шаги</div>
              <div className="dq-suggestions">
                {suggestedSteps.length ? (
                  suggestedSteps.map((suggestion) => (
                    <button key={suggestion.title} className="btn btn-ghost" type="button" onClick={() => addSuggestedStep(suggestion)}>
                      {suggestion.title}
                    </button>
                  ))
                ) : (
                  <div className="hint">Добавьте поля в CSV, чтобы появились предложения.</div>
                )}
              </div>

              {(validationErrors.length || validationWarnings.length) ? (
                <div className="dq-validate-box">
                  <strong>Проверка перед запуском</strong>
                  {validationErrors.length ? (
                    <ul className="dq-list dq-list--error">
                      {validationErrors.map((message) => (
                        <li key={message}>{message}</li>
                      ))}
                    </ul>
                  ) : null}
                  {validationWarnings.length ? (
                    <ul className="dq-list">
                      {validationWarnings.map((message) => (
                        <li key={`warn-${message}`}>{message}</li>
                      ))}
                    </ul>
                  ) : null}
                </div>
              ) : null}

              <div className="dq-step-list">
                {steps.length === 0 ? <div className="hint">Добавьте шаг, либо выберите одно из предложений.</div> : null}
                {steps.map((step) => {
                  const state = validation.byStep[step.id] ?? { errors: [], warnings: [] }
                  const isActive = activeStep?.id === step.id
                  return (
                    <div key={step.id} className={`dq-step-item ${isActive ? 'is-active' : ''}`}>
                      <button className="dq-step-button" type="button" onClick={() => setActiveStepId(step.id)}>
                        <strong>{stepLabel(step)}</strong>
                        <span className="dq-step-badges">
                          {state.errors.length ? <span className="dq-badge dq-badge--error">Ошибка</span> : null}
                          {state.warnings.length ? <span className="dq-badge dq-badge--warn">Предупреждение</span> : null}
                        </span>
                      </button>
                      <button className="btn btn-ghost" type="button" onClick={() => removeStep(step.id)}>
                        Удалить
                      </button>
                    </div>
                  )
                })}
              </div>

              {activeStep ? (
                <div className="dq-step-editor">
                  <div className="dq-card-header">Актуальный шаг</div>
                  {renderStepEditor(activeStep)}
                </div>
              ) : null}
            </div>
          </div>
        </section>

        <section className="dq-context">
          <div className="panel-header">CONTEXT</div>
          <div className="dq-context-body">
            <div className="dq-context-block">
              <div className="dq-context-block-title">Properties</div>
              <ul className="dq-metadata">
                <li>Источн. колонок: {columns.length}</li>
                <li>Шагов: {steps.length}</li>
                <li>Колонки после последнего шага: {dataColumns.length}</li>
                <li>Строк до запуска: {records.length}</li>
                <li>Выбрана колонка: {activeColumn ?? 'не выбрана'}</li>
              </ul>
            </div>
            <div className="dq-context-block">
              <div className="dq-context-block-title">Applied Steps</div>
              {steps.length ? (
                <ol className="dq-list">
                  {steps.map((step) => (
                    <li key={`ctx-${step.id}`}>{stepLabel(step)}</li>
                  ))}
                </ol>
              ) : (
                <div className="hint">Пока пусто.</div>
              )}
            </div>
            <div className="dq-context-block">
              <div className="dq-context-block-title">Formula / generated library code</div>
              <pre className="dq-code">{runtimeCode}</pre>
            </div>
          </div>
        </section>
      </main>

      <footer className="dq-footer">
        <div className="dq-tabs">
          {outputTabs.map((tab) => (
            <button
              key={tab.key}
              className={`dq-tab ${outputMode === tab.key ? 'is-active' : ''}`}
              type="button"
              onClick={() => setOutputMode(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <div className="dq-tab-body">
          {outputMode === 'results' ? (
            <>
              <div className="dq-result-header">
                {preview ? `${preview.title}: ${preview.rowCount} строк, ${preview.columns.length} колонок` : 'Результат появится после запуска'}
              </div>
              <div className="dq-grid-wrap">
                <table className="dq-table">
                  <thead>
                    <tr>
                      {(preview?.columns ?? dataColumns).map((column) => (
                        <th key={column}>{column}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(preview ? preview.head : dataRows).map((row, rowIndex) => (
                      <tr key={`res-${rowIndex}`}>
                        {(preview ? preview.columns : dataColumns).map((column) => {
                          const value = row[column]
                          const text = value === null || value === undefined ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value)
                          return <td key={`${column}-${rowIndex}`}>{text}</td>
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : null}

          {outputMode === 'charts' ? <div className="hint">Графики будут добавлены позже.</div> : null}

          {outputMode === 'diagnostics' ? (
            <>
              <div className="dq-context-block-title">Диагностика</div>
              {validationErrors.length ? (
                <ul className="dq-list dq-list--error">
                  {validationErrors.map((message) => (
                    <li key={`diag-${message}`}>{message}</li>
                  ))}
                </ul>
              ) : (
                <div className="hint">Ошибок не обнаружено.</div>
              )}
              {validationWarnings.length ? (
                <ul className="dq-list">
                  {validationWarnings.map((message) => (
                    <li key={`diag-warn-${message}`}>{message}</li>
                  ))}
                </ul>
              ) : null}
            </>
          ) : null}

          {outputMode === 'log' ? (
            <div className="dq-log">
              {executionLog.length ? executionLog.map((message) => <div key={message}>{message}</div>) : <div className="hint">Журнал пока пуст.</div>}
              {runError ? <div className="dq-log-error">{runError}</div> : null}
            </div>
          ) : null}

          {outputMode === 'warnings' ? (
            <div className="dq-warning-list">
              {validationWarnings.length ? (
                validationWarnings.map((message) => <div key={`warn-${message}`}>⚠ {message}</div>)
              ) : (
                <div className="hint">Предупреждения после проверки не найдены.</div>
              )}
            </div>
          ) : null}
        </div>
      </footer>
    </div>
  )
}




