/**
 * Fused kernel for arithmetic / math chains over numeric columns.
 *
 * evalVec materializes one Float64Array per expression node, so `(x − μ) / σ` costs two full passes
 * and two 8n-byte temporaries. For a tree made only of numeric columns, numeric literals, arithmetic
 * (+ − × ÷ mod pow), element-wise math (abs neg sqrt log … round) and optionally one comparison at
 * the root, this module compiles the whole tree into a single `for` loop:
 *
 *   out[i] = ((c0[i] - 50.1) / 10.2)
 *
 * The loop is generated as JS source and compiled once per distinct expression via `new Function`
 * (V8 JITs it like hand-written code; cached by source). Where `new Function` is unavailable (CSP
 * without unsafe-eval) the same tree is evaluated through a closure per node — still one pass and no
 * temporaries, just slower per element.
 *
 * Null semantics match the node-by-node path: the output is null wherever any referenced column is
 * null (bitmap AND); values at those rows are computed but masked.
 */
import { getColumn, isNumeric, type Column, type TableView } from '@columna/arrow'
import type { ExprNode, MathOp } from './types.js'
import { roundHalfAway } from './math.js'

const ARITH: Record<string, string> = { add: '+', sub: '-', mul: '*', div: '/', mod: '%', pow: '**' }
const CMP: Record<string, string> = { eq: '===', neq: '!==', gt: '>', gte: '>=', lt: '<', lte: '<=' }
const MATH_FN: Record<string, string> = {
  abs: 'Math.abs',
  sqrt: 'Math.sqrt',
  log10: 'Math.log10',
  log2: 'Math.log2',
  exp: 'Math.exp',
  floor: 'Math.floor',
  ceil: 'Math.ceil',
  sign: 'Math.sign',
}

export type FusedResult =
  | { kind: 'num'; data: Float64Array; bitmap: Uint8Array | undefined }
  | { kind: 'bool'; data: Uint8Array; bitmap: Uint8Array | undefined }

/** Minimum number of fusable operator nodes before fusing pays off (a single op is already one pass). */
const MIN_OPS = 2

type Ctx = { table: TableView; cols: Column[]; names: Map<string, number>; ops: number }

function isNumericCol(c: Column): boolean {
  return isNumeric(c.field.dtype) || c.field.dtype === 'datetime'
}

/** Emit a JS expression string for a numeric subtree, or null if any node is outside the fusable subset. */
function emit(expr: ExprNode, ctx: Ctx): string | null {
  switch (expr.type) {
    case 'lit': {
      const v = expr.value
      if (typeof v !== 'number') return null
      return v < 0 || Object.is(v, -0) ? `(${String(v)})` : String(v)
    }
    case 'col': {
      let k = ctx.names.get(expr.name)
      if (k === undefined) {
        const c = getColumn(ctx.table, expr.name)
        if (!isNumericCol(c)) return null
        k = ctx.cols.length
        ctx.cols.push(c)
        ctx.names.set(expr.name, k)
      }
      return `c${k}[i]`
    }
    case 'alias':
      return emit(expr.expr, ctx)
    case 'binary': {
      const op = ARITH[expr.op]
      if (!op) return null
      const l = emit(expr.left, ctx)
      if (l === null) return null
      const r = emit(expr.right, ctx)
      if (r === null) return null
      ctx.ops++
      return `(${l} ${op} ${r})`
    }
    case 'unary': {
      const inner = emit(expr.expr, ctx)
      if (inner === null) return null
      ctx.ops++
      switch (expr.op) {
        case 'neg':
          return `(-${inner})`
        case 'log':
          return expr.base === undefined ? `Math.log(${inner})` : `(Math.log(${inner}) * ${1 / Math.log(Number(expr.base))})`
        case 'round':
          // Number(): the node may come from JSON / untyped callers; only a numeric literal may reach the generated source
          return `rha(${inner}, ${Number(expr.decimals ?? 0)})`
        default: {
          const fn = MATH_FN[expr.op]
          return fn ? `${fn}(${inner})` : null
        }
      }
    }
    default:
      return null
  }
}

type Kernel = (cols: ArrayLike<number>[], rha: typeof roundHalfAway, n: number, out: Float64Array | Uint8Array) => void

const cache = new Map<string, Kernel>()
const CACHE_MAX = 256
let canCompile: boolean | null = null

function compile(body: string, cmp: boolean, nCols: number): Kernel {
  const key = (cmp ? 'b:' : 'n:') + body
  const hit = cache.get(key)
  if (hit) return hit
  if (canCompile === null) {
    try {
      new Function('return 1')()
      canCompile = true
    } catch {
      canCompile = false
    }
  }
  let kernel: Kernel
  if (canCompile) {
    const decl = Array.from({ length: nCols }, (_, k) => `const c${k} = cols[${k}];`).join(' ')
    const store = cmp ? `out[i] = (${body}) ? 1 : 0;` : `out[i] = ${body};`
    kernel = new Function('cols', 'rha', 'n', 'out', `${decl} for (let i = 0; i < n; i++) { ${store} }`) as Kernel
  } else {
    kernel = null as unknown as Kernel // closure path is built per call (needs the tree, not the source)
  }
  if (cache.size >= CACHE_MAX) cache.clear()
  if (kernel) cache.set(key, kernel)
  return kernel
}

// ---- closure fallback (no eval) ---------------------------------------------------------------
type Fn = (i: number) => number

function closure(expr: ExprNode, ctx: Ctx): Fn | null {
  switch (expr.type) {
    case 'lit': {
      const v = expr.value
      if (typeof v !== 'number') return null
      return () => v
    }
    case 'col': {
      const k = ctx.names.get(expr.name)!
      const d = ctx.cols[k]!.data as ArrayLike<number>
      return (i) => d[i]!
    }
    case 'alias':
      return closure(expr.expr, ctx)
    case 'binary': {
      const l = closure(expr.left, ctx)
      const r = closure(expr.right, ctx)
      if (!l || !r) return null
      switch (expr.op) {
        case 'add':
          return (i) => l(i) + r(i)
        case 'sub':
          return (i) => l(i) - r(i)
        case 'mul':
          return (i) => l(i) * r(i)
        case 'div':
          return (i) => l(i) / r(i)
        case 'mod':
          return (i) => l(i) % r(i)
        case 'pow':
          return (i) => l(i) ** r(i)
        case 'eq':
          return (i) => (l(i) === r(i) ? 1 : 0)
        case 'neq':
          return (i) => (l(i) !== r(i) ? 1 : 0)
        case 'gt':
          return (i) => (l(i) > r(i) ? 1 : 0)
        case 'gte':
          return (i) => (l(i) >= r(i) ? 1 : 0)
        case 'lt':
          return (i) => (l(i) < r(i) ? 1 : 0)
        case 'lte':
          return (i) => (l(i) <= r(i) ? 1 : 0)
        default:
          return null
      }
    }
    case 'unary': {
      const a = closure(expr.expr, ctx)
      if (!a) return null
      const op = expr.op as MathOp | 'abs' | 'neg'
      switch (op) {
        case 'neg':
          return (i) => -a(i)
        case 'abs':
          return (i) => Math.abs(a(i))
        case 'sqrt':
          return (i) => Math.sqrt(a(i))
        case 'log': {
          if (expr.base === undefined) return (i) => Math.log(a(i))
          const inv = 1 / Math.log(expr.base)
          return (i) => Math.log(a(i)) * inv
        }
        case 'log10':
          return (i) => Math.log10(a(i))
        case 'log2':
          return (i) => Math.log2(a(i))
        case 'exp':
          return (i) => Math.exp(a(i))
        case 'round': {
          const d = expr.decimals ?? 0
          return (i) => roundHalfAway(a(i), d)
        }
        case 'floor':
          return (i) => Math.floor(a(i))
        case 'ceil':
          return (i) => Math.ceil(a(i))
        case 'sign':
          return (i) => Math.sign(a(i))
        default:
          return null
      }
    }
    default:
      return null
  }
}

function andBitmaps(cols: Column[], n: number): Uint8Array | undefined {
  let out: Uint8Array | undefined
  for (const c of cols) {
    const bm = c.nullBitmap
    if (!bm) continue
    if (!out) {
      out = bm
      continue
    }
    const len = Math.min(out.length, bm.length, (n + 7) >> 3)
    const merged = new Uint8Array(len)
    for (let i = 0; i < len; i++) merged[i] = out[i]! & bm[i]!
    out = merged
  }
  return out
}

/**
 * Try to evaluate `expr` as one fused pass. Returns null when the tree is not fusable or is too
 * small to benefit (fewer than MIN_OPS operator nodes) — the caller then falls back to per-node evaluation.
 */
export function tryFused(table: TableView, expr: ExprNode, n: number): FusedResult | null {
  // Root may be a comparison (→ bool); everything below must be numeric.
  let root = expr
  while (root.type === 'alias') root = root.expr
  const cmp = root.type === 'binary' && root.op in CMP ? root : null

  const ctx: Ctx = { table, cols: [], names: new Map(), ops: 0 }
  let body: string | null
  if (cmp) {
    const l = emit(cmp.left, ctx)
    if (l === null) return null
    const r = emit(cmp.right, ctx)
    if (r === null) return null
    body = `${l} ${CMP[cmp.op]} ${r}`
    // A bare `col cmp lit` is handled by dedicated filter kernels; fuse only real chains.
    if (ctx.ops < 1) return null
  } else {
    body = emit(root, ctx)
    if (body === null || ctx.ops < MIN_OPS) return null
  }
  if (ctx.cols.length === 0) return null // constant expression — leave to the scalar path

  const out = cmp ? new Uint8Array(n) : new Float64Array(n)
  const kernel = compile(body, Boolean(cmp), ctx.cols.length)
  if (kernel) {
    kernel(
      ctx.cols.map((c) => c.data as ArrayLike<number>),
      roundHalfAway,
      n,
      out,
    )
  } else {
    const fn = closure(root, ctx)
    if (!fn) return null
    for (let i = 0; i < n; i++) out[i] = fn(i)
  }
  const bitmap = andBitmaps(ctx.cols, n)
  return cmp ? { kind: 'bool', data: out as Uint8Array, bitmap } : { kind: 'num', data: out as Float64Array, bitmap }
}

/** Test hook: force the closure path (as if `new Function` were blocked by CSP). */
export function __setFusedCompile(enabled: boolean | null): void {
  canCompile = enabled
  cache.clear()
}
