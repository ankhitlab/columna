import type { MathOp } from './types.js'

/** Round half away from zero (Minitab / Excel ROUND); the (1+ε) nudge absorbs binary drift like 1.005·100 = 100.49999…. */
export function roundHalfAway(v: number, decimals: number): number {
  if (!Number.isFinite(v)) return v
  const f = decimals ? 10 ** decimals : 1
  const r = Math.round(Math.abs(v) * f * (1 + Number.EPSILON)) / f
  return v < 0 ? -r : r
}

/** Scalar kernel for MathOp; mirrors the vector loops in evalVec. */
export function applyMathOp(op: MathOp, v: number, decimals = 0, base?: number): number {
  switch (op) {
    case 'sqrt':
      return Math.sqrt(v)
    case 'log':
      return base === undefined ? Math.log(v) : Math.log(v) / Math.log(base)
    case 'log10':
      return Math.log10(v)
    case 'log2':
      return Math.log2(v)
    case 'exp':
      return Math.exp(v)
    case 'round':
      return roundHalfAway(v, decimals)
    case 'floor':
      return Math.floor(v)
    case 'ceil':
      return Math.ceil(v)
    case 'sign':
      return Math.sign(v)
  }
}
