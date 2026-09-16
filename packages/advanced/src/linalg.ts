/**
 * Small dense linear algebra for the regression modules: row-major Float64 matrices, Householder QR
 * least squares with rank detection, Cholesky, general inverse, and a one-sided Jacobi SVD.
 * Sizes are n × p with p small (tens of columns), so O(np²) algorithms are used throughout.
 */

export interface Matrix {
  rows: number
  cols: number
  /** Row-major data, length rows × cols. */
  data: Float64Array
}

export function matrix(rows: number, cols: number, data?: Float64Array): Matrix {
  return { rows, cols, data: data ?? new Float64Array(rows * cols) }
}

export function fromRows(rows: ArrayLike<number>[]): Matrix {
  const r = rows.length
  const c = r ? rows[0]!.length : 0
  const m = matrix(r, c)
  for (let i = 0; i < r; i++) {
    const row = rows[i]!
    if (row.length !== c) throw new RangeError(`ragged matrix: row ${i} has ${row.length} entries, expected ${c}`)
    for (let j = 0; j < c; j++) m.data[i * c + j] = row[j]!
  }
  return m
}

export function fromColumns(cols: ArrayLike<number>[]): Matrix {
  const c = cols.length
  const r = c ? cols[0]!.length : 0
  const m = matrix(r, c)
  for (let j = 0; j < c; j++) {
    const col = cols[j]!
    if (col.length !== r) throw new RangeError(`ragged matrix: column ${j} has ${col.length} entries, expected ${r}`)
    for (let i = 0; i < r; i++) m.data[i * c + j] = col[i]!
  }
  return m
}

export function column(m: Matrix, j: number): Float64Array {
  const out = new Float64Array(m.rows)
  for (let i = 0; i < m.rows; i++) out[i] = m.data[i * m.cols + j]!
  return out
}

export function transpose(m: Matrix): Matrix {
  const t = matrix(m.cols, m.rows)
  for (let i = 0; i < m.rows; i++) for (let j = 0; j < m.cols; j++) t.data[j * m.rows + i] = m.data[i * m.cols + j]!
  return t
}

export function matmul(a: Matrix, b: Matrix): Matrix {
  if (a.cols !== b.rows) throw new RangeError(`matmul: ${a.rows}×${a.cols} · ${b.rows}×${b.cols}`)
  const out = matrix(a.rows, b.cols)
  for (let i = 0; i < a.rows; i++) {
    for (let k = 0; k < a.cols; k++) {
      const aik = a.data[i * a.cols + k]!
      if (aik === 0) continue
      for (let j = 0; j < b.cols; j++) out.data[i * b.cols + j] += aik * b.data[k * b.cols + j]!
    }
  }
  return out
}

/** y = A·x */
export function matvec(a: Matrix, x: ArrayLike<number>): Float64Array {
  const out = new Float64Array(a.rows)
  for (let i = 0; i < a.rows; i++) {
    let s = 0
    for (let j = 0; j < a.cols; j++) s += a.data[i * a.cols + j]! * x[j]!
    out[i] = s
  }
  return out
}

/** AᵀA (p × p, symmetric). */
export function gram(a: Matrix): Matrix {
  const p = a.cols
  const out = matrix(p, p)
  for (let i = 0; i < a.rows; i++) {
    const off = i * p
    for (let j = 0; j < p; j++) {
      const v = a.data[off + j]!
      if (v === 0) continue
      for (let k = j; k < p; k++) out.data[j * p + k] += v * a.data[off + k]!
    }
  }
  for (let j = 0; j < p; j++) for (let k = 0; k < j; k++) out.data[j * p + k] = out.data[k * p + j]!
  return out
}

/** Aᵀy */
export function gramVec(a: Matrix, y: ArrayLike<number>): Float64Array {
  const out = new Float64Array(a.cols)
  for (let i = 0; i < a.rows; i++) {
    const yi = y[i]!
    for (let j = 0; j < a.cols; j++) out[j] += a.data[i * a.cols + j]! * yi
  }
  return out
}

/** Cholesky factor L (lower) of a symmetric positive-definite matrix, or null if not PD. */
export function cholesky(a: Matrix): Matrix | null {
  const n = a.rows
  const L = matrix(n, n)
  for (let j = 0; j < n; j++) {
    let d = a.data[j * n + j]!
    for (let k = 0; k < j; k++) d -= L.data[j * n + k]! ** 2
    if (!(d > 0)) return null
    const ljj = Math.sqrt(d)
    L.data[j * n + j] = ljj
    for (let i = j + 1; i < n; i++) {
      let s = a.data[i * n + j]!
      for (let k = 0; k < j; k++) s -= L.data[i * n + k]! * L.data[j * n + k]!
      L.data[i * n + j] = s / ljj
    }
  }
  return L
}

/** Solve (L Lᵀ) x = b given the Cholesky factor. */
export function choleskySolve(L: Matrix, b: ArrayLike<number>): Float64Array {
  const n = L.rows
  const y = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = b[i]!
    for (let k = 0; k < i; k++) s -= L.data[i * n + k]! * y[k]!
    y[i] = s / L.data[i * n + i]!
  }
  const x = new Float64Array(n)
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i]!
    for (let k = i + 1; k < n; k++) s -= L.data[k * n + i]! * x[k]!
    x[i] = s / L.data[i * n + i]!
  }
  return x
}

/** Inverse by Gauss–Jordan with partial pivoting; throws if singular. */
export function inverse(a: Matrix): Matrix {
  const n = a.rows
  if (a.cols !== n) throw new RangeError('inverse: matrix must be square')
  const w = new Float64Array(n * 2 * n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) w[i * 2 * n + j] = a.data[i * n + j]!
    w[i * 2 * n + n + i] = 1
  }
  const W = 2 * n
  for (let c = 0; c < n; c++) {
    let piv = c
    let best = Math.abs(w[c * W + c]!)
    for (let r = c + 1; r < n; r++) {
      const v = Math.abs(w[r * W + c]!)
      if (v > best) {
        best = v
        piv = r
      }
    }
    if (!(best > 1e-300)) throw new RangeError('inverse: singular matrix')
    if (piv !== c) for (let j = 0; j < W; j++) [w[c * W + j], w[piv * W + j]] = [w[piv * W + j]!, w[c * W + j]!]
    const d = w[c * W + c]!
    for (let j = 0; j < W; j++) w[c * W + j] = w[c * W + j]! / d
    for (let r = 0; r < n; r++) {
      if (r === c) continue
      const f = w[r * W + c]!
      if (f === 0) continue
      for (let j = 0; j < W; j++) w[r * W + j] = w[r * W + j]! - f * w[c * W + j]!
    }
  }
  const out = matrix(n, n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out.data[i * n + j] = w[i * W + n + j]!
  return out
}

export interface QRResult {
  /** Thin Q (n × p) with orthonormal columns (rank-deficient columns are still orthonormalised). */
  q: Matrix
  /** Upper-triangular R (p × p). */
  r: Matrix
  /** Numerical rank and the columns judged dependent (|r_jj| below tolerance). */
  rank: number
  dependent: number[]
}

/**
 * Householder QR of an n × p matrix (n ≥ p). Columns whose diagonal of R falls below
 * `tol · max|r_jj|` are flagged as linearly dependent (their coefficients are set to 0 by `lstsq`).
 */
export function qr(a: Matrix, tol = 1e-10): QRResult {
  const n = a.rows
  const p = a.cols
  if (n < p) throw new RangeError(`qr: needs rows ≥ cols, got ${n} × ${p}`)
  const R = matrix(n, p, Float64Array.from(a.data))
  const vs: Float64Array[] = []
  const betas: number[] = []
  for (let k = 0; k < p; k++) {
    let norm = 0
    for (let i = k; i < n; i++) norm += R.data[i * p + k]! ** 2
    norm = Math.sqrt(norm)
    const v = new Float64Array(n - k)
    const x0 = R.data[k * p + k]!
    const alpha = x0 >= 0 ? -norm : norm
    v[0] = x0 - alpha
    for (let i = k + 1; i < n; i++) v[i - k] = R.data[i * p + k]!
    let vnorm2 = 0
    for (let i = 0; i < v.length; i++) vnorm2 += v[i]! ** 2
    const beta = vnorm2 > 0 ? 2 / vnorm2 : 0
    vs.push(v)
    betas.push(beta)
    if (beta === 0) continue
    for (let j = k; j < p; j++) {
      let s = 0
      for (let i = k; i < n; i++) s += v[i - k]! * R.data[i * p + j]!
      s *= beta
      if (s === 0) continue
      for (let i = k; i < n; i++) R.data[i * p + j] = R.data[i * p + j]! - s * v[i - k]!
    }
  }
  // thin Q: apply reflectors in reverse to the first p columns of I
  const Q = matrix(n, p)
  for (let j = 0; j < p; j++) Q.data[j * p + j] = 1
  for (let k = p - 1; k >= 0; k--) {
    const v = vs[k]!
    const beta = betas[k]!
    if (beta === 0) continue
    for (let j = 0; j < p; j++) {
      let s = 0
      for (let i = k; i < n; i++) s += v[i - k]! * Q.data[i * p + j]!
      s *= beta
      if (s === 0) continue
      for (let i = k; i < n; i++) Q.data[i * p + j] = Q.data[i * p + j]! - s * v[i - k]!
    }
  }
  const Rp = matrix(p, p)
  let maxDiag = 0
  for (let i = 0; i < p; i++) {
    for (let j = i; j < p; j++) Rp.data[i * p + j] = R.data[i * p + j]!
    maxDiag = Math.max(maxDiag, Math.abs(R.data[i * p + i]!))
  }
  const dependent: number[] = []
  for (let i = 0; i < p; i++) if (!(Math.abs(Rp.data[i * p + i]!) > tol * maxDiag)) dependent.push(i)
  return { q: Q, r: Rp, rank: p - dependent.length, dependent }
}

export interface LstsqResult {
  coef: Float64Array
  /** (XᵀX)⁻¹ restricted to the independent columns (zeros elsewhere). */
  xtxInv: Matrix
  fitted: Float64Array
  residuals: Float64Array
  /** Hat-matrix diagonal. */
  leverage: Float64Array
  rank: number
  dependent: number[]
  sse: number
  /** Qᵀy for the kept (independent) columns in design order — (Qᵀy)ₖ² is the sequential SS of column k. */
  qty: Float64Array
}

/**
 * Least squares y ≈ X β via QR. Dependent columns get coefficient 0 and are removed from
 * (XᵀX)⁻¹; the remaining columns are refitted so the result is a proper least-squares solution.
 */
export function lstsq(X: Matrix, y: ArrayLike<number>, tol = 1e-10): LstsqResult {
  const n = X.rows
  const p = X.cols
  const full = qr(X, tol)
  let keep = Array.from({ length: p }, (_, i) => i)
  let Xk = X
  let f = full
  if (full.dependent.length) {
    const dep = new Set(full.dependent)
    keep = keep.filter((j) => !dep.has(j))
    Xk = matrix(n, keep.length)
    for (let i = 0; i < n; i++) for (let jj = 0; jj < keep.length; jj++) Xk.data[i * keep.length + jj] = X.data[i * p + keep[jj]!]!
    f = qr(Xk, tol)
  }
  const pk = keep.length
  // Qᵀy and back-substitution
  const qty = new Float64Array(pk)
  for (let i = 0; i < n; i++) {
    const yi = y[i]!
    for (let j = 0; j < pk; j++) qty[j] += f.q.data[i * pk + j]! * yi
  }
  const bk = new Float64Array(pk)
  for (let i = pk - 1; i >= 0; i--) {
    let s = qty[i]!
    for (let j = i + 1; j < pk; j++) s -= f.r.data[i * pk + j]! * bk[j]!
    bk[i] = s / f.r.data[i * pk + i]!
  }
  // R⁻¹ then (XᵀX)⁻¹ = R⁻¹ R⁻ᵀ
  const rinv = matrix(pk, pk)
  for (let c = 0; c < pk; c++) {
    for (let i = pk - 1; i >= 0; i--) {
      let s = i === c ? 1 : 0
      for (let j = i + 1; j < pk; j++) s -= f.r.data[i * pk + j]! * rinv.data[j * pk + c]!
      rinv.data[i * pk + c] = s / f.r.data[i * pk + i]!
    }
  }
  const invK = matrix(pk, pk)
  for (let i = 0; i < pk; i++) for (let j = 0; j < pk; j++) {
    let s = 0
    for (let k = 0; k < pk; k++) s += rinv.data[i * pk + k]! * rinv.data[j * pk + k]!
    invK.data[i * pk + j] = s
  }
  const coef = new Float64Array(p)
  const xtxInv = matrix(p, p)
  for (let i = 0; i < pk; i++) {
    coef[keep[i]!] = bk[i]!
    for (let j = 0; j < pk; j++) xtxInv.data[keep[i]! * p + keep[j]!] = invK.data[i * pk + j]!
  }
  const fitted = matvec(X, coef)
  const residuals = new Float64Array(n)
  const leverage = new Float64Array(n)
  let sse = 0
  for (let i = 0; i < n; i++) {
    residuals[i] = y[i]! - fitted[i]!
    sse += residuals[i]! ** 2
    let h = 0
    for (let j = 0; j < pk; j++) h += f.q.data[i * pk + j]! ** 2
    leverage[i] = h
  }
  return { coef, xtxInv, fitted, residuals, leverage, rank: pk, dependent: full.dependent, sse, qty }
}

export interface SVDResult {
  /** U (n × k), singular values (k, descending), V (p × k) with k = min(n, p). */
  u: Matrix
  s: Float64Array
  v: Matrix
}

/** One-sided Jacobi SVD (Hestenes) — accurate for the small matrices used here. */
export function svd(a: Matrix, maxSweeps = 60): SVDResult {
  const transposed = a.rows < a.cols
  const A = transposed ? transpose(a) : a
  const n = A.rows
  const p = A.cols
  const U = matrix(n, p, Float64Array.from(A.data))
  const V = matrix(p, p)
  for (let i = 0; i < p; i++) V.data[i * p + i] = 1
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0
    for (let j = 0; j < p - 1; j++) {
      for (let k = j + 1; k < p; k++) {
        let alpha = 0
        let beta = 0
        let gamma = 0
        for (let i = 0; i < n; i++) {
          const uj = U.data[i * p + j]!
          const uk = U.data[i * p + k]!
          alpha += uj * uj
          beta += uk * uk
          gamma += uj * uk
        }
        if (Math.abs(gamma) <= 1e-15 * Math.sqrt(alpha * beta) || gamma === 0) continue
        off = Math.max(off, Math.abs(gamma) / Math.sqrt(alpha * beta))
        const zeta = (beta - alpha) / (2 * gamma)
        const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta))
        const c = 1 / Math.sqrt(1 + t * t)
        const s = c * t
        for (let i = 0; i < n; i++) {
          const uj = U.data[i * p + j]!
          const uk = U.data[i * p + k]!
          U.data[i * p + j] = c * uj - s * uk
          U.data[i * p + k] = s * uj + c * uk
        }
        for (let i = 0; i < p; i++) {
          const vj = V.data[i * p + j]!
          const vk = V.data[i * p + k]!
          V.data[i * p + j] = c * vj - s * vk
          V.data[i * p + k] = s * vj + c * vk
        }
      }
    }
    if (off < 1e-14) break
  }
  const sv = new Float64Array(p)
  for (let j = 0; j < p; j++) {
    let s = 0
    for (let i = 0; i < n; i++) s += U.data[i * p + j]! ** 2
    sv[j] = Math.sqrt(s)
    if (sv[j]! > 0) for (let i = 0; i < n; i++) U.data[i * p + j] = U.data[i * p + j]! / sv[j]!
  }
  // sort descending
  const order = Array.from({ length: p }, (_, i) => i).sort((x, y) => sv[y]! - sv[x]!)
  const Us = matrix(n, p)
  const Vs = matrix(p, p)
  const ss = new Float64Array(p)
  order.forEach((src, dst) => {
    ss[dst] = sv[src]!
    for (let i = 0; i < n; i++) Us.data[i * p + dst] = U.data[i * p + src]!
    for (let i = 0; i < p; i++) Vs.data[i * p + dst] = V.data[i * p + src]!
  })
  return transposed ? { u: Vs, s: ss, v: Us } : { u: Us, s: ss, v: Vs }
}

export interface EigenResult {
  /** Eigenvalues, descending. */
  values: Float64Array
  /** Eigenvectors as columns of a p × p matrix (same order as values). */
  vectors: Matrix
}

/** Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations (p small). */
export function symmetricEigen(S: Matrix): EigenResult {
  const p = S.rows
  const A = matrix(p, p, Float64Array.from(S.data))
  const V = matrix(p, p)
  for (let i = 0; i < p; i++) V.data[i * p + i] = 1
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0
    for (let i = 0; i < p; i++) for (let j = i + 1; j < p; j++) off += A.data[i * p + j]! ** 2
    if (off < 1e-30) break
    for (let i = 0; i < p - 1; i++) {
      for (let j = i + 1; j < p; j++) {
        const aij = A.data[i * p + j]!
        if (Math.abs(aij) < 1e-300) continue
        const aii = A.data[i * p + i]!
        const ajj = A.data[j * p + j]!
        const tau = (ajj - aii) / (2 * aij)
        const t = Math.sign(tau || 1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau))
        const c = 1 / Math.sqrt(1 + t * t)
        const sn = t * c
        for (let k = 0; k < p; k++) {
          const aik = A.data[i * p + k]!
          const ajk = A.data[j * p + k]!
          A.data[i * p + k] = c * aik - sn * ajk
          A.data[j * p + k] = sn * aik + c * ajk
        }
        for (let k = 0; k < p; k++) {
          const aki = A.data[k * p + i]!
          const akj = A.data[k * p + j]!
          A.data[k * p + i] = c * aki - sn * akj
          A.data[k * p + j] = sn * aki + c * akj
        }
        for (let k = 0; k < p; k++) {
          const vki = V.data[k * p + i]!
          const vkj = V.data[k * p + j]!
          V.data[k * p + i] = c * vki - sn * vkj
          V.data[k * p + j] = sn * vki + c * vkj
        }
      }
    }
  }
  const order = Array.from({ length: p }, (_, i) => i).sort((a, b) => A.data[b * p + b]! - A.data[a * p + a]!)
  const values = Float64Array.from(order, (i) => A.data[i * p + i]!)
  const vectors = matrix(p, p)
  order.forEach((src, dst) => {
    // sign convention: largest-magnitude component positive
    let big = 0
    for (let k = 0; k < p; k++) if (Math.abs(V.data[k * p + src]!) > Math.abs(big)) big = V.data[k * p + src]!
    const sgn = big < 0 ? -1 : 1
    for (let k = 0; k < p; k++) vectors.data[k * p + dst] = sgn * V.data[k * p + src]!
  })
  return { values, vectors }
}

/** Alias for `symmetricEigen` (NumPy `eigh` style). */
export const eigh = symmetricEigen

/**
 * Non-negative least squares (Lawson–Hanson active-set MVP).
 * Solves min ‖Ax − b‖² s.t. x ≥ 0.
 */
export function nnls(A: Matrix, b: ArrayLike<number>, maxIter = 200): { x: Float64Array; residual: Float64Array } {
  const m = A.rows
  const n = A.cols
  const bb = Float64Array.from({ length: m }, (_, i) => Number(b[i]))
  const x = new Float64Array(n)
  const P = new Set<number>()
  const resid = new Float64Array(m)
  const updateResid = () => {
    for (let i = 0; i < m; i++) {
      let s = bb[i]!
      for (let j = 0; j < n; j++) s -= A.data[i * n + j]! * x[j]!
      resid[i] = s
    }
  }
  updateResid()
  for (let iter = 0; iter < maxIter; iter++) {
    const w = new Float64Array(n)
    for (let j = 0; j < n; j++) {
      let s = 0
      for (let i = 0; i < m; i++) s += A.data[i * n + j]! * resid[i]!
      w[j] = s
    }
    let maxW = -Infinity
    let t = -1
    for (let j = 0; j < n; j++) {
      if (!P.has(j) && w[j]! > maxW) {
        maxW = w[j]!
        t = j
      }
    }
    if (t < 0 || maxW <= 1e-12) break
    P.add(t)
    for (;;) {
      const cols = [...P]
      const k = cols.length
      const sub = matrix(m, k)
      for (let i = 0; i < m; i++) for (let c = 0; c < k; c++) sub.data[i * k + c] = A.data[i * n + cols[c]!]!
      const sol = lstsq(sub, bb)
      const sVec = new Float64Array(n)
      for (let c = 0; c < k; c++) sVec[cols[c]!] = sol.coef[c]!
      if (cols.every((j) => sVec[j]! > 1e-14)) {
        for (let j = 0; j < n; j++) x[j] = Math.max(0, sVec[j]!)
        break
      }
      let alpha = 1
      let drop = -1
      for (const j of cols) {
        if (sVec[j]! <= 0 && x[j]! > 0) {
          const a = x[j]! / (x[j]! - sVec[j]!)
          if (a < alpha) {
            alpha = a
            drop = j
          }
        }
      }
      for (let j = 0; j < n; j++) x[j] = x[j]! + alpha * (sVec[j]! - x[j]!)
      if (drop >= 0) {
        P.delete(drop)
        x[drop] = 0
      }
      for (const j of [...P]) {
        if (x[j]! <= 1e-14) {
          x[j] = 0
          P.delete(j)
        }
      }
      if (P.size === 0) break
    }
    updateResid()
  }
  updateResid()
  return { x, residual: resid }
}
