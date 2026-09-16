/**
 * Discrete Fourier transform of a real sequence of any length: iterative radix-2 FFT for powers of
 * two, Bluestein's chirp-z algorithm otherwise (O(n log n)). Returns re / im arrays of length n.
 */

function fftPow2(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length
  // bit reversal
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      ;[re[i], re[j]] = [re[j]!, re[i]!]
      ;[im[i], im[j]] = [im[j]!, im[i]!]
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len
    const wr = Math.cos(ang)
    const wi = Math.sin(ang)
    for (let i = 0; i < n; i += len) {
      let cr = 1
      let ci = 0
      const half = len >> 1
      for (let j = 0; j < half; j++) {
        const a = i + j
        const b = a + half
        const xr = re[b]! * cr - im[b]! * ci
        const xi = re[b]! * ci + im[b]! * cr
        re[b] = re[a]! - xr
        im[b] = im[a]! - xi
        re[a] = re[a]! + xr
        im[a] = im[a]! + xi
        const ncr = cr * wr - ci * wi
        ci = cr * wi + ci * wr
        cr = ncr
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) {
    re[i] = re[i]! / n
    im[i] = im[i]! / n
  }
}

/** X_k = Σ x_t e^{−2πi k t / n}, k = 0 … n − 1. */
export function dft(x: ArrayLike<number>): { re: Float64Array; im: Float64Array } {
  const n = x.length
  if (n === 0) return { re: new Float64Array(0), im: new Float64Array(0) }
  if ((n & (n - 1)) === 0) {
    const re = Float64Array.from(x)
    const im = new Float64Array(n)
    fftPow2(re, im)
    return { re, im }
  }
  // Bluestein: X_k = conj(w_k) · Σ (x_t w_t) · w*_{k−t}, with w_t = e^{−iπ t²/n}
  let m = 1
  while (m < 2 * n - 1) m <<= 1
  const wr = new Float64Array(n)
  const wi = new Float64Array(n)
  for (let t = 0; t < n; t++) {
    const ang = (-Math.PI * ((t * t) % (2 * n))) / n
    wr[t] = Math.cos(ang)
    wi[t] = Math.sin(ang)
  }
  const ar = new Float64Array(m)
  const ai = new Float64Array(m)
  for (let t = 0; t < n; t++) {
    ar[t] = x[t]! * wr[t]!
    ai[t] = x[t]! * wi[t]!
  }
  const br = new Float64Array(m)
  const bi = new Float64Array(m)
  br[0] = wr[0]!
  bi[0] = -wi[0]!
  for (let t = 1; t < n; t++) {
    br[t] = wr[t]!
    bi[t] = -wi[t]!
    br[m - t] = wr[t]!
    bi[m - t] = -wi[t]!
  }
  fftPow2(ar, ai)
  fftPow2(br, bi)
  for (let i = 0; i < m; i++) {
    const r = ar[i]! * br[i]! - ai[i]! * bi[i]!
    const im2 = ar[i]! * bi[i]! + ai[i]! * br[i]!
    ar[i] = r
    ai[i] = im2
  }
  fftPow2(ar, ai, true)
  const re = new Float64Array(n)
  const im = new Float64Array(n)
  for (let k = 0; k < n; k++) {
    re[k] = ar[k]! * wr[k]! - ai[k]! * wi[k]!
    im[k] = ar[k]! * wi[k]! + ai[k]! * wr[k]!
  }
  return { re, im }
}
