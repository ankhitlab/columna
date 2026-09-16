/**
 * Browser stub — Node worker_threads are unavailable in Vite builds.
 * Sync dualGtIndices is used by parallel.ts when this module is not loaded;
 * this stub exists so Rollup can resolve the dynamic import target.
 */
type NumArr = Float64Array | Float32Array | Int32Array | Uint32Array

export async function runParallelDualGt(
  _a: NumArr,
  _b: NumArr,
  _la: number,
  _lb: number,
): Promise<Uint32Array> {
  throw new Error('parallel-node is Node-only')
}
