/**
 * Optional native Rayon kernels (Node). Safe no-op when the addon is missing.
 */
type DualI32F64 = (
  a: Int32Array,
  b: Float64Array,
  opA: number,
  litA: number,
  opB: number,
  litB: number,
) => Uint32Array

type DualI32I32 = (
  a: Int32Array,
  b: Int32Array,
  opA: number,
  litA: number,
  opB: number,
  litB: number,
) => Uint32Array

type GatherF64Fn = (src: Float64Array, indices: Uint32Array) => Float64Array
type GatherI32Fn = (src: Int32Array, indices: Uint32Array) => Int32Array
type JoinProbeFn = (leftKeys: Int32Array, dense: Int32Array, rMin: number) => Int32Array
type JoinSemiFn = (leftKeys: Int32Array, dense: Int32Array, rMin: number, wantHit: boolean) => Uint32Array
type GroupSumsFn = (
  codes: Uint32Array,
  card: number,
  cols: Float64Array[],
) => { sums: Float64Array; counts: Float64Array; used: Uint8Array }
type StrContainsFn = (values: string[], needle: string) => Uint8Array
type StrToLowerFn = (values: string[]) => string[]

type DualF64F64 = (
  a: Float64Array,
  b: Float64Array,
  opA: number,
  litA: number,
  opB: number,
  litB: number,
) => Uint32Array

type ArgsortF64Fn = (values: Float64Array, descending: boolean, nullsLast: boolean) => Uint32Array
type ArgsortI32Fn = (
  values: Int32Array,
  descending: boolean,
  nullsLast: boolean,
  nullBitmap?: Uint8Array,
) => Uint32Array
type ArgsortMultiF64Fn = (
  keys: Float64Array[],
  descending: boolean[],
  nullsLast: boolean[],
) => Uint32Array
type FilterF64Fn = (cols: Float64Array[], ops: number[], lits: number[]) => Uint32Array
type UniqueF64Fn = (cols: Float64Array[]) => Uint32Array
type JoinBuildDenseI32Fn = (rightKeys: Int32Array, rMin: number) => Int32Array
type GroupMinMaxFn = (
  codes: Uint32Array,
  card: number,
  cols: Float64Array[],
) => { mins: Float64Array; maxs: Float64Array; counts: Float64Array; used: Uint8Array }

export type NativeKernels = {
  filterAnd2I32F64?: DualI32F64
  filterAnd2I32I32?: DualI32I32
  filterAnd2F64F64?: DualF64F64
  filterF64?: FilterF64Fn
  gatherF64?: GatherF64Fn
  gatherI32?: GatherI32Fn
  joinProbeDenseI32?: JoinProbeFn
  joinSemiDenseI32?: JoinSemiFn
  joinBuildDenseI32?: JoinBuildDenseI32Fn
  groupbySumsF64?: GroupSumsFn
  groupbyMinmaxF64?: GroupMinMaxFn
  argsortF64?: ArgsortF64Fn
  argsortI32?: ArgsortI32Fn
  argsortMultiF64?: ArgsortMultiF64Fn
  uniqueF64?: UniqueF64Fn
  strContains?: StrContainsFn
  strToLower?: StrToLowerFn
}

let kernels: NativeKernels = {}
let loaded = false

export function setNativeKernels(next: NativeKernels): void {
  kernels = next
  loaded = Boolean(next.filterAnd2I32F64)
}

export function getNativeKernels(): NativeKernels {
  return kernels
}

export function isNativeKernelsLoaded(): boolean {
  return loaded
}

/** Try dynamic import of `@columna/native` (optional dependency). */
export async function tryLoadNativeKernels(): Promise<boolean> {
  try {
    // Keep the module id unbundled (optional .node addon).
    const id = '@columna/' + 'native'
    const mod = (await import(/* webpackIgnore: true */ /* @vite-ignore */ id)) as NativeKernels & {
      isNativeLoaded?: boolean
    }
    if (mod.isNativeLoaded && typeof mod.filterAnd2I32F64 === 'function') {
      setNativeKernels({
        filterAnd2I32F64: mod.filterAnd2I32F64,
        filterAnd2I32I32: mod.filterAnd2I32I32,
        filterAnd2F64F64: (mod as NativeKernels).filterAnd2F64F64,
        filterF64: (mod as NativeKernels).filterF64,
        gatherF64: mod.gatherF64,
        gatherI32: mod.gatherI32,
        joinProbeDenseI32: mod.joinProbeDenseI32,
        joinSemiDenseI32: mod.joinSemiDenseI32,
        joinBuildDenseI32: (mod as NativeKernels).joinBuildDenseI32,
        groupbySumsF64: mod.groupbySumsF64,
        groupbyMinmaxF64: (mod as NativeKernels).groupbyMinmaxF64,
        argsortF64: (mod as NativeKernels).argsortF64,
        argsortI32: (mod as NativeKernels).argsortI32,
        argsortMultiF64: (mod as NativeKernels).argsortMultiF64,
        uniqueF64: (mod as NativeKernels).uniqueF64,
        strContains: mod.strContains,
        strToLower: mod.strToLower,
      })
      return true
    }
  } catch {
    // optional
  }
  loaded = false
  kernels = {}
  return false
}

/** Min rows before native dual-gt is preferred over JS (amortize napi call). */
export const NATIVE_FILTER_MIN_ROWS = 1_000_000

/** Min take size before native gather is used for typed columns. */
export const NATIVE_GATHER_MIN_ROWS = 50_000

/** Min left rows for native dense join probe / semi. */
export const NATIVE_JOIN_MIN_ROWS = 500_000

/** Min rows for native dense groupby sum/mean/count. */
export const NATIVE_GROUPBY_MIN_ROWS = 500_000

/** Min rows for native single-key numeric argsort. */
export const NATIVE_SORT_MIN_ROWS = 500_000

/** Min rows for native multi-key f64 argsort. */
export const NATIVE_SORT_MULTI_MIN_ROWS = 500_000

/** Min rows for native generic multi-column f64 filter. */
export const NATIVE_FILTER_GENERIC_MIN_ROWS = 1_000_000

/** Min rows for native parallel unique (f64). */
export const NATIVE_UNIQUE_MIN_ROWS = 1_000_000

/** Min right rows for native hash-join build (dense probe table). */
export const NATIVE_JOIN_BUILD_MIN_ROWS = 200_000

/** Min utf8 rows before native str contains/lower.
 * String→Rust copy usually dominates; JS wins on short/interned utf8. Keep very high. */
export const NATIVE_STR_MIN_ROWS = 10_000_000
