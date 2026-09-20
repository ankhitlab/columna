import { executeCpu, getDefaultRuntime, Runtime, tryLoadNativeKernels } from '@columna/runtime'
import { WasmBackend, tryLoadRustKernels } from '@columna/wasm'
import { WebGpuBackend } from '@columna/webgpu'

export { DataFrame, LazyFrame, GroupBy, Series, Expr, col, cols, lit, aggExpr, when, dt, daysBetween } from '@columna/core'
// IO policy / explicit sources / SQL client ownership — the boundaries a server needs, same surface as @columna/core
export { openSqlClient, setIoPolicy, getIoPolicy, io } from '@columna/core'
export type { IoPolicy, IoLoadOptions, IoSourceMode, CsvWriteOptions, Row, JoinResult, InferColumns, AnyExpr, ColRefs, DTypeValue } from '@columna/core'
export type {
  AggSpec,
  IoSource,
  KafkaClient,
  KafkaCodec,
  KafkaConnection,
  KafkaFlattenOptions,
  KafkaMetaField,
  KafkaRawMessage,
  KafkaSaslOptions,
  ReadCsvOptions,
  ReadExcelOptions,
  ReadJsonOptions,
  ReadKafkaOptions,
  ReadParquetOptions,
  ReadSqlOptions,
  SqlClient,
  SqlConnection,
  SqlConnectionConfig,
  SqlDialect,
  SqlParams,
} from '@columna/core'
export type { ArrowLike, ArrowIpcWriteOptions, DType, Schema, TableView } from '@columna/arrow'
export { toArrowIpc, fromArrowIpc } from '@columna/arrow'
export type { AggKind, CorrMethod, EngineKind, ExecuteOptions, ExecutionEvent, ExecutionReport, JoinKind, MathOp, PlanNode, RankMethod } from '@columna/runtime'
export { Session, createSession, type SessionOptions } from '@columna/core'
export {
  EngineStrictError,
  ExecutionAbortedError,
  formatExecutionReport,
  PersistCache,
  type PersistCacheOptions,
  type RuntimeOptions,
  Runtime,
  getDefaultRuntime,
  setDefaultRuntime,
  CpuBackend,
  executeCpu,
  optimizePlan,
  estimatePlanRows,
  tryLoadNativeKernels,
  isNativeKernelsLoaded,
  closeParallelPool,
  setNativeKernels,
  setMemoryPolicy,
  getMemoryPolicy,
  clearMemoryPolicy,
  clearPersistCache,
  persistCacheStats,
  estimateTableBytes,
  type MemoryPolicy,
} from '@columna/runtime'
export { WasmBackend, tryLoadRustKernels, writeParquetLike, readParquetLike, wasmStringContains, wasmStringLength } from '@columna/wasm'
export { WebGpuBackend, detectWebGPU } from '@columna/webgpu'
export { isRustKernelsLoaded } from '@columna/wasm'

let bootstrapped = false
let webgpuBackend: WebGpuBackend | null = null

/**
 * Start the optional infrastructure — this is the only place it starts. Importing `columna` has no side
 * effects beyond registering inert backend objects; `init()` requests the WebGPU adapter / device, loads the
 * native Node addon and the Rust WASM kernels when they are available, and resolves when all of that is
 * settled. Without it the CPU engine handles everything, deterministically.
 * `gpuLossyF32: true` lets the WebGPU filter / map kernels run f64 and datetime columns (and literals the
 * column type cannot represent) in float32 — faster, but the GPU may then return different rows than the
 * CPU (16 777 217 rounds to 16 777 216). Off by default: results are identical across backends.
 */
export async function init(options: { rust?: boolean; native?: boolean; gpuLossyF32?: boolean } = {}): Promise<Runtime> {
  const runtime = getDefaultRuntime()
  if (webgpuBackend && options.gpuLossyF32 !== undefined) webgpuBackend.options.lossyF32 = options.gpuLossyF32
  if (!bootstrapped) {
    runtime.register(new WasmBackend(executeCpu))
    webgpuBackend = new WebGpuBackend(executeCpu)
    runtime.register(webgpuBackend)
    bootstrapped = true
  }
  if (webgpuBackend) await webgpuBackend.waitReady()
  if (options.native !== false) {
    await tryLoadNativeKernels()
  }
  if (options.rust !== false) {
    await tryLoadRustKernels()
  }
  return runtime
}

/** Whether WebGPU device was acquired (call after init). */
export async function isWebGpuAvailable(): Promise<boolean> {
  if (!webgpuBackend) return false
  return webgpuBackend.waitReady()
}

// Registration at import is inert: constructing the backends allocates two objects and nothing else.
// No adapter request, no worker, no native module load happens until `init()` — until then every plan
// runs on the CPU engine, and `engine('webgpu')` without `init()` reports "does not support this plan".
getDefaultRuntime().register(new WasmBackend(executeCpu))
webgpuBackend = new WebGpuBackend(executeCpu)
getDefaultRuntime().register(webgpuBackend)
bootstrapped = true
