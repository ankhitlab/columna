import { executeCpu, getDefaultRuntime, Runtime, tryLoadNativeKernels } from '@columna/runtime'
import { WasmBackend, tryLoadRustKernels } from '@columna/wasm'
import { WebGpuBackend } from '@columna/webgpu'

export { DataFrame, LazyFrame, GroupBy, Series, Expr, col, lit, aggExpr, when, dt, daysBetween } from '@columna/core'
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
export type { ArrowLike, DType, Schema, TableView } from '@columna/arrow'
export type { AggKind, CorrMethod, EngineKind, JoinKind, MathOp, PlanNode, RankMethod } from '@columna/runtime'
export {
  Runtime,
  getDefaultRuntime,
  setDefaultRuntime,
  CpuBackend,
  executeCpu,
  tryLoadNativeKernels,
  isNativeKernelsLoaded,
  setNativeKernels,
} from '@columna/runtime'
export { WasmBackend, tryLoadRustKernels, writeParquetLike, readParquetLike, wasmStringContains, wasmStringLength } from '@columna/wasm'
export { WebGpuBackend, detectWebGPU } from '@columna/webgpu'
export { isRustKernelsLoaded } from '@columna/wasm'

let bootstrapped = false
let webgpuBackend: WebGpuBackend | null = null

/** Register WASM + WebGPU backends on the default runtime (idempotent). */
export async function init(options: { rust?: boolean; native?: boolean } = {}): Promise<Runtime> {
  const runtime = getDefaultRuntime()
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

// Eager sync registration so `import 'columna'` already has backends.
getDefaultRuntime().register(new WasmBackend(executeCpu))
webgpuBackend = new WebGpuBackend(executeCpu)
getDefaultRuntime().register(webgpuBackend)
bootstrapped = true
