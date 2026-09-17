export {
  setMemoryPolicy,
  getMemoryPolicy,
  clearMemoryPolicy,
  type MemoryPolicy,
} from './memory.js'

/** Re-export estimate helper from arrow for callers that only depend on runtime. */
export { estimateTableBytes as estimateTableBytesReexport } from '@columna/arrow'
export { estimateTableBytes } from '@columna/arrow'
