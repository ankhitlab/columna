import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)

/** @type {typeof import('./index.js')} */
let native
try {
  native = require('./index.js')
} catch {
  native = {}
}

export const filterAnd2I32F64 = native.filterAnd2I32F64
export const filterAnd2I32I32 = native.filterAnd2I32I32
export const gatherF64 = native.gatherF64
export const gatherI32 = native.gatherI32
export const joinProbeDenseI32 = native.joinProbeDenseI32
export const joinSemiDenseI32 = native.joinSemiDenseI32
export const groupbySumsF64 = native.groupbySumsF64
export const strContains = native.strContains
export const strToLower = native.strToLower
export const parseCsvUnquoted = native.parseCsvUnquoted
export const writeCsvUnquoted = native.writeCsvUnquoted
export const isNativeLoaded = typeof native.filterAnd2I32F64 === 'function'
