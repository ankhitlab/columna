/**
 * Compare join/groupby with native on vs off at 1M.
 */
import { pathToFileURL } from 'node:url'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const { init, DataFrame, isNativeKernelsLoaded, setNativeKernels } = await import(
  pathToFileURL(join(root, 'packages/columna/dist/index.js')).href
)

const N = 1_000_000
const user_id = new Int32Array(N)
const salary = new Float64Array(N)
const region = new Uint32Array(N)
const id = new Int32Array(N)
for (let i = 0; i < N; i++) {
  id[i] = i
  user_id[i] = (i * 7) % 100_000
  salary[i] = 40_000 + (i % 50_000)
  region[i] = i % 8
}
const left = DataFrame.fromColumns({
  id,
  user_id,
  salary,
  region: { codes: region, dictionary: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] },
})
const rIds = new Int32Array(100_000)
const score = new Float64Array(100_000)
for (let i = 0; i < 100_000; i++) {
  rIds[i] = i
  score[i] = i
}
const right = DataFrame.fromColumns({ user_id: rIds, score })

async function med(label, fn, runs = 5) {
  await fn()
  const s = []
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now()
    await fn()
    s.push(performance.now() - t0)
  }
  s.sort((a, b) => a - b)
  console.log(`${label}: ${s[Math.floor(s.length / 2)].toFixed(2)} ms`)
}

await init({ native: true })
console.log('native:', isNativeKernelsLoaded())
await med('join native', () => left.innerJoin(right, 'user_id').collect())
await med('semi native', () => left.join(right, { on: 'user_id', how: 'semi' }).collect())
await med('groupby native', () => left.groupBy('region').agg({ salary: 'mean', id: 'count' }).collect())

// correctness vs JS: disable kernels
setNativeKernels({})
console.log('native after clear:', isNativeKernelsLoaded())
const jN = await left.innerJoin(right, 'user_id').collect()
await init({ native: true })
const jY = await left.innerJoin(right, 'user_id').collect()
console.log('join rows match', jN.shape[0] === jY.shape[0], jN.shape[0])

setNativeKernels({})
const gN = await left.groupBy('region').agg({ salary: 'mean', id: 'count' }).collect()
await init({ native: true })
const gY = await left.groupBy('region').agg({ salary: 'mean', id: 'count' }).collect()
const aN = gN.getColumn('salary').toArray()
const aY = gY.getColumn('salary').toArray()
let ok = aN.length === aY.length
for (let i = 0; i < aN.length; i++) if (Math.abs(aN[i] - aY[i]) > 1e-6) ok = false
console.log('groupby mean match', ok)

setNativeKernels({})
await med('join js', () => left.innerJoin(right, 'user_id').collect())
await med('semi js', () => left.join(right, { on: 'user_id', how: 'semi' }).collect())
await med('groupby js', () => left.groupBy('region').agg({ salary: 'mean', id: 'count' }).collect())

const require = createRequire(import.meta.url)
const native = require(join(root, 'packages/native/index.js'))
console.log('kernels:', Object.keys(native).filter((k) => typeof native[k] === 'function').join(', '))
