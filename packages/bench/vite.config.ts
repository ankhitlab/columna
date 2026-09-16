import { defineConfig } from 'vite'
import { resolve } from 'node:path'

const root = resolve(__dirname, '../..')

export default defineConfig({
  root: resolve(__dirname, 'web'),
  server: { port: 5179, open: true },
  resolve: {
    alias: {
      '@columna/arrow': resolve(root, 'packages/arrow/src/index.ts'),
      '@columna/core': resolve(root, 'packages/core/src/index.ts'),
      '@columna/runtime': resolve(root, 'packages/runtime/src/index.ts'),
      '@columna/wasm': resolve(root, 'packages/wasm/src/index.ts'),
      '@columna/webgpu': resolve(root, 'packages/webgpu/src/index.ts'),
      columna: resolve(root, 'packages/columna/src/index.ts'),
    },
  },
})
