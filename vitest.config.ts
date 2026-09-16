import { defineConfig } from 'vitest/config'
import { resolve } from 'node:path'

export default defineConfig({
  test: {
    globals: false,
    environment: 'node',
    include: ['packages/**/src/**/*.test.ts', 'packages/**/tests/**/*.test.ts'],
  },
  resolve: {
    alias: {
      '@columna/arrow': resolve(__dirname, 'packages/arrow/src/index.ts'),
      '@columna/core': resolve(__dirname, 'packages/core/src/index.ts'),
      '@columna/advanced': resolve(__dirname, 'packages/advanced/src/index.ts'),
      '@columna/runtime': resolve(__dirname, 'packages/runtime/src/index.ts'),
      '@columna/wasm': resolve(__dirname, 'packages/wasm/src/index.ts'),
      '@columna/webgpu': resolve(__dirname, 'packages/webgpu/src/index.ts'),
      columna: resolve(__dirname, 'packages/columna/src/index.ts'),
    },
  },
})
