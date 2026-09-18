import { defineConfig } from 'vitest/config'
import { resolve } from 'node:path'

export default defineConfig({
  test: {
    globals: false,
    environment: 'node',
    include: ['packages/**/src/**/*.test.ts', 'packages/**/tests/**/*.test.ts'],
    coverage: {
      // `pnpm test:coverage` — CI prints the summary into the job summary and uploads the full report.
      provider: 'v8',
      include: ['packages/*/src/**/*.ts'],
      exclude: ['packages/bench/**', 'packages/browser-smoke/**', 'packages/studio/**', 'packages/native/**', '**/*.test.ts', '**/*.d.ts'],
      reporter: ['text-summary', 'json-summary', 'html'],
      reportsDirectory: 'coverage',
    },
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
