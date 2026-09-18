import { defineConfig } from 'vitest/config'
import base from './vitest.config.js'

// `pnpm test:interop` — the Arrow IPC interop suite (needs apache-arrow, a dev dependency of
// packages/arrow-interop only). The default `pnpm test` excludes it so the core suite stays dependency-free.
export default defineConfig({
  ...base,
  test: {
    ...base.test,
    include: ['packages/arrow-interop/tests/**/*.test.ts'],
    exclude: ['**/node_modules/**'],
    coverage: { ...base.test?.coverage, enabled: false },
  },
})
