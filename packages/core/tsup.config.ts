import { defineConfig } from 'tsup'

export default defineConfig({
  // Node-only accelerators are separate output files next to index.js so the runtime's opaque
  // `import('./csv-parallel.js')` resolves there; they are never inlined into the main bundle.
  entry: { index: 'src/index.ts', 'csv-parallel': 'src/io/csv-parallel.ts', 'csv-worker': 'src/io/csv-worker.ts' },
  format: ['esm', 'cjs'],
  dts: { entry: { index: 'src/index.ts' } },
  clean: true,
  external: ['pg', 'mssql', '@clickhouse/client', 'mysql2', 'better-sqlite3', 'kafkajs', '@columna/native'],
})
