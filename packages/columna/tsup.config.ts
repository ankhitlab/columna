import { defineConfig } from 'tsup'

/** Bundle workspace packages into the public `columna` tarball. */
const workspace = [/^@columna\//]

/** Stay external — dynamic optional peers / heavy IO deps / native addon. */
const external = [
  'hyparquet',
  'hyparquet-compressors',
  'xlsx',
  'pg',
  'mssql',
  '@clickhouse/client',
  'mysql2',
  'mysql2/promise',
  'better-sqlite3',
  'kafkajs',
  '@columna/native',
]

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    core: 'src/core.ts',
    advanced: 'src/advanced.ts',
    // Sibling modules required by runtime dynamic imports after bundling.
    'parallel-node': '../runtime/src/parallel-node.ts',
    'filter-worker': '../runtime/src/filter-worker.ts',
    'csv-worker': '../core/src/io/csv-worker.ts',
    'csv-parallel': '../core/src/io/csv-parallel.ts',
  },
  format: ['esm', 'cjs'],
  dts: {
    // Only generate declarations for public entry points.
    entry: {
      index: 'src/index.ts',
      core: 'src/core.ts',
      advanced: 'src/advanced.ts',
    },
    resolve: true,
  },
  clean: true,
  sourcemap: true,
  splitting: false,
  noExternal: workspace,
  external,
})
