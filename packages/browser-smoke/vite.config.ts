import { defineConfig } from 'vite'

// No aliases on purpose: `columna` resolves through the workspace package's `exports` to its built dist,
// i.e. exactly what an npm consumer bundles.
export default defineConfig({
  build: { outDir: 'dist', target: 'es2022', minify: false, sourcemap: false },
})
