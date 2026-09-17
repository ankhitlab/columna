import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'node:path'

/** Redirect Node-only parallel-node into a browser stub so Vite/Rollup never sees node:*. */
function stubParallelNode(): Plugin {
  const stub = resolve(__dirname, 'src/stubs/parallel-node.ts')
  return {
    name: 'stub-parallel-node',
    enforce: 'pre',
    resolveId(id, importer) {
      if (!id.includes('parallel-node')) return null
      if (importer && (importer.includes('runtime') || importer.includes('parallel'))) {
        return stub
      }
      if (id.endsWith('parallel-node.ts') || id.endsWith('parallel-node.js')) {
        return stub
      }
      return null
    },
  }
}

export default defineConfig({
  plugins: [stubParallelNode(), react()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      columna: resolve(__dirname, '../columna/src/index.ts'),
      '@columna/arrow': resolve(__dirname, '../arrow/src/index.ts'),
      '@columna/core': resolve(__dirname, '../core/src/index.ts'),
      '@columna/runtime': resolve(__dirname, '../runtime/src/index.ts'),
      '@columna/wasm': resolve(__dirname, '../wasm/src/index.ts'),
      '@columna/webgpu': resolve(__dirname, '../webgpu/src/index.ts'),
    },
  },
  server: {
    port: 5173,
    open: true,
    // Studio executes whatever is typed into it with the page's privileges. Keep the dev server on
    // loopback; exposing it (`vite --host`) makes every device on the network a local user.
    host: '127.0.0.1',
    strictPort: false,
  },
  optimizeDeps: {
    exclude: ['columna'],
  },
  build: {
    rollupOptions: {
      // optional rust artifact may be missing
      external: [/columna_wasm/],
    },
  },
})
