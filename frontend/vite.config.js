import { defineConfig } from 'vite'

// Minimal Vite config. We avoid configuring esbuild loaders manually so the
// default JSX handling works reliably across environments.
export default defineConfig({
  server: {
    port: 3000,
    strictPort: false,
  },
})
