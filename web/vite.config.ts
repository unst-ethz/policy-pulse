import { fileURLToPath, URL } from 'node:url'
import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vitest/config'

// The API (`uv run start-api`) runs on :8000; in development the browser talks to Vite only and
// /api is proxied, so there is no CORS setup to get wrong and production can serve both under
// one origin.
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    port: 5173,
    proxy: { '/api': process.env.API_URL ?? 'http://127.0.0.1:8000' },
  },
  build: {
    // ECharts is ~210 kB gzipped. It sits in its own chunk that only the chart pages load (the
    // home page and the resolution list don't), so the default 500 kB warning is expected.
    chunkSizeWarningLimit: 700,
    rolldownOptions: {
      output: {
        advancedChunks: {
          groups: [{ name: 'echarts', test: /node_modules[\\/](echarts|zrender)/ }],
        },
      },
    },
  },
  test: {
    environment: 'jsdom',
  },
})
