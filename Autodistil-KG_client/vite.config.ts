import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true, // forward WebSocket upgrades (e.g. /api/ws -> backend /ws)
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
      '/ws': {
        target: 'http://localhost:8000',
        ws: true,
      },
    },
    allowedHosts: ['bumblebee'],
  },
})
