import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

const root = path.resolve(__dirname, '..')

export default defineConfig({
  root,
  plugins: [react()],
  build: {
    assetsInlineLimit: 0,
    rollupOptions: {
      input: {
        qwen3Tts: path.resolve(root, 'examples/minimal-qwen3-tts/index.html'),
        kokoro: path.resolve(root, 'examples/minimal-kokoro/index.html'),
      },
    },
  },
})
