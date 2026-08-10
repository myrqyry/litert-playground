import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { Readable } from 'node:stream'
import type { Connect, Plugin } from 'vite'

const root = path.resolve(__dirname, '..')
const modelPrefix = '/models/qwen3-tts/'
const modelRepository = 'litert-community/Qwen3-TTS-12Hz-0.6B-Base'

function qwenModelProxy(): Plugin {
  const middleware: Connect.NextHandleFunction = async (req, res, next) => {
    const requestPath = req.url?.split('?')[0] ?? ''
    if (!requestPath.startsWith(modelPrefix)) {
      next()
      return
    }

    const encodedPath = requestPath.slice(modelPrefix.length)
    let modelPath: string
    try {
      modelPath = decodeURIComponent(encodedPath)
    } catch {
      res.statusCode = 400
      res.end('Invalid model path')
      return
    }
    if (!modelPath || modelPath.includes('..')) {
      res.statusCode = 400
      res.end('Invalid model path')
      return
    }

    const upstream = new URL(
      `https://huggingface.co/${modelRepository}/resolve/main/${modelPath}`,
    )
    const headers = new Headers()
    for (const name of ['range', 'if-range', 'if-none-match', 'if-modified-since']) {
      const value = req.headers[name]
      if (value) headers.set(name, Array.isArray(value) ? value[0] : value)
    }

    try {
      const response = await fetch(upstream, {
        method: req.method === 'HEAD' ? 'HEAD' : 'GET',
        headers,
      })
      res.statusCode = response.status
      response.headers.forEach((value, name) => res.setHeader(name, value))
      if (req.method === 'HEAD' || !response.body) {
        res.end()
        return
      }
      Readable.fromWeb(response.body).pipe(res)
    } catch (cause) {
      res.statusCode = 502
      res.end(`Model proxy failed: ${String(cause)}`)
    }
  }

  return {
    name: 'qwen-model-proxy',
    configureServer(server) {
      server.middlewares.use(middleware)
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware)
    },
  }
}

export default defineConfig({
  root,
  plugins: [react(), qwenModelProxy()],
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
