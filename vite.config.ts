import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'node:path'
import fs from 'node:fs'

export default defineConfig({
  plugins: [
    tailwindcss(),
    react(),
    {
      name: 'static-models',
      configureServer(server) {
        server.middlewares.use('/models', (req, res, next) => {
          const url = req.url || '/'
          const filePath = path.resolve('static-models', url.slice(1))
          if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            res.setHeader('Content-Type', 'application/octet-stream')
            const stream = fs.createReadStream(filePath)
            stream.pipe(res)
            stream.on('error', () => { res.statusCode = 404; res.end() })
          } else {
            next()
          }
        })
      },
    },
  ],
  test: {
    environment: 'node',
  },
  build: {
    assetsInlineLimit: 0,
  },
})
