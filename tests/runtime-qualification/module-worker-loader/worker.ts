import { loadLiteRt } from '@litertjs/core'

try {
  await loadLiteRt('/node_modules/@litertjs/core/wasm')
  self.postMessage({ status: 'pass' })
} catch (error) {
  self.postMessage({
    status: 'fail',
    stage: 'worker-load',
    error: { message: error instanceof Error ? error.message : String(error) },
  })
}
