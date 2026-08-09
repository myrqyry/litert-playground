import { loadLiteRt, loadAndCompile } from '@litertjs/core'
import { type RuntimeContext, type LiteRtRuntime, InferenceError } from '../core/types'
import { parseNpy } from '../adapters/qwen3-tts/npy-parser'

export { type RuntimeContext, type LiteRtRuntime }

export async function createRuntimeContext(
  assetBase: string,
  assets: import('../core/types').AssetResolver,
): Promise<RuntimeContext> {
  await loadLiteRt('https://cdn.jsdelivr.net/npm/@litertjs/core/wasm/', { jspi: true })

  const liteRt: LiteRtRuntime = {
    async loadModel(path: string) {
      const buffer = await assets.resolve({ id: path, path })
      return loadAndCompile(new Uint8Array(buffer))
    },
    async loadNpy(path: string) {
      const buffer = await assets.resolve({ id: path, path })
      return parseNpy(buffer)
    },
    async fetchBuffer(path: string) {
      return assets.resolve({ id: path, path })
    },
  }

  return { backend: 'wasm', assets, liteRt }
}