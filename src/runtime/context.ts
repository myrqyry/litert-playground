import { loadLiteRt, loadAndCompile, setWebGpuDevice } from '@litertjs/core'
import { type RuntimeContext, type LiteRtRuntime, InferenceError } from '../core/types'
import { parseNpy } from '../adapters/qwen3-tts/npy-parser'

export { type RuntimeContext, type LiteRtRuntime }

export async function createRuntimeContext(
  assetBase: string,
  assets: import('../core/types').AssetResolver,
  signal?: AbortSignal,
): Promise<RuntimeContext> {
  const pageBase = (globalThis as { location?: { href: string } }).location?.href ?? 'http://localhost/'
  const runtimeUrl = new URL('wasm/', new URL(assetBase, pageBase)).href
  try {
    await loadLiteRt(runtimeUrl, { jspi: true })
  } catch (e) {
    throw new InferenceError('BACKEND_UNAVAILABLE', `Failed to load LiteRT runtime: ${String(e)}`, { cause: e })
  }

  let backend: 'webgpu' | 'wasm' = 'wasm'
  const gpu = (globalThis as {
    navigator?: { gpu?: { requestAdapter(): Promise<{ requestDevice(): Promise<unknown> } | null> } }
  }).navigator?.gpu
  if (gpu) {
    try {
      const adapter = await gpu.requestAdapter()
      const device = await adapter?.requestDevice()
      if (device) {
        setWebGpuDevice(device as GPUDevice)
        backend = 'webgpu'
      }
    } catch {
      backend = 'wasm'
    }
  }

  const liteRt: LiteRtRuntime = {
    async loadModel(path: string) {
      const buffer = await assets.resolve({ id: path, path }, signal)
      try {
        return await loadAndCompile(new Uint8Array(buffer), { accelerator: backend })
      } catch (e) {
        if (backend !== 'webgpu') throw e
        backend = 'wasm'
        return loadAndCompile(new Uint8Array(buffer), { accelerator: 'wasm' })
      }
    },
    async loadNpy(path: string) {
      const buffer = await assets.resolve({ id: path, path }, signal)
      return parseNpy(buffer)
    },
    async fetchBuffer(path: string) {
      return assets.resolve({ id: path, path }, signal)
    },
  }

  return {
    get backend() { return backend },
    assets,
    signal,
    liteRt,
  }
}
