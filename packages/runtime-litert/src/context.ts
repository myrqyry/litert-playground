import { loadAndCompile, loadLiteRt, setWebGpuDevice } from '@litertjs/core'
import {
  InferenceError,
  type AssetResolver,
  type Backend,
  type RuntimeContext,
} from '@litert-playground/inference-core'
import { probeRuntimeCapabilities, selectBackend, type BackendPreference } from './capabilities'
import { parseNpy } from './npy'

export interface LiteRtRuntimeOptions {
  assetBase?: string
  backend?: BackendPreference
  assets: AssetResolver
  signal?: AbortSignal
  supportedBackends?: Partial<Record<Backend, boolean | 'experimental'>>
}

function resolveBase(assetBase: string | undefined): string {
  const pageBase = (globalThis as { location?: { href: string } }).location?.href ?? 'http://localhost/'
  return new URL(assetBase ?? 'https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/', pageBase).href
}

export async function createLiteRtRuntime(options: LiteRtRuntimeOptions): Promise<RuntimeContext> {
  const backendPreference = options.backend ?? 'auto'
  const runtimeUrl = new URL('wasm/', resolveBase(options.assetBase)).href
  try {
    await loadLiteRt(runtimeUrl, { jspi: true })
  } catch (cause) {
    throw new InferenceError('BACKEND_UNAVAILABLE', `Failed to load LiteRT runtime: ${String(cause)}`, { cause })
  }

  const capabilities = await probeRuntimeCapabilities()
  let backend: Backend
  try {
    backend = selectBackend(capabilities, options.supportedBackends, backendPreference)
  } catch (cause) {
    throw new InferenceError('BACKEND_UNAVAILABLE', String(cause), { cause })
  }
  if (backend === 'webgpu') {
    const gpu = (globalThis as { navigator?: { gpu?: { requestAdapter(): Promise<{ requestDevice(): Promise<GPUDevice> } | null> } } }).navigator?.gpu
    const adapter = await gpu?.requestAdapter()
    const device = await adapter?.requestDevice()
    if (!device) throw new InferenceError('BACKEND_UNAVAILABLE', 'WebGPU adapter is no longer usable')
    setWebGpuDevice(device)
  }

  const context: RuntimeContext = {
    get backend() { return backend },
    assets: options.assets,
    signal: options.signal,
    liteRt: {
      async loadModel(path: string): Promise<any> {
        const buffer = await options.assets.resolve({ id: path, path }, { signal: options.signal })
        try {
          return await loadAndCompile(new Uint8Array(buffer), { accelerator: backend })
        } catch (cause) {
          if (backendPreference !== 'auto' || backend !== 'webgpu') {
            throw new InferenceError('MODEL_COMPILE_FAILED', `Failed to compile ${path}`, { asset: path, cause })
          }
          try {
            backend = 'wasm'
            return await loadAndCompile(new Uint8Array(buffer), { accelerator: 'wasm' })
          } catch (fallbackCause) {
            throw new InferenceError('MODEL_COMPILE_FAILED', `Failed to compile ${path} on WebGPU and WASM`, { asset: path, cause: fallbackCause })
          }
        }
      },
      async loadNpy(path: string): Promise<Float32Array> {
        return parseNpy(await options.assets.resolve({ id: path, path }, { signal: options.signal }))
      },
      async fetchBuffer(path: string): Promise<ArrayBuffer> {
        return options.assets.resolve({ id: path, path }, { signal: options.signal })
      },
    },
  }
  return context
}

export type { RuntimeContext }
