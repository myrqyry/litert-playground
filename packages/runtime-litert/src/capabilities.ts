import type { Backend, RuntimeCapabilities } from '@litert-playground/inference-core'

type WebGpu = {
  requestAdapter(): Promise<{ requestDevice(): Promise<GPUDevice> } | null>
}

export async function probeRuntimeCapabilities(): Promise<RuntimeCapabilities> {
  const caps: RuntimeCapabilities = {
    webgpu: { available: false },
    wasm: { available: true, simd: false, threads: false, jspi: true },
    webnn: { available: false, reason: 'WebNN probing is not implemented' },
  }
  const gpu = (globalThis as { navigator?: { gpu?: WebGpu } }).navigator?.gpu
  if (gpu) {
    try {
      const adapter = await gpu.requestAdapter()
      if (adapter) {
        await adapter.requestDevice()
        caps.webgpu = { available: true }
      }
    } catch {
      caps.webgpu = { available: false }
    }
  }
  return caps
}

export type BackendPreference = 'auto' | Backend

export function selectBackend(
  capabilities: RuntimeCapabilities,
  supported: Partial<Record<Backend, boolean | 'experimental'>> = {},
  preference: BackendPreference = 'auto',
): Backend {
  const candidates = preference === 'auto' ? ['webgpu', 'wasm', 'webnn'] as Backend[] : [preference]
  for (const backend of candidates) {
    if (supported[backend] === false) continue
    if (supported[backend] || preference === 'auto') {
      if (capabilities[backend].available) return backend
    }
  }
  throw new Error(`No usable backend for preference ${preference}`)
}
