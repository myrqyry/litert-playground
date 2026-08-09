import { type RuntimeCapabilities, type Backend } from '../core/types'

export { type RuntimeCapabilities }

export function probeRuntimeCapabilities(): RuntimeCapabilities {
  const caps: RuntimeCapabilities = {
    webgpu: { available: false },
    wasm: { available: true, simd: false, threads: false, jspi: false },
    webnn: { available: false },
  }

  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    caps.webgpu.available = true
  }

  if (typeof navigator !== 'undefined' && 'ml' in navigator) {
    caps.webnn.available = true
  }

  return caps
}

export function selectBackend(
  caps: RuntimeCapabilities,
  supportedList: Partial<Record<Backend, boolean | 'experimental'>>,
  preference: Backend = 'webgpu',
): Backend {
  if (supportedList[preference] && caps[preference]?.available) return preference
  for (const b of ['webgpu', 'wasm', 'webnn'] as Backend[]) {
    if (supportedList[b] && caps[b]?.available) return b
  }
  return 'wasm'
}