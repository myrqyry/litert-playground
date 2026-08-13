import type { Backend, RuntimeCapabilities } from '@litert-playground/inference-core'

type WebGpu = {
  requestAdapter(): Promise<{ requestDevice(): Promise<unknown> } | null>
}

type NavigatorWithAccelerators = {
  gpu?: WebGpu
  ml?: unknown
}

export async function probeRuntimeCapabilities(): Promise<RuntimeCapabilities> {
  const caps: RuntimeCapabilities = {
    webgpu: { available: false },
    wasm: { available: typeof WebAssembly !== 'undefined', simd: false, threads: false, jspi: true },
    webnn: { available: false, reason: 'WebNN is not exposed by this browser' },
  }

  const navigatorLike = (globalThis as { navigator?: NavigatorWithAccelerators }).navigator
  const gpu = navigatorLike?.gpu
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

  if (navigatorLike?.ml) {
    caps.webnn = { available: true }
  }

  return caps
}

export type BackendPreference = 'auto' | Backend

export const AUTO_BACKEND_ORDER = ['webgpu', 'webnn', 'wasm'] as const satisfies readonly Backend[]

export function rankBackends(
  capabilities: RuntimeCapabilities,
  supported: Partial<Record<Backend, boolean | 'experimental'>> = {},
  preference: BackendPreference = 'auto',
): Backend[] {
  const candidates: readonly Backend[] = preference === 'auto' ? AUTO_BACKEND_ORDER : [preference]
  return candidates.filter((backend) => supported[backend] !== false && capabilities[backend].available)
}

export function selectBackend(
  capabilities: RuntimeCapabilities,
  supported: Partial<Record<Backend, boolean | 'experimental'>> = {},
  preference: BackendPreference = 'auto',
): Backend {
  const [backend] = rankBackends(capabilities, supported, preference)
  if (!backend) throw new Error(`No usable backend for preference ${preference}`)
  return backend
}
