import type { Backend, RuntimeCapabilities } from '@litert-playground/inference-core'

type WebGpu = {
  requestAdapter(): Promise<{ requestDevice(): Promise<unknown> } | null>
}

type NavigatorWithAccelerators = {
  gpu?: WebGpu
  ml?: unknown
}

function probeWasmSimd(): boolean {
  try {
    // Minimal WASM module with a SIMD v128.const op (0xfd 0x0c). If validated, SIMD is supported.
    return typeof WebAssembly !== 'undefined' && WebAssembly.validate(new Uint8Array([
      0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
      0x03, 0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01, 0x08, 0x00, 0xfd, 0x0c, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x0b,
    ]))
  } catch { return false }
}

function probeWasmThreads(): boolean {
  try {
    if (typeof SharedArrayBuffer === 'undefined') return false
    // Threads require crossOriginIsolated + shared WebAssembly.Memory
    if ((globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === false) return false
    return WebAssembly.validate(new Uint8Array([
      0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ])) && typeof SharedArrayBuffer !== 'undefined'
      && (() => { try { new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true } as WebAssembly.MemoryDescriptor); return true } catch { return false } })()
  } catch { return false }
}

function probeWasmJspi(): boolean {
  try {
    // JSPI aka JSPromiseIntegration / WebAssembly.Function with promising
    const w = WebAssembly as unknown as { Function?: unknown; promising?: unknown; JSPromise?: unknown }
    return typeof w.Function !== 'undefined' || typeof w.promising !== 'undefined' || typeof w.JSPromise !== 'undefined'
  } catch { return false }
}

export async function probeRuntimeCapabilities(): Promise<RuntimeCapabilities> {
  const caps: RuntimeCapabilities = {
    webgpu: { available: false },
    wasm: {
      available: typeof WebAssembly !== 'undefined',
      simd: probeWasmSimd(),
      threads: probeWasmThreads(),
      jspi: probeWasmJspi(),
    },
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
