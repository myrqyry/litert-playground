import { afterEach, describe, expect, it, vi } from 'vitest'
import type { RuntimeCapabilities } from '@litert-playground/inference-core'
import { AUTO_BACKEND_ORDER, probeRuntimeCapabilities, rankBackends, selectBackend } from './capabilities'

const allAvailable: RuntimeCapabilities = {
  webgpu: { available: true },
  webnn: { available: true },
  wasm: { available: true, simd: false, threads: false, jspi: true },
}

describe('LiteRT backend selection', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('prefers WebGPU, then WebNN, then WASM in auto mode', () => {
    expect(AUTO_BACKEND_ORDER).toEqual(['webgpu', 'webnn', 'wasm'])
    expect(rankBackends(allAvailable)).toEqual(['webgpu', 'webnn', 'wasm'])
    expect(selectBackend(allAvailable)).toBe('webgpu')
  })

  it('respects per-model backend support when ranking fallbacks', () => {
    expect(rankBackends(allAvailable, { webgpu: false })).toEqual(['webnn', 'wasm'])
    expect(rankBackends(allAvailable, { webgpu: false, webnn: false })).toEqual(['wasm'])
  })

  it('does not silently substitute an explicitly requested backend', () => {
    expect(rankBackends(allAvailable, { webgpu: false }, 'webgpu')).toEqual([])
    expect(() => selectBackend(allAvailable, { webgpu: false }, 'webgpu')).toThrow(/No usable backend/)
  })

  it('detects WebNN when navigator.ml is exposed', async () => {
    vi.stubGlobal('navigator', { ml: {} })
    const capabilities = await probeRuntimeCapabilities()
    expect(capabilities.webnn.available).toBe(true)
  })
})
