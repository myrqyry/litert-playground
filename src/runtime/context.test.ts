import { beforeEach, describe, expect, it, vi } from 'vitest'
import { loadAndCompile, loadLiteRt, setWebGpuDevice } from '@litertjs/core'
import { createRuntimeContext } from './context'

vi.mock('@litertjs/core', async () => {
  const actual = await vi.importActual<typeof import('@litertjs/core')>('@litertjs/core')
  return {
    ...actual,
    loadLiteRt: vi.fn().mockResolvedValue({}),
    loadAndCompile: vi.fn().mockResolvedValue({}),
    setWebGpuDevice: vi.fn(),
  }
})

describe('createRuntimeContext', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal('navigator', {})
  })

  it('uses the supplied asset base and selects WebGPU when available', async () => {
    const device = {}
    vi.stubGlobal('navigator', {
      gpu: { requestAdapter: vi.fn().mockResolvedValue({ requestDevice: vi.fn().mockResolvedValue(device) }) },
    })
    const assets = { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(1)) }
    const context = await createRuntimeContext('https://assets.test/qwen/', assets)

    expect(loadLiteRt).toHaveBeenCalledWith('https://assets.test/qwen/wasm/', { jspi: true })
    expect(context.backend).toBe('webgpu')
    await context.liteRt.loadModel('model.bin')
    expect(assets.resolve).toHaveBeenCalledWith({ id: 'model.bin', path: 'model.bin' }, undefined)
    expect(loadAndCompile).toHaveBeenCalledWith(expect.any(Uint8Array), { accelerator: 'webgpu' })
    expect(setWebGpuDevice).toHaveBeenCalledWith(device)
  })

  it('falls back to WASM when WebGPU is unavailable', async () => {
    const assets = { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(1)) }
    const context = await createRuntimeContext('https://assets.test/qwen/', assets)

    expect(context.backend).toBe('wasm')
    await context.liteRt.loadModel('model.bin')
    expect(loadAndCompile).toHaveBeenCalledWith(expect.any(Uint8Array), { accelerator: 'wasm' })
  })

  it('falls back to WASM when WebGPU compilation fails', async () => {
    vi.stubGlobal('navigator', {
      gpu: { requestAdapter: vi.fn().mockResolvedValue({ requestDevice: vi.fn().mockResolvedValue({}) }) },
    })
    vi.mocked(loadAndCompile)
      .mockRejectedValueOnce(new Error('gpu compile failed'))
      .mockResolvedValueOnce({} as never)
    const assets = { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(1)) }
    const context = await createRuntimeContext('https://assets.test/qwen/', assets)

    await context.liteRt.loadModel('model.bin')
    expect(context.backend).toBe('wasm')
    expect(loadAndCompile).toHaveBeenNthCalledWith(2, expect.any(Uint8Array), { accelerator: 'wasm' })
  })
})
