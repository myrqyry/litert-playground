import { beforeEach, describe, expect, it, vi } from 'vitest'
import { loadAndCompile, loadLiteRt, setWebGpuDevice } from '@litertjs/core'
import { createLiteRtRuntime } from './context'

vi.mock('@litertjs/core', async () => {
  const actual = await vi.importActual<typeof import('@litertjs/core')>('@litertjs/core')
  return { ...actual, loadLiteRt: vi.fn().mockResolvedValue({}), loadAndCompile: vi.fn().mockResolvedValue({}), setWebGpuDevice: vi.fn() }
})

describe('createLiteRtRuntime', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal('navigator', {})
  })

  it('selects WebGPU only after an adapter and device are usable', async () => {
    const device = {}
    vi.stubGlobal('navigator', { gpu: { requestAdapter: vi.fn().mockResolvedValue({ requestDevice: vi.fn().mockResolvedValue(device) }) } })
    const assets = { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(1)) }
    const context = await createLiteRtRuntime({ assetBase: 'https://assets.test/qwen/', assets })

    expect(loadLiteRt).toHaveBeenCalledWith('https://assets.test/qwen/wasm/', { jspi: true })
    expect(context.backend).toBe('webgpu')
    await context.liteRt.loadModel('model.bin')
    expect(assets.resolve).toHaveBeenCalledWith({ id: 'model.bin', path: 'model.bin' }, { signal: undefined })
    expect(loadAndCompile).toHaveBeenCalledWith(expect.any(Uint8Array), { accelerator: 'webgpu' })
    expect(setWebGpuDevice).toHaveBeenCalledWith(device)
  })

  it('falls back to WASM when WebGPU has no usable adapter', async () => {
    vi.stubGlobal('navigator', { gpu: { requestAdapter: vi.fn().mockResolvedValue(null) } })
    const context = await createLiteRtRuntime({ assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(1)) } })
    expect(context.backend).toBe('wasm')
  })

  it('does not fall back for an explicit unavailable backend', async () => {
    await expect(createLiteRtRuntime({ backend: 'webgpu', assets: { resolve: vi.fn() } }))
      .rejects.toMatchObject({ code: 'BACKEND_UNAVAILABLE' })
  })

  it('falls back to WASM after an automatic GPU compile failure', async () => {
    vi.stubGlobal('navigator', { gpu: { requestAdapter: vi.fn().mockResolvedValue({ requestDevice: vi.fn().mockResolvedValue({}) }) } })
    vi.mocked(loadAndCompile).mockRejectedValueOnce(new Error('gpu compile failed')).mockResolvedValueOnce({} as never)
    const context = await createLiteRtRuntime({ assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(1)) } })
    await context.liteRt.loadModel('model.bin')
    expect(context.backend).toBe('wasm')
    expect(loadAndCompile).toHaveBeenNthCalledWith(2, expect.any(Uint8Array), { accelerator: 'wasm' })
  })
})
