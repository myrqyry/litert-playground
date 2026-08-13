import { beforeEach, describe, expect, it, vi } from 'vitest'
import { loadAndCompile, loadLiteRt, setWebGpuDevice } from '@litertjs/core'
import { createLiteRtRuntime } from './context'

vi.mock('@litertjs/core', async () => {
  const actual = await vi.importActual<typeof import('@litertjs/core')>('@litertjs/core')
  return {
    ...actual,
    loadLiteRt: vi.fn(),
    loadAndCompile: vi.fn(),
    setWebGpuDevice: vi.fn(),
  }
})

describe('managed LiteRT runtime', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal('navigator', {})
    vi.mocked(loadLiteRt).mockResolvedValue({} as never)
    vi.mocked(loadAndCompile).mockResolvedValue({} as never)
  })

  it('caches compiled models and emits compile telemetry', async () => {
    const model = {}
    vi.mocked(loadAndCompile).mockResolvedValue(model as never)
    const assets = { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) }
    const onTelemetry = vi.fn()
    const context = await createLiteRtRuntime({ backend: 'wasm', assets, onTelemetry })

    await context.liteRt.loadModel('vision.tflite')
    await context.liteRt.loadModel('vision.tflite')

    expect(assets.resolve).toHaveBeenCalledTimes(1)
    expect(loadAndCompile).toHaveBeenCalledTimes(1)
    expect(context.liteRt.getModelInfo('vision.tflite')).toMatchObject({
      modelPath: 'vision.tflite',
      requestedBackend: 'wasm',
      resolvedBackend: 'wasm',
      fallbackCount: 0,
    })
    expect(context.liteRt.getTelemetry()).toHaveLength(1)
    expect(onTelemetry).toHaveBeenCalledWith(expect.objectContaining({ event: 'compile' }))
  })

  it('deduplicates concurrent model loads', async () => {
    let releaseCompile!: (model: unknown) => void
    const compileGate = new Promise((resolve) => { releaseCompile = resolve })
    vi.mocked(loadAndCompile).mockReturnValue(compileGate as never)
    const assets = { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) }
    const context = await createLiteRtRuntime({ backend: 'wasm', assets })

    const first = context.liteRt.loadModel('shared.tflite')
    const second = context.liteRt.loadModel('shared.tflite')
    await vi.waitFor(() => expect(loadAndCompile).toHaveBeenCalledTimes(1))
    releaseCompile({})

    await expect(Promise.all([first, second])).resolves.toHaveLength(2)
    expect(assets.resolve).toHaveBeenCalledTimes(1)
    expect(loadAndCompile).toHaveBeenCalledTimes(1)
  })

  it('uses WebNN before WASM when WebGPU compilation fails in auto mode', async () => {
    const device = {}
    vi.stubGlobal('navigator', {
      gpu: { requestAdapter: vi.fn().mockResolvedValue({ requestDevice: vi.fn().mockResolvedValue(device) }) },
      ml: {},
    })
    vi.mocked(loadAndCompile)
      .mockRejectedValueOnce(new Error('webgpu compile failed'))
      .mockResolvedValueOnce({} as never)
    const context = await createLiteRtRuntime({
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) },
      webNNOptions: { devicePreference: 'gpu', precision: 'fp16' },
    })

    await context.liteRt.loadModel('model.tflite')

    expect(setWebGpuDevice).toHaveBeenCalledWith(device)
    expect(loadAndCompile).toHaveBeenNthCalledWith(
      2,
      expect.any(Uint8Array),
      expect.objectContaining({
        accelerator: 'webnn',
        webNNOptions: { devicePreference: 'gpu', precision: 'fp16' },
      }),
    )
    expect(context.backend).toBe('webnn')
    expect(context.liteRt.getModelInfo('model.tflite')?.fallbackCount).toBe(1)
  })

  it('preflights a model through the shared coordinator and records a receipt-like trace', async () => {
    const output = {} as never
    const model = {
      getInputDetails: vi.fn().mockReturnValue([{ shape: [1, 4], dtype: 'float32' }]),
      getOutputDetails: vi.fn().mockReturnValue([{ shape: [1, 2], dtype: 'float32' }]),
      run: vi.fn().mockResolvedValue([output]),
    }
    vi.mocked(loadAndCompile).mockResolvedValue(model as never)
    const context = await createLiteRtRuntime({
      backend: 'wasm',
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) },
    })

    const result = await context.liteRt.preflight('model.tflite', { createInputs: () => [] })

    expect(model.run).toHaveBeenCalledWith([])
    expect(result).toMatchObject({ resolvedBackend: 'wasm', outputCount: 1, fallbackCount: 0 })
    expect(context.liteRt.getTelemetry().map((entry) => entry.event)).toEqual(['compile', 'preflight'])
  })

  it('supports named-signature prediction and inference telemetry', async () => {
    const result = [{}] as never
    const model = { run: vi.fn().mockResolvedValue(result) }
    vi.mocked(loadAndCompile).mockResolvedValue(model as never)
    const context = await createLiteRtRuntime({
      backend: 'wasm',
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) },
    })
    const input = {} as never

    await context.liteRt.predictWithSignature('model.tflite', 'decode', input, { label: 'decode-step' })

    expect(model.run).toHaveBeenCalledWith('decode', input)
    const telemetry = context.liteRt.getTelemetry()
    expect(telemetry[telemetry.length - 1]).toMatchObject({
      event: 'inference',
      modelPath: 'model.tflite',
      resolvedBackend: 'wasm',
      outputCount: 1,
    })
  })

  it('bounds telemetry history for long-running consumers', async () => {
    const model = { run: vi.fn().mockResolvedValue([{}]) }
    vi.mocked(loadAndCompile).mockResolvedValue(model as never)
    const context = await createLiteRtRuntime({
      backend: 'wasm',
      telemetryLimit: 2,
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) },
    })

    await context.liteRt.predict('model.tflite', {} as never)
    await context.liteRt.predict('model.tflite', {} as never)
    await context.liteRt.predict('model.tflite', {} as never)

    expect(context.liteRt.getTelemetry()).toHaveLength(2)
    expect(context.liteRt.getTelemetry().every((entry) => entry.event === 'inference')).toBe(true)
  })
})
