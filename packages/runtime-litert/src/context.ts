import {
  loadAndCompile,
  loadLiteRt,
  setWebGpuDevice,
  Tensor,
  type CompiledModel,
  type TensorDetails,
} from '@litertjs/core'
import {
  InferenceError,
  type Backend,
  type RuntimeCapabilities,
} from '@litert-playground/inference-core'
import { parseNpy } from './npy'
import { probeRuntimeCapabilities, rankBackends, selectBackend } from './capabilities'
import { inferenceCoordinator as defaultCoordinator } from './coordinator'
import type {
  LiteRtModelInfo,
  LiteRtModelInput,
  LiteRtModelOptions,
  LiteRtModelOutput,
  LiteRtPreflightOptions,
  LiteRtPreflightResult,
  LiteRtRuntimeOptions,
  LiteRtTelemetryRecord,
  ManagedLiteRtRuntime,
  ManagedLiteRtRuntimeContext,
  WebNNRuntimeOptions,
} from './types'

interface LoadedModel extends LiteRtModelInfo {
  model: CompiledModel
}

function resolveBase(assetBase: string | undefined): string {
  const pageBase = (globalThis as { location?: { href: string } }).location?.href ?? 'http://localhost/'
  return new URL(assetBase ?? 'https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/', pageBase).href
}

function serializeOptions(value: unknown): string {
  return JSON.stringify(value ?? {})
}

function outputCount(output: LiteRtModelOutput): number {
  return Array.isArray(output) ? output.length : Object.keys(output).length
}

function inputCount(input: LiteRtModelInput): number {
  if (Array.isArray(input)) return input.length
  if (input instanceof Tensor) return 1
  return Object.keys(input).length
}

function createZeroTensor(details: TensorDetails, maxTensorElements: number): Tensor {
  const shape = [...details.shape]
  if (shape.some((dimension) => !Number.isInteger(dimension) || dimension <= 0)) {
    throw new InferenceError(
      'INVALID_INPUT',
      `Dynamic or invalid preflight tensor shape: ${shape.join('x')}`,
      { stage: 'preflight' },
    )
  }

  const size = shape.reduce((total, dimension) => total * dimension, 1)
  if (size > maxTensorElements) {
    throw new InferenceError('INVALID_INPUT', `Preflight tensor is too large: ${size} elements`, {
      stage: 'preflight',
    })
  }

  switch (details.dtype) {
    case 'float32':
      return Tensor.fromTypedArray(new Float32Array(size), shape)
    case 'int32':
      return Tensor.fromTypedArray(new Int32Array(size), shape)
    case 'uint8':
      return Tensor.fromTypedArray(new Uint8Array(size), shape)
    default:
      throw new InferenceError('INVALID_INPUT', `Unsupported preflight tensor type: ${details.dtype}`, {
        stage: 'preflight',
      })
  }
}

class LiteRtRuntimeManager implements ManagedLiteRtRuntime {
  private readonly models = new Map<string, LoadedModel>()
  private readonly telemetry: LiteRtTelemetryRecord[] = []
  private lastResolvedBackend: Backend
  private webGpuDevicePromise: Promise<unknown> | null = null

  constructor(
    private readonly options: LiteRtRuntimeOptions,
    private readonly capabilities: RuntimeCapabilities,
    defaultBackend: Backend,
  ) {
    this.lastResolvedBackend = defaultBackend
  }

  get backend(): Backend {
    return this.lastResolvedBackend
  }

  async loadModel(path: string, options: LiteRtModelOptions = {}): Promise<CompiledModel> {
    const requestKey = this.requestKey(path, options)
    const cached = this.models.get(requestKey)
    if (cached) return cached.model

    const preference = options.accelerator ?? this.options.backend ?? 'auto'
    const supported = options.supportedBackends ?? this.options.supportedBackends ?? {}
    const candidates = rankBackends(this.capabilities, supported, preference)
    if (candidates.length === 0) {
      throw new InferenceError('BACKEND_UNAVAILABLE', `No usable backend for ${path} (${preference})`, {
        stage: 'compile',
        asset: path,
      })
    }

    const buffer = await this.resolve(path)
    let lastError: unknown

    for (let index = 0; index < candidates.length; index += 1) {
      const backend = candidates[index]
      const compileStart = performance.now()
      try {
        await this.prepareBackend(backend)
        const model = await loadAndCompile(
          new Uint8Array(buffer),
          this.compileOptions(backend, options.webNNOptions ?? this.options.webNNOptions),
        )
        const entry: LoadedModel = {
          model,
          modelPath: path,
          requestedBackend: preference,
          resolvedBackend: backend,
          compileDurationMs: performance.now() - compileStart,
          fallbackCount: index,
        }
        this.models.set(requestKey, entry)
        this.lastResolvedBackend = backend
        const { model: _model, ...info } = entry
        this.record({ ...info, event: 'compile', timestamp: new Date().toISOString() })
        return model
      } catch (cause) {
        lastError = cause
        if (preference !== 'auto') break
      }
    }

    throw new InferenceError('MODEL_COMPILE_FAILED', `Failed to compile ${path} on ${candidates.join(', ')}`, {
      stage: 'compile',
      asset: path,
      cause: lastError,
    })
  }

  async loadNpy(path: string): Promise<Float32Array> {
    return parseNpy(await this.resolve(path))
  }

  async fetchBuffer(path: string): Promise<ArrayBuffer> {
    return this.resolve(path)
  }

  async predict(
    path: string,
    input: LiteRtModelInput,
    options: LiteRtModelOptions & { signal?: AbortSignal; label?: string } = {},
  ): Promise<LiteRtModelOutput> {
    return this.run(path, input, undefined, options)
  }

  async predictWithSignature(
    path: string,
    signature: string,
    input: LiteRtModelInput,
    options: LiteRtModelOptions & { signal?: AbortSignal; label?: string } = {},
  ): Promise<LiteRtModelOutput> {
    return this.run(path, input, signature, options)
  }

  async preflight(path: string, options: LiteRtPreflightOptions = {}): Promise<LiteRtPreflightResult> {
    const model = await this.loadModel(path, options)
    const info = this.requireModelInfo(path, options)
    const inputDetails = model.getInputDetails()
    const outputDetails = model.getOutputDetails()
    const maxTensorElements = options.maxTensorElements ?? 10_000_000
    const inputs = options.createInputs?.(inputDetails)
      ?? inputDetails.map((details) => createZeroTensor(details, maxTensorElements))

    const startedAt = performance.now()
    const output = await (this.options.coordinator ?? defaultCoordinator).run(
      async () => {
        const result = options.signature
          ? await model.run(options.signature, inputs)
          : await model.run(inputs)
        return (Array.isArray(result) ? result : result) as LiteRtModelOutput
      },
      options.signal ?? this.options.signal,
      `litert-preflight:${path}`,
    )
    const inferenceDurationMs = performance.now() - startedAt
    const result: LiteRtPreflightResult = {
      ...info,
      inputDetails,
      outputDetails,
      outputCount: outputCount(output),
      inferenceDurationMs,
    }
    this.record({
      ...info,
      event: 'preflight',
      timestamp: new Date().toISOString(),
      inferenceDurationMs,
      inputCount: inputCount(inputs),
      outputCount: result.outputCount,
    })
    return result
  }

  getModelInfo(path: string, options: LiteRtModelOptions = {}): LiteRtModelInfo | undefined {
    const entry = this.models.get(this.requestKey(path, options))
      ?? [...this.models.values()].find((candidate) => candidate.modelPath === path)
    if (!entry) return undefined
    const { model: _model, ...info } = entry
    return info
  }

  getTelemetry(): readonly LiteRtTelemetryRecord[] {
    return [...this.telemetry]
  }

  clearTelemetry(): void {
    this.telemetry.length = 0
  }

  createTensor(data: Float32Array | Int32Array | Uint8Array, shape: number[]): Tensor {
    return Tensor.fromTypedArray(data, shape)
  }

  readTensor<T extends Float32Array | Int32Array | Uint8Array>(tensor: Tensor): T {
    return tensor.toTypedArray() as T
  }

  supportsGpuBufferTensors(): boolean {
    return typeof (Tensor as unknown as { fromGpuBuffer?: unknown }).fromGpuBuffer === 'function'
  }

  disposeModel(path: string): void {
    for (const [key, entry] of this.models.entries()) {
      if (entry.modelPath === path) this.models.delete(key)
    }
  }

  dispose(): void {
    this.models.clear()
    this.telemetry.length = 0
    this.webGpuDevicePromise = null
  }

  private async run(
    path: string,
    input: LiteRtModelInput,
    signature: string | undefined,
    options: LiteRtModelOptions & { signal?: AbortSignal; label?: string },
  ): Promise<LiteRtModelOutput> {
    const model = await this.loadModel(path, options)
    const info = this.requireModelInfo(path, options)
    const startedAt = performance.now()
    const result = await (this.options.coordinator ?? defaultCoordinator).run(
      async () => {
        const output = signature ? await model.run(signature, input) : await model.run(input)
        return (Array.isArray(output) ? output : output) as LiteRtModelOutput
      },
      options.signal ?? this.options.signal,
      options.label ?? `litert:${path}`,
    )
    const inferenceDurationMs = performance.now() - startedAt
    this.record({
      ...info,
      event: 'inference',
      timestamp: new Date().toISOString(),
      inferenceDurationMs,
      inputCount: inputCount(input),
      outputCount: outputCount(result),
    })
    return result
  }

  private requireModelInfo(path: string, options: LiteRtModelOptions): LiteRtModelInfo {
    const info = this.getModelInfo(path, options)
    if (!info) throw new InferenceError('INFERENCE_FAILED', `Model ${path} is not loaded`)
    return info
  }

  private requestKey(path: string, options: LiteRtModelOptions): string {
    return [
      path,
      options.accelerator ?? this.options.backend ?? 'auto',
      serializeOptions(options.supportedBackends ?? this.options.supportedBackends ?? {}),
      serializeOptions(options.webNNOptions ?? this.options.webNNOptions ?? {}),
    ].join('::')
  }

  private async resolve(path: string): Promise<ArrayBuffer> {
    try {
      return await this.options.assets.resolve({ id: path, path }, { signal: this.options.signal })
    } catch (cause) {
      throw new InferenceError('ASSET_FETCH_FAILED', `Failed to resolve ${path}`, { asset: path, cause })
    }
  }

  private async prepareBackend(backend: Backend): Promise<void> {
    if (backend !== 'webgpu') return
    if (!this.webGpuDevicePromise) {
      this.webGpuDevicePromise = (async () => {
        const gpu = (globalThis as {
          navigator?: { gpu?: { requestAdapter(): Promise<{ requestDevice(): Promise<unknown> } | null> } }
        }).navigator?.gpu
        const adapter = await gpu?.requestAdapter()
        const device = await adapter?.requestDevice()
        if (!device) throw new InferenceError('BACKEND_UNAVAILABLE', 'WebGPU adapter is no longer usable')
        setWebGpuDevice(device as Parameters<typeof setWebGpuDevice>[0])
        return device
      })().catch((error) => {
        this.webGpuDevicePromise = null
        throw error
      })
    }
    await this.webGpuDevicePromise
  }

  private compileOptions(backend: Backend, webNNOptions?: WebNNRuntimeOptions): Parameters<typeof loadAndCompile>[1] {
    const options: { accelerator: Backend; webNNOptions?: WebNNRuntimeOptions } = { accelerator: backend }
    if (backend === 'webnn' && webNNOptions) options.webNNOptions = webNNOptions
    return options as Parameters<typeof loadAndCompile>[1]
  }

  private record(record: LiteRtTelemetryRecord): void {
    this.telemetry.push(record)
    this.options.onTelemetry?.(record)
  }
}

export async function createLiteRtRuntime(options: LiteRtRuntimeOptions): Promise<ManagedLiteRtRuntimeContext> {
  const runtimeUrl = new URL('wasm/', resolveBase(options.assetBase)).href
  try {
    await loadLiteRt(runtimeUrl, { jspi: true })
  } catch (cause) {
    throw new InferenceError('BACKEND_UNAVAILABLE', `Failed to load LiteRT runtime: ${String(cause)}`, { cause })
  }

  const capabilities = await probeRuntimeCapabilities()
  let defaultBackend: Backend
  try {
    defaultBackend = selectBackend(capabilities, options.supportedBackends, options.backend ?? 'auto')
  } catch (cause) {
    throw new InferenceError('BACKEND_UNAVAILABLE', String(cause), { cause })
  }

  const runtime = new LiteRtRuntimeManager(options, capabilities, defaultBackend)
  const context: ManagedLiteRtRuntimeContext = {
    get backend() {
      return runtime.backend
    },
    assets: options.assets,
    signal: options.signal,
    liteRt: runtime,
  }
  return context
}

export type { RuntimeContext } from '@litert-playground/inference-core'
