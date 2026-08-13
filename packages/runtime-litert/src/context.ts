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

const DEFAULT_TELEMETRY_LIMIT = 512

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

function isAbort(cause: unknown, signal?: AbortSignal): boolean {
  if (signal?.aborted) return true
  return cause instanceof Error && cause.name === 'AbortError'
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
  private readonly pendingLoads = new Map<string, Promise<LoadedModel>>()
  private readonly telemetry: LiteRtTelemetryRecord[] = []
  private lastResolvedBackend: Backend
  private webGpuDevicePromise: Promise<unknown> | null = null
  private tensorCopyCount = 0
  private disposed = false

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
    this.assertUsable()
    const requestKey = this.requestKey(path, options)
    const cached = this.models.get(requestKey)
    if (cached) return cached.model

    const pending = this.pendingLoads.get(requestKey)
    if (pending) return (await pending).model

    const load = this.compileModel(path, options, requestKey)
    this.pendingLoads.set(requestKey, load)
    try {
      return (await load).model
    } finally {
      this.pendingLoads.delete(requestKey)
    }
  }

  async loadNpy(path: string, signal?: AbortSignal): Promise<Float32Array> {
    this.assertUsable()
    return parseNpy(await this.resolve(path, signal))
  }

  async fetchBuffer(path: string, signal?: AbortSignal): Promise<ArrayBuffer> {
    this.assertUsable()
    return this.resolve(path, signal)
  }

  async predict(
    path: string,
    input: LiteRtModelInput,
    options: LiteRtModelOptions & { label?: string } = {},
  ): Promise<LiteRtModelOutput> {
    return this.run(path, input, undefined, options)
  }

  async predictWithSignature(
    path: string,
    signature: string,
    input: LiteRtModelInput,
    options: LiteRtModelOptions & { label?: string } = {},
  ): Promise<LiteRtModelOutput> {
    return this.run(path, input, signature, options)
  }

  async preflight(path: string, options: LiteRtPreflightOptions = {}): Promise<LiteRtPreflightResult> {
    this.assertUsable()
    const signal = options.signal ?? this.options.signal
    const model = await this.loadModel(path, options)
    const info = this.requireModelInfo(path, options)
    const inputDetails = model.getInputDetails()
    const outputDetails = model.getOutputDetails()
    const maxTensorElements = options.maxTensorElements ?? 10_000_000
    const inputs = options.createInputs?.(inputDetails)
      ?? inputDetails.map((details) => createZeroTensor(details, maxTensorElements))

    let inferenceStartedAt = 0
    try {
      const output = await (this.options.coordinator ?? defaultCoordinator).run(
        async () => {
          inferenceStartedAt = performance.now()
          const result = options.signature
            ? await model.run(options.signature, inputs)
            : await model.run(inputs)
          return (Array.isArray(result) ? result : result) as LiteRtModelOutput
        },
        signal,
        `litert-preflight:${path}`,
      )
      const inferenceDurationMs = performance.now() - inferenceStartedAt
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
        tensorCopyCount: this.tensorCopyCount,
      })
      return result
    } catch (cause) {
      if (cause instanceof InferenceError) throw cause
      if (isAbort(cause, signal)) {
        throw new InferenceError('CANCELLED', `Preflight cancelled for ${path}`, { stage: 'preflight', cause })
      }
      throw new InferenceError('INFERENCE_FAILED', `Preflight failed for ${path}`, { stage: 'preflight', cause })
    }
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
    this.tensorCopyCount = 0
  }

  createTensor(data: Float32Array | Int32Array | Uint8Array, shape: number[]): Tensor {
    this.assertUsable()
    return Tensor.fromTypedArray(data, shape)
  }

  readTensor<T extends Float32Array | Int32Array | Uint8Array>(tensor: Tensor): T {
    this.assertUsable()
    this.tensorCopyCount += 1
    const latest = this.telemetry[this.telemetry.length - 1]
    if (latest && (latest.event === 'inference' || latest.event === 'preflight')) {
      latest.tensorCopyCount = this.tensorCopyCount
    }
    return tensor.toTypedArray() as T
  }

  supportsGpuBufferTensors(): boolean {
    return typeof (Tensor as unknown as { fromGpuBuffer?: unknown }).fromGpuBuffer === 'function'
  }

  disposeModel(path: string): void {
    for (const [key, entry] of this.models.entries()) {
      if (entry.modelPath === path) this.models.delete(key)
    }
    for (const [key] of this.pendingLoads.entries()) {
      if (key.startsWith(`${path}::`)) this.pendingLoads.delete(key)
    }
  }

  dispose(): void {
    this.disposed = true
    this.models.clear()
    this.pendingLoads.clear()
    this.telemetry.length = 0
    this.webGpuDevicePromise = null
    this.tensorCopyCount = 0
  }

  private async compileModel(
    path: string,
    options: LiteRtModelOptions,
    requestKey: string,
  ): Promise<LoadedModel> {
    const signal = options.signal ?? this.options.signal
    const preference = options.accelerator ?? this.options.backend ?? 'auto'
    const supported = options.supportedBackends ?? this.options.supportedBackends ?? {}
    const candidates = rankBackends(this.capabilities, supported, preference)
    if (candidates.length === 0) {
      throw new InferenceError('BACKEND_UNAVAILABLE', `No usable backend for ${path} (${preference})`, {
        stage: 'compile',
        asset: path,
      })
    }

    const buffer = await this.resolve(path, signal)
    let lastError: unknown

    for (let index = 0; index < candidates.length; index += 1) {
      if (signal?.aborted) {
        throw new InferenceError('CANCELLED', `Model load cancelled for ${path}`, { stage: 'compile', asset: path })
      }

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
        this.record({
          ...info,
          event: 'compile',
          timestamp: new Date().toISOString(),
          tensorCopyCount: this.tensorCopyCount,
        })
        return entry
      } catch (cause) {
        lastError = cause
        if (backend === 'webgpu') this.webGpuDevicePromise = null
        if (isAbort(cause, signal)) {
          throw new InferenceError('CANCELLED', `Model load cancelled for ${path}`, {
            stage: 'compile',
            asset: path,
            cause,
          })
        }
        if (preference !== 'auto') break
      }
    }

    throw new InferenceError('MODEL_COMPILE_FAILED', `Failed to compile ${path} on ${candidates.join(', ')}`, {
      stage: 'compile',
      asset: path,
      cause: lastError,
    })
  }

  private async run(
    path: string,
    input: LiteRtModelInput,
    signature: string | undefined,
    options: LiteRtModelOptions & { label?: string },
  ): Promise<LiteRtModelOutput> {
    this.assertUsable()
    const signal = options.signal ?? this.options.signal
    const model = await this.loadModel(path, options)
    const info = this.requireModelInfo(path, options)
    let inferenceStartedAt = 0
    try {
      const result = await (this.options.coordinator ?? defaultCoordinator).run(
        async () => {
          inferenceStartedAt = performance.now()
          const output = signature ? await model.run(signature, input) : await model.run(input)
          return (Array.isArray(output) ? output : output) as LiteRtModelOutput
        },
        signal,
        options.label ?? `litert:${path}`,
      )
      const inferenceDurationMs = performance.now() - inferenceStartedAt
      this.record({
        ...info,
        event: 'inference',
        timestamp: new Date().toISOString(),
        inferenceDurationMs,
        inputCount: inputCount(input),
        outputCount: outputCount(result),
        tensorCopyCount: this.tensorCopyCount,
      })
      return result
    } catch (cause) {
      if (cause instanceof InferenceError) throw cause
      if (isAbort(cause, signal)) {
        throw new InferenceError('CANCELLED', `Inference cancelled for ${path}`, { stage: 'inference', cause })
      }
      throw new InferenceError('INFERENCE_FAILED', `Inference failed for ${path}`, { stage: 'inference', cause })
    }
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

  private async resolve(path: string, signal?: AbortSignal): Promise<ArrayBuffer> {
    const effectiveSignal = signal ?? this.options.signal
    try {
      return await this.options.assets.resolve({ id: path, path }, { signal: effectiveSignal })
    } catch (cause) {
      if (isAbort(cause, effectiveSignal)) {
        throw new InferenceError('CANCELLED', `Asset fetch cancelled for ${path}`, { asset: path, cause })
      }
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
    const limit = Math.max(1, this.options.telemetryLimit ?? DEFAULT_TELEMETRY_LIMIT)
    if (this.telemetry.length > limit) this.telemetry.splice(0, this.telemetry.length - limit)
    this.options.onTelemetry?.(record)
  }

  private assertUsable(): void {
    if (this.disposed) throw new InferenceError('INFERENCE_FAILED', 'LiteRT runtime has been disposed')
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
