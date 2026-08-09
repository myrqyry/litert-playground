export type Capability =
  | 'text-to-speech'
  | 'speech-to-text'
  | 'image-generation'
  | 'image-edit'
  | 'image-classification'
  | 'image-embedding'
  | 'image-segmentation'
  | 'image-super-resolution'
  | 'image-depth'
  | 'image-matting'
  | 'image-detection'
  | 'image-pose'
  | 'audio-generation'
  | 'text-embedding'
  | 'audio-embedding'

export type Backend = 'webgpu' | 'wasm' | 'webnn'

export interface ModelAsset {
  id: string
  path: string
  bytes?: number
  sha256?: string
  optional?: boolean
}

export interface AssetResolver {
  resolve(asset: ModelAsset): Promise<ArrayBuffer>
  stream?(asset: ModelAsset): Promise<ReadableStream<Uint8Array>>
}

export interface RuntimeContext {
  backend: Backend
  assets: AssetResolver
  signal?: AbortSignal
  liteRt: LiteRtRuntime
}

export interface LiteRtRuntime {
  loadModel(path: string): Promise<import('@litertjs/core').CompiledModel>
  loadNpy(path: string): Promise<Float32Array>
  fetchBuffer(path: string): Promise<ArrayBuffer>
}

export type PipelineStatus = 'idle' | 'loading' | 'ready' | 'running' | 'error' | 'disposed'

export interface Pipeline<I = unknown, O = unknown, C = unknown> {
  readonly manifest: ModelManifest
  readonly status: PipelineStatus
  load(context: RuntimeContext): Promise<void>
  run(input: I, config?: C, signal?: AbortSignal): Promise<O>
  dispose(): Promise<void>
}

export interface PipelineProgress {
  phase: string
  step: number
  total: number
}

export interface PipelineProgressReporter {
  report(progress: PipelineProgress): void
}

export type InferenceResult =
  | AudioInferenceResult
  | TextInferenceResult
  | ImageInferenceResult
  | EmbeddingInferenceResult
  | TensorInferenceResult

export interface AudioInferenceResult {
  kind: 'audio'
  samples: Float32Array
  sampleRate: number
  channels: number
  durationSeconds: number
}

export interface TextInferenceResult {
  kind: 'text'
  text: string
}

export interface ImageInferenceResult {
  kind: 'image'
  width: number
  height: number
  pixels: Uint8ClampedArray
}

export interface EmbeddingInferenceResult {
  kind: 'embedding'
  values: Float32Array
  dimensions: number
}

export interface TensorInferenceResult {
  kind: 'tensor'
  tensors: Record<string, import('@litertjs/core').Tensor>
}

export interface ModelManifest {
  modelId: string
  name: string
  version: string
  capabilities: Capability[]
  backends: Partial<Record<Backend, boolean | 'experimental'>>
  memory: { downloadBytes: number; residentBytes: number }
  assets: ModelAsset[]
  verification?: ModelVerification
}

export interface ModelVerification {
  compile: 'pass' | 'fail' | 'untested'
  inference: 'pass' | 'fail' | 'untested'
  output: 'pass' | 'fail' | 'untested'
  lastVerifiedAt?: string
  environment?: string
}

export type InferenceErrorCode =
  | 'BACKEND_UNAVAILABLE'
  | 'ASSET_FETCH_FAILED'
  | 'MODEL_COMPILE_FAILED'
  | 'INVALID_INPUT'
  | 'OUT_OF_MEMORY'
  | 'INFERENCE_FAILED'
  | 'OUTPUT_INVALID'
  | 'CANCELLED'

export class InferenceError extends Error {
  readonly code: InferenceErrorCode
  readonly stage?: string
  readonly asset?: string
  readonly cause?: unknown

  constructor(code: InferenceErrorCode, message: string, opts?: { stage?: string; asset?: string; cause?: unknown }) {
    super(message)
    this.name = 'InferenceError'
    this.code = code
    this.stage = opts?.stage
    this.asset = opts?.asset
    if (opts?.cause !== undefined) this.cause = opts.cause
  }
}

export interface InferenceReceipt {
  modelId: string
  pipelineVersion: string
  backend: Backend
  timestamp: string
  loadMs: number
  compileMs: number
  inferenceMs: number
  inputSummary: string
  outputSummary: string
  warnings: string[]
}

export interface RuntimeCapabilities {
  webgpu: { available: boolean; adapter?: Record<string, unknown> }
  wasm: { available: boolean; simd: boolean; threads: boolean; jspi: boolean }
  webnn: { available: boolean; reason?: string }
}