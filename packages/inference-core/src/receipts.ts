import type { Backend, InferenceReceipt, ModelManifest } from './types'

export interface InferenceReceiptOptions {
  manifest: Pick<ModelManifest, 'modelId' | 'version'>
  backend: Backend
  loadMs: number
  compileMs: number
  inferenceStart: number
  inputSummary: string
  outputSummary: string
  warnings: string[]
}

function environment(): string | undefined {
  const userAgent = (globalThis as { navigator?: { userAgent?: string } }).navigator?.userAgent
  return userAgent ? `browser: ${userAgent}` : undefined
}

export function createInferenceReceipt(options: InferenceReceiptOptions): InferenceReceipt {
  return {
    modelId: options.manifest.modelId,
    pipelineVersion: options.manifest.version,
    backend: options.backend,
    timestamp: new Date().toISOString(),
    loadMs: options.loadMs,
    compileMs: options.compileMs,
    inferenceMs: performance.now() - options.inferenceStart,
    inputSummary: options.inputSummary,
    outputSummary: options.outputSummary,
    warnings: options.warnings,
    environment: environment(),
  }
}
