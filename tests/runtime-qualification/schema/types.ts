import type { InferenceDiagnostics } from '../../../packages/inference-core/src/types'

export type QualificationStatus = 'pass' | 'known-limitation' | 'fail'
export type QualificationBackend = 'wasm' | 'webgpu'

export interface QualificationEnvironment {
  browser?: string
  browserVersion?: string
  operatingSystem?: string
  device?: string
  gpu?: string
  runtimePackage: string
  runtimeVersion: string
  requestedBackend: QualificationBackend
  resolvedBackend?: QualificationBackend
  webgpuAvailable?: boolean
}

export interface ModelAssetDescriptor {
  id: string
  url: string
  bytes: number
  sha256: string
  mimeType?: string
}

export interface QualificationModel {
  id: string
  variant?: string
  revision?: string
  assets: ModelAssetDescriptor[]
}

export interface QualificationError {
  code?: string
  stage?: string
  message: string
}

export interface QualificationObservation {
  status: QualificationStatus
  stage?: string
  resolvedBackend?: QualificationBackend
  diagnostics?: InferenceDiagnostics
  error?: QualificationError
}

export interface QualificationContext {
  requestedBackend: QualificationBackend
  environment: QualificationEnvironment
  fetchAsset(asset: ModelAssetDescriptor): Promise<ArrayBuffer>
  runtime: QualificationRuntime
}

export interface QualificationRuntime {
  initialize?(): Promise<void>
  loadAndCompile(
    model: Uint8Array,
    options: { accelerator: QualificationBackend },
  ): Promise<QualificationCompiledModel>
}

export interface QualificationCompiledModel {
  getInputDetails(): readonly QualificationTensorDetails[]
  getOutputDetails(): readonly QualificationTensorDetails[]
  run(input: QualificationTensor[]): Promise<QualificationTensor[]>
  delete(): void
}

export interface QualificationTensorDetails {
  shape: readonly number[]
  dtype: string
}

export interface QualificationTensor {
  data: unknown
  shape?: readonly number[]
}

export interface QualificationCase {
  id: string
  description: string
  model?: QualificationModel
  environments: QualificationEnvironment[]
  expected: {
    status: 'pass' | 'known-limitation'
    error?: {
      code?: string
      stage?: string
      messagePattern?: string
    }
  }
  run(context: QualificationContext): Promise<QualificationObservation>
}

export interface QualificationResult {
  schemaVersion: 1
  caseId: string
  timestamp: string
  playgroundRevision: string
  runtimePackage: string
  runtimeVersion: string
  environment: QualificationEnvironment
  model?: QualificationModel
  expected: QualificationCase['expected']
  observed: QualificationObservation
  matchesExpectation: boolean
}

export interface QualificationSelection {
  caseIds?: string[]
  backends?: QualificationBackend[]
}

export interface QualificationRunOptions {
  playgroundRevision: string
  runtimePackage: string
  runtimeVersion: string
  resultsDirectory?: string
  now?: () => Date
}

export function matchQualificationExpectation(
  expected: QualificationCase['expected'],
  observed: QualificationObservation,
): boolean {
  if (expected.status === 'pass') return observed.status === 'pass'
  return observed.status === 'fail'
}

export function mapQualificationStatus(
  result: Pick<QualificationObservation, 'status'>,
): 'limited' | 'qualified' {
  return result.status === 'pass' ? 'qualified' : 'limited'
}
