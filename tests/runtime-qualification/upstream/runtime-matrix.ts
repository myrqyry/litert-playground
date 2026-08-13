import type { QualificationBackend, QualificationError } from '../schema/types'

export interface RuntimeLane {
  id: 'production-2.5.3' | 'upstream'
  packageName: string
  version: string
  installCommand: string
}

export interface ModelTesterResult {
  modelId: string
  backend: QualificationBackend
  status: 'pass' | 'fail'
  stage?: string
  error?: QualificationError
}

export function getRuntimeLanes(): RuntimeLane[] {
  return [
    {
      id: 'production-2.5.3',
      packageName: '@litertjs/core',
      version: '2.5.3',
      installCommand: 'pnpm install --frozen-lockfile',
    },
    {
      id: 'upstream',
      packageName: '@litertjs/core',
      version: 'isolated checkout or prerelease',
      installCommand: 'run in a temporary qualification directory',
    },
  ]
}

export function classifyRuntimeFailure(
  result: ModelTesterResult,
): 'managed-runtime' | 'litert-runtime' | 'backend' | 'unknown' {
  if (result.stage === 'managed-runtime') return 'managed-runtime'
  if (result.stage === 'runtime') return 'litert-runtime'
  if (result.stage === 'backend') return 'backend'
  return 'unknown'
}
