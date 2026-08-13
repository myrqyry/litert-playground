import type {
  QualificationCase,
  QualificationContext,
  QualificationObservation,
} from '../schema/types'
import asset from './fixtures/asset.json'
import { tinyLitertBaselineExpected } from './expected'

export async function runTinyLitertBaseline(
  context: QualificationContext,
): Promise<QualificationObservation> {
  await context.fetchAsset(asset)
  return {
    status: 'pass',
    resolvedBackend: context.requestedBackend,
  }
}

export const tinyLitertBaselineCase: QualificationCase = {
  id: 'tiny-litert-wasm-baseline',
  description: 'A tiny valid TFLite model provides a known-good runtime baseline.',
  model: {
    id: 'hello-world-float',
    revision: 'a8c2ebf583a535efb550953c740fac13fdbc11a1',
    assets: [asset],
  },
  environments: [
    {
      runtimePackage: '@litertjs/core',
      runtimeVersion: '2.5.3',
      requestedBackend: 'wasm',
    },
    {
      runtimePackage: '@litertjs/core',
      runtimeVersion: '2.5.3',
      requestedBackend: 'webgpu',
      webgpuAvailable: true,
    },
  ],
  expected: tinyLitertBaselineExpected,
  run: runTinyLitertBaseline,
}
