import type { QualificationCase, QualificationContext, QualificationObservation } from '../schema/types'
import { moduleWorkerLoaderExpected } from './expected'

export async function runModuleWorkerLoader(
  context: QualificationContext,
): Promise<QualificationObservation> {
  if (!context.runtime.runModuleWorkerLoader) {
    throw new Error('Module worker browser adapter is unavailable')
  }
  return context.runtime.runModuleWorkerLoader()
}

export const moduleWorkerLoaderCase: QualificationCase = {
  id: 'module-worker-loader',
  description: 'Reproduces Qwen LiteRT loader behavior in a module worker.',
  evidenceKind: 'browser-observation',
  environments: [{
    runtimePackage: '@litertjs/core',
    runtimeVersion: '2.5.3',
    requestedBackend: 'wasm',
  }],
  expected: moduleWorkerLoaderExpected,
  run: runModuleWorkerLoader,
}
