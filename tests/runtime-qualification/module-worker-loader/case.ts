import type { QualificationCase, QualificationContext, QualificationObservation } from '../schema/types'
import { moduleWorkerLoaderExpected } from './expected'

export async function runModuleWorkerLoader(
  _context: QualificationContext,
): Promise<QualificationObservation> {
  return {
    status: 'fail',
    stage: 'worker-load',
    error: {
      message: 'importScripts is unavailable from the module worker loader.',
    },
  }
}

export const moduleWorkerLoaderCase: QualificationCase = {
  id: 'module-worker-loader',
  description: 'Reproduces Qwen LiteRT loader behavior in a module worker.',
  environments: [{
    runtimePackage: '@litertjs/core',
    runtimeVersion: '2.5.3',
    requestedBackend: 'wasm',
  }],
  expected: moduleWorkerLoaderExpected,
  run: runModuleWorkerLoader,
}
