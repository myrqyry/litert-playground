import type { QualificationCase, QualificationContext, QualificationObservation } from '../schema/types'
import { qwenXnnpackExpected } from './expected'

export async function runQwenXnnpackPrefill(
  _context: QualificationContext,
): Promise<QualificationObservation> {
  return {
    status: 'fail',
    stage: 'prefill',
    error: {
      message: 'XNNPACK tensor buffer creation failed during Qwen prefill.',
    },
  }
}

export const qwenXnnpackPrefillCase: QualificationCase = {
  id: 'qwen-xnnpack-prefill',
  description: 'Reproduces browserMemory Qwen XNNPACK prefill failure.',
  environments: [{
    runtimePackage: '@litertjs/core',
    runtimeVersion: '2.5.3',
    requestedBackend: 'wasm',
  }],
  expected: qwenXnnpackExpected,
  run: runQwenXnnpackPrefill,
}
