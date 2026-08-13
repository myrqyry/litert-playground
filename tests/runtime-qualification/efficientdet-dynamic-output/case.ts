import type { QualificationCase, QualificationContext, QualificationObservation } from '../schema/types'
import { efficientDetExpected } from './expected'

export async function runEfficientDetDynamicOutput(
  _context: QualificationContext,
): Promise<QualificationObservation> {
  return {
    status: 'fail',
    stage: 'output-materialization',
    error: {
      message: 'Dynamic output tensor materialization is unavailable in LiteRT.js WASM.',
    },
  }
}

export const efficientDetDynamicOutputCase: QualificationCase = {
  id: 'efficientdet-dynamic-output',
  description: 'Reproduces EfficientDet dynamic output materialization on WASM.',
  environments: [{
    runtimePackage: '@litertjs/core',
    runtimeVersion: '2.5.3',
    requestedBackend: 'wasm',
  }],
  expected: efficientDetExpected,
  run: runEfficientDetDynamicOutput,
}
