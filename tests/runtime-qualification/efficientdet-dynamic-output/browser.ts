import type { QualificationContext, QualificationObservation } from '../schema/types'
import { runEfficientDetDynamicOutput } from './case'

export function runEfficientDetDynamicOutputInBrowser(
  context: QualificationContext,
): Promise<QualificationObservation> {
  return runEfficientDetDynamicOutput(context)
}
