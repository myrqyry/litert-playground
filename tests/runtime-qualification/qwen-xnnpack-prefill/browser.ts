import type { QualificationContext, QualificationObservation } from '../schema/types'
import { runQwenXnnpackPrefill } from './case'

export function runQwenXnnpackPrefillInBrowser(
  context: QualificationContext,
): Promise<QualificationObservation> {
  return runQwenXnnpackPrefill(context)
}
