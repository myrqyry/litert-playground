import type { QualificationContext, QualificationObservation } from '../schema/types'
import { runTinyLitertBaseline } from './case'

export function runTinyLitertBaselineInBrowser(
  context: QualificationContext,
): Promise<QualificationObservation> {
  return runTinyLitertBaseline(context)
}
