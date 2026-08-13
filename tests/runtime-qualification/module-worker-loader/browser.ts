import type { QualificationContext, QualificationObservation } from '../schema/types'
import { runModuleWorkerLoader } from './case'

export function runModuleWorkerLoaderInBrowser(
  context: QualificationContext,
): Promise<QualificationObservation> {
  return runModuleWorkerLoader(context)
}
