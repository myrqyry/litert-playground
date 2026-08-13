import type { QualificationCase } from '../schema/types'

export const moduleWorkerLoaderExpected: QualificationCase['expected'] = {
  status: 'known-limitation',
  error: {
    stage: 'worker-load',
    messagePattern: 'importScripts|module worker|loader',
  },
}
