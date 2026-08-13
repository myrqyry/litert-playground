import type { QualificationCase } from '../schema/types'

export const efficientDetExpected: QualificationCase['expected'] = {
  status: 'known-limitation',
  error: {
    stage: 'output-materialization',
    messagePattern: 'dynamic|output|tensor',
  },
}
