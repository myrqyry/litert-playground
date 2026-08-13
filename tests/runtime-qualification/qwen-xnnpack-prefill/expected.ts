import type { QualificationCase } from '../schema/types'

export const qwenXnnpackExpected: QualificationCase['expected'] = {
  status: 'known-limitation',
  error: {
    stage: 'prefill',
    messagePattern: 'tensor_buffer|XNNPACK|runtime',
  },
}
