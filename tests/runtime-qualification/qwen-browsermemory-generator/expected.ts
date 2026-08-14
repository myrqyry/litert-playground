import type { QualificationCase } from '../schema/types'

export const qwenBrowserMemoryGeneratorExpected: QualificationCase['expected'] = {
  status: 'known-limitation',
  error: {
    stage: 'talker-prefill',
    messagePattern: 'tensor_buffer|TensorBuffer|runtime',
  },
}
