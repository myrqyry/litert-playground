import { describe, expect, it } from 'vitest'
import { qwenBrowserMemoryGeneratorCase, traceStages } from './case'

describe('Qwen browserMemory generator qualification contract', () => {
  it('records the explicit Talker and Omni MTP revisions', () => {
    expect(qwenBrowserMemoryGeneratorCase).toMatchObject({
      id: 'qwen-browsermemory-generator',
      evidenceKind: 'browser-observation',
      model: {
        variant: 'browserMemoryOmni',
        revisions: {
          talker: '0eb3b8a4714972b065c160faec6a12158caa9dc0',
          mtp: '791880469d874546d884a0e6cf68564a61c04ca9',
        },
      },
      expected: {
        status: 'known-limitation',
        error: { stage: 'talker-prefill' },
      },
    })
  })

  it('requires all composed-stage receipt names', () => {
    expect(traceStages).toEqual([
      'talker-compile',
      'talker-prefill',
      'talker-output-read',
      'mtp-input-build',
      'mtp-compile',
      'mtp-run',
      'mtp-output-read',
      'state-update',
    ])
  })
})
