import { describe, expect, it } from 'vitest'
import { qwenXnnpackPrefillCase } from './case'

describe('Qwen XNNPACK prefill contract', () => {
  it('uses the smallest real browserMemory prefill model descriptor', () => {
    expect(qwenXnnpackPrefillCase.evidenceKind).toBe('browser-observation')
    expect(qwenXnnpackPrefillCase.model).toMatchObject({
      id: 'qwen3-tts-browser-memory',
      variant: 'browserMemory',
      revision: '0eb3b8a4714972b065c160faec6a12158caa9dc0',
      assets: [{
        id: 'mtp',
        bytes: 229608368,
        sha256: 'f5ab8f826e3dd68f14667af422145fe57233b445046e5ef42c01b59f82191b4b',
      }],
    })
    expect(qwenXnnpackPrefillCase.expected).toMatchObject({
      status: 'known-limitation',
      error: { stage: 'prefill' },
    })
  })
})
