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
        id: 'talker',
        bytes: 255998768,
        sha256: 'e03df54e73ed1f88b2ae6d47bbf82dd64ea90a3620d753a0f3c8d6a8d60848db',
      }],
    })
    expect(qwenXnnpackPrefillCase.expected).toMatchObject({
      status: 'known-limitation',
      error: { stage: 'prefill' },
    })
  })
})
