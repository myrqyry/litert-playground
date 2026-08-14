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
        bytes: 440528628,
        sha256: '7e808fb554fdf443e70e5ccdd3fdccd3cd74cdec606d3375fa4c5877d4f46e0b',
      }],
    })
    expect(qwenXnnpackPrefillCase.expected).toMatchObject({
      status: 'known-limitation',
      error: { stage: 'prefill' },
    })
  })
})
