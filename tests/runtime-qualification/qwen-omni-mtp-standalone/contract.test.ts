import { describe, expect, it } from 'vitest'
import { qwenOmniMtpStandaloneCase } from './case'

describe('standalone Omni MTP qualification contract', () => {
  it('records the exact passing Omni MTP descriptor', () => {
    expect(qwenOmniMtpStandaloneCase.evidenceKind).toBe('browser-observation')
    expect(qwenOmniMtpStandaloneCase.model).toMatchObject({
      id: 'qwen3-tts-browser-memory',
      variant: 'browserMemory',
      revision: '791880469d874546d884a0e6cf68564a61c04ca9',
      assets: [{
        id: 'mtp',
        bytes: 440528628,
        sha256: '7e808fb554fdf443e70e5ccdd3fdccd3cd74cdec606d3375fa4c5877d4f46e0b',
      }],
    })
    expect(qwenOmniMtpStandaloneCase.expected).toEqual({ status: 'pass' })
  })
})
