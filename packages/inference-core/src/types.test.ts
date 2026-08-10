import { describe, expect, it } from 'vitest'
import { checkAudioValid, InferenceError, type AudioInferenceResult, type Pipeline } from './index'

describe('inference core', () => {
  it('keeps audio validation model-agnostic', () => {
    expect(checkAudioValid(new Float32Array(), 24_000, 1, 0)).toContain('audio: empty samples')
  })

  it('exposes shared pipeline and error contracts', () => {
    const pipeline: Pipeline<unknown, AudioInferenceResult> | null = null
    expect(pipeline).toBeNull()
    expect(new InferenceError('CANCELLED', 'cancelled').code).toBe('CANCELLED')
  })
})
