import { describe, expect, it } from 'vitest'
import { createQwen3TtsManifest, qwen3TtsVariants } from './manifest'

describe('Qwen variants', () => {
  it('keeps artifact filenames in variant metadata', () => {
    const manifest = createQwen3TtsManifest(qwen3TtsVariants.int4)
    expect(manifest.assets.find((asset) => asset.id === 'talker')?.path).toBe('talker_int4.tflite')
    expect(manifest.name).toContain('int4')
  })
})
