import { describe, expect, it } from 'vitest'
import { qwen3TtsManifest, createQwen3TtsManifest, qwen3TtsVariants } from './manifest'

describe('Qwen variants', () => {
  it('keeps artifact filenames in variant metadata', () => {
    const manifest = createQwen3TtsManifest(qwen3TtsVariants.int4)
    expect(manifest.assets.find((asset) => asset.id === 'talker')?.path).toBe('talker_int4.tflite')
    expect(manifest.name).toContain('int4')
  })

  it('matches the official Qwen3-TTS LiteRT repository', () => {
    expect(qwen3TtsManifest.modelId).toBe('qwen3-tts-12hz-0.6b-base')
    expect(qwen3TtsManifest.assets.find(asset => asset.id === 'tokenizer')?.bytes)
      .toBe(11_424_262)
    expect(qwen3TtsManifest.assets.find(asset => asset.id === 'talker')?.path)
      .toBe('talker_fp32.tflite')
    expect(qwen3TtsManifest.memory.downloadBytes).toBe(3_415_668_132)
  })
})
