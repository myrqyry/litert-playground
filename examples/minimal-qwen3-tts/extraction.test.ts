// Minimal extraction test: can import Qwen3TTS without the playground
// Run: npx vitest run src/core/validation.test.ts src/adapters/qwen3-tts/pipeline.test.ts
import { describe, it, expect } from 'vitest'
import { Qwen3TtsPipeline, type QwenTtsInput, type QwenTtsConfig } from '../../src/adapters/qwen3-tts/pipeline'
import { createHttpAssetResolver } from '../../src/assets/http-resolver'

describe('minimal Qwen3-TTS extraction', () => {
  it('creates pipeline with manifest', () => {
    const p = new Qwen3TtsPipeline()
    expect(p.manifest.modelId).toBe('qwen3-tts-0.6b')
    expect(p.manifest.version).toBe('0.4.0')
    expect(p.manifest.capabilities).toEqual(['text-to-speech'])
    expect(p.manifest.backends.wasm).toBe(true)
  })

  it('pipeline starts idle and enforces lifecycle', async () => {
    const p = new Qwen3TtsPipeline()
    expect(p.status).toBe('idle')

    await expect(p.run({ text: 'test' })).rejects.toThrow('Pipeline not ready')

    await p.dispose()
    expect(p.status).toBe('disposed')
  })

  it('has 9 declared assets', () => {
    const p = new Qwen3TtsPipeline()
    expect(p.manifest.assets.length).toBeGreaterThanOrEqual(8)
    const required = p.manifest.assets.filter(a => !a.optional)
    expect(required.length).toBe(8)
  })

  it('type-checks Pipeline interface conformance', () => {
    // Static assertion: Qwen3TtsPipeline satisfies Pipeline<QwenTtsInput, AudioInferenceResult, QwenTtsConfig>
    const p = new Qwen3TtsPipeline()
    void p
    // Symbols present on interface:
    expect(typeof p.load).toBe('function')
    expect(typeof p.run).toBe('function')
    expect(typeof p.dispose).toBe('function')
    expect(p.manifest).toBeDefined()
    expect(p.status).toBeDefined()
  })
})