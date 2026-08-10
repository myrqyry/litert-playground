import { describe, it, expect } from 'vitest'
import { Qwen3TtsPipeline } from './pipeline'

describe('Qwen3TtsPipeline', () => {
  it('throws on run before load', async () => {
    const p = new Qwen3TtsPipeline()
    await expect(p.run({ text: 'hello' })).rejects.toThrow('Pipeline not ready')
  })

  it('has correct manifest', () => {
    const p = new Qwen3TtsPipeline()
    expect(p.manifest.modelId).toBe('qwen3-tts-12hz-0.6b-base')
    expect(p.manifest.capabilities).toContain('text-to-speech')
  })

  it('starts idle', () => {
    const p = new Qwen3TtsPipeline()
    expect(p.status).toBe('idle')
  })
})
