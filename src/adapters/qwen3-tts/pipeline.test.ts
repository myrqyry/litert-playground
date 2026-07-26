import { describe, it, expect } from 'vitest'
import { Qwen3TtsPipeline } from './pipeline'

describe('Qwen3TtsPipeline', () => {
  it('throws on synthesize before any models are loaded', async () => {
    const p = new Qwen3TtsPipeline('/models')
    await expect(p.synthesize('hello')).rejects.toThrow('Pipeline not loaded')
  })
})
