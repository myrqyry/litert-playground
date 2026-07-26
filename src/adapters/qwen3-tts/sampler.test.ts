import { describe, it, expect } from 'vitest'
import { sample } from './sampler'

describe('sample', () => {
  it('returns the argmax at temperature=0', () => {
    const logits = new Float32Array([0.1, 0.2, 5.0, 0.0, -1.0])
    const result = sample(logits, { temperature: 0, topK: 0, repetitionPenalty: 1, prevTokens: [] })
    expect(result).toBe(2)
  })

  it('applies repetition penalty', () => {
    const logits = new Float32Array([1.0, 1.0, 1.0])
    const result = sample(logits, { temperature: 0, topK: 0, repetitionPenalty: 1.2, prevTokens: [0] })
    expect(result).toBeGreaterThan(0)
  })

  it('filters to top-k tokens', () => {
    const logits = new Float32Array([0.1, 0.5, 10.0, 0.3, 9.0])
    let top2Count = 0
    for (let trial = 0; trial < 50; trial++) {
      const r = sample(logits, { temperature: 0.5, topK: 2, repetitionPenalty: 1, prevTokens: [] })
      if (r === 2 || r === 4) top2Count++
    }
    expect(top2Count).toBe(50)
  })
})
