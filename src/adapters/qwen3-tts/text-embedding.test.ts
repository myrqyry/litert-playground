import { describe, it, expect } from 'vitest'
import { TextEmbedder } from './text-embedding'

describe('TextEmbedder', () => {
  it('embeds token IDs and projects to output dim', () => {
    const dim = 4
    const outputDim = 2
    const hiddenDim = outputDim * 4
    const w1Size = dim * hiddenDim
    const b1Size = hiddenDim
    const w2Size = hiddenDim * outputDim
    const b2Size = outputDim
    const total = w1Size + b1Size + w2Size + b2Size
    const proj = new Float32Array(total)

    // w1[0,0] = 1 so hidden[0] gets input[0]
    proj[0] = 1
    // w2[0,0] = 1 so out[0] gets hidden[0]
    proj[w1Size + b1Size + 0] = 1
    // w2[1,1] = 1 so out[1] gets hidden[1]
    proj[w1Size + b1Size + 1 * hiddenDim + 1] = 1

    const embedding = new Float32Array(4 * dim)
    embedding[0] = 2
    embedding[1] = 3
    embedding[4] = 5
    embedding[5] = 7

    const embedder = new TextEmbedder(embedding, proj, dim, outputDim)
    const result = embedder.embed([0, 1])
    expect(result.length).toBe(outputDim)
    expect(result.some(v => v !== 0)).toBe(true)
  })

  it('computes silu correctly', () => {
    const embedder = new TextEmbedder(new Float32Array(4), new Float32Array(1), 1, 1)
    expect(embedder.silu(0)).toBeCloseTo(0)
    expect(embedder.silu(1)).toBeGreaterThan(0)
    expect(embedder.silu(-2)).toBeLessThan(0)
  })
})
