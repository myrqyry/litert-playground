import { describe, expect, it, vi } from 'vitest'
import {
  EncoderPipeline,
  encoder230mManifest,
  encoderPolicyLinterManifest,
  encoderSpellcheckerManifest,
  meanPool,
  selectEncoderManifest,
} from './index'

vi.mock('@huggingface/transformers', () => ({
  AutoTokenizer: {
    from_pretrained: vi.fn(async () => ({
      encode: vi.fn(async () => ({
        input_ids: { data: new Int32Array([1, 1, 1, 1]), dims: [1, 4] },
      })),
    })),
  },
}))

const context = {
  backend: 'wasm' as const,
  assets: { resolve: vi.fn() },
  signal: undefined,
  liteRt: {
    loadModel: vi.fn(async () => fakeModel),
    loadNpy: vi.fn(),
    fetchBuffer: vi.fn(),
  },
}

const fakeModel = {
  run: vi.fn(async () => [{ data: async () => new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) }]),
}

describe('selectEncoderManifest', () => {
  it('maps capability to the right manifest', () => {
    expect(selectEncoderManifest('text-embedding')).toBe(encoder230mManifest)
    expect(selectEncoderManifest('token-classification')).toBe(encoderSpellcheckerManifest)
    expect(selectEncoderManifest('policy-classification')).toBe(encoderPolicyLinterManifest)
  })
})

describe('EncoderPipeline', () => {
  it('declares the embedding capability', () => {
    expect(encoder230mManifest.capabilities).toContain('text-embedding')
  })

  it('loads model and tokenizer from manifest paths', async () => {
    const pipeline = new EncoderPipeline({ manifest: encoder230mManifest })
    await pipeline.load(context)
    expect(context.liteRt.loadModel).toHaveBeenCalledWith(
      'litert-community/LFM2.5-Encoder-230M/resolve/main/LFM2.5-Encoder-230M_fp16.tflite',
    )
    expect(pipeline.status).toBe('ready')
  })

  it('mean-pools embeddings over tokens', async () => {
    const pipeline = new EncoderPipeline({ manifest: encoder230mManifest })
    await pipeline.load(context)
    const result = await pipeline.run({ text: 'hello' }, { maxTokens: 4 })
    expect(result).toEqual({ kind: 'embedding', values: meanPool(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]), 4, 2), dimensions: 2 })
  })

  it('returns raw token scores for token-classification manifests', async () => {
    const pipeline = new EncoderPipeline({ manifest: encoderSpellcheckerManifest })
    await pipeline.load(context)
    const result = await pipeline.run({ text: 'helo' }, { maxTokens: 4 })
    expect(result.kind).toBe('token-classification')
    if (result.kind === 'token-classification') {
      expect(result.tokens).toBe(4)
      expect(result.dimensions).toBe(2)
    }
  })
})

describe('meanPool', () => {
  it('averages rows', () => {
    const pooled = meanPool(new Float32Array([2, 4, 6, 8]), 2, 2)
    expect(Array.from(pooled)).toEqual([4, 6])
  })

  it('returns zeros for empty input', () => {
    expect(Array.from(meanPool(new Float32Array(0), 2, 3))).toEqual([0, 0, 0])
  })
})
