import { describe, expect, it } from 'vitest'
import {
  checkAudioValid,
  checkMultiVectorEmbeddingValid,
  InferenceError,
  type AudioInferenceResult,
  type Capability,
  type InferenceResult,
  type Pipeline,
  type RetrievalResult,
  type VisionLanguageResult,
} from './index'

describe('inference core', () => {
  it('keeps audio validation model-agnostic', () => {
    expect(checkAudioValid(new Float32Array(), 24_000, 1, 0)).toContain('audio: empty samples')
  })

  it('exposes shared pipeline and error contracts', () => {
    const pipeline: Pipeline<unknown, AudioInferenceResult> | null = null
    expect(pipeline).toBeNull()
    expect(new InferenceError('CANCELLED', 'cancelled').code).toBe('CANCELLED')
  })

  it('covers the LFM model families in the capability registry', () => {
    const lfm: Capability[] = [
      'image-text-to-text',
      'token-classification',
      'text-classification',
      'reranking',
      'multi-vector-embedding',
      'policy-classification',
    ]
    expect(lfm).toHaveLength(6)
  })

  it('narrows multi-vector embeddings and retrieval results', () => {
    const multi: InferenceResult = {
      kind: 'multi-vector-embedding',
      values: new Float32Array([1, 0, 0, 1]),
      tokens: 2,
      dimensions: 2,
    }
    expect(multi).toMatchObject({ kind: 'multi-vector-embedding', tokens: 2, dimensions: 2 })
    expect(checkMultiVectorEmbeddingValid(multi.values, multi.tokens, multi.dimensions)).toEqual([])
    expect(checkMultiVectorEmbeddingValid(new Float32Array(0), 1, 2)).toContain(
      'multi-vector-embedding: empty values',
    )

    const ranked: RetrievalResult = { id: 'episode-1', score: 0.91, payload: { episode: 'x' } }
    expect(ranked.score).toBe(0.91)
  })

  it('carries a reasoning channel on text results and composes vision-language contracts', () => {
    const text: InferenceResult = { kind: 'text', text: 'answer', reasoning: 'thinking' }
    expect(text.reasoning).toBe('thinking')

    const vl: VisionLanguageResult = {
      text: 'found it',
      boxes: [{ x: 0, y: 0, width: 10, height: 10, score: 0.9 }],
      receipt: {
        modelId: 'lfm2.5-vl-450m',
        pipelineVersion: '0.1.0',
        backend: 'webgpu',
        timestamp: new Date(0).toISOString(),
        loadMs: 1,
        compileMs: 1,
        inferenceMs: 1,
        inputSummary: 'image 10x10 + prompt',
        outputSummary: 'text + 1 box',
        warnings: [],
      },
    }
    expect(vl.boxes).toHaveLength(1)
    expect(vl.receipt.modelId).toBe('lfm2.5-vl-450m')
  })
})
