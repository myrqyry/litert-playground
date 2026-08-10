import { describe, expect, it, vi } from 'vitest'
import { Qwen3TtsPipeline } from './pipeline'

vi.mock('./prompt', () => ({
  buildPrompt: () => ({
    prefill: new Float32Array(1024),
    trailing: [new Float32Array(1024)],
    ttsPad: new Float32Array(1024),
  }),
}))
vi.mock('./sampler', () => ({ sample: () => 1 }))
vi.mock('./npy-parser', () => ({
  parseNpy: () => new Float32Array(1024),
  parseNpz: vi.fn(),
}))
vi.mock('@litert-playground/inference-core', async () => ({
  ...(await vi.importActual<typeof import('@litert-playground/inference-core')>('@litert-playground/inference-core')),
  checkAudioValid: () => ['audio warning'],
}))

describe('Qwen3TtsPipeline receipts', () => {
  it('attaches an automatic receipt to successful audio output', async () => {
    const signal = new AbortController().signal
    const assets = { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(1)) }
    const pipeline = new Qwen3TtsPipeline()
    const internals = pipeline as any

    internals.context = {
      backend: 'webgpu',
      assets,
      liteRt: {},
    }
    internals.tokenizer = { encode: () => [], decode: () => '' }
    internals.codecEmb = new Float32Array(2048)
    internals.mtpEmb = new Float32Array(0)
    internals.talker = {
      createEmptyKv: vi.fn().mockReturnValue({}),
      prefill: vi.fn().mockResolvedValue({
        logits: new Float32Array(3072), hidden: new Float32Array(1024), kvCache: {},
      }),
      decode: vi.fn().mockResolvedValue({
        logits: new Float32Array(3072), hidden: new Float32Array(1024), kvCache: {},
      }),
    }
    internals.mtp = { predict: vi.fn().mockResolvedValue([]) }
    internals.codec = { decode: vi.fn().mockResolvedValue(new Float32Array([0, 0.1, 0.2])) }
    pipeline.status = 'ready'

    const result = await pipeline.run({ text: 'hello' }, { maxFrames: 1 }, signal)

    expect(result.kind).toBe('audio')
    expect(result.receipt.modelId).toBe('qwen3-tts-12hz-0.6b-base')
    expect(result.receipt.pipelineVersion).toBe('0.4.0')
    expect(result.receipt.backend).toBe('webgpu')
    expect(Number.isNaN(Date.parse(result.receipt.timestamp))).toBe(false)
    expect(result.receipt.inferenceMs).toBeGreaterThanOrEqual(0)
    expect(result.receipt.inputSummary).toContain('5 characters')
    expect(result.receipt.outputSummary).toContain('24000Hz')
    expect(result.receipt.warnings).toEqual(['audio warning'])
    expect(assets.resolve).toHaveBeenCalledWith(
      { id: 'voice', path: 'voices/demo_speaker.npy', optional: true }, { signal },
    )
  })
})
