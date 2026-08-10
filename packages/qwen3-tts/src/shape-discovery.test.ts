import { describe, expect, it } from 'vitest'
import { discoverCodecShapes, discoverMtpShapes, discoverTalkerShapes } from './shape-discovery'

function details(names: Array<[string, number[]]>) {
  return names.map(([name, shape]) => ({ name, index: 0, dtype: 'float32' as const, shape: Int32Array.from(shape), supportedBufferTypes: new Set() }))
}

describe('Qwen compiled-model shape discovery', () => {
  it('discovers talker KV names, shapes, and mask length', () => {
    const decode = { getInputDetails: () => details([
      ['embeddings', [1, 1, 1024]],
      ['mask', [1, 1, 1, 128]],
      ['kv_cache_0', [1, 8, 128, 128]],
      ['kv_cache_1', [1, 8, 128, 128]],
    ]) }
    const model = { signatures: { decode }, getInputDetails: decode.getInputDetails } as never

    expect(discoverTalkerShapes(model)).toEqual({
      kvNames: ['kv_cache_0', 'kv_cache_1'],
      kvShapes: [[1, 8, 128, 128], [1, 8, 128, 128]],
      cacheLen: 128,
    })
  })

  it('discovers MTP cache shape and codec chunk size', () => {
    const mtp = { getInputDetails: () => details([
      ['args_2', [1, 1, 1, 23]],
      ['args_3', [1, 23, 1024]],
    ]) } as never
    const codec = { getInputDetails: () => details([['args_0', [1, 16, 96]]]) } as never

    expect(discoverMtpShapes(mtp)).toEqual({ cacheLen: 23, kvShape: [1, 23, 1024] })
    expect(discoverCodecShapes(codec)).toEqual({ chunkSize: 96 })
  })
})
