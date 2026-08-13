import { describe, expect, it } from 'vitest'
import asset from './fixtures/asset.json'
import { isImmutableAsset } from '../shared/modelAssets'
import { tinyLitertBaselineCase } from './case'

describe('tiny LiteRT baseline contract', () => {
  it('uses a verified immutable model descriptor', () => {
    expect(isImmutableAsset(asset)).toBe(true)
  })

  it('declares passing WASM and WebGPU environments', () => {
    expect(tinyLitertBaselineCase.expected).toEqual({ status: 'pass' })
    expect(tinyLitertBaselineCase.environments).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ requestedBackend: 'wasm' }),
        expect.objectContaining({ requestedBackend: 'webgpu' }),
      ]),
    )
  })
})
