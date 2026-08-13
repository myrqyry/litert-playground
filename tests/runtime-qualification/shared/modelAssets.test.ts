import { describe, expect, it } from 'vitest'
import {
  assertImmutableAssetDescriptor,
  getAssetCacheKey,
  isImmutableAsset,
} from './modelAssets'
import type { ModelAssetDescriptor } from '../schema/types'

const asset: ModelAssetDescriptor = {
  id: 'model',
  url: 'https://example.test/model.tflite',
  bytes: 4,
  sha256: 'a'.repeat(64),
}

describe('qualification model assets', () => {
  it('accepts immutable HTTPS descriptors', () => {
    expect(() => assertImmutableAssetDescriptor(asset)).not.toThrow()
    expect(isImmutableAsset(asset)).toBe(true)
  })

  it.each([
    ['http://example.test/model.tflite', 4, 'a'.repeat(64)],
    [asset.url, 0, asset.sha256],
    [asset.url, 4, 'not-a-hash'],
  ])('rejects mutable descriptor values', (url, bytes, sha256) => {
    const invalid = { ...asset, url, bytes, sha256 }
    expect(() => assertImmutableAssetDescriptor(invalid)).toThrow()
    expect(isImmutableAsset(invalid)).toBe(false)
  })

  it('includes identity and content hash in its cache key', () => {
    const key = getAssetCacheKey(asset)

    expect(key).toContain(asset.id)
    expect(key).toContain(asset.url)
    expect(key).toContain(asset.sha256)
  })
})
