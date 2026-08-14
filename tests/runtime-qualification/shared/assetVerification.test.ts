import { describe, expect, it } from 'vitest'
import { verifyQualificationAsset } from './assetVerification'

const asset = {
  id: 'model',
  url: 'https://example.test/model.tflite',
  bytes: 3,
  sha256: '039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81',
}

describe('qualification asset verification', () => {
  it('accepts bytes matching both immutable facts', async () => {
    await expect(verifyQualificationAsset(new Uint8Array([1, 2, 3]).buffer, asset))
      .resolves.toBeInstanceOf(ArrayBuffer)
  })

  it('rejects a hash mismatch after checking the complete buffer', async () => {
    await expect(verifyQualificationAsset(new Uint8Array([1, 2, 4]).buffer, asset))
      .rejects.toThrow('expected SHA-256')
  })
})
