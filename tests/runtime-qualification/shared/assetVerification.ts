import type { ModelAssetDescriptor } from '../schema/types'

export async function verifyQualificationAsset(
  buffer: ArrayBuffer,
  asset: Pick<ModelAssetDescriptor, 'id' | 'bytes' | 'sha256'>,
): Promise<ArrayBuffer> {
  if (buffer.byteLength !== asset.bytes) {
    throw new Error(
      `Asset ${asset.id} expected ${asset.bytes} bytes, got ${buffer.byteLength}`,
    )
  }

  const digest = new Uint8Array(
    await globalThis.crypto.subtle.digest('SHA-256', buffer),
  )
  const actual = Array.from(digest, (value) => value.toString(16).padStart(2, '0')).join('')
  if (actual !== asset.sha256) {
    throw new Error(`Asset ${asset.id} expected SHA-256 ${asset.sha256}, got ${actual}`)
  }
  return buffer
}
