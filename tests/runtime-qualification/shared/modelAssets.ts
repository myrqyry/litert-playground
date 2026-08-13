import type { ModelAssetDescriptor } from '../schema/types'

export function assertImmutableAssetDescriptor(
  asset: ModelAssetDescriptor,
): void {
  if (!asset.id) throw new Error('Asset id is required')

  const url = new URL(asset.url)
  if (url.protocol !== 'https:') {
    throw new Error(`Asset ${asset.id} must use HTTPS`)
  }

  if (!Number.isInteger(asset.bytes) || asset.bytes < 1) {
    throw new Error(`Asset ${asset.id} must declare a positive byte count`)
  }

  if (!/^[a-f0-9]{64}$/.test(asset.sha256)) {
    throw new Error(`Asset ${asset.id} must declare a lowercase SHA-256 hash`)
  }
}

export function isImmutableAsset(asset: ModelAssetDescriptor): boolean {
  try {
    assertImmutableAssetDescriptor(asset)
    return true
  } catch {
    return false
  }
}

export function getAssetCacheKey(asset: ModelAssetDescriptor): string {
  assertImmutableAssetDescriptor(asset)
  return `${asset.id}:${asset.url}:${asset.bytes}:${asset.sha256}`
}
