import { type ModelAsset, type AssetResolver, InferenceError } from '../core/types'

export type { ModelAsset, AssetResolver }

export function createHttpAssetResolver(baseUrl: string): AssetResolver {
  return {
    async resolve(asset: ModelAsset, signal?: AbortSignal): Promise<ArrayBuffer> {
      const url = new URL(asset.path, baseUrl).href
      try {
        const resp = await fetch(url, { signal })
        if (!resp.ok) {
          throw new InferenceError('ASSET_FETCH_FAILED', `HTTP ${resp.status} fetching ${asset.id}`, { asset: asset.id })
        }
        return resp.arrayBuffer()
      } catch (e) {
        if (e instanceof InferenceError) throw e
        throw new InferenceError('ASSET_FETCH_FAILED', `Failed to fetch ${asset.id}: ${String(e)}`, {
          asset: asset.id,
          cause: e,
        })
      }
    },

    async stream(asset: ModelAsset, signal?: AbortSignal): Promise<ReadableStream<Uint8Array>> {
      const url = new URL(asset.path, baseUrl).href
      const resp = await fetch(url, { signal })
      if (!resp.ok || !resp.body) {
        throw new InferenceError('ASSET_FETCH_FAILED', `HTTP ${resp.status} streaming ${asset.id}`, { asset: asset.id })
      }
      return resp.body
    },
  }
}

// ponytail: single key-value store, add LRU if cache-hit rate drops
export function createCachingAssetResolver(inner: AssetResolver): AssetResolver {
  const cache = new Map<string, Promise<ArrayBuffer>>()

  return {
    async resolve(asset: ModelAsset, signal?: AbortSignal): Promise<ArrayBuffer> {
      const key = asset.path
      if (cache.has(key)) return cache.get(key)!
      const p = inner.resolve({ ...asset, path: asset.path }, signal)
      cache.set(key, p)
      void p.catch(() => {
        if (cache.get(key) === p) cache.delete(key)
      })
      return p
    },

    stream: inner.stream ? (asset, signal) => inner.stream!(asset, signal) : undefined,
  }
}
