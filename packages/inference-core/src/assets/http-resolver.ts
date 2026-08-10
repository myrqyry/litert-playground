import {
  type AssetProgress,
  type AssetRequestOptions,
  type AssetResolver,
  type ModelAsset,
  InferenceError,
} from '../types'

function progressFor(asset: ModelAsset, options: AssetRequestOptions | undefined, loadedBytes: number, totalBytes?: number): void {
  options?.onProgress?.({ asset: asset.id, loadedBytes, totalBytes })
}

function requestError(asset: ModelAsset, error: unknown, signal?: AbortSignal): InferenceError {
  const code = signal?.aborted ? 'CANCELLED' : 'ASSET_FETCH_FAILED'
  return new InferenceError(code, `Failed to fetch ${asset.id}: ${String(error)}`, { asset: asset.id, cause: error })
}

async function responseStream(asset: ModelAsset, response: Response, options?: AssetRequestOptions): Promise<ReadableStream<Uint8Array>> {
  if (!response.ok || !response.body) {
    throw new InferenceError('ASSET_FETCH_FAILED', `HTTP ${response.status} streaming ${asset.id}`, { asset: asset.id })
  }
  const totalBytes = Number(response.headers.get('content-length') ?? asset.bytes ?? 0) || undefined
  let loadedBytes = 0
  const reader = response.body.getReader()
  return new ReadableStream({
    async pull(controller) {
      try {
        const next = await reader.read()
        if (next.done) {
          controller.close()
          return
        }
        loadedBytes += next.value.byteLength
        progressFor(asset, options, loadedBytes, totalBytes)
        controller.enqueue(next.value)
      } catch (error) {
        controller.error(requestError(asset, error, options?.signal))
      }
    },
    async cancel(reason) {
      await reader.cancel(reason)
    },
  })
}

export function createHttpAssetResolver(baseUrl: string): AssetResolver {
  return {
    async resolve(asset: ModelAsset, options?: AssetRequestOptions): Promise<ArrayBuffer> {
      const url = new URL(asset.path, baseUrl).href
      try {
        const response = await fetch(url, { signal: options?.signal })
        const stream = await responseStream(asset, response, options)
        const reader = stream.getReader()
        const chunks: Uint8Array[] = []
        let total = 0
        while (true) {
          const next = await reader.read()
          if (next.done) break
          chunks.push(next.value)
          total += next.value.byteLength
        }
        const bytes = new Uint8Array(total)
        let offset = 0
        for (const chunk of chunks) {
          bytes.set(chunk, offset)
          offset += chunk.byteLength
        }
        return bytes.buffer
      } catch (error) {
        if (error instanceof InferenceError) throw error
        throw requestError(asset, error, options?.signal)
      }
    },

    async stream(asset: ModelAsset, options?: AssetRequestOptions): Promise<ReadableStream<Uint8Array>> {
      const url = new URL(asset.path, baseUrl).href
      try {
        return await responseStream(asset, await fetch(url, { signal: options?.signal }), options)
      } catch (error) {
        if (error instanceof InferenceError) throw error
        throw requestError(asset, error, options?.signal)
      }
    },
  }
}

export function createCachingAssetResolver(inner: AssetResolver): AssetResolver {
  const cache = new Map<string, Promise<ArrayBuffer>>()
  return {
    resolve(asset: ModelAsset, options?: AssetRequestOptions): Promise<ArrayBuffer> {
      const key = asset.path
      const existing = cache.get(key)
      if (existing) return existing
      const promise = inner.resolve(asset, options)
      cache.set(key, promise)
      void promise.catch(() => {
        if (cache.get(key) === promise) cache.delete(key)
      })
      return promise
    },
    stream: inner.stream ? (asset, options) => inner.stream!(asset, options) : undefined,
  }
}

export type { AssetProgress, AssetRequestOptions, AssetResolver, ModelAsset }
