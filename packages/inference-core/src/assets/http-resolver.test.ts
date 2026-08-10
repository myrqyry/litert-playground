import { describe, expect, it, vi } from 'vitest'
import { createCachingAssetResolver, createHttpAssetResolver } from './http-resolver'

describe('HTTP asset resolver', () => {
  it('forwards abort signals and reports buffered progress', async () => {
    const signal = new AbortController().signal
    const progress: number[] = []
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(new Uint8Array([1, 2, 3]), { status: 200 }),
    )

    await createHttpAssetResolver('https://assets.test/').resolve(
      { id: 'model', path: 'model.bin' },
      { signal, onProgress: ({ loadedBytes }) => progress.push(loadedBytes) },
    )

    expect(fetchMock).toHaveBeenCalledWith('https://assets.test/model.bin', { signal })
    expect(progress[progress.length - 1]).toBe(3)
    fetchMock.mockRestore()
  })

  it('forwards abort signals and reports streamed progress', async () => {
    const signal = new AbortController().signal
    const progress: number[] = []
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(new Uint8Array([1]), { status: 200 }),
    )
    const stream = await createHttpAssetResolver('https://assets.test/').stream!(
      { id: 'model', path: 'model.bin' },
      { signal, onProgress: ({ loadedBytes }) => progress.push(loadedBytes) },
    )
    await stream.cancel()

    expect(fetchMock).toHaveBeenCalledWith('https://assets.test/model.bin', { signal })
    expect(progress).toEqual([1])
    fetchMock.mockRestore()
  })

  it('evicts rejected cached requests so retries can succeed', async () => {
    const inner = {
      resolve: vi.fn().mockRejectedValueOnce(new Error('temporary')).mockResolvedValueOnce(new ArrayBuffer(1)),
    }
    const resolver = createCachingAssetResolver(inner)
    const asset = { id: 'model', path: 'model.bin' }

    await expect(resolver.resolve(asset)).rejects.toThrow('temporary')
    await expect(resolver.resolve(asset)).resolves.toBeInstanceOf(ArrayBuffer)
    expect(inner.resolve).toHaveBeenCalledTimes(2)
  })

  it('names HTTP failures with the asset ID', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(null, { status: 503 }))
    await expect(createHttpAssetResolver('https://assets.test/').resolve({ id: 'talker', path: 'talker.tflite' }))
      .rejects.toMatchObject({ code: 'ASSET_FETCH_FAILED', asset: 'talker' })
    vi.restoreAllMocks()
  })
})
