import { describe, expect, it, vi } from 'vitest'
import { createCachingAssetResolver, createHttpAssetResolver } from './http-resolver'

describe('HTTP asset resolver', () => {
  it('passes the signal to resolve fetch', async () => {
    const signal = new AbortController().signal
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(new Uint8Array([1, 2, 3]), { status: 200 }),
    )

    await createHttpAssetResolver('https://assets.test/').resolve(
      { id: 'model', path: 'model.bin' }, signal,
    )

    expect(fetchMock).toHaveBeenCalledWith('https://assets.test/model.bin', { signal })
    fetchMock.mockRestore()
  })

  it('passes the signal to stream fetch', async () => {
    const signal = new AbortController().signal
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(new Uint8Array([1]), { status: 200 }),
    )

    await createHttpAssetResolver('https://assets.test/').stream!(
      { id: 'model', path: 'model.bin' }, signal,
    )

    expect(fetchMock).toHaveBeenCalledWith('https://assets.test/model.bin', { signal })
    fetchMock.mockRestore()
  })

  it('evicts a rejected cached request so retry can succeed', async () => {
    const inner = {
      resolve: vi.fn()
        .mockRejectedValueOnce(new Error('temporary'))
        .mockResolvedValueOnce(new ArrayBuffer(1)),
    }
    const resolver = createCachingAssetResolver(inner)
    const asset = { id: 'model', path: 'model.bin' }

    await expect(resolver.resolve(asset)).rejects.toThrow('temporary')
    await expect(resolver.resolve(asset)).resolves.toBeInstanceOf(ArrayBuffer)
    expect(inner.resolve).toHaveBeenCalledTimes(2)
  })
})
