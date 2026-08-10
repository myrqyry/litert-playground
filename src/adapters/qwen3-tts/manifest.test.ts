import { describe, expect, it } from 'vitest'
import { qwen3TtsManifest } from './manifest'

describe('qwen3TtsManifest', () => {
  it('reports the sum of required asset sizes', () => {
    const expected = qwen3TtsManifest.assets
      .filter(asset => !asset.optional)
      .reduce((total, asset) => total + (asset.bytes ?? 0), 0)

    expect(qwen3TtsManifest.memory.downloadBytes).toBe(expected)
    expect(qwen3TtsManifest.memory.downloadBytes).not.toBe(
      qwen3TtsManifest.assets.reduce((total, asset) => total + (asset.bytes ?? 0), 0),
    )
  })
})
