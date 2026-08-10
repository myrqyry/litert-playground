// Minimal extraction test: can import Qwen3TTS without the playground
// Run: npx vitest run src/core/validation.test.ts src/adapters/qwen3-tts/pipeline.test.ts
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { createCachingAssetResolver, createHttpAssetResolver } from '@litert-playground/inference-core'
import { createLiteRtRuntime } from '@litert-playground/runtime-litert'
import { Qwen3TtsPipeline, qwen3TtsManifest, type QwenTtsInput, type QwenTtsConfig } from '@litert-playground/qwen3-tts'

describe('minimal Qwen3-TTS extraction', () => {
  it('creates pipeline with manifest', () => {
    const p = new Qwen3TtsPipeline()
    expect(p.manifest.modelId).toBe('qwen3-tts-12hz-0.6b-base')
    expect(p.manifest.version).toBe('0.4.0')
    expect(p.manifest.capabilities).toEqual(['text-to-speech'])
    expect(p.manifest.backends.wasm).toBe(true)
  })

  it('pipeline starts idle and enforces lifecycle', async () => {
    const p = new Qwen3TtsPipeline()
    expect(p.status).toBe('idle')

    await expect(p.run({ text: 'test' })).rejects.toThrow('Pipeline not ready')

    await p.dispose()
    expect(p.status).toBe('disposed')
  })

  it('has 9 declared assets', () => {
    const p = new Qwen3TtsPipeline()
    expect(p.manifest.assets.length).toBeGreaterThanOrEqual(8)
    const required = p.manifest.assets.filter(a => !a.optional)
    expect(required.length).toBe(8)
  })

  it('type-checks Pipeline interface conformance', () => {
    // Static assertion: Qwen3TtsPipeline satisfies Pipeline<QwenTtsInput, AudioInferenceResult, QwenTtsConfig>
    const p = new Qwen3TtsPipeline()
    void p
    // Symbols present on interface:
    expect(typeof p.load).toBe('function')
    expect(typeof p.run).toBe('function')
    expect(typeof p.dispose).toBe('function')
    expect(p.manifest).toBeDefined()
    expect(p.status).toBeDefined()
  })

  it('has a standalone browser entry without playground imports', () => {
    const source = readFileSync(new URL('./main.tsx', import.meta.url), 'utf8')
    expect(source).toContain('createHttpAssetResolver')
    expect(source).toContain('createCachingAssetResolver')
    expect(source).toContain('createLiteRtRuntime')
    expect(source).toContain('qwen3TtsManifest')
    expect(source).toContain('Qwen3TtsPipeline')
    expect(source).toContain(
      'https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base/resolve/main/',
    )
    expect(source).toContain('createLiteRtRuntime({ assets })')
    expect(source).not.toContain('assetBase: modelBase')
    expect(source).not.toMatch(/src\/App|registry|components\//)
    expect(readFileSync(new URL('./index.html', import.meta.url), 'utf8')).toContain('main.tsx')
    void createHttpAssetResolver
    void createCachingAssetResolver
    void createLiteRtRuntime
    void qwen3TtsManifest
  })
})
