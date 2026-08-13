import { readFile } from 'node:fs/promises'
import { describe, expect, it } from 'vitest'

async function text(path: string): Promise<string> {
  return readFile(new URL(`../${path}`, import.meta.url), 'utf8')
}

describe('workspace package boundaries', () => {
  it('keeps examples on public package entrypoints', async () => {
    const qwen = await text('examples/minimal-qwen3-tts/main.tsx')
    const kokoro = await text('examples/minimal-kokoro/main.ts')
    expect(`${qwen}\n${kokoro}`).not.toMatch(/\.\.\/\.\/src|apps\/playground|podqast/)
  })

  it('keeps package manifests directed from generic to specific code', async () => {
    const core = await text('packages/inference-core/package.json')
    const runtime = await text('packages/runtime-litert/package.json')
    const kokoro = await text('packages/kokoro/package.json')
    const qwen = await text('packages/qwen3-tts/package.json')
    expect(core).not.toMatch(/kokoro|qwen3-tts|@litertjs\/core/)
    expect(runtime).toContain('@litert-playground/inference-core')
    expect(kokoro).toContain('@litert-playground/inference-core')
    expect(qwen).toContain('@litert-playground/inference-core')
  })

  it('keeps the playground on the shared managed runtime instead of rebuilding LiteRT plumbing', async () => {
    const runner = await text('apps/playground/src/hooks/useModelRunner.ts')
    expect(runner).toContain("from '@litert-playground/runtime-litert'")
    expect(runner).toContain('createLiteRtRuntime')
    expect(runner).not.toMatch(/\bloadLiteRt\b|\bloadAndCompile\b/)
  })

  it('keeps managed runtime policy generic and product-independent', async () => {
    const runtime = [
      await text('packages/runtime-litert/src/context.ts'),
      await text('packages/runtime-litert/src/coordinator.ts'),
      await text('packages/runtime-litert/src/types.ts'),
    ].join('\n')
    expect(runtime).not.toMatch(/podqast|episode|streamer|earthbound|\bobs\b/i)
    expect(runtime).toContain('preflight')
    expect(runtime).toContain('telemetry')
  })

  it('keeps Kokoro externally consumable through the shared core contract', async () => {
    const manifest = JSON.parse(await text('packages/kokoro/package.json')) as {
      dependencies?: Record<string, string>
      peerDependencies?: Record<string, string>
      devDependencies?: Record<string, string>
    }
    const source = await text('packages/kokoro/src/pipeline.ts')
    const entrypoint = await text('packages/kokoro/src/index.ts')

    expect(manifest.dependencies).toMatchObject({ 'kokoro-js': '^1.2.1' })
    expect(manifest.dependencies?.['@litert-playground/inference-core']).toBeUndefined()
    expect(manifest.peerDependencies).toMatchObject({ '@litert-playground/inference-core': '0.1.x' })
    expect(manifest.devDependencies).toMatchObject({ '@litert-playground/inference-core': 'workspace:*' })
    expect(source).toMatch(
      /from ['"]@litert-playground\/inference-core['"]/,
    )
    expect(entrypoint).toContain('KokoroPipeline')
    expect(entrypoint).toContain('kokoroManifest')
    expect(entrypoint).toContain('KokoroInput')
    expect(entrypoint).toContain('KokoroConfig')
  })

  it('keeps runtime and Qwen packages externally consumable through peer contracts', async () => {
    const runtime = JSON.parse(await text('packages/runtime-litert/package.json')) as {
      dependencies?: Record<string, string>
      peerDependencies?: Record<string, string>
      devDependencies?: Record<string, string>
    }
    const qwen = JSON.parse(await text('packages/qwen3-tts/package.json')) as {
      dependencies?: Record<string, string>
      peerDependencies?: Record<string, string>
      devDependencies?: Record<string, string>
    }

    expect(runtime.dependencies?.['@litert-playground/inference-core']).toBeUndefined()
    expect(runtime.peerDependencies).toMatchObject({ '@litert-playground/inference-core': '0.1.x' })
    expect(runtime.devDependencies).toMatchObject({ '@litert-playground/inference-core': 'workspace:*' })

    expect(qwen.dependencies?.['@litert-playground/inference-core']).toBeUndefined()
    expect(qwen.dependencies?.['@litert-playground/runtime-litert']).toBeUndefined()
    expect(qwen.peerDependencies).toMatchObject({
      '@litert-playground/inference-core': '0.1.x',
      '@litert-playground/runtime-litert': '0.1.x',
    })
    expect(qwen.devDependencies).toMatchObject({
      '@litert-playground/inference-core': 'workspace:*',
      '@litert-playground/runtime-litert': 'workspace:*',
    })
    expect(qwen.dependencies?.['@litertjs/core']).toBe('^2.5.3')
  })
})
