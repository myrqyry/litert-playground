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
})
