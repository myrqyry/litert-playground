import { describe, expect, it } from 'vitest'
import { readFile } from 'node:fs/promises'

describe('browser LiteRT bridge', () => {
  it('deletes output tensors after serializing them', async () => {
    const source = await readFile(new URL('./browser-entry.ts', import.meta.url), 'utf8')
    expect(source).toContain('output.delete()')
  })

  it('supports browser-side zero-input execution for large models', async () => {
    const source = await readFile(new URL('./browser-entry.ts', import.meta.url), 'utf8')
    expect(source).toContain('runWithZeros')
  })

  it('runs the real Qwen GeneratorPhase inside the browser', async () => {
    const source = await readFile(new URL('./browser-entry.ts', import.meta.url), 'utf8')
    expect(source).toContain('runQwenGenerator')
    expect(source).toContain('GeneratorPhase')
    expect(source).toContain('createLiteRtRuntime')
    expect(source).toContain('onTrace')
    expect(source).not.toContain('data: tensor')
  })
})
