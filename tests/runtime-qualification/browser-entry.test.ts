import { describe, expect, it } from 'vitest'
import { readFile } from 'node:fs/promises'

describe('browser LiteRT bridge', () => {
  it('deletes output tensors after serializing them', async () => {
    const source = await readFile(new URL('./browser-entry.ts', import.meta.url), 'utf8')
    expect(source).toContain('output.delete()')
  })
})
