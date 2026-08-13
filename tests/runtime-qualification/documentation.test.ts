import { describe, expect, it } from 'vitest'
import { readFile } from 'node:fs/promises'

describe('qualification documentation', () => {
  it('documents deterministic and browser lanes', async () => {
    const readme = await readFile(new URL('./README.md', import.meta.url), 'utf8')
    expect(readme).toContain('pnpm test:qualification')
    expect(readme).toContain('pnpm qualify')
    expect(readme).toContain('--case')
    expect(readme).toContain('--backend')
    expect(readme).toContain('2.5.3')
    expect(readme).toContain('matchesExpectation')
  })
})
