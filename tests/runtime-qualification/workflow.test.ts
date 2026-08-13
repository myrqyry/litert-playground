import { describe, expect, it } from 'vitest'
import { readFile } from 'node:fs/promises'

describe('runtime qualification workflow', () => {
  it('keeps browser qualification opt-in', async () => {
    const workflow = await readFile(new URL('../../.github/workflows/runtime-qualification.yml', import.meta.url), 'utf8')
    expect(workflow).toContain('workflow_dispatch:')
    expect(workflow).toContain('pnpm qualify')
    expect(workflow).toContain('tests/runtime-qualification/results')
  })

  it('keeps qualification tests in the normal verify gate', async () => {
    const packageJson = JSON.parse(await readFile(new URL('../../package.json', import.meta.url), 'utf8')) as {
      scripts: Record<string, string>
    }
    expect(packageJson.scripts.verify).toContain('pnpm test:qualification')
    expect(packageJson.scripts.verify).not.toContain('pnpm qualify')
  })
})
