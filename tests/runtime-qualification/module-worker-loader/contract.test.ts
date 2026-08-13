import { describe, expect, it } from 'vitest'
import { readFile } from 'node:fs/promises'
import { moduleWorkerLoaderCase } from './case'

describe('module worker loader contract', () => {
  it('keeps the module worker failure separate from classic workers', async () => {
    const worker = await readFile(new URL('./worker.ts', import.meta.url), 'utf8')
    expect(worker).toContain('self')
    expect(worker).toContain('postMessage')
    expect(moduleWorkerLoaderCase.expected).toMatchObject({
      status: 'known-limitation',
      error: { stage: 'worker-load' },
    })
  })
})
