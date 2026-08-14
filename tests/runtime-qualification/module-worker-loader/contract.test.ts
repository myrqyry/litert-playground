import { describe, expect, it } from 'vitest'
import { readFile } from 'node:fs/promises'
import { moduleWorkerLoaderCase } from './case'

describe('module worker loader contract', () => {
  it('keeps the module worker failure separate from classic workers', async () => {
    const worker = await readFile(new URL('./worker.ts', import.meta.url), 'utf8')
    expect(worker).toContain("from '@litertjs/core'")
    expect(worker).toContain('loadLiteRt')
    expect(worker).toContain('postMessage')
    expect(worker).toContain('catch')
    expect(moduleWorkerLoaderCase.expected).toMatchObject({
      status: 'known-limitation',
      error: { stage: 'worker-load' },
    })
  })

  it('delegates the observation to the browser worker adapter', async () => {
    const observation = await moduleWorkerLoaderCase.run({
      requestedBackend: 'wasm',
      environment: moduleWorkerLoaderCase.environments[0],
      fetchAsset: async () => new ArrayBuffer(0),
      runtime: {
        runModuleWorkerLoader: async () => ({ status: 'pass' }),
      },
    })

    expect(observation).toEqual({ status: 'pass' })
  })
})
