import { describe, expect, it } from 'vitest'
import { runBrowserQualification } from './browserHarness'
import type { QualificationCase } from '../schema/types'

describe('browser qualification harness', () => {
  it('passes the requested backend to an injected launcher', async () => {
    const calls: string[] = []
    const testCase: QualificationCase = {
      id: 'case',
      description: 'fake browser case',
      environments: [{
        runtimePackage: '@litertjs/core',
        runtimeVersion: '2.5.3',
        requestedBackend: 'wasm',
      }],
      expected: { status: 'pass' },
      run: async (context) => {
        calls.push(context.requestedBackend)
        return { status: 'pass', resolvedBackend: context.requestedBackend }
      },
    }

    const results = await runBrowserQualification([testCase], {
      launch: {
        browserName: 'chromium',
        headless: true,
      },
      selection: { backends: ['wasm'] },
      playgroundRevision: 'abc',
      runtimePackage: '@litertjs/core',
      runtimeVersion: '2.5.3',
      launcher: async (_options, run) => run({
        browser: 'Chromium',
        browserVersion: '140',
        webgpuAvailable: false,
      }),
    })

    expect(calls).toEqual(['wasm'])
    expect(results[0]).toMatchObject({ matchesExpectation: true })
  })
})
