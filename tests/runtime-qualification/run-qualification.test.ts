import { describe, expect, it } from 'vitest'
import { formatQualificationMatrix, parseQualificationArgs } from './run-qualification'

describe('qualification CLI', () => {
  it('parses repeated case and backend filters', () => {
    expect(parseQualificationArgs([
      '--case', 'qwen-xnnpack-prefill',
      '--case', 'module-worker-loader',
      '--backend', 'wasm',
    ])).toEqual({
      caseIds: ['qwen-xnnpack-prefill', 'module-worker-loader'],
      backends: ['wasm'],
      browserName: 'chromium',
      headed: false,
    })
  })

  it('formats one compact row per result', () => {
    expect(formatQualificationMatrix([{
      schemaVersion: 1,
      caseId: 'case',
      timestamp: '2026-08-13T00:00:00.000Z',
      playgroundRevision: 'abc',
      runtimePackage: '@litertjs/core',
      runtimeVersion: '2.5.3',
      environment: {
        runtimePackage: '@litertjs/core',
        runtimeVersion: '2.5.3',
        requestedBackend: 'wasm',
      },
      expected: { status: 'pass' },
      observed: { status: 'pass' },
      matchesExpectation: true,
    }])).toBe('case\twasm\tpass\tmatch')
  })
})
