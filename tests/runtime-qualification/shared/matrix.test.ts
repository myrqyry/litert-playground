import { describe, expect, it } from 'vitest'
import {
  runQualificationCase,
  selectQualificationCases,
} from './matrix'
import type {
  QualificationCase,
  QualificationContext,
} from '../schema/types'

const context = (requestedBackend: 'wasm' | 'webgpu'): QualificationContext => ({
  requestedBackend,
  environment: {
    runtimePackage: '@litertjs/core',
    runtimeVersion: '2.5.3',
    requestedBackend,
  },
  fetchAsset: async () => new ArrayBuffer(0),
  runtime: {},
})

const qualificationCase: QualificationCase = {
  id: 'case',
  description: 'controlled case',
  evidenceKind: 'contract',
  environments: [context('wasm').environment],
  expected: { status: 'pass' },
  run: async ({ requestedBackend }) => ({
    status: 'pass',
    resolvedBackend: requestedBackend,
  }),
}

describe('qualification matrix', () => {
  it('selects cases by id and backend', () => {
    expect(selectQualificationCases(
      [qualificationCase],
      { caseIds: ['case'], backends: ['wasm'] },
    )).toEqual([qualificationCase])
  })

  it('rejects an unknown explicit case', () => {
    expect(() => selectQualificationCases(
      [qualificationCase],
      { caseIds: ['missing'] },
    )).toThrow('missing')
  })

  it('executes a case and creates evidence', async () => {
    const result = await runQualificationCase(
      qualificationCase,
      context('wasm'),
      {
        playgroundRevision: 'abc',
        runtimePackage: '@litertjs/core',
        runtimeVersion: '2.5.3',
        now: () => new Date('2026-08-13T00:00:00.000Z'),
      },
    )

    expect(result).toMatchObject({
      caseId: 'case',
      observed: { status: 'pass', resolvedBackend: 'wasm' },
      matchesExpectation: true,
    })
  })

  it('does not run a case on an undeclared backend', async () => {
    const results = await (await import('./matrix')).runQualificationMatrix(
      [qualificationCase],
      { backends: ['webgpu'] },
      [context('wasm'), context('webgpu')],
      {
        playgroundRevision: 'abc',
        runtimePackage: '@litertjs/core',
        runtimeVersion: '2.5.3',
      },
    )

    expect(results).toHaveLength(0)
  })

  it('reports unavailable WebGPU without running the model', async () => {
    let ran = false
    const webgpuCase: QualificationCase = {
      ...qualificationCase,
      id: 'webgpu-case',
      evidenceKind: 'browser-observation',
      environments: [context('webgpu').environment],
      run: async () => {
        ran = true
        return { status: 'pass' }
      },
    }
    const unavailable = context('webgpu')
    unavailable.environment.webgpuAvailable = false

    const [result] = await (await import('./matrix')).runQualificationMatrix(
      [webgpuCase],
      {},
      [unavailable],
      {
        playgroundRevision: 'abc',
        runtimePackage: '@litertjs/core',
        runtimeVersion: '2.5.3',
      },
    )

    expect(ran).toBe(false)
    expect(result.observed).toMatchObject({
      status: 'unsupported',
      stage: 'capability',
      error: { code: 'BACKEND_UNAVAILABLE' },
    })
    expect(result.matchesExpectation).toBe(false)
  })
})
