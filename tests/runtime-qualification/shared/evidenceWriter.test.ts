import { describe, expect, it } from 'vitest'
import {
  createQualificationResult,
  matchQualificationExpectation,
  normalizeQualificationError,
  writeQualificationResult,
} from './evidenceWriter'
import type {
  QualificationCase,
  QualificationObservation,
} from '../schema/types'
import { mkdtemp, readFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

describe('qualification evidence', () => {
  it('matches a passing observation', () => {
    expect(matchQualificationExpectation(
      { status: 'pass' },
      { status: 'pass' },
    )).toBe(true)
  })

  it('matches a confirmed known limitation', () => {
    const expected: QualificationCase['expected'] = {
      status: 'known-limitation',
      error: {
        code: 'ASSET_FETCH_FAILED',
        stage: 'prefill',
        messagePattern: 'XNNPACK',
      },
    }
    const observed: QualificationObservation = {
      status: 'fail',
      stage: 'prefill',
      error: {
        code: 'ASSET_FETCH_FAILED',
        stage: 'prefill',
        message: 'XNNPACK runtime creation failed',
      },
    }

    expect(matchQualificationExpectation(expected, observed)).toBe(true)
  })

  it('does not treat an upstream fix as a known limitation match', () => {
    expect(matchQualificationExpectation(
      { status: 'known-limitation' },
      { status: 'pass' },
    )).toBe(false)
  })

  it('normalizes errors with a fallback stage', () => {
    expect(normalizeQualificationError(new Error('broken'), 'load')).toEqual({
      message: 'broken',
      stage: 'load',
    })
  })

  it('writes JSON evidence without runtime-only values', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'qualification-'))
    const result = createQualificationResult({
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
    })

    const path = await writeQualificationResult(directory, result)
    expect(JSON.parse(await readFile(path, 'utf8'))).toEqual(result)
  })
})
