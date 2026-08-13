import { describe, expect, it } from 'vitest'
import { classifyRuntimeFailure, getRuntimeLanes } from './runtime-matrix'

describe('isolated runtime lanes', () => {
  it('keeps production pinned to LiteRT.js 2.5.3', () => {
    expect(getRuntimeLanes()[0]).toMatchObject({
      id: 'production-2.5.3',
      packageName: '@litertjs/core',
      version: '2.5.3',
    })
  })

  it('classifies wrapper, runtime, and backend failures', () => {
    expect(classifyRuntimeFailure({ modelId: 'm', backend: 'wasm', status: 'fail', stage: 'managed-runtime' })).toBe('managed-runtime')
    expect(classifyRuntimeFailure({ modelId: 'm', backend: 'wasm', status: 'fail', stage: 'runtime' })).toBe('litert-runtime')
    expect(classifyRuntimeFailure({ modelId: 'm', backend: 'wasm', status: 'fail', stage: 'backend' })).toBe('backend')
  })
})
