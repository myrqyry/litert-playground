import { describe, it, expect } from 'vitest'
import type { TensorSpec, ModelMetadata, ModelAdapter } from './types'

describe('TensorSpec', () => {
  it('defines a valid input spec', () => {
    const spec: TensorSpec = {
      name: 'input',
      dtype: 'float32',
      shape: [1, 256, 1],
      description: 'Audio input frame'
    }
    expect(spec.name).toBe('input')
  })

  it('supports optional constraints', () => {
    const spec: TensorSpec = {
      name: 'temperature_harmonic',
      dtype: 'float32',
      shape: [1],
      description: 'Harmonic temperature',
      constraints: { min: 0.1, max: 5.0 }
    }
    expect(spec.constraints!.min).toBe(0.1)
  })
})

describe('ModelMetadata', () => {
  it('holds model info', () => {
    const meta: ModelMetadata = {
      name: 'Magenta RealTime 2',
      description: 'Music generation model',
      modelPath: '/models/magenta.tflite',
      tags: ['audio', 'music']
    }
    expect(meta.name).toContain('Magenta')
  })
})

describe('ModelAdapter interface', () => {
  it('is structurally compatible', () => {
    const adapter: ModelAdapter = {
      modelId: 'test',
      metadata: { name: 'Test', description: '', modelPath: '', tags: [] },
      inputSpecs: [],
      outputSpecs: [],
      prepareInputs: () => ({}),
      parseOutputs: async () => ({})
    }
    expect(adapter.modelId).toBe('test')
  })
})
