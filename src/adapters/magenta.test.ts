import { describe, it, expect, vi } from 'vitest'
import { magentaAdapter } from './magenta'

describe('magentaAdapter', () => {
  it('defines modelId and metadata', () => {
    expect(magentaAdapter.modelId).toBe('magenta-realtime-2')
    expect(magentaAdapter.metadata.name).toContain('Magenta')
    expect(magentaAdapter.metadata.tags).toContain('magenta')
  })

  it('defines input specs', () => {
    expect(magentaAdapter.inputSpecs).toHaveLength(3)
    const names = magentaAdapter.inputSpecs.map(s => s.name)
    expect(names).toEqual(['input', 'length', 'temperature_harmonic'])
  })

  it('defines output specs', () => {
    expect(magentaAdapter.outputSpecs).toHaveLength(2)
    const names = magentaAdapter.outputSpecs.map(s => s.name)
    expect(names).toEqual(['output', 'state'])
  })

  it('input specs have correct shapes', () => {
    const input = magentaAdapter.inputSpecs.find(s => s.name === 'input')!
    expect(input.shape).toEqual([1, 256, 1])
    expect(input.dtype).toBe('float32')
  })

  it('has constraints on temperature and length', () => {
    const temp = magentaAdapter.inputSpecs.find(s => s.name === 'temperature_harmonic')!
    expect(temp.constraints?.min).toBe(0.1)
    expect(temp.constraints?.max).toBe(5.0)

    const len = magentaAdapter.inputSpecs.find(s => s.name === 'length')!
    expect(len.constraints?.min).toBe(0.5)
    expect(len.constraints?.max).toBe(30.0)
  })
})
