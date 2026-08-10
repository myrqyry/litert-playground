import { describe, expect, it, vi } from 'vitest'
import { runBonsaiGraph, type BonsaiGraphTensor } from './graph'

function tensor(data: Float32Array): BonsaiGraphTensor {
  return { toTypedArray: () => data, delete: vi.fn() }
}

describe('runBonsaiGraph', () => {
  it('copies outputs and deletes input/output tensors', async () => {
    const input = tensor(new Float32Array([1]))
    const output = tensor(new Float32Array([2, 3]))
    const model = { run: vi.fn().mockResolvedValue([output]) }

    const result = await runBonsaiGraph(model, [{ data: new Float32Array([1]), shape: [1] }], () => input)

    expect(result).toEqual([new Float32Array([2, 3])])
    expect(result[0]).not.toBe(output.toTypedArray())
    expect(input.delete).toHaveBeenCalledOnce()
    expect(output.delete).toHaveBeenCalledOnce()
    expect(model.run).toHaveBeenCalledWith([input])
  })

  it('cleans input tensors when graph execution fails', async () => {
    const input = tensor(new Float32Array([1]))
    const model = { run: vi.fn().mockRejectedValue(new Error('compile failed')) }

    await expect(runBonsaiGraph(model, [{ data: new Float32Array([1]), shape: [1] }], () => input))
      .rejects.toThrow('compile failed')
    expect(input.delete).toHaveBeenCalledOnce()
  })
})
