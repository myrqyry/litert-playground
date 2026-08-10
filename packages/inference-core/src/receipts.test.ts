import { describe, expect, it, vi } from 'vitest'
import { createInferenceReceipt } from './receipts'

describe('inference receipts', () => {
  it('adds shared timing and environment fields', () => {
    vi.stubGlobal('navigator', { userAgent: 'test-browser' })
    const now = vi.spyOn(performance, 'now').mockReturnValue(25)
    const receipt = createInferenceReceipt({
      manifest: { modelId: 'model', version: '1.0.0' },
      backend: 'wasm', loadMs: 2, compileMs: 3, inferenceStart: 10,
      inputSummary: 'input', outputSummary: 'output', warnings: [],
    })
    expect(receipt.inferenceMs).toBe(15)
    expect(receipt.environment).toBe('browser: test-browser')
    now.mockRestore()
  })
})
