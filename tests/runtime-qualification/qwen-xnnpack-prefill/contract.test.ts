import { describe, expect, it } from 'vitest'
import { qwenXnnpackPrefillCase } from './case'

describe('Qwen XNNPACK prefill contract', () => {
  it('records browserMemory WASM as a known limitation', () => {
    expect(qwenXnnpackPrefillCase.model).toBeUndefined()
    expect(qwenXnnpackPrefillCase.expected).toMatchObject({
      status: 'known-limitation',
      error: { stage: 'prefill' },
    })
  })
})
