import { describe, expect, it } from 'vitest'
import { efficientDetDynamicOutputCase } from './case'

describe('EfficientDet dynamic output contract', () => {
  it('records a WASM known limitation without claiming a pass', () => {
    expect(efficientDetDynamicOutputCase.expected.status).toBe('known-limitation')
    expect(efficientDetDynamicOutputCase.expected.error?.stage).toBe('output-materialization')
    expect(efficientDetDynamicOutputCase.model).toBeUndefined()
  })
})
