import { describe, expect, it } from 'vitest'
import { efficientDetDynamicOutputCase } from './case'

describe('EfficientDet dynamic output contract', () => {
  it('uses an immutable real model descriptor for browser evidence', () => {
    expect(efficientDetDynamicOutputCase.expected.status).toBe('known-limitation')
    expect(efficientDetDynamicOutputCase.expected.error?.stage).toBe('output-materialization')
    expect(efficientDetDynamicOutputCase.expected.error?.messagePattern).toContain('Target crashed')
    expect(efficientDetDynamicOutputCase.evidenceKind).toBe('browser-observation')
    expect(efficientDetDynamicOutputCase.model).toMatchObject({
      id: 'efficientdet-lite0',
      revision: 'tfhub-v1',
      assets: [{
        id: 'model',
        bytes: 4563519,
        sha256: '2e04c53bfeac0ac2a30c057c7e2a777594ce39baaac35a92f74fb1e8c4fc4e0b',
      }],
    })
  })
})
