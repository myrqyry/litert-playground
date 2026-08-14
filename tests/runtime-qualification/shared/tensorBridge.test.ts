import { describe, expect, it } from 'vitest'
import {
  createQualificationTypedArray,
  createQualificationZeroTypedArray,
  serializeQualificationInput,
} from './tensorBridge'

describe('qualification tensor bridge', () => {
  it('creates the LiteRT-supported typed array for each dtype', () => {
    expect(createQualificationTypedArray('float32', [1.5])).toBeInstanceOf(Float32Array)
    expect(createQualificationTypedArray('int32', [2])).toBeInstanceOf(Int32Array)
    expect(createQualificationTypedArray('uint8', [3])).toBeInstanceOf(Uint8Array)
  })

  it('preserves named inputs and signature selection', () => {
    expect(serializeQualificationInput({
      signature: 'serving_default',
      input: {
        token_ids: { data: new Int32Array([1, 2]), shape: [1, 2], dtype: 'int32' },
      },
    })).toEqual({
      signature: 'serving_default',
      input: {
        kind: 'named',
        tensors: {
          token_ids: { data: [1, 2], shape: [1, 2], dtype: 'int32' },
        },
      },
    })
  })

  it('rejects a dtype the installed LiteRT.js bridge cannot materialize', () => {
    expect(() => createQualificationTypedArray('float16', [1])).toThrow('Unsupported tensor dtype')
  })

  it('allocates zero-filled tensors without an intermediate number array', () => {
    expect(createQualificationZeroTypedArray('int32', 2)).toEqual(new Int32Array(2))
  })
})
