import { describe, expect, it } from 'vitest'
import { createFp16Table } from './fp16-table'

describe('FP16 text embedding table', () => {
  it('keeps storage packed and converts only requested rows', () => {
    const storage = new Uint16Array([0x3c00, 0xc000, 0x3800, 0x7c00])
    const table = createFp16Table(storage, 2)

    expect(table.storage).toBe(storage)
    expect(table.row(0)).toEqual(new Float32Array([1, -2]))
    expect(table.row(1)).toEqual(new Float32Array([0.5, Infinity]))
  })
})
