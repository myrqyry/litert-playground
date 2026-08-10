import { describe, expect, it } from 'vitest'
import { parseNpy } from './npy'

function float32Npy(values: number[]): ArrayBuffer {
  const headerText = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }"
  const padding = 16 - ((10 + headerText.length + 1) % 16)
  const header = `${headerText}${' '.repeat(padding - 1)}\n`
  const bytes = new Uint8Array(10 + header.length + values.length * 4)
  bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0)
  new DataView(bytes.buffer).setUint16(8, header.length, true)
  bytes.set(new TextEncoder().encode(header), 10)
  bytes.set(new Uint8Array(new Float32Array(values).buffer), 10 + header.length)
  return bytes.buffer
}

describe('parseNpy', () => {
  it('parses NumPy v1 headers with a 16-bit header length', () => {
    expect(Array.from(parseNpy(float32Npy([1.5, -2])))).toEqual([1.5, -2])
  })
})
