export interface Fp16Table {
  readonly storage: Uint16Array
  readonly width: number
  readonly length: number
  row(index: number): Float32Array
}

function halfToFloat(value: number): number {
  const sign = (value & 0x8000) ? -1 : 1
  const exponent = (value >> 10) & 0x1f
  const mantissa = value & 0x3ff
  if (exponent === 0) return sign * 2 ** -14 * (mantissa / 1024)
  if (exponent === 31) return mantissa === 0 ? sign * Infinity : NaN
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024)
}

export function createFp16Table(storage: Uint16Array, width: number): Fp16Table {
  if (width <= 0 || storage.length % width !== 0) throw new RangeError('invalid FP16 table width')
  return {
    storage,
    width,
    length: storage.length,
    row(index: number) {
      if (index < 0 || (index + 1) * width > storage.length) throw new RangeError(`FP16 row out of range: ${index}`)
      const output = new Float32Array(width)
      const offset = index * width
      for (let i = 0; i < width; i++) output[i] = halfToFloat(storage[offset + i])
      return output
    },
  }
}

export function parseFp16Npy(buffer: ArrayBuffer): Fp16Table {
  const bytes = new Uint8Array(buffer)
  if (bytes[0] !== 0x93 || new TextDecoder().decode(bytes.subarray(1, 6)) !== 'NUMPY') throw new Error('Invalid .npy magic')
  const version = bytes[6]
  const view = new DataView(buffer)
  const headerLength = version === 1 ? view.getUint16(8, true) : view.getUint32(8, true)
  const headerOffset = version === 1 ? 10 : 12
  const header = new TextDecoder().decode(bytes.subarray(headerOffset, headerOffset + headerLength))
  const descr = header.match(/'descr':\s*'([^']+)'/)
  const shape = header.match(/'shape':\s*\(([^)]*)\)/)
  if (!descr || !shape || descr[1] !== '<f2') throw new Error('Expected little-endian FP16 NPY table')
  const dimensions = shape[1].split(',').map((value) => Number(value.trim())).filter(Number.isFinite)
  if (dimensions.length !== 2) throw new Error('Expected a two-dimensional FP16 NPY table')
  const dataOffset = headerOffset + headerLength
  return createFp16Table(new Uint16Array(buffer, dataOffset, dimensions[0] * dimensions[1]), dimensions[1])
}
