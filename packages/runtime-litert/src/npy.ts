export function parseNpy(buffer: ArrayBuffer): Float32Array {
  const view = new DataView(buffer)
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]
  for (let i = 0; i < 6; i++) if (view.getUint8(i) !== magic[i]) throw new Error('Invalid .npy magic')
  const version = view.getUint8(6)
  const headerLen = version === 1 ? view.getUint32(8, true) : Number(view.getBigUint64(8, true))
  const headerOffset = version === 1 ? 12 : 16
  const header = new TextDecoder().decode(new Uint8Array(buffer, headerOffset, headerLen))
  const descr = header.match(/'descr':\s*'<([fiu])(\d+)'/)
  const shape = header.match(/'shape':\s*\(([^)]+)\)/)
  if (!descr || !shape) throw new Error('Failed to parse .npy header')
  const bytesPer = Number(descr[2])
  const total = shape[1].replace(/\s/g, '').split(',').filter(Boolean).map(Number).reduce((a, b) => a * b, 1)
  const dataOffset = headerOffset + headerLen
  const raw = new Uint8Array(buffer).slice(dataOffset, dataOffset + total * bytesPer)
  if (descr[1] === 'f' && bytesPer === 4) return new Float32Array(raw.buffer)
  if (descr[1] !== 'f' || bytesPer !== 2) throw new Error(`Unsupported dtype: ${descr[1]}${bytesPer}`)
  const input = new Uint16Array(raw.buffer)
  const output = new Float32Array(total)
  for (let i = 0; i < total; i++) {
    const value = input[i]
    const sign = (value >> 15) & 1
    const exponent = (value >> 10) & 0x1f
    const mantissa = value & 0x3ff
    if (exponent === 0) output[i] = mantissa === 0 ? (sign ? -0 : 0) : (sign ? -1 : 1) * 2 ** -14 * (mantissa / 1024)
    else if (exponent === 31) output[i] = mantissa === 0 ? (sign ? -Infinity : Infinity) : NaN
    else output[i] = (sign ? -1 : 1) * 2 ** (exponent - 15) * (1 + mantissa / 1024)
  }
  return output
}
