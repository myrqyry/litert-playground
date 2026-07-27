export function parseNpy(buffer: ArrayBuffer): Float32Array {
  const view = new DataView(buffer)
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]
  for (let i = 0; i < 6; i++) {
    if (view.getUint8(i) !== magic[i]) throw new Error('Invalid .npy magic')
  }
  const version = view.getUint8(6)
  const headerLen = version === 1
    ? view.getUint32(8, true)
    : Number(view.getBigUint64(8, true))
  const hdrOff = version === 1 ? 12 : 16
  const headerStr = new TextDecoder().decode(
    new Uint8Array(buffer, hdrOff, headerLen)
  )
  const descrMatch = headerStr.match(/'descr':\s*'<([fiu])(\d+)'/)
  const shapeMatch = headerStr.match(/'shape':\s*\(([^)]+)\)/)
  if (!descrMatch || !shapeMatch) throw new Error('Failed to parse .npy header')

  const dtype = descrMatch[1] as 'f' | 'i' | 'u'
  const bytesPer = parseInt(descrMatch[2])
  const shapeStr = shapeMatch[1]
    .replace(/\s/g, '')
    .split(',')
    .filter(s => s.length > 0)
    .map(Number)
  const totalElems = shapeStr.reduce((a, b) => a * b, 1)

  const dataOff = hdrOff + headerLen
  const raw = new Uint8Array(buffer).slice(dataOff, dataOff + totalElems * bytesPer)

  if (dtype === 'f') {
    if (bytesPer === 4) return new Float32Array(raw.buffer)
    if (bytesPer === 2) {
      const u16 = new Uint16Array(raw.buffer)
      const out = new Float32Array(totalElems)
      for (let i = 0; i < totalElems; i++) {
        const f = u16[i]
        const sign = (f >> 15) & 1
        const exp = (f >> 10) & 0x1f
        const mant = f & 0x3ff
        if (exp === 0) {
          out[i] = mant === 0 ? (sign ? -0 : 0) : (sign ? -1 : 1) * Math.pow(2, -14) * (mant / 1024)
        } else if (exp === 31) {
          out[i] = mant === 0 ? (sign ? -Infinity : Infinity) : NaN
        } else {
          out[i] = (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024)
        }
      }
      return out
    }
  }
  throw new Error(`Unsupported dtype: ${dtype}${bytesPer}`)
}

export async function parseNpz(buffer: ArrayBuffer): Promise<Record<string, Float32Array>> {
  const u8 = new Uint8Array(buffer)
  const result: Record<string, Float32Array> = {}
  let offset = 0
  while (offset + 30 <= u8.length) {
    if (
      u8[offset] !== 0x50 || u8[offset + 1] !== 0x4b ||
      u8[offset + 2] !== 0x03 || u8[offset + 3] !== 0x04
    ) {
      if (u8[offset] === 0x50 && u8[offset + 1] === 0x4b) break
      offset++
      continue
    }
    const view = new DataView(buffer, offset)
    const compMethod = view.getUint16(8, true)
    const compSize = view.getUint32(18, true)
    const uncompSize = view.getUint32(22, true)
    const nameLen = view.getUint16(26, true)
    const extraLen = view.getUint16(28, true)
    const name = new TextDecoder().decode(
      new Uint8Array(buffer, offset + 30, nameLen)
    )
    const dataOff = offset + 30 + nameLen + extraLen

    if (name.endsWith('.npy')) {
      const key = name.replace('.npy', '')
      if (compMethod === 0) {
        result[key] = parseNpy(buffer.slice(dataOff, dataOff + compSize))
      } else if (compMethod === 8) {
        if (typeof window !== 'undefined') {
          throw new Error('Decompressed .npz data is not supported in the browser — use uncompressed .npy files')
        }
        const { inflateSync } = await import(/* @vite-ignore */ 'zlib')
        const inflated = inflateSync(new Uint8Array(buffer, dataOff, compSize))
        result[key] = parseNpy(inflated.buffer as ArrayBuffer)
      }
    }
    offset = dataOff + compSize
  }
  return result
}
