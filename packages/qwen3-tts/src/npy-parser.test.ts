import { describe, it, expect } from 'vitest'
import { parseNpy, parseNpz } from './npy-parser'

describe('parseNpy', () => {
  it('parses a simple float32 .npy header and data', () => {
    const magic = new Uint8Array([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59])
    const header = "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }"
    const headerBytes = new TextEncoder().encode(header)
    const padLen = (64 - (10 + headerBytes.length) % 64) % 64
    const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    const buf = new Uint8Array(10 + headerBytes.length + padLen + data.byteLength)
    let off = 0
    buf.set(magic, off); off += magic.length
    buf[off++] = 1; buf[off++] = 0
    const dv = new DataView(buf.buffer, off, 2)
    dv.setUint16(0, headerBytes.length + padLen, true); off += 2
    buf.set(headerBytes, off); off += headerBytes.length
    off += padLen
    buf.set(new Uint8Array(data.buffer), off)

    const result = parseNpy(buf.buffer)
    expect(result.length).toBe(12)
    expect(Array.from(result.slice(0, 4))).toEqual([1, 2, 3, 4])
  })

  it('rejects invalid magic bytes', () => {
    const buf = new ArrayBuffer(128)
    const view = new Uint8Array(buf)
    view[0] = 0x00; view[1] = 0x4e; view[2] = 0x55; view[3] = 0x4d; view[4] = 0x50; view[5] = 0x59
    expect(() => parseNpy(buf)).toThrow('Invalid .npy magic')
  })

  it('parses float16 as float32 output', () => {
    const header = "{'descr': '<f2', 'fortran_order': False, 'shape': (2,), }"
    const magic = new Uint8Array([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59])
    const hb = new TextEncoder().encode(header)
    const padLen = (64 - (10 + hb.length) % 64) % 64
    const f16 = new Uint16Array([0x3c00, 0x4000])
    const buf = new Uint8Array(10 + hb.length + padLen + f16.byteLength)
    let off = 0
    buf.set(magic, off); off += magic.length
    buf[off++] = 1; buf[off++] = 0
    const dv = new DataView(buf.buffer, off, 2)
    dv.setUint16(0, hb.length + padLen, true); off += 2
    buf.set(hb, off); off += hb.length
    off += padLen
    buf.set(new Uint8Array(f16.buffer), off)

    const result = parseNpy(buf.buffer)
    expect(result.length).toBe(2)
    expect(Math.round(result[0])).toBe(1)
    expect(Math.round(result[1])).toBe(2)
  })

  it('rejects unsupported dtype', () => {
    const header = "{'descr': '<i4', 'fortran_order': False, 'shape': (2,), }"
    const magic = new Uint8Array([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59])
    const hb = new TextEncoder().encode(header)
    const padLen = (64 - (10 + hb.length) % 64) % 64
    const buf = new Uint8Array(10 + hb.length + padLen + 8)
    let off = 0
    buf.set(magic, off); off += magic.length
    buf[off++] = 1; buf[off++] = 0
    const dv = new DataView(buf.buffer, off, 2)
    dv.setUint16(0, hb.length + padLen, true); off += 2
    buf.set(hb, off); off += hb.length
    expect(() => parseNpy(buf.buffer)).toThrow(/Unsupported dtype/)
  })
})

function npy(values: number[]): ArrayBuffer {
  const headerText = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }"
  const padding = 16 - ((10 + headerText.length + 1) % 16)
  const header = `${headerText}${' '.repeat(padding - 1)}\n`
  const bytes = new Uint8Array(10 + header.length + values.length * 4)
  bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0])
  new DataView(bytes.buffer).setUint16(8, header.length, true)
  bytes.set(new TextEncoder().encode(header), 10)
  bytes.set(new Uint8Array(new Float32Array(values).buffer), 10 + header.length)
  return bytes.buffer
}

function zip64Npz(data: ArrayBuffer): ArrayBuffer {
  const name = new TextEncoder().encode('w1.npy')
  const extraLength = 20
  const dataOffset = 30 + name.length + extraLength
  const bytes = new Uint8Array(dataOffset + data.byteLength)
  const view = new DataView(bytes.buffer)
  view.setUint32(0, 0x04034b50, true)
  view.setUint16(6, 0, true)
  view.setUint16(8, 0, true)
  view.setUint32(18, 0xffffffff, true)
  view.setUint32(22, 0xffffffff, true)
  view.setUint16(26, name.length, true)
  view.setUint16(28, extraLength, true)
  bytes.set(name, 30)
  view.setUint16(36, 0x0001, true)
  view.setUint16(38, 16, true)
  view.setBigUint64(40, BigInt(data.byteLength), true)
  view.setBigUint64(48, BigInt(data.byteLength), true)
  bytes.set(new Uint8Array(data), dataOffset)
  return bytes.buffer
}

describe('parseNpz', () => {
  it('parses a .npz with a single stored .npy entry', async () => {
    const npyMagic = new Uint8Array([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59])
    const npyHeader = "{'descr': '<f4', 'fortran_order': False, 'shape': (4,), }"
    const npyHb = new TextEncoder().encode(npyHeader)
    const npyTotalHdr = npyMagic.length + 1 + 1 + 2 + npyHb.length
    const npyPad = (64 - (npyTotalHdr % 64)) % 64
    const npyData = new Float32Array([1, 2, 3, 4])
    const npyBytes = new Uint8Array(npyTotalHdr + npyPad + npyData.byteLength)
    let off = 0
    npyBytes.set(npyMagic, off); off += npyMagic.length
    npyBytes[off++] = 1; npyBytes[off++] = 0
    const dv = new DataView(npyBytes.buffer, off, 2)
    dv.setUint16(0, npyHb.length + npyPad, true); off += 2
    npyBytes.set(npyHb, off); off += npyHb.length; off += npyPad
    npyBytes.set(new Uint8Array(npyData.buffer), off)

    const nameBytes = new TextEncoder().encode('test.npy')
    const localHdr = new Uint8Array(30)
    const lv = new DataView(localHdr.buffer)
    lv.setUint32(0, 0x04034b50, true)
    lv.setUint16(4, 20, true)
    lv.setUint16(8, 0, true)
    lv.setUint16(10, 0, true)
    lv.setUint16(12, 0, true)
    lv.setUint32(14, 0, true)
    lv.setUint32(18, npyBytes.length, true)
    lv.setUint32(22, npyBytes.length, true)
    lv.setUint16(26, nameBytes.length, true)
    lv.setUint16(28, 0, true)

    const npz = new Uint8Array(localHdr.length + nameBytes.length + npyBytes.length)
    npz.set(localHdr, 0)
    npz.set(nameBytes, localHdr.length)
    npz.set(npyBytes, localHdr.length + nameBytes.length)

    const result = await parseNpz(npz.buffer)
    expect(result['test']).toBeDefined()
    expect(Array.from(result['test'])).toEqual([1, 2, 3, 4])
  })

  it('reads ZIP64 sizes for uncompressed NPY entries', async () => {
    const parsed = await parseNpz(zip64Npz(npy([1.5, -2])))
    expect(Array.from(parsed.w1)).toEqual([1.5, -2])
  })
})
