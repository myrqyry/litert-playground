import { describe, expect, it } from 'vitest'
import {
  createImagePositionIds,
  createTextPositionIds,
  flowMatchSigmas,
  unpatchifyLatent,
} from './host'

describe('Bonsai host transforms', () => {
  it('matches the published four-step sigma schedule shape', () => {
    const sigmas = flowMatchSigmas(4)

    expect(sigmas).toHaveLength(5)
    expect(sigmas[0]).toBeCloseTo(1, 6)
    expect(sigmas[3]).toBeCloseTo(0.71749656, 6)
    expect(sigmas[4]).toBe(0)
    expect(sigmas[0]).toBeGreaterThan(sigmas[1])
    expect(sigmas[1]).toBeGreaterThan(sigmas[2])
  })

  it('creates image and text position IDs in graph order', () => {
    const image = createImagePositionIds(2)
    const text = createTextPositionIds(3)

    expect(image).toEqual(new Float32Array([
      0, 0, 0, 0,
      0, 0, 1, 0,
      0, 1, 0, 0,
      0, 1, 1, 0,
    ]))
    expect(text).toEqual(new Float32Array([
      0, 0, 0, 0,
      0, 0, 0, 1,
      0, 0, 0, 2,
    ]))
  })

  it('unpatchifies packed channels into BCHW latent order', () => {
    const lat = new Float32Array(4)
    lat.set([1, 2, 3, 4])
    const scale = new Float32Array([2, 2, 2, 2])
    const shift = new Float32Array([10, 20, 30, 40])

    const output = unpatchifyLatent(lat, scale, shift, 1)

    expect(output).toHaveLength(4)
    expect(output).toEqual(new Float32Array([12, 24, 36, 48]))
  })
})
