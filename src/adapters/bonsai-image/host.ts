export const BONSAI_IMAGE_SIZE = 512
export const BONSAI_LATENT_GRID = 32
export const BONSAI_IMAGE_TOKENS = BONSAI_LATENT_GRID * BONSAI_LATENT_GRID
export const BONSAI_TEXT_TOKENS = 256

/** Matches the FLUX.2-klein flow-matching schedule published with Bonsai. */
export function flowMatchSigmas(
  steps: number,
  tokens = BONSAI_IMAGE_TOKENS,
): Float32Array {
  if (!Number.isInteger(steps) || steps < 1) {
    throw new RangeError('steps must be a positive integer')
  }

  const m200 = 0.00016927 * tokens + 0.45666666
  const m10 = 8.73809524e-05 * tokens + 1.89833333
  const a = (m200 - m10) / 190
  const mu = a * steps + (m200 - 200 * a)
  const output = new Float32Array(steps + 1)

  for (let i = 0; i < steps; i++) {
    const linear = steps === 1 ? 1 : 1 - (i / (steps - 1)) * (1 - 1 / steps)
    output[i] = Math.exp(mu) / (Math.exp(mu) + (1 / linear - 1))
  }
  output[steps] = 0
  return output
}

/** Creates [batch, row, column, ...] image token position IDs. */
export function createImagePositionIds(
  grid = BONSAI_LATENT_GRID,
): Float32Array {
  if (!Number.isInteger(grid) || grid < 1) {
    throw new RangeError('grid must be a positive integer')
  }
  const output = new Float32Array(grid * grid * 4)
  let offset = 0
  for (let row = 0; row < grid; row++) {
    for (let column = 0; column < grid; column++) {
      output[offset++] = 0
      output[offset++] = row
      output[offset++] = column
      output[offset++] = 0
    }
  }
  return output
}

/** Creates [batch, 0, 0, token] text token position IDs. */
export function createTextPositionIds(
  tokens = BONSAI_TEXT_TOKENS,
): Float32Array {
  if (!Number.isInteger(tokens) || tokens < 1) {
    throw new RangeError('tokens must be a positive integer')
  }
  const output = new Float32Array(tokens * 4)
  for (let token = 0; token < tokens; token++) {
    output[token * 4 + 3] = token
  }
  return output
}

/** Converts DiT packed tokens into the VAE latent layout. */
export function unpatchifyLatent(
  lat: Float32Array,
  batchNormScale: Float32Array,
  batchNormShift: Float32Array,
  grid = BONSAI_LATENT_GRID,
): Float32Array {
  const channels = batchNormScale.length
  const expectedLength = grid * grid * channels
  if (lat.length !== expectedLength || batchNormShift.length !== channels || channels % 4 !== 0) {
    throw new RangeError('latent and normalization shapes do not match Bonsai packing')
  }

  const output = new Float32Array((channels / 4) * grid * 2 * grid * 2)
  for (let row = 0; row < grid; row++) {
    for (let column = 0; column < grid; column++) {
      const tokenOffset = (row * grid + column) * channels
      for (let channel = 0; channel < channels; channel++) {
        const packed = lat[tokenOffset + channel] * batchNormScale[channel] + batchNormShift[channel]
        const outputChannel = Math.floor(channel / 4)
        const patchRow = Math.floor((channel % 4) / 2)
        const patchColumn = channel % 2
        const outputOffset =
          outputChannel * (grid * 2) * (grid * 2) +
          (row * 2 + patchRow) * (grid * 2) +
          column * 2 + patchColumn
        output[outputOffset] = packed
      }
    }
  }
  return output
}
