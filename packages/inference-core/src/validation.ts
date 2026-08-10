export function checkAudioValid(
  samples: Float32Array,
  sampleRate: number,
  channels: number,
  durationSeconds: number,
): string[] {
  const warnings: string[] = []
  if (samples.length === 0) warnings.push('audio: empty samples')
  for (let i = 0; i < samples.length; i++) {
    if (!isFinite(samples[i])) { warnings.push('audio: contains NaN/Infinity'); break }
  }
  if (sampleRate <= 0) warnings.push('audio: invalid sampleRate')
  if (channels <= 0) warnings.push('audio: invalid channels')
  if (durationSeconds <= 0.01) warnings.push('audio: duration too short (<10ms)')
  let sumSq = 0
  for (let i = 0; i < samples.length; i++) sumSq += samples[i] * samples[i]
  const rms = samples.length === 0 ? 0 : Math.sqrt(sumSq / samples.length)
  if (rms < 1e-6) warnings.push('audio: near-silent (RMS below threshold)')
  let clipCount = 0
  for (let i = 0; i < samples.length; i++) if (Math.abs(samples[i]) > 0.999) clipCount++
  if (samples.length > 0 && clipCount / samples.length > 0.01) {
    warnings.push(`audio: high clipping ratio (${((clipCount / samples.length) * 100).toFixed(1)}%)`)
  }
  return warnings
}

export function checkImageValid(pixels: Uint8ClampedArray, width: number, height: number): string[] {
  const warnings: string[] = []
  if (pixels.length === 0) warnings.push('image: empty pixels')
  if (pixels.length !== width * height * 4) warnings.push('image: pixel count mismatch for RGBA')
  if (width <= 0 || height <= 0) warnings.push('image: invalid dimensions')
  let nonBlack = false
  for (let i = 0; i < pixels.length; i += 4) {
    if (pixels[i] > 0 || pixels[i + 1] > 0 || pixels[i + 2] > 0) { nonBlack = true; break }
  }
  if (!nonBlack) warnings.push('image: all black pixels')
  return warnings
}

export function checkEmbeddingValid(values: Float32Array, dimensions: number): string[] {
  const warnings: string[] = []
  if (values.length === 0) warnings.push('embedding: empty values')
  if (values.length !== dimensions) warnings.push('embedding: dimension mismatch')
  for (let i = 0; i < values.length; i++) {
    if (!isFinite(values[i])) { warnings.push('embedding: contains NaN/Infinity'); break }
  }
  let normSq = 0
  for (let i = 0; i < values.length; i++) normSq += values[i] * values[i]
  if (normSq < 1e-12) warnings.push('embedding: near-zero vector')
  return warnings
}
