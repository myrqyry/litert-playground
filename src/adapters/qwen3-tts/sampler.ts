export interface SampleOpts {
  temperature: number
  topK: number
  repetitionPenalty: number
  prevTokens: number[]
}

function applyRepetitionPenalty(logits: Float32Array, penalty: number, prev: number[]): Float32Array {
  if (penalty === 1 || prev.length === 0) return logits
  const out = new Float32Array(logits)
  for (const id of prev) {
    if (id >= 0 && id < out.length) {
      out[id] = out[id] > 0 ? out[id] / penalty : out[id] * penalty
    }
  }
  return out
}

function applyTemperature(logits: Float32Array, temp: number): Float32Array {
  if (temp === 0) return logits
  const out = new Float32Array(logits)
  for (let i = 0; i < out.length; i++) out[i] = out[i] / temp
  return out
}

function topKFilter(logits: Float32Array, k: number): Float32Array {
  if (k <= 0 || k >= logits.length) return logits
  const out = new Float32Array(logits.length).fill(-Infinity)
  const indexed = Array.from(logits).map((v, i) => ({ v, i }))
  indexed.sort((a, b) => b.v - a.v)
  const threshold = indexed[k - 1].v
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] >= threshold) out[i] = logits[i]
  }
  return out
}

function softmax(logits: Float32Array): Float32Array {
  const out = new Float32Array(logits)
  let maxVal = -Infinity
  for (let i = 0; i < out.length; i++) {
    if (out[i] > maxVal) maxVal = out[i]
  }
  let sum = 0
  for (let i = 0; i < out.length; i++) {
    if (out[i] > -Infinity) {
      out[i] = Math.exp(out[i] - maxVal)
    } else {
      out[i] = 0
    }
    sum += out[i]
  }
  if (sum > 0) for (let i = 0; i < out.length; i++) out[i] = out[i] / sum
  return out
}

export function sample(logits: Float32Array, opts: SampleOpts): number {
  let adjusted = applyRepetitionPenalty(logits, opts.repetitionPenalty, opts.prevTokens)
  adjusted = applyTemperature(adjusted, opts.temperature)
  adjusted = topKFilter(adjusted, opts.topK)

  if (opts.temperature === 0) {
    let maxIdx = 0
    for (let i = 1; i < adjusted.length; i++) {
      if (adjusted[i] > adjusted[maxIdx]) maxIdx = i
    }
    return maxIdx
  }

  const probs = softmax(adjusted)
  const rand = Math.random()
  let accum = 0
  for (let i = 0; i < probs.length; i++) {
    accum += probs[i]
    if (accum >= rand) return i
  }
  return 0
}
