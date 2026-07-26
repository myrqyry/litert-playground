export class TextEmbedder {
  constructor(
    private embeddingTable: Float32Array,
    private projectionWeights: Float32Array,
    private dim: number,
    private outputDim: number
  ) {}

  silu(x: number): number {
    return x / (1 + Math.exp(-x))
  }

  embedToken(id: number): Float32Array {
    const vec = new Float32Array(this.dim)
    const base = id * this.dim
    if (base + this.dim <= this.embeddingTable.length) {
      vec.set(this.embeddingTable.subarray(base, base + this.dim))
    }
    return vec
  }

  project(vec: Float32Array): Float32Array {
    const hiddenDim = this.outputDim * 4
    const w1 = this.projectionWeights.subarray(0, this.dim * hiddenDim)
    const b1 = this.projectionWeights.subarray(this.dim * hiddenDim, this.dim * hiddenDim + hiddenDim)
    const w2 = this.projectionWeights.subarray(this.dim * hiddenDim + hiddenDim, this.dim * hiddenDim + hiddenDim + hiddenDim * this.outputDim)
    const b2 = this.projectionWeights.subarray(this.dim * hiddenDim + hiddenDim + hiddenDim * this.outputDim)

    const hidden = new Float32Array(hiddenDim)
    for (let i = 0; i < hiddenDim; i++) {
      let sum = i < b1.length ? b1[i] : 0
      for (let j = 0; j < this.dim; j++) sum += w1[i * this.dim + j] * vec[j]
      hidden[i] = this.silu(sum)
    }

    const out = new Float32Array(this.outputDim)
    for (let i = 0; i < this.outputDim; i++) {
      let sum = i < b2.length ? b2[i] : 0
      for (let j = 0; j < hiddenDim; j++) sum += w2[i * hiddenDim + j] * hidden[j]
      out[i] = sum
    }
    return out
  }

  embed(ids: number[]): Float32Array {
    if (ids.length === 0) return new Float32Array(this.outputDim)

    let sum = new Float32Array(this.outputDim)
    for (const id of ids) {
      const vec = this.embedToken(id)
      const projected = this.project(vec)
      for (let i = 0; i < this.outputDim; i++) sum[i] += projected[i]
    }

    const norm = Math.sqrt(this.outputDim)
    for (let i = 0; i < this.outputDim; i++) sum[i] = sum[i] / norm
    return sum
  }
}
