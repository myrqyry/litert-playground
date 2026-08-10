import { CompiledModel, Tensor } from '@litertjs/core'

const NUM_CODE_GROUPS = 16
const UPSAMPLE = 1920
const OVERLAP_LEFT = 25

export interface CodecConfig {
  chunkSize?: number
  overlapLeft?: number
}

export class CodecDecoder {
  private chunkSize: number
  private overlapLeft: number

  constructor(
    private model: CompiledModel,
    config?: CodecConfig,
  ) {
    this.chunkSize = config?.chunkSize ?? 64
    this.overlapLeft = config?.overlapLeft ?? OVERLAP_LEFT
  }

  async decode(frameCodes: number[][]): Promise<Float32Array> {
    const pieces: Float32Array[] = []
    const codes = frameCodes
    const chunk = this.chunkSize
    const ctx = this.overlapLeft

    let i = 0
    while (i < codes.length) {
      const c = Math.min(ctx, i)
      const j = Math.min(i + chunk - c, codes.length)
      const windowLen = j - (i - c)

      const buf = new Int32Array(NUM_CODE_GROUPS * chunk).fill(0)
      for (let g = 0; g < NUM_CODE_GROUPS; g++) {
        for (let f = 0; f < windowLen; f++) {
          buf[g * chunk + f] = codes[i - c + f][g]
        }
      }

      const inputs: Record<string, Tensor> = {
        'args_0': new Tensor(buf, [1, NUM_CODE_GROUPS, chunk]),
      }

      const result = await this.model.run(inputs)
      const outKey = Object.keys(result)[0]
      const wav = new Float32Array(await (result[outKey] as Tensor).data())

      const validFrames = windowLen - c
      if (validFrames > 0) {
        const ctxSamples = c * UPSAMPLE
        const validSamples = validFrames * UPSAMPLE
        pieces.push(wav.slice(ctxSamples, ctxSamples + validSamples))
      }

      i = j
    }

    if (pieces.length === 0) return new Float32Array(0)
    const totalLen = pieces.reduce((s, p) => s + p.length, 0)
    const out = new Float32Array(totalLen)
    let off = 0
    for (const p of pieces) { out.set(p, off); off += p.length }
    return out
  }
}
