import { CompiledModel, Tensor } from '@litertjs/core'
import { sample, SampleOpts } from './sampler'

const MTP_CACHE_SLOTS = 17
const MTP_CODEBOOKS = 15
const HIDDEN = 1024
const CODEC_VOCAB = 3072
const NEG_INF = -1e9

export interface MTPConfig {
  mtpEmbeddings: Float32Array
  codecEmbeddings: Float32Array
  numCacheSlots?: number
  numCodebooks?: number
  cacheShape?: number[]
}

export class MTP {
  private mtpEmb: Float32Array
  private codecEmb: Float32Array
  private numCacheSlots: number
  private numCodebooks: number
  private cacheShape: number[]

  constructor(
    private model: CompiledModel,
    config: MTPConfig,
  ) {
    this.mtpEmb = config.mtpEmbeddings
    this.codecEmb = config.codecEmbeddings
    this.numCacheSlots = config.numCacheSlots ?? MTP_CACHE_SLOTS
    this.numCodebooks = config.numCodebooks ?? MTP_CODEBOOKS
    this.cacheShape = config.cacheShape ?? [1, this.numCacheSlots, HIDDEN]
  }

  async predict(
    hiddenState: Float32Array,
    cb0: number,
    opts?: Partial<SampleOpts>,
  ): Promise<number[]> {
    const sampleOpts: SampleOpts = {
      temperature: opts?.temperature ?? 0,
      topK: opts?.topK ?? 0,
      repetitionPenalty: 1,
      prevTokens: [],
    }

    const cacheSize = this.cacheShape.reduce((a, b) => a * b, 1)
    const kAll = new Float32Array(cacheSize)
    const vAll = new Float32Array(cacheSize)

    const codes: number[] = []

    for (let t = 0; t < this.numCodebooks + 1; t++) {
      let embed: Float32Array
      if (t === 0) {
        embed = hiddenState
      } else if (t === 1) {
        embed = this.codecEmb.slice(cb0 * HIDDEN, (cb0 + 1) * HIDDEN)
      } else {
        const step = t - 2
        const prevCode = codes[codes.length - 1]
        const offset = (step * CODEC_VOCAB + prevCode) * HIDDEN
        embed = this.mtpEmb.slice(offset, offset + HIDDEN)
      }

      const mask = new Float32Array(this.numCacheSlots).fill(NEG_INF)
      for (let i = 0; i <= t; i++) mask[i] = 0

      const inputs: Record<string, Tensor> = {
        'args_0': new Tensor(embed, [1, 1, HIDDEN]),
        'args_1': new Tensor(new Int32Array([t]), [1]),
        'args_2': new Tensor(mask, [1, 1, 1, this.numCacheSlots]),
        'args_3': new Tensor(kAll, this.cacheShape),
        'args_4': new Tensor(vAll, this.cacheShape),
      }

      const result = await this.model.run(inputs)

      const kUpd = new Float32Array(await (result['output_1'] as Tensor).data())
      const vUpd = new Float32Array(await (result['output_2'] as Tensor).data())
      kAll.set(kUpd)
      vAll.set(vUpd)

      if (t >= 1) {
        const logits = new Float32Array(await (result['output_0'] as Tensor).data())
        const headLogits = logits.slice((t - 1) * CODEC_VOCAB, t * CODEC_VOCAB)
        codes.push(sample(headLogits, sampleOpts))
      }
    }

    return codes
  }
}
