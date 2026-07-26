import { CompiledModel, Tensor } from '@litertjs/core'

export interface TalkerConfig {
  /** Number of KV cache slots (64 for talker_fp32) */
  numKvSlots: number
  /** Hidden dimension (1024 for Qwen3-TTS) */
  hiddenDim: number
  /** Codec vocabulary size (3072) */
  codecVocab: number
}

const DEFAULT_CONFIG: TalkerConfig = {
  numKvSlots: 64,
  hiddenDim: 1024,
  codecVocab: 3072,
}

export class Talker {
  private config: TalkerConfig

  constructor(
    private model: CompiledModel,
    config?: Partial<TalkerConfig>,
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config }
  }

  get numKvSlots(): number { return this.config.numKvSlots }

  async prefill(
    embeddings: Float32Array,
    inputPos: Int32Array,
    kvCache: Record<string, Tensor>,
  ): Promise<{ logits: Float32Array; hidden: Float32Array; kvCache: Record<string, Tensor>; }> {
    const inputs: Record<string, Tensor> = { ...kvCache }
    // prefill_32 signature
    const negInf = -1e9
    const seqLen = embeddings.length / this.config.hiddenDim
    const mask = new Float32Array(1 * 1 * 32 * 32).fill(negInf)
    for (let i = 0; i < 32 && i < seqLen; i++) {
      for (let j = 0; j <= i && j < seqLen; j++) {
        mask[i * 32 + j] = 0
      }
    }

    inputs['embeddings'] = new Tensor(embeddings, [1, 32, this.config.hiddenDim])
    inputs['input_pos'] = new Tensor(inputPos, [32])
    inputs['mask'] = new Tensor(mask, [1, 1, 32, 32])

    const result = await this.model.run('prefill_32', inputs)
    const logits = new Float32Array(await (result['logits'] as Tensor).data())
    // logits shape: [1, 32, codecVocab + hiddenDim]; take last position
    const lastLogits = logits.slice(-(this.config.codecVocab + this.config.hiddenDim))
    const cb0Logits = lastLogits.slice(0, this.config.codecVocab)
    const hidden = lastLogits.slice(this.config.codecVocab)

    const outKv: Record<string, Tensor> = {}
    for (const [key, tensor] of Object.entries(result)) {
      if (key.startsWith('kv_cache') || key.startsWith('StateArray')) {
        outKv[key] = tensor as Tensor
      }
    }

    return { logits: cb0Logits, hidden, kvCache: outKv }
  }

  async decode(
    embeddings: Float32Array,
    inputPos: Int32Array,
    kvCache: Record<string, Tensor>,
  ): Promise<{ logits: Float32Array; hidden: Float32Array; kvCache: Record<string, Tensor>; }> {
    const inputs: Record<string, Tensor> = { ...kvCache }
    const negInf = -1e9
    const pos = inputPos[0]
    const mask = new Float32Array(1 * 1 * 1 * 32).fill(negInf)
    for (let i = 0; i <= pos; i++) mask[i] = 0

    inputs['embeddings'] = new Tensor(embeddings, [1, 1, this.config.hiddenDim])
    inputs['input_pos'] = inputPos.length === 1
      ? new Tensor(inputPos, [1])
      : new Tensor(inputPos, [1])
    inputs['mask'] = new Tensor(mask, [1, 1, 1, 32])

    const result = await this.model.run('decode', inputs)
    const logits = new Float32Array(await (result['logits'] as Tensor).data())
    const cb0Logits = logits.slice(0, this.config.codecVocab)
    const hidden = logits.slice(this.config.codecVocab)

    const outKv: Record<string, Tensor> = {}
    for (const [key, tensor] of Object.entries(result)) {
      if (key.startsWith('kv_cache') || key.startsWith('StateArray')) {
        outKv[key] = tensor as Tensor
      }
    }

    return { logits: cb0Logits, hidden, kvCache: outKv }
  }
}
