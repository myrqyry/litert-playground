import { loadAndCompile, CompiledModel, Tensor } from '@litertjs/core'
import { BPETokenizer } from './tokenizer'
import { Talker } from './talker'
import { MTP } from './mtp'
import { CodecDecoder } from './codec'
import { sample, SampleOpts } from './sampler'
import { parseNpy, parseNpz } from './npy-parser'

export interface TTSConfig {
  temperature?: number
  topK?: number
  repetitionPenalty?: number
  voice?: string
  maxFrames?: number
  language?: string
}

export interface TTSProgress {
  phase: 'loading' | 'prefill' | 'decode' | 'mtp' | 'codec' | 'done'
  step: number
  total: number
}

const DEFAULT_CONFIG: TTSConfig = {
  temperature: 0.85,
  topK: 25,
  repetitionPenalty: 1.05,
  voice: 'demo_speaker',
  maxFrames: 512,
  language: 'english',
}

const HIDDEN = 1024
const CODEC_VOCAB = 3072
const CODEC_PAD = 2148
const CODEC_BOS = 2149
const CODEC_EOS = 2150
const CODEC_THINK = 2154
const CODEC_THINK_BOS = 2156
const CODEC_THINK_EOS = 2157
const CODEC_NOTHINK = 2155
const TTS_PAD = 151671
const TTS_BOS = 151672
const TTS_EOS = 151673
const NEG_INF = -1e9

const LANGUAGE_IDS: Record<string, number> = {
  english: 2050, chinese: 2055, japanese: 2058,
  korean: 2064, german: 2053, french: 2061,
  spanish: 2054, italian: 2070, portuguese: 2071,
  russian: 2069,
}

export class Qwen3TtsPipeline {
  tokenizer: BPETokenizer | null = null

  private talkerModel: CompiledModel | null = null
  private mtpModel: CompiledModel | null = null
  private codecModel: CompiledModel | null = null
  private talker!: Talker
  private mtp!: MTP
  private codec!: CodecDecoder
  private codecEmb!: Float32Array
  private mtpEmb!: Float32Array
  private textEmbData!: Float32Array
  private projW1!: Float32Array
  private projB1!: Float32Array
  private projW2!: Float32Array
  private projB2!: Float32Array

  constructor(private modelDir: string) {}

  onProgress?: (progress: TTSProgress) => void

  async load(): Promise<void> {
    this.onProgress?.({ phase: 'loading', step: 0, total: 7 })
    const base = this.modelDir

    const tokRes = await fetch(`${base}/tokenizer.json`)
    this.tokenizer = new BPETokenizer(await tokRes.json())
    this.onProgress?.({ phase: 'loading', step: 1, total: 7 })

    const codecEmb = await this.loadNpy(`${base}/tables/codec_embedding_fp32.npy`)
    this.codecEmb = codecEmb as Float32Array
    this.onProgress?.({ phase: 'loading', step: 2, total: 7 })

    const mtpEmb = await this.loadNpy(`${base}/tables/mtp_embeddings_fp16.npy`)
    this.mtpEmb = mtpEmb as Float32Array
    this.onProgress?.({ phase: 'loading', step: 3, total: 7 })

    // text_embedding_fp16 is ~1.2GB fp16, use fp16 via direct .npy
    const textEmb = await this.loadNpy(`${base}/tables/text_embedding_fp16.npy`)
    this.textEmbData = textEmb as Float32Array
    this.onProgress?.({ phase: 'loading', step: 4, total: 7 })

    const projRes = await fetch(`${base}/tables/text_projection_fp32.npz`)
    const proj = await parseNpz(await projRes.arrayBuffer())
    this.projW1 = proj['w1'] as Float32Array
    this.projB1 = proj['b1'] as Float32Array
    this.projW2 = proj['w2'] as Float32Array
    this.projB2 = proj['b2'] as Float32Array
    this.onProgress?.({ phase: 'loading', step: 5, total: 7 })

    this.talkerModel = await this.loadModel(`${base}/talker_fp32.tflite`)
    this.onProgress?.({ phase: 'loading', step: 6, total: 7 })

    this.mtpModel = await this.loadModel(`${base}/mtp_fp32.tflite`)
    const codecModel = await this.loadModel(`${base}/codec_decoder_fp32.tflite`)

    this.talker = new Talker(this.talkerModel)
    this.mtp = new MTP(this.mtpModel, {
      mtpEmbeddings: this.mtpEmb,
      codecEmbeddings: this.codecEmb,
    })
    this.codec = new CodecDecoder(codecModel)
    this.onProgress?.({ phase: 'loading', step: 7, total: 7 })
  }

  async synthesize(text: string, config?: TTSConfig): Promise<Float32Array> {
    if (!this.tokenizer) throw new Error('Pipeline not loaded')

    const cfg = { ...DEFAULT_CONFIG, ...config }
    const lang = LANGUAGE_IDS[cfg.language || 'english'] || LANGUAGE_IDS.english

    const speakerResp = await fetch(`${this.modelDir}/voices/${cfg.voice}.npy`)
    const speakerEmb = parseNpy(await speakerResp.arrayBuffer())

    // Build prompt
    const { prefill, trailing, ttsPad } = this.buildPrompt(text, speakerEmb, lang)

    // Prefill talker
    this.onProgress?.({ phase: 'prefill', step: 0, total: 1 })
    const kv: Record<string, Tensor> = {}
    const sl = prefill.length / HIDDEN
    const { logits, hidden, kvCache } = await this.talker.prefill(prefill, kv, sl)

    // Decode loop
    const sampleOpts: SampleOpts = {
      temperature: cfg.temperature || 0.85,
      topK: cfg.topK || 25,
      repetitionPenalty: cfg.repetitionPenalty || 1.05,
      prevTokens: [],
    }

    const allFrames: number[][] = []
    let currentLogits = logits
    let currentHidden = hidden
    let currentKv = kvCache
    const maxFrames = cfg.maxFrames || 512

    for (let frame = 0; frame < maxFrames; frame++) {
      this.onProgress?.({ phase: 'decode', step: frame, total: maxFrames })

      // Suppress control tokens (2048+)
      const scores = new Float32Array(currentLogits)
      for (let i = 2048; i < CODEC_VOCAB; i++) scores[i] = NEG_INF
      scores[CODEC_EOS] = 0

      if (frame < 2) scores[CODEC_EOS] = NEG_INF

      for (const token of sampleOpts.prevTokens) {
        scores[token] = scores[token] > 0
          ? scores[token] / sampleOpts.repetitionPenalty
          : scores[token] * sampleOpts.repetitionPenalty
      }

      const cb0 = sample(scores, { ...sampleOpts, prevTokens: [] })
      if (cb0 === CODEC_EOS) break
      sampleOpts.prevTokens.push(cb0)

      this.onProgress?.({ phase: 'mtp', step: frame, total: maxFrames })

      const mtpOpts: Partial<SampleOpts> = {
        temperature: cfg.temperature,
        topK: cfg.topK,
      }
      const residual = await this.mtp.predict(currentHidden, cb0, mtpOpts)
      allFrames.push([cb0, ...residual])

      // Build decoder input embedding:
      // codec_emb[cb0] + mtp_emb.mean(residual) + trailing[frame] or tts_pad
      const frameIdx = frame < trailing.length ? frame : trailing.length - 1
      const textCond = frameIdx >= 0 ? trailing[frameIdx] : ttsPad

      let sumEmb = new Float32Array(HIDDEN)
      // codec embedding for cb0
      const cb0Emb = this.codecEmb.slice(cb0 * HIDDEN, (cb0 + 1) * HIDDEN)
      for (let i = 0; i < HIDDEN; i++) sumEmb[i] += cb0Emb[i]

      // mtp embedding mean for residuals
      for (let r = 0; r < residual.length; r++) {
        const re = this.mtpEmb.slice(r * CODEC_VOCAB * HIDDEN + residual[r] * HIDDEN,
          r * CODEC_VOCAB * HIDDEN + (residual[r] + 1) * HIDDEN)
        for (let i = 0; i < HIDDEN; i++) sumEmb[i] += re[i] / residual.length
      }

      for (let i = 0; i < HIDDEN; i++) sumEmb[i] += textCond[i]

      this.onProgress?.({ phase: 'decode', step: frame, total: maxFrames })
      const pos = frame + 1
      const result = await this.talker.decode(sumEmb, currentKv, pos)
      currentLogits = result.logits
      currentHidden = result.hidden
      currentKv = result.kvCache
    }

    if (allFrames.length === 0) return new Float32Array(0)

    this.onProgress?.({ phase: 'codec', step: 0, total: 1 })
    const audio = await this.codec.decode(allFrames)
    this.onProgress?.({ phase: 'done', step: 1, total: 1 })
    return audio
  }

  private silu(x: number): number {
    return x / (1 + Math.exp(-x))
  }

  private embedText(ids: number[]): Float32Array[] {
    return ids.map(id => {
      const base = id * HIDDEN
      if (base + HIDDEN > this.textEmbData.length) {
        return new Float32Array(HIDDEN)
      }
      // fp16 stored as float32 after parseNpy
      const row = this.textEmbData.slice(base, base + HIDDEN)
      return this.projectText(row)
    })
  }

  private projectText(row: Float32Array): Float32Array {
    // SiLU MLP: silu(x @ W1 + b1) @ W2 + b2
    const hiddenDim = HIDDEN * 4
    const h = new Float32Array(hiddenDim)
    for (let i = 0; i < hiddenDim; i++) {
      let sum = 0
      for (let j = 0; j < HIDDEN; j++) sum += this.projW1[j * hiddenDim + i] * row[j]
      h[i] = this.silu(sum + this.projB1[i])
    }
    const out = new Float32Array(HIDDEN)
    for (let i = 0; i < HIDDEN; i++) {
      let sum = 0
      for (let j = 0; j < hiddenDim; j++) sum += this.projW2[j * HIDDEN + i] * h[j]
      out[i] = sum + this.projB2[i]
    }
    return out
  }

  private buildPrompt(text: string, speakerEmb: Float32Array, langId: number) {
    const tok = this.tokenizer!
    const ids = tok.encode(`<|im_start|>assistant\n${text}<|im_end|>\n<|im_start|>assistant\n`)

    const ttsBos = this.embedText([TTS_BOS])[0]
    const ttsEos = this.embedText([TTS_EOS])[0]
    const ttsPad = this.embedText([TTS_PAD])[0]

    const control: number[] = [CODEC_THINK, CODEC_THINK_BOS, langId, CODEC_THINK_EOS]
    const codecPreEmb = control.map(c => this.codecEmb.slice(c * HIDDEN, (c + 1) * HIDDEN))

    const roleIds = ids.slice(0, 3)
    const roleEmb = this.embedText(roleIds)
    // role + codecPre + speaker + pad + bos
    const bodyEmb: Float32Array[] = []
    for (const ce of codecPreEmb) {
      const p = new Float32Array(HIDDEN)
      for (let i = 0; i < HIDDEN; i++) p[i] = ttsPad[i] + ce[i]
      bodyEmb.push(p)
    }
    const speakerRow = new Float32Array(HIDDEN)
    for (let i = 0; i < HIDDEN; i++) speakerRow[i] = ttsPad[i] + speakerEmb[i]
    bodyEmb.push(speakerRow)
    const bosRow = new Float32Array(HIDDEN)
    for (let i = 0; i < HIDDEN; i++) bosRow[i] = ttsPad[i] + this.codecEmb[CODEC_PAD * HIDDEN + i]
    bodyEmb.push(bosRow)

    const firstTextId = ids[3]
    const firstTextEmb = this.embedText([firstTextId])[0]
    const codecEosEmb = this.codecEmb.slice(CODEC_BOS * HIDDEN, (CODEC_BOS + 1) * HIDDEN)
    const firstTextRow = new Float32Array(HIDDEN)
    for (let i = 0; i < HIDDEN; i++) firstTextRow[i] = firstTextEmb[i] + codecEosEmb[i]

    const prefill = new Float32Array((roleEmb.length + bodyEmb.length + 1) * HIDDEN)
    let off = 0
    for (const e of roleEmb) { prefill.set(e, off); off += HIDDEN }
    for (const e of bodyEmb) { prefill.set(e, off); off += HIDDEN }
    prefill.set(firstTextRow, off)

    // trailing text tokens
    const trailingIds = ids.slice(4)
    const trailingEmb = trailingIds.map(id => {
      const e = this.embedText([id])[0]
      const r = new Float32Array(HIDDEN)
      for (let i = 0; i < HIDDEN; i++) r[i] = e[i] + ttsPad[i]
      return r
    })

    return { prefill, trailing: [...trailingEmb, ttsEos], ttsPad }
  }

  private async loadModel(path: string): Promise<CompiledModel> {
    const resp = await fetch(path)
    const buffer = await resp.arrayBuffer()
    return loadAndCompile(new Uint8Array(buffer))
  }

  private async loadNpy(path: string): Promise<Float32Array> {
    const resp = await fetch(path)
    return parseNpy(await resp.arrayBuffer())
  }
}
