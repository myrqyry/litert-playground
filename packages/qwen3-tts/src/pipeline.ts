import { CompiledModel } from '@litertjs/core'
import { BPETokenizer } from './tokenizer'
import { Talker } from './talker'
import { MTP } from './mtp'
import { CodecDecoder } from './codec'
import { sample, SampleOpts } from './sampler'
import { parseNpy, parseNpz } from './npy-parser'
import { buildPrompt } from './prompt'
import { createQwen3TtsManifest, qwen3TtsVariants, type Qwen3TtsVariant } from './manifest'
import {
  type Pipeline,
  type PipelineStatus,
  type PipelineProgress,
  type RuntimeContext,
  type AudioInferenceResult,
  type InferenceReceipt,
  InferenceError,
  checkAudioValid,
  createInferenceReceipt,
} from '@litert-playground/inference-core'
import { discoverCodecShapes, discoverMtpShapes, discoverTalkerShapes } from './shape-discovery'
import { parseFp16Npy, type Fp16Table } from './fp16-table'

const HIDDEN = 1024
const CODEC_VOCAB = 3072
const CODEC_EOS = 2150
const NEG_INF = -1e9

const LANGUAGE_IDS: Record<string, number> = {
  english: 2050, chinese: 2055, japanese: 2058,
  korean: 2064, german: 2053, french: 2061,
  spanish: 2054, italian: 2070, portuguese: 2071,
  russian: 2069,
}

export interface QwenTtsInput { text: string }

export interface QwenTtsConfig {
  temperature?: number
  topK?: number
  repetitionPenalty?: number
  voice?: string
  maxFrames?: number
  language?: string
}

const DEFAULTS: QwenTtsConfig = {
  temperature: 0.85, topK: 25, repetitionPenalty: 1.05,
  voice: 'demo_speaker', maxFrames: 512, language: 'english',
}

export class Qwen3TtsPipeline implements Pipeline<QwenTtsInput, AudioInferenceResult, QwenTtsConfig> {
  readonly manifest
  status: PipelineStatus = 'idle'

  onProgress?: (progress: PipelineProgress) => void

  private context: RuntimeContext | null = null
  private tokenizer: BPETokenizer | null = null
  private talker!: Talker
  private mtp!: MTP
  private codec!: CodecDecoder
  private codecEmb!: Float32Array
  private mtpEmb!: Float32Array
  private textEmbData!: Fp16Table
  private projW1!: Float32Array
  private projB1!: Float32Array
  private projW2!: Float32Array
  private projB2!: Float32Array
  private talkerModel: CompiledModel | null = null
  private mtpModel: CompiledModel | null = null
  private codecModel: CompiledModel | null = null
  private loadMs = 0
  private compileMs = 0

  constructor(private readonly variant: Qwen3TtsVariant = qwen3TtsVariants.fp32) {
    this.manifest = createQwen3TtsManifest(variant)
  }

  async load(context: RuntimeContext): Promise<void> {
    if (this.status === 'ready') return
    this.status = 'loading'
    this.context = context
    const loadStart = performance.now()
    let compileMs = 0

    try {
      this.report({ phase: 'loading', step: 0, total: 7 })

      const tokData = await context.assets.resolve({ id: 'tokenizer', path: 'tokenizer.json' })
      this.tokenizer = new BPETokenizer(JSON.parse(new TextDecoder().decode(tokData)))
      this.report({ phase: 'loading', step: 1, total: 7 })

      this.codecEmb = await context.liteRt.loadNpy('tables/codec_embedding_fp32.npy')
      this.report({ phase: 'loading', step: 2, total: 7 })

      this.mtpEmb = await context.liteRt.loadNpy('tables/mtp_embeddings_fp16.npy')
      this.report({ phase: 'loading', step: 3, total: 7 })

       this.textEmbData = parseFp16Npy(await context.liteRt.fetchBuffer('tables/text_embedding_fp16.npy'))
      this.report({ phase: 'loading', step: 4, total: 7 })

      const projBuf = await context.assets.resolve({ id: 'text-projection', path: 'tables/text_projection_fp32.npz' })
      const proj = await parseNpz(projBuf)
      this.projW1 = proj['w1'] as Float32Array
      this.projB1 = proj['b1'] as Float32Array
      this.projW2 = proj['w2'] as Float32Array
      this.projB2 = proj['b2'] as Float32Array
      this.report({ phase: 'loading', step: 5, total: 7 })

       const compileStart = performance.now()
       this.talkerModel = await context.liteRt.loadModel(this.variant.talker)
       this.report({ phase: 'loading', step: 6, total: 7 })

       this.mtpModel = await context.liteRt.loadModel(this.variant.mtp)
       this.codecModel = await context.liteRt.loadModel(this.variant.codec)
       compileMs = performance.now() - compileStart

       const talkerShapes = discoverTalkerShapes(this.talkerModel!)
       const mtpShapes = discoverMtpShapes(this.mtpModel!)
       const codecShapes = discoverCodecShapes(this.codecModel!)
       this.talker = new Talker(this.talkerModel!, talkerShapes)
       this.mtp = new MTP(this.mtpModel!, {
         mtpEmbeddings: this.mtpEmb,
         codecEmbeddings: this.codecEmb,
         numCacheSlots: mtpShapes.cacheLen,
         cacheShape: mtpShapes.kvShape,
       })
       this.codec = new CodecDecoder(this.codecModel!, { chunkSize: codecShapes.chunkSize })
       this.report({ phase: 'loading', step: 7, total: 7 })
       this.status = 'ready'
       this.loadMs = performance.now() - loadStart
       this.compileMs = compileMs
    } catch (e) {
      this.status = 'error'
      throw e instanceof InferenceError ? e : new InferenceError('MODEL_COMPILE_FAILED', String(e), { cause: e })
    }
  }

  async run(input: QwenTtsInput, cfg?: QwenTtsConfig, signal?: AbortSignal): Promise<AudioInferenceResult> {
    if (this.status !== 'ready') throw new InferenceError('INFERENCE_FAILED', 'Pipeline not ready')
    this.status = 'running'

    const config = { ...DEFAULTS, ...cfg }
    const ctx = this.context!

    try {
      const lang = LANGUAGE_IDS[config.language || 'english'] || LANGUAGE_IDS.english
      const voicePath = `voices/${config.voice}.npy`

      const inferenceStart = performance.now()
      const speakerBuf = await ctx.assets.resolve({ id: 'voice', path: voicePath, optional: true }, { signal })

      const speakerEmb = parseNpy(speakerBuf)

      const { prefill, trailing, ttsPad } = buildPrompt(
        input.text, speakerEmb, lang, this.tokenizer!, this.codecEmb, this.textEmbData,
        (row: Float32Array) => this.projectText(row),
      )

      if (signal?.aborted) throw new InferenceError('CANCELLED', 'Cancelled before prefill')

      this.report({ phase: 'prefill', step: 0, total: 1 })
      const kv = this.talker.createEmptyKv()
      const sl = prefill.length / HIDDEN
      const { logits, hidden, kvCache } = await this.talker.prefill(prefill, kv, sl)

      const sampleOpts: SampleOpts = {
        temperature: config.temperature || 0.85,
        topK: config.topK || 25,
        repetitionPenalty: config.repetitionPenalty || 1.05,
        prevTokens: [],
      }

      const allFrames: number[][] = []
      let currentLogits = logits
      let currentHidden = hidden
      let currentKv = kvCache
      const maxFrames = config.maxFrames || 512

      for (let frame = 0; frame < maxFrames; frame++) {
        if (signal?.aborted) throw new InferenceError('CANCELLED', 'Cancelled during generation')

        this.report({ phase: 'decode', step: frame, total: maxFrames })

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

        this.report({ phase: 'mtp', step: frame, total: maxFrames })
        const residual = await this.mtp.predict(currentHidden, cb0, {
          temperature: config.temperature,
          topK: config.topK,
        })
        allFrames.push([cb0, ...residual])

        const frameIdx = frame < trailing.length ? frame : trailing.length - 1
        const textCond = frameIdx >= 0 ? trailing[frameIdx] : ttsPad

        let sumEmb = new Float32Array(HIDDEN)
        const cb0Emb = this.codecEmb.slice(cb0 * HIDDEN, (cb0 + 1) * HIDDEN)
        for (let i = 0; i < HIDDEN; i++) sumEmb[i] += cb0Emb[i]

        for (let r = 0; r < residual.length; r++) {
          const re = this.mtpEmb.slice(r * CODEC_VOCAB * HIDDEN + residual[r] * HIDDEN,
            r * CODEC_VOCAB * HIDDEN + (residual[r] + 1) * HIDDEN)
          for (let i = 0; i < HIDDEN; i++) sumEmb[i] += re[i] / residual.length
        }

        for (let i = 0; i < HIDDEN; i++) sumEmb[i] += textCond[i]

        const result = await this.talker.decode(sumEmb, currentKv, frame + 1)
        currentLogits = result.logits
        currentHidden = result.hidden
        currentKv = result.kvCache
      }

      if (allFrames.length === 0) {
        const samples = new Float32Array(0)
        this.status = 'ready'
        return {
          kind: 'audio', samples, sampleRate: 24000, channels: 1, durationSeconds: 0,
          receipt: createInferenceReceipt({
            manifest: this.manifest,
            backend: this.context?.backend ?? 'wasm',
            loadMs: this.loadMs,
            compileMs: this.compileMs,
            inferenceStart,
            inputSummary: `${input.text.length} characters`,
            outputSummary: '0 samples at 24000Hz, 1 channel',
            warnings: [],
          }),
        }
      }

      this.report({ phase: 'codec', step: 0, total: 1 })
      const audio = await this.codec.decode(allFrames)

      const duration = audio.length / 24000
      const warnings = checkAudioValid(audio, 24000, 1, duration)

      this.report({ phase: 'done', step: 1, total: 1 })

      if (warnings.length > 0) {
        console.warn('Qwen3TTS output warnings:', warnings)
      }

      this.status = 'ready'
      return {
        kind: 'audio', samples: audio, sampleRate: 24000, channels: 1, durationSeconds: duration,
      receipt: createInferenceReceipt({
        manifest: this.manifest,
        backend: this.context?.backend ?? 'wasm',
        loadMs: this.loadMs,
        compileMs: this.compileMs,
        inferenceStart,
        inputSummary: `${input.text.length} characters`,
        outputSummary: `${audio.length} samples at 24000Hz, 1 channel`,
        warnings,
      }),
      }
    } catch (e) {
      this.status = 'ready'
      throw e instanceof InferenceError ? e : new InferenceError('INFERENCE_FAILED', String(e), { cause: e })
    }
  }

  async dispose(): Promise<void> {
    this.talkerModel = null
    this.mtpModel = null
    this.codecModel = null
    this.tokenizer = null
    this.context = null
    this.status = 'disposed'
  }

  // ---- internals ----

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress)
  }

  private silu(x: number): number {
    return x / (1 + Math.exp(-x))
  }

  private projectText(row: Float32Array): Float32Array {
    const hiddenDim = HIDDEN * 4
    const inputDim = this.projW1.length / hiddenDim
    const h = new Float32Array(hiddenDim)
    for (let i = 0; i < hiddenDim; i++) {
      let sum = 0
      for (let j = 0; j < inputDim; j++) sum += this.projW1[j * hiddenDim + i] * row[j]
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
}
