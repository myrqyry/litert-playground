import { BPETokenizer } from './tokenizer'
import { TextEmbedder } from './text-embedding'
import { parseNpz } from './npy-parser'

export interface TTSConfig {
  temperature?: number
  topK?: number
  voice?: string
}

export interface TTSProgress {
  phase: 'prefill' | 'decode' | 'mtp' | 'codec' | 'done'
  token: number
  totalTokens: number
}

export class Qwen3TtsPipeline {
  tokenizer: BPETokenizer | null = null
  embedder: TextEmbedder | null = null

  constructor(private modelDir: string) {}

  async load(): Promise<void> {
    const tokRes = await fetch(`${this.modelDir}/tokenizer.json`)
    const tokJson = await tokRes.json()
    this.tokenizer = new BPETokenizer(tokJson)
  }

  async synthesize(text: string, config?: TTSConfig): Promise<Float32Array> {
    throw new Error('Not implemented')
  }
}
