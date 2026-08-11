import { type ModelAsset, type ModelManifest } from '@litert-playground/inference-core'

export interface Qwen3TtsVariant {
  id: string
  talker: string
  mtp: string
  codec: string
  quantization: string
  backendSupport: Partial<Record<'webgpu' | 'wasm' | 'webnn', boolean | 'experimental'>>
}

export const qwen3TtsVariants: Record<string, Qwen3TtsVariant> = {
  fp32: {
    id: 'fp32', talker: 'talker_fp32.tflite', mtp: 'mtp_fp32.tflite', codec: 'codec_decoder_fp32.tflite',
    quantization: 'fp32', backendSupport: { webgpu: 'experimental', wasm: true },
  },
  int4: {
    id: 'int4', talker: 'talker_int4.tflite', mtp: 'mtp_fp32.tflite', codec: 'codec_decoder_fp32.tflite',
    quantization: 'int4 talker / fp32 auxiliary graphs', backendSupport: { webgpu: 'experimental', wasm: true },
  },
  browserMemory: {
    id: 'browserMemory', talker: 'talker_int4.tflite', mtp: 'mtp_folded_int8.tflite', codec: 'codec_decoder_fp32.tflite',
    quantization: 'int4 talker / int8 folded mtp / fp32 codec', backendSupport: { webgpu: 'experimental', wasm: true },
  },
}

function assetsFor(variant: Qwen3TtsVariant): ModelAsset[] {
  return [
   { id: 'tokenizer', path: 'tokenizer.json', bytes: 11_424_262 },
   { id: 'talker', path: variant.talker, bytes: variant.id === 'fp32' ? 1_783_890_064 : 255_998_768 },
   { id: 'mtp', path: variant.mtp, bytes: variant.id === 'browserMemory' ? 229_608_368 : 440_526_692 },
   { id: 'codec-decoder', path: variant.codec, bytes: 456_820_324 },
   { id: 'codec-embedding', path: 'tables/codec_embedding_fp32.npy', bytes: 12_583_040 },
   { id: 'mtp-embeddings', path: 'tables/mtp_embeddings_fp16.npy', bytes: 62_914_688 },
   { id: 'text-embedding', path: 'tables/text_embedding_fp16.npy', bytes: 622_329_984 },
   { id: 'text-projection', path: 'tables/text_projection_fp32.npz', bytes: 25_179_078 },
   { id: 'voice', path: 'voices/demo_speaker.npy', bytes: 4_224, optional: true },
  ]
}

export function createQwen3TtsManifest(variant: Qwen3TtsVariant = qwen3TtsVariants.fp32): ModelManifest {
  const assets = assetsFor(variant)
  const requiredDownloadBytes = assets
    .filter(asset => !asset.optional)
    .reduce((total, asset) => total + (asset.bytes ?? 0), 0)
  return {
   modelId: 'qwen3-tts-12hz-0.6b-base',
  name: `Qwen3-TTS 0.6B (${variant.quantization})`,
  version: '0.4.0',
  capabilities: ['text-to-speech'],
  backends: variant.backendSupport,
  memory: { downloadBytes: requiredDownloadBytes, residentBytes: 2_500_000_000 },
  assets,
  }
}

export const qwen3TtsManifest = createQwen3TtsManifest()
