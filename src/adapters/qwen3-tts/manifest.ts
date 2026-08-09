import { type ModelManifest } from '../../core/types'

export const qwen3TtsManifest: ModelManifest = {
  modelId: 'qwen3-tts-0.6b',
  name: 'Qwen3-TTS 0.6B',
  version: '0.4.0',
  capabilities: ['text-to-speech'],
  backends: { webgpu: 'experimental', wasm: true },
  memory: { downloadBytes: 1_600_000_000, residentBytes: 2_500_000_000 },
  assets: [
    { id: 'tokenizer', path: 'tokenizer.json', bytes: 2_600_000 },
    { id: 'talker', path: 'talker_fp32.tflite', bytes: 2_700_000_000 },
    { id: 'mtp', path: 'mtp_fp32.tflite', bytes: 2_000_000_000 },
    { id: 'codec-decoder', path: 'codec_decoder_fp32.tflite', bytes: 200_000_000 },
    { id: 'codec-embedding', path: 'tables/codec_embedding_fp32.npy', bytes: 25_000_000 },
    { id: 'mtp-embeddings', path: 'tables/mtp_embeddings_fp16.npy', bytes: 86_000_000 },
    { id: 'text-embedding', path: 'tables/text_embedding_fp16.npy', bytes: 1_200_000_000 },
    { id: 'text-projection', path: 'tables/text_projection_fp32.npz', bytes: 13_000_000 },
    { id: 'voice', path: 'voices/demo_speaker.npy', bytes: 4_000_000, optional: true },
  ],
}