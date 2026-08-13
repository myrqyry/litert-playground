import { type ModelManifest } from '@litert-playground/inference-core';

// Int4/int8 sizes for the 1.2B family are estimates; corrected during qualification.
const LFM25_1_2B_INT4_BYTES = 645_000_000;
const LFM25_1_2B_INT8_BYTES = 1_290_000_000;

export const litertLmManifest: ModelManifest = {
  modelId: 'qwen3-0.6b',
  name: 'Qwen 3 0.6B (LiteRT-LM)',
  version: '0.6.0',
  capabilities: ['text-generation'],
  backends: { webgpu: true, wasm: true },
  memory: {
    downloadBytes: 614_400_000,
    residentBytes: 614_400_000,
  },
  assets: [
    {
      id: 'model',
      path: 'litert-community/Qwen3-0.6B/resolve/main/Qwen3-0.6B.litertlm',
      bytes: 614_400_000,
    },
  ],
};

export const lfm2_5InstructManifest: ModelManifest = {
  modelId: 'lfm2.5-1.2b-instruct',
  name: 'LFM2.5 1.2B Instruct (LiteRT-LM, int4)',
  version: '1.2.0',
  capabilities: ['text-generation'],
  backends: { webgpu: true, wasm: true },
  memory: {
    downloadBytes: LFM25_1_2B_INT4_BYTES,
    residentBytes: LFM25_1_2B_INT4_BYTES,
  },
  assets: [
    {
      id: 'model',
      path: 'litert-community/LFM2.5-1.2B-Instruct/resolve/main/LFM2.5-1.2B-Instruct_int4.litertlm',
      bytes: LFM25_1_2B_INT4_BYTES,
    },
  ],
};

export const lfm2_5InstructInt8Manifest: ModelManifest = {
  modelId: 'lfm2.5-1.2b-instruct-int8',
  name: 'LFM2.5 1.2B Instruct (LiteRT-LM, int8)',
  version: '1.2.0',
  capabilities: ['text-generation'],
  backends: { webgpu: true, wasm: true },
  memory: {
    downloadBytes: LFM25_1_2B_INT8_BYTES,
    residentBytes: LFM25_1_2B_INT8_BYTES,
  },
  assets: [
    {
      id: 'model',
      path: 'litert-community/LFM2.5-1.2B-Instruct/resolve/main/LFM2.5-1.2B-Instruct_int8.litertlm',
      bytes: LFM25_1_2B_INT8_BYTES,
    },
  ],
};

export const lfm2_5ThinkingManifest: ModelManifest = {
  modelId: 'lfm2.5-1.2b-thinking',
  name: 'LFM2.5 1.2B Thinking (LiteRT-LM, int4)',
  version: '1.2.0',
  capabilities: ['text-generation', 'reasoning'],
  backends: { webgpu: true, wasm: true },
  memory: {
    downloadBytes: LFM25_1_2B_INT4_BYTES,
    residentBytes: LFM25_1_2B_INT4_BYTES,
  },
  assets: [
    {
      id: 'model',
      path: 'litert-community/LFM2.5-1.2B-Thinking/resolve/main/LFM2.5-1.2B-Thinking_int4.litertlm',
      bytes: LFM25_1_2B_INT4_BYTES,
    },
  ],
};

export const lfm2_5ThinkingInt8Manifest: ModelManifest = {
  modelId: 'lfm2.5-1.2b-thinking-int8',
  name: 'LFM2.5 1.2B Thinking (LiteRT-LM, int8)',
  version: '1.2.0',
  capabilities: ['text-generation', 'reasoning'],
  backends: { webgpu: true, wasm: true },
  memory: {
    downloadBytes: LFM25_1_2B_INT8_BYTES,
    residentBytes: LFM25_1_2B_INT8_BYTES,
  },
  assets: [
    {
      id: 'model',
      path: 'litert-community/LFM2.5-1.2B-Thinking/resolve/main/LFM2.5-1.2B-Thinking_int8.litertlm',
      bytes: LFM25_1_2B_INT8_BYTES,
    },
  ],
};

export const gemma4E2bManifest: ModelManifest = {
  modelId: 'gemma-4-e2b-it',
  name: 'Gemma 4 E2B Instruct (LiteRT-LM)',
  version: '4.0.0',
  capabilities: ['text-generation'],
  backends: { webgpu: true, wasm: true },
  memory: {
    downloadBytes: 2_580_000_000,
    residentBytes: 2_580_000_000,
  },
  assets: [
    {
      id: 'model',
      path: 'litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm',
      bytes: 2_580_000_000,
    },
  ],
};

export const gemma4E4bManifest: ModelManifest = {
  modelId: 'gemma-4-e4b-it',
  name: 'Gemma 4 E4B Instruct (LiteRT-LM)',
  version: '4.0.0',
  capabilities: ['text-generation'],
  backends: { webgpu: true, wasm: true },
  memory: {
    downloadBytes: 3_650_000_000,
    residentBytes: 3_650_000_000,
  },
  assets: [
    {
      id: 'model',
      path: 'litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm',
      bytes: 3_650_000_000,
    },
  ],
};

export type TextGenCapability = 'text-generation' | 'reasoning';
export type TextGenPreference = 'low_latency' | 'deep';

export function selectTextGenerationManifest(
  capability: TextGenCapability,
  preference: TextGenPreference = 'low_latency',
): ModelManifest {
  const thinking = capability === 'reasoning';
  return preference === 'deep'
    ? thinking
      ? lfm2_5ThinkingInt8Manifest
      : lfm2_5InstructInt8Manifest
    : thinking
      ? lfm2_5ThinkingManifest
      : lfm2_5InstructManifest;
}

export const transformersTextManifest: ModelManifest = {
  modelId: 'qwen3-0.6b-litertlm',
  name: 'Qwen 3 0.6B (Transformers.js)',
  version: '0.6.0',
  capabilities: ['text-generation'],
  backends: { webgpu: true },
  memory: {
    downloadBytes: 629_145_600,
    residentBytes: 629_145_600,
  },
  assets: [
    {
      id: 'model',
      path: 'onnx-community/Qwen3-0.6B-ONNX',
      bytes: 629_145_600,
    },
  ],
};
