import { type ModelManifest } from '@litert-playground/inference-core';

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
