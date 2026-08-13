import type { ModelManifest } from '@litert-playground/inference-core';

const MODEL_URL =
  'https://huggingface.co/litert-community/MoViNet-A0-Stream-LiteRT/resolve/c2ceda0efa7344ba5a95c3eeaa9925cb0940e453/movinet_a0_stream.tflite';

export const moViNetManifest: ModelManifest = {
  modelId: 'movinet-a0-stream',
  name: 'MoViNet-A0-Stream',
  version: '1.0.0',
  capabilities: ['image-classification'],
  backends: { webgpu: true, wasm: true },
  memory: { downloadBytes: 2_500_000, residentBytes: 4_000_000 },
  assets: [{ id: 'model', path: MODEL_URL }],
};
