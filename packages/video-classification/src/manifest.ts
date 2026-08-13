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
  assets: [{ id: 'model', path: MODEL_URL, role: 'model' }],
  verification: {
    assets: 'untested',
    compile: 'untested',
    inference: 'untested',
    output: 'untested',
    qualification: 'limited',
    upstreamRevision: 'c2ceda0efa7344ba5a95c3eeaa9925cb0940e453',
    environments: [
      { browser: 'CI contract tests', backend: 'webgpu', runtime: 'manifest-only' },
      { browser: 'CI contract tests', backend: 'wasm', runtime: 'manifest-only' },
    ],
    expectedOutput: {
      preprocessing: [
        'resize frames to 172x172',
        'RGB planar channels',
        'normalize channel values by dividing by 255',
      ],
      outputShape: [1, 600],
      labels: {
        assetId: 'labels',
        count: 600,
        mapping: 'zero-based output index to Kinetics-600 label',
      },
      behavior: ['softmax scores sum to 1', 'top classes use Kinetics-600 labels'],
    },
  },
};
