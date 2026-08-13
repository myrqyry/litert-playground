import type { ModelManifest } from '@litert-playground/inference-core';

const MODEL_BASE = 'https://huggingface.co/mlboydaisuke/clip-vit-b32-litert/resolve/c9253a36391d0881d4349f07d14e8b2d2054a955';

export const clipImageEmbeddingManifest: ModelManifest = {
  modelId: 'clip-vit-b32',
  name: 'CLIP ViT-B/32',
  version: '1.0.0',
  capabilities: ['image-embedding'],
  backends: { webgpu: true, wasm: true },
  memory: { downloadBytes: 180_000_000, residentBytes: 260_000_000 },
  assets: [
    { id: 'model', path: `${MODEL_BASE}/clip_image_encoder.tflite` },
    { id: 'text-embeddings', path: `${MODEL_BASE}/text_embeddings.bin` },
    { id: 'labels', path: `${MODEL_BASE}/labels.txt` },
  ],
};
