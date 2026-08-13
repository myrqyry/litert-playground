import { describe, expect, it } from 'vitest';
import { clipImageEmbeddingManifest } from './manifest';

describe('CLIP image embedding manifest', () => {
  it('records roles and the pinned qualification contract', () => {
    expect(clipImageEmbeddingManifest.assets).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: 'model', role: 'image encoder' }),
        expect.objectContaining({
          id: 'text-embeddings',
          role: 'text embedding table',
        }),
        expect.objectContaining({ id: 'labels', role: 'label mapping' }),
      ]),
    );
    expect(clipImageEmbeddingManifest.verification).toMatchObject({
      qualification: 'limited',
      upstreamRevision: 'c9253a36391d0881d4349f07d14e8b2d2054a955',
      environments: [
        expect.objectContaining({ backend: 'webgpu' }),
        expect.objectContaining({ backend: 'wasm' }),
      ],
      expectedOutput: {
        outputDimension: 512,
      },
    });
  });

  it('records the exact CLIP preprocessing constants as JSON data', () => {
    expect(clipImageEmbeddingManifest.verification?.expectedOutput?.preprocessing).toEqual([
      'center crop input to a square',
      'resize to 224x224',
      'RGB channel order',
      'divide channel values by 255',
      'normalize with mean [0.48145466, 0.4578275, 0.40821073]',
      'normalize with standard deviation [0.26862954, 0.26130258, 0.27577711]',
    ]);
    expect(JSON.parse(JSON.stringify(clipImageEmbeddingManifest))).toEqual(
      clipImageEmbeddingManifest,
    );
  });
});
