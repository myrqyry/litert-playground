import { describe, expect, it } from 'vitest';
import { decodeClipTextEmbeddings, rankClipLabels } from './pipeline';

describe('CLIP text embeddings', () => {
  it('decodes a binary header and payload', () => {
    const count = 2;
    const dimension = 3;
    const buffer = new ArrayBuffer(8 + count * dimension * 4);
    const view = new DataView(buffer);
    view.setInt32(0, count, true);
    view.setInt32(4, dimension, true);
    const decoded = decodeClipTextEmbeddings(buffer);
    expect(decoded.count).toBe(count);
    expect(decoded.dimension).toBe(dimension);
    expect(decoded.embeddings.length).toBe(count * dimension);
  });

  it('rejects an invalid header', () => {
    const buffer = new ArrayBuffer(4);
    expect(() => decodeClipTextEmbeddings(buffer)).toThrow('Invalid CLIP text embeddings header');
  });
});

describe('rankClipLabels', () => {
  it('ranks labels by cosine similarity with CLIP scaling', () => {
    const image = new Float32Array([1, 0]);
    const text = new Float32Array([1, 0, 0, 1]);
    const ranked = rankClipLabels(image, text, ['cat', 'dog']);
    expect(ranked[0].label).toBe('cat');
    expect(ranked[0].index).toBe(0);
  });
});
