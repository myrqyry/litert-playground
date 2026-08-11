import { describe, expect, it, vi, beforeEach } from 'vitest';

vi.mock('../sampler', () => ({ sample: () => 42 }));
vi.mock('../prompt', () => ({
  buildPrompt: () => ({
    prefill: new Float32Array(1024),
    trailing: [new Float32Array(1024)],
    ttsPad: new Float32Array(1024),
  }),
}));
vi.mock('../npy-parser', () => ({
  parseNpy: () => new Float32Array(1024),
  parseNpz: vi.fn().mockResolvedValue({
    w1: new Float32Array(4096 * 1024),
    b1: new Float32Array(4096),
    w2: new Float32Array(1024 * 1024),
    b2: new Float32Array(1024),
  }),
}));
vi.mock('../talker', () => ({
  Talker: class {
    createEmptyKv = vi.fn(() => ({}));
    prefill = vi.fn(async () => ({ logits: new Float32Array(3072), hidden: new Float32Array(1024), kvCache: {} }));
    decode = vi.fn(async () => ({ logits: new Float32Array(3072), hidden: new Float32Array(1024), kvCache: {} }));
  },
}));
vi.mock('../mtp', () => ({
  MTP: class {
    predict = vi.fn(async () => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
  },
}));

import { GeneratorPhase } from './generator';
import { qwen3TtsVariants } from '../manifest';

function fakeModel() {
  const inputDetails = [
    { name: 'mask', shape: [1, 1, 32, 32] },
    { name: 'kv_cache_0', shape: [1, 32, 1024] },
    { name: 'args_2', shape: [1, 1, 1, 17] },
    { name: 'args_3', shape: [1, 17, 1024] },
  ];
  return {
    signatures: {
      decode: {
        getInputDetails: () => inputDetails,
      },
    },
    getInputDetails: () => inputDetails,
    run: vi.fn(async () => {
      const data = () => new Float32Array(32 * (3072 + 1024));
      return {
        logits: { data },
        output_0: { data: () => new Float32Array(16 * 3072) },
        output_1: { data },
        output_2: { data },
        kv_cache_0: { data },
      };
    }),
  };
}

function fakeLiteRt() {
  return {
    loadModel: vi.fn().mockResolvedValue(fakeModel()),
    loadNpy: vi.fn().mockResolvedValue(new Float32Array(3072 * 1024)),
    fetchBuffer: vi.fn().mockResolvedValue(fakeFp16Npy(16)),
  };
}

function fakeFp16Npy(width: number): ArrayBuffer {
  const header = `{'descr': '<f2', 'fortran_order': False, 'shape': (1, ${width}), }`;
  const headerBytes = new TextEncoder().encode(header);
  const dataOffset = 10 + headerBytes.length;
  const padded = dataOffset + width * 2;
  const aligned = Math.ceil(padded / 64) * 64;
  const buffer = new ArrayBuffer(aligned);
  const bytes = new Uint8Array(buffer);
  bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00]);
  new DataView(buffer).setUint16(8, headerBytes.length, true);
  bytes.set(headerBytes, 10);
  return buffer;
}

function fakeContext() {
  return {
    backend: 'wasm' as const,
    assets: {
      resolve: vi.fn().mockResolvedValue(new TextEncoder().encode('{"model":{"vocab":{}}}').buffer),
    },
    liteRt: fakeLiteRt(),
  };
}

describe('GeneratorPhase', () => {
  let phase: GeneratorPhase;
  beforeEach(() => {
    phase = new GeneratorPhase(qwen3TtsVariants.int4);
  });

  it('loads tokenizer, tables, talker and mtp (not codec)', async () => {
    const ctx = fakeContext();
    await phase.load(ctx);
    const { liteRt } = ctx;
    expect(liteRt.loadModel).toHaveBeenCalledWith('talker_int4.tflite');
    expect(liteRt.loadModel).toHaveBeenCalledWith('mtp_fp32.tflite');
    expect(liteRt.loadModel).not.toHaveBeenCalledWith('codec_decoder_fp32.tflite');
    expect(phase.name).toBe('generator');
  });

  it('generates CodecFrames with flat Uint16Array layout', async () => {
    await phase.load(fakeContext());
    const frames = await phase.generate({ text: 'hello' }, { maxFrames: 1 });
    expect(frames.frameCount).toBe(1);
    expect(frames.codebooks).toBe(16);
    expect(frames.frames).toBeInstanceOf(Uint16Array);
  });
});
