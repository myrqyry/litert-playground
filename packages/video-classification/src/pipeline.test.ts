import { Tensor } from '@litertjs/core';
import type { RuntimeContext } from '@litert-playground/inference-core';
import { describe, expect, it, vi } from 'vitest';
import { MoViNetPipeline } from './pipeline';

const inputDetails = Array.from({ length: 47 }, (_, i) => ({
  index: i,
  shape: i === 0 ? [1, 3, 172, 172] : [1, 1, 1, 1],
}));

function fakeTensor() {
  return { toTypedArray: () => new Float32Array([1]), delete: vi.fn() };
}

function fakeContext(): RuntimeContext {
  return {
    backend: 'wasm',
    assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(0)) },
    liteRt: {
      loadModel: vi.fn().mockResolvedValue({ getInputDetails: () => inputDetails }),
      loadNpy: vi.fn(),
      fetchBuffer: vi.fn(),
    },
  };
}

describe('MoViNetPipeline', () => {
  it('builds 47 stateful input tensors after load', async () => {
    const pipeline = new MoViNetPipeline();
    await pipeline.load(fakeContext());
    expect(pipeline.status).toBe('ready');

    const state = (pipeline as any).state;
    const spy = vi.spyOn(Tensor, 'fromTypedArray').mockImplementation(() => fakeTensor() as unknown as Tensor);
    const frame = fakeTensor() as unknown as Tensor;
    try {
      const inputs = await state.buildInputTensors(frame);
      expect(inputs.length).toBe(47);
      expect(inputs[0]).toBe(frame);
      expect((inputs[45] as unknown as { toTypedArray(): Float32Array }).toTypedArray()[0]).toBe(1);
    } finally {
      spy.mockRestore();
    }
  });

  it('throws not-ready when run before load', async () => {
    const pipeline = new MoViNetPipeline();
    await expect(
      pipeline.run({ canvas: {} as HTMLCanvasElement }),
    ).rejects.toThrow('Pipeline not ready');
  });
});
