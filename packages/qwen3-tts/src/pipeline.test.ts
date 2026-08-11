import { describe, expect, it, vi, beforeEach } from 'vitest';
import { Qwen3TtsPipeline } from './pipeline';
import { qwen3TtsVariants } from './manifest';

vi.mock('./phases/generator', () => ({
  GeneratorPhase: class GeneratorPhase {
    name = 'generator';
    loadMs = 1;
    compileMs = 2;
    inferenceMs = 3;
    constructor(_variant: unknown, _options?: { onProgress?: (p: unknown) => void }) {}
    load = vi.fn().mockResolvedValue(undefined);
    generate = vi.fn().mockResolvedValue({ frames: new Uint16Array([1, 2, 3, 4]), frameCount: 1, codebooks: 4 });
    dispose = vi.fn().mockResolvedValue(undefined);
  },
}));

vi.mock('./phases/decoder', () => ({
  DecoderPhase: class DecoderPhase {
    name = 'decoder';
    loadMs = 4;
    compileMs = 5;
    inferenceMs = 6;
    constructor(_variant: unknown, _options?: { onProgress?: (p: unknown) => void }) {}
    load = vi.fn().mockResolvedValue(undefined);
    decode = vi.fn().mockResolvedValue(new Float32Array([0, 0.1, 0.2]));
    dispose = vi.fn().mockResolvedValue(undefined);
  },
}));

describe('Qwen3TtsPipeline', () => {
  let pipeline: Qwen3TtsPipeline;
  beforeEach(() => {
    pipeline = new Qwen3TtsPipeline(qwen3TtsVariants.int4);
  });

  it('starts idle', () => {
    expect(pipeline.status).toBe('idle');
  });

  it('exposes the manifest', () => {
    expect(pipeline.manifest.modelId).toBe('qwen3-tts-12hz-0.6b-base');
    expect(pipeline.manifest.capabilities).toContain('text-to-speech');
  });

  it('throws on run before load', async () => {
    await expect(pipeline.run({ text: 'hi' })).rejects.toThrow('Pipeline not ready');
  });

  it('load() validates context and becomes ready without compiling models', async () => {
    const context = {
      backend: 'wasm' as const,
      assets: { resolve: vi.fn() },
      liteRt: { loadModel: vi.fn(), loadNpy: vi.fn(), fetchBuffer: vi.fn() },
    };
    await pipeline.load(context);
    expect(pipeline.status).toBe('ready');
    expect(context.liteRt.loadModel).not.toHaveBeenCalled();
    expect(context.liteRt.loadNpy).not.toHaveBeenCalled();
  });

  it('run() in direct mode (no Worker) executes both phases and attaches phase receipts', async () => {
    vi.stubGlobal('Worker', undefined);
    const context = {
      backend: 'wasm' as const,
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(1)) },
      liteRt: {
        loadModel: vi.fn().mockResolvedValue({
          signatures: { decode: { getInputDetails: () => [{ name: 'mask', shape: [1, 1, 32, 32] }] } },
        }),
        loadNpy: vi.fn().mockResolvedValue(new Float32Array(3072 * 1024)),
        fetchBuffer: vi.fn().mockResolvedValue(new ArrayBuffer(1)),
      },
    };
    await pipeline.load(context);
    const result = await pipeline.run({ text: 'hello' }, { maxFrames: 1 });
    expect(result.kind).toBe('audio');
    expect(result.receipt.phases).toBeDefined();
    expect(result.receipt.phases!.map((p) => p.name)).toEqual(['generator', 'decoder']);
    vi.unstubAllGlobals();
  });

  it('run() in worker mode posts the expected protocol messages', async () => {
    const sent: (Record<string, unknown> & { target: { postMessage: (m: unknown, t?: unknown[]) => void }; id: string })[] = [];
    const FakeWorker = vi.fn().mockImplementation(function (this: {
      postMessage: (m: unknown, t?: unknown[]) => void;
      terminate: () => void;
      id: string;
      onmessage: ((e: unknown) => void) | undefined;
      _respond: ((m: unknown) => void) | undefined;
    }) {
      this.id = `w${sent.length}`;
      this.postMessage = vi.fn((m: unknown, t?: unknown[]) => {
        sent.push({ target: this, ...(m as object) } as never);
        const msg = m as { type: string; requestId?: number };
        if (msg.type === 'initialize') {
          queueMicrotask(() => this._respond && this._respond({ data: { type: 'ready' } }));
        } else if (msg.type === 'generate') {
          queueMicrotask(() => this._respond && this._respond({ data: { type: 'frames', requestId: msg.requestId, frames: { frames: new Uint16Array([1, 2, 3, 4]), frameCount: 1, codebooks: 4 }, phaseReceipt: { name: 'generator', backend: 'wasm' } } }));
        } else if (msg.type === 'decode') {
          queueMicrotask(() => this._respond && this._respond({ data: { type: 'audio', requestId: msg.requestId, audio: new Float32Array([0.1, 0.2]), phaseReceipt: { name: 'decoder', backend: 'wasm' } } }));
        }
      });
      this.terminate = vi.fn();
      this._respond = undefined;
      Object.defineProperty(this, 'onmessage', {
        set: (handler: ((e: unknown) => void) | undefined) => {
          this._respond = handler;
        },
        get: () => this._respond,
      });
    });
    vi.stubGlobal('Worker', FakeWorker);
    const context = { backend: 'wasm' as const, assets: { resolve: vi.fn() }, liteRt: { loadModel: vi.fn(), loadNpy: vi.fn(), fetchBuffer: vi.fn() } };
    await pipeline.load(context);
    const result = await pipeline.run({ text: 'hello' }, { maxFrames: 1 });
    expect(result.kind).toBe('audio');
    expect(sent.filter((s) => s.type === 'initialize').length).toBe(2);
    expect(sent.filter((s) => s.type === 'generate').length).toBe(1);
    expect(sent.filter((s) => s.type === 'decode').length).toBe(1);
    vi.unstubAllGlobals();
  });
});
