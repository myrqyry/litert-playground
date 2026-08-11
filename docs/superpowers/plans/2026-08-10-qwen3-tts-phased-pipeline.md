# Qwen3-TTS Phased Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `Qwen3TtsPipeline` into two execution phases (Generator → CodecFrames → Decoder) so the three Qwen3-TTS LiteRT graphs never share one WASM page, unblocking browser inference.

**Architecture:** `Qwen3TtsPipeline.run()` orchestrates two disposable phases. In the browser it spawns two sequential classic Web Workers (Generator worker compiles Talker+MTP and produces `CodecFrames`, is terminated, then Decoder worker compiles codec and decodes to `Float32Array`, is terminated) — the proven `f25097c` residency split. In Node/vitest (`typeof Worker === 'undefined'`) it runs the same phase classes in-process (direct mode). All inference logic moves from `pipeline.ts` into `phases/generator.ts` and `phases/decoder.ts`; thin module bridges `workers/*.worker.ts` forward messages; the public API `new Qwen3TtsPipeline(variant) → load(context) → run({text})` is unchanged.

**Tech Stack:** TypeScript, Vite dev server (port 5176), Playwright headless Chrome harness at `/tmp/opencode/qwen-browser/`, vitest, `@litertjs/core@2.5.3`.

## Global Constraints

- Public API must NOT change: `new Qwen3TtsPipeline(variant)` → `load(context)` → `run({ text }, config?, signal?)` → `AudioInferenceResult`. No `generateFrames()`/`decodeFrames()` exposed.
- `CodecFrames = { frames: Uint16Array; frameCount: number; codebooks: number }` — ONE contiguous flat typed array `[frame0 cb0..cbN, frame1 cb0..cbN, ...]`, transferred via `postMessage(message, [frames.frames.buffer])`. NOT `number[][]`.
- Phase ownership: GeneratorPhase owns tokenizer / speaker emb / codec+mtp emb tables / FP16 text emb / projection weights / Talker / MTP / sampling / prompt / KV caches → produces only CodecFrames → then `worker.terminate()` (worker/process lifetime is the memory boundary). DecoderPhase owns codec graph / shape discovery / frame chunking → consumes CodecFrames → produces `Float32Array` → then terminates.
- Worker protocol (boring unions, defined in `src/workers/protocol.ts`):
  - GeneratorWorkerRequest = `initialize{variant, modelBase}` | `generate{requestId, input, config}` | `cancel{requestId}`
  - GeneratorWorkerResponse = `ready` | `progress{requestId, progress}` | `frames{requestId, frames, phaseReceipt}` | `error{requestId?, error}`
  - DecoderWorkerRequest = `initialize{variant, modelBase}` | `decode{requestId, frames}`
  - DecoderWorkerResponse = `ready` | `audio{requestId, audio, phaseReceipt}` | `error{requestId?, error}`
  - Errors serialized as `SerializedInferenceError { code: string; message: string; stage?: string }`.
- Receipts become phase-aware (non-breaking): add `InferencePhaseReceipt { name; backend; loadMs?; compileMs?; inferenceMs?; warnings? }` and `phases?: InferencePhaseReceipt[]` to BOTH `InferenceReceipt` and `InferenceReceiptOptions`; `createInferenceReceipt` passes `phases` through.
- `load()` semantics change: validate manifest/runtime/assets/configuration and prepare orchestration → `ready`. It does NOT compile or keep models resident. `run()` launches Generator worker → load/compile → generate → terminate → Decoder worker → load/compile → decode → terminate.
- Cancellation = kill the phase: on `signal.abort`, terminate the active worker and reject with `InferenceError('CANCELLED', ...)`. WASM memory dies with the worker.
- Worker mechanisms: MODULE workers break LiteRT (`importScripts`, proven b11-b13); classic workers at a real http URL work when served verbatim (proven `f25097c`). `litertWasmProxy` middleware in `examples/vite.config.ts` stays as-is and is extended to serve the two classic worker shells verbatim from disk.
- Scope boundaries: NO changes to `@litert-playground/runtime-litert`, `@litert-playground/text-gen`, PodQast, or model conversion. `qwenModelProxy` + `litertWasmProxy` in `examples/vite.config.ts` stay (extend only the shell-serving part).
- Deferred (do NOT implement): PodQast batching, OPFS/Cache Storage persistence, lighter/quantized codec, WebGPU escape hatch.
- All logic is testable in direct mode under vitest (Node has no `Worker`); phases hold the logic, worker files are ~30-line bridges.
- Existing `packages/qwen3-tts/src/pipeline.test.ts` (3 tests) and `receipt.test.ts` must be updated to the new internals but keep their assertions' intent.
- Every task runs `pnpm verify`-style gates for the affected package: `pnpm --filter <pkg> typecheck` then `pnpm --filter <pkg> test`.

---

### Task 1: Phase receipts in inference-core

**Files:**
- Modify: `packages/inference-core/src/types.ts`
- Modify: `packages/inference-core/src/receipts.ts`
- Test: `packages/inference-core/src/receipts.test.ts`

**Interfaces:**
- Consumes: existing `Backend` type, `InferenceReceipt`, `InferenceReceiptOptions`, `createInferenceReceipt` from `@litert-playground/inference-core`.
- Produces: `InferencePhaseReceipt` type; `phases?: InferencePhaseReceipt[]` field on `InferenceReceipt` and `InferenceReceiptOptions`; `createInferenceReceipt` returns a receipt carrying `phases` when provided. Later tasks and `packages/qwen3-tts` consume these.

- [ ] **Step 1: Write the failing tests**

Append to `packages/inference-core/src/receipts.test.ts`:

```ts
import { createInferenceReceipt } from './receipts';
import type { InferencePhaseReceipt } from './receipts';

describe('phase receipts', () => {
  it('includes an optional phases array on the receipt', () => {
    const phases: InferencePhaseReceipt[] = [
      { name: 'generator', backend: 'wasm', loadMs: 10, compileMs: 20, inferenceMs: 30 },
      { name: 'decoder', backend: 'wasm', loadMs: 40, compileMs: 50, inferenceMs: 60 },
    ];
    const receipt = createInferenceReceipt({
      manifest: { modelId: 'model', version: '1.0.0' },
      backend: 'wasm',
      loadMs: 10,
      compileMs: 20,
      inferenceStart: 0,
      inputSummary: 'input',
      outputSummary: 'output',
      warnings: [],
      phases,
    });
    expect(receipt.phases).toEqual(phases);
  });

  it('omits phases when none are provided', () => {
    const receipt = createInferenceReceipt({
      manifest: { modelId: 'model', version: '1.0.0' },
      backend: 'wasm',
      loadMs: 10,
      compileMs: 20,
      inferenceStart: 0,
      inputSummary: 'input',
      outputSummary: 'output',
      warnings: [],
    });
    expect(receipt.phases).toBeUndefined();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm --filter @litert-playground/inference-core test`
Expected: FAIL — `receipt.phases` is `undefined` / type errors on the new `InferencePhaseReceipt` import.

- [ ] **Step 3: Add the type and the field**

In `packages/inference-core/src/receipts.ts`, add the phase type and thread `phases` through:

```ts
import type { Backend, InferenceReceipt } from './types';

export interface InferencePhaseReceipt {
  name: string;
  backend: Backend;
  loadMs?: number;
  compileMs?: number;
  inferenceMs?: number;
  warnings?: string[];
}

export interface InferenceReceiptOptions {
  manifest: Pick<ModelManifest, 'modelId' | 'version'>;
  backend: Backend;
  loadMs: number;
  compileMs: number;
  inferenceStart: number;
  inputSummary: string;
  outputSummary: string;
  warnings: string[];
  phases?: InferencePhaseReceipt[];
}
```

(Keep the existing `ModelManifest` import; add it if not already present.) Then in `createInferenceReceipt`'s returned object add `phases: options.phases,`. Update the `InferenceReceipt` interface in `packages/inference-core/src/types.ts` to add `phases?: InferencePhaseReceipt[]` (import the type from `./receipts` as `import type { InferencePhaseReceipt } from './receipts';` — if a cycle arises, declare `phases?: readonly { name: string; backend: Backend; loadMs?: number; compileMs?: number; inferenceMs?: number; warnings?: string[] }[]` inline instead).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm --filter @litert-playground/inference-core test`
Expected: PASS (all existing tests + the 2 new ones).

- [ ] **Step 5: Typecheck and commit**

```bash
pnpm --filter @litert-playground/inference-core typecheck
git add packages/inference-core/src/types.ts packages/inference-core/src/receipts.ts packages/inference-core/src/receipts.test.ts
git commit -m "feat(inference-core): add phase receipts to inference receipt"
```

---

### Task 2: CodecFrames pack/unpack

**Files:**
- Create: `packages/qwen3-tts/src/codec-frames.ts`
- Test: `packages/qwen3-tts/src/codec-frames.test.ts`

**Interfaces:**
- Consumes: nothing (pure module).
- Produces: `interface CodecFrames { frames: Uint16Array; frameCount: number; codebooks: number }`, `packCodecFrames(allFrames: number[][], codebooks?: number): CodecFrames`, `unpackCodecFrames(frames: CodecFrames): number[][]`. Later tasks: GeneratorPhase returns `packCodecFrames(...)`; DecoderPhase consumes via `unpackCodecFrames(...)`.

- [ ] **Step 1: Write the failing test**

Create `packages/qwen3-tts/src/codec-frames.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { packCodecFrames, unpackCodecFrames } from './codec-frames';

describe('codec-frames', () => {
  it('round-trips number[][] through the flat Uint16Array layout', () => {
    const frames = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ];
    const packed = packCodecFrames(frames);
    expect(packed.frameCount).toBe(3);
    expect(packed.codebooks).toBe(3);
    expect(Array.from(packed.frames)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(unpackCodecFrames(packed)).toEqual(frames);
  });

  it('handles an empty frame list', () => {
    const packed = packCodecFrames([], 16);
    expect(packed.frameCount).toBe(0);
    expect(packed.frames.length).toBe(0);
    expect(unpackCodecFrames(packed)).toEqual([]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --filter @litert-playground/qwen3-tts test codec-frames`
Expected: FAIL — module `./codec-frames` does not exist.

- [ ] **Step 3: Implement**

Create `packages/qwen3-tts/src/codec-frames.ts`:

```ts
export interface CodecFrames {
  frames: Uint16Array;
  frameCount: number;
  codebooks: number;
}

export function packCodecFrames(allFrames: number[][], codebooks?: number): CodecFrames {
  const cb = codebooks ?? (allFrames.length > 0 ? allFrames[0].length : 16);
  const frames = new Uint16Array(allFrames.length * cb);
  for (let f = 0; f < allFrames.length; f++) {
    for (let c = 0; c < cb; c++) {
      frames[f * cb + c] = allFrames[f][c] ?? 0;
    }
  }
  return { frames, frameCount: allFrames.length, codebooks: cb };
}

export function unpackCodecFrames(frames: CodecFrames): number[][] {
  const out: number[][] = [];
  for (let f = 0; f < frames.frameCount; f++) {
    const row: number[] = [];
    for (let c = 0; c < frames.codebooks; c++) {
      row.push(frames.frames[f * frames.codebooks + c]);
    }
    out.push(row);
  }
  return out;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm --filter @litert-playground/qwen3-tts test codec-frames`
Expected: PASS.

- [ ] **Step 5: Typecheck and commit**

```bash
pnpm --filter @litert-playground/qwen3-tts typecheck
git add packages/qwen3-tts/src/codec-frames.ts packages/qwen3-tts/src/codec-frames.test.ts
git commit -m "feat(qwen3-tts): add CodecFrames flat packing helpers"
```

---

### Task 3: Worker protocol types + error serializer

**Files:**
- Create: `packages/qwen3-tts/src/workers/protocol.ts`
- Test: `packages/qwen3-tts/src/workers/protocol.test.ts`

**Interfaces:**
- Consumes: `Qwen3TtsVariant` from `../manifest`, `CodecFrames` from `../codec-frames`, `QwenTtsConfig` from `../types` (created in Task 4), `InferencePhaseReceipt` from `@litert-playground/inference-core`.
- Produces: `SerializedInferenceError`, `serializeError(e: unknown): SerializedInferenceError`, `GeneratorWorkerRequest`, `GeneratorWorkerResponse`, `DecoderWorkerRequest`, `DecoderWorkerResponse`. Consumed by worker bridges (Task 6) and `pipeline.ts` (Task 7).

- [ ] **Step 1: Write the failing test**

Create `packages/qwen3-tts/src/workers/protocol.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { serializeError } from './protocol';

describe('serializeError', () => {
  it('serializes an Error', () => {
    const out = serializeError(new Error('boom'));
    expect(out.message).toBe('boom');
    expect(out.code).toBe('UNKNOWN');
  });

  it('serializes a string', () => {
    expect(serializeError('kaboom').message).toBe('kaboom');
  });

  it('serializes an InferenceError-like object with code and stage', () => {
    const out = serializeError({ code: 'CANCELLED', message: 'stopped', stage: 'decode' });
    expect(out).toEqual({ code: 'CANCELLED', message: 'stopped', stage: 'decode' });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --filter @litert-playground/qwen3-tts test protocol`
Expected: FAIL — module `./protocol` does not exist.

- [ ] **Step 3: Implement**

Create `packages/qwen3-tts/src/workers/protocol.ts` (also defining `QwenTtsConfig` import from `../types`; create `../types` first if Task 4 hasn't landed yet — if it hasn't, temporarily type `config` fields inline as `Record<string, unknown>` and fix the import after Task 4):

```ts
import type { InferencePhaseReceipt } from '@litert-playground/inference-core';
import type { CodecFrames } from '../codec-frames';
import type { Qwen3TtsVariant } from '../manifest';
import type { QwenTtsConfig } from '../types';

export interface SerializedInferenceError {
  code: string;
  message: string;
  stage?: string;
}

export function serializeError(e: unknown): SerializedInferenceError {
  if (typeof e === 'object' && e !== null && 'code' in e && 'message' in e) {
    const as = e as { code?: string; message?: string; stage?: string };
    return { code: as.code ?? 'UNKNOWN', message: String(as.message), stage: as.stage };
  }
  if (e instanceof Error) return { code: 'UNKNOWN', message: e.message };
  return { code: 'UNKNOWN', message: String(e) };
}

export type GeneratorWorkerRequest =
  | { type: 'initialize'; variant: Qwen3TtsVariant; modelBase: string }
  | { type: 'generate'; requestId: number; input: { text: string }; config: QwenTtsConfig }
  | { type: 'cancel'; requestId: number };

export type GeneratorWorkerResponse =
  | { type: 'ready' }
  | { type: 'progress'; requestId?: number; progress: { phase: string; step: number; total: number } }
  | { type: 'frames'; requestId: number; frames: CodecFrames; phaseReceipt: InferencePhaseReceipt }
  | { type: 'error'; requestId?: number; error: SerializedInferenceError };

export type DecoderWorkerRequest =
  | { type: 'initialize'; variant: Qwen3TtsVariant; modelBase: string }
  | { type: 'decode'; requestId: number; frames: CodecFrames };

export type DecoderWorkerResponse =
  | { type: 'ready' }
  | { type: 'audio'; requestId: number; audio: Float32Array; phaseReceipt: InferencePhaseReceipt }
  | { type: 'error'; requestId?: number; error: SerializedInferenceError };
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm --filter @litert-playground/qwen3-tts test protocol`
Expected: PASS.

- [ ] **Step 5: Typecheck and commit**

```bash
pnpm --filter @litert-playground/qwen3-tts typecheck
git add packages/qwen3-tts/src/workers/protocol.ts packages/qwen3-tts/src/workers/protocol.test.ts
git commit -m "feat(qwen3-tts): define worker protocol types and error serializer"
```

---

### Task 4: Shared types + GeneratorPhase

**Files:**
- Create: `packages/qwen3-tts/src/types.ts`
- Create: `packages/qwen3-tts/src/phases/generator.ts`
- Create: `packages/qwen3-tts/src/phases/generator.test.ts`
- Modify: `packages/qwen3-tts/src/pipeline.ts` (move `QwenTtsInput`/`QwenTtsConfig`/`DEFAULTS`/constants into `types.ts`, re-export from `pipeline.ts`)

**Interfaces:**
- Consumes: `BPETokenizer` from `../tokenizer`, `Talker` from `../talker`, `MTP` from `../mtp`, `sample`/`SampleOpts` from `../sampler`, `parseNpy`/`parseNpz` from `../npy-parser`, `buildPrompt` from `../prompt`, `discoverTalkerShapes`/`discoverMtpShapes` from `../shape-discovery`, `parseFp16Npy`/`Fp16Table` from `../fp16-table`, `createQwen3TtsManifest`/`Qwen3TtsVariant` from `../manifest`, `RuntimeContext`/`PipelineProgress`/`InferenceError` from `@litert-playground/inference-core`, `CodecFrames`/`packCodecFrames` from `../codec-frames`.
- Produces: `QwenTtsInput`, `QwenTtsConfig`, `DEFAULTS`, model constants (`HIDDEN`, `CODEC_VOCAB`, `CODEC_EOS`, `NEG_INF`, `LANGUAGE_IDS`) exported from `src/types.ts`; `GeneratorPhase` class with `manifest`, `name = 'generator'`, `loadMs`, `compileMs`, `inferenceMs`, `async load(context: RuntimeContext): Promise<void>`, `async generate(input, config, signal?): Promise<CodecFrames>`, `dispose()`. Consumed by decoder task, worker bridges, pipeline.

- [ ] **Step 1: Create `src/types.ts`**

Create `packages/qwen3-tts/src/types.ts`:

```ts
export interface QwenTtsInput {
  text: string;
}

export interface QwenTtsConfig {
  temperature?: number;
  topK?: number;
  repetitionPenalty?: number;
  voice?: string;
  maxFrames?: number;
  language?: string;
}

export const DEFAULTS: Required<QwenTtsConfig> = {
  temperature: 0.85,
  topK: 25,
  repetitionPenalty: 1.05,
  voice: 'demo_speaker',
  maxFrames: 512,
  language: 'english',
};

export const HIDDEN = 1024;
export const CODEC_VOCAB = 3072;
export const CODEC_EOS = 2150;
export const NEG_INF = -1e9;

export const LANGUAGE_IDS: Record<string, number> = {
  english: 2050,
  chinese: 2055,
  japanese: 2058,
  korean: 2064,
  german: 2053,
  french: 2061,
  spanish: 2054,
  italian: 2070,
  portuguese: 2071,
  russian: 2069,
};
```

- [ ] **Step 2: Write the failing GeneratorPhase test**

Create `packages/qwen3-tts/src/phases/generator.test.ts` — mocks the model-adjacent modules so the phase is exercised with a fake `liteRt`:

```ts
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

import { GeneratorPhase } from './generator';
import { qwen3TtsVariants } from '../manifest';

function fakeModel() {
  return {
    signatures: {
      decode: {
        getInputDetails: () => [
          { name: 'mask', shape: [1, 1, 32, 32] },
          { name: 'kv_cache_0', shape: [1, 32, 1024] },
          { name: 'args_2', shape: [1, 1, 1, 17] },
          { name: 'args_3', shape: [1, 17, 1024] },
        ],
      },
    },
  };
}

function fakeLiteRt() {
  return {
    loadModel: vi.fn().mockResolvedValue(fakeModel()),
    loadNpy: vi.fn().mockResolvedValue(new Float32Array(3072 * 1024)),
    fetchBuffer: vi.fn().mockResolvedValue(new ArrayBuffer(1)),
  };
}

function fakeContext() {
  return {
    backend: 'wasm',
    assets: {
      resolve: vi.fn().mockResolvedValue(new TextEncoder().encode('{}').buffer),
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
```

Note: the decode loop samples cb0=42 on frame 0 (never EOS=2150), MTP predict must exist — `phases/generator.ts` will call `mtp.predict` and `talker.decode`. Since the loop runs at most `maxFrames=1`, only one iteration executes. The `talker`/`mtp` instances come from the real classes wrapping `fakeModel()` (their `.run` calls will hit the real `CompiledModel.run` on the fake object — see Step 3; the test mocks at the `../sampler`/`../prompt`/`../npy-parser` layer, so the generation loop itself is exercised).

- [ ] **Step 3: Run test to verify it fails**

Run: `pnpm --filter @litert-playground/qwen3-tts test phases/generator`
Expected: FAIL — module `./generator` does not exist (or type errors).

- [ ] **Step 4: Implement `GeneratorPhase`**

Create `packages/qwen3-tts/src/phases/generator.ts` (extracted from the current `pipeline.ts` load+run minus the codec, returning `CodecFrames`):

```ts
import type { RuntimeContext, PipelineProgress, ModelManifest } from '@litert-playground/inference-core';
import { InferenceError } from '@litert-playground/inference-core';
import { BPETokenizer } from '../tokenizer';
import { Talker } from '../talker';
import { MTP } from '../mtp';
import { sample, type SampleOpts } from '../sampler';
import { parseNpy, parseNpz } from '../npy-parser';
import { buildPrompt } from '../prompt';
import { createQwen3TtsManifest, type Qwen3TtsVariant } from '../manifest';
import { discoverMtpShapes, discoverTalkerShapes } from '../shape-discovery';
import { parseFp16Npy, type Fp16Table } from '../fp16-table';
import { HIDDEN, CODEC_VOCAB, CODEC_EOS, NEG_INF, LANGUAGE_IDS, DEFAULTS, type QwenTtsInput, type QwenTtsConfig } from '../types';
import { packCodecFrames, type CodecFrames } from '../codec-frames';

function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

export interface GeneratorPhaseOptions {
  onProgress?: (progress: PipelineProgress) => void;
}

export class GeneratorPhase {
  readonly manifest: ModelManifest;
  readonly name = 'generator';
  loadMs = 0;
  compileMs = 0;
  inferenceMs = 0;

  private readonly variant: Qwen3TtsVariant;
  private readonly onProgress?: (progress: PipelineProgress) => void;
  private context?: RuntimeContext;
  private tokenizer?: BPETokenizer;
  private talker?: Talker;
  private mtp?: MTP;
  private codecEmb?: Float32Array;
  private mtpEmb?: Float32Array;
  private textEmbData?: Fp16Table;
  private projW1?: Float32Array;
  private projB1?: Float32Array;
  private projW2?: Float32Array;
  private projB2?: Float32Array;

  constructor(variant: Qwen3TtsVariant, options: GeneratorPhaseOptions = {}) {
    this.variant = variant;
    this.onProgress = options.onProgress;
    this.manifest = createQwen3TtsManifest(variant);
  }

  async load(context: RuntimeContext): Promise<void> {
    this.context = context;
    const loadStart = performance.now();
    this.report({ phase: 'loading', step: 0, total: 7 });
    const tokData = await context.assets.resolve({ id: 'tokenizer', path: 'tokenizer.json' });
    this.tokenizer = new BPETokenizer(JSON.parse(new TextDecoder().decode(tokData)));
    this.report({ phase: 'loading', step: 1, total: 7 });
    this.codecEmb = await context.liteRt.loadNpy('tables/codec_embedding_fp32.npy');
    this.report({ phase: 'loading', step: 2, total: 7 });
    this.mtpEmb = await context.liteRt.loadNpy('tables/mtp_embeddings_fp16.npy');
    this.report({ phase: 'loading', step: 3, total: 7 });
    this.textEmbData = parseFp16Npy(await context.liteRt.fetchBuffer('tables/text_embedding_fp16.npy'));
    this.report({ phase: 'loading', step: 4, total: 7 });
    const projBuf = await context.assets.resolve({ id: 'text-projection', path: 'tables/text_projection_fp32.npz' });
    const proj = await parseNpz(projBuf);
    this.projW1 = proj['w1'];
    this.projB1 = proj['b1'];
    this.projW2 = proj['w2'];
    this.projB2 = proj['b2'];
    this.report({ phase: 'loading', step: 5, total: 7 });
    const compileStart = performance.now();
    const talkerModel = await context.liteRt.loadModel(this.variant.talker);
    this.report({ phase: 'loading', step: 6, total: 7 });
    const mtpModel = await context.liteRt.loadModel(this.variant.mtp);
    this.compileMs = performance.now() - compileStart;
    const talkerShapes = discoverTalkerShapes(talkerModel);
    const mtpShapes = discoverMtpShapes(mtpModel);
    this.talker = new Talker(talkerModel, talkerShapes);
    this.mtp = new MTP(mtpModel, {
      mtpEmbeddings: this.mtpEmb,
      codecEmbeddings: this.codecEmb,
      numCacheSlots: mtpShapes.cacheLen,
      cacheShape: mtpShapes.kvShape,
    });
    this.loadMs = performance.now() - loadStart;
  }

  async generate(input: QwenTtsInput, config: QwenTtsConfig, signal?: AbortSignal): Promise<CodecFrames> {
    const ctx = this.context!;
    const cfg = { ...DEFAULTS, ...config };
    const lang = LANGUAGE_IDS[cfg.language] ?? LANGUAGE_IDS.english;
    const inferenceStart = performance.now();
    try {
      const voicePath = `voices/${cfg.voice}.npy`;
      const speakerBuf = await ctx.assets.resolve({ id: 'voice', path: voicePath, optional: true }, { signal });
      const speakerEmb = parseNpy(speakerBuf);
      const { prefill, trailing, ttsPad } = buildPrompt(
        input.text,
        speakerEmb,
        lang,
        this.tokenizer!,
        this.codecEmb!,
        this.textEmbData!,
        (row) => this.projectText(row),
      );
      if (signal?.aborted) throw new InferenceError('CANCELLED', 'Cancelled before prefill');
      this.report({ phase: 'prefill', step: 0, total: 1 });
      const kv = this.talker!.createEmptyKv();
      const sl = prefill.length / HIDDEN;
      let { logits, hidden, kvCache } = await this.talker!.prefill(prefill, kv, sl);
      const sampleOpts: SampleOpts = {
        temperature: cfg.temperature,
        topK: cfg.topK,
        repetitionPenalty: cfg.repetitionPenalty,
        prevTokens: [],
      };
      const allFrames: number[][] = [];
      let currentLogits = logits;
      let currentHidden = hidden;
      let currentKv = kvCache;
      const maxFrames = cfg.maxFrames;
      for (let frame = 0; frame < maxFrames; frame++) {
        if (signal?.aborted) throw new InferenceError('CANCELLED', 'Cancelled during generation');
        this.report({ phase: 'decode', step: frame, total: maxFrames });
        const scores = new Float32Array(currentLogits);
        for (let i = 2048; i < CODEC_VOCAB; i++) scores[i] = NEG_INF;
        scores[CODEC_EOS] = 0;
        if (frame < 2) scores[CODEC_EOS] = NEG_INF;
        for (const token of sampleOpts.prevTokens) {
          scores[token] = scores[token] > 0 ? scores[token] / sampleOpts.repetitionPenalty : scores[token] * sampleOpts.repetitionPenalty;
        }
        const cb0 = sample(scores, { ...sampleOpts, prevTokens: [] });
        if (cb0 === CODEC_EOS) break;
        sampleOpts.prevTokens.push(cb0);
        this.report({ phase: 'mtp', step: frame, total: maxFrames });
        const residual = await this.mtp!.predict(currentHidden, cb0, { temperature: cfg.temperature, topK: cfg.topK });
        allFrames.push([cb0, ...residual]);
        const frameIdx = frame < trailing.length ? frame : trailing.length - 1;
        const textCond = frameIdx >= 0 ? trailing[frameIdx] : ttsPad;
        const sumEmb = new Float32Array(HIDDEN);
        const cb0Emb = this.codecEmb!.slice(cb0 * HIDDEN, (cb0 + 1) * HIDDEN);
        for (let i = 0; i < HIDDEN; i++) sumEmb[i] += cb0Emb[i];
        for (let r = 0; r < residual.length; r++) {
          const re = this.mtpEmb!.slice(r * CODEC_VOCAB * HIDDEN + residual[r] * HIDDEN, r * CODEC_VOCAB * HIDDEN + (residual[r] + 1) * HIDDEN);
          for (let i = 0; i < HIDDEN; i++) sumEmb[i] += re[i] / residual.length;
        }
        for (let i = 0; i < HIDDEN; i++) sumEmb[i] += textCond[i];
        const result = await this.talker!.decode(sumEmb, currentKv, frame + 1);
        currentLogits = result.logits;
        currentHidden = result.hidden;
        currentKv = result.kvCache;
      }
      this.inferenceMs = performance.now() - inferenceStart;
      return packCodecFrames(allFrames);
    } catch (e) {
      throw e instanceof InferenceError ? e : new InferenceError('INFERENCE_FAILED', String(e), { cause: e });
    }
  }

  dispose(): void {
    this.talker = undefined;
    this.mtp = undefined;
    this.tokenizer = undefined;
    this.codecEmb = undefined;
    this.mtpEmb = undefined;
    this.textEmbData = undefined;
    this.projW1 = undefined;
    this.projB1 = undefined;
    this.projW2 = undefined;
    this.projB2 = undefined;
    this.context = undefined;
  }

  private projectText(row: Float32Array): Float32Array {
    const hiddenDim = HIDDEN * 4;
    const inputDim = this.projW1!.length / hiddenDim;
    const h = new Float32Array(hiddenDim);
    for (let i = 0; i < hiddenDim; i++) {
      let acc = this.projB1![i];
      for (let j = 0; j < inputDim; j++) acc += this.projW1![j * hiddenDim + i] * row[j];
      h[i] = silu(acc);
    }
    const out = new Float32Array(HIDDEN);
    for (let i = 0; i < HIDDEN; i++) {
      let acc = this.projB2![i];
      for (let j = 0; j < hiddenDim; j++) acc += this.projW2![j * HIDDEN + i] * h[j];
      out[i] = acc;
    }
    return out;
  }

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress);
  }
}
```

- [ ] **Step 5: Update `pipeline.ts` exports**

In `packages/qwen3-tts/src/pipeline.ts`, replace the local declarations of `QwenTtsInput`, `QwenTtsConfig`, `DEFAULTS`, `HIDDEN`, `CODEC_VOCAB`, `CODEC_EOS`, `NEG_INF`, `LANGUAGE_IDS` with a re-export (keep pipeline compiling by deleting the moved constants and adding):

```ts
export type { QwenTtsInput, QwenTtsConfig } from './types';
```

(the constants are consumed internally by whatever remains in pipeline.ts — pipeline.ts will be rewritten in Task 7, so for now importing `DEFAULTS`/`HIDDEN` etc. from `./types` where still used is acceptable, or simply leave the pipeline.ts body referencing them via a new import from `./types`.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `pnpm --filter @litert-playground/qwen3-tts test phases/generator`
Expected: PASS (2 tests). If the `talker.run`/`mtp.predict` real classes fail against `fakeModel()`, adjust the test to mock `../talker`, `../mtp`, `../tokenizer` too (returning stubs like the old `receipt.test.ts` did) — the point is to exercise the load/generate orchestration, not real inference.

- [ ] **Step 7: Typecheck and commit**

```bash
pnpm --filter @litert-playground/qwen3-tts typecheck
git add packages/qwen3-tts/src/types.ts packages/qwen3-tts/src/phases/generator.ts packages/qwen3-tts/src/phases/generator.test.ts packages/qwen3-tts/src/pipeline.ts
git commit -m "feat(qwen3-tts): extract GeneratorPhase and shared TTS types"
```

---

### Task 5: DecoderPhase

**Files:**
- Create: `packages/qwen3-tts/src/phases/decoder.ts`
- Create: `packages/qwen3-tts/src/phases/decoder.test.ts`

**Interfaces:**
- Consumes: `CodecDecoder` from `../codec`, `discoverCodecShapes` from `../shape-discovery`, `unpackCodecFrames`/`CodecFrames` from `../codec-frames`, `createQwen3TtsManifest`/`Qwen3TtsVariant` from `../manifest`, `RuntimeContext`/`PipelineProgress`/`ModelManifest`/`InferenceError` from `@litert-playground/inference-core`.
- Produces: `DecoderPhase` class with `manifest`, `name = 'decoder'`, `loadMs`, `compileMs`, `inferenceMs`, `async load(context)`, `async decode(frames: CodecFrames, signal?): Promise<Float32Array>`, `dispose()`.

- [ ] **Step 1: Write the failing test**

Create `packages/qwen3-tts/src/phases/decoder.test.ts`:

```ts
import { describe, expect, it, beforeEach } from 'vitest';

vi.mock('../codec', () => ({
  CodecDecoder: class {
    decode = vi.fn().mockResolvedValue(new Float32Array([0, 0.1, 0.2]));
  },
}));

import { DecoderPhase } from './decoder';
import { qwen3TtsVariants } from '../manifest';
import { packCodecFrames } from '../codec-frames';

describe('DecoderPhase', () => {
  let phase: DecoderPhase;
  beforeEach(() => {
    phase = new DecoderPhase(qwen3TtsVariants.int4);
  });

  it('compiles only the codec graph', async () => {
    const liteRt = {
      loadModel: vi.fn().mockResolvedValue({
        signatures: { decode: { getInputDetails: () => [{ name: 'args_0', shape: [1, 16, 64] }] } },
      }),
    };
    const ctx = { backend: 'wasm', assets: {}, liteRt };
    await phase.load(ctx);
    expect(liteRt.loadModel).toHaveBeenCalledWith('codec_decoder_fp32.tflite');
    expect(phase.name).toBe('decoder');
  });

  it('decodes CodecFrames into a Float32Array', async () => {
    const liteRt = {
      loadModel: vi.fn().mockResolvedValue({
        signatures: { decode: { getInputDetails: () => [{ name: 'args_0', shape: [1, 16, 64] }] } },
      }),
    };
    await phase.load({ backend: 'wasm', assets: {}, liteRt });
    const audio = await phase.decode(packCodecFrames([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]));
    expect(audio).toBeInstanceOf(Float32Array);
    expect(audio.length).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --filter @litert-playground/qwen3-tts test phases/decoder`
Expected: FAIL — module `./decoder` does not exist.

- [ ] **Step 3: Implement**

Create `packages/qwen3-tts/src/phases/decoder.ts`:

```ts
import type { RuntimeContext, PipelineProgress, ModelManifest } from '@litert-playground/inference-core';
import { InferenceError } from '@litert-playground/inference-core';
import { CodecDecoder } from '../codec';
import { discoverCodecShapes } from '../shape-discovery';
import { createQwen3TtsManifest, type Qwen3TtsVariant } from '../manifest';
import { unpackCodecFrames, type CodecFrames } from '../codec-frames';

export interface DecoderPhaseOptions {
  onProgress?: (progress: PipelineProgress) => void;
}

export class DecoderPhase {
  readonly manifest: ModelManifest;
  readonly name = 'decoder';
  loadMs = 0;
  compileMs = 0;
  inferenceMs = 0;

  private readonly variant: Qwen3TtsVariant;
  private readonly onProgress?: (progress: PipelineProgress) => void;
  private context?: RuntimeContext;
  private codec?: CodecDecoder;

  constructor(variant: Qwen3TtsVariant, options: DecoderPhaseOptions = {}) {
    this.variant = variant;
    this.onProgress = options.onProgress;
    this.manifest = createQwen3TtsManifest(variant);
  }

  async load(context: RuntimeContext): Promise<void> {
    this.context = context;
    const loadStart = performance.now();
    this.report({ phase: 'loading', step: 0, total: 1 });
    const compileStart = performance.now();
    const codecModel = await context.liteRt.loadModel(this.variant.codec);
    this.compileMs = performance.now() - compileStart;
    const codecShapes = discoverCodecShapes(codecModel);
    this.codec = new CodecDecoder(codecModel, { chunkSize: codecShapes.chunkSize });
    this.loadMs = performance.now() - loadStart;
  }

  async decode(frames: CodecFrames, signal?: AbortSignal): Promise<Float32Array> {
    if (signal?.aborted) throw new InferenceError('CANCELLED', 'Cancelled before decode');
    const inferenceStart = performance.now();
    try {
      this.report({ phase: 'codec', step: 0, total: 1 });
      const allFrames = unpackCodecFrames(frames);
      const audio = await this.codec!.decode(allFrames);
      this.inferenceMs = performance.now() - inferenceStart;
      return audio;
    } catch (e) {
      throw e instanceof InferenceError ? e : new InferenceError('INFERENCE_FAILED', String(e), { cause: e });
    }
  }

  dispose(): void {
    this.codec = undefined;
    this.context = undefined;
  }

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress);
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm --filter @litert-playground/qwen3-tts test phases/decoder`
Expected: PASS.

- [ ] **Step 5: Typecheck and commit**

```bash
pnpm --filter @litert-playground/qwen3-tts typecheck
git add packages/qwen3-tts/src/phases/decoder.ts packages/qwen3-tts/src/phases/decoder.test.ts
git commit -m "feat(qwen3-tts): extract DecoderPhase"
```

---

### Task 6: Worker module bridges

**Files:**
- Create: `packages/qwen3-tts/src/workers/generator.worker.ts`
- Create: `packages/qwen3-tts/src/workers/decoder.worker.ts`

**Interfaces:**
- Consumes: `GeneratorPhase`/`DecoderPhase`, protocol types + `serializeError`, `createLiteRtRuntime` from `@litert-playground/runtime-litert`, `createCachingAssetResolver`/`createHttpAssetResolver`/`InferencePhaseReceipt` from `@litert-playground/inference-core`.
- Produces: two ESM modules that each set `self.onmessage`. These are dynamically `import()`-ed by the classic worker shells (Task 8). The host-side `pipeline.ts` never imports these modules directly — it talks to the workers via the protocol.

- [ ] **Step 1: Implement `generator.worker.ts`**

Create `packages/qwen3-tts/src/workers/generator.worker.ts`:

```ts
/// <reference lib="webworker" />
import { createLiteRtRuntime } from '@litert-playground/runtime-litert';
import { createCachingAssetResolver, createHttpAssetResolver } from '@litert-playground/inference-core';
import { GeneratorPhase } from '../phases/generator';
import type { GeneratorWorkerRequest, GeneratorWorkerResponse } from './protocol';
import { serializeError } from './protocol';

let phase: GeneratorPhase | undefined;

async function buildContext(modelBase: string) {
  const assets = createCachingAssetResolver(createHttpAssetResolver(modelBase));
  return createLiteRtRuntime({ assets });
}

self.onmessage = async (event: MessageEvent<GeneratorWorkerRequest>) => {
  const req = event.data;
  try {
    if (req.type === 'initialize') {
      const context = await buildContext(req.modelBase);
      phase = new GeneratorPhase(req.variant, {
        onProgress: (progress) => {
          self.postMessage({ type: 'progress', progress } satisfies GeneratorWorkerResponse);
        },
      });
      await phase.load(context);
      self.postMessage({ type: 'ready' } satisfies GeneratorWorkerResponse);
      return;
    }
    if (req.type === 'generate') {
      if (!phase) throw new Error('generator not initialized');
      const frames = await phase.generate(req.input, req.config);
      const phaseReceipt = {
        name: phase.name,
        backend: 'wasm',
        loadMs: phase.loadMs,
        compileMs: phase.compileMs,
        inferenceMs: phase.inferenceMs,
      };
      self.postMessage(
        { type: 'frames', requestId: req.requestId, frames, phaseReceipt } satisfies GeneratorWorkerResponse,
        [frames.frames.buffer],
      );
      return;
    }
    if (req.type === 'cancel') {
      // Host terminates the worker for cancellation; nothing to do here.
      return;
    }
  } catch (cause) {
    const requestId = 'requestId' in req ? req.requestId : undefined;
    self.postMessage({ type: 'error', requestId, error: serializeError(cause) } satisfies GeneratorWorkerResponse);
  }
};

export {};
```

- [ ] **Step 2: Implement `decoder.worker.ts`**

Create `packages/qwen3-tts/src/workers/decoder.worker.ts`:

```ts
/// <reference lib="webworker" />
import { createLiteRtRuntime } from '@litert-playground/runtime-litert';
import { createCachingAssetResolver, createHttpAssetResolver } from '@litert-playground/inference-core';
import { DecoderPhase } from '../phases/decoder';
import type { DecoderWorkerRequest, DecoderWorkerResponse } from './protocol';
import { serializeError } from './protocol';

let phase: DecoderPhase | undefined;

async function buildContext(modelBase: string) {
  const assets = createCachingAssetResolver(createHttpAssetResolver(modelBase));
  return createLiteRtRuntime({ assets });
}

self.onmessage = async (event: MessageEvent<DecoderWorkerRequest>) => {
  const req = event.data;
  try {
    if (req.type === 'initialize') {
      const context = await buildContext(req.modelBase);
      phase = new DecoderPhase(req.variant, {
        onProgress: (progress) => {
          self.postMessage({ type: 'progress', progress } satisfies DecoderWorkerResponse);
        },
      });
      await phase.load(context);
      self.postMessage({ type: 'ready' } satisfies DecoderWorkerResponse);
      return;
    }
    if (req.type === 'decode') {
      if (!phase) throw new Error('decoder not initialized');
      const audio = await phase.decode(req.frames);
      const phaseReceipt = {
        name: phase.name,
        backend: 'wasm',
        loadMs: phase.loadMs,
        compileMs: phase.compileMs,
        inferenceMs: phase.inferenceMs,
      };
      self.postMessage(
        { type: 'audio', requestId: req.requestId, audio, phaseReceipt } satisfies DecoderWorkerResponse,
        [audio.buffer],
      );
      return;
    }
  } catch (cause) {
    const requestId = 'requestId' in req ? req.requestId : undefined;
    self.postMessage({ type: 'error', requestId, error: serializeError(cause) } satisfies DecoderWorkerResponse);
  }
};

export {};
```

- [ ] **Step 3: Typecheck and commit**

Run: `pnpm --filter @litert-playground/qwen3-tts typecheck`
Expected: clean. (These files are compiled under vitest/tsc with the `lib: ["webworker"]` need — if `tsconfig` lacks it, `self`/`postMessage` resolve via the `/// <reference lib="webworker" />` directive; adjust only if tsc complains.)

```bash
git add packages/qwen3-tts/src/workers/generator.worker.ts packages/qwen3-tts/src/workers/decoder.worker.ts
git commit -m "feat(qwen3-tts): add generator and decoder worker bridges"
```

---

### Task 7: Rewrite pipeline.ts as the phased orchestrator

**Files:**
- Modify: `packages/qwen3-tts/src/pipeline.ts` (rewrite)
- Modify: `packages/qwen3-tts/src/pipeline.test.ts`
- Modify: `packages/qwen3-tts/src/receipt.test.ts`
- Create: `packages/qwen3-tts/src/workers/host.ts` (host-side worker client helpers, so pipeline.ts stays small and testable)

**Interfaces:**
- Consumes: `GeneratorPhase`/`DecoderPhase`, protocol types, `CodecFrames`, `checkAudioValid`/`createInferenceReceipt`/`InferenceError`/`Pipeline`/`PipelineStatus`/`AudioInferenceResult` from `@litert-playground/inference-core`, `QwenTtsInput`/`QwenTtsConfig`/`DEFAULTS` from `./types`.
- Produces: rewritten `Qwen3TtsPipeline` with same public surface; `Qwen3TtsPipelineOptions { workerBase?: string; modelBase?: string }` as optional 2nd constructor arg (defaults `workerBase: '/litert-wasm/'`, `modelBase: '/models/qwen3-tts/'`); `runHostGenerator(...)` / `runHostDecoder(...)` helpers in `workers/host.ts` that manage the Worker lifecycle and message protocol (used by pipeline; unit-testable with a fake `Worker` global).

- [ ] **Step 1: Write the failing tests**

Rewrite `packages/qwen3-tts/src/pipeline.test.ts`:

```ts
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { Qwen3TtsPipeline } from './pipeline';
import { qwen3TtsVariants } from './manifest';

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
      backend: 'wasm',
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
      backend: 'wasm',
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
    const sent: { target: { postMessage: (m: unknown, t?: unknown[]) => void }; id: string }[] = [];
    const FakeWorker = vi.fn().mockImplementation(function (this: { postMessage: (m: unknown, t?: unknown[]) => void; terminate: () => void; id: string }) {
      this.id = `w${sent.length}`;
      this.postMessage = vi.fn((m: unknown, t?: unknown[]) => {
        sent.push({ target: this, ...(m as object) } as never);
        // respond as the worker would
        const msg = m as { type: string; requestId?: number; frames?: unknown };
        if (msg.type === 'initialize') {
          queueMicrotask(() => this._respond && this._respond({ type: 'ready' }));
        } else if (msg.type === 'generate') {
          queueMicrotask(() => this._respond && this._respond({ type: 'frames', requestId: msg.requestId, frames: { frames: new Uint16Array([1, 2, 3, 4]), frameCount: 1, codebooks: 4 }, phaseReceipt: { name: 'generator', backend: 'wasm' } }));
        } else if (msg.type === 'decode') {
          queueMicrotask(() => this._respond && this._respond({ type: 'audio', requestId: msg.requestId, audio: new Float32Array([0.1, 0.2]), phaseReceipt: { name: 'decoder', backend: 'wasm' } }));
        }
      });
      this.terminate = vi.fn();
      this._respond = undefined;
    });
    // wire _respond by pushing FakeWorker into globalThis
    vi.stubGlobal('Worker', FakeWorker);
    // hook responses: patch FakeWorker instances to capture their message handler
    (FakeWorker as unknown as { instances: unknown[] }).instances = sent;
    const context = { backend: 'wasm', assets: { resolve: vi.fn() }, liteRt: {} };
    await pipeline.load(context);
    const result = await pipeline.run({ text: 'hello' }, { maxFrames: 1 });
    expect(result.kind).toBe('audio');
    expect(sent.filter((s) => s.type === 'initialize').length).toBe(2);
    expect(sent.filter((s) => s.type === 'generate').length).toBe(1);
    expect(sent.filter((s) => s.type === 'decode').length).toBe(1);
    vi.unstubAllGlobals();
  });
});
```

Note: the worker-mode test's FakeWorker must support the message listener wiring. Implement `workers/host.ts` so the pipeline sets `worker.onmessage` (not `addEventListener`); then the FakeWorker's `onmessage` setter can capture `_respond`. If the harness proves fiddly, simplify the test to assert that `new Worker` was called with the two expected shell URLs and that `terminate` was called twice. The essential assertions: two workers created (generator + decoder), both `terminate()`d, protocol messages sent.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm --filter @litert-playground/qwen3-tts test pipeline`
Expected: FAIL — `load()` currently compiles models (loadModel called) and run() lacks phases.

- [ ] **Step 3: Implement `workers/host.ts`**

Create `packages/qwen3-tts/src/workers/host.ts`:

```ts
import type { InferencePhaseReceipt } from '@litert-playground/inference-core';
import { InferenceError } from '@litert-playground/inference-core';
import type { CodecFrames } from '../codec-frames';
import type { QwenTtsInput, QwenTtsConfig } from '../types';
import type { Qwen3TtsVariant } from '../manifest';
import type {
  GeneratorWorkerRequest,
  GeneratorWorkerResponse,
  DecoderWorkerRequest,
  DecoderWorkerResponse,
} from './protocol';

export interface GeneratorOutcome {
  frames: CodecFrames;
  phaseReceipt: InferencePhaseReceipt;
}

export interface DecoderOutcome {
  audio: Float32Array;
  phaseReceipt: InferencePhaseReceipt;
}

type ProgressHandler = (p: { phase: string; step: number; total: number }) => void;

function waitFor<T extends { type: string }>(
  worker: Worker,
  expectType: T['type'],
  requestId?: number,
  onProgress?: ProgressHandler,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    worker.onmessage = (event: MessageEvent<T>) => {
      const msg = event.data;
      if (msg.type === 'progress') {
        onProgress?.(msg.progress);
        return;
      }
      if (msg.type === 'error') {
        reject(new InferenceError(msg.error.code === 'CANCELLED' ? 'CANCELLED' : 'INFERENCE_FAILED', msg.error.message));
        return;
      }
      if (msg.type === expectType && (requestId === undefined || msg.requestId === requestId)) {
        resolve(msg);
      }
    };
    worker.onerror = (event) => {
      reject(new InferenceError('INFERENCE_FAILED', `worker error: ${event.message}`));
    };
  });
}

export async function runHostGenerator(
  worker: Worker,
  variant: Qwen3TtsVariant,
  modelBase: string,
  input: QwenTtsInput,
  config: QwenTtsConfig,
  onProgress?: ProgressHandler,
): Promise<GeneratorOutcome> {
  const init = waitFor<GeneratorWorkerResponse>(worker, 'ready');
  worker.postMessage({ type: 'initialize', variant, modelBase } satisfies GeneratorWorkerRequest);
  await init;
  const framesWait = waitFor<GeneratorWorkerResponse>(worker, 'frames', 1, onProgress);
  worker.postMessage({ type: 'generate', requestId: 1, input, config } satisfies GeneratorWorkerRequest);
  const msg = (await framesWait) as Extract<GeneratorWorkerResponse, { type: 'frames' }>;
  return { frames: msg.frames, phaseReceipt: msg.phaseReceipt };
}

export async function runHostDecoder(
  worker: Worker,
  variant: Qwen3TtsVariant,
  modelBase: string,
  frames: CodecFrames,
  onProgress?: ProgressHandler,
): Promise<DecoderOutcome> {
  const init = waitFor<DecoderWorkerResponse>(worker, 'ready');
  worker.postMessage({ type: 'initialize', variant, modelBase } satisfies DecoderWorkerRequest);
  await init;
  const audioWait = waitFor<DecoderWorkerResponse>(worker, 'audio', 1, onProgress);
  worker.postMessage({ type: 'decode', requestId: 1, frames } satisfies DecoderWorkerRequest, [frames.frames.buffer]);
  const msg = (await audioWait) as Extract<DecoderWorkerResponse, { type: 'audio' }>;
  return { audio: msg.audio, phaseReceipt: msg.phaseReceipt };
}
```

- [ ] **Step 4: Implement the rewritten `pipeline.ts`**

Rewrite `packages/qwen3-tts/src/pipeline.ts`:

```ts
import type { PipelineStatus, PipelineProgress, RuntimeContext, AudioInferenceResult } from '@litert-playground/inference-core';
import { InferenceError, checkAudioValid, createInferenceReceipt } from '@litert-playground/inference-core';
import { createQwen3TtsManifest, type Qwen3TtsVariant } from './manifest';
import type { QwenTtsInput, QwenTtsConfig } from './types';
import { DEFAULTS } from './types';
import { GeneratorPhase } from './phases/generator';
import { DecoderPhase } from './phases/decoder';
import { runHostGenerator, runHostDecoder } from './workers/host';

export type { QwenTtsInput, QwenTtsConfig } from './types';

export interface Qwen3TtsPipelineOptions {
  workerBase?: string;
  modelBase?: string;
}

const WORKER_BASE = '/litert-wasm/';
const MODEL_BASE = '/models/qwen3-tts/';

export class Qwen3TtsPipeline {
  readonly manifest;
  status: PipelineStatus = 'idle';
  onProgress?: (progress: PipelineProgress) => void;

  private readonly variant: Qwen3TtsVariant;
  private readonly workerBase: string;
  private readonly modelBase: string;
  private context?: RuntimeContext;
  private disposed = false;

  constructor(
    variant: Qwen3TtsVariant = qwen3TtsVariants.fp32,
    options: Qwen3TtsPipelineOptions = {},
  ) {
    this.variant = variant;
    this.workerBase = options.workerBase ?? WORKER_BASE;
    this.modelBase = options.modelBase ?? MODEL_BASE;
    this.manifest = createQwen3TtsManifest(variant);
  }

  async load(context: RuntimeContext): Promise<void> {
    if (this.disposed) throw new InferenceError('INFERENCE_FAILED', 'Pipeline disposed');
    if (this.status === 'ready') return;
    this.context = context;
    this.status = 'loading';
    this.report({ phase: 'loading', step: 0, total: 1 });
    if (!context.liteRt || !context.assets) {
      this.status = 'error';
      throw new InferenceError('INFERENCE_FAILED', 'Invalid runtime context');
    }
    this.status = 'ready';
  }

  async run(input: QwenTtsInput, config?: QwenTtsConfig, signal?: AbortSignal): Promise<AudioInferenceResult> {
    if (this.status !== 'ready') throw new InferenceError('INFERENCE_FAILED', 'Pipeline not ready');
    this.status = 'running';
    const cfg = { ...DEFAULTS, ...config };
    const inferenceStart = performance.now();
    const backend = this.context?.backend ?? 'wasm';
    try {
      let audio: Float32Array;
      let phases: NonNullable<AudioInferenceResult['receipt']> extends infer R ? NonNullable<R['phases']> : never;

      if (typeof Worker === 'undefined') {
        // Direct mode (Node / vitest): run phases in-process.
        const genPhase = new GeneratorPhase(this.variant, { onProgress: (p) => this.report(p) });
        let frames;
        try {
          await genPhase.load(this.context!);
          frames = await genPhase.generate(input, cfg, signal);
        } finally {
          genPhase.dispose();
        }
        const decPhase = new DecoderPhase(this.variant, { onProgress: (p) => this.report(p) });
        try {
          await decPhase.load(this.context!);
          audio = await decPhase.decode(frames, signal);
        } finally {
          decPhase.dispose();
        }
        phases = [
          { name: genPhase.name, backend, loadMs: genPhase.loadMs, compileMs: genPhase.compileMs, inferenceMs: genPhase.inferenceMs },
          { name: decPhase.name, backend, loadMs: decPhase.loadMs, compileMs: decPhase.compileMs, inferenceMs: decPhase.inferenceMs },
        ];
      } else {
        // Worker mode (browser): two disposable classic workers.
        const genWorker = new Worker(this.workerBase + 'generator-worker.js');
        const gen = await runHostGenerator(genWorker, this.variant, this.modelBase, input, cfg, (p) => this.report({ ...p }));
        genWorker.terminate();
        const decWorker = new Worker(this.workerBase + 'decoder-worker.js');
        const dec = await runHostDecoder(decWorker, this.variant, this.modelBase, gen.frames, (p) => this.report({ ...p }));
        decWorker.terminate();
        audio = dec.audio;
        phases = [gen.phaseReceipt, dec.phaseReceipt];
      }

      const duration = audio.length / 24000;
      const warnings = checkAudioValid(audio, 24000, 1, duration);
      if (warnings.length > 0) console.warn('Qwen3TTS output warnings:', warnings);
      this.status = 'ready';
      return {
        kind: 'audio',
        samples: audio,
        sampleRate: 24000,
        channels: 1,
        durationSeconds: duration,
        receipt: createInferenceReceipt({
          manifest: this.manifest,
          backend,
          loadMs: 0,
          compileMs: 0,
          inferenceStart,
          inputSummary: `${input.text.length} characters`,
          outputSummary: `${audio.length} samples at 24000Hz, 1 channel`,
          warnings,
          phases,
        }),
      };
    } catch (e) {
      this.status = 'ready';
      throw e instanceof InferenceError ? e : new InferenceError('INFERENCE_FAILED', String(e), { cause: e });
    }
  }

  async dispose(): Promise<void> {
    this.disposed = true;
    this.context = undefined;
    this.status = 'disposed';
  }

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress);
  }
}
```

Add `import { qwen3TtsVariants } from './manifest';` to the import list (the code above references it in the constructor default).

- [ ] **Step 5: Update `receipt.test.ts`**

Rewrite `packages/qwen3-tts/src/receipt.test.ts` to drive the pipeline in direct mode with a real (mock-backed) phase path. Simplest robust approach: mock `./phases/generator` and `./phases/decoder` so run() gets deterministic phase outputs, then assert the receipt:

```ts
import { describe, expect, it, vi } from 'vitest';

const genPhase = {
  name: 'generator',
  loadMs: 1,
  compileMs: 2,
  inferenceMs: 3,
  load: vi.fn().mockResolvedValue(undefined),
  generate: vi.fn().mockResolvedValue({ frames: new Uint16Array([1, 2, 3, 4]), frameCount: 1, codebooks: 4 }),
  dispose: vi.fn(),
};
const decPhase = {
  name: 'decoder',
  loadMs: 4,
  compileMs: 5,
  inferenceMs: 6,
  load: vi.fn().mockResolvedValue(undefined),
  decode: vi.fn().mockResolvedValue(new Float32Array([0, 0.1, 0.2])),
  dispose: vi.fn(),
};

vi.mock('./phases/generator', () => ({ GeneratorPhase: class { constructor() {} } }));
vi.mock('./phases/decoder', () => ({ DecoderPhase: class { constructor() {} } }));

import { Qwen3TtsPipeline } from './pipeline';
import { qwen3TtsVariants } from './manifest';

describe('Qwen3TtsPipeline receipt', () => {
  it('attaches phase receipts to successful audio output in direct mode', async () => {
    vi.stubGlobal('Worker', undefined);
    const { GeneratorPhase } = await import('./phases/generator');
    const { DecoderPhase } = await import('./phases/decoder');
    (GeneratorPhase as unknown as { prototype: object }).prototype = Object.assign(Object.create(null), genPhase);
    (DecoderPhase as unknown as { prototype: object }).prototype = Object.assign(Object.create(null), decPhase);
    const pipeline = new Qwen3TtsPipeline(qwen3TtsVariants.int4);
    await pipeline.load({ backend: 'webgpu', assets: { resolve: vi.fn() }, liteRt: {} });
    const result = await pipeline.run({ text: 'hello' }, { maxFrames: 1 });
    expect(result.kind).toBe('audio');
    expect(result.receipt.modelId).toBe('qwen3-tts-12hz-0.6b-base');
    expect(result.receipt.pipelineVersion).toBe('0.4.0');
    expect(result.receipt.backend).toBe('webgpu');
    expect(result.receipt.inputSummary).toContain('5 characters');
    expect(result.receipt.outputSummary).toContain('24000Hz');
    expect(result.receipt.phases).toHaveLength(2);
    expect(result.receipt.phases!.map((p) => p.name)).toEqual(['generator', 'decoder']);
    vi.unstubAllGlobals();
  });
});
```

(If prototype replacement is awkward, instead mock the two phase modules to return the stubs from their constructors directly: `vi.mock('./phases/generator', () => ({ GeneratorPhase: class { ...same shape as genPhase } }))` — either approach is fine as long as `run()` sees deterministic phases.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `pnpm --filter @litert-playground/qwen3-tts test`
Expected: PASS — pipeline.test.ts, receipt.test.ts, phases/*, protocol, codec-frames all green.

- [ ] **Step 7: Typecheck and commit**

```bash
pnpm --filter @litert-playground/qwen3-tts typecheck
git add packages/qwen3-tts/src/pipeline.ts packages/qwen3-tts/src/pipeline.test.ts packages/qwen3-tts/src/receipt.test.ts packages/qwen3-tts/src/workers/host.ts
git commit -m "feat(qwen3-tts): orchestrate phased generation in Qwen3TtsPipeline"
```

---

### Task 8: Example wiring + classic worker shells

**Files:**
- Modify: `examples/vite.config.ts` (extend `litertWasmProxy` to serve two more shell files verbatim from disk)
- Create: `examples/minimal-qwen3-tts/generator-worker.js`
- Create: `examples/minimal-qwen3-tts/decoder-worker.js`
- Modify: `examples/minimal-qwen3-tts/main.tsx` (pass `modelBase` into the pipeline constructor options)

**Interfaces:**
- Consumes: the package worker bridges (`packages/qwen3-tts/src/workers/generator.worker.ts` / `decoder.worker.ts`) via dynamic `import()`; `litertWasmProxy` middleware; `Qwen3TtsPipelineOptions`.
- Produces: two classic worker shells served at `/litert-wasm/generator-worker.js` and `/litert-wasm/decoder-worker.js`; `main.tsx` constructs `new Qwen3TtsPipeline(qwen3TtsVariants.int4, { modelBase })`.

- [ ] **Step 1: Write the classic worker shells**

Create `examples/minimal-qwen3-tts/generator-worker.js`:

```js
// Classic worker shell for the generator phase. Served verbatim by litertWasmProxy
// so self.location is a real http URL (Emscripten resolves the wasm relative to it).
// The module bridge is loaded via dynamic import() (legal in classic workers).
(async () => {
  const origin = self.location.origin;
  const moduleUrl = `${origin}/packages/qwen3-tts/src/workers/generator.worker.ts`;
  try {
    await import(moduleUrl);
  } catch (cause) {
    self.postMessage({ type: 'error', error: { code: 'WORKER_BOOTSTRAP_FAILED', message: String(cause), stage: 'worker' } });
  }
})();
```

Create `examples/minimal-qwen3-tts/decoder-worker.js`:

```js
// Classic worker shell for the decoder phase. See generator-worker.js.
(async () => {
  const origin = self.location.origin;
  const moduleUrl = `${origin}/packages/qwen3-tts/src/workers/decoder.worker.ts`;
  try {
    await import(moduleUrl);
  } catch (cause) {
    self.postMessage({ type: 'error', error: { code: 'WORKER_BOOTSTRAP_FAILED', message: String(cause), stage: 'worker' } });
  }
})();
```

- [ ] **Step 2: Extend `litertWasmProxy`**

In `examples/vite.config.ts`, add consts next to `residencyWorkerFile`:

```ts
const generatorWorkerFile = path.resolve(__dirname, 'minimal-qwen3-tts/generator-worker.js');
const decoderWorkerFile = path.resolve(__dirname, 'minimal-qwen3-tts/decoder-worker.js');
const workerShells: Record<string, string> = {
  'residency-worker.js': residencyWorkerFile,
  'generator-worker.js': generatorWorkerFile,
  'decoder-worker.js': decoderWorkerFile,
};
```

Replace the `if (rest === 'residency-worker.js')` branch in the `litertWasmProxy` middleware with:

```ts
if (workerShells[rest]) {
  const file = workerShells[rest];
  res.setHeader('content-type', 'application/javascript');
  res.setHeader('cache-control', 'no-store');
  res.end(fs.readFileSync(file));
  return;
}
```

Add `import fs from 'node:fs';` to the top of the file. (Everything else in `litertWasmProxy` — the jsdelivr proxy branch, header stripping, streaming — stays as-is.)

- [ ] **Step 3: Wire `main.tsx`**

In `examples/minimal-qwen3-tts/main.tsx`, change the pipeline construction to pass `modelBase`:

```tsx
const [pipeline] = useState(() => new Qwen3TtsPipeline(qwen3TtsVariants.int4, { modelBase }));
```

(`modelBase` already exists in the file as `new URL('/models/qwen3-tts/', window.location.href).href`.)

- [ ] **Step 4: Verify the shells are served verbatim**

Restart the dev server (it must be restarted to pick up `vite.config.ts` changes):

```bash
# kill the existing server on 5176, then:
pnpm --filter @litert-playground/example-qwen3-tts dev --host 0.0.0.0 --port 5176 --strictPort
```

In another shell:

```bash
curl -s http://localhost:5176/litert-wasm/generator-worker.js | head -3
curl -s http://localhost:5176/litert-wasm/decoder-worker.js | head -3
curl -s -o /dev/null -w '%{http_code} %{content_type}\n' http://localhost:5176/packages/qwen3-tts/src/workers/generator.worker.ts
curl -s -o /dev/null -w '%{http_code} %{content_type}\n' http://localhost:5176/packages/qwen3-tts/src/workers/decoder.worker.ts
```

Expected: first two return the shell source (no `export {}`); the last two return 200 with `text/javascript` (Vite serving the package module). If the package-source path is not served (Vite may 404 because the workspace package is under `node_modules/@litert-playground/qwen3-tts`), adjust the shell `moduleUrl` to the resolved dev URL by inspecting the URL Vite serves when the example imports the package (open the example in a browser and read `window.importMap`/network tab, or run the example once and read the transformed `main.tsx`). **This is the one empirical step** — it must be resolved by observation, exactly like every earlier browser step.

- [ ] **Step 5: Typecheck the example and commit**

```bash
pnpm --filter @litert-playground/example-qwen3-tts typecheck
git add examples/vite.config.ts examples/minimal-qwen3-tts/generator-worker.js examples/minimal-qwen3-tts/decoder-worker.js examples/minimal-qwen3-tts/main.tsx
git commit -m "feat(example): wire phased Qwen3-TTS classic worker shells"
```

---

### Task 9: Browser acceptance verification

**Files:**
- Use existing: `/tmp/opencode/qwen-browser/check-live.mjs` (streams console/crash, polls status every 10s; adapt target/selectors if needed) and the Vite dev server on 5176.
- Update (record): `docs/superpowers/verification/2026-08-10-package-extraction.md` (append phased-pipeline results) — doc commit happens after evidence.

**Interfaces:**
- Consumes: the running phased pipeline on `http://localhost:5176/examples/minimal-qwen3-tts/`; the 10-step acceptance ladder from the spec.

- [ ] **Step 1: Run the acceptance ladder in the browser**

Run the phased example in headless Chromium via the Playwright harness. Watch for the exact acceptance sequence (stop at the first failure):

| # | Ladder step | Observed signal |
|---|-------------|-----------------|
| 1 | Generator worker initializes | console: runtime init + generator ready |
| 2 | Talker+MTP compile | `Flatbuffer model initialized` × 2 in generator worker |
| 3 | "Testing one two three." produces non-empty CodecFrames | generator `frames` posted; no error |
| 4 | Generator worker terminates | no further generator worker console |
| 5 | Decoder worker initializes | decoder worker runtime init |
| 6 | Codec compiles | `Flatbuffer model initialized` in decoder worker |
| 7 | CodecFrames decode into non-empty Float32 PCM | decoder `audio` posted, length > 0 |
| 8 | `checkAudioValid()` passes | receipt `warnings` empty (or recorded) |
| 9 | `AudioInferenceResult` receipt appears with `phases` | `result.receipt.phases` = [generator, decoder] on page |
| 10 | User hears intelligible speech | audio plays; capture via page `AudioContext` — requires a headed/human check |

Expected timing: talker ~25s + mtp ~180s (generator worker), then codec ~90s (decoder worker) — each run re-downloads because the proxy streams and the browser profile does not cache those responses. Use a short prompt (e.g. "Hello from LiteRT.") and `maxFrames: 128` as `main.tsx` already does.

- [ ] **Step 2: Record evidence**

Append to `docs/superpowers/verification/2026-08-10-package-extraction.md` a `### Phased pipeline (worker residency)` section with the ladder results, per-step pass/fail, timings, receipt `phases`, and any warnings. Do NOT claim audible playback (ladder step 10) unless actually heard.

- [ ] **Step 3: Commit the verification record**

```bash
git add docs/superpowers/verification/2026-08-10-package-extraction.md
git commit -m "docs: record phased pipeline browser acceptance results"
```

---

## Self-Review

**Spec coverage:**
- Public API unchanged → Task 7 (constructor keeps `variant`; options are optional 2nd arg). ✓
- CodecFrames flat Uint16Array + transferable → Tasks 2, 6, 7. ✓
- Phase ownership split → Tasks 4, 5. ✓
- Boring worker protocol with SerializedInferenceError → Task 3 (protocol types) + Task 6 (bridges). ✓
- load() validate-only → Task 7. ✓
- Receipts phase-aware (InferencePhaseReceipt + phases passthrough) → Task 1. ✓
- Cancellation = terminate + CANCELLED → Task 7 run() + worker bridges (hosts terminate workers; bridges throw CANCELLED on abort in direct mode). ✓
- No runtime-litert/text-gen/PodQast/model-conversion changes → all tasks scoped to inference-core, qwen3-tts, examples. ✓
- litertWasmProxy/qwenModelProxy stay; only shell-serving extended → Task 8. ✓
- Deferred items (PodQast batching, OPFS/Cache, lighter codec) → not implemented. ✓
- Acceptance ladder 10 steps → Task 9. ✓

**Placeholder scan:** No TBD/TODO. The one intentionally-empirical step (Task 8 Step 4 module URL) is explicitly marked with the observation procedure, consistent with the project's established browser-verification pattern.

**Type consistency:** `CodecFrames` shape consistent across Tasks 2/4/5/6/7. `GeneratorWorkerRequest`/`DecoderWorkerRequest` type names consistent between Task 3 and Tasks 6/7. `InferencePhaseReceipt` name consistent between Task 1 and Tasks 6/7. `Qwen3TtsPipelineOptions { workerBase, modelBase }` used in Task 7 (pipeline) and Task 8 (main.tsx). `runHostGenerator`/`runHostDecoder` signatures match Task 7 usage.
