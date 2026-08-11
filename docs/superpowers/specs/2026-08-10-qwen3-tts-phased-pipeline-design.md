# Qwen3-TTS Phased Pipeline Design

**Date:** 2026-08-10
**Status:** Approved

## Goal

Restructure the Qwen3-TTS browser runtime into a phased, worker-isolated
pipeline so that all three LiteRT graphs (INT4 Talker, FP32 MTP, FP32 codec
decoder) can be compiled and run in WASM without exceeding the renderer memory
ceiling. The public API must remain `text → Qwen3TtsPipeline → AudioInferenceResult`.

## Background

The three-graph compile blocker was characterized experimentally (commits
`5abd0a9`, `f25097c`):

- Talker + MTP compile together: pass
- Codec alone: pass
- All three in one page: renderer crashes (heap 497/497 MB, not tunable via
  `--max-old-space-size`)
- Hard worker teardown frees enough residency: pass

Staged worker residency is therefore the **intended** Qwen3-TTS execution
model for WASM, not a workaround.

## Architecture

```
Qwen3TtsPipeline.run(text)
        │
        ▼
┌──────────────────────────────┐
│ Generator Worker             │
│ tokenizer / speaker emb      │
│ tables / projection          │
│ Talker / MTP                 │
│ text → CodecFrames           │
└──────────────┬───────────────┘
               │ transferable
               ▼
       CodecFrames artifact
               │
      TERMINATE WORKER
               │
               ▼
┌──────────────────────────────┐
│ Decoder Worker               │
│ codec_decoder                │
│ CodecFrames → Float32 PCM    │
└──────────────┬───────────────┘
               │
      TERMINATE WORKER
               ▼
      AudioInferenceResult
```

### Public API (unchanged)

```ts
const pipeline = new Qwen3TtsPipeline(variant)
await pipeline.load(context)
const audio = await pipeline.run({ text: "Testing one two three." })
```

No `generateFrames()`, `decodeFrames()`, or worker details exposed to callers.

### Direct-mode fallback

- `phases/generator.ts` and `phases/decoder.ts` contain all logic as plain
  classes.
- `workers/generator.worker.ts` and `workers/decoder.worker.ts` are thin
  bridges (postMessage in, postMessage out) that instantiate the phase class.
- `run()` checks `typeof Worker === 'undefined'`; when absent (Node/vitest,
  non-browser runtimes) it executes the phases in-process, passing `CodecFrames`
  by return value instead of postMessage transfer.
- Existing `pipeline.test.ts` continues to run under vitest; the browser path
  exercises the real worker lifecycle.

## CodecFrames (internal phase boundary)

```ts
interface CodecFrames {
  frames: Uint16Array   // flat, contiguous
  frameCount: number
  codebooks: number     // 1 (cb0) + MTP residual count
}
```

Layout is flat: `[frame0 cb0..cbN, frame1 cb0..cbN, ...]`. One contiguous typed
array; transferred Generator → host → Decoder via
`postMessage(message, [frames.frames.buffer])` without copying.

## Phase ownership

### GeneratorPhase

Owns: tokenizer, speaker embedding, codec embedding table, MTP embeddings,
FP16 text embedding table, projection weights, Talker compiled graph, MTP
compiled graph, sampling, prompt construction, KV caches.

Produces only `CodecFrames`. After output crosses the worker boundary:
`generatorWorker.terminate()` (worker/process lifetime is the reliable memory
boundary — never rely on `talker = null` + GC).

### DecoderPhase

Owns: codec decoder compiled graph, codec-specific shape discovery, frame
chunking. Consumes `CodecFrames`. Produces `Float32Array`. Terminates after.

## Worker protocol

Generator:

```ts
type GeneratorWorkerRequest =
  | { type: 'initialize'; variant: Qwen3TtsVariant; modelBase: string }
  | { type: 'generate'; requestId: string; input: QwenTtsInput; config: QwenTtsConfig }
  | { type: 'cancel'; requestId: string }

type GeneratorWorkerResponse =
  | { type: 'ready' }
  | { type: 'progress'; requestId: string; progress: PipelineProgress }
  | { type: 'frames'; requestId: string; frames: CodecFrames }
  | { type: 'error'; requestId?: string; error: SerializedInferenceError }
```

Decoder:

```ts
type DecoderWorkerRequest =
  | { type: 'initialize'; variant: Qwen3TtsVariant }
  | { type: 'decode'; requestId: string; frames: CodecFrames }

type DecoderWorkerResponse =
  | { type: 'ready' }
  | { type: 'audio'; requestId: string; audio: Float32Array }
  | { type: 'error'; requestId?: string; error: SerializedInferenceError }
```

`CodecFrames` and PCM `Float32Array` are transferred by postMessage transfer
list; do not duplicate intermediate data.

## load() semantics

`load()` revalidates manifest/runtime/assets/configuration and prepares
orchestration, then reports `ready`. It does NOT keep models resident.

`run()`:
1. launch Generator worker
2. load/compile Generator phase
3. generate
4. destroy (terminate)
5. launch Decoder worker
6. load/compile Decoder phase
7. decode
8. destroy (terminate)

Per-invocation cost is higher (re-download + recompile); acceptable. Optimize
residency only after one complete speech path exists.

## Receipts (phase-aware)

`inference-core` changes (non-breaking, optional field):

```ts
interface InferencePhaseReceipt {
  name: string          // 'generator' | 'decoder'
  backend: Backend
  loadMs?: number
  compileMs?: number
  inferenceMs?: number
  warnings?: string[]
}

// InferenceReceipt gains:
phases?: InferencePhaseReceipt[]
```

`InferenceReceiptOptions` gains the same optional `phases` field;
`createInferenceReceipt` passes it through.

## Cancellation

`signal.abort()` terminates the active phase worker:

- generation cancellation → terminate Generator
- decoding cancellation → terminate Decoder

Pipeline returns `CANCELLED` (existing `InferenceErrorCode`). WASM memory is
released with the worker — a stronger boundary than LiteRT abort semantics.

## Deferred (not designed now)

- PodQast batching (Generator worker processes multiple utterances before
  teardown; Decoder worker batch-decodes — do not build batching into
  `Qwen3TtsPipeline` now, but keep the generator able to serve multiple
  requests before teardown).
- Asset persistence caching (OPFS / Cache Storage). Separation to preserve:
  raw model assets persist; compiled models, runtime, KV cache, graph working
  memory stay ephemeral.
- Memory-lighter codec export.

## Scope boundaries

- No changes to `runtime-litert` (generic runtime; staged graph relationship is
  Qwen3-TTS-specific).
- No changes to `text-gen`, PodQast, or model conversion.
- Asset proxy (`litertWasmProxy`, `qwenModelProxy` in `examples/vite.config.ts`)
  stays as-is.

## File layout

```
packages/qwen3-tts/src/
  pipeline.ts                 (rewritten orchestrator)
  codec-frames.ts
  phases/
    generator.ts
    decoder.ts
  workers/
    generator.worker.ts
    decoder.worker.ts
packages/inference-core/src/
  types.ts                    (InferencePhaseReceipt + phases on InferenceReceipt)
  receipts.ts                 (pass phases through)
```

## Acceptance ladder

Stop at the first failure. Retain verification vocabulary: generator compile /
generator inference / runtime teardown / decoder compile / decoder inference /
output validation / audible playback.

1. Generator worker initializes
2. Talker + MTP compile
3. "Testing one two three." produces non-empty `CodecFrames`
4. Generator worker terminates
5. Decoder worker initializes
6. Codec compiles
7. `CodecFrames` decode into non-empty Float32 PCM
8. `checkAudioValid()` passes
9. `AudioInferenceResult` receipt appears (with `phases`)
10. User hears intelligible speech

## Verification record

Update `docs/superpowers/verification/2026-08-10-package-extraction.md` as
stages pass; record each gate with browser evidence.
