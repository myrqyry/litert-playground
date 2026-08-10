# Monorepo inference packages design

## Goal

Turn the reusable LiteRT inference code in `litert-playground` into canonical
pnpm workspace packages that PodQast and other applications can consume through
public package entrypoints without copying source files.

This migration stops before text-generation packages. It preserves the current
playground behavior and does not redesign the playground UI.

## Public packages

The workspace uses the `@litert-playground/*` scope:

- `@litert-playground/inference-core`
- `@litert-playground/runtime-litert`
- `@litert-playground/kokoro`
- `@litert-playground/qwen3-tts`

The repository also contains `apps/playground` and two standalone examples:

```text
apps/playground/
packages/inference-core/
packages/runtime-litert/
packages/kokoro/
packages/qwen3-tts/
examples/minimal-qwen3-tts/
examples/minimal-kokoro/
```

The examples are browser consumers of public package entrypoints. They do not
import source files from `apps/playground` or package internals.

## Dependency direction

Dependencies point toward generic infrastructure:

```text
inference-core
    ^
runtime-litert
    ^
kokoro / qwen3-tts
    ^
apps/playground and examples
```

`inference-core` is model-agnostic. It owns pipeline contracts, runtime
context types, assets, normalized inference results, errors, receipts,
verification state, asset resolution, progress types, and generic audio
validation. It does not depend on model packages, LiteRT, Kokoro, Qwen, or
PodQast concepts such as `PodcastTts`.

`runtime-litert` owns `@litertjs/core`, LiteRT initialization, model
compilation, backend probing and selection, and runtime-generic NPY loading.
Model packages
receive a `RuntimeContext`; they never call `loadLiteRt()` directly.

The model packages depend on core and expose small public entrypoints. They do
not depend on the playground application.

## Core API

The core package exposes the existing generic contracts without duplicate
definitions:

- `Pipeline<I, O, C>` and `PipelineStatus`
- `RuntimeContext`, `AssetResolver`, `ModelAsset`, and `ModelManifest`
- `Capability` and `Backend`
- `AudioInferenceResult`, `TextInferenceResult`, `ImageInferenceResult`,
  `EmbeddingInferenceResult`, and `TensorInferenceResult`
- `InferenceError`
- `InferenceReceipt` and structured verification state
- generic audio validation helpers

Asset access uses one options shape for buffered and streamed downloads:

```ts
resolve(asset, options?: {
  signal?: AbortSignal
  onProgress?: (progress: AssetProgress) => void
}): Promise<ArrayBuffer>

stream(asset, options?: {
  signal?: AbortSignal
  onProgress?: (progress: AssetProgress) => void
}): Promise<ReadableStream<Uint8Array>>
```

Failed cached promises are evicted, cancellation reaches `fetch`, streamed and
buffered errors identify the asset, and large downloads report progress. The
resolver remains model-agnostic and does not add OPFS behavior.

## LiteRT runtime API

The runtime package exposes runtime construction through:

```ts
createLiteRtRuntime(options: {
  backend?: 'auto' | 'webgpu' | 'wasm' | 'webnn'
  assets: AssetResolver
  signal?: AbortSignal
}): Promise<RuntimeContext>
```

Backend selection probes a usable adapter/device rather than treating
`navigator.gpu` as sufficient. It respects explicit preferences, intersects
available capabilities with model support at compilation time, and falls back
only where policy permits. The observable selected backend is reflected in the
returned context and receipts.

Runtime errors distinguish an unavailable backend from a model compile failure.
Receipt construction is shared infrastructure and includes model ID, pipeline
version, backend, load time, compile time where available, inference time,
input/output summaries, warnings, timestamp, and safe browser/environment
information.

## Kokoro package

`@litert-playground/kokoro` exposes `KokoroPipeline`, its manifest, and its
public input/config types. It depends on `@litert-playground/inference-core`
and `kokoro-js`, returns the normalized audio result, validates output, checks
cancellation, and emits shared receipt semantics.

It has no dependency on playground UI or PodQast.

## Qwen package

`@litert-playground/qwen3-tts` exposes only:

```ts
Qwen3TtsPipeline
qwen3TtsManifest
QwenTtsInput
QwenTtsConfig
Qwen3TtsVariant
```

Model filenames and quantization metadata live in declarative variant metadata.
Only variants supported by known repository assets are included, including the
existing FP32 talker and the known `talker_int4.tflite` variant.

The implementation reconciles the current playground version with PodQast's
newer implementation rather than replacing either blindly. It must preserve:

- compiled-model input introspection
- dynamic KV names and shapes
- cache and mask length discovery
- MTP cache/KV discovery
- codec chunk discovery
- cancellation during generation
- normalized audio output, validation, and receipts

The text embedding table remains FP16 in `Uint16Array` storage. Only the rows
needed for an inference are converted to FP32. The pipeline must not assume one
fixed FP32 tensor layout.

## Applications and examples

The existing app moves to `apps/playground` without a UI redesign. Its manifest
depends on workspace packages, and model/runtime/core imports use public
entrypoints. Any temporary compatibility forwarding exists only during the
same migration phase and is removed before that phase closes.

The Qwen example keeps its current lifecycle and behavior while switching to
public packages. The Kokoro example is a small independent browser app that
accepts text, synthesizes audio, displays receipt warnings, sample rate, and
duration, and plays returned `Float32Array` audio through plain Web Audio.

`PodcastTts` remains app-oriented and outside the generic packages. PodQast
continues to own speaker mapping, episode orchestration, and playback
integration.

## Verification model

Structured verification distinguishes these independent states:

```ts
assets: 'pass' | 'fail' | 'untested'
compile: 'pass' | 'fail' | 'untested'
inference: 'pass' | 'fail' | 'untested'
output: 'pass' | 'fail' | 'untested'
```

The migration adds lightweight boundary tests that verify public imports,
forbid deep relative imports in examples, prevent package imports from
`apps/playground`, and ensure core has no model-package dependencies. Both
model packages must satisfy the shared audio pipeline contract.

Every phase requires a clean install, type-check, unit tests, and production
build. Browser and audio claims remain unverified unless a real browser run
produces and audibly verifies output. The final report separately records
workspace structure, package compilation, tests, playground package usage,
standalone example exercise, actual audio verification, and remaining PodQast
duplication.

## Execution phases

The work is split into three plans so each boundary can be reviewed and tested
independently:

1. **Workspace, core, runtime, and assets:** establish pnpm packages, move the
   generic contracts and resolver, move LiteRT runtime ownership, and migrate
   the playground imports.
2. **Kokoro and Qwen packages:** extract both model packages, reconcile
   PodQast's adaptive Qwen implementation, add declarative variants, and
   preserve focused tests.
3. **Examples and boundary verification:** add the standalone Kokoro example,
   migrate Qwen example imports, add package-boundary tests, and run structured
   browser verification.

The migration does not extract text generation or modify PodQast source.
