# Standalone Qwen3-TTS Browser App

## Goal

Make `examples/minimal-qwen3-tts` a real standalone Vite browser app that proves an external consumer can import the Qwen3-TTS adapter, construct a runtime, load real assets, synthesize audio, and play it without importing playground components, the registry, or `src/App`.

The milestone also closes the concrete runtime gaps identified in the architecture review:

- resolver retry behavior;
- resolver abort propagation;
- actual backend selection;
- manifest size accuracy;
- automatic inference receipts;
- standalone end-to-end verification.

This milestone does not introduce a monorepo, package split, second model consumer, giant base class, or new dependency.

## Runtime Design

The public data flow is:

```text
createHttpAssetResolver(baseUrl)
  -> createRuntimeContext(baseUrl, assets)
  -> createQwen3TtsPipeline()
  -> pipeline.load(context)
  -> pipeline.run(input)
  -> audio playback
```

`createRuntimeContext` remains responsible for constructing the LiteRT WASM runtime and its adapters. It must honor the supplied asset base URL instead of ignoring it.

Backend selection is runtime-owned and automatic:

1. Probe WebGPU support.
2. Configure the LiteRT WASM runtime to use WebGPU when available.
3. Report `backend: "webgpu"` when GPU execution is active.
4. Fall back to WASM CPU and report `backend: "wasm"` when WebGPU is unavailable or initialization fails.

The consumer does not implement capability detection or fallback policy. An optional explicit preference may exist for tests or debugging, but automatic selection is the default.

## Asset Resolution

`AssetResolver` operations accept an `AbortSignal` and propagate it to every underlying fetch, including voice assets used by Qwen3-TTS.

The caching resolver must remove a cache entry when its underlying promise rejects. Successful results remain cached. This allows transient network failures to retry without retaining a permanently rejected promise.

Aborted requests reject with `AbortError` and must not leave failed entries in the cache.

## Manifest and Receipts

Manifest aggregate download size is derived from the declared asset metadata rather than maintained as a conflicting hardcoded value. The baseline total includes required assets only; optional voice assets are excluded from that total and are accounted for separately when selected. The same rule is used consistently in displayed totals and tests.

`InferenceReceipt` is created automatically by the pipeline run path. A completed run includes at least:

- model identity and version;
- selected backend;
- elapsed inference time;
- input summary;
- output summary.

Receipt creation is part of the existing pipeline result path, not a separate telemetry system.

## Standalone App

The example app uses only the extracted public runtime and adapter APIs. It must:

- load the real Qwen3-TTS manifest assets from its configured base URL;
- show loading, ready, running, and error states;
- disable synthesis until loading succeeds;
- synthesize a short phrase;
- create playable audio and expose playback controls;
- display the selected backend, timing, and receipt information;
- show actionable errors for asset, runtime, and inference failures.

The existing extraction test remains a static boundary test. It must continue to verify that the example has no imports from playground-only application code. It is not a substitute for browser execution.

## Verification

Unit tests cover:

- abort signal propagation through asset resolution;
- rejected cache entry eviction and retry;
- derived manifest aggregate sizes;
- backend selection and WASM fallback;
- automatic inference receipt creation.

Repository verification runs `npm test` and `npm run build`.

Browser verification runs the standalone example against real assets and records the result in `docs/superpowers/verification/`, including the browser/runtime environment and whether WebGPU or WASM CPU fallback was used. A run that cannot access WebGPU or download the model must report that limitation clearly and must not be recorded as a successful end-to-end synthesis.

## Failure and Lifecycle Behavior

The pipeline lifecycle remains:

```text
idle -> loading -> ready -> running -> ready
                         running -> error
```

Load and inference failures enter `error` and remain visible to the example app. `dispose()` releases pipeline and runtime resources. WebGPU initialization failure is the one intentional fallback path; model and asset failures are not silently swallowed or retried forever.
