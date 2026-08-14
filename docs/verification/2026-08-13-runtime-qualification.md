# Runtime qualification record

The Playground qualification laboratory owns this record. It distinguishes
deterministic contracts from real browser evidence and does not claim a model
or backend is qualified without a recorded environment.

## Current production lane

Production uses `@litertjs/core` 2.5.3. The deterministic qualification
contracts are green at the current Playground revision. Real headless
Chromium WASM runs now cover the tiny baseline and module-worker case; the
EfficientDet and Qwen observations below remain separate from manifest
promotion.

The tiny baseline descriptor is immutable and points to the LiteRT.js core
`add_10x10.tflite` asset at commit
`16d8551be578965fe194e4d75f414f48c7b4e75a`. Its measured size is 708 bytes
and its SHA-256 is
`1317a76ceedc6e0a2b39c4ee2802f80b3b831b16ac96a99e48540113472aaee2`.

## Browser observations

The current browser observations were run with Chromium 151.0.7922.34 and
requested WASM. The tiny baseline passes with the immutable `add_10x10.tflite`
asset. The module-worker case fails with `Failed to execute 'importScripts' on
'WorkerGlobalScope': Module scripts don't support importScripts().` and matches
the expected `worker-load` limitation.

The real EfficientDet Lite0 descriptor is pinned to TF Hub version `1`, with
4,563,519 bytes and SHA-256
`2e04c53bfeac0ac2a30c057c7e2a777594ce39baaac35a92f74fb1e8c4fc4e0b`. WASM
reaches the output-materialization cleanup path and Chromium reports `Target
crashed`; this matches the limitation contract but needs a smaller upstream
error report before any manifest claim.

The standalone Omni MTP qualification is pinned to the supplied LiteRT-LM
Omni repository at commit
`791880469d874546d884a0e6cf68564a61c04ca9`, with 440,528,628 bytes and
SHA-256 `7e808fb554fdf443e70e5ccdd3fdccd3cd74cdec606d3375fa4c5877d4f46e0b`.
Its Chromium WASM compile and default-input execution both pass. The base
folded MTP graph and talker `prefill_32` candidate also pass separately.
These are component qualifications, not evidence that the composed Qwen
browserMemory generator is qualified.

The composed `qwen-browsermemory-generator` case uses the base INT4 Talker and
the Omni FP32 MTP together through the real `GeneratorPhase` with
`maxFrames: 1`. In Chromium 151 WASM, both graphs compile, then Talker
prefill fails during tensor materialization with
`litert_tensor_buffer.h:101`. The browser receipt records `embeddings` as
`float32 [1,32,1024]`, `input_pos` as `int32 [32]`, `mask` as
`float32 [1,1,32,1024]`, and the real `kv_cache_k_0` through
`kv_cache_v_27` tensors as `float32` KV shapes with element counts. No tensor
contents cross the browser boundary. This moves the known limitation to the
composed Talker prefill seam. The result is classified as `limited` with
limitation `resource-exhausted`; the standalone Omni MTP result remains a pass.

The headless environment reports WebGPU as unsupported, so WebGPU cases return
`unsupported` with `BACKEND_UNAVAILABLE` instead of attempting model execution.

## Evidence rule

Manifest verification metadata can be updated only when a result records the
exact model revision, Playground revision, LiteRT.js version, browser, and
backend. A reproduced limitation maps to `limited`. A passing result maps to
`qualified` only when the environment and model revision are present.

## Upstream issue inputs

The minimal cases provide issue inputs for dynamic output tensor
materialization, Qwen/XNNPACK runtime creation, and module-worker loader
compatibility. Upstream reports must include the LiteRT.js version, browser,
operating system, model URL and revision, requested/resolved backend, and the
smallest reproduction.
