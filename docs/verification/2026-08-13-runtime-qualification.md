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

The real Qwen browserMemory talker descriptor is pinned to Hugging Face commit
`0eb3b8a4714972b065c160faec6a12158caa9dc0`, with 255,998,768 bytes and SHA-256
`e03df54e73ed1f88b2ae6d47bbf82dd64ea90a3620d753a0f3c8d6a8d60848db`. Its
`prefill_32` run passes in this Chromium WASM environment, so it intentionally
mismatches the known-limitation expectation. This is evidence against the
current talker-only repro, not evidence that Qwen browserMemory is qualified.

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
