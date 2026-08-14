# Runtime qualification record

The Playground qualification laboratory owns this record. It distinguishes
deterministic contracts from real browser evidence and does not claim a model
or backend is qualified without a recorded environment.

## Current production lane

Production uses `@litertjs/core` 2.5.3. The deterministic qualification
contracts are green at the current Playground revision, but browser evidence
for the EfficientDet, Qwen, and module-worker reproductions requires an
explicit `pnpm qualify` run on a named browser and device.

The tiny baseline descriptor is immutable and points to the LiteRT.js core
`add_10x10.tflite` asset at commit
`16d8551be578965fe194e4d75f414f48c7b4e75a`. Its measured size is 708 bytes
and its SHA-256 is
`1317a76ceedc6e0a2b39c4ee2802f80b3b831b16ac96a99e48540113472aaee2`.

## Known limitation contracts

The lab keeps these cases as expected limitations until browser evidence
confirms the exact error and stage:

- EfficientDet dynamic output materialization on WASM.
- Qwen browserMemory XNNPACK prefill.
- Qwen LiteRT loader behavior from a module worker.

The recorded verification notes contain a prior Qwen tensor-buffer failure at
`tensor_buffer.h:101`; the qualification case still requires a current
browser-run normalized error before this becomes manifest evidence.

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
