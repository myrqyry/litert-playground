# Qwen browserMemory generator qualification

## Goal

Qualify the real `browserMemory` Qwen generator topology in Chromium WASM
without duplicating its inference logic. The qualification must call the
existing `GeneratorPhase.load()` and `GeneratorPhase.generate()` methods
inside the browser page, using the real browser-side asset and LiteRT runtime
implementations.

## Scope

The first qualification run covers `GeneratorPhase.generate()` with
`maxFrames: 1`. This executes prompt construction, Talker prefill, initial
Talker decode, MTP prediction, conditioned embedding construction, KV/state
update, and the next Talker decode. Codec decoding and full audio synthesis
remain outside this first case.

The existing standalone Omni MTP qualification becomes a passing component
qualification. The composed sequence gets a separate case named
`qwen-browsermemory-generator` and uses an explicit model description for the
Talker and MTP revisions it loads.

## Browser boundary

The qualification page creates a real `RuntimeContext` with:

- `createLiteRtRuntime()` and the requested WASM backend;
- an `AssetResolver` that fetches immutable Qwen asset URLs in the browser;
- browser-side size and SHA-256 verification for every model and table asset;
- browser-side parsing of tokenizer, NPY, and NPZ assets.

Playwright receives only environment metadata, stage receipts, tensor metadata,
timings, and the final observation. Model bytes and large table contents never
cross the Playwright boundary.

## Trace contract

`GeneratorPhaseOptions` gains an optional `onTrace` callback. The callback
observes the existing generator values and does not alter tensor creation or
execution. Trace receipts cover `talker-compile`, `talker-prefill`,
`talker-output-read`, `mtp-input-build`, `mtp-compile`, `mtp-run`,
`mtp-output-read`, and `state-update`. Events may carry `phase: 'start'` or
`phase: 'end'` when timing a boundary.

Each event may include a frame, duration, and tensor receipts. A tensor receipt
contains only its name, dtype, shape, and element count. It never contains
tensor values.

## Error and cleanup behavior

The qualification records the first failing stage and message, while preserving
the existing exception behavior. `GeneratorPhase.dispose()` runs in a `finally`
block after each run so the phase releases its model references and browser-side
runtime cleanup follows the existing ownership semantics.

## Verification

The implementation adds contract tests for the case identity, exact asset
revisions, expected pass status, and trace metadata shape. It runs the focused
qualification tests, the real browser qualification with `maxFrames: 1`, and
the repository `pnpm verify` command. If the one-frame sequence passes, later
frame counts remain follow-up qualifications rather than being inferred from
the first result.
