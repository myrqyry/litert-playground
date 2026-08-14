# Qwen TTS browser-WASM status

This note closes the current Qwen TTS browser-WASM investigation. The failure
is a measured browser residency ceiling, not evidence of incorrect Qwen tensor
construction or a standalone MTP incompatibility.

## Evidence

The real `qwen-browsermemory-generator` qualification runs the existing
`GeneratorPhase` with the base INT4 Talker and Omni FP32 MTP in Chromium 151
WASM.

- LiteRT WASM runtime loads: pass.
- Talker compilation: pass.
- MTP compilation: pass.
- Graph setup and real prompt construction: pass.
- Failure stage: real Talker `prefill` tensor allocation.
- Failure: `litert_tensor_buffer.h:101` during managed tensor materialization.
- Standalone Omni MTP compile and execution: pass.
- Evidence boundary: tensor names, dtypes, shapes, and element counts only;
  tensor contents never leave the browser.

The same prefill allocation failure remains after prior experiments with INT8
MTP, prompt-worker teardown, separate generator workers, and reduced KV
exports. These experiments rule out the current lifetime-management hypotheses.

## Decision

The browser Qwen TTS route is **blocked by browser-WASM residency**. Do not
resume buffer-lifetime or cleanup-tweak work without new evidence that changes
the measured memory envelope. This limitation must not block PodQast or the
larger runtime architecture; browser UI and control can remain separate from
the inference backend.

## Next valid experiments

Prioritize the non-browser runtime path, with native LiteRT on Android as the
cleanest product target and local or server inference as alternatives. Compare
native Qwen3-TTS through `audio.cpp` with the LiteRT/native path before making
consumer integration decisions.

Treat a smaller or mixed-precision browser export as experimental. It must
demonstrate all of the following before the browser route can be reconsidered:

1. Talker prefill allocation succeeds.
2. Full short synthesis succeeds.
3. Repeated synthesis succeeds.
4. Output quality remains acceptable.
5. Measured memory remains below a documented browser ceiling.
