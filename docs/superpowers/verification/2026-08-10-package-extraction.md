# Package extraction verification

This record covers the workspace/package boundary, not real model execution.

## Automated gates

- Workspace install: pass with `pnpm install --frozen-lockfile`.
- Package tests: pass across core, runtime, Kokoro, Qwen, and examples.
- Type-check: pass across all workspace projects.
- Playground production build: pass.
- Standalone example builds: pass.
- Package-boundary tests: pass.

## Runtime gates

The repository does not contain the model assets required for browser inference.

| Boundary | Kokoro | Qwen3-TTS |
| --- | --- | --- |
| Assets | untested | partial pass; speaker embedding untested |
| Compile | untested | Talker pass; MTP pass; codec pass (staged worker residency) |
| Inference | untested | untested |
| Output | untested | untested |
| Audible audio | untested | untested |

No model or audio result is marked as working from build output alone.

The Qwen proxy was exercised locally on August 10, 2026. The tokenizer
returned HTTP 200 with the official 11,424,262-byte length, and an INT4 talker
range request returned HTTP 206 with `Content-Range: bytes 0-15/255998768`.

The standalone run used Qwen INT4 talker plus FP32 auxiliary graphs from
Hugging Face revision
`66855540b3b34679f06c3ff07859603fc9514c66` in Chrome 151.0.7922.34 on Linux.
LiteRT initialization passed in 186 ms with a warm persistent browser profile;
WebGPU had no adapter, so automatic selection used WASM/XNNPACK. Tokenizer,
codec embeddings, MTP FP16 embeddings, the 622 MB lazy FP16 text table, and
the ZIP64 text projection parsed successfully. Talker and MTP compiled
successfully. The browser stalled at progress `loading: 6/7` while reaching the
codec boundary; the process was aborted after the bounded wait. Inference,
audio validation, and audible playback remain untested.

### Codec compile blocker

Isolated instrumentation of the codec graph established the following:

- Codec compiles alone: pass. A fresh page that fetched
  `codec_decoder_fp32.tflite` (456,820,324 bytes) and compiled only it
  completed in 635 ms after a 229,667 ms fetch.
- Codec stalls neither on fetch nor on compile by itself. Fetch always
  completes (577-230 s depending on run); compilation in isolation succeeds.
- Codec compile crashes the renderer whenever `talker_int4.tflite` and
  `mtp_fp32.tflite` are already compiled in the same page. This reproduces both
  in the isolated three-graph diagnostic (no embedding tables loaded) and in
  the real pipeline load, so the pressure comes from the three compiled graphs'
  accumulated memory, not from the embedding tables.
- Hard evidence: the live diagnostic logged `codec compile start heap=`
  `{"jsHeapUsedMB":497,"jsHeapTotalMB":497}` immediately before the renderer
  crash. The JS heap is fully exhausted at the moment codec compilation begins.
- Reordering load so all three graphs compile before any table loads does not
  fix the crash; the graphs still accumulate in one page.
- Raising the headless renderer V8 old-space budget does not help. The same
  full three-graph diagnostic run with `--js-flags=--max-old-space-size=4096`
  (all else identical) crashed at the exact same point with the exact same
  heap values: `codec compile start heap={"jsHeapUsedMB":497,
  "jsHeapTotalMB":497}` followed by a renderer process crash. The failure is a
  renderer process crash, not a JavaScript heap out of memory error, so the
  `--max-old-space-size` flag does not affect it. This falsifies the
  "artificial headless heap cap" hypothesis for this stage.

Side-by-side comparison:

| Launch condition | talker | mtp | codec fetch | codec compile | heap before codec |
|---|---|---|---|---|---|
| Default headless heap | pass | pass | pass | renderer crash | 497/497 MB |
| `--max-old-space-size=4096` | pass | pass | pass | renderer crash | 497/497 MB |

Blocker state: `codec_decoder_fp32.tflite` cannot compile in the same WASM
page as already-compiled INT4 talker and FP32 MTP graphs; the renderer
process crashes at 497/497 MB JS heap, and this is not tunable via
`--max-old-space-size`. Reaching the next stage (inference) requires either a
memory-lighter codec path or a way to free compiled-graph memory between
stages.

### Staged worker residency: blocker resolved

A two-worker experiment proved the memory is reclaimable by process
isolation. Two disposable classic Web Workers (served at
`/litert-wasm/residency-worker.js` via a Vite proxy to
`@litertjs/core@2.5.3`):

- Worker A initialized LiteRT and compiled `talker_int4.tflite` (fetch
  15,841 ms, compile 3,982 ms) and `mtp_fp32.tflite` (fetch 42,877 ms,
  compile 492 ms), then was fully terminated.
- A fresh Worker B initialized LiteRT and compiled only
  `codec_decoder_fp32.tflite` (fetch 64,347 ms, compile 614 ms): **pass**.

Both workers reported ok=true with no crashes or page errors, and JS heap
pressure stayed at 0/0 MB (each worker is a separate renderer process, so the
accumulated three-graph memory never coexists). This proves terminating the
Talker/MTP worker releases enough WASM residency for codec compilation, and
gives the pipeline a route forward: compile/run the generator graphs in one
worker, tear it down, then compile the codec in a second worker. Inference,
audio validation, and audible playback remain untested.
