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
| Compile | untested | Talker pass; MTP pass; codec untested |
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
