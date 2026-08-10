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
| Assets | untested | proxy transport verified; model loading untested |
| Compile | untested | untested |
| Inference | untested | untested |
| Output | untested | untested |
| Audible audio | untested | untested |

No model or audio result is marked as working from build output alone.

The Qwen proxy was exercised locally on August 10, 2026. The tokenizer
returned HTTP 200 with the official 11,424,262-byte length, and an INT4 talker
range request returned HTTP 206 with `Content-Range: bytes 0-15/255998768`.
The browser runtime, graph compilation, inference, validated audio, and
audible playback were not exercised because browser automation was unavailable
in this environment.
