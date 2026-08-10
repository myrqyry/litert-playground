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
| Assets | untested | untested |
| Compile | untested | untested |
| Inference | untested | untested |
| Output | untested | untested |
| Audible audio | untested | untested |

No model or audio result is marked as working from build output alone.
