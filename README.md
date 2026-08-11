# litert-playground

A pnpm monorepo for exploring LiteRT (the successor to TensorFlow Lite) on-device
inference in the browser, with a focus on text-to-speech packages and the
runtime plumbing around them.

## Workspace layout

| Path | Package | Purpose |
|------|---------|---------|
| `apps/playground` | `playground` | React + Vite + Tailwind app for interactive model demos |
| `packages/inference-core` | `@litert-playground/inference-core` | Shared contracts: model manifests, asset resolvers, receipts, audio validation, error types |
| `packages/runtime-litert` | `@litert-playground/runtime-litert` | `createLiteRtRuntime` over `@litertjs/core`; capability probing and backend selection |
| `packages/kokoro` | `@litert-playground/kokoro` | `KokoroPipeline` — Kokoro TTS via `kokoro-js` (q8, wasm) |
| `packages/qwen3-tts` | `@litert-playground/qwen3-tts` | `Qwen3TtsPipeline` — phased worker architecture over three LiteRT graphs (talker, MTP, codec) |
| `packages/text-gen` | `@litert-playground/text-gen` | Text generation pipelines (frozen) |
| `examples/minimal-kokoro` | `@litert-playground/example-kokoro` | Standalone Kokoro browser demo |
| `examples/minimal-qwen3-tts` | `@litert-playground/example-qwen3-tts` | Standalone Qwen3-TTS browser demo |

Workspace globs cover `apps/*`, `packages/*`, and `examples/*`. Native build
scripts for `onnxruntime-node`, `protobufjs`, and `sharp` are disabled.

## Commands

| Command | Runs |
|---------|------|
| `pnpm install` | Install workspace dependencies |
| `pnpm dev` | Start the playground dev server (`apps/playground`) |
| `pnpm build` | Build all workspace projects (`pnpm -r --if-present build`) |
| `pnpm preview` | Preview the playground production build |
| `pnpm test` | Run tests in all workspace projects (`pnpm -r --if-present test`) |
| `pnpm test:boundaries` | Verify package dependency boundaries |
| `pnpm test:watch` | Watch-mode tests for the playground |
| `pnpm typecheck` | Type-check all workspace projects (`pnpm -r --if-present typecheck`) |
| `pnpm verify` | Authoritative static gate: typecheck + test + boundaries + build |

`pnpm verify` is the gate for local development and CI.

## TTS capability split

| Capability | Backend | Status |
|------------|---------|--------|
| **Kokoro** | Browser WASM (`kokoro-js`, q8) | Verified audible browser proof — synthesizes "Testing one two three." to 24 kHz mono Float32 audio, validated and exported as WAV |
| **Qwen3-TTS** | Native / local runtime (Android LiteRT, desktop/server, future WebGPU) | Classified native/local-runtime capability, not browser-WASM |

Qwen3-TTS runs as three host-orchestrated LiteRT graphs (talker, MTP, codec)
using a phased worker architecture with hard worker boundaries. Its combined
runtime residency exceeds the practical browser WASM/JS memory budget for a
single prefill call, independent of headed/headless mode, MTP quantization,
prompt residency, codec residency, and KV-cache capacity. The `browserMemory`
manifest variant (`mtp_folded_int8`) and a short-KV (256) talker export remain
as compatibility probes for future LiteRT.js improvements.

## Verification record

Real-model and audio results are recorded in
`docs/superpowers/verification/2026-08-10-package-extraction.md`, including the
Kokoro browser proof, the Qwen3-TTS runtime classification, and the model-proxy
exercises. No model or audio result is marked as working from build output
alone.
