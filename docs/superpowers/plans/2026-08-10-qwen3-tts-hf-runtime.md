# Qwen3-TTS Hugging Face runtime proof Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Point the standalone Qwen3-TTS example at the official Hugging Face
LiteRT model repository and verify the first available browser runtime stage
without conflating model assets with LiteRT WASM assets.

**Architecture:** Keep `createLiteRtRuntime()` unchanged and let it load its
existing LiteRT WASM CDN default. Configure the example's `AssetResolver` with
the same-origin `/models/qwen3-tts/` namespace and add a narrow Vite middleware
that follows Hugging Face redirects server-side and streams the final response.
Update the Qwen manifest with the repository's exact paths and byte sizes.
Browser verification records each stage independently and updates the existing
verification record only with observed results.

**Tech Stack:** TypeScript, pnpm workspaces, LiteRT JavaScript runtime,
Vite, Vitest, browser automation, and Hugging Face model hosting.

## Global Constraints

- Use `litert-community/Qwen3-TTS-12Hz-0.6B-Base` as the canonical model
  asset source.
- Keep LiteRT runtime assets on the existing CDN default.
- Do not vendor the 2+ GB model assets into the repository.
- Proxy only `/models/qwen3-tts/` and reject traversal paths.
- Follow Hugging Face redirects server-side and preserve range metadata.
- Do not claim model loading, compilation, inference, validated audio, or
  audible playback without exercising that stage in a browser.
- Keep `text-gen` frozen and do not modify PodQast in this cycle.

---

### Task 1: Align the Qwen manifest with the official model repository

**Files:**
- Modify: `packages/qwen3-tts/src/manifest.ts`
- Test: `packages/qwen3-tts/src/manifest.test.ts`
- Update expectations: `packages/qwen3-tts/src/pipeline.test.ts`,
  `packages/qwen3-tts/src/receipt.test.ts`,
  `examples/minimal-qwen3-tts/extraction.test.ts`

**Interfaces:**
- Consumes: `Qwen3TtsVariant`, `createQwen3TtsManifest()`, and the existing
  `ModelAsset` contract.
- Produces: manifests whose asset paths match the Hugging Face repository and
  whose required byte totals are `3_415_668_132` for `fp32` and
  `1_887_776_836` for `int4`.

- [ ] **Step 1: Add a failing metadata test**

Add assertions that the default manifest uses the official model identity,
contains the exact repository paths, and calculates the FP32 total:

```ts
import { qwen3TtsManifest, qwen3TtsVariants } from './manifest'

it('matches the official Qwen3-TTS LiteRT repository', () => {
  expect(qwen3TtsManifest.modelId).toBe('qwen3-tts-12hz-0.6b-base')
  expect(qwen3TtsManifest.assets.find(asset => asset.id === 'tokenizer')?.bytes)
    .toBe(11_424_262)
  expect(qwen3TtsManifest.assets.find(asset => asset.id === 'talker')?.path)
    .toBe('talker_fp32.tflite')
  expect(qwen3TtsManifest.memory.downloadBytes).toBe(3_415_668_132)
})
```

Add this test to `packages/qwen3-tts/src/manifest.test.ts` without removing
the existing variant filename test.

Run:

```bash
pnpm --filter @litert-playground/qwen3-tts test --run
```

Expected: the new test fails because the current manifest uses stale identity
and byte metadata.

- [ ] **Step 2: Update exact repository metadata**

Use these repository file sizes in `assetsFor()`:

```ts
tokenizer: 11_424_262
talker_fp32.tflite: 1_783_890_064
talker_int4.tflite: 255_998_768
mtp_fp32.tflite: 440_526_692
codec_decoder_fp32.tflite: 456_820_324
tables/codec_embedding_fp32.npy: 12_583_040
tables/mtp_embeddings_fp16.npy: 62_914_688
tables/text_embedding_fp16.npy: 622_329_984
tables/text_projection_fp32.npz: 25_179_078
voices/demo_speaker.npy: 4_224
```

Set `modelId` to `qwen3-tts-12hz-0.6b-base` and retain the existing variant
selection. The INT4 variant continues to use `mtp_fp32.tflite` and
`codec_decoder_fp32.tflite` because those are the available auxiliary graphs.

- [ ] **Step 3: Run the package tests**

Run:

```bash
pnpm --filter @litert-playground/qwen3-tts test --run
pnpm --filter @litert-playground/qwen3-tts typecheck
```

Expected: all Qwen tests pass and the package type-checks.

- [ ] **Step 4: Commit the manifest update**

```bash
git add packages/qwen3-tts/src/manifest.ts packages/qwen3-tts/src/manifest.test.ts \
  packages/qwen3-tts/src/pipeline.test.ts packages/qwen3-tts/src/receipt.test.ts \
  examples/minimal-qwen3-tts/extraction.test.ts
git commit -m "fix: align Qwen TTS manifest with Hugging Face"
```

### Task 2: Separate model and runtime asset bases in the standalone example

**Files:**
- Modify: `examples/minimal-qwen3-tts/main.tsx:8-30`
- Modify: `examples/vite.config.ts`
- Test: `examples/minimal-qwen3-tts/extraction.test.ts`

**Interfaces:**
- Consumes: `createHttpAssetResolver()`, `createCachingAssetResolver()`, and
  `createLiteRtRuntime()`.
- Produces: a browser consumer that resolves model files through the
  same-origin proxy and leaves LiteRT WASM resolution on its default CDN.

- [ ] **Step 1: Add extraction assertions for the model proxy boundary**

Extend the example extraction test to assert that the source uses the local
model namespace and does not pass it as `assetBase` to
`createLiteRtRuntime()`. Add these assertions after the existing resolver
assertions:

```ts
expect(source).toContain("const modelBase = '/models/qwen3-tts/'")
expect(source).toContain('createLiteRtRuntime({ assets })')
expect(source).not.toContain('assetBase: modelBase')
```

- [ ] **Step 2: Configure the same-origin model resolver and proxy**

```ts
const modelBase = '/models/qwen3-tts/'
```

Keep `createHttpAssetResolver(modelBase)` and call the runtime without
`assetBase`. Add `qwenModelProxy()` to `examples/vite.config.ts` with these
requirements:

- Match only `/models/qwen3-tts/`.
- Rewrite to the official Hugging Face `resolve/main/` URL.
- Use server-side `fetch()` so the 302 to Xet is followed before responding.
- Forward `Range`, `If-Range`, `If-None-Match`, and `If-Modified-Since`.
- Stream the response body with `Readable.fromWeb()`.
- Reject empty or traversal paths with status `400`.

The example code remains:

```ts
const assets = createCachingAssetResolver(createHttpAssetResolver(modelBase))
const context = await createLiteRtRuntime({ assets })
```

This makes model requests same-origin while the proxy uses Hugging Face and the
runtime continues to use the default LiteRT CDN base.

- [ ] **Step 3: Run example tests and typecheck**

Run:

```bash
pnpm --filter @litert-playground/example-qwen3-tts test --run
pnpm --filter @litert-playground/example-qwen3-tts typecheck
```

Expected: extraction tests and typecheck pass. A local proxy request to
`/models/qwen3-tts/tokenizer.json` returns HTTP 200 with the official byte
length, and a range request to `talker_int4.tflite` returns HTTP 206 with
`Content-Range`.

- [ ] **Step 4: Commit the example deployment change**

```bash
git add examples/minimal-qwen3-tts/main.tsx examples/minimal-qwen3-tts/extraction.test.ts
git commit -m "feat: use Hugging Face assets in Qwen example"
```

### Task 3: Perform staged browser verification

**Files:**
- Modify: `docs/superpowers/verification/2026-08-10-package-extraction.md`

**Interfaces:**
- Consumes: the built standalone Qwen example and its browser-visible status,
  error, progress, receipt, and audio behavior.
- Produces: an evidence-backed verification table with only exercised stages
  marked as passed.

- [ ] **Step 1: Build and start the standalone example**

Run:

```bash
pnpm --filter @litert-playground/example-qwen3-tts build
pnpm --filter @litert-playground/example-qwen3-tts dev --host 0.0.0.0
```

Open the example in a new browser tab. Confirm that the first visible failure,
if any, identifies a concrete URL, CORS response, WASM initialization error,
graph compilation error, or inference error.

- [ ] **Step 2: Record each exercised stage**

Capture the browser status and receipt for these stages independently:

1. Hugging Face asset request and response.
2. LiteRT WASM runtime initialization.
3. Talker, MTP, and codec graph compilation.
4. Qwen text-to-speech inference.
5. `AudioInferenceResult` validation and receipt output.
6. Audible playback.

If a stage fails, record the exact failure and leave later stages unverified.

- [ ] **Step 3: Update the structured verification record**

Change only the Qwen column in the runtime-gates table of
`docs/superpowers/verification/2026-08-10-package-extraction.md`. Preserve
`untested` for any stage not exercised successfully and include the observed
failure or receipt evidence below the table.

- [ ] **Step 4: Run static verification**

Run:

```bash
pnpm verify
```

Expected: the complete workspace static verification passes regardless of
whether browser model execution reaches inference.

- [ ] **Step 5: Commit the evidence record**

```bash
git add docs/superpowers/verification/2026-08-10-package-extraction.md
git commit -m "docs: record Qwen runtime verification"
```
