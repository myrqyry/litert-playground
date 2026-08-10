# Model packages implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract canonical Kokoro and shape-adaptive Qwen3-TTS packages with small public APIs.

**Architecture:** Both model packages depend on `inference-core` and receive a
`RuntimeContext`. Qwen variant metadata owns filenames and quantization while
shared generation code discovers compiled tensor layouts at load time.

**Tech Stack:** TypeScript, Vitest, `@litertjs/core`, `kokoro-js`.

## Global Constraints

- Use `@litert-playground/kokoro` and `@litert-playground/qwen3-tts`.
- Do not import PodQast source or depend on playground UI.
- Preserve normalized audio results, validation, cancellation, and receipts.
- Support only known FP32 and `talker_int4.tflite` variants.
- Do not materialize the full Qwen text embedding table as FP32.

---

### Task 1: Extract Kokoro package

**Files:**
- Create: `packages/kokoro/src/manifest.ts`
- Create: `packages/kokoro/src/pipeline.ts`
- Create: `packages/kokoro/src/types.ts`
- Create: `packages/kokoro/src/index.ts`
- Create: `packages/kokoro/src/pipeline.test.ts`
- Create: `packages/kokoro/package.json`
- Remove: `src/adapters/kokoro/manifest.ts`
- Remove: `src/adapters/kokoro/pipeline.ts`
- Remove: `src/adapters/kokoro/pipeline.test.ts`

**Interfaces:**
- Produces `KokoroPipeline`, manifest, input/config types, and package-level
  exports only.

- [ ] Update imports to core package contracts and shared validation/receipts.
- [ ] Preserve cancellation and normalized `AudioInferenceResult` behavior.
- [ ] Run Kokoro tests through the package boundary.
- [ ] Commit: `feat: extract Kokoro package`.

### Task 2: Add declarative Qwen variants

**Files:**
- Modify: `packages/qwen3-tts/src/manifest.ts`
- Create: `packages/qwen3-tts/src/variants.test.ts`
- Create: `packages/qwen3-tts/package.json`

**Interfaces:**
- Produces `Qwen3TtsVariant` with `id`, `talker`, `mtp`, `codec`,
  `quantization`, and backend support metadata.

- [ ] Define only the known FP32 and `talker_int4.tflite` variants.
- [ ] Make the manifest's required assets derive from the selected variant.
- [ ] Test that filenames are metadata and that unsupported variants cannot be
  selected.
- [ ] Commit: `feat: declare Qwen model variants`.

### Task 3: Reconcile Qwen shape discovery

**Files:**
- Move: `src/adapters/qwen3-tts/*` to `packages/qwen3-tts/src/`
- Modify: `packages/qwen3-tts/src/pipeline.ts`
- Modify: `packages/qwen3-tts/src/talker.ts`
- Modify: `packages/qwen3-tts/src/mtp.ts`
- Modify: `packages/qwen3-tts/src/codec.ts`
- Modify: `packages/qwen3-tts/src/text-embedding.ts`
- Create: `packages/qwen3-tts/src/shape-discovery.ts`
- Create: `packages/qwen3-tts/src/shape-discovery.test.ts`

**Interfaces:**
- Produces internal discovery helpers for compiled model input details and a
  package-level `Qwen3TtsPipeline` implementing the shared `Pipeline` contract.

- [ ] Compare each current playground helper with the corresponding PodQast
  implementation before changing behavior.
- [ ] Add failing tests for dynamic KV names/shapes, cache length, mask length,
  MTP cache/KV shape, and codec chunk discovery.
- [ ] Implement discovery from `CompiledModel.getInputDetails()` and use the
  discovered values in talker, MTP, and codec calls.
- [ ] Add `talker_int4.tflite` selection through variant metadata, not filename
  literals in generation logic.
- [ ] Run focused discovery tests before integration tests.
- [ ] Commit: `feat: make Qwen runtime shape adaptive`.

### Task 4: Preserve FP16 text embeddings and receipts

**Files:**
- Modify: `packages/qwen3-tts/src/text-embedding.ts`
- Modify: `packages/qwen3-tts/src/pipeline.ts`
- Create: `packages/qwen3-tts/src/text-embedding.test.ts`
- Modify: `packages/qwen3-tts/src/receipt.test.ts`

**Interfaces:**
- Produces lazy row conversion from `Uint16Array` FP16 storage to FP32 output
  rows and automatic shared receipt creation.

- [ ] Write a test proving loading preserves FP16 storage and row conversion
  returns correct FP32 values.
- [ ] Load only requested rows during inference.
- [ ] Keep cancellation checks in text generation, MTP, and codec loops.
- [ ] Use shared receipt infrastructure and structured verification fields.
- [ ] Run all Qwen tests and commit: `perf: keep Qwen text embeddings in FP16`.

### Task 5: Migrate package consumers

**Files:**
- Modify: `apps/playground/src/components/Qwen3TtsPanel.tsx`
- Modify: `apps/playground/src/adapters/registry.ts`
- Modify: `examples/minimal-qwen3-tts/main.tsx`
- Modify: `examples/minimal-qwen3-tts/extraction.test.ts`
- Modify: `apps/playground/package.json`

**Interfaces:**
- Consumers use only `@litert-playground/qwen3-tts` and
  `@litert-playground/runtime-litert` public exports.

- [ ] Remove all deep relative imports into old Qwen/runtime files.
- [ ] Run Qwen extraction tests and the playground build.
- [ ] Commit: `refactor: consume Qwen package APIs`.

### Phase gate

- [ ] Run clean frozen install.
- [ ] Run Kokoro and Qwen package tests.
- [ ] Run workspace type-check and production build.
- [ ] Verify no model package imports `apps/playground` or PodQast.
