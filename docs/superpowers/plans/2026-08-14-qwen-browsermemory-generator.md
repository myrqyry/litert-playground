# Qwen browserMemory generator qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Qualify the real browserMemory Qwen generator path by running the existing `GeneratorPhase` inside Chromium with browser-side assets, runtime, and forensic tensor receipts.

**Architecture:** Keep inference behavior in `packages/qwen3-tts/src/phases/generator.ts`. Add an optional observational trace callback and use a browser-page adapter that creates a real `RuntimeContext` backed by `createLiteRtRuntime()` and an immutable, verified asset resolver. The qualification uses a distinct `browserMemoryOmni` variant: the base-revision `talker_int4.tflite` plus Omni-revision `mtp_fp32.tflite`. It invokes `GeneratorPhase.load()` and `generate({ text: 'Testing one two three.' }, { temperature: 0, topK: 0, repetitionPenalty: 1, maxFrames: 1, voice: 'demo_speaker', language: 'english' })`, then returns metadata-only receipts through Playwright.

**Tech Stack:** TypeScript, Vitest, Vite, Playwright, `@litertjs/core` 2.5.3, `@litert-playground/runtime-litert`, `@litert-playground/qwen3-tts`.

## Global Constraints

- Keep large model and table bytes inside the browser page.
- Send only stage events, environment metadata, tensor metadata, timings, and final status through Playwright.
- Verify every immutable asset URL in the browser by byte count and SHA-256.
- Never record tensor contents.
- Reuse `GeneratorPhase` and its existing Talker/MTP ownership and state transitions.
- Keep the standalone Omni MTP component qualification separate from the composed generator qualification.
- Do not repin PodQast or Live Streamer.

---

### Task 1: Correct and graduate the standalone Omni qualification

**Files:**
- Rename: `tests/runtime-qualification/qwen-xnnpack-prefill/` to `tests/runtime-qualification/qwen-omni-mtp-standalone/`
- Modify: `tests/runtime-qualification/run-qualification.ts`
- Modify: renamed `case.ts`, `expected.ts`, `contract.test.ts`, `README.md`, and `fixtures/asset.json`
- Modify: `docs/verification/2026-08-13-runtime-qualification.md`

**Interfaces:**
- Produces case id `qwen-omni-mtp-standalone`.
- Produces `expected: { status: 'pass' }`.
- Records Omni revision `791880469d874546d884a0e6cf68564a61c04ca9` in both the case model and asset URL.

- [ ] **Step 1: Write failing contract assertions**

Update the contract test to assert the renamed id, Omni revision, Omni MTP URL, byte count `440528628`, SHA-256 `7e808fb554fdf443e70e5ccdd3fdccd3cd74cdec606d3375fa4c5877d4f46e0b`, and expected pass status.

- [ ] **Step 2: Run the contract test and verify it fails**

Run:

```bash
pnpm exec vitest run --config tests/runtime-qualification/vitest.config.ts tests/runtime-qualification/qwen-omni-mtp-standalone/contract.test.ts
```

Expected: FAIL because the old case path and expected limitation still exist.

- [ ] **Step 3: Rename the case and update metadata**

Rename the directory and change imports, case id, description, expected status, model revision, README, verification record, and case registration. Keep the browser run as compile plus default-input execution of the exact Omni MTP asset.

- [ ] **Step 4: Run focused tests**

Run the renamed contract test and the manifest consistency tests. Expected: all pass.

- [ ] **Step 5: Commit the standalone qualification**

```bash
git add tests/runtime-qualification/qwen-omni-mtp-standalone docs/verification/2026-08-13-runtime-qualification.md tests/runtime-qualification/run-qualification.ts
git commit -m "test: graduate Omni MTP qualification"
```

### Task 2: Add observational generator trace events

**Files:**
- Modify: `packages/qwen3-tts/src/phases/generator.ts`
- Modify: `packages/qwen3-tts/src/talker.ts`
- Modify: `packages/qwen3-tts/src/mtp.ts`
- Create or modify: `packages/qwen3-tts/src/generator-trace.test.ts`

**Interfaces:**
- Produces `GeneratorTraceEvent`, `GeneratorTraceTensor`, and `GeneratorTraceStage` types.
- `GeneratorPhaseOptions` accepts `onTrace?: (event: GeneratorTraceEvent) => void`.
- Trace tensors expose `name`, `dtype`, `shape`, and `elementCount` only.
- Talker and MTP expose optional trace callbacks for existing input/output tensors without changing execution APIs.
- `GeneratorTraceStage` includes exactly `talker-compile`, `talker-prefill`, `talker-output-read`, `mtp-input-build`, `mtp-compile`, `mtp-run`, `mtp-output-read`, and `state-update`.

- [ ] **Step 1: Write failing trace tests**

Add tests that construct the trace receipt helper with a typed tensor-like object and assert it returns only name, dtype, shape, and element count. Add a generator-phase test with mocked Talker/MTP classes asserting `onTrace` receives all eight required stages, with `phase: 'start' | 'end'` around executions where timing is meaningful.

- [ ] **Step 2: Run tests and verify the trace contract fails**

Run:

```bash
pnpm exec vitest run packages/qwen3-tts/src/generator-trace.test.ts
```

Expected: FAIL because the trace types and callback are absent.

- [ ] **Step 3: Implement metadata-only tracing**

Add a small tensor receipt helper that reads tensor type metadata and computes element count from shape. Thread optional callbacks through the existing Talker `prefill`/`decode` and MTP `predict` methods. Emit events around existing calls and state assignment. Do not allocate alternate inputs, copy values into receipts, or add qualification-specific branches.

- [ ] **Step 4: Run focused tests**

Run the trace test and existing Qwen package tests. Expected: all pass.

- [ ] **Step 5: Commit the trace seam**

```bash
git add packages/qwen3-tts/src/generator-trace.test.ts packages/qwen3-tts/src/phases/generator.ts packages/qwen3-tts/src/talker.ts packages/qwen3-tts/src/mtp.ts
git commit -m "feat: trace Qwen generator stages"
```

### Task 3: Expose a browser-side GeneratorPhase runner

**Files:**
- Modify: `tests/runtime-qualification/browser-entry.ts`
- Modify: `tests/runtime-qualification/shared/browserHarness.ts`
- Modify: `tests/runtime-qualification/schema/types.ts`
- Modify: `tests/runtime-qualification/browser-entry.test.ts`

**Interfaces:**
- Browser page method `runQwenGenerator(request)` returns `{ observation, receipts }` and never returns model or tensor bytes.
- Browser request contains exact variant, asset descriptors, backend, text, config, and trace limits.
- Browser receipts contain stage, frame, duration, and tensor metadata only.

- [ ] **Step 1: Write failing browser API contract tests**

Assert that the browser entry source exposes `runQwenGenerator`, constructs a browser-side `AssetResolver`, invokes `GeneratorPhase.load` and `generate`, and serializes no tensor `data` field in its receipts.

- [ ] **Step 2: Run the contract test and verify it fails**

```bash
pnpm exec vitest run --config tests/runtime-qualification/vitest.config.ts tests/runtime-qualification/browser-entry.test.ts
```

Expected: FAIL because the method is not exposed.

- [ ] **Step 3: Implement the page-side runner**

Inside the page, create an asset map from immutable descriptors. The resolver fetches each URL, validates bytes and SHA-256 with `verifyQualificationAsset`, and returns the `ArrayBuffer`. Create `createLiteRtRuntime({ assets, backend, packageName: '@litert-playground/qwen3-tts' })`, call `initialize` first, instantiate `GeneratorPhase` with the requested variant and trace collector, call `load`, call `generate` with `maxFrames: 1`, and call `dispose` in `finally`. Return stage receipts and status only.

- [ ] **Step 4: Wire Playwright without crossing large data**

Add the typed browser API and a harness method that calls `page.evaluate` with only the request metadata. Do not use the existing generic tensor bridge for this path.

- [ ] **Step 5: Run focused tests**

Run browser-entry, browser-harness, and Qwen package tests. Expected: all pass.

### Task 4: Add the composed browserMemory generator case

**Files:**
- Create: `tests/runtime-qualification/qwen-browsermemory-generator/case.ts`
- Create: `tests/runtime-qualification/qwen-browsermemory-generator/fixtures/assets.json`
- Create: `tests/runtime-qualification/qwen-browsermemory-generator/contract.test.ts`
- Create: `tests/runtime-qualification/qwen-browsermemory-generator/expected.ts`
- Create: `tests/runtime-qualification/qwen-browsermemory-generator/README.md`
- Modify: `tests/runtime-qualification/run-qualification.ts`
- Modify: `tests/runtime-qualification/schema/types.ts`

**Interfaces:**
- Produces case id `qwen-browsermemory-generator`.
- Uses the explicit `browserMemoryOmni` variant: Talker `talker_int4.tflite` from base revision `0eb3b8a4714972b065c160faec6a12158caa9dc0` and Omni `mtp_fp32.tflite` from revision `791880469d874546d884a0e6cf68564a61c04ca9`.
- Uses real tokenizer, codec embedding, MTP embedding, text embedding, text projection, and voice assets through browser-side immutable descriptors.
- Expected status is `pass`; any failure records the first stage and error message.

- [ ] **Step 1: Write failing case contracts**

Assert the case id, browser-observation evidence kind, backend, `browserMemory` variant, exact talker and MTP revisions, `maxFrames: 1`, expected pass status, and the eight required stage names in the receipt schema.

- [ ] **Step 2: Run the contract test and verify it fails**

```bash
pnpm exec vitest run --config tests/runtime-qualification/vitest.config.ts tests/runtime-qualification/qwen-browsermemory-generator/contract.test.ts
```

Expected: FAIL because the case and receipt schema do not exist.

- [ ] **Step 3: Implement the case**

Call the browser runner with the deterministic text and configuration. Map returned receipts into the qualification observation, preserve the first failing stage, and delete the phase through the page-side `finally` path.

- [ ] **Step 4: Register the case and document its boundary**

Register the case in `run-qualification.ts`. Document that this is the real generator boundary, not full codec/audio synthesis, and that passing it does not qualify the complete Qwen pipeline.

- [ ] **Step 5: Run focused qualification tests**

Run all qualification tests. Expected: all contract and schema tests pass.

### Task 5: Execute and verify the real one-frame sequence

**Files:**
- Modify: `docs/verification/2026-08-13-runtime-qualification.md`
- Generate locally: `tests/runtime-qualification/results/qwen-browsermemory-generator-*.json`

- [ ] **Step 1: Run the focused real browser qualification**

```bash
pnpm qualify -- --case qwen-browsermemory-generator --backend wasm
```

Expected on success: `qwen-browsermemory-generator wasm pass match`, with receipts for prompt construction, Talker prefill, first decode, MTP prediction, state update, and post-MTP Talker decode.

- [ ] **Step 2: If it fails, preserve the evidence and stop at the first failing stage**

Record the exact stage, tensor metadata, timings, and runtime error. Do not change the expected status to known limitation unless the composed sequence itself reproduces the failure.

- [ ] **Step 3: Update the verification record**

Record the exact revisions, browser, runtime version, backend, observed status, and the stage receipt summary. Do not claim full Qwen qualification from this generator-only result.

- [ ] **Step 4: Run the full verification gate**

```bash
pnpm verify
git diff --check
```

Expected: `pnpm verify` passes and `git diff --check` produces no output.

- [ ] **Step 5: Commit the composed qualification**

```bash
git add packages/qwen3-tts tests/runtime-qualification docs/verification/2026-08-13-runtime-qualification.md
git commit -m "test: qualify Qwen generator sequence"
```
