# Examples and boundaries implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove external consumers use public package APIs and add a standalone Kokoro browser path with honest runtime verification.

**Architecture:** Examples are independent Vite entrypoints that construct the
shared runtime and model packages directly. Boundary tests scan imports and
package manifests rather than adding an architecture framework.

**Tech Stack:** Vite 8, React where already used, TypeScript, Vitest, Web Audio.

## Global Constraints

- Examples import only public `@litert-playground/*` entrypoints.
- Do not import PodQast or playground UI code.
- Audio/runtime verification is only `pass` after real browser/audio exercise.
- Do not extract text generation in this phase.

---

### Task 1: Move and migrate the Qwen example

**Files:**
- Move: `examples/minimal-qwen3-tts/*` into the workspace example location if
  the final Vite layout requires it
- Modify: `examples/minimal-qwen3-tts/main.tsx`
- Modify: `examples/minimal-qwen3-tts/extraction.test.ts`
- Modify: workspace Vite input configuration

**Interfaces:**
- Produces a Qwen example importing `Qwen3TtsPipeline`, manifest, runtime, and
  core types from public package entrypoints.

- [ ] Replace every `../../src/...` import with a package import.
- [ ] Preserve lifecycle, errors, receipt display, and audio playback behavior.
- [ ] Run the extraction test and production build.
- [ ] Commit: `refactor: migrate standalone Qwen example imports`.

### Task 2: Add standalone Kokoro example

**Files:**
- Create: `examples/minimal-kokoro/index.html`
- Create: `examples/minimal-kokoro/main.ts`
- Create: `examples/minimal-kokoro/app.css`
- Create: `examples/minimal-kokoro/extraction.test.ts`
- Modify: workspace Vite input configuration

**Interfaces:**
- Produces a browser page that accepts text, loads Kokoro, synthesizes audio,
  displays receipt/warnings/sample rate/duration, and plays Float32 audio with
  plain Web Audio.

- [ ] Write the extraction test asserting public package imports and absence of
  playground/PodQast imports.
- [ ] Implement explicit loading, ready, running, error, and disposed states.
- [ ] Add Web Audio playback using the returned sample rate and channels.
- [ ] Run extraction tests and build.
- [ ] Commit: `feat: add standalone Kokoro example`.

### Task 3: Add package-boundary tests

**Files:**
- Create: `tests/package-boundaries.test.ts`
- Modify: package `exports` and manifests if tests expose a leak

**Interfaces:**
- Produces lightweight assertions that public entrypoints resolve and dependency
  direction remains generic-to-specific.

- [ ] Assert examples have no `../../src` or `apps/playground` imports.
- [ ] Assert model packages do not import playground paths.
- [ ] Assert core has no model-package dependencies.
- [ ] Assert Kokoro and Qwen public pipeline types are assignable to the shared
  audio pipeline contract.
- [ ] Run the boundary test file and commit: `test: protect package boundaries`.

### Task 4: Align structured verification records

**Files:**
- Modify: `packages/inference-core/src/types.ts`
- Modify: `packages/inference-core/src/receipts.ts`
- Modify: existing standalone verification documentation
- Create: `docs/superpowers/verification/<date>-package-examples.md`

**Interfaces:**
- Produces shared receipt creation and verification records with separate
  `assets`, `compile`, `inference`, and `output` states.

- [ ] Ensure receipts include model ID, package/pipeline version, backend,
  load/compile/inference timings, summaries, warnings, timestamp, and safe
  environment information.
- [ ] Record Qwen and Kokoro browser gates independently.
- [ ] Mark unavailable assets, compile, inference, or audio as `untested` or
  `fail`, never as successful based only on build output.
- [ ] Commit: `docs: record package example verification`.

### Task 5: Final migration gate

**Files:**
- Modify only files identified by failing checks.

- [ ] Run `pnpm install --frozen-lockfile`.
- [ ] Run all workspace type-checks.
- [ ] Run all unit and boundary tests.
- [ ] Run production builds for playground and examples.
- [ ] Exercise Kokoro in a real browser if model assets are available; record
  whether audible output was verified.
- [ ] Exercise Qwen in a real browser if assets are available; record whether
  audio output was verified.
- [ ] Report remaining duplicated PodQast files without modifying PodQast.
