# Workspace, core, and runtime implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the pnpm workspace and move generic contracts, asset resolution, and LiteRT runtime ownership into consumable packages.

**Architecture:** Create `apps/playground`, `packages/inference-core`, and
`packages/runtime-litert`. Core owns model-agnostic contracts and resolver
behavior; runtime owns `@litertjs/core` and constructs `RuntimeContext`.

**Tech Stack:** pnpm workspaces, TypeScript, Vite 8, Vitest 4, `@litertjs/core`.

## Global Constraints

- Use package names under `@litert-playground/*`.
- Preserve playground UI behavior; do not redesign the app.
- Do not add OPFS, a monorepo tool, or a new test framework.
- Model packages must not call `loadLiteRt()` directly.
- Keep `PodcastTts` outside `inference-core`.

---

### Task 1: Create workspace manifests

**Files:**
- Create: `pnpm-workspace.yaml`
- Create: `apps/playground/package.json`
- Create: `packages/inference-core/package.json`
- Create: `packages/runtime-litert/package.json`
- Modify: `package.json`
- Modify: `pnpm-lock.yaml`

**Interfaces:**
- Produces workspace packages named `@litert-playground/inference-core` and
  `@litert-playground/runtime-litert`.

- [ ] Write package manifests with `name`, `version`, `private`, `type`, and
  `exports` pointing at `src/index.ts`.
- [ ] Move root app dependencies and scripts into `apps/playground` without
  changing React/Vite versions.
- [ ] Add workspace dependencies using `workspace:*`.
- [ ] Run `pnpm install --frozen-lockfile`.
- [ ] Commit: `chore: create inference workspace packages`.

### Task 2: Extract inference core contracts

**Files:**
- Create: `packages/inference-core/src/types.ts`
- Create: `packages/inference-core/src/validation.ts`
- Create: `packages/inference-core/src/index.ts`
- Create: `packages/inference-core/src/types.test.ts`
- Modify: `src/core/types.ts`
- Modify: `src/core/validation.ts`

**Interfaces:**
- Produces public exports for `Pipeline`, `RuntimeContext`, `AssetResolver`,
  `ModelAsset`, `ModelManifest`, `Capability`, `Backend`, all normalized
  result types, `InferenceError`, `InferenceReceipt`, and verification state.

- [ ] Copy the existing contracts into package-owned files, adding structured
  verification fields `assets`, `compile`, `inference`, and `output`.
- [ ] Keep `inference-core` free of imports from model packages, LiteRT, React,
  and PodQast.
- [ ] Export only the intended public symbols from `src/index.ts`.
- [ ] Add a dependency-graph test that rejects imports containing `/kokoro`,
  `/qwen3-tts`, `@litertjs/core`, or `apps/playground`.
- [ ] Run `pnpm --filter @litert-playground/inference-core test` and type-check.
- [ ] Commit: `feat: extract inference core package`.

### Task 3: Move and harden asset resolution

**Files:**
- Create: `packages/inference-core/src/assets/http-resolver.ts`
- Create: `packages/inference-core/src/assets/http-resolver.test.ts`
- Create: `packages/inference-core/src/assets/index.ts`
- Remove: `src/assets/http-resolver.ts`
- Remove: `src/assets/http-resolver.test.ts`
- Modify: `packages/inference-core/src/types.ts`
- Modify: `packages/inference-core/src/index.ts`

**Interfaces:**
- Produces `createHttpAssetResolver(baseUrl)` with
  `resolve(asset, { signal, onProgress })` and
  `stream(asset, { signal, onProgress })`.

- [ ] Write failing tests for fetch signal forwarding, rejected-cache eviction,
  retry after failure, asset-named errors, and progress callbacks.
- [ ] Implement one shared request path for buffered and streamed fetches.
- [ ] Evict a cache entry when its promise rejects.
- [ ] Run the resolver test file and confirm all cases pass.
- [ ] Commit: `feat: extract abortable asset resolver`.

### Task 4: Extract LiteRT runtime ownership

**Files:**
- Create: `packages/runtime-litert/src/context.ts`
- Create: `packages/runtime-litert/src/capabilities.ts`
- Create: `packages/runtime-litert/src/npy.ts`
- Create: `packages/runtime-litert/src/index.ts`
- Create: `packages/runtime-litert/src/context.test.ts`
- Remove: `src/runtime/context.ts`
- Remove: `src/runtime/context.test.ts`
- Remove: `src/runtime/capabilities.ts`
- Remove: `src/runtime/index.ts`

**Interfaces:**
- Produces `createLiteRtRuntime({ backend, assets, signal })` returning a
  `RuntimeContext` whose observable backend is `webgpu`, `wasm`, or `webnn`.

- [ ] Write tests for usable WebGPU adapter selection, missing adapter fallback,
  explicit WASM, unavailable explicit backend, and GPU compile fallback.
- [ ] Implement capability probing with adapter/device acquisition, not only
  `navigator.gpu` existence.
- [ ] Wrap LiteRT initialization, model compilation, NPY loading, and asset
  access in runtime-owned helpers.
- [ ] Convert backend and compile errors to distinct `InferenceError` codes.
- [ ] Run runtime tests and type-check the package.
- [ ] Commit: `feat: extract LiteRT runtime package`.

### Task 5: Move playground into the workspace

**Files:**
- Move: root app source files into `apps/playground/src/`
- Move: `index.html` to `apps/playground/index.html`
- Move: `vite.config.ts` to `apps/playground/vite.config.ts`
- Modify: `apps/playground/src/App.tsx`
- Modify: `apps/playground/src/main.tsx`
- Modify: `apps/playground/package.json`
- Create: root `package.json` workspace scripts if needed

**Interfaces:**
- Playground imports core and runtime APIs from workspace package names, not
  relative paths into `packages` or old root `src`.

- [ ] Update app imports to `@litert-playground/inference-core` and
  `@litert-playground/runtime-litert`.
- [ ] Preserve existing non-TTS adapters and UI behavior while they remain app
  owned.
- [ ] Run `pnpm --filter playground typecheck` and `pnpm --filter playground build`.
- [ ] Start the app with `pnpm --filter playground dev` and verify the root UI
  loads.
- [ ] Commit: `refactor: move playground into workspace app`.

### Phase gate

- [ ] Run `pnpm install --frozen-lockfile`.
- [ ] Run workspace type-checks.
- [ ] Run all unit tests.
- [ ] Run the production build.
- [ ] Confirm no app import reaches old core/runtime source paths.
