# Root verification orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make root verification cover every workspace test, typecheck, and
build script while preserving the explicit package-boundary test.

**Architecture:** Keep package-local scripts as the source of truth. Root
scripts delegate through `pnpm -r --if-present`, which automatically includes
future workspace members that expose the relevant script. `verify` sequences
static checks only and does not claim runtime model success.

**Tech Stack:** pnpm 11 workspaces, package-local TypeScript, Vitest, and Vite.

## Global Constraints

- Include standalone examples in recursive builds.
- Keep `test:watch` scoped to the playground.
- Keep `test:boundaries` as an explicit root-only test.
- Keep runtime model, browser, hardware, and audio verification outside
  `pnpm verify`.

---

### Task 1: Authoritative root scripts

**Files:**
- Modify: `package.json:7-15`
- Test: `package.json` scripts through `pnpm verify`

**Interfaces:**
- Consumes: package-local `test`, `typecheck`, and `build` scripts exposed by
  workspace members.
- Produces: root commands `test`, `typecheck`, `build`, `test:boundaries`,
  `verify`, and `test:watch` with the following exact definitions:

```json
{
  "test": "pnpm -r --if-present test",
  "test:boundaries": "vitest run tests/package-boundaries.test.ts",
  "test:watch": "pnpm --filter playground test:watch",
  "typecheck": "pnpm -r --if-present typecheck",
  "build": "pnpm -r --if-present build",
  "verify": "pnpm typecheck && pnpm test && pnpm test:boundaries && pnpm build"
}
```

- [ ] **Step 1: Update the root script definitions**

Replace only the existing root `test`, `typecheck`, and `build` wrappers and
add `verify`. Leave `dev`, `preview`, `test:boundaries`, and `test:watch`
unchanged.

- [ ] **Step 2: Run the authoritative verification command**

Run:

```bash
pnpm verify
```

Expected: every workspace member with a matching script runs; package tests,
all workspace typechecks, the boundary suite, the playground build, and both
standalone example builds complete successfully.

- [ ] **Step 3: Confirm the root diff is limited to scripts**

Run:

```bash
git diff -- package.json
git diff --check
```

Expected: only the four root script entries are changed or added, and
`git diff --check` emits no output.

- [ ] **Step 4: Commit the implementation**

```bash
git add package.json
git commit -m "ci: verify all workspace projects"
```
