# Root verification orchestration

## Overview

The repository root becomes the authoritative entrypoint for static workspace
verification. Root commands delegate to package-local scripts through pnpm's
recursive workspace execution, so new applications, packages, and examples
are included automatically when they expose the relevant script.

## Scope

This change updates only the root `package.json` scripts. It does not add
runtime model verification, browser automation, hardware checks, or a custom
verification framework. Those checks remain separate until they have enough
variation to justify dedicated commands.

## Commands

The root scripts have these responsibilities:

- `test` runs every workspace `test` script with
  `pnpm -r --if-present test`.
- `typecheck` runs every workspace `typecheck` script with
  `pnpm -r --if-present typecheck`.
- `build` runs every workspace `build` script with
  `pnpm -r --if-present build`, including standalone examples that define a
  build script.
- `test:boundaries` continues to run the root package-boundary suite.
- `verify` runs `typecheck`, `test`, `test:boundaries`, and `build` in that
  order.
- `test:watch` remains scoped to the playground for interactive development.

## Verification

The implementation is complete when `pnpm verify` exits successfully and
reports the workspace package checks, boundary tests, and builds. A successful
`pnpm verify` proves repository-level static correctness only; it does not
prove that model assets load, a device compiles a model, or generated audio is
audible.
