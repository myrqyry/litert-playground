# playground

React + Vite + Tailwind model lab. Consumes `@litert-playground/runtime-litert` for all LiteRT loading — never import `@litertjs/core` directly.

## Key constraints

- **Build output**: `apps/playground/dist`. Vercel config (`vercel.json`) sets `buildCommand: "pnpm --filter playground build"` and `outputDirectory: "apps/playground/dist"`.
- **Runtime contract**: `useModelRunner.ts` must import `createLiteRtRuntime` from `@litert-playground/runtime-litert`. Boundary tests enforce this.
- **Dev server**: `pnpm dev` starts Vite from the workspace root.

## Verification

```bash
pnpm --filter playground typecheck
pnpm --filter playground test
pnpm --filter playground test:watch   # watch mode
```
