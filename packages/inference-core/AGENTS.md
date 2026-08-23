# inference-core

Contracts, asset resolvers, receipts, and validation. No model-specific logic.

## Key constraints

- **Must stay independent**: no references to kokoro, qwen3-tts, @litertjs/core, or any model package.
- **Asset verification**: `verifyAssetIntegrity(asset, buffer)` checks SHA-256 and size. Exported from `src/assets/manifest-resolver.ts` and re-exported in `src/assets/index.ts`. Consumers (e.g., `browser-cache`) call it on both cache-hit and fresh paths.
- GOTCHA: `resolve()`'s `assetFromManifest` lookup by `asset.id` overrides any `sha256` on the argument — tests that supply a wrong-hash variant must match by `id`, not by passing a new asset object.

## Verification

```bash
pnpm --filter @litert-playground/inference-core typecheck
pnpm --filter @litert-playground/inference-core test
```
