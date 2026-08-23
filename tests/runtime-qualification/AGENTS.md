# runtime-qualification

Qualification suite for verifying LiteRT runtime and model behavior.

## Structure

- `tests/runtime-qualification/vitest.config.ts` — separate vitest config with `environment: 'node'`.
- `shared/` — evidenceWriter and shared harness.
- `qwen-browsermemory-generator/` — Qwen browser memory qualification case.
- Other subdirectories (`upstream/`, `tiny-litert-baseline/`, etc.) hold per-model or per-scenario suites.

## Invocation

```bash
pnpm test:qualification     # vitest with the qualification config
pnpm qualify                # tsx runner (tests/runtime-qualification/run-qualification.ts)
```

`pnpm verify` runs `test:qualification` as one of its gates — do not skip it.

## Conventions

- Known limitations are promoted to `status: 'known-limitation'` in case results so `evidenceWriter` can match them as `pass` (not `fail`).
- `evidenceWriter` accepts observed `'fail'` or `'known-limitation'` for an expected known-limitation.
