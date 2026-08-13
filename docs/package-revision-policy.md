# Package revision policy

Consuming applications must use one LiteRT Playground revision for all
Playground packages. This keeps package contracts, runtime behavior, and model
qualification metadata aligned.

## Rules

- Select one LiteRT Playground Git SHA for each consuming application.
- Resolve every Playground package used by that application from the same SHA.
- Do not mix packages from different Playground revisions.
- Use SHA pinning as the supported revision mechanism today.
- Treat peer dependency ranges as compatibility information, not as a license
  to mix revisions.
- Future release tags such as `runtime-v0.2.x` may improve ergonomics, but they
  do not change the single-revision rule.

The packed compatibility surface currently covers these packages:

- `@litert-playground/inference-core`
- `@litert-playground/runtime-litert`
- `@litert-playground/text-gen`
- `@litert-playground/kokoro`
- `@litert-playground/qwen3-tts`
- `@litert-playground/image-embedding`
- `@litert-playground/video-classification`

Each package is consumed through its public `.` entrypoint. Consumers must not
import files from a package's `src` directory.
