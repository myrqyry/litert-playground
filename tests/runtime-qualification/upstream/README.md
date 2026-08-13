# Isolated runtime lanes

The production lane remains `@litertjs/core` 2.5.3. Upstream or prerelease
LiteRT.js builds run only from a temporary qualification directory and never
modify the Playground lockfile or production dependency graph.

The optional Google `@litertjs/model-tester` lane accepts immutable standalone
TFLite asset descriptors and reports WASM/WebGPU results. It exists to
separate managed-runtime wrapper failures from LiteRT.js runtime and backend
failures. It is not part of `pnpm test:qualification`.
