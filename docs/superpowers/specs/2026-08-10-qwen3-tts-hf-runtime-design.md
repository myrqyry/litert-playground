# Qwen3-TTS Hugging Face runtime proof

## Overview

The standalone Qwen3-TTS example uses the official
`litert-community/Qwen3-TTS-12Hz-0.6B-Base` repository as its model asset
source and keeps LiteRT runtime assets on their existing CDN path. This fixes
the current missing-runtime boundary without changing the reusable runtime
API.

## Asset boundaries

The example creates two separate concerns:

- The model resolver uses
  `https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base/resolve/main/`.
- `createLiteRtRuntime()` receives no model URL as `assetBase`, so its existing
  LiteRT WASM CDN default remains active.

The model manifest remains aligned with the repository's host-side pipeline:
the tokenizer, talker, MTP, codec, embedding tables, text projection, and
demo speaker assets are resolved from the model repository. Manifest byte
counts are updated from the repository API so progress and receipts describe
the actual download sizes.

## Runtime behavior

The example must reach LiteRT runtime initialization before attempting model
loading. The existing pipeline then loads and compiles the three graphs and
host-side tables, reports progress, runs the text-to-speech loop, validates the
PCM result, displays the receipt, and plays the audio.

This change does not claim runtime success from a build. Browser verification
must record each stage independently: model asset resolution, LiteRT runtime
initialization, graph compilation, inference, validated audio output, and
audible playback.

## Verification

Automated workspace verification must continue to pass. A browser run against
the standalone example must capture the first concrete failure if remote model
assets, browser capabilities, graph compilation, or inference remain blocked.
The structured verification record must mark only exercised stages as passed.
