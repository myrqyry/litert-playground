# Qwen XNNPACK prefill

This case runs the smallest browserMemory Qwen graph that exposes the
LiteRT-LM Omni MTP graph: the immutable 440,528,628-byte `mtp_fp32.tflite`
asset from the pinned Hugging Face revision. The expected stage is prefill.
The current runtime passes this standalone path, so the result intentionally
mismatches the known-limitation expectation and keeps the qualification lane
red until the failing browserMemory graph is identified. The talker
`prefill_32` candidate also passed separately.
