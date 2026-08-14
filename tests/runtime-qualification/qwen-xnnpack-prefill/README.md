# Qwen XNNPACK prefill

This case runs the smallest browserMemory Qwen graph that exposes the
`prefill_32` signature: the immutable 255,998,768-byte `talker_int4.tflite`
asset from the pinned Hugging Face revision. The expected stage is prefill.
If the current runtime passes this path, the result intentionally mismatches
the known-limitation expectation and keeps the qualification lane red until
the failing browserMemory graph is identified.
