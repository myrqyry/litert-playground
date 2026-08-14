# Qwen browserMemory generator

This case runs the real `GeneratorPhase` in Chromium WASM with the explicit
`browserMemoryOmni` variant. It retains the base-revision INT4 Talker and the
Omni-revision FP32 MTP graph in one browser runtime, then executes one frame.

The one-frame run reaches prompt construction and both model compilation, then
fails while materializing the real Talker prefill inputs in the current
Chromium WASM runtime. Receipts contain only stage, timing, tensor name, dtype,
shape, and element count. This is the composed browserMemory limitation; it is
not a standalone MTP limitation and does not qualify codec decoding or
complete Qwen audio synthesis.
