# EfficientDet dynamic output

This case isolates the known LiteRT.js 2.5.3 WASM limitation around dynamic
output tensor materialization. It is a runtime reproduction, not a product
fallback or an accuracy claim. The model descriptor remains absent until a
real reproduction records an immutable model URL, byte count, and SHA-256.
