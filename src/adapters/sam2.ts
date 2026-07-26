import type { ModelAdapter } from './types'

export const sam2EncoderAdapter: ModelAdapter = {
  modelId: 'sam2-encoder',
  metadata: { name: 'SAM2 — Image Encoder', description: 'SAM2.1 Hiera-Tiny image encoder (1024×1024)', modelPath: '/models/sam2-encoder/sam2_tiny_image_encoder_fp16.tflite', tags: ['vision', 'segmentation'] },
  inputSpecs: [
    { name: 'image', dtype: 'float32', shape: [1, 3, 1024, 1024], description: 'RGB ImageNet-normalized NCHW' },
  ],
  outputSpecs: [
    { name: 'image_embeddings', dtype: 'float32', shape: [1, 256, 64, 64], description: 'Image feature embeddings' },
    { name: 'feat_s1', dtype: 'float32', shape: [1, 64, 128, 128], description: 'FPN feature map s1' },
    { name: 'feat_s0', dtype: 'float32', shape: [1, 32, 256, 256], description: 'FPN feature map s0' },
  ],
  prepareInputs: () => ({}),
  parseOutputs: async () => ({}),
}

export const sam2DecoderAdapter: ModelAdapter = {
  modelId: 'sam2-decoder',
  metadata: { name: 'SAM2 — Mask Decoder', description: 'SAM2.1 Hiera-Tiny mask decoder (promptable segmentation)', modelPath: '/models/sam2-mask/sam2_tiny_mask_decoder_fp16.tflite', tags: ['vision', 'segmentation'] },
  inputSpecs: [
    { name: 'image_embeddings', dtype: 'float32', shape: [1, 256, 64, 64], description: 'From SAM2 image encoder' },
    { name: 'feat_s1', dtype: 'float32', shape: [1, 64, 128, 128], description: 'FPN feature s1 from encoder' },
    { name: 'feat_s0', dtype: 'float32', shape: [1, 32, 256, 256], description: 'FPN feature s0 from encoder' },
    { name: 'sparse_prompt', dtype: 'float32', shape: [1, 2, 256], description: 'Host-side prompt encoding (point coords + labels)' },
  ],
  outputSpecs: [
    { name: 'masks', dtype: 'float32', shape: [1, 3, 256, 256], description: '3 mask predictions' },
    { name: 'iou_predictions', dtype: 'float32', shape: [1, 3], description: 'IoU scores per mask' },
  ],
  prepareInputs: () => ({}),
  parseOutputs: async () => ({}),
}

export const sam2Adapters: ModelAdapter[] = [sam2EncoderAdapter, sam2DecoderAdapter]
