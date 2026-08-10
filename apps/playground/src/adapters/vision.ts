import type { ModelAdapter } from './types'

const inpSpec = (n: string, s: number[], d: 'float32' | 'int32', desc: string) =>
  ({ name: n, dtype: d, shape: s, description: desc })
const outSpec = (n: string, s: number[], d: 'float32' | 'int32', desc: string) =>
  ({ name: n, dtype: d, shape: s, description: desc })

export const headpose6drepnetAdapter: ModelAdapter = {
  modelId: '6drepnet',
  metadata: { name: '6DRepNet — Head Pose', description: '6D head pose estimation (Euler angles)', modelPath: '/models/6drepnet/6drepnet.tflite', tags: ['vision', 'pose'] },
  inputSpecs: [inpSpec('input', [1, 3, 224, 224], 'float32', 'RGB ImageNet-normalized NCHW')],
  outputSpecs: [outSpec('output', [6], 'float32', '6D rotation vector → Gram-Schmidt → Euler')],
  prepareInputs: (v) => ({}),
  parseOutputs: async (o) => ({}),
}

export const blazeFaceAdapter: ModelAdapter = {
  modelId: 'blaze-face',
  metadata: { name: 'BlazeFace — Face Detection', description: 'MediaPipe face detection (full-range)', modelPath: '/models/blaze-face/blaze_face_full_range.tflite', tags: ['vision', 'face'] },
  inputSpecs: [inpSpec('input', [1, 128, 128, 3], 'float32', 'RGB 0-255 NHWC')],
  outputSpecs: [
    outSpec('regressors', [1, 896, 16], 'float32', 'Bounding box + 6 keypoint regressors'),
    outSpec('classificators', [1, 896, 1], 'float32', 'Face presence scores'),
  ],
  prepareInputs: (v) => ({}),
  parseOutputs: async (o) => ({}),
}

export const yoloxAdapter: ModelAdapter = {
  modelId: 'yolox',
  metadata: { name: 'YOLOX-M — Object Detection', description: 'YOLOX-M COCO detection (640×640)', modelPath: '/models/yolox/yolox_m.tflite', tags: ['vision', 'detection'] },
  inputSpecs: [inpSpec('images', [1, 640, 640, 3], 'float32', 'BGR 0-255 NHWC, letterbox pad 114')],
  outputSpecs: [outSpec('output', [1, 8400, 85], 'float32', 'Raw heads: 4 box + 1 obj + 80 class')],
  prepareInputs: (v) => ({}),
  parseOutputs: async (o) => ({}),
}

export const u2netAdapter: ModelAdapter = {
  modelId: 'u2net',
  metadata: { name: 'U2-Net — Portrait Sketch', description: 'Photo to pencil line drawing', modelPath: '/models/u2net/portrait.tflite', tags: ['vision', 'creative'] },
  inputSpecs: [inpSpec('input', [1, 3, 512, 512], 'float32', 'RGB ImageNet-normalized NCHW')],
  outputSpecs: [outSpec('output', [1, 1, 512, 512], 'float32', 'Sketch map [0,1], invert for dark-on-white')],
  prepareInputs: (v) => ({}),
  parseOutputs: async (o) => ({}),
}

export const edsrAdapter: ModelAdapter = {
  modelId: 'edsr',
  metadata: { name: 'EDSR ×4 — Super Resolution', description: '4× super resolution (128→512)', modelPath: '/models/edsr/edsr.tflite', tags: ['vision', 'enhancement'] },
  inputSpecs: [inpSpec('input', [1, 3, 128, 128], 'float32', 'RGB x/255 NCHW')],
  outputSpecs: [outSpec('output', [1, 3, 512, 512], 'float32', 'RGB 0-1 NCHW, clamp ×255')],
  prepareInputs: (v) => ({}),
  parseOutputs: async (o) => ({}),
}

export const miganAdapter: ModelAdapter = {
  modelId: 'migan',
  metadata: { name: 'MI-GAN — Image Inpainting', description: 'Object removal / image inpainting (512×512)', modelPath: '/models/migan/migan_fp16.tflite', tags: ['vision', 'inpainting'] },
  inputSpecs: [inpSpec('input', [1, 4, 512, 512], 'float32', 'concat(mask-0.5, rgb·mask) NCHW')],
  outputSpecs: [outSpec('output', [1, 3, 512, 512], 'float32', 'Inpainted RGB [-1,1] NCHW')],
  prepareInputs: (v) => ({}),
  parseOutputs: async (o) => ({}),
}

const sharedStyleSpecs = {
  inputSpecs: [inpSpec('input', [1, 3, 256, 256], 'float32', 'RGB 0-255 NCHW (no normalization)')],
  outputSpecs: [outSpec('output', [1, 3, 256, 256], 'float32', 'RGB 0-255 NCHW (clamp)')],
}

export const styleAdapters: ModelAdapter[] = [
  { modelId: 'style-candy', metadata: { name: 'Neural Style — Candy', description: 'Fast Neural Style Transfer (candy)', modelPath: '/models/neural-style/style_candy_fp16.tflite', tags: ['vision', 'creative'] }, ...sharedStyleSpecs, prepareInputs: (v) => ({}), parseOutputs: async (o) => ({}) },
  { modelId: 'style-mosaic', metadata: { name: 'Neural Style — Mosaic', description: 'Fast Neural Style Transfer (mosaic)', modelPath: '/models/neural-style/style_mosaic_fp16.tflite', tags: ['vision', 'creative'] }, ...sharedStyleSpecs, prepareInputs: (v) => ({}), parseOutputs: async (o) => ({}) },
  { modelId: 'style-rain-princess', metadata: { name: 'Neural Style — Rain Princess', description: 'Fast Neural Style Transfer (rain princess)', modelPath: '/models/neural-style/style_rain_princess_fp16.tflite', tags: ['vision', 'creative'] }, ...sharedStyleSpecs, prepareInputs: (v) => ({}), parseOutputs: async (o) => ({}) },
  { modelId: 'style-udnie', metadata: { name: 'Neural Style — Udnie', description: 'Fast Neural Style Transfer (udnie)', modelPath: '/models/neural-style/style_udnie_fp16.tflite', tags: ['vision', 'creative'] }, ...sharedStyleSpecs, prepareInputs: (v) => ({}), parseOutputs: async (o) => ({}) },
]

export const visionAdapters: ModelAdapter[] = [
  headpose6drepnetAdapter,
  blazeFaceAdapter,
  yoloxAdapter,
  u2netAdapter,
  edsrAdapter,
  miganAdapter,
  ...styleAdapters,
]
