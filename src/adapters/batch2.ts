import type { ModelAdapter } from './types'

const s = (n: string, d: 'float32' | 'int32', sh: number[], desc: string) =>
  ({ name: n, dtype: d, shape: sh, description: desc })

const stub = { prepareInputs: (v: any) => ({}), parseOutputs: async (o: any) => ({}) }

export const depth3Adapter: ModelAdapter = {
  modelId: 'depth-anything-3',
  metadata: { name: 'Depth Anything 3', description: 'Monocular depth estimation (896×504)', modelPath: '/models/depth-anything/da3_small_gpu_fp16.tflite', tags: ['vision', 'depth'] },
  inputSpecs: [s('input', 'float32', [1, 3, 896, 504], 'RGB ImageNet-normalized NCHW')],
  outputSpecs: [s('depth', 'float32', [1, 1, 896, 504], 'Depth map')],
  ...stub,
}

export const rtmposeFaceAdapter: ModelAdapter = {
  modelId: 'rtmpose-face',
  metadata: { name: 'RTMPose Face', description: '98-point face alignment WFLW (256×256)', modelPath: '/models/rtmpose-face/rtm_face_fp16.tflite', tags: ['vision', 'pose'] },
  inputSpecs: [s('input', 'float32', [1, 3, 256, 256], 'RGB mmpose mean/std NCHW')],
  outputSpecs: [
    s('simcc_x', 'float32', [1, 98, 512], 'SimCC X bins (argmax/2 → pixel)'),
    s('simcc_y', 'float32', [1, 98, 512], 'SimCC Y bins (argmax/2 → pixel)'),
  ],
  ...stub,
}

export const rtmposeHandAdapter: ModelAdapter = {
  modelId: 'rtmpose-hand',
  metadata: { name: 'RTMPose Hand', description: '21-keypoint hand pose (256×256)', modelPath: '/models/rtmpose-hand/rtmhand_fp16.tflite', tags: ['vision', 'pose'] },
  inputSpecs: [s('input', 'float32', [1, 3, 256, 256], 'RGB ImageNet norm NCHW')],
  outputSpecs: [
    s('simcc_x', 'float32', [1, 21, 512], 'SimCC X bins'),
    s('simcc_y', 'float32', [1, 21, 512], 'SimCC Y bins'),
  ],
  ...stub,
}

export const rtmposeSAdapter: ModelAdapter = {
  modelId: 'rtmpose-s',
  metadata: { name: 'RTMPose-s', description: '17-keypoint body pose (256×192)', modelPath: '/models/rtmpose-s/rtmpose_s_fp16.tflite', tags: ['vision', 'pose'] },
  inputSpecs: [s('input', 'float32', [1, 3, 256, 192], 'RGB mmpose mean/std NCHW')],
  outputSpecs: [
    s('simcc_x', 'float32', [1, 17, 384], 'SimCC X bins'),
    s('simcc_y', 'float32', [1, 17, 512], 'SimCC Y bins'),
  ],
  ...stub,
}

export const rtmwAdapter: ModelAdapter = {
  modelId: 'rtmw-wholebody',
  metadata: { name: 'RTMW-m WholeBody', description: '133-keypoint whole-body pose (256×192)', modelPath: '/models/rtmw-wholebody/rtmw_fp16.tflite', tags: ['vision', 'pose'] },
  inputSpecs: [s('input', 'float32', [1, 3, 256, 192], 'RGB mmpose mean/std NCHW')],
  outputSpecs: [
    s('simcc_x', 'float32', [1, 133, 384], 'SimCC X bins'),
    s('simcc_y', 'float32', [1, 133, 512], 'SimCC Y bins'),
  ],
  ...stub,
}

export const yolactAdapter: ModelAdapter = {
  modelId: 'yolact',
  metadata: { name: 'YOLACT — Instance Seg', description: 'Real-time instance segmentation ResNet50 (550×550)', modelPath: '/models/yolact/yolact.tflite', tags: ['vision', 'segmentation'] },
  inputSpecs: [s('input', 'float32', [1, 3, 550, 550], 'BGR normalized NCHW')],
  outputSpecs: [
    s('loc', 'float32', [1, 19248, 4], 'Box regressors'),
    s('conf', 'float32', [1, 19248, 81], 'Class scores (softmax)'),
    s('mask', 'float32', [1, 19248, 32], 'Mask coefficients'),
    s('proto', 'float32', [1, 138, 138, 32], 'Prototype masks'),
  ],
  ...stub,
}

export const ormbgAdapter: ModelAdapter = {
  modelId: 'ormbg',
  metadata: { name: 'ORMBG — BG Removal', description: 'Background removal (1024×1024)', modelPath: '/models/ormbg/ormbg.tflite', tags: ['vision', 'matting'] },
  inputSpecs: [s('input', 'float32', [1, 3, 1024, 1024], 'RGB x/255 NCHW')],
  outputSpecs: [s('alpha', 'float32', [1, 1, 1024, 1024], 'Alpha matte')],
  ...stub,
}

export const u2netSalientAdapter: ModelAdapter = {
  modelId: 'u2net-salient',
  metadata: { name: 'U-2-Net — Saliency', description: 'Salient object detection (320×320)', modelPath: '/models/u2net-salient/u2net_fp16.tflite', tags: ['vision', 'segmentation'] },
  inputSpecs: [s('input', 'float32', [1, 3, 320, 320], 'RGB ImageNet-norm NCHW')],
  outputSpecs: [s('saliency', 'float32', [1, 1, 320, 320], 'Saliency mask [0,1]')],
  ...stub,
}

export const modnetAdapter: ModelAdapter = {
  modelId: 'modnet',
  metadata: { name: 'MODNet — Matting', description: 'Trimap-free portrait matting (512×512)', modelPath: '/models/modnet/modnet.tflite', tags: ['vision', 'matting'] },
  inputSpecs: [s('input', 'float32', [1, 3, 512, 512], 'RGB [-1,1] NCHW')],
  outputSpecs: [s('alpha', 'float32', [1, 1, 512, 512], 'Soft alpha matte [0,1]')],
  ...stub,
}

export const yolo26SegAdapter: ModelAdapter = {
  modelId: 'yolo26-seg',
  metadata: { name: 'YOLO26m Segmentation', description: 'EdgeFirst YOLO26 segmentation INT8', modelPath: '/models/yolo26-seg/yolo26m-seg-int8-smart.tflite', tags: ['vision', 'segmentation'] },
  inputSpecs: [s('input', 'float32', [1, 640, 640, 3], 'RGB NHWC')],
  outputSpecs: [s('output', 'float32', [1, 116, 8400], 'Detection + mask coeffs')],
  ...stub,
}

export const yolo11SegAdapter: ModelAdapter = {
  modelId: 'yolo11-seg',
  metadata: { name: 'YOLO11m Segmentation', description: 'EdgeFirst YOLO11 segmentation INT8', modelPath: '/models/yolo11-seg/yolo11m-seg-int8-smart.tflite', tags: ['vision', 'segmentation'] },
  inputSpecs: [s('input', 'float32', [1, 640, 640, 3], 'RGB NHWC')],
  outputSpecs: [s('output', 'float32', [1, 116, 8400], 'Detection + mask coeffs')],
  ...stub,
}

export const yolov8SegAdapter: ModelAdapter = {
  modelId: 'yolov8-seg',
  metadata: { name: 'YOLOv8m Segmentation', description: 'EdgeFirst YOLOv8 segmentation INT8', modelPath: '/models/yolov8-seg/yolov8m-seg-int8-smart.tflite', tags: ['vision', 'segmentation'] },
  inputSpecs: [s('input', 'float32', [1, 640, 640, 3], 'RGB NHWC')],
  outputSpecs: [s('output', 'float32', [1, 116, 8400], 'Detection + mask coeffs')],
  ...stub,
}

export const clipsegAdapters: ModelAdapter[] = [
  {
    modelId: 'clipseg-text',
    metadata: { name: 'CLIPSeg — Text Encoder', description: 'Text encoder for text-prompted segmentation', modelPath: '/models/clipseg/clipseg_text_fp16.tflite', tags: ['vision', 'segmentation'] },
    inputSpecs: [s('input', 'float32', [1, 77], 'Text tokens')],
    outputSpecs: [s('text_emb', 'float32', [1, 64], 'Text embedding')],
    ...stub,
  },
  {
    modelId: 'clipseg-vision',
    metadata: { name: 'CLIPSeg — Vision Encoder', description: 'Vision encoder for text-prompted segmentation', modelPath: '/models/clipseg/clipseg_vision_fp16.tflite', tags: ['vision', 'segmentation'] },
    inputSpecs: [s('input', 'float32', [1, 3, 352, 352], 'RGB x/255 NCHW')],
    outputSpecs: [s('vision_emb', 'float32', [1, 64], 'Visual embedding')],
    ...stub,
  },
  {
    modelId: 'clipseg-decoder',
    metadata: { name: 'CLIPSeg — Decoder', description: 'Decoder for text-prompted segmentation', modelPath: '/models/clipseg/clipseg_decoder.tflite', tags: ['vision', 'segmentation'] },
    inputSpecs: [
      s('text_emb', 'float32', [1, 64], 'From CLIPSeg text encoder'),
      s('vision_emb', 'float32', [1, 64], 'From CLIPSeg vision encoder'),
    ],
    outputSpecs: [s('mask', 'float32', [1, 1, 352, 352], 'Segmentation mask')],
    ...stub,
  },
]

export const adapters13: ModelAdapter[] = [
  depth3Adapter, rtmposeFaceAdapter, rtmposeHandAdapter, rtmposeSAdapter, rtmwAdapter,
  yolactAdapter, ormbgAdapter, u2netSalientAdapter, modnetAdapter,
  yolo26SegAdapter, yolo11SegAdapter, yolov8SegAdapter, ...clipsegAdapters,
]
