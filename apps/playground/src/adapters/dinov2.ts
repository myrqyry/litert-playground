import { Tensor } from '@litertjs/core'
import type { ModelAdapter, TensorSpec } from './types'
import { flatten } from './util'

const INPUT_SPECS: TensorSpec[] = [
  {
    name: 'args_0',
    dtype: 'float32',
    shape: [1, 3, 448, 448],
    description: 'Input image (NCHW, 448x448, RGB, 0-1)',
    constraints: { min: 0, max: 1 },
  },
]

const OUTPUT_SPECS: TensorSpec[] = [
  {
    name: 'output',
    dtype: 'float32',
    shape: [1, 1024, 384],
    description: 'DINOv2 patch embeddings (1024 patches, 384-dim each)',
  },
]

export const dinov2Adapter: ModelAdapter = {
  modelId: 'dinov2',
  metadata: {
    name: 'DINOv2 ViT-S/14',
    description: 'Vision Transformer patch embeddings (384-dim)',
    modelPath: '/models/dinov2_s_fp16.tflite',
    tags: ['vision', 'embeddings'],
  },
  inputSpecs: INPUT_SPECS,
  outputSpecs: OUTPUT_SPECS,

  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const spec = INPUT_SPECS[0]
    const raw = values[spec.name] ?? 0.5
    const data = flatten(raw)
    return { [spec.name]: new Tensor(data, spec.shape) }
  },

  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const entries = await Promise.all(
      Object.entries(outputs).map(async ([name, tensor]) => {
        const data = await tensor.data()
        return [name, Array.from(new Float32Array(data))]
      })
    )
    return Object.fromEntries(entries)
  },
}
