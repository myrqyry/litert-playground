import { Tensor } from '@litertjs/core'
import type { ModelAdapter, TensorSpec } from './types'

const INPUT_SPECS: TensorSpec[] = [
  {
    name: 'args_0',
    dtype: 'float32',
    shape: [1, 3, 256, 256],
    description: 'Noisy image (NCHW, 256x256, RGB, 0-1)',
    constraints: { min: 0, max: 1 },
  },
]

const OUTPUT_SPECS: TensorSpec[] = [
  {
    name: 'output',
    dtype: 'float32',
    shape: [1, 3, 256, 256],
    description: 'Denoised image (NCHW, 256x256)',
  },
]

function flatten(arr: any): Float32Array {
  if (arr instanceof Float32Array) return arr
  if (typeof arr[0] === 'number') return new Float32Array(arr)
  const flat = arr.flat(Infinity) as number[]
  return new Float32Array(flat)
}

export const nafnetAdapter: ModelAdapter = {
  modelId: 'nafnet',
  metadata: {
    name: 'NAFNet SIDD width32',
    description: 'Image denoising (SIDD benchmark)',
    modelPath: 'models/nafnet_sidd_width32_fp16.tflite',
    tags: ['vision', 'denoising'],
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
