import { loadAndCompile, loadLiteRt, Tensor } from '@litertjs/core'

const models = new Map<number, Awaited<ReturnType<typeof loadAndCompile>>>()
let nextModelId = 1

Object.assign(window, {
  litertQualification: {
    async initialize(path: string) {
      await loadLiteRt(path)
    },
    async loadAndCompile(bytes: number[], accelerator: 'wasm' | 'webgpu') {
      const model = await loadAndCompile(new Uint8Array(bytes), { accelerator })
      const id = nextModelId++
      models.set(id, model)
      return {
        id,
        inputs: model.getInputDetails().map((detail) => ({
          shape: Array.from(detail.shape),
          dtype: detail.dtype,
        })),
        outputs: model.getOutputDetails().map((detail) => ({
          shape: Array.from(detail.shape),
          dtype: detail.dtype,
        })),
      }
    },
    async run(id: number, values: Float32Array[], shapes: number[][]) {
      const model = models.get(id)
      if (!model) throw new Error(`Unknown qualification model: ${id}`)
      const tensors = values.map((value, index) => Tensor.fromTypedArray(value, shapes[index]))
      try {
        const outputs = await model.run(tensors)
        return outputs.map((output) => Array.from(output.toTypedArray() as Float32Array))
      } finally {
        tensors.forEach((tensor) => tensor.delete())
      }
    },
    delete(id: number) {
      models.get(id)?.delete()
      models.delete(id)
    },
  },
})
