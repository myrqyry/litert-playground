export interface TensorSpec {
  name: string
  dtype: 'float32' | 'int32' | 'int8' | 'uint8'
  shape: number[]
  description: string
  constraints?: {
    min?: number
    max?: number
    enum?: string[]
    items?: string[]
  }
}

export interface ModelMetadata {
  name: string
  description: string
  modelPath: string
  tags: string[]
}

export interface ModelAdapter {
  modelId: string
  metadata: ModelMetadata
  inputSpecs: TensorSpec[]
  outputSpecs: TensorSpec[]
  prepareInputs(values: Record<string, any>): Record<string, import('@litertjs/core').Tensor>
  parseOutputs(outputs: Record<string, import('@litertjs/core').Tensor>): Promise<Record<string, any>>
}
