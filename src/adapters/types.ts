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
  load(model: import('@litertjs/core').LiteRtModel): Promise<void>
  applyInputs(values: Record<string, any>, session: import('@litertjs/core').LiteRtSession): Promise<void>
  parseOutputs(session: import('@litertjs/core').LiteRtSession): Promise<Record<string, any>>
}
