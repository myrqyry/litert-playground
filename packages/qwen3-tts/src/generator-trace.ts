export type GeneratorTraceStage =
  | 'talker-compile'
  | 'talker-prefill'
  | 'talker-output-read'
  | 'mtp-input-build'
  | 'mtp-compile'
  | 'mtp-run'
  | 'mtp-output-read'
  | 'state-update'

export interface GeneratorTraceTensor {
  name: string
  dtype: string
  shape: number[]
  elementCount: number
}

export interface GeneratorTraceEvent {
  stage: GeneratorTraceStage
  phase?: 'start' | 'end'
  frame?: number
  durationMs?: number
  tensors?: GeneratorTraceTensor[]
}

export interface TraceTensorLike {
  type: {
    dtype: string
    layout: { dimensions: ArrayLike<number> }
  }
}

export function traceTensor(name: string, tensor: TraceTensorLike): GeneratorTraceTensor {
  const shape = Array.from(tensor.type.layout.dimensions)
  return {
    name,
    dtype: tensor.type.dtype,
    shape,
    elementCount: shape.reduce((total, dimension) => total * dimension, 1),
  }
}

export function traceArray(
  name: string,
  dtype: string,
  shape: readonly number[],
): GeneratorTraceTensor {
  return {
    name,
    dtype,
    shape: Array.from(shape),
    elementCount: shape.reduce((total, dimension) => total * dimension, 1),
  }
}
