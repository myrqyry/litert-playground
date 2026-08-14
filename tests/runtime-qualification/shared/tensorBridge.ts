import type {
  QualificationDType,
  QualificationTensor,
  QualificationTensorInput,
} from '../schema/types'

export interface SerializedTensor {
  data: number[]
  shape: number[]
  dtype: QualificationDType
}

export type SerializedInput =
  | { kind: 'positional'; tensors: SerializedTensor[] }
  | { kind: 'named'; tensors: Record<string, SerializedTensor> }

export function createQualificationTypedArray(
  dtype: QualificationDType | string,
  values: ArrayLike<number>,
): Float32Array | Int32Array | Uint8Array {
  switch (dtype) {
    case 'float32': return new Float32Array(values)
    case 'int32': return new Int32Array(values)
    case 'uint8': return new Uint8Array(values)
    default: throw new Error(`Unsupported tensor dtype: ${dtype}`)
  }
}

export function createQualificationZeroTypedArray(
  dtype: QualificationDType | string,
  length: number,
): Float32Array | Int32Array | Uint8Array {
  switch (dtype) {
    case 'float32': return new Float32Array(length)
    case 'int32': return new Int32Array(length)
    case 'uint8': return new Uint8Array(length)
    default: throw new Error(`Unsupported tensor dtype: ${dtype}`)
  }
}

export function serializeQualificationInput(options: {
  signature?: string
  input: QualificationTensorInput
}): { signature?: string; input: SerializedInput } {
  const serialize = (tensor: QualificationTensor): SerializedTensor => ({
    data: Array.from(tensor.data as ArrayLike<number>),
    shape: [...(tensor.shape ?? [])],
    dtype: tensor.dtype ?? inferQualificationDType(tensor.data),
  })

  const input = Array.isArray(options.input)
    ? { kind: 'positional' as const, tensors: options.input.map(serialize) }
    : {
      kind: 'named' as const,
      tensors: Object.fromEntries(
        Object.entries(options.input).map(([name, tensor]) => [name, serialize(tensor)]),
      ),
    }

  return options.signature ? { signature: options.signature, input } : { input }
}

function inferQualificationDType(data: unknown): QualificationDType {
  if (data instanceof Int32Array) return 'int32'
  if (data instanceof Uint8Array) return 'uint8'
  return 'float32'
}
