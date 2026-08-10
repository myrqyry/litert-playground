import { Tensor, type CompiledModel, type TypedArray } from '@litertjs/core'

export type BonsaiGraphInput = {
  data: Float32Array | Int32Array
  shape: number[]
}

export type BonsaiGraphTensor = {
  toTypedArray(): TypedArray
  delete(): void
}

type TensorFactory = (
  data: Float32Array | Int32Array,
  shape: number[],
) => BonsaiGraphTensor

const createTensor: TensorFactory = (data, shape) => Tensor.fromTypedArray(data, shape)

function copyTypedArray(value: TypedArray): TypedArray {
  if (value instanceof Float32Array) return new Float32Array(value)
  if (value instanceof Int32Array) return new Int32Array(value)
  return new Uint8Array(value)
}

/** Runs one fixed-shape Bonsai graph and returns host-owned output arrays. */
export async function runBonsaiGraph(
  model: Pick<CompiledModel, 'run'>,
  inputs: BonsaiGraphInput[],
  tensorFactory: TensorFactory = createTensor,
): Promise<TypedArray[]> {
  const inputTensors = inputs.map(({ data, shape }) => tensorFactory(data, shape))
  let outputTensors: BonsaiGraphTensor[] = []
  try {
    const result = await model.run(inputTensors as never)
    outputTensors = Array.isArray(result)
      ? result as BonsaiGraphTensor[]
      : Object.values(result as Record<string, BonsaiGraphTensor>)
    return outputTensors.map((tensor) => copyTypedArray(tensor.toTypedArray()))
  } finally {
    for (const tensor of inputTensors) tensor.delete()
    for (const tensor of outputTensors) tensor.delete()
  }
}
