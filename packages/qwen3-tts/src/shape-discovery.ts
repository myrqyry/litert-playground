import type { CompiledModel, TensorDetails } from '@litertjs/core'

export interface TalkerShapeInfo {
  kvNames: string[]
  kvShapes: number[][]
  cacheLen: number
}

export interface MtpShapeInfo {
  cacheLen: number
  kvShape: number[]
}

export interface CodecShapeInfo {
  chunkSize: number
}

function lastDimension(details: TensorDetails | undefined, fallback: number): number {
  return details?.shape[details.shape.length - 1] ?? fallback
}

function findInput(details: readonly TensorDetails[], name: string): TensorDetails | undefined {
  return details.find((detail) => detail.name === name)
}

export function discoverTalkerShapes(model: CompiledModel): TalkerShapeInfo {
  const details = model.signatures.decode?.getInputDetails() ?? model.getInputDetails()
  const kv = details.filter((detail) => detail.name.startsWith('kv_cache') || detail.name.startsWith('StateArray'))
  return {
    kvNames: kv.map((detail) => detail.name),
    kvShapes: kv.map((detail) => Array.from(detail.shape)),
    cacheLen: lastDimension(findInput(details, 'mask'), 32),
  }
}

export function discoverMtpShapes(model: CompiledModel): MtpShapeInfo {
  const details = model.getInputDetails()
  return {
    cacheLen: lastDimension(findInput(details, 'args_2'), 17),
    kvShape: Array.from(findInput(details, 'args_3')?.shape ?? [1, 17, 1024]),
  }
}

export function discoverCodecShapes(model: CompiledModel): CodecShapeInfo {
  const details = model.getInputDetails()
  return { chunkSize: lastDimension(findInput(details, 'args_0'), 64) }
}
