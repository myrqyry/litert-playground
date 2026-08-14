import type {
  QualificationCase,
  QualificationContext,
  QualificationObservation,
} from '../schema/types'
import asset from './fixtures/asset.json'
import { tinyLitertBaselineExpected } from './expected'

export async function runTinyLitertBaseline(
  context: QualificationContext,
): Promise<QualificationObservation> {
  const bytes = new Uint8Array(await context.fetchAsset(asset))
  await context.runtime.initialize?.()
  const model = await context.runtime.loadAndCompile(bytes, {
    accelerator: context.requestedBackend,
  })
  try {
    const inputs = model.getInputDetails()
    if (inputs.length === 0) throw new Error('Baseline model has no input tensors')
    const inputTensors = inputs.map((input) => ({
      data: new Float32Array(input.shape.reduce((size, value) => size * value, 1)),
      shape: input.shape,
    }))
    const output = await model.run(inputTensors)
    if (output.length === 0) throw new Error('Baseline model returned no output tensors')
    return {
      status: 'pass',
      resolvedBackend: context.requestedBackend,
    }
  } finally {
    await model.delete()
  }
}

export const tinyLitertBaselineCase: QualificationCase = {
  id: 'tiny-litert-wasm-baseline',
  description: 'A tiny valid TFLite model provides a known-good runtime baseline.',
  evidenceKind: 'browser-observation',
  model: {
    id: 'add-10x10',
    revision: '16d8551be578965fe194e4d75f414f48c7b4e75a',
    assets: [asset],
  },
  environments: [
    {
      runtimePackage: '@litertjs/core',
      runtimeVersion: '2.5.3',
      requestedBackend: 'wasm',
    },
    {
      runtimePackage: '@litertjs/core',
      runtimeVersion: '2.5.3',
      requestedBackend: 'webgpu',
      webgpuAvailable: true,
    },
  ],
  expected: tinyLitertBaselineExpected,
  run: runTinyLitertBaseline,
}
