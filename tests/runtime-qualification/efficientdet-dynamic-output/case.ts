import type { QualificationCase, QualificationContext, QualificationObservation } from '../schema/types'
import asset from './fixtures/asset.json'
import { efficientDetExpected } from './expected'

export async function runEfficientDetDynamicOutput(
  context: QualificationContext,
): Promise<QualificationObservation> {
  await context.runtime.initialize?.()
  let model
  let observation: QualificationObservation
  try {
    if (!context.runtime.loadAndCompileAsset) {
      throw new Error('Browser asset compilation adapter is unavailable')
    }
    model = await context.runtime.loadAndCompileAsset(asset, {
      accelerator: context.requestedBackend,
    })
    const inputs = model.getModelDetails
      ? (await model.getModelDetails()).inputs
      : model.getInputDetails()
    const inputTensors = inputs.map((input) => ({
      data: new Float32Array(input.shape.reduce((size, value) => size * value, 1)),
      shape: input.shape,
      dtype: input.dtype,
    }))
    await model.run(inputTensors)
    observation = { status: 'pass', resolvedBackend: context.requestedBackend }
  } catch (error) {
    observation = {
      status: 'fail',
      stage: 'output-materialization',
      error: { message: error instanceof Error ? error.message : String(error) },
    }
  }
  try {
    await model?.delete()
  } catch (error) {
    observation = {
      status: 'fail',
      stage: 'output-materialization',
      error: { message: error instanceof Error ? error.message : String(error) },
    }
  }
  return observation
}

export const efficientDetDynamicOutputCase: QualificationCase = {
  id: 'efficientdet-dynamic-output',
  description: 'Reproduces EfficientDet dynamic output materialization on WASM.',
  evidenceKind: 'browser-observation',
  model: {
    id: 'efficientdet-lite0',
    revision: 'tfhub-v1',
    assets: [asset],
  },
  environments: [{
    runtimePackage: '@litertjs/core',
    runtimeVersion: '2.5.3',
    requestedBackend: 'wasm',
  }],
  expected: efficientDetExpected,
  run: runEfficientDetDynamicOutput,
}
