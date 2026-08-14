import type { QualificationCase, QualificationContext, QualificationObservation } from '../schema/types'
import asset from './fixtures/asset.json'
import { qwenXnnpackExpected } from './expected'

export async function runQwenXnnpackPrefill(
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
    if (!model.runWithZeros) {
      throw new Error('Browser zero-input execution adapter is unavailable')
    }
    await model.runWithZeros()
    observation = { status: 'pass', resolvedBackend: context.requestedBackend }
  } catch (error) {
    observation = {
      status: 'fail',
      stage: 'prefill',
      error: { message: error instanceof Error ? error.message : String(error) },
    }
  }
  try {
    await model?.delete()
  } catch (error) {
    observation = {
      status: 'fail',
      stage: 'prefill',
      error: { message: error instanceof Error ? error.message : String(error) },
    }
  }
  return observation
}

export const qwenXnnpackPrefillCase: QualificationCase = {
  id: 'qwen-xnnpack-prefill',
  description: 'Reproduces browserMemory Qwen XNNPACK prefill failure.',
  evidenceKind: 'browser-observation',
  model: {
    id: 'qwen3-tts-browser-memory',
    variant: 'browserMemory',
    revision: '0eb3b8a4714972b065c160faec6a12158caa9dc0',
    assets: [asset],
  },
  environments: [{
    runtimePackage: '@litertjs/core',
    runtimeVersion: '2.5.3',
    requestedBackend: 'wasm',
  }],
  expected: qwenXnnpackExpected,
  run: runQwenXnnpackPrefill,
}
