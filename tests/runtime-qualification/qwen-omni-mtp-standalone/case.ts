import type { QualificationCase, QualificationContext, QualificationObservation } from '../schema/types'
import asset from './fixtures/asset.json'
import { qwenOmniMtpStandaloneExpected } from './expected'

export async function runQwenOmniMtpStandalone(
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

export const qwenOmniMtpStandaloneCase: QualificationCase = {
  id: 'qwen-omni-mtp-standalone',
  description: 'Qualifies standalone Omni MTP graph execution in Chromium WASM.',
  evidenceKind: 'browser-observation',
  model: {
    id: 'qwen3-tts-browser-memory',
    variant: 'browserMemory',
    revision: '791880469d874546d884a0e6cf68564a61c04ca9',
    assets: [asset],
  },
  environments: [{
    runtimePackage: '@litertjs/core',
    runtimeVersion: '2.5.3',
    requestedBackend: 'wasm',
  }],
  expected: qwenOmniMtpStandaloneExpected,
  run: runQwenOmniMtpStandalone,
}
