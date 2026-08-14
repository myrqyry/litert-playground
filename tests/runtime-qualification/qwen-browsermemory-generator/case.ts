import type {
  QualificationCase,
  QualificationContext,
  QualificationObservation,
} from '../schema/types'
import { qwen3TtsVariants } from '../../../packages/qwen3-tts/src/manifest'
import assets from './fixtures/assets.json'
import { qwenBrowserMemoryGeneratorExpected } from './expected'

const talkerRevision = '0eb3b8a4714972b065c160faec6a12158caa9dc0'
const mtpRevision = '791880469d874546d884a0e6cf68564a61c04ca9'
const traceStages = [
  'talker-compile',
  'talker-prefill',
  'talker-output-read',
  'mtp-input-build',
  'mtp-compile',
  'mtp-run',
  'mtp-output-read',
  'state-update',
] as const

export { traceStages }

export async function runQwenBrowserMemoryGenerator(
  context: QualificationContext,
): Promise<QualificationObservation> {
  if (!context.runtime.runQwenGenerator) {
    return {
      status: 'fail',
      stage: 'browser-generator',
      error: { message: 'Browser Qwen generator adapter is unavailable' },
    }
  }

  const result = await context.runtime.runQwenGenerator({
    variant: qwen3TtsVariants.browserMemoryOmni,
    assets,
    backend: context.requestedBackend,
    text: 'Testing one two three.',
    config: {
      temperature: 0,
      topK: 0,
      repetitionPenalty: 1,
      voice: 'demo_speaker',
      language: 'english',
      maxFrames: 1,
    },
  })
  return { ...result.observation, receipts: result.receipts }
}

export const qwenBrowserMemoryGeneratorCase: QualificationCase = {
  id: 'qwen-browsermemory-generator',
  description: 'Runs the real browserMemory Talker-to-MTP generator sequence.',
  evidenceKind: 'browser-observation',
  model: {
    id: 'qwen3-tts-browser-memory-generator',
    variant: 'browserMemoryOmni',
    revisions: { talker: talkerRevision, mtp: mtpRevision },
    assets: assets.map(({ path: _path, ...asset }) => asset),
  },
  environments: [{
    runtimePackage: '@litertjs/core',
    runtimeVersion: '2.5.3',
    requestedBackend: 'wasm',
  }],
  expected: qwenBrowserMemoryGeneratorExpected,
  run: runQwenBrowserMemoryGenerator,
}
