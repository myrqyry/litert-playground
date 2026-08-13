import type { ModelAssetDescriptor } from '../schema/types'
import type { ModelTesterResult } from './runtime-matrix'

export interface ModelTesterOptions {
  modelId: string
  assets: ModelAssetDescriptor[]
  backends: Array<'wasm' | 'webgpu'>
}

export async function runStandaloneModelTester(
  _options: ModelTesterOptions,
): Promise<ModelTesterResult[]> {
  throw new Error(
    'The optional @litertjs/model-tester lane requires an isolated installation and is not run by deterministic tests.',
  )
}
