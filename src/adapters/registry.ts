import type { ModelAdapter } from './types'
import { realesrganAdapter } from './realesrgan'
import { nafnetAdapter } from './nafnet'
import { dinov2Adapter } from './dinov2'
import { registeredAdapters as musicCocaAdapters } from './magenta'
import { visionAdapters } from './vision'
import { sam2Adapters } from './sam2'
import { adapters13 } from './batch2'

export const registeredAdapters: ModelAdapter[] = [
  ...musicCocaAdapters,
  realesrganAdapter,
  nafnetAdapter,
  dinov2Adapter,
  ...visionAdapters,
  ...sam2Adapters,
  ...adapters13,
]
