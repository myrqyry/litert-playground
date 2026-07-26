import type { ModelAdapter } from './types'
import { realesrganAdapter } from './realesrgan'
import { nafnetAdapter } from './nafnet'
import { dinov2Adapter } from './dinov2'
import { registeredAdapters as musicCocaAdapters } from './magenta'

export const registeredAdapters: ModelAdapter[] = [
  ...musicCocaAdapters,
  realesrganAdapter,
  nafnetAdapter,
  dinov2Adapter,
]
