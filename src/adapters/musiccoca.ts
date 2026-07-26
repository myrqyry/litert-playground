import { Tensor } from '@litertjs/core'
import type { ModelAdapter } from './types'

function flatten(arr: any[]): number[] {
  const out: number[] = []
  function go(x: any) {
    if (Array.isArray(x)) x.forEach(go)
    else out.push(x)
  }
  go(arr)
  return out
}

function toNestedArray(flat: Float32Array, dims: number[]): any {
  if (dims.length === 0) return flat[0]
  const size = dims[0]
  const rest = dims.slice(1)
  const result: any[] = []
  let offset = 0
  for (let i = 0; i < size; i++) {
    const subLen = dims.slice(1).reduce((a, b) => a * b, 1)
    if (rest.length === 1) {
      result.push(Array.from(flat.slice(offset, offset + subLen)))
    } else {
      result.push(toNestedArray(flat.slice(offset, offset + subLen), rest))
    }
    offset += subLen
  }
  return result
}

export const audioPreprocessor: ModelAdapter = {
  modelId: 'audio_preprocessor',
  metadata: {
    name: 'Audio Preprocessor',
    description: 'Converts raw audio waveform (10s @ 16kHz) into mel-spectrogram features [1, 992, 128]',
    modelPath: '/models/audio_preprocessor.tflite',
    tags: ['musiccoca', 'audio'],
  },
  inputSpecs: [{
    name: 'waveform',
    dtype: 'float32',
    shape: [1, 160000],
    description: 'Raw audio samples (10 seconds at 16kHz, mean-centered)',
    constraints: { min: -1, max: 1 },
  }],
  outputSpecs: [{
    name: 'features',
    dtype: 'float32',
    shape: [1, 992, 128],
    description: 'Mel-spectrogram features',
  }],
  prepareInputs(values: Record<string, any>) {
    const arr = flatten(values['waveform'] ?? [])
    if (arr.length !== 160000) arr.length = 160000
    return { waveform: new Tensor(new Float32Array(arr), [1, 160000]) }
  },
  async parseOutputs(outputs) {
    const t = outputs['Identity']
    const data = await t.data()
    return { features: toNestedArray(data as Float32Array, [1, 992, 128]) }
  },
}

export const musicEncoder: ModelAdapter = {
  modelId: 'music_encoder',
  metadata: {
    name: 'Music Encoder',
    description: 'Encodes audio features [1, 992, 128] into a 768-dim music embedding',
    modelPath: '/models/music_encoder.tflite',
    tags: ['musiccoca', 'audio', 'embedding'],
  },
  inputSpecs: [{
    name: 'features',
    dtype: 'float32',
    shape: [1, 992, 128],
    description: 'Mel-spectrogram features from Audio Preprocessor',
  }],
  outputSpecs: [{
    name: 'embedding',
    dtype: 'float32',
    shape: [1, 768],
    description: 'Music embedding vector',
  }],
  prepareInputs(values: Record<string, any>) {
    const arr = flatten(values['features'] ?? [[]])
    if (arr.length !== 992 * 128) arr.length = 992 * 128
    return { 'serving_default_args_0:0': new Tensor(new Float32Array(arr), [1, 992, 128]) }
  },
  async parseOutputs(outputs) {
    const t = outputs['StatefulPartitionedCall:0']
    const data = await t.data()
    return { embedding: toNestedArray(data as Float32Array, [1, 768]) }
  },
}

export const textEncoder: ModelAdapter = {
  modelId: 'text_encoder',
  metadata: {
    name: 'Text Encoder',
    description: 'Encodes tokenized text (max 128 tokens) into a 768-dim text embedding',
    modelPath: '/models/text_encoder.tflite',
    tags: ['musiccoca', 'text', 'embedding'],
  },
  inputSpecs: [
    {
      name: 'ids',
      dtype: 'int32',
      shape: [1, 128],
      description: 'Token IDs (pad to 128)',
    },
    {
      name: 'paddings',
      dtype: 'float32',
      shape: [1, 128],
      description: 'Attention mask (1.0 for real tokens, 0.0 for padding)',
      constraints: { min: 0, max: 1 },
    },
  ],
  outputSpecs: [{
    name: 'embedding',
    dtype: 'float32',
    shape: [1, 768],
    description: 'Text embedding vector',
  }],
  prepareInputs(values: Record<string, any>) {
    const ids = flatten(values['ids'] ?? []).map(Math.round)
    if (ids.length !== 128) ids.length = 128
    const paddings = flatten(values['paddings'] ?? [])
    if (paddings.length !== 128) paddings.length = 128
    return {
      'serving_default_ids:0': new Tensor(new Int32Array(ids), [1, 128]),
      'serving_default_paddings:0': new Tensor(new Float32Array(paddings), [1, 128]),
    }
  },
  async parseOutputs(outputs) {
    const t = outputs['StatefulPartitionedCall:0']
    const data = await t.data()
    return { embedding: toNestedArray(data as Float32Array, [1, 768]) }
  },
}

export const mapper: ModelAdapter = {
  modelId: 'mapper',
  metadata: {
    name: 'Mapper',
    description: 'Projects between music and text embedding spaces [1, 768] → [1, 768]',
    modelPath: '/models/mapper.tflite',
    tags: ['musiccoca', 'projection'],
  },
  inputSpecs: [
    {
      name: 'input_a',
      dtype: 'float32',
      shape: [1, 768],
      description: 'First embedding (e.g. text embedding)',
    },
    {
      name: 'input_b',
      dtype: 'float32',
      shape: [1, 768],
      description: 'Second embedding (e.g. music embedding)',
    },
  ],
  outputSpecs: [{
    name: 'output',
    dtype: 'float32',
    shape: [1, 768],
    description: 'Projected embedding',
  }],
  prepareInputs(values: Record<string, any>) {
    const a = flatten(values['input_a'] ?? [])
    if (a.length !== 768) a.length = 768
    const b = flatten(values['input_b'] ?? [])
    if (b.length !== 768) b.length = 768
    return {
      'serving_default_args_0:0': new Tensor(new Float32Array(a), [1, 768]),
      'serving_default_args_1:0': new Tensor(new Float32Array(b), [1, 768]),
    }
  },
  async parseOutputs(outputs) {
    const t = outputs['StatefulPartitionedCall:0']
    const data = await t.data()
    return { output: toNestedArray(data as Float32Array, [1, 768]) }
  },
}

export const quantizer: ModelAdapter = {
  modelId: 'quantizer',
  metadata: {
    name: 'Pretrained Vector Quantizer',
    description: 'Quantizes a 768-dim embedding into 12 discrete tokens (int32)',
    modelPath: '/models/pretrained_vector_quantizer.tflite',
    tags: ['musiccoca', 'quantization'],
  },
  inputSpecs: [{
    name: 'embedding',
    dtype: 'float32',
    shape: [1, 768],
    description: 'Embedding vector to quantize',
  }],
  outputSpecs: [{
    name: 'tokens',
    dtype: 'int32',
    shape: [12, 1],
    description: '12 discrete quantized tokens',
  }],
  prepareInputs(values: Record<string, any>) {
    const arr = flatten(values['embedding'] ?? [])
    if (arr.length !== 768) arr.length = 768
    return { 'quantize_inputs:0': new Tensor(new Float32Array(arr), [1, 768]) }
  },
  async parseOutputs(outputs) {
    const t = outputs['StatefulPartitionedCall_1:0']
    const data = await t.data()
    return { tokens: Array.from(data as Int32Array) }
  },
}

export const registeredAdapters: ModelAdapter[] = [
  audioPreprocessor,
  musicEncoder,
  textEncoder,
  mapper,
  quantizer,
]
