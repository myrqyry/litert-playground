import { useState } from 'react'
import ModelRunner from './components/ModelRunner'
import { Qwen3TtsPanel } from './components/Qwen3TtsPanel'
import { registeredAdapters } from './adapters/registry'
import type { ModelAdapter } from './adapters/types'

const ttsEntry: ModelAdapter & { isPipeline: true } = {
  modelId: 'qwen3-tts',
  metadata: { name: 'Qwen3-TTS (Speech Synthesis)', description: 'Multilingual TTS via 3-graph pipeline', modelPath: '', tags: ['tts', 'pipeline'] },
  inputSpecs: [],
  outputSpecs: [],
  isPipeline: true,
  prepareInputs: () => ({}),
  parseOutputs: async () => ({}),
}

export default function App() {
  const [selectedId, setSelectedId] = useState<string | null>(null)

  if (selectedId === 'qwen3-tts') {
    return (
      <div className="max-w-2xl mx-auto p-4">
        <button
          className="text-sm text-blue-600 mb-4 hover:underline"
          onClick={() => setSelectedId(null)}
        >
          ← Back to model list
        </button>
        <h1 className="text-xl font-bold mb-4">Qwen3-TTS</h1>
        <Qwen3TtsPanel />
      </div>
    )
  }

  return (
    <ModelRunner
      adapters={[...registeredAdapters, ttsEntry]}
      selectedId={selectedId}
      onSelect={setSelectedId}
    />
  )
}
